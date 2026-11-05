"""Assemble one same-track multi-burst SAFE directory (path-SAFE).

A path-SAFE bundles the bursts of one ``(relative_orbit, orbit_direction)``
track group into a single SAFE tree that :func:`open_safe_product` reads
like any ESA SAFE subset::

    <dest_root>/<YYYYMMDD>/<parentBox>/<product_id>.SAFE/
        manifest.safe                      minimal manifest
        measurement/<stem>_b<absid>.tiff   complex64 burst raster (per burst)
        annotation/<stem>_b<absid>.xml     one-burst annotation (per burst)
        annotation/calibration/...         calibration + noise XML (copied)

Each written burst keeps its own ``measurement``/``annotation`` pair, so
:func:`open_safe_product` returns one ``S1Swath`` per burst and the
existing single-burst segment merge (:func:`open_acquisition_products`)
can reassemble them into one product — the same merge path ASF SLC-BURST
downloads already use.  File stems embed ``b{absoluteBurstID}`` when the
catalog-derived absolute id is known (stage-1
``S1Burst.absolute_burst_id``) or ``b{index:03d}`` when it is not.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import Affine

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.io.canvas import _local
from faninsar.missions.s1.io.export import _annotation_stem, _read_annotation_source
from faninsar.stack.layout import staged_acquisition_path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.missions.s1.types import S1Burst, S1Swath

logger = setup_logger(__name__)

_SAFE_PRODUCT_RE = re.compile(
    r"^(?P<mission>S1[ABCD])_(?P<beam>[A-Z]{2})_SLC__1S(?P<pols>[A-Z]{2})_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_"
    r"(?P<absorbit>\d{6})_(?P<datatake>[0-9A-F]{6})_(?P<uid>[0-9A-F]{4})$"
)

_DATATAKE_DEFAULT = "000000"


@dataclass(frozen=True, slots=True)
class PathBurstPayload:
    """One kept burst's pixels and identity for :func:`write_path_safe`.

    Attributes
    ----------
    burst_index : int
        Original burst index within the parent swath.
    samples : numpy.ndarray
        Burst samples of shape ``(lines_per_burst, samples_per_burst)``
        (complex64).  Rows outside the fetched envelope may be zero-filled;
        ``line_valid`` governs the annotation mask.
    line_valid : tuple of tuple of int, optional
        Per-burst-line ``(first, last)`` valid column range, length
        ``lines_per_burst``.  ``None`` marks every line fully valid.
    absolute_burst_id : int, optional
        Mission-independent ESA burst identifier when the catalog carries
        it (ASF ``BURST_ID_ABSOLUTE``); overrides
        ``S1Burst.absolute_burst_id`` for naming.
    data_take : str, optional
        Six-hex-digit data-take id for the SAFE product name; falls back
        to the parent product name's field or ``000000``.
    annotation_xml : str, optional
        Per-payload parent annotation document.  ASF SLC-BURST downloads
        arrive as independent one-burst products, so each payload supplies
        its own annotation instead of indexing a shared parent's
        ``burstList``.  When set, ``burst_index`` selects within *this*
        document (usually 0) and several payloads may share the same
        ``burst_index`` value.  The group's
        ``(azimuth_time, sensing_time)`` slot order, not the raw index,
        decides written order.

    """

    burst_index: int
    samples: np.ndarray
    line_valid: tuple[tuple[int, int], ...] | None = None
    absolute_burst_id: int | None = None
    data_take: str | None = None
    annotation_xml: str | None = None


@dataclass(frozen=True, slots=True)
class PathSafeResult:
    """Handle to one assembled path-SAFE directory.

    Attributes
    ----------
    safe_root : pathlib.Path
        Generated ``<product_id>.SAFE`` directory.
    product_id : str
        SAFE product id (directory name without ``.SAFE``).
    track_key : str
        ``"<relOrbit><A|D>"`` group label the assembly was written under.
    swath_paths : mapping
        ``{swath_name: ((annotation_path, measurement_path), ...)}`` of
        per-burst asset pairs in written order.
    kept_bursts : mapping
        ``{swath_name: (burst_index, ...)}`` in written slot order.

    """

    safe_root: Path
    product_id: str
    track_key: str
    swath_paths: dict[str, tuple[tuple[Path, Path], ...]] = field(default_factory=dict)
    kept_bursts: dict[str, tuple[int, ...]] = field(default_factory=dict)


def _track_parts(track_key: str) -> tuple[str, str]:
    """Split ``"88A"``-style key into ``(digits, letter)``."""
    text = str(track_key or "").strip().upper()
    digits = ""
    for char in text:
        if not char.isdigit():
            break
        digits += char
    return digits or "000", text[len(digits) :] or "?"


def _child(parent: ET.Element, name: str) -> ET.Element:
    for child in parent:
        if _local(child.tag) == name:
            return child
    message = f"missing required annotation element: {name}"
    reject_product(message)
    raise AssertionError(message)


def _has(parent: ET.Element, name: str) -> bool:
    """Return whether ``parent`` has a child with the local tag ``name``."""
    return any(_local(child.tag) == name for child in parent)


def _set_text(parent: ET.Element, name: str, value: str) -> None:
    _child(parent, name).text = value


def _fmt_safe_time(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S")


def _burst_data_take(product_id: str, payload: PathBurstPayload) -> str:
    """Resolve the six-hex data-take id for the SAFE name."""
    if payload.data_take:
        return payload.data_take.upper()[:6].ljust(6, "0")
    match = _SAFE_PRODUCT_RE.fullmatch(product_id)
    if match:
        return match.group("datatake")
    return _DATATAKE_DEFAULT


def _annotation_mission(xml_text: str) -> str:
    """Return the ``adsHeader/missionId`` (``S1A``-style) from annotation XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    for element in root.iter():
        if _local(element.tag) == "missionId" and element.text:
            return element.text.strip().upper()[:3]
    return ""


def _path_safe_product_id(
    swath: S1Swath,
    bursts: Sequence[S1Burst],
    track_key: str,
    payloads: Sequence[PathBurstPayload],
    source_product_id: str,
    annotation_xml: str,
    *,
    sensing: Sequence[datetime] | None = None,
) -> str:
    """Build the path-SAFE product id from track identity and burst times.

    ``{MISSION}_IW_SLC__1S{pols}_{t_start}_{t_stop}_{relOrbit}{A|D}_{dataTake}_{hash}``
    where ``t_start``/``t_stop`` are the group's sensing min/max and the
    final field is a four-hex digest of the absolute burst ids (or slot
    indices) so distinct subsets of one track never collide.
    """
    match = _SAFE_PRODUCT_RE.fullmatch(source_product_id)
    mission = _annotation_mission(annotation_xml) or (
        match.group("mission") if match else "S1?"
    )
    beam = match.group("beam") if match else swath.swath[:2]
    pols = (
        match.group("pols")
        if match
        else ("DV" if swath.polarization in {"VV", "VH"} else "DH")
    )
    if sensing is None:
        sensing = [burst.sensing_time for burst in bursts]
    t_start = min(sensing)
    t_stop = max(sensing)
    digits, letter = _track_parts(track_key)
    track_field = f"{int(digits):05d}{letter}"
    data_take = _burst_data_take(source_product_id, payloads[0])
    id_parts: list[str] = []
    for i, payload in enumerate(payloads):
        if payload.absolute_burst_id is not None:
            id_parts.append(str(payload.absolute_burst_id))
        elif bursts[i].absolute_burst_id is not None:
            id_parts.append(str(bursts[i].absolute_burst_id))
        elif payload.annotation_xml is not None:
            # Per-document payloads may repeat burst_index; their own
            # sensing time keeps the digest unique.
            id_parts.append(f"{payload.burst_index}@{sensing[i].isoformat()}")
        else:
            id_parts.append(str(payload.burst_index))
    id_bits = "-".join(id_parts)
    digest = hashlib.md5(id_bits.encode()).hexdigest()[:4].upper()
    return (
        f"{mission}_{beam}_SLC__1S{pols}_{_fmt_safe_time(t_start)}_"
        f"{_fmt_safe_time(t_stop)}_{track_field}_{data_take}_{digest}"
    )


def _build_burst_annotation(
    xml_text: str,
    keep_index: int,
    payload: PathBurstPayload,
    lines_per_burst: int,
    t_start: datetime,
    t_stop: datetime,
) -> bytes:
    """Return a one-burst annotation XML for ``keep_index``.

    The parent ``burstList`` is trimmed to the single kept burst, an
    ``absoluteBurstID`` element is injected when the id is known, the
    valid-sample mask is rewritten when ``payload.line_valid`` is set,
    and ``adsHeader``/``imageInformation`` reflect the one-burst extent.
    """
    root = ET.fromstring(xml_text)
    swath_timing = next(c for c in root if _local(c.tag) == "swathTiming")
    burst_list = next(c for c in swath_timing if _local(c.tag) == "burstList")
    burst_elements = [c for c in burst_list if _local(c.tag) == "burst"]
    if keep_index < 0 or keep_index >= len(burst_elements):
        reject_product(
            f"burst_index {keep_index} out of range for parent burstList "
            f"({len(burst_elements)} bursts)"
        )
    for index, burst_el in enumerate(burst_elements):
        if index != keep_index:
            burst_list.remove(burst_el)
            continue
        absolute = payload.absolute_burst_id
        if absolute is None:
            continue
        if not _has(burst_el, "absoluteBurstID"):
            element = ET.SubElement(burst_el, "absoluteBurstID")
            element.text = str(absolute)
    burst_list.set("count", "1")

    kept_el = burst_elements[keep_index]
    if payload.line_valid is not None:
        first = [int(pair[0]) for pair in payload.line_valid]
        last = [int(pair[1]) for pair in payload.line_valid]
        _set_text(kept_el, "firstValidSample", " ".join(str(v) for v in first))
        _set_text(kept_el, "lastValidSample", " ".join(str(v) for v in last))

    image_info = _child(_child(root, "imageAnnotation"), "imageInformation")
    _set_text(image_info, "numberOfLines", str(lines_per_burst))
    ads = _child(root, "adsHeader")
    _set_text(ads, "startTime", t_start.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    _set_text(ads, "stopTime", t_stop.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _burst_stem(base: str, payload: PathBurstPayload, burst: S1Burst) -> str:
    """Return the per-burst file stem embedding ``b{absoluteBurstID}``."""
    absolute = (
        payload.absolute_burst_id
        if payload.absolute_burst_id is not None
        else burst.absolute_burst_id
    )
    tag = f"b{absolute}" if absolute is not None else f"b{payload.burst_index:03d}"
    return f"{base}_{tag}"


def manifest_processing_lines(ipf_version: str | None) -> list[str]:
    """Return the ``processing`` metadata object recording the IPF version.

    Materialized SAFEs keep the source product's IPF version so loaders can
    still decide whether the elevation antenna phase correction applies.
    """
    if not ipf_version:
        return []
    return [
        '    <metadataObject ID="processing">',
        "      <metadataWrap><xmlData>",
        (
            "        <safe:processing><safe:facility>"
            '<safe:software name="Sentinel-1 IPF" '
            f'version="{ipf_version}"/></safe:facility></safe:processing>'
        ),
        "      </xmlData></metadataWrap>",
        "    </metadataObject>",
    ]


def write_minimal_manifest(
    safe_root: Path,
    product_id: str,
    names: Sequence[str],
    *,
    ipf_version: str | None = None,
) -> None:
    """Write a minimal manifest.safe listing the emitted assets.

    Parameters
    ----------
    safe_root : pathlib.Path
        ``.SAFE`` directory receiving ``manifest.safe``.
    product_id : str
        Product id whose first three characters name the platform unit.
    names : sequence of str
        SAFE-relative asset paths listed in the file section.
    ipf_version : str, optional
        Source product IPF version, recorded in a ``processing`` object.

    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" '
            'xmlns:safe="http://www.esa.int/safe/sentinel-1.0" '
            'version="esa/safe/sentinel-1.0/sentinel-1">'
        ),
        "  <metadataSection>",
        '    <metadataObject ID="platform">',
        "      <metadataWrap><xmlData>",
        (
            "        <safe:platform><safe:familyName>SENTINEL-1</safe:familyName>"
            f"<safe:number>{product_id[:3]}</safe:number></safe:platform>"
        ),
        "      </xmlData></metadataWrap>",
        "    </metadataObject>",
        *manifest_processing_lines(ipf_version),
        "  </metadataSection>",
        "  <fileSection>",
    ]
    lines.extend(
        f'    <file ID="{name}"><location href="./{name}" /></file>' for name in names
    )
    lines += ["  </fileSection>", "</xfdu:XFDU>", ""]
    (safe_root / "manifest.safe").write_text("\n".join(lines), encoding="utf-8")


def _payload_burst(payload: PathBurstPayload, swath: S1Swath) -> S1Burst:
    """Return the ``S1Burst`` metadata for one payload.

    Shared-parent payloads index ``swath.bursts``; per-document payloads
    (ASF one-burst products) always address burst 0 of their own
    annotation and reuse ``swath.bursts[0]`` since every member of the
    same-track group shares geometry.
    """
    index = int(payload.burst_index)
    if payload.annotation_xml is not None:
        index = 0
    if index < 0 or index >= len(swath.bursts):
        reject_product(f"burst_index {index} out of range for {swath.swath}")
    return swath.bursts[index]


def _payload_sensing_time(payload: PathBurstPayload, fallback: S1Burst) -> datetime:
    """Return the burst sensing time of one payload.

    Per-document payloads read ``sensingTime`` from their own annotation
    (each ASF one-burst product has a distinct time); shared-parent
    payloads use the parent's burst entry.
    """
    if payload.annotation_xml is not None:
        try:
            root = ET.fromstring(payload.annotation_xml)
        except ET.ParseError:
            return fallback.sensing_time
        for burst_el in root.iter():
            if _local(burst_el.tag) != "burst":
                continue
            for child in burst_el:
                if _local(child.tag) == "sensingTime" and child.text:
                    try:
                        return datetime.fromisoformat(child.text.strip())
                    except ValueError:
                        return fallback.sensing_time
            return fallback.sensing_time
    return fallback.sensing_time


def write_path_safe(
    swath: S1Swath,
    payloads: Sequence[PathBurstPayload],
    track_key: str,
    dest_root: str | Path,
    *,
    annotation_xml: str | None = None,
    product_id: str | None = None,
    ipf_version: str | None = None,
) -> PathSafeResult:
    """Assemble one same-track burst group into a SAFE directory.

    Parameters
    ----------
    swath : S1Swath
        Parsed parent sub-swath supplying burst geometry and the source
        annotation document.
    payloads : sequence of PathBurstPayload
        Kept bursts (any order); each is written as its own
        ``measurement``/``annotation`` pair.  Shared-parent payloads
        (``annotation_xml=None``) are ordered by ``burst_index`` and must
        be unique; per-document payloads (``annotation_xml`` set) order
        by their own burst ``sensing_time`` and may repeat index 0.
    track_key : str
        ``"<relOrbit><A|D>"`` grouping key (see
        :func:`~faninsar.missions.s1.types.track_group_key`).
    dest_root : str or pathlib.Path
        Staging root (e.g. ``tracks/{track_key}/acquisitions/``); the
        ``<YYYYMMDD>/<parentBox>/<product_id>.SAFE`` folder is created
        inside it.
    annotation_xml : str, optional
        Parent annotation XML text; defaults to ``swath.annotation_path``.
    product_id : str, optional
        Override SAFE product id; defaults to the assembled
        ``{MISSION}_IW_SLC__1S{pols}_{t_start}_{t_stop}_{track}_{take}_{hash}``
        name.
    ipf_version : str, optional
        Source product IPF version recorded in ``manifest.safe`` so the
        generated SAFE keeps its EAP-correction decision.

    Returns
    -------
    PathSafeResult
        Handle with ``safe_root`` and per-burst asset paths, compatible
        with :func:`~faninsar.missions.s1.safe.open_safe_product`.

    Raises
    ------
    Sentinel1ProductError
        If ``payloads`` is empty, a burst index is out of range or
        duplicated, or a sample block does not match the swath geometry.

    """
    if not payloads:
        reject_product("write_path_safe requires at least one burst payload")
    shared = [p for p in payloads if p.annotation_xml is None]
    if len({int(p.burst_index) for p in shared}) != len(shared):
        reject_product(
            "duplicate shared-parent burst indices in path-SAFE payloads: "
            f"{tuple(int(p.burst_index) for p in shared)}"
        )
    lines_per_burst = int(swath.lines_per_burst)
    samples_per_burst = int(swath.samples_per_burst)

    # Resolve every payload to (payload, burst, source_xml) before writing.
    resolved: list[tuple[PathBurstPayload, S1Burst, str]] = []
    for payload in payloads:
        source_text = (
            payload.annotation_xml
            if payload.annotation_xml is not None
            else (
                annotation_xml
                if annotation_xml is not None
                else _read_annotation_source(swath)
            )
        )
        burst = _payload_burst(payload, swath)
        array = np.asarray(payload.samples, dtype=np.complex64)
        expected = (lines_per_burst, samples_per_burst)
        if array.ndim == 1 and array.size == expected[0] * expected[1]:
            array = array.reshape(expected)
        if array.shape != expected:
            reject_product(
                f"path-SAFE burst {payload.burst_index} samples shape "
                f"{array.shape} exceeds parent burst geometry {expected}"
            )
        resolved.append(
            (
                payload,
                burst,
                source_text,
            )
        )
    # Written order: per-document payloads by sensing_time (ASF groups),
    # shared-parent by burst_index.  A uniform tuple key keeps the sort
    # total even if a group ever blends the two forms.
    sensing = [
        _payload_sensing_time(payload, burst) for payload, burst, _source in resolved
    ]
    resolved = [
        entry
        for _t, entry in sorted(
            zip(sensing, resolved, strict=True),
            key=lambda pair: (
                pair[1][0].annotation_xml is None,
                (
                    int(pair[1][0].burst_index)
                    if pair[1][0].annotation_xml is None
                    else pair[0].timestamp()
                ),
            ),
        )
    ]
    ordered = [entry[0] for entry in resolved]
    bursts = [entry[1] for entry in resolved]
    sensing = [
        _payload_sensing_time(payload, burst) for payload, burst, _source in resolved
    ]

    first_source = resolved[0][2]
    source_id = product_id or _annotation_stem(swath.annotation_path)
    pid = _path_safe_product_id(
        swath, bursts, track_key, ordered, source_id, first_source, sensing=sensing
    )
    dest_root = Path(dest_root)
    safe_root = staged_acquisition_path(dest_root, f"{pid}.SAFE")
    (safe_root / "measurement").mkdir(parents=True, exist_ok=True)
    (safe_root / "annotation").mkdir(parents=True, exist_ok=True)

    stem_base = _annotation_stem(swath.annotation_path)
    names: list[str] = []
    pairs: list[tuple[Path, Path]] = []

    for position, (payload, burst, source_text) in enumerate(resolved):
        stem = _burst_stem(stem_base, payload, burst)
        # Distinct per-document payloads can collide on index-0 stems when
        # no absolute id is known; disambiguate by written slot.
        if payload.annotation_xml is not None and payload.absolute_burst_id is None:
            stem = f"{stem}_s{position:02d}"
        tiff_path = safe_root / "measurement" / f"{stem}.tiff"
        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=lines_per_burst,
            width=samples_per_burst,
            count=1,
            dtype="complex64",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            crs=None,
            transform=Affine.identity(),
        ) as dst:
            dst.write(
                np.ascontiguousarray(
                    np.asarray(payload.samples, dtype=np.complex64).reshape(
                        (lines_per_burst, samples_per_burst)
                    ),
                    dtype=np.complex64,
                ),
                1,
            )
        names.append(f"measurement/{stem}.tiff")

        rewritten = _build_burst_annotation(
            source_text,
            int(payload.burst_index) if payload.annotation_xml is None else 0,
            payload,
            lines_per_burst,
            sensing[position],
            sensing[position],
        )
        ann_path = safe_root / "annotation" / f"{stem}.xml"
        ann_path.write_bytes(rewritten)
        names.append(f"annotation/{stem}.xml")
        pairs.append((ann_path, tiff_path))

    src_dir = Path(str(swath.annotation_path)).parent
    cal_src = src_dir / "calibration"
    if cal_src.is_dir():
        cal_dst = safe_root / "annotation" / "calibration"
        cal_dst.mkdir(parents=True, exist_ok=True)
        stem = _burst_stem(stem_base, ordered[0], bursts[0])
        for name in (f"calibration-{stem}.xml", f"noise-{stem}.xml"):
            candidate = cal_src / name
            if candidate.exists():
                shutil.copy2(candidate, cal_dst / name)
                names.append(f"annotation/calibration/{name}")

    write_minimal_manifest(safe_root, pid, names, ipf_version=ipf_version)
    logger.info(
        "Wrote path-SAFE %s: %d burst(s) of %s under track %s",
        safe_root,
        len(ordered),
        swath.swath,
        track_key,
    )
    return PathSafeResult(
        safe_root=safe_root,
        product_id=pid,
        track_key=track_key,
        swath_paths={swath.swath: tuple(pairs)},
        kept_bursts={swath.swath: tuple(int(p.burst_index) for p in ordered)},
    )


__all__ = ["PathBurstPayload", "PathSafeResult", "write_path_safe"]

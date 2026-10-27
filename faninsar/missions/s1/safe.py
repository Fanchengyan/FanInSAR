"""Open Sentinel-1 Level-1 SAFE products from directories or ZIP archives."""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

from .annotation import parse_annotation_xml
from .errors import UnsupportedPolarizationError, reject_product
from .types import S1Burst, S1Product, S1Swath

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = setup_logger(__name__)

_SUPPORTED_POLARIZATIONS = frozenset({"VV", "VH", "HH", "HV"})
_MISSION_PREFIXES = ("S1A", "S1B", "S1C", "S1D")
_MISSION_ID_RE = re.compile(r"<missionId>\s*(S1[ABCD])\s*</missionId>", re.IGNORECASE)


def open_safe_product(
    path: str | Path,
    *,
    polarizations: tuple[str, ...] | None = None,
) -> S1Product:
    """Open a SAFE directory or ZIP and parse annotation metadata lazily.

    Measurement GeoTIFF pixels are not loaded. Paths to measurement assets are
    retained for later burst-window reads.

    Parameters
    ----------
    path : str or pathlib.Path
        SAFE directory or ``.zip`` archive.
    polarizations : tuple of str, optional
        Subset of polarizations to load. Defaults to all present.

    Returns
    -------
    S1Product
        Typed product handle with sub-swath and burst metadata.

    Raises
    ------
    Sentinel1ProductError
        If the product structure or required metadata is invalid.
    UnsupportedPolarizationError
        If a requested polarization is not present.

    """
    source = Path(path)
    if not source.exists():
        reject_product(f"SAFE product does not exist: {source}")

    if source.is_dir():
        return _open_safe_directory(source, polarizations=polarizations)
    if source.suffix.lower() == ".zip":
        return _open_safe_zip(source, polarizations=polarizations)
    return reject_product(f"unsupported SAFE source type: {source}")


def open_acquisition_products(
    paths: Sequence[str | Path],
    *,
    polarizations: tuple[str, ...] | None = None,
) -> list[S1Product]:
    """Open one date's SAFE inputs, merging complementary same-date extracts.

    Full-scene frames stay separate. Same-date ASF burst ZIPs that each carry
    a single burst are assembled into one product so NESD sees adjacent bursts
    of one swath. Same-date ROI canvases that each carry a disjoint sub-swath
    of one parent scene are assembled the same way.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        SAFE directories or ZIP archives of one acquisition date.
    polarizations : tuple of str, optional
        Polarization subset forwarded to :func:`open_safe_product`.

    Returns
    -------
    list of S1Product
        One product per remaining frame after complementary extracts merge.

    Raises
    ------
    Sentinel1ProductError
        If a path is not a SAFE product or required metadata is invalid.
    UnsupportedPolarizationError
        If a requested polarization is not present.

    """
    products = [open_safe_product(path, polarizations=polarizations) for path in paths]
    if len(products) <= 1:
        return products
    if _all_complementary_swath_canvases(products):
        return [_merge_swath_canvas_products(products)]
    if _all_single_burst_segments(products):
        return [_merge_burst_segment_products(products)]
    return products


def open_safe_acquisition(
    paths: Sequence[str | Path],
    *,
    polarizations: tuple[str, ...] | None = None,
) -> S1Product:
    """Open one logical acquisition, merging complementary SAFE segments.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        SAFE directories or ZIP archives of one acquisition date.
    polarizations : tuple of str, optional
        Polarization subset forwarded to :func:`open_safe_product`.

    Returns
    -------
    S1Product
        The single merged product for that date.

    Raises
    ------
    Sentinel1ProductError
        If the inputs do not collapse to one logical product.
    UnsupportedPolarizationError
        If a requested polarization is not present.

    """
    products = open_acquisition_products(paths, polarizations=polarizations)
    if len(products) != 1:
        reject_product(
            "open_safe_acquisition requires one logical product; "
            f"got {len(products)} frames"
        )
    return products[0]


def _open_safe_directory(
    root: Path,
    *,
    polarizations: tuple[str, ...] | None,
) -> S1Product:
    if (root / "manifest.safe").exists():
        safe_roots = [root]
    else:
        safe_roots = list(root.glob("*.SAFE"))
    if (
        not safe_roots
        and (root / "annotation").is_dir()
        and (root / "measurement").is_dir()
    ):
        safe_roots = [root]
    if len(safe_roots) != 1:
        reject_product(f"expected one SAFE root under {root}")
    safe_root = safe_roots[0]
    annotation_files = sorted(
        path
        for path in (safe_root / "annotation").glob("*.xml")
        if path.is_file() and "calibration" not in path.parts
    )
    if not annotation_files:
        reject_product(f"no annotation XML found under {safe_root}")

    swaths: list[S1Swath] = []
    xml_texts: list[str] = []
    for annotation in annotation_files:
        measurement = _measurement_for_annotation(safe_root, annotation.name)
        xml_text = annotation.read_text(encoding="utf-8")
        xml_texts.append(xml_text)
        swath = parse_annotation_xml(
            xml_text,
            annotation_path=str(annotation),
            measurement_path=str(measurement),
        )
        swaths.append(swath)
    return _assemble_product(
        source_path=root,
        swaths=swaths,
        polarizations=polarizations,
        identity_names=(
            root.name,
            safe_root.name,
            *(path.name for path in annotation_files),
        ),
        xml_texts=xml_texts,
    )


def _open_safe_zip(
    zip_path: Path,
    *,
    polarizations: tuple[str, ...] | None,
) -> S1Product:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        annotation_names = sorted(
            name
            for name in names
            if "/annotation/" in name
            and name.endswith(".xml")
            and "/calibration/" not in name
            and "/rfi/" not in name
            and not name.endswith("/")
        )
        if not annotation_names:
            reject_product(f"no annotation XML found in {zip_path}")

        swaths: list[S1Swath] = []
        xml_texts: list[str] = []
        identity_names: list[str] = [zip_path.name]
        for annotation_name in annotation_names:
            measurement_name = _measurement_member_for_annotation(
                names,
                annotation_name,
            )
            xml_text = archive.read(annotation_name).decode("utf-8")
            xml_texts.append(xml_text)
            identity_names.append(annotation_name)
            identity_names.extend(
                part for part in Path(annotation_name).parts if part.endswith(".SAFE")
            )
            measurement_uri = f"/vsizip/{zip_path}/{measurement_name}"
            swath = parse_annotation_xml(
                xml_text,
                annotation_path=f"/vsizip/{zip_path}/{annotation_name}",
                measurement_path=measurement_uri,
            )
            swaths.append(swath)
    return _assemble_product(
        source_path=zip_path,
        swaths=swaths,
        polarizations=polarizations,
        identity_names=identity_names,
        xml_texts=xml_texts,
    )


def _measurement_for_annotation(safe_root: Path, annotation_name: str) -> Path:
    stem = Path(annotation_name).stem
    candidate = safe_root / "measurement" / f"{stem}.tiff"
    if not candidate.exists():
        matches = list((safe_root / "measurement").glob(f"{stem}.*"))
        if not matches:
            reject_product(f"missing measurement for annotation {annotation_name}")
        return matches[0]
    return candidate


def _measurement_member_for_annotation(
    names: list[str],
    annotation_name: str,
) -> str:
    stem = Path(annotation_name).stem
    for name in names:
        if "/measurement/" in name and Path(name).stem == stem:
            return name
    return reject_product(f"missing measurement member for {annotation_name}")


def _assemble_product(
    *,
    source_path: Path,
    swaths: list[S1Swath],
    polarizations: tuple[str, ...] | None,
    identity_names: Sequence[str] = (),
    xml_texts: Sequence[str] = (),
) -> S1Product:
    present = tuple(sorted({swath.polarization for swath in swaths}))
    requested = polarizations or present
    allowed = sorted(_SUPPORTED_POLARIZATIONS)
    for pol in requested:
        if pol not in _SUPPORTED_POLARIZATIONS:
            message = f"unsupported polarization {pol!r}; allowed={allowed}"
            logger.error(message)
            raise UnsupportedPolarizationError(message)
        if pol not in present:
            message = (
                f"polarization {pol!r} not present in {source_path}; "
                f"available={present}"
            )
            logger.error(message)
            raise UnsupportedPolarizationError(message)

    filtered = tuple(swath for swath in swaths if swath.polarization in requested)
    if not filtered:
        reject_product(f"no swaths remain after polarization filter for {source_path}")

    product_id, mission_id = _product_identity(
        source_path,
        identity_names=identity_names,
        xml_texts=xml_texts,
    )
    mode = "IW" if "IW" in product_id.upper() else filtered[0].swath[:2]
    absolute_orbit = _absolute_orbit(product_id)
    if absolute_orbit == 0:
        for name in identity_names:
            absolute_orbit = _absolute_orbit(name)
            if absolute_orbit:
                break

    return S1Product(
        product_id=product_id,
        source_path=source_path,
        mission_id=mission_id,
        product_type="SLC",
        mode=mode,
        absolute_orbit=absolute_orbit,
        polarizations=tuple(requested),
        swaths=filtered,
    )


def _product_identity(
    source_path: Path,
    *,
    identity_names: Sequence[str],
    xml_texts: Sequence[str],
) -> tuple[str, str]:
    """Return ``(product_id, mission_id)`` from inner SAFE / annotation identity."""
    product_id: str | None = None
    mission_id: str | None = None
    for xml_text in xml_texts:
        match = _MISSION_ID_RE.search(xml_text)
        if match is not None:
            mission_id = match.group(1).upper()
            break
    for name in identity_names:
        candidate = _safe_product_id(name)
        if candidate is not None:
            product_id = candidate
            mission_id = mission_id or candidate[:3]
            break
        mission_id = mission_id or _mission_prefix(name)
    if product_id is None:
        product_id = source_path.stem.replace(".SAFE", "")
    if mission_id is None:
        mission_id = _mission_prefix(product_id) or "S1"
    return product_id, mission_id


def _safe_product_id(name: str) -> str | None:
    """Return a Sentinel-1 product id from a SAFE directory or annotation name."""
    stem = Path(str(name)).name.removesuffix(".SAFE").removesuffix(".zip")
    upper = stem.upper()
    if upper.startswith(_MISSION_PREFIXES) and "_" in stem:
        return stem
    return None


def _mission_prefix(name: str) -> str | None:
    """Return S1A/B/C/D when ``name`` starts with a mission identifier."""
    compact = str(name).upper().replace("-", "")
    for prefix in _MISSION_PREFIXES:
        if compact.startswith(prefix):
            return prefix
    return None


def _absolute_orbit(name: str) -> int:
    """Return the six-digit absolute orbit encoded in a product name, if any."""
    for part in str(name).replace("-", "_").split("_"):
        if part.isdigit() and len(part) == 6:
            return int(part)
    return 0


def _all_complementary_swath_canvases(products: Sequence[S1Product]) -> bool:
    """Return whether products are disjoint same-orbit sub-swath canvases."""
    if len(products) < 2:
        return False
    mission_id = products[0].mission_id
    orbit = products[0].absolute_orbit
    seen: set[tuple[str, str]] = set()
    for product in products:
        if product.mission_id != mission_id or product.absolute_orbit != orbit:
            return False
        if not product.swaths:
            return False
        keys = {(swath.swath, swath.polarization) for swath in product.swaths}
        if keys & seen:
            return False
        seen.update(keys)
    return len({swath for swath, _pol in seen}) > 1


def _merge_swath_canvas_products(products: Sequence[S1Product]) -> S1Product:
    """Assemble disjoint same-date sub-swath canvases into one product."""
    merged_swaths = sorted(
        (swath for product in products for swath in product.swaths),
        key=lambda item: item.slant_range_time_s,
    )
    first = products[0]
    polarizations = tuple(
        sorted({pol for product in products for pol in product.polarizations})
    )
    return S1Product(
        product_id=first.product_id,
        source_path=first.source_path,
        mission_id=first.mission_id,
        product_type=first.product_type,
        mode=first.mode,
        absolute_orbit=first.absolute_orbit,
        polarizations=polarizations,
        swaths=tuple(merged_swaths),
    )


def _all_single_burst_segments(products: Sequence[S1Product]) -> bool:
    """Return whether every product is a one-burst extract that can be merged."""
    if len(products) < 2:
        return False
    mission_id = products[0].mission_id
    orbit = products[0].absolute_orbit
    for product in products:
        if product.mission_id != mission_id or product.absolute_orbit != orbit:
            return False
        if not product.swaths:
            return False
        if any(len(swath.bursts) != 1 for swath in product.swaths):
            return False
    return True


def _merge_burst_segment_products(products: Sequence[S1Product]) -> S1Product:
    """Assemble one-burst SAFE extracts into one product per swath/polarization."""
    groups: dict[tuple[str, str], list[tuple[S1Product, S1Swath]]] = defaultdict(list)
    for product in products:
        for swath in product.swaths:
            groups[(swath.swath, swath.polarization)].append((product, swath))

    merged_swaths: list[S1Swath] = []
    for _key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda pair: pair[1].bursts[0].azimuth_time)
        template = ordered[0][1]
        bursts: list[S1Burst] = []
        for index, (_product, swath) in enumerate(ordered):
            burst = swath.bursts[0]
            bursts.append(
                replace(
                    burst,
                    index=index,
                    measurement_path=burst.measurement_path or swath.measurement_path,
                )
            )
        vectors = []
        seen: set[object] = set()
        for _product, swath in ordered:
            for vector in swath.orbit.vectors:
                if vector.time in seen:
                    continue
                seen.add(vector.time)
                vectors.append(vector)
        vectors.sort(key=lambda vector: vector.time)
        merged_swaths.append(
            replace(
                template,
                lines=len(bursts) * template.lines_per_burst,
                bursts=tuple(bursts),
                sensing_start=min(swath.sensing_start for _product, swath in ordered),
                sensing_stop=max(swath.sensing_stop for _product, swath in ordered),
                orbit=replace(
                    template.orbit,
                    vectors=tuple(vectors),
                    source="merged-burst-segments",
                ),
                doppler_centroid=tuple(
                    term
                    for _product, swath in ordered
                    for term in swath.doppler_centroid
                ),
                azimuth_fm_rate=tuple(
                    term
                    for _product, swath in ordered
                    for term in swath.azimuth_fm_rate
                ),
            )
        )

    first = products[0]
    polarizations = tuple(
        sorted({pol for product in products for pol in product.polarizations})
    )
    return S1Product(
        product_id=first.product_id,
        source_path=first.source_path,
        mission_id=first.mission_id,
        product_type=first.product_type,
        mode=first.mode,
        absolute_orbit=first.absolute_orbit,
        polarizations=polarizations,
        swaths=tuple(merged_swaths),
    )

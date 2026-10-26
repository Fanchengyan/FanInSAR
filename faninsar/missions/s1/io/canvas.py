"""SAFE-subset full-burst-height canvases for ROI partial SLC transfer.

PROPOSAL-0061 materializes CDSE ROI fetches as a SAFE-compatible subset
instead of cropped GeoTIFFs, so ``from_safes`` / :func:`read_full_burst` /
ESD see parent burst dimensions.  Per kept burst this module writes a
full-burst-height x full-``samplesPerBurst`` canvas, rewrites the
valid-sample XML mask to the actually-fetched column envelope, remaps the
geolocation grid by original burst identity, and splices overlap-strip
rows back at their original burst-local lines before the tree is handed to
``from_safes``.

The canvas tree mirrors :mod:`faninsar.missions.s1.io.export`::

    <out_dir>/<product_id>.SAFE/
        measurement/<...>.tiff          complex64 canvas (kept bursts)
        annotation/<...>.xml            annotation with rewritten mask
        annotation/calibration/...      calibration + noise XML (copied)
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import rasterio
from rasterio.windows import Window

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.io.export import (
    _annotation_stem,
    _read_annotation_source,
    _write_burst_tiff,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from faninsar.missions.s1.overlap import BurstOverlapWindow
    from faninsar.missions.s1.types import S1Swath

logger = setup_logger(__name__)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass(frozen=True, slots=True)
class BurstFetchedWindow:
    """Fetched sample rectangle of one kept burst in burst-local coordinates.

    Attributes
    ----------
    burst_index : int
        Original burst index within the parent swath.  Canvas slot order is
        ascending ``burst_index``, independent of the order given here.
    row0 : int
        Burst-local line of the first fetched row.
    col0 : int
        Burst-local column of the first fetched column.
    samples : numpy.ndarray
        Fetched samples of shape ``(n_lines, n_samples)`` placed at
        ``(row0, col0)`` on the canvas.  Pixels outside fetched windows
        stay zero.

    """

    burst_index: int
    row0: int
    col0: int
    samples: np.ndarray


@dataclass(frozen=True, slots=True)
class RoiSafeCanvas:
    """Handle to a SAFE-subset canvas tree written for ROI transfer.

    Attributes
    ----------
    safe_root : pathlib.Path
        Generated ``<product_id>.SAFE`` directory.
    measurement_path : pathlib.Path
        Canvas GeoTIFF of shape
        ``(n_kept * lines_per_burst, samples_per_burst)``.
    annotation_path : pathlib.Path
        Rewritten annotation XML (valid-sample mask + remapped grid).
    kept_indices : tuple of int
        Original parent burst indices in ascending order; canvas slot ``j``
        holds ``kept_indices[j]``.
    lines_per_burst : int
        Parent ``linesPerBurst`` (canvas burst height).
    samples_per_burst : int
        Parent ``samplesPerBurst`` (canvas width).

    """

    safe_root: Path
    measurement_path: Path
    annotation_path: Path
    kept_indices: tuple[int, ...]
    lines_per_burst: int
    samples_per_burst: int

    def slot_of(self, burst_index: int) -> int:
        """Return the canvas slot holding an original parent burst index.

        Parameters
        ----------
        burst_index : int
            Original parent burst index.

        Returns
        -------
        int
            Zero-based canvas slot (``row0 = slot * lines_per_burst``).

        Raises
        ------
        Sentinel1ProductError
            If the burst was not kept in this canvas.

        """
        try:
            return self.kept_indices.index(burst_index)
        except ValueError:
            reject_product(
                f"burst {burst_index} is not kept in canvas {self.safe_root}; "
                f"kept={list(self.kept_indices)}"
            )


def _checked_samples(
    window: BurstFetchedWindow,
    lines: int,
    samples: int,
) -> np.ndarray:
    """Return the window samples as complex64, rejecting bad geometry."""
    array = np.asarray(window.samples)
    if array.ndim != 2:
        reject_product(
            f"fetched window for burst {window.burst_index} must be 2-D, "
            f"got shape {array.shape}"
        )
    n_lines, n_cols = array.shape
    row0, col0 = int(window.row0), int(window.col0)
    if n_lines <= 0 or n_cols <= 0:
        reject_product(
            f"fetched window for burst {window.burst_index} must be non-empty, "
            f"got shape {array.shape}"
        )
    if row0 < 0 or col0 < 0:
        reject_product(
            f"fetched window for burst {window.burst_index} has negative "
            f"origin ({row0}, {col0})"
        )
    if row0 + n_lines > lines or col0 + n_cols > samples:
        reject_product(
            f"fetched window for burst {window.burst_index} at ({row0}, {col0}) "
            f"size {array.shape} exceeds parent burst geometry "
            f"({lines}, {samples})"
        )
    return np.ascontiguousarray(array, dtype=np.complex64)


def _window_mask(
    window: BurstFetchedWindow,
    lines_per_burst: int,
) -> tuple[list[int], list[int]]:
    """Return per-line ``(first, last)`` valid samples for one fetched window.

    Fetched lines carry the fetched column envelope; unfetched lines stay
    ``(-1, -1)`` so ``from_safes`` treats them as invalid azimuth lines.
    """
    samples = np.asarray(window.samples)
    n_lines, n_cols = samples.shape
    row0, col0 = int(window.row0), int(window.col0)
    first = [-1] * lines_per_burst
    last = [-1] * lines_per_burst
    for line in range(row0, row0 + n_lines):
        first[line] = col0
        last[line] = col0 + n_cols - 1
    return first, last


def _burst_elements(root: ET.Element) -> tuple[ET.Element, list[ET.Element]]:
    """Return the ``burstList`` element and its ``burst`` children."""
    swath_timing = next(c for c in root if _local(c.tag) == "swathTiming")
    burst_list = next(c for c in swath_timing if _local(c.tag) == "burstList")
    bursts = [c for c in burst_list if _local(c.tag) == "burst"]
    return burst_list, bursts


def _child(parent: ET.Element, name: str) -> ET.Element:
    """Return the first child element with the given local tag."""
    for child in parent:
        if _local(child.tag) == name:
            return child
    message = f"missing required annotation element: {name}"
    reject_product(message)
    raise AssertionError(message)


def _set_valid_samples(
    burst_el: ET.Element,
    first: Sequence[int],
    last: Sequence[int],
) -> None:
    """Rewrite one burst element's ``firstValidSample`` / ``lastValidSample``."""
    _child(burst_el, "firstValidSample").text = " ".join(str(v) for v in first)
    _child(burst_el, "lastValidSample").text = " ".join(str(v) for v in last)


def _remap_geolocation_grid(
    root: ET.Element,
    slot_by_index: dict[int, int],
    lines_per_burst: int,
) -> None:
    """Remap grid points to kept-burst canvas lines by original burst identity.

    Grid points are attributed to bursts by ``line // lines_per_burst`` in the
    parent image and rewritten to ``slot * lines_per_burst + burst_local`` so
    trimmed bursts do not inherit earlier bursts' footprints.  Points of
    dropped bursts are removed.
    """
    grid = None
    for child in root:
        if _local(child.tag) == "geolocationGrid":
            grid = child
            break
    if grid is None:
        return

    def _walk(
        parent: ET.Element,
    ) -> Iterator[tuple[ET.Element, ET.Element]]:
        for child in list(parent):
            yield parent, child
            yield from _walk(child)

    for parent, child in list(_walk(grid)):
        if _local(child.tag) != "geolocationGridPoint":
            continue
        line_el = _child(child, "line")
        line = int((line_el.text or "0").strip())
        block, local_line = divmod(line, lines_per_burst)
        if block not in slot_by_index:
            parent.remove(child)
            continue
        line_el.text = str(slot_by_index[block] * lines_per_burst + local_line)

    for element in grid.iter():
        if _local(element.tag) != "geolocationGridPointList":
            continue
        count = sum(1 for c in element if _local(c.tag) == "geolocationGridPoint")
        element.set("count", str(count))


def _build_canvas_annotation(
    xml_text: str,
    kept: Sequence[tuple[int, tuple[list[int], list[int]]]],
    lines_per_burst: int,
) -> bytes:
    """Return annotation XML for the canvas tree.

    Parameters
    ----------
    xml_text : str
        Parent annotation XML document body.
    kept : sequence of (int, (list of int, list of int))
        Ascending ``(original burst index, (first, last))`` pairs; the mask
        lists have ``lines_per_burst`` entries each.
    lines_per_burst : int
        Parent ``linesPerBurst``.

    Returns
    -------
    bytes
        Rewritten annotation XML with only kept bursts, the fetched-column
        valid-sample mask, ``numberOfLines = n_kept * linesPerBurst``, and a
        geolocation grid remapped by original burst identity.

    """
    root = ET.fromstring(xml_text)
    slot_by_index = {index: slot for slot, (index, _) in enumerate(kept)}
    mask_by_index = dict(kept)

    burst_list, bursts = _burst_elements(root)
    if max(slot_by_index, default=-1) >= len(bursts):
        reject_product(
            f"kept burst indices {sorted(slot_by_index)} exceed the parent "
            f"burstList count {len(bursts)}"
        )
    for index, burst_el in enumerate(bursts):
        if index not in slot_by_index:
            burst_list.remove(burst_el)
            continue
        first, last = mask_by_index[index]
        _set_valid_samples(burst_el, first, last)
    burst_list.set("count", str(len(kept)))

    image_info = _child(_child(root, "imageAnnotation"), "imageInformation")
    _child(image_info, "numberOfLines").text = str(len(kept) * lines_per_burst)

    _remap_geolocation_grid(root, slot_by_index, lines_per_burst)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _copy_calibration(swath: S1Swath, safe_root: Path, stem: str) -> None:
    """Copy calibration/noise XMLs when the source is a directory SAFE."""
    src_dir = Path(str(swath.annotation_path)).parent
    cal_src = src_dir / "calibration"
    if cal_src.is_dir():
        cal_dst = safe_root / "annotation" / "calibration"
        cal_dst.mkdir(parents=True, exist_ok=True)
        for name in (f"calibration-{stem}.xml", f"noise-{stem}.xml"):
            candidate = cal_src / name
            if candidate.exists():
                shutil.copy2(candidate, cal_dst / name)


def write_roi_safe_canvas(
    swath: S1Swath,
    windows: Sequence[BurstFetchedWindow],
    out_dir: str | Path,
    *,
    annotation_xml: str | None = None,
    product_id: str | None = None,
) -> RoiSafeCanvas:
    """Write a SAFE-subset canvas tree for ROI-fetched bursts.

    Every kept burst gets a full-burst-height x full-``samplesPerBurst``
    zero canvas with its fetched rectangle placed at the original
    burst-local ``(line, column)``.  The rewritten annotation marks
    unfetched lines ``firstValidSample = -1`` and ROI lines with the fetched
    column envelope, keeps parent ``linesPerBurst`` / ``azimuthTime`` /
    ``byteOffset`` / ``samplesPerBurst``, sets
    ``numberOfLines = n_kept * linesPerBurst``, and remaps the geolocation
    grid by original burst identity.

    Parameters
    ----------
    swath : S1Swath
        Parsed *parent* sub-swath whose burst geometry is inherited.
    windows : sequence of BurstFetchedWindow
        Fetched rectangles, one per kept burst.  Order does not matter;
        canvas slots follow ascending ``burst_index``.
    out_dir : str or pathlib.Path
        Parent directory; a ``<product_id>.SAFE`` folder is created inside.
    annotation_xml : str, optional
        Parent annotation XML text.  Defaults to reading
        ``swath.annotation_path`` (directory or ``/vsizip/`` source).
    product_id : str, optional
        SAFE product id (defaults to the annotation stem).

    Returns
    -------
    RoiSafeCanvas
        Handle to the generated tree, including the original-index to slot
        mapping needed by :func:`splice_overlap_into_canvas`.

    Raises
    ------
    Sentinel1ProductError
        If ``windows`` is empty, names an unknown or duplicate burst, or a
        fetched rectangle exceeds the parent burst geometry.

    Examples
    --------
    >>> canvas = write_roi_safe_canvas(swath, [window], out_dir)
    >>> canvas.measurement_path.name.endswith(".tiff")
    True

    """
    if not windows:
        reject_product("write_roi_safe_canvas requires at least one fetched window")
    ordered = sorted(windows, key=lambda w: int(w.burst_index))
    kept_indices = tuple(int(w.burst_index) for w in ordered)
    if len(set(kept_indices)) != len(kept_indices):
        reject_product(f"duplicate burst indices in fetched windows: {kept_indices}")

    lines_per_burst = int(swath.lines_per_burst)
    samples_per_burst = int(swath.samples_per_burst)
    canvas = np.zeros(
        (len(ordered) * lines_per_burst, samples_per_burst),
        dtype=np.complex64,
    )
    kept_masks: list[tuple[int, tuple[list[int], list[int]]]] = []
    for slot, window in enumerate(ordered):
        if window.burst_index < 0 or window.burst_index >= len(swath.bursts):
            reject_product(
                f"burst_index {window.burst_index} out of range for {swath.swath}"
            )
        samples = _checked_samples(window, lines_per_burst, samples_per_burst)
        n_lines, n_cols = samples.shape
        row0, col0 = int(window.row0), int(window.col0)
        row_base = slot * lines_per_burst
        canvas[row_base + row0 : row_base + row0 + n_lines, col0 : col0 + n_cols] = (
            samples
        )
        kept_masks.append(
            (int(window.burst_index), _window_mask(window, lines_per_burst))
        )

    out_dir = Path(out_dir)
    pid = product_id or _annotation_stem(swath.annotation_path)
    stem = _annotation_stem(swath.annotation_path)
    safe_root = out_dir / f"{pid}.SAFE"
    (safe_root / "measurement").mkdir(parents=True, exist_ok=True)
    (safe_root / "annotation").mkdir(parents=True, exist_ok=True)

    tiff_path = safe_root / "measurement" / f"{stem}.tiff"
    _write_burst_tiff(tiff_path, canvas)

    source = (
        annotation_xml if annotation_xml is not None else _read_annotation_source(swath)
    )
    rewritten = _build_canvas_annotation(source, kept_masks, lines_per_burst)
    ann_path = safe_root / "annotation" / f"{stem}.xml"
    ann_path.write_bytes(rewritten)

    _copy_calibration(swath, safe_root, stem)

    logger.info(
        "Wrote ROI SAFE canvas %s: %d kept burst(s) %s, shape %s",
        safe_root,
        len(ordered),
        list(kept_indices),
        canvas.shape,
    )
    return RoiSafeCanvas(
        safe_root=safe_root,
        measurement_path=tiff_path,
        annotation_path=ann_path,
        kept_indices=kept_indices,
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
    )


def _read_valid_samples(burst_el: ET.Element) -> tuple[list[int], list[int]]:
    """Return one burst element's ``firstValidSample`` / ``lastValidSample`` lists."""
    first = [
        int(part) for part in (_child(burst_el, "firstValidSample").text or "").split()
    ]
    last = [
        int(part) for part in (_child(burst_el, "lastValidSample").text or "").split()
    ]
    return first, last


def _overlap_rectangle(
    samples: np.ndarray | bytes | bytearray | memoryview,
    window: BurstOverlapWindow,
) -> np.ndarray:
    """Return the fetched overlap rectangle as complex64.

    Completeness is the parent overlap rectangle (full width of the window,
    ``n_lines`` rows); anything else is rejected.
    """
    expected = (int(window.n_lines), int(window.n_samples))
    if isinstance(samples, np.ndarray):
        array = np.asarray(samples)
    else:
        try:
            array = np.frombuffer(samples, dtype=np.complex64)
        except ValueError:
            reject_product("overlap samples are not a complex64 buffer")
        if array.size == expected[0] * expected[1]:
            array = array.reshape(expected)
        else:
            array = array.reshape(-1)
    if array.shape != expected:
        reject_product(
            "overlap completeness is the parent overlap rectangle: "
            f"expected shape {expected}, got shape {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array, dtype=np.complex64)


def splice_overlap_into_canvas(
    canvas: RoiSafeCanvas,
    window: BurstOverlapWindow,
    *,
    role: Literal["top", "bottom"],
    samples: np.ndarray | bytes | bytearray | memoryview,
) -> None:
    """Splice one fetched overlap rectangle into a canvas tree.

    The rectangle is written at the burst's *original* burst-local rows
    (``top_row0`` for ``role="top"``, ``bottom_row0`` for ``role="bottom"``)
    inside the burst's canvas slot, so the raster height never changes.
    Afterwards the spliced lines' ``firstValidSample`` / ``lastValidSample``
    are rewritten to the parent overlap-window valid samples
    (``col0`` through ``col0 + n_samples - 1``) before the tree is handed to
    ``from_safes``.

    Parameters
    ----------
    canvas : RoiSafeCanvas
        Canvas tree previously written by :func:`write_roi_safe_canvas`.
    window : BurstOverlapWindow
        Parent overlap window (original burst indices and burst-local rows).
    role : {"top", "bottom"}
        Which burst of the overlap pair ``samples`` belongs to.
    samples : numpy.ndarray or bytes-like
        Fetched rectangle of shape ``(window.n_lines, window.n_samples)``
        (complex64).  Bytes-like input is reinterpreted as complex64.

    Raises
    ------
    Sentinel1ProductError
        If the role is unknown, the burst is not kept, the rectangle is not
        the full parent overlap rectangle, or the window falls outside the
        burst geometry.

    """
    if role == "top":
        burst_index = int(window.top_index)
        row0 = int(window.top_row0)
    elif role == "bottom":
        burst_index = int(window.bottom_index)
        row0 = int(window.bottom_row0)
    else:
        reject_product(f"unknown overlap role {role!r}; expected 'top' or 'bottom'")

    slot = canvas.slot_of(burst_index)
    rectangle = _overlap_rectangle(samples, window)
    n_lines, n_samples = rectangle.shape
    col0 = int(window.col0)
    if row0 < 0 or col0 < 0:
        reject_product(
            f"overlap window {window.top_index}/{window.bottom_index} has "
            f"negative origin ({row0}, {col0})"
        )
    if row0 + n_lines > canvas.lines_per_burst or col0 + n_samples > (
        canvas.samples_per_burst
    ):
        reject_product(
            f"overlap window {window.top_index}/{window.bottom_index} at "
            f"({row0}, {col0}) size {rectangle.shape} exceeds burst geometry "
            f"({canvas.lines_per_burst}, {canvas.samples_per_burst})"
        )

    row_base = slot * canvas.lines_per_burst

    # Validate the annotation state before mutating the raster.
    xml_text = canvas.annotation_path.read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)
    _burst_list, bursts = _burst_elements(root)
    if slot >= len(bursts):
        reject_product(
            f"canvas annotation {canvas.annotation_path} has {len(bursts)} "
            f"bursts; slot {slot} for burst {burst_index} is out of range"
        )
    first, last = _read_valid_samples(bursts[slot])
    if len(first) != canvas.lines_per_burst or len(last) != canvas.lines_per_burst:
        reject_product(
            f"canvas annotation {canvas.annotation_path} slot {slot} has "
            f"valid-sample lengths ({len(first)}, {len(last)}), expected "
            f"{canvas.lines_per_burst}"
        )
    last_col = col0 + n_samples - 1
    for line in range(row0, row0 + n_lines):
        first[line] = col0
        last[line] = last_col

    with rasterio.open(canvas.measurement_path, "r+") as dst:
        dst.write(
            rectangle,
            1,
            window=Window(
                col_off=col0,
                row_off=row_base + row0,
                width=n_samples,
                height=n_lines,
            ),
        )
    _set_valid_samples(bursts[slot], first, last)
    canvas.annotation_path.write_bytes(
        ET.tostring(root, encoding="utf-8", xml_declaration=True)
    )

    logger.info(
        "Spliced %s overlap %s/%s into %s: burst %s slot %s rows[%s:%s] cols[%s:%s]",
        role,
        window.top_index,
        window.bottom_index,
        canvas.safe_root,
        burst_index,
        slot,
        row_base + row0,
        row_base + row0 + n_lines,
        col0,
        last_col + 1,
    )


__all__ = [
    "BurstFetchedWindow",
    "RoiSafeCanvas",
    "splice_overlap_into_canvas",
    "write_roi_safe_canvas",
]

"""Export extracted bursts as a minimal SAFE-compatible subset directory.

The output mirrors the structure produced by the ``burst2safe`` tool so the
resulting directory can be consumed by ISCE2 ``topsStack`` and other tools
that expect a SAFE product on disk:

    <out_dir>/<product_id>.SAFE/
        measurement/<...>.tiff          complex64 GeoTIFF (selected bursts)
        annotation/<...>.xml            annotation with trimmed burstList
        annotation/calibration/...      calibration + noise XML (copied)
        manifest.safe                   minimal manifest

Only the sub-swaths and bursts requested are written.  This is an upper-layer
convenience on top of :mod:`faninsar.missions.s1.io.extract`.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import Affine

from faninsar.logging import setup_logger
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.io.extract import extract_bursts

if TYPE_CHECKING:
    from faninsar.missions.s1.types import S1Product, S1Swath

logger = setup_logger(__name__)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_annotation_source(swath: S1Swath) -> str:
    """Return the raw annotation XML text for a swath.

    Handles directory annotations (``annotation_path`` on disk) and ZIP-backed
    annotations (``/vsizip/`` URI).
    """
    ap = str(swath.annotation_path)
    if ap.startswith("/vsizip/"):
        # /vsizip//<zip>/<member>
        body = ap[len("/vsizip/") :].lstrip("/")
        sep = body.find(".zip/")
        if sep == -1:
            reject_product(f"unparseable vsizip annotation path: {ap}")
        zip_path = Path(body[: sep + 4])
        member = body[sep + 5 :].lstrip("/")
        with zipfile.ZipFile(zip_path) as archive:
            return archive.read(member).decode("utf-8")
    p = Path(ap)
    if p.exists():
        return p.read_text(encoding="utf-8")
    reject_product(f"annotation source not found: {ap}")
    return ""


def _trim_annotation(xml_text: str, keep_indices: list[int]) -> bytes:
    """Return annotation XML with only the selected bursts in its burstList."""
    root = ET.fromstring(xml_text)
    swath_timing = next(c for c in root if _local(c.tag) == "swathTiming")
    burst_list = next(c for c in swath_timing if _local(c.tag) == "burstList")
    bursts = [c for c in burst_list if _local(c.tag) == "burst"]
    keep_set = set(keep_indices)
    for idx, burst in enumerate(bursts):
        if idx not in keep_set:
            burst_list.remove(burst)
    burst_list.set("count", str(len(keep_indices)))
    # ET.tostring round-trips namespaces well enough for downstream tools;
    # we emit with encoding="utf-8" to match the SAFE convention.
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _write_burst_tiff(
    out_path: Path,
    samples: np.ndarray,
) -> None:
    """Write a complex64 GeoTIFF containing one stitched burst block."""
    height, width = samples.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="complex64",
        tiled=False,
        compress=None,
        crs=None,
        transform=Affine.identity(),
    ) as dst:
        dst.write(np.ascontiguousarray(samples), 1)


def _annotation_stem(annotation_path: str) -> str:
    """Return the annotation filename stem (no extension)."""
    from_path = str(annotation_path).split("/")[-1]
    return from_path.removesuffix(".xml")


def export_swath_bursts(
    swath: S1Swath,
    burst_indices: list[int],
    out_dir: str | Path,
    *,
    product_id: str | None = None,
) -> Path:
    """Export selected bursts of one sub-swath as a minimal SAFE subset.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath whose bursts will be exported.
    burst_indices : list of int
        Burst indices (within ``swath``) to keep, in ascending order.
    out_dir : str or pathlib.Path
        Parent directory; a ``<product_id>.SAFE`` folder is created inside.
    product_id : str, optional
        SAFE product id (defaults to the annotation stem).

    Returns
    -------
    pathlib.Path
        Path to the generated ``.SAFE`` directory.

    Raises
    ------
    Sentinel1ProductError
        If ``burst_indices`` is empty or out of range.

    """
    if not burst_indices:
        reject_product("burst_indices must be non-empty for export")
    for i in burst_indices:
        if i < 0 or i >= len(swath.bursts):
            reject_product(f"burst_index {i} out of range for {swath.swath}")

    out_dir = Path(out_dir)
    pid = product_id or _annotation_stem(swath.annotation_path)
    safe_root = out_dir / f"{pid}.SAFE"
    (safe_root / "measurement").mkdir(parents=True, exist_ok=True)
    (safe_root / "annotation").mkdir(parents=True, exist_ok=True)

    bursts = extract_bursts(swath, burst_indices=burst_indices)
    stacked = np.concatenate([b.samples for b in bursts], axis=0)

    stem = _annotation_stem(swath.annotation_path)
    tiff_path = safe_root / "measurement" / f"{stem}.tiff"
    _write_burst_tiff(tiff_path, stacked)

    xml_text = _read_annotation_source(swath)
    trimmed = _trim_annotation(xml_text, burst_indices)
    ann_path = safe_root / "annotation" / f"{stem}.xml"
    ann_path.write_bytes(trimmed)

    # Copy calibration/noise XMLs when the source is a directory SAFE.
    src_dir = Path(str(swath.annotation_path)).parent
    cal_src = src_dir / "calibration"
    if cal_src.is_dir():
        cal_dst = safe_root / "annotation" / "calibration"
        cal_dst.mkdir(parents=True, exist_ok=True)
        for name in (f"calibration-{stem}.xml", f"noise-{stem}.xml"):
            candidate = cal_src / name
            if candidate.exists():
                shutil.copy2(candidate, cal_dst / name)

    logger.info(
        "Exported %d burst(s) from %s -> %s (%s, %d lines)",
        len(burst_indices),
        swath.swath,
        safe_root,
        stacked.shape,
        stacked.shape[0],
    )
    return safe_root


def export_burst_safe(
    product: S1Product,
    selections: dict[str, list[int]],
    out_dir: str | Path,
) -> Path:
    """Export bursts across multiple sub-swaths of a product.

    Parameters
    ----------
    product : S1Product
        Open SAFE product.
    selections : dict[str, list[int]]
        Mapping ``{swath_name: [burst_index, ...]}``, e.g.
        ``{"IW2": [4, 5, 6]}``.
    out_dir : str or pathlib.Path
        Parent directory for the generated ``.SAFE`` folder.

    Returns
    -------
    pathlib.Path
        Path to the generated ``.SAFE`` directory.

    """
    out_dir = Path(out_dir)
    pid = product.product_id
    safe_root = out_dir / f"{pid}.SAFE"
    safe_root.mkdir(parents=True, exist_ok=True)
    for swath_name, indices in selections.items():
        if not indices:
            continue
        swath = product.swath(swath_name)
        export_swath_bursts(
            swath,
            indices,
            safe_root.parent,
            product_id=f"{pid}_{swath_name}",
        )
    return safe_root

"""On-disk manifest for dataset file metadata.

A manifest is a small JSON file (``faninsar.manifest.json``) written next to
the dataset's ``root_dir``.  It stores the per-file metadata produced by the
initial filesystem scan so subsequent instantiations of the same dataset can
skip the expensive per-file ``rasterio.open`` pass and restore the state
directly.

Design contract
---------------
* **Location** — always ``<root_dir>/faninsar.manifest.json``.  There is no
  user-facing cache directory; the file travels with the data.
* **Ownership** — entries are keyed by a human-readable dataset slug
  (``_manifest_name()`` on the instance) so several datasets sharing one
  ``root_dir`` (e.g. unw / coh / LOS) coexist in one file.
* **Validation** — an entry is trusted only when (a) the envelope passes the
  ``faninsar`` / ``schema`` gate, (b) the stored ``paths`` list (after the
  class-specific path-resolution hook) matches the paths resolved for the
  current instantiation.  Anything else falls back to a full rescan and the
  entry is rewritten.
* **Raw facts only** — entries store per-file *source* metadata
  (``file_crs``, ``file_bounds``, ``file_res``, ``file_dtype``,
  ``file_nodata``, ``width``, ``height``, ``colormap``) plus class-specific
  extras (``pair`` / ``date``).  The transformed ``crs`` / ``bounds`` /
  ``res`` columns are *recomputed* on restore via
  :func:`rasterio.warp.calculate_default_transform`, so a manifest written
  with ``crs=None`` still serves an instantiation that passes an explicit
  ``crs``.
* **Corruption is silent** — a malformed JSON file or a missing key is
  treated as a cache miss, never as an error.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio.crs
from pyproj.crs import CRS
from rasterio.warp import calculate_default_transform

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, MutableMapping
    from os import PathLike

MANIFEST_FILENAME = "faninsar.manifest.json"
MANIFEST_SCHEMA = 1


def manifest_path(root_dir: str | PathLike) -> Path:
    """Return the manifest path for a dataset rooted at ``root_dir``."""
    return Path(root_dir) / MANIFEST_FILENAME


def slugify(name: str) -> str:
    """Return a lowercase slug for ``name`` used as a manifest section key."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "dataset"


def encode_file_row(row: MutableMapping[str, Any]) -> dict[str, Any]:
    """Serialize one file-metadata dict to JSON-friendly primitives."""
    out: dict[str, Any] = {}
    for key, val in row.items():
        if val is None or isinstance(val, (str, bool, int, float)):
            out[key] = val
        elif isinstance(val, (CRS, rasterio.crs.CRS)):
            out[key] = {"__crs__": val.to_wkt()}
        elif isinstance(val, np.generic):
            out[key] = val.item()
        elif isinstance(val, (tuple, list)) and all(
            isinstance(v, (int, float, np.integer, np.floating)) for v in val
        ):
            out[key] = list(val)
        elif isinstance(val, dict) and all(isinstance(k, int) for k in val):
            out[key] = {str(k): list(v) for k, v in val.items()}
        else:
            out[key] = val
    return out


def decode_file_row(row: MutableMapping[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`encode_file_row`."""
    out: dict[str, Any] = {}
    for key, val in row.items():
        if isinstance(val, dict) and "__crs__" in val:
            out[key] = CRS.from_wkt(val["__crs__"])
        elif (
            key in {"bounds", "file_bounds", "res", "file_res"}
            and isinstance(val, list)
            and len(val) in {2, 4}
        ):
            out[key] = tuple(val)
        elif key == "colormap" and isinstance(val, dict):
            out[key] = {int(k): tuple(v) for k, v in val.items()}
        else:
            out[key] = val
    return out


def _transformed_bounds_and_res(
    file_crs: CRS,
    file_bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    target_crs: CRS,
) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    """Reproduce ``WarpedVRT`` bounds/res without opening the file.

    The metadata scan uses :class:`rasterio.vrt.WarpedVRT` when ``target_crs``
    differs from the file CRS.  ``calculate_default_transform`` computes the
    same geotransform and output shape deterministically, so the cached raw
    (``file_crs``, ``file_bounds``, ``file_res``, ``width``, ``height``) is
    sufficient to reconstruct the transformed bounds/res exactly.
    """
    tf, w, h = calculate_default_transform(
        file_crs, target_crs, width, height, *file_bounds
    )
    # WarpedVRT bounds are the raster pixel-grid box in the target CRS, i.e.
    # the default transform applied to the pixel corners.
    bounds = (tf.c, tf.f + h * tf.e, tf.c + w * tf.a, tf.f)
    res = (abs(tf.a), abs(tf.e))
    return bounds, res


def _restore_files_rows(
    raw_files: list[dict[str, Any]],
    paths: list[str],
    target_crs: CRS | None,
) -> list[dict[str, Any]]:
    """Rebuild the per-file metadata dicts expected by ``_process_scan_results``.

    The manifest stores *raw* file facts; this layer re-derives the
    ``crs`` / ``bounds`` / ``res`` columns (and any extra class keys such as
    ``pair``/``date`` survive untouched).
    """
    out: list[dict[str, Any]] = []
    for raw, current_path in zip(raw_files, paths, strict=True):
        row = decode_file_row(raw)
        # Use the current instantiation's path form (e.g. a freshly built
        # HDF5 subdataset URI) rather than the stored one.
        row["paths"] = str(current_path)
        if not row.get("valid", False):
            # Invalid rows keep all None fields from the manifest.
            for k in (
                "file_crs",
                "file_bounds",
                "file_res",
                "file_dtype",
                "file_nodata",
                "crs",
                "bounds",
                "res",
                "colormap",
            ):
                row.setdefault(k, None)
            out.append(row)
            continue
        file_crs = row["file_crs"]
        file_bounds = tuple(row["file_bounds"])
        width = int(row["width"])
        height = int(row["height"])
        if target_crs is not None and target_crs != file_crs:
            bounds, res = _transformed_bounds_and_res(
                file_crs, file_bounds, width, height, target_crs
            )
            row["crs"] = target_crs
            row["bounds"] = bounds
            row["res"] = res
        else:
            row["crs"] = file_crs
            row["bounds"] = file_bounds
            row["res"] = tuple(row["file_res"])
        out.append(row)
    return out


def _normalize_paths_for_compare(
    paths: Iterable[str],
    to_comparable_path: Callable[[str], str | list[str]],
) -> list[str]:
    """Map raw paths (possibly HDF5 subdataset URIs) to stat-able strings."""
    out = []
    for p in paths:
        comparable = to_comparable_path(p)
        if isinstance(comparable, Path):
            out.append(str(comparable))
        elif isinstance(comparable, (list, tuple)):
            out.extend(str(x) for x in comparable)
        else:
            out.append(str(comparable))
    return out


def read_raw_entry(  # noqa: PLR0911
    root_dir: str | PathLike, name: str
) -> dict[str, Any] | None:
    """Return the raw ``{paths, files}`` entry for ``name``, or None."""
    path = manifest_path(root_dir)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(doc, dict)
        or doc.get("faninsar") is None
        or doc.get("schema") != MANIFEST_SCHEMA
    ):
        return None
    datasets = doc.get("datasets")
    if not isinstance(datasets, dict):
        return None
    entry = datasets.get(name)
    if not isinstance(entry, dict):
        return None
    entry_paths = entry.get("paths")
    raw_files = entry.get("files")
    if not isinstance(entry_paths, list) or not isinstance(raw_files, list):
        return None
    return {"paths": entry_paths, "files": raw_files}


def read_entry(
    root_dir: str | PathLike,
    name: str,
    paths: list[str],
    to_comparable_path: Callable[[str], str | list[str]],
    target_crs: CRS | None,
) -> list[dict[str, Any]] | None:
    """Return restored file metadata if the manifest has a usable entry.

    Parameters
    ----------
    root_dir : str or PathLike
        Dataset root directory; the manifest lives directly under it.
    name : str
        Dataset slug returned by ``_manifest_name``.
    paths : list of str
        Paths resolved for the current instantiation, *before* the
        ``to_comparable_path`` normalization.
    to_comparable_path : callable
        Maps one path to a filesystem-comparable string (or list of strings).
        ``RasterDataset`` returns the path unchanged; ``NisarH5Mixin`` maps
        ``hdf5:"..."`` URIs to the backing granule file.
    target_crs : CRS or None
        The ``crs`` argument the caller passed to ``__init__``.

    Returns
    -------
    list of dict or None
        Decoded per-file metadata ready for ``_process_scan_results``, or
        ``None`` when the entry is missing or stale.

    """
    entry = read_raw_entry(root_dir, name)
    if entry is None:
        return None
    entry_paths = entry["paths"]
    raw_files = entry["files"]
    if len(raw_files) != len(paths):
        return None
    # Compare paths after normalization: the manifest stores the same
    # pre-normalization form (possibly ``hdf5:"..."`` URIs) as the scan.
    expected = _normalize_paths_for_compare(entry_paths, to_comparable_path)
    actual = _normalize_paths_for_compare(paths, to_comparable_path)
    if expected != actual:
        return None
    return _restore_files_rows(raw_files, paths, target_crs)


def write_entry(
    root_dir: str | PathLike,
    name: str,
    paths: list[str],
    file_metadata: list[dict[str, Any]],
    version: str,
) -> None:
    """Upsert ``name``'s entry into ``root_dir/faninsar.manifest.json``.

    The write is atomic (write to ``*.tmp`` then :func:`os.replace`).  Other
    entries already present in the file are preserved.
    """
    path = manifest_path(root_dir)
    doc: dict[str, Any] = {"faninsar": version, "schema": MANIFEST_SCHEMA}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                doc["datasets"] = existing.get("datasets") or {}
        except (OSError, json.JSONDecodeError):
            doc["datasets"] = {}
    datasets = doc.setdefault("datasets", {})
    datasets[name] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "paths": [str(p) for p in paths],
        "files": [encode_file_row(row) for row in file_metadata],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "manifest_path",
    "read_entry",
    "slugify",
    "write_entry",
]

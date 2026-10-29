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
  ``faninsar`` / ``schema`` gate, and (b) the stored ``paths`` list has the
  same length and same first element as the current instantiation's paths.
  Anything else falls back to a full rescan and the entry is rewritten.
* **Baked values** — each entry stores the *resolved* ``crs`` / ``res`` /
  ``dtype`` / ``nodata`` / ``bounds`` captured at the end of ``__init__``.
  Restoring is a lookup, not a recompute — the caller's ``crs`` argument is
  ignored on a hit.
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

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from os import PathLike

MANIFEST_FILENAME = "faninsar.manifest.json"
MANIFEST_SCHEMA = 1

_CRS_CACHE: dict[str, CRS] = {}


def _crs_from_wkt(wkt: str) -> CRS:
    """Return a CRS for ``wkt``, memoized."""
    crs = _CRS_CACHE.get(wkt)
    if crs is None:
        crs = CRS.from_wkt(wkt)
        _CRS_CACHE[wkt] = crs
    return crs


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
        elif isinstance(val, np.dtype):
            out[key] = {"__dtype__": val.str}
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
            out[key] = _crs_from_wkt(val["__crs__"])
        elif isinstance(val, dict) and "__dtype__" in val:
            out[key] = np.dtype(val["__dtype__"])
        elif key in {"bounds", "file_bounds", "res", "file_res"} and isinstance(
            val, list
        ):
            out[key] = tuple(val)
        elif key == "colormap" and isinstance(val, dict):
            out[key] = {int(k): tuple(v) for k, v in val.items()}
        else:
            out[key] = val
    return out


def _decode_files_rows(
    raw_files: list[dict[str, Any]],
    paths: list[str],
    entry_crs: CRS | None,
    entry_bounds: tuple[float, float, float, float] | None,
    entry_res: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    """Rebuild per-file metadata dicts expected by ``_process_scan_results``.

    The manifest stores the *resolved* ``crs`` / ``bounds`` / ``res`` at
    entry level; every valid row takes those values verbatim — no transform
    is recomputed on restore.
    """
    out: list[dict[str, Any]] = []
    for raw, current_path in zip(raw_files, paths, strict=True):
        row = decode_file_row(raw)
        row["paths"] = str(current_path)
        if not row.get("valid", False):
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
        else:
            row["crs"] = entry_crs
            row["bounds"] = entry_bounds
            row["res"] = entry_res
        out.append(row)
    return out


def _paths_match(entry_paths: list[str], paths: list[str]) -> bool:
    """Cheap sanity check: length and the first path must agree."""
    if len(entry_paths) != len(paths):
        return False
    if not entry_paths:
        return True
    return str(entry_paths[0]) == str(paths[0])


def read_entry(  # noqa: PLR0911
    root_dir: str | PathLike,
    name: str,
    paths: list[str],
) -> dict[str, Any] | None:
    """Return a manifest-restored snapshot for ``name``, or None on a miss.

    On a hit the dict carries:
      - ``files``: per-file metadata dicts ready for ``_process_scan_results``
      - ``attrs``: the resolved ``crs`` / ``res`` / ``dtype`` / ``nodata`` /
        ``bounds`` recorded when the manifest was written
      - ``paths``: the stored path list
    """
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
    attrs_raw = entry.get("attrs")
    if (
        not isinstance(entry_paths, list)
        or not isinstance(raw_files, list)
        or not isinstance(attrs_raw, dict)
        or len(raw_files) != len(paths)
        or not _paths_match(entry_paths, paths)
    ):
        return None
    entry_crs = attrs_raw.get("crs")
    if isinstance(entry_crs, dict) and "__crs__" in entry_crs:
        entry_crs = _crs_from_wkt(entry_crs["__crs__"])
    entry_bounds = attrs_raw.get("bounds")
    if isinstance(entry_bounds, list):
        entry_bounds = tuple(entry_bounds)
    entry_res = attrs_raw.get("res")
    if isinstance(entry_res, list):
        entry_res = tuple(entry_res)
    files = _decode_files_rows(raw_files, paths, entry_crs, entry_bounds, entry_res)
    attrs = dict(attrs_raw)
    attrs["crs"] = entry_crs
    attrs["bounds"] = entry_bounds
    attrs["res"] = entry_res
    return {"paths": entry_paths, "files": files, "attrs": attrs}


def write_entry(
    root_dir: str | PathLike,
    name: str,
    paths: list[str],
    file_metadata: list[dict[str, Any]],
    attrs: dict[str, Any],
    version: str,
) -> None:
    """Upsert ``name``'s entry into ``root_dir/faninsar.manifest.json``.

    ``attrs`` holds the resolved output attributes (``crs``, ``res``,
    ``dtype``, ``nodata``, ``bounds``) captured at the end of ``__init__``;
    restoring them verbatim skips any recomputation on a manifest hit.

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
        "attrs": {k: encode_file_row({"v": v})["v"] for k, v in attrs.items()},
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

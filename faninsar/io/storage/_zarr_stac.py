"""Small helpers for canonical Zarr payloads and local STAC records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping


def digest_path(path: Path) -> str:
    """Return a deterministic SHA-256 digest for a file or directory tree."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir() or path.is_symlink():
        reject_invalid_state(f"Zarr asset is missing or unsafe: {path}")
    # Generation leases expose a descriptor-backed Path whose filesystem
    # spelling is ``/dev/fd/...``.  Use its stable display path for recursive
    # Zarr traversal; the lease still pins the immutable generation lifetime.
    root = os.fspath(getattr(path, "_display_path", path))
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directories.sort()
        for filename in sorted(filenames):
            child = Path(directory) / filename
            if child.is_symlink():
                reject_invalid_state(f"Zarr asset contains a symbolic link: {child}")
            relative = os.path.relpath(child, root).replace(os.sep, "/").encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def write_zarr_array(path: Path, value: np.ndarray) -> None:
    """Write one array as a self-contained Zarr directory."""
    import zarr

    if value.ndim == 0 or value.dtype.hasobject:
        reject_invalid_state("Zarr payload must be a non-scalar non-object array")
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            reject_invalid_state(f"Zarr payload path is unsafe: {path}")
        for child in path.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    chunks = tuple(max(1, min(int(size), 256)) for size in value.shape)
    array = zarr.open_array(
        store=str(path), mode="w", shape=value.shape, chunks=chunks, dtype=value.dtype
    )
    array[...] = value
    array.attrs["faninsar:format"] = "stac-zarr"


def read_zarr_array(path: Path) -> np.ndarray:
    """Read one validated Zarr array from a canonical asset directory."""
    import zarr

    if not path.is_dir() or path.is_symlink():
        reject_invalid_state(f"Zarr payload is missing or unsafe: {path}")
    try:
        return np.asarray(zarr.open_array(store=str(path), mode="r"))
    except (OSError, ValueError, TypeError) as error:
        reject_invalid_state(f"Zarr payload cannot be read: {error}")


def validate_spatial_metadata(
    domain: object,
    shape: tuple[int, int],
    grid_identity: object,
    *,
    crs: object = None,
    transform: object = None,
) -> tuple[str, tuple[int, int], str, str | None, tuple[float, ...] | None]:
    """Validate the shared radar/geo spatial metadata contract."""
    if domain not in {"radar", "geo"}:
        reject_invalid_state("artifact domain must be radar or geo")
    if len(shape) != 2 or any(not isinstance(size, int) or size <= 0 for size in shape):
        reject_invalid_state("artifact shape must be a positive 2-D shape")
    if (
        not isinstance(grid_identity, str)
        or len(grid_identity) != 64
        or any(character not in "0123456789abcdef" for character in grid_identity)
    ):
        reject_invalid_state("artifact grid_identity must be a lowercase SHA-256")
    if domain == "geo":
        if crs is None or transform is None:
            reject_invalid_state("Geo artifact requires CRS and transform metadata")
        try:
            normalized_transform = tuple(float(value) for value in transform)
        except (TypeError, ValueError):
            reject_invalid_state("Geo artifact transform must be numeric")
        if len(normalized_transform) != 6:
            reject_invalid_state("Geo artifact transform must have six values")
        normalized_crs = str(crs)
        if not normalized_crs:
            reject_invalid_state("Geo artifact CRS must not be empty")
        return str(domain), shape, grid_identity, normalized_crs, normalized_transform
    if crs is not None or transform is not None:
        reject_invalid_state("Radar artifacts must not carry geographic CRS metadata")
    return str(domain), shape, grid_identity, None, None


def write_stac_item(
    path: Path,
    *,
    item_id: str,
    assets: Mapping[str, str],
    domain: str,
    shape: tuple[int, int],
    grid_identity: str,
    crs: str | None = None,
    transform: tuple[float, ...] | None = None,
) -> None:
    """Write a compact STAC Item after all referenced assets are complete."""
    properties: dict[str, Any] = {
        "faninsar:domain": domain,
        "faninsar:shape": list(shape),
        "faninsar:grid_identity": grid_identity,
    }
    if crs is not None:
        properties["proj:code"] = crs
    if transform is not None:
        properties["proj:transform"] = list(transform)
    item = {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": item_id,
        "geometry": None,
        "bbox": None,
        "properties": properties,
        "assets": {
            name: {"href": href, "roles": ["data"]} for name, href in assets.items()
        },
        "links": [],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(item, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_stac_item(
    path: Path,
    *,
    item_id: str,
    domain: str,
    shape: tuple[int, int],
    grid_identity: str,
    crs: str | None = None,
    transform: tuple[float, ...] | None = None,
) -> None:
    """Validate the local STAC Item and every referenced asset path."""
    if not path.is_file() or path.is_symlink():
        reject_invalid_state(f"STAC record is missing or unsafe: {path}")
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        reject_invalid_state(f"STAC record cannot be read: {error}")
    if (
        not isinstance(item, dict)
        or item.get("type") != "Feature"
        or item.get("id") != item_id
        or item.get("geometry") is not None
    ):
        reject_invalid_state("STAC record identity or geometry is invalid")
    properties = item.get("properties")
    if not isinstance(properties, dict):
        reject_invalid_state("STAC record properties are invalid")
    if (
        properties.get("faninsar:domain") != domain
        or properties.get("faninsar:shape") != list(shape)
        or properties.get("faninsar:grid_identity") != grid_identity
    ):
        reject_invalid_state("STAC record spatial metadata differs from manifest")
    if domain == "geo" and (
        properties.get("proj:code") != crs
        or properties.get("proj:transform") != list(transform or ())
    ):
        reject_invalid_state("STAC record geo metadata is incomplete")
    if domain == "radar" and any(
        key in properties for key in ("proj:code", "proj:transform")
    ):
        reject_invalid_state("Radar STAC record must not fabricate CRS metadata")
    assets = item.get("assets")
    if not isinstance(assets, dict) or not assets:
        reject_invalid_state("STAC record assets are missing")
    for asset in assets.values():
        if not isinstance(asset, dict) or not isinstance(asset.get("href"), str):
            reject_invalid_state("STAC record asset is invalid")
        target = path.parent / asset["href"]
        if not target.is_dir() or target.is_symlink():
            reject_invalid_state(f"STAC record asset is missing or unsafe: {target}")


__all__ = [
    "digest_path",
    "read_zarr_array",
    "validate_spatial_metadata",
    "validate_stac_item",
    "write_stac_item",
    "write_zarr_array",
]

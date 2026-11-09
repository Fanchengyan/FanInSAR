"""Small helpers for canonical Zarr payloads and local STAC records."""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping


def inspect_zarr_array_metadata(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    """Return ``(shape, dtype)`` from Zarr metadata without reading samples.

    Parameters
    ----------
    path : pathlib.Path
        Canonical Zarr array directory.

    Returns
    -------
    tuple of tuple of int and numpy.dtype
        Declared array shape and dtype.

    Raises
    ------
    InvalidProcessingStateError
        If the asset is missing, unsafe, or its array metadata cannot be read.

    """
    import zarr

    if not path.is_dir() or path.is_symlink():
        reject_invalid_state(f"Zarr payload is missing or unsafe: {path}")
    try:
        array = zarr.open_array(store=str(path), mode="r")
        shape = tuple(int(size) for size in array.shape)
        return shape, np.dtype(array.dtype)
    except (OSError, ValueError, TypeError, KeyError) as error:
        reject_invalid_state(f"Zarr payload cannot be read: {error}")


def require_basename(value: object, field: str) -> str:
    """Return ``value`` when it is a single path basename, else reject."""
    if not isinstance(value, str):
        reject_invalid_state(f"{field} must be a filename component")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != value
        or value in {"", ".", ".."}
    ):
        reject_invalid_state(f"{field} must be a basename: {value!r}")
    return value


def normalize_read_window(
    window: object,
    shape: tuple[int, ...],
) -> tuple[slice, slice]:
    """Validate a local half-open ``(r0, r1, c0, c1)`` window on a 2-D array."""
    if len(shape) != 2:
        reject_invalid_state("windowed Zarr reads require a 2-D array")
    if not isinstance(window, tuple) or len(window) != 4:
        reject_invalid_state("read window must be a 4-tuple of ints (r0, r1, c0, c1)")
    if any(isinstance(value, bool) for value in window):
        reject_invalid_state("read window must be a 4-tuple of ints (r0, r1, c0, c1)")
    try:
        r0, r1, c0, c1 = (operator.index(value) for value in window)
    except TypeError:
        reject_invalid_state("read window must be a 4-tuple of ints (r0, r1, c0, c1)")
    rows, cols = int(shape[0]), int(shape[1])
    if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
        reject_invalid_state(
            f"read window {window!r} is outside array shape {(rows, cols)}"
        )
    return slice(r0, r1), slice(c0, c1)


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


def read_zarr_array(
    path: Path, window: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    """Read one validated Zarr array, optionally a local 2-D window.

    ``window`` is a half-open ``(r0, r1, c0, c1)`` in the array's own
    coordinates. The slice is taken from the opened Zarr array before
    materializing a full ndarray. ``None`` reads the whole array.

    Parameters
    ----------
    path : pathlib.Path
        Canonical Zarr asset directory.
    window : tuple of int, optional
        Local half-open row/column window.

    Returns
    -------
    numpy.ndarray
        The selected samples.

    Raises
    ------
    InvalidProcessingStateError
        If the asset is missing or unsafe, or ``window`` is not a valid
        in-bounds 2-D selection.

    """
    import zarr

    if not path.is_dir() or path.is_symlink():
        reject_invalid_state(f"Zarr payload is missing or unsafe: {path}")
    try:
        array = zarr.open_array(store=str(path), mode="r")
        if window is None:
            return np.asarray(array)
        row_slice, col_slice = normalize_read_window(
            window, tuple(int(size) for size in array.shape)
        )
        return np.asarray(array[row_slice, col_slice])
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
    "inspect_zarr_array_metadata",
    "normalize_read_window",
    "read_zarr_array",
    "require_basename",
    "validate_spatial_metadata",
    "validate_stac_item",
    "write_stac_item",
    "write_zarr_array",
]

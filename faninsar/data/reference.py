"""Shared reference-point sampling, validity, and mean selection.

This module owns the pixel-centre nearest-sample rule, footprint checks,
valid-point subset policy, and arithmetic-mean offsets used by Stack
inversion and layer plotting (PROPOSAL-0085). Coordinate projection into
a geographic affine or a radar geometry stays with those owners.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from faninsar.core.parameters import Parameter
from faninsar.data.query.points import Points
from faninsar.logging import reject, setup_logger

logger = setup_logger(__name__)

RefPolicy = Literal["common", "per_image"]
WGS84_CRS = "EPSG:4326"
_REFERENCE_METHOD = "arithmetic_mean"


@dataclass(frozen=True, slots=True)
class ReferencePixelMap:
    """Nearest-pixel mapping of requested points onto one layer grid.

    Attributes
    ----------
    row_frac, col_frac : numpy.ndarray
        Pixel-centre fractional coordinates in the layer frame, one per
        requested point. Radar mappings use the multilook box-average
        centre convention; geographic mappings use the inverse affine
        shifted from pixel corners to centres.
    rows, cols : numpy.ndarray
        Nearest integer indices ``floor(center + 0.5)``. Values outside
        the footprint are not clipped.
    mapped : numpy.ndarray
        True where the point is finite, in-footprint, and (for radar)
        converged with a finite DEM height.

    """

    row_frac: np.ndarray
    col_frac: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    mapped: np.ndarray


def is_inherit(value: object) -> bool:
    """Return whether ``value`` requests parent or session settings."""
    return value is Parameter.INHERIT or (isinstance(value, str) and value == "inherit")


def copy_points(points: Points | None) -> Points | None:
    """Return a defensive copy, preserving dtype and CRS.

    Parameters
    ----------
    points : Points or None
        Points to copy. ``None`` is returned unchanged.

    Returns
    -------
    Points or None
        A new :class:`~faninsar.data.query.Points` whose array is independent
        of ``points.values``.

    """
    if points is None:
        return None
    dtype = np.dtype(points.dtype)
    values = np.array(points.values, dtype=dtype, copy=True)
    return Points(values, crs=points.crs, dtype=dtype)


def normalize_stored_points(points: Points) -> Points:
    """Validate and copy points for Stack persistence.

    CRS-less input is stored as WGS84 longitude, latitude. Projected CRS
    is kept as given; radar sampling converts to WGS84 later. Empty or
    non-finite coordinates are rejected. The returned object is a copy.

    Parameters
    ----------
    points : Points
        Requested reference points.

    Returns
    -------
    Points
        Finite nonempty points with an explicit CRS and the original dtype.

    Raises
    ------
    TypeError
        If ``points`` is not a :class:`~faninsar.data.query.Points`.
    ValueError
        If the set is empty or contains non-finite coordinates.

    """
    if not isinstance(points, Points):
        message = (
            "ref_points must be a faninsar.data.query.Points instance, "
            f"got {type(points).__name__}"
        )
        reject(message, TypeError, logger=logger)
    if len(points) == 0:
        message = "ref_points must contain at least one point"
        reject(message, ValueError, logger=logger)
    dtype = np.dtype(points.dtype)
    values = np.array(points.values, dtype=dtype, copy=True)
    if values.ndim != 2 or values.shape[1] != 2:
        message = "ref_points must be a 2-D array with shape (n, 2)"
        reject(message, ValueError, logger=logger)
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        message = "ref_points coordinates must be finite"
        reject(message, ValueError, logger=logger)
    crs = WGS84_CRS if points.crs is None else points.crs
    return Points(values, crs=crs, dtype=dtype)


def encode_points_payload(points: Points) -> dict[str, Any]:
    """Return a JSON-safe ``faninsar:points`` payload preserving dtype."""
    stored = normalize_stored_points(points)
    crs = None if stored.crs is None else stored.crs.to_string()
    values = np.array(stored.values, dtype=stored.dtype, copy=True)
    return {
        "coordinates": values.tolist(),
        "crs": crs,
        "dtype": str(np.dtype(stored.dtype)),
    }


def decode_points_payload(payload: Mapping[str, Any]) -> Points:
    """Rebuild :class:`Points` from a ``faninsar:points`` payload."""
    if not isinstance(payload, Mapping):
        message = "faninsar:points payload must be a mapping"
        reject(message, ValueError, logger=logger)
    dtype_name = payload.get("dtype", "float32")
    try:
        dtype = np.dtype(dtype_name)
    except (TypeError, ValueError):
        message = f"faninsar:points dtype is invalid: {dtype_name!r}"
        reject(message, ValueError, logger=logger)
    coordinates = payload.get("coordinates")
    try:
        values = np.asarray(coordinates, dtype=dtype)
    except (TypeError, ValueError):
        message = "faninsar:points coordinates must be a numeric (n, 2) array"
        reject(message, ValueError, logger=logger)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        message = "faninsar:points coordinates must be a nonempty (n, 2) array"
        reject(message, ValueError, logger=logger)
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        message = "faninsar:points coordinates must be finite"
        reject(message, ValueError, logger=logger)
    return Points(values, crs=payload.get("crs"), dtype=dtype)


def resolve_ref_points_argument(
    value: object,
    inherited_points: Points | None,
    inherited_policy: str = "common",
) -> tuple[Points | None, str]:
    """Resolve inherit / None / explicit Points against session defaults.

    Parameters
    ----------
    value : Points, None, ``"inherit"``, or Parameter.INHERIT
        Operation argument. ``Parameter.AUTO`` is not accepted.
    inherited_points : Points or None
        Session points. ``None`` when there is no parent context.
    inherited_policy : {"common", "per_image"}, optional
        Session policy, or ``"common"`` when no session exists.

    Returns
    -------
    points : Points or None
        ``None`` disables referencing. Explicit :class:`Points` are
        normalized; inherit uses a copy of the session points.
    policy : str
        The session policy is always retained. Standalone callers without
        a session use ``"common"``.

    Raises
    ------
    TypeError
        If ``value`` is not an accepted type.
    ValueError
        If automatic selection is requested or inherited points are invalid.

    """
    policy = _require_policy(inherited_policy)
    if is_inherit(value):
        return (
            (None, policy)
            if inherited_points is None
            else (
                copy_points(inherited_points),
                policy,
            )
        )
    if value is None:
        return None, policy
    if value is Parameter.AUTO or (isinstance(value, str) and value == "auto"):
        message = "ref_points does not support automatic selection"
        reject(message, ValueError, logger=logger)
    if not isinstance(value, Points):
        message = (
            f"ref_points must be Points, None, or 'inherit'; got {type(value).__name__}"
        )
        reject(message, TypeError, logger=logger)
    return normalize_stored_points(value), policy


def _require_policy(policy: object) -> str:
    """Return ``policy`` when it is ``common`` or ``per_image``."""
    if not isinstance(policy, str) or policy not in {"common", "per_image"}:
        message = "ref_policy must be 'common' or 'per_image'"
        reject(message, ValueError, logger=logger)
    return policy


def coordinates_in_crs(points: Points, crs: object) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` in ``crs`` as float64.

    :meth:`~faninsar.data.query.Points.to_crs` reconstructs points with the
    constructor default dtype (float32). Mapping and radar sampling keep
    double precision here instead.

    Parameters
    ----------
    points : Points
        Points with an explicit CRS.
    crs : object
        Target CRS accepted by :meth:`pyproj.crs.CRS.from_user_input`.

    Returns
    -------
    x, y : numpy.ndarray
        Float64 coordinates in ``crs``.

    Raises
    ------
    ValueError
        If ``points`` has no CRS.

    """
    from pyproj import CRS
    from rasterio.warp import transform as warp_transform

    if points.crs is None:
        message = "reference points need a CRS for coordinate conversion"
        reject(message, ValueError, logger=logger)
    target = CRS.from_user_input(crs)
    xs = np.asarray(points.x, dtype=np.float64)
    ys = np.asarray(points.y, dtype=np.float64)
    if points.crs == target:
        return xs, ys
    x_out, y_out = warp_transform(points.crs, target, xs, ys)
    return np.asarray(x_out, dtype=np.float64), np.asarray(y_out, dtype=np.float64)


def nearest_pixels(
    row_center: np.ndarray,
    col_center: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map pixel-centre coordinates to nearest indices without clipping.

    A point is in-footprint when its centre coordinate lies in
    ``[-0.5, size - 0.5)``. The nearest index is ``floor(center + 0.5)``.
    Out-of-footprint points are not snapped to an edge pixel.

    Parameters
    ----------
    row_center, col_center : numpy.ndarray
        Pixel-centre fractional coordinates.
    shape : tuple[int, int]
        Layer ``(height, width)``.

    Returns
    -------
    rows, cols : numpy.ndarray
        Integer nearest indices (unclipped). Non-finite centres stay 0
        and are excluded by ``in_footprint``.
    in_footprint : numpy.ndarray
        Boolean mask of finite in-footprint samples.

    """
    row = np.asarray(row_center, dtype=np.float64)
    col = np.asarray(col_center, dtype=np.float64)
    height, width = int(shape[0]), int(shape[1])
    finite = np.isfinite(row) & np.isfinite(col)
    in_footprint = (
        finite
        & (row >= -0.5)
        & (row < (height - 0.5))
        & (col >= -0.5)
        & (col < (width - 0.5))
    )
    nearest_row = np.zeros(row.shape, dtype=np.intp)
    nearest_col = np.zeros(col.shape, dtype=np.intp)
    if np.any(finite):
        nearest_row[finite] = np.floor(row[finite] + 0.5).astype(np.intp)
        nearest_col[finite] = np.floor(col[finite] + 0.5).astype(np.intp)
    return nearest_row, nearest_col, in_footprint


def map_geographic_points(
    points: Points,
    *,
    transform: object,
    shape: tuple[int, int],
    crs: object | None = None,
) -> ReferencePixelMap:
    """Project geographic points through a layer affine into pixel centres.

    ``transform`` is the six-coefficient rasterio/Affine order used by
    product ``proj:transform``: ``x = a * col + b * row + c``. Inverse
    coordinates are pixel corners; they are shifted by ``-0.5`` onto the
    same centre convention as radar layer coordinates.

    Parameters
    ----------
    points : Points
        Normalized points with an explicit CRS.
    transform : Affine or sequence of float
        Layer affine.
    shape : tuple[int, int]
        Layer ``(height, width)``.
    crs : object, optional
        Layer CRS. Points are converted when it differs.

    Returns
    -------
    ReferencePixelMap
        Fractional centres, nearest indices, and in-footprint mask.

    """
    from affine import Affine

    if crs is not None:
        xs, ys = coordinates_in_crs(points, crs)
    else:
        xs = np.asarray(points.x, dtype=np.float64)
        ys = np.asarray(points.y, dtype=np.float64)
    if isinstance(transform, Affine):
        affine = transform
    else:
        values = tuple(float(item) for item in transform)
        affine = Affine(*values[:6])
    inverse = ~affine
    cols_corner, rows_corner = inverse * (xs, ys)
    row_center = np.asarray(rows_corner, dtype=np.float64) - 0.5
    col_center = np.asarray(cols_corner, dtype=np.float64) - 0.5
    nearest_row, nearest_col, in_footprint = nearest_pixels(
        row_center, col_center, shape
    )
    return ReferencePixelMap(
        row_frac=row_center,
        col_frac=col_center,
        rows=nearest_row,
        cols=nearest_col,
        mapped=in_footprint,
    )


def sample_at_pixels(
    array: np.ndarray,
    mapping: ReferencePixelMap,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sample ``array`` at mapped nearest pixels.

    Validity requires the mapping footprint, a finite array value, and
    (when given) a true support-mask sample. Connected-component labels
    are never used as a validity mask. Duplicate points that fall on the
    same pixel keep independent samples so they retain equal weight.

    Parameters
    ----------
    array : numpy.ndarray
        2-D layer values. The original array is not modified.
    mapping : ReferencePixelMap
        Pixel centres and nearest indices.
    support_mask : numpy.ndarray, optional
        Boolean support of the same shape. ``None`` skips mask checks.

    Returns
    -------
    numpy.ndarray
        Float64 samples with ``NaN`` at invalid points.

    """
    n_points = int(np.shape(mapping.mapped)[0])
    out = np.full(n_points, np.nan, dtype=np.float64)
    usable = np.asarray(mapping.mapped, dtype=bool)
    if not np.any(usable):
        return out
    rows = np.asarray(mapping.rows)
    cols = np.asarray(mapping.cols)
    values = np.asarray(array)[rows[usable], cols[usable]]
    sampled = np.asarray(values, dtype=np.float64)
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        sampled = np.where(support[rows[usable], cols[usable]], sampled, np.nan)
    sampled = np.where(np.isfinite(sampled), sampled, np.nan)
    out[usable] = sampled
    return out


def select_valid_subsets(
    valid_by_image: Mapping[str, np.ndarray],
    policy: str,
) -> dict[str, np.ndarray]:
    """Select the valid-point subset for ``common`` or ``per_image`` policy.

    ``common`` uses the nonempty intersection across every image and emits
    one warning when requested points were excluded. ``per_image`` uses
    each image's own nonempty subset and warns once when those subsets
    differ. An empty selection raises.

    Parameters
    ----------
    valid_by_image : mapping
        Per-image boolean masks aligned with the requested points.
    policy : {"common", "per_image"}
        Selection policy.

    Returns
    -------
    dict[str, numpy.ndarray]
        Per-image boolean selection masks.

    Raises
    ------
    ValueError
        If the policy is invalid or a required subset is empty.

    """
    policy = _require_policy(policy)
    keys = list(valid_by_image)
    if not keys:
        message = "no images available for reference-point selection"
        reject(message, ValueError, logger=logger)
    if policy == "common":
        intersection = np.asarray(valid_by_image[keys[0]], dtype=bool).copy()
        for key in keys[1:]:
            intersection &= np.asarray(valid_by_image[key], dtype=bool)
        n_valid = int(np.count_nonzero(intersection))
        n_requested = int(intersection.size)
        if n_valid == 0:
            message = "no requested reference point is valid on every image"
            reject(message, ValueError, logger=logger)
        if n_valid < n_requested:
            logger.warning(
                "Reference-point common intersection excluded %s of %s "
                "requested points",
                n_requested - n_valid,
                n_requested,
            )
        return {key: intersection.copy() for key in keys}

    selected = {
        key: np.asarray(mask, dtype=bool).copy() for key, mask in valid_by_image.items()
    }
    empty = [key for key, mask in selected.items() if not np.any(mask)]
    if empty:
        message = "no valid reference points on images: " + ", ".join(empty)
        reject(message, ValueError, logger=logger)
    first = selected[keys[0]]
    if any(not np.array_equal(selected[key], first) for key in keys[1:]):
        logger.warning("Reference-point per-image valid subsets differ across images")
    return selected


def mean_offsets(
    samples_by_image: Mapping[str, np.ndarray],
    selected_by_image: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Return the arithmetic mean of selected samples for each image.

    Duplicate points that share a pixel contribute separately. Component
    labels are not consulted.

    """
    offsets: dict[str, float] = {}
    for key, samples in samples_by_image.items():
        mask = np.asarray(selected_by_image[key], dtype=bool)
        values = np.asarray(samples, dtype=np.float64)[mask]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            message = f"no finite reference samples on image {key}"
            reject(message, ValueError, logger=logger)
        offsets[key] = float(np.mean(finite))
    return offsets


def subtract_offsets(
    arrays: Mapping[str, np.ndarray],
    offsets: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Return copies of ``arrays`` with each image's scalar offset subtracted."""
    referenced: dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        copied = np.array(array, copy=True)
        offset = np.asarray(offsets[key], dtype=copied.dtype)
        referenced[key] = copied - offset
    return referenced


def reference_definition(
    points: Points | None,
    policy: str,
) -> dict[str, Any] | None:
    """Return the JSON-safe requested reference definition, or ``None``."""
    if points is None:
        return None
    policy = _require_policy(policy)
    crs = None if points.crs is None else points.crs.to_string()
    return {
        "method": _REFERENCE_METHOD,
        "policy": policy,
        "coordinates": np.asarray(points.values, dtype=np.float64).tolist(),
        "crs": crs,
        "dtype": str(np.dtype(points.dtype)),
    }


def definitions_match(stored: object, requested: object) -> bool:
    """Return whether two reference definitions describe the same request.

    Missing metadata (``None``) matches only a disabled request. Coordinate
    comparison uses float64 values after JSON round-trip.

    """
    if stored is None and requested is None:
        return True
    if not isinstance(stored, dict) or not isinstance(requested, dict):
        return False
    if any(
        stored.get(key) != requested.get(key) for key in ("method", "policy", "crs")
    ):
        return False
    left = np.asarray(stored.get("coordinates"), dtype=np.float64)
    right = np.asarray(requested.get("coordinates"), dtype=np.float64)
    if left.shape != right.shape:
        return False
    return bool(np.allclose(left, right, rtol=1e-7, atol=1e-7))


def applied_reference_record(
    *,
    points: Points,
    policy: str,
    mapping: ReferencePixelMap,
    selected_by_image: Mapping[str, np.ndarray],
    offsets: Mapping[str, float],
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the provenance record stored on a time-series result."""
    definition = reference_definition(points, policy)
    selected = {
        key: np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int).tolist()
        for key, mask in selected_by_image.items()
    }
    pixels: dict[str, list[list[int]]] = {}
    for key, mask in selected_by_image.items():
        take = np.asarray(mask, dtype=bool)
        rows = np.asarray(mapping.rows)[take]
        cols = np.asarray(mapping.cols)[take]
        pixels[key] = [
            [int(row), int(col)] for row, col in zip(rows, cols, strict=True)
        ]
    record = dict(definition or {})
    record["grid"] = dict(grid)
    record["selected"] = selected
    record["pixels"] = pixels
    record["offsets"] = {key: float(value) for key, value in offsets.items()}
    return record


def stored_definition_from_record(record: object) -> dict[str, Any] | None:
    """Extract the requested definition from a stored provenance record."""
    if not isinstance(record, dict):
        return None
    if "coordinates" not in record or "policy" not in record:
        return None
    return {
        "method": record.get("method", _REFERENCE_METHOD),
        "policy": record.get("policy"),
        "coordinates": record.get("coordinates"),
        "crs": record.get("crs"),
        "dtype": record.get("dtype"),
    }

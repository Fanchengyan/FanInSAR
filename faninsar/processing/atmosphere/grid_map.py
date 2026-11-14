"""Coordinate-aware mapping of unwrapped ion phase onto a target IFG grid.

Native pixel centers have integer coordinates. A block-mean sample that
begins at native index ``s`` with ``L`` looks is centered at
``s + (L - 1) / 2`` with spacing ``L``. Mapping is bilinear in unwrapped
radians: invalid or non-positive-quality neighbors are not renormalized,
and samples outside the source-center support stay invalid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "map_unwrapped_phase_bilinear",
    "native_block_center_anchor",
]

_SNAP_ATOL = 1e-9
_WEIGHT_ATOL = 1e-12


def native_block_center_anchor(
    native_origin: Sequence[int | float],
    looks: Sequence[int],
) -> tuple[float, float]:
    """Return the first-sample center of a block-mean grid.

    Parameters
    ----------
    native_origin : sequence of int or float
        Native-pixel index ``(azimuth, range)`` of the first contributing
        sample of the first block.
    looks : sequence of int
        Placement-ordered ``(azimuth, range)`` absolute look counts. This is
        a radar-grid placement pair, not the range-first look-factor value.

    Returns
    -------
    tuple of float
        First-sample center in parent native-pixel coordinates.

    """
    origin = _as_pair(native_origin, "native_origin", positive=False)
    look_pair = _as_looks(looks)
    return (
        origin[0] + (look_pair[0] - 1) / 2.0,
        origin[1] + (look_pair[1] - 1) / 2.0,
    )


def map_unwrapped_phase_bilinear(
    phase: np.ndarray,
    quality: np.ndarray,
    *,
    source_anchor: Sequence[float],
    source_spacing: Sequence[float],
    target_shape: tuple[int, int],
    target_anchor: Sequence[float],
    target_spacing: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map unwrapped phase onto a target grid with masked bilinear weights.

    Parameters
    ----------
    phase, quality : numpy.ndarray
        Source unwrapped phase (radians) and finite-positive quality weights
        on the same 2-D grid.
    source_anchor, target_anchor : sequence of float
        First-sample centers ``(azimuth, range)`` in native-pixel units.
    source_spacing, target_spacing : sequence of float
        Sample spacing ``(azimuth, range)`` in native-pixel units.
    target_shape : tuple of int
        Output ``(azimuth, range)`` shape.

    Returns
    -------
    mapped : numpy.ndarray
        Phase on the target grid; unsupported samples are NaN.
    valid : numpy.ndarray
        Boolean support of ``mapped``.

    Raises
    ------
    InvalidProcessingStateError
        If the source arrays or placement metadata are inconsistent.

    """
    source_phase = np.asarray(phase, dtype=np.float64)
    source_quality = np.asarray(quality, dtype=np.float64)
    if source_phase.ndim != 2 or source_quality.shape != source_phase.shape:
        reject_invalid_state("ion mapping phase and quality must share one 2-D grid")
    if any(size < 1 for size in source_phase.shape):
        reject_invalid_state("ion mapping source grid must be non-empty")
    src_anchor = _as_pair(source_anchor, "source_anchor", positive=False)
    src_step = _as_pair(source_spacing, "source_spacing", positive=True)
    dst_anchor = _as_pair(target_anchor, "target_anchor", positive=False)
    dst_step = _as_pair(target_spacing, "target_spacing", positive=True)
    height, width = _as_shape(target_shape)
    source_ok = (
        np.isfinite(source_phase) & np.isfinite(source_quality) & (source_quality > 0.0)
    )
    if (
        source_phase.shape == (height, width)
        and np.allclose(src_anchor, dst_anchor, rtol=0.0, atol=_SNAP_ATOL)
        and np.allclose(src_step, dst_step, rtol=0.0, atol=_SNAP_ATOL)
    ):
        mapped = np.where(source_ok, source_phase, np.nan)
        return mapped, source_ok.copy()

    rows = dst_anchor[0] + np.arange(height, dtype=np.float64) * dst_step[0]
    cols = dst_anchor[1] + np.arange(width, dtype=np.float64) * dst_step[1]
    north, east = np.meshgrid(rows, cols, indexing="ij")
    row_index = (north - src_anchor[0]) / src_step[0]
    col_index = (east - src_anchor[1]) / src_step[1]
    row_index = _snap_integer(row_index)
    col_index = _snap_integer(col_index)
    source_rows, source_cols = source_phase.shape
    in_support = (
        (row_index >= 0.0)
        & (row_index <= source_rows - 1.0)
        & (col_index >= 0.0)
        & (col_index <= source_cols - 1.0)
    )
    row0 = np.floor(row_index).astype(np.int64)
    col0 = np.floor(col_index).astype(np.int64)
    weight_row = row_index - row0
    weight_col = col_index - col0
    value00, ok00 = _gather(source_phase, source_ok, row0, col0, in_support=in_support)
    value01, ok01 = _gather(
        source_phase, source_ok, row0, col0 + 1, in_support=in_support
    )
    value10, ok10 = _gather(
        source_phase, source_ok, row0 + 1, col0, in_support=in_support
    )
    value11, ok11 = _gather(
        source_phase, source_ok, row0 + 1, col0 + 1, in_support=in_support
    )
    weight00 = (1.0 - weight_row) * (1.0 - weight_col)
    weight01 = (1.0 - weight_row) * weight_col
    weight10 = weight_row * (1.0 - weight_col)
    weight11 = weight_row * weight_col
    valid = (
        in_support
        & _neighbor_ok(weight00, ok00)
        & _neighbor_ok(weight01, ok01)
        & _neighbor_ok(weight10, ok10)
        & _neighbor_ok(weight11, ok11)
    )
    mapped = (
        weight00 * np.where(ok00, value00, 0.0)
        + weight01 * np.where(ok01, value01, 0.0)
        + weight10 * np.where(ok10, value10, 0.0)
        + weight11 * np.where(ok11, value11, 0.0)
    )
    mapped = np.where(valid, mapped, np.nan)
    return mapped, valid


def _as_looks(value: Sequence[int]) -> tuple[int, int]:
    """Validate two placement-ordered positive integer look counts."""
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in value
        )
    ):
        reject_invalid_state("looks must be two positive integers")
    return int(value[0]), int(value[1])


def _as_pair(
    value: Sequence[float],
    name: str,
    *,
    positive: bool,
) -> tuple[float, float]:
    """Validate a two-element finite placement pair."""
    try:
        pair = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        reject_invalid_state(f"{name} must be two finite numbers")
    if len(value) != 2 or not np.isfinite(pair).all():
        reject_invalid_state(f"{name} must be two finite numbers")
    if positive and (pair[0] <= 0.0 or pair[1] <= 0.0):
        reject_invalid_state(f"{name} must be two positive numbers")
    return pair


def _as_shape(value: tuple[int, int]) -> tuple[int, int]:
    """Validate a non-empty 2-D shape."""
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in value
        )
    ):
        reject_invalid_state("target_shape must be two positive integers")
    return int(value[0]), int(value[1])


def _snap_integer(index: np.ndarray) -> np.ndarray:
    """Treat near-integer fractional indices as exact sample centers."""
    rounded = np.round(index)
    return np.where(
        np.isclose(index, rounded, rtol=0.0, atol=_SNAP_ATOL), rounded, index
    )


def _gather(
    phase: np.ndarray,
    source_ok: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    in_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one bilinear neighbor, marking out-of-bounds samples invalid."""
    height, width = phase.shape
    inside = in_support & (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    clipped_rows = np.clip(rows, 0, height - 1)
    clipped_cols = np.clip(cols, 0, width - 1)
    values = phase[clipped_rows, clipped_cols]
    ok = inside & source_ok[clipped_rows, clipped_cols]
    return values, ok


def _neighbor_ok(weight: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Require validity only of neighbors with nonzero bilinear weight."""
    return (~(weight > _WEIGHT_ATOL)) | ok

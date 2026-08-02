"""Production Sentinel-1 measurement I/O: full burst and multi-burst swath reads.

This module hosts the rasterio-window read path used by the production
loaders.  For a byte-offset fast path that targets SAFE ZIP archives see
:mod:`faninsar.missions.sentinel1.io.extract`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rasterio.windows import Window

from faninsar.logging import setup_logger
from faninsar.missions.sentinel1.errors import reject_product

if TYPE_CHECKING:
    from faninsar.missions.sentinel1.types import S1Burst, S1Swath

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class BurstArray:
    """Full (or cropped) complex burst with absolute raster placement."""

    samples: np.ndarray
    row0: int
    col0: int
    burst_index: int
    valid_mask: np.ndarray


def _burst_line_offset(swath: S1Swath, burst: S1Burst) -> int:
    """Return starting azimuth line of a burst in the measurement raster."""
    return int(burst.index * swath.lines_per_burst)


def _valid_column_bounds(burst: S1Burst) -> tuple[int, int]:
    """Return inclusive-exclusive column bounds covering valid samples."""
    first = np.asarray(burst.first_valid_sample, dtype=np.int32)
    last = np.asarray(burst.last_valid_sample, dtype=np.int32)
    valid = first >= 0
    if not np.any(valid):
        reject_product(f"burst {burst.index} has no valid samples")
    col0 = int(np.min(first[valid]))
    col1 = int(np.max(last[valid])) + 1
    col0 = max(col0, 0)
    col1 = min(col1, burst.samples)
    if col1 <= col0:
        reject_product(f"burst {burst.index} has empty valid column range")
    return col0, col1


def read_full_burst(
    swath: S1Swath,
    *,
    burst_index: int = 0,
    range_looks_crop: tuple[int, int] | None = None,
    geocoding_layout: bool = False,
    full_range: bool = False,
) -> BurstArray:
    """Read a TOPS burst in radar or direct-geocoding layout.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath with measurement path.
    burst_index : int, optional
        Burst index within the swath.
    range_looks_crop : tuple[int, int], optional
        Optional ``(col0, col1)`` absolute range crop inside the burst.
        When omitted, uses the burst valid-sample envelope.
    geocoding_layout : bool, optional
        Use valid azimuth lines and the complete divisible range extent,
        matching direct geographic SLC transform conventions.
    full_range : bool, optional
        Read the full swath range extent (column 0 through the burst width)
        instead of cropping to the valid-sample envelope, matching ISCE2's
        full-width burst layout.  Invalid samples are still zeroed via the
        valid mask.

    Returns
    -------
    BurstArray
        Complex64 samples of shape ``(lines_per_burst, n_range)``.

    """
    if burst_index < 0 or burst_index >= len(swath.bursts):
        reject_product(f"burst_index {burst_index} out of range for {swath.swath}")
    burst = swath.bursts[burst_index]
    local_row0 = 0
    height = burst.lines
    if geocoding_layout:
        valid_lines = np.asarray(burst.first_valid_sample, dtype=np.int32) >= 0
        if not np.any(valid_lines):
            reject_product(f"burst {burst.index} has no valid azimuth lines")
        valid_indices = np.flatnonzero(valid_lines)
        local_row0 = int(valid_indices[0])
        valid_height = int(valid_indices[-1] - local_row0)
        height = valid_height - valid_height % 4
        if height <= 0:
            reject_product(f"burst {burst.index} has empty valid azimuth extent")
        col0 = 0
        col1 = int(burst.samples) - int(burst.samples) % 4
    elif full_range:
        col0 = 0
        col1 = int(burst.samples)
    else:
        col0, col1 = _valid_column_bounds(burst)
    if range_looks_crop is not None:
        c0, c1 = range_looks_crop
        valid_col0, valid_col1 = _valid_column_bounds(burst)
        col0 = max(valid_col0, int(c0))
        col1 = min(valid_col1, int(c1))
        if col1 <= col0:
            reject_product("range_looks_crop is empty after intersecting valid bounds")

    row0 = _burst_line_offset(swath, burst) + local_row0
    width = col1 - col0
    import rasterio

    logger.info(
        "Reading full burst %s: rows[%s:%s] cols[%s:%s] (%.1f MiB complex64)",
        burst_index,
        row0,
        row0 + height,
        col0,
        col1,
        height * width * 8 / (1024**2),
    )
    with rasterio.open(swath.measurement_path) as dataset:
        window = Window.from_slices(
            (row0, row0 + height),
            (col0, col0 + width),
        )
        samples = dataset.read(1, window=window)
    samples = np.asarray(samples, dtype=np.complex64)
    if samples.shape != (height, width):
        reject_product(
            f"unexpected full-burst shape {samples.shape}, expected {(height, width)}"
        )

    first = np.asarray(
        burst.first_valid_sample[local_row0 : local_row0 + height],
        dtype=np.int32,
    )
    last = np.asarray(
        burst.last_valid_sample[local_row0 : local_row0 + height],
        dtype=np.int32,
    )
    cols = np.arange(col0, col1, dtype=np.int32)[None, :]
    valid_mask = (
        (first[:, None] >= 0) & (cols >= first[:, None]) & (cols <= last[:, None])
    )
    # zero invalid samples so downstream kernels do not use garbage
    samples = np.where(valid_mask, samples, 0)

    return BurstArray(
        samples=samples,
        row0=row0,
        col0=col0,
        burst_index=burst_index,
        valid_mask=valid_mask,
    )


def read_swath_bursts(
    swath: S1Swath,
    *,
    burst_indices: list[int] | None = None,
) -> list[BurstArray]:
    """Read one or more full bursts from a sub-swath.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath.
    burst_indices : list of int, optional
        Bursts to read. Defaults to all bursts.

    Returns
    -------
    list[BurstArray]
        Full-burst arrays in burst-index order.

    """
    if burst_indices is None:
        burst_indices = list(range(len(swath.bursts)))
    return [read_full_burst(swath, burst_index=i) for i in burst_indices]


def stitch_bursts(
    bursts: list[BurstArray],
    *,
    overlap_blend: bool = True,
) -> BurstArray:
    """Stitch full bursts along azimuth into a continuous swath image.

    Parameters
    ----------
    bursts : list[BurstArray]
        Ordered burst arrays (same range origin/width required).
    overlap_blend : bool, optional
        When True, linearly blend overlapping azimuth lines.

    Returns
    -------
    BurstArray
        Stitched complex array. ``burst_index`` is set to -1.

    """
    if not bursts:
        reject_product("no bursts to stitch")

    # Find the common range intersection across all bursts.
    common_col0 = max(b.col0 for b in bursts)
    common_col1 = min(b.col0 + b.samples.shape[1] for b in bursts)
    if common_col1 <= common_col0:
        reject_product("bursts have no common range overlap for stitching")
    common_width = common_col1 - common_col0

    row0 = min(b.row0 for b in bursts)
    row1 = max(b.row0 + b.samples.shape[0] for b in bursts)
    height = row1 - row0
    out = np.zeros((height, common_width), dtype=np.complex64)
    weight = np.zeros((height, common_width), dtype=np.float32)
    valid = np.zeros((height, common_width), dtype=bool)

    for burst in bursts:
        r0 = burst.row0 - row0
        r1 = r0 + burst.samples.shape[0]
        # Slice the burst to the common range intersection.
        c0 = common_col0 - burst.col0
        c1 = c0 + common_width
        burst_slice = burst.samples[:, c0:c1]
        mask_slice = burst.valid_mask[:, c0:c1]
        if overlap_blend:
            # triangular weights peaking at burst centre
            n = burst.samples.shape[0]
            w_line = np.hanning(n).astype(np.float32)
            w_line = np.maximum(w_line, 0.05)
            w = w_line[:, None] * mask_slice.astype(np.float32)
        else:
            w = mask_slice.astype(np.float32)
        out[r0:r1] += burst_slice * w
        weight[r0:r1] += w
        valid[r0:r1] |= mask_slice

    mask = weight > 0
    out[mask] /= weight[mask]
    logger.info(
        "Stitched %s bursts -> shape %s (valid fraction %.3f)",
        len(bursts),
        out.shape,
        float(np.mean(valid)),
    )
    return BurstArray(
        samples=out,
        row0=row0,
        col0=common_col0,
        burst_index=-1,
        valid_mask=valid,
    )


# Keep the small-window helper for debugging, but production path uses full burst.
def read_burst_window(
    swath: S1Swath,
    *,
    burst_index: int = 0,
    height: int = 128,
    width: int = 128,
    row_offset: int = 100,
    col_offset: int = 100,
) -> BurstArray:
    """Read a sub-window of a burst (debug only; prefer :func:`read_full_burst`)."""
    full = read_full_burst(swath, burst_index=burst_index)
    first = np.asarray(swath.bursts[burst_index].first_valid_sample, dtype=np.int32)
    valid_lines = np.where(first >= 0)[0]
    local_row = int(valid_lines[0] + row_offset)
    local_row = min(max(local_row, 0), full.samples.shape[0] - height)
    local_col = int(col_offset)
    local_col = min(max(local_col, 0), full.samples.shape[1] - width)
    samples = full.samples[
        local_row : local_row + height, local_col : local_col + width
    ]
    valid = full.valid_mask[
        local_row : local_row + height, local_col : local_col + width
    ]
    return BurstArray(
        samples=np.asarray(samples, dtype=np.complex64),
        row0=full.row0 + local_row,
        col0=full.col0 + local_col,
        burst_index=burst_index,
        valid_mask=valid,
    )

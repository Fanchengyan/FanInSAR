"""Filtering utilities for ionospheric phase screens.

Torch implementations of the reference behaviors: inverse-variance weighted
Gaussian smoothing and small-cluster removal equivalent to ISCE3's
``remove_small_components`` (4-connectivity via ``scipy.ndimage.label``
default), implemented as a bounded, convergent min-label propagation loop.
"""

from __future__ import annotations

import torch

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "gaussian_kernel_1d",
    "remove_small_components",
    "smooth_inverse_variance",
]

_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def remove_small_components(
    valid_mask: torch.Tensor,
    min_cluster_pixels: int,
    *,
    max_iterations: int | None = None,
) -> torch.Tensor:
    """Remove 4-connected valid clusters smaller than ``min_cluster_pixels``.

    Semantics mirror ISCE3 ``remove_small_components`` (which labels the
    valid mask with the default 4-connected ``scipy.ndimage.label`` structure
    and drops clusters below the pixel threshold): clusters are sets of
    True pixels connected along rows/columns.

    The exact labeling is computed by bounded min-label propagation: every
    valid pixel starts from its unique linear index and repeatedly adopts
    the smallest neighboring valid label until a fixpoint.  Convergence is
    guaranteed within ``rows * cols`` iterations (the largest possible
    geodesic diameter plus one); ``max_iterations`` allows early capping for
    callers that prefer bounded latency.

    Parameters
    ----------
    valid_mask : torch.Tensor
        Boolean mask of valid pixels.
    min_cluster_pixels : int
        Clusters with fewer valid pixels are removed (set False).
        ``0`` disables removal entirely.
    max_iterations : int, optional
        Hard cap on propagation sweeps; defaults to ``rows * cols`` so the
        exact fixpoint is always reached.  Provide a smaller value only when
        bounded latency matters more than exact labeling.

    Returns
    -------
    torch.Tensor
        Cleaned boolean mask.  Input is never mutated.

    Raises
    ------
    ValueError
        On non-boolean masks or negative thresholds.

    """
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        message = "valid_mask must be a boolean tensor"
        logger.error(message)
        raise ValueError(message)
    if min_cluster_pixels < 0:
        message = f"min_cluster_pixels must be >= 0, got {min_cluster_pixels!r}"
        logger.error(message)
        raise ValueError(message)

    cleaned = valid_mask.clone()
    if min_cluster_pixels == 0 or not bool(cleaned.any()):
        return cleaned

    rows, cols = int(valid_mask.shape[-2]), int(valid_mask.shape[-1])
    linear = (
        torch.arange(rows * cols, device=valid_mask.device, dtype=torch.int64).reshape(
            rows, cols
        )
        + 1
    )
    labels = torch.where(cleaned, linear, torch.zeros_like(linear))

    max_steps = max_iterations if max_iterations is not None else rows * cols
    huge = torch.iinfo(torch.int64).max

    for _ in range(max(1, max_steps)):
        padded = torch.nn.functional.pad(labels, (1, 1, 1, 1), mode="constant", value=0)
        candidates = [padded[1:-1, 1:-1]]
        for dy, dx in _NEIGHBOR_OFFSETS:
            shifted = padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
            candidates.append(shifted)
        # only positive (valid) labels compete; every candidate slot falls
        # back to the huge sentinel so an isolated pixel keeps its own label
        stacked = torch.stack(candidates, dim=0)
        competing = torch.where(stacked > 0, stacked, torch.full_like(stacked, huge))
        merged = torch.where(
            cleaned,
            competing.min(dim=0).values,
            torch.zeros_like(labels),
        )

        if bool(torch.equal(merged, labels)):
            break
        labels = merged

    # bincount index k equals label id k (raw ids kept; invalid background
    # id 0 has size = pixel count but is force-zeroed below)
    sizes = torch.bincount(labels.reshape(-1))
    sizes[0] = 0
    keep_by_id = sizes >= min_cluster_pixels
    return keep_by_id[labels]


def gaussian_kernel_1d(sigma: float, *, truncate: float = 3.0) -> torch.Tensor:
    """Return a normalized 1-D Gaussian kernel of odd length."""
    if sigma <= 0.0:
        return torch.ones(1, dtype=torch.float64)
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def smooth_inverse_variance(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    sigma_y: float,
    sigma_x: float,
) -> torch.Tensor:
    """One pass of inverse-variance weighted Gaussian smoothing.

    Computes ``G*(v*w)/G*w`` with separable Gaussian kernels ``G``, i.e. a
    variance-weighted local average identical in form to the weighted
    smoothing step of the reference screens.  Invalid values take zero
    weight; output is NaN where accumulated support vanishes.
    """
    v = values.to(torch.float64)
    finite_v = torch.isfinite(v)
    # NaN values must neither contribute nor drag fake zeros into the mean
    v = torch.where(finite_v, v, torch.zeros_like(v))
    w = weights.to(torch.float64).clamp(min=0.0) * finite_v
    vw = v * w

    def separable(img: torch.Tensor, kernel: torch.Tensor, axis: int) -> torch.Tensor:
        radius = (kernel.numel() - 1) // 2
        moved = img.unsqueeze(0).unsqueeze(0)
        # F.pad flattens as (W_left, W_right, H_top, H_bottom)
        padded = (
            torch.nn.functional.pad(moved, (0, 0, radius, radius))
            if axis == 0
            else torch.nn.functional.pad(moved, (radius, radius, 0, 0))
        )
        weight4d = (
            kernel.reshape(1, 1, -1, 1) if axis == 0 else kernel.reshape(1, 1, 1, -1)
        )
        out = torch.nn.functional.conv2d(padded, weight4d)
        return out.squeeze(0).squeeze(0)

    ky = gaussian_kernel_1d(float(sigma_y)).to(values.device)
    kx = gaussian_kernel_1d(float(sigma_x)).to(values.device)
    num = separable(separable(vw, ky, 0), kx, 1)
    den = separable(separable(w, ky, 0), kx, 1)

    fill = float("nan")
    smoothed = torch.where(den > 0, num / den, torch.full_like(num, fill))
    return smoothed.to(values.dtype)

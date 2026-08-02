"""Overlap masks and feather weights for burst merge products."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "apply_feather",
    "compute_feather",
    "compute_hanning_weight",
    "compute_weight_stack",
    "overlap_mask",
]


def compute_feather(
    valid_mask: np.ndarray,
    *,
    feather_width_px: float,
) -> np.ndarray:
    """Compute the feather weight profile of a valid mask.

    The feather weight is 0 at the valid-region boundary, rising linearly
    to 1 at ``feather_width_px`` pixels inside the region, and clipped to 1
    deeper inside. Invalid pixels are exactly 0.

    Parameters
    ----------
    valid_mask : numpy.ndarray
        Boolean mask of valid pixels.
    feather_width_px : float
        Distance (in pixels) over which the weight ramps from 0 to 1.
        A value of 0 disables feathering (returns 1 on valid, 0 elsewhere).

    Returns
    -------
    numpy.ndarray
        Float32 feather weights in ``[0, 1]``.

    """
    if feather_width_px <= 0.0:
        return valid_mask.astype(np.float32)
    # distance_transform_edt computes, for each True pixel, the Euclidean
    # distance to the nearest False pixel (or array edge when not padded).
    dist = distance_transform_edt(valid_mask)
    d = np.clip(dist / feather_width_px, 0.0, 1.0).astype(np.float32)
    d[~valid_mask] = 0.0
    return d


def apply_feather(
    weight: np.ndarray,
    valid_mask: np.ndarray,
    *,
    feather_width_px: float,
) -> np.ndarray:
    """Multiply a feather profile into an existing weight array.

    Parameters
    ----------
    weight : np.ndarray
        Per-pixel weight to be feathered.
    valid_mask : np.ndarray
        Boolean mask of valid pixels.
    feather_width_px : float
        Feather half-width in pixels.

    Returns
    -------
    np.ndarray
        Feathereight as a float32 array.

    """
    feather = compute_feather(valid_mask, feather_width_px=feather_width_px)
    return (weight.astype(np.float32) * feather).astype(np.float32)


def compute_hanning_weight(
    valid_mask: np.ndarray,
    *,
    min_weight: float = 0.05,
    axis: int = 0,
) -> np.ndarray:
    """Compute a Hanning-window weight over the valid extent of a burst.

    The weight peaks at the burst centre along *axis* and tapers to
    ``min_weight`` at the edges, matching the blending strategy of
    :func:`faninsar.missions.sentinel1.io.stitch_bursts`.  Compared to
    :func:`compute_feather` (which only ramps over a narrow border), the
    Hanning window covers the entire burst height so that two overlapping
    bursts form complementary cos²/sin²-style transitions with no
    "double-1" plateau in the overlap centre.

    Parameters
    ----------
    valid_mask : numpy.ndarray
        Boolean mask of valid pixels (2-D).
    min_weight : float, optional
        Floor value at the burst edges (default 0.05, same as
        ``stitch_bursts``).
    axis : int, optional
        Axis along which the Hanning window is applied. ``0`` (default)
        is azimuth (rows) — the typical burst-overlap direction for
        Sentinel-1 TOPS.

    Returns
    -------
    numpy.ndarray
        Float32 weights of the same shape as *valid_mask*.  Invalid
        pixels are exactly 0.

    Notes
    -----
    The valid extent along *axis* is determined from ``valid_mask.any()``
    on the complementary axis.  For a burst whose footprint is a slanted
    parallelogram on the geo grid, each column may have a slightly
    different valid-row range; this implementation uses the global row
    extent, which is sufficient for the ~25 % azimuth overlap typical of
    Sentinel-1 IW bursts.

    """
    if valid_mask.ndim != 2:
        msg = "compute_hanning_weight expects a 2-D valid_mask"
        raise ValueError(msg)
    other = 1 - axis
    valid_any = valid_mask.any(axis=other)
    valid_idx = np.where(valid_any)[0]
    if valid_idx.size == 0:
        return np.zeros_like(valid_mask, dtype=np.float32)

    r0, r1 = int(valid_idx[0]), int(valid_idx[-1])
    extent = r1 - r0 + 1
    w_1d = np.maximum(np.hanning(extent).astype(np.float32), float(min_weight))

    w_full = np.zeros(valid_mask.shape[axis], dtype=np.float32)
    w_full[r0 : r1 + 1] = w_1d

    if axis == 0:
        w_2d = w_full[:, None]
    else:
        w_2d = w_full[None, :]
    return (w_2d * valid_mask.astype(np.float32)).astype(np.float32)


def overlap_mask(
    weight_i: np.ndarray,
    weight_j: np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Return the boolean overlap of two weighted bursts.

    Parameters
    ----------
    weight_i, weight_j : numpy.ndarray
        Per-pixel weights of two bursts on the same grid.
    threshold : float, optional
        A pixel is in the overlap when both weights exceed this value.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the overlap region.

    """
    return (weight_i > threshold) & (weight_j > threshold)


def compute_weight_stack(
    *,
    valid_mask: np.ndarray,
    coherence: np.ndarray | None,
    coherence_exponent: float,
    feather_width_px: float,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    r"""Combine the weight components into a single per-pixel weight.

    The weight is

    .. math::

        w(p) = m_{\text{valid}}(p) \cdot d_{\text{feather}}(p)
               \cdot \gamma(p)^{p_\gamma} \cdot w_{\text{extra}}(p)

    Parameters
    ----------
    valid_mask : numpy.ndarray
        Boolean valid mask.
    coherence : numpy.ndarray or None
        Coherence in ``[0, 1]``. When ``None`` the coherence factor is 1.
    coherence_exponent : float
        Exponent applied to the coherence.
    feather_width_px : float
        Feather half-width in pixels.
    extra : numpy.ndarray, optional
        Additional per-pixel weight factor (e.g. look-direction prior).

    Returns
    -------
    numpy.ndarray
        Float32 per-pixel weight.

    """
    feather = compute_feather(valid_mask, feather_width_px=feather_width_px)
    if coherence is None:
        coh_factor = np.ones_like(feather)
    else:
        coh_factor = np.power(
            np.clip(coherence.astype(np.float32), 0.0, 1.0),
            float(coherence_exponent),
        )
    extra_factor = np.ones_like(feather) if extra is None else extra.astype(np.float32)
    return (valid_mask.astype(np.float32) * feather * coh_factor * extra_factor).astype(
        np.float32
    )

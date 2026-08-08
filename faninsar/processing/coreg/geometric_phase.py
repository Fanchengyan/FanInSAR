"""Geometric phase from dense range offsets (flat / flat+topo).

Single removal site for stack and pair coregistration (PROPOSAL-0017).
There is no public ``stage_topo``; callers use this helper from coreg only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.interferometry.flatten import remove_topographic_phase

logger = setup_logger(__name__)

GeometricPhaseMode = Literal["topo", "flat", "none"]


def phase_per_range_pixel(
    *,
    range_spacing_m: float,
    wavelength_m: float,
) -> float:
    """Return radians of geometric phase per range pixel of path difference.

    Parameters
    ----------
    range_spacing_m : float
        Slant-range sampling interval (m).
    wavelength_m : float
        Radar wavelength (m).

    Returns
    -------
    float
        ``4π · Δr / λ`` scale for one range pixel of offset.

    """
    if range_spacing_m <= 0.0 or wavelength_m <= 0.0:
        reject_invalid_state("range_spacing_m and wavelength_m must be positive")
    return float(4.0 * np.pi * range_spacing_m / wavelength_m)


def geometric_phase_from_range_offset(
    range_offset_px: np.ndarray | float,
    *,
    range_spacing_m: float,
    wavelength_m: float,
) -> np.ndarray:
    """Build geometric phase (rad) from a range-offset field.

    With DEM-driven geometry offsets this is flat+topo; without DEM the same
    formula yields flat-earth only (ellipsoid path).

    Parameters
    ----------
    range_offset_px : array or float
        Dense or scalar range offset (master − secondary), pixels.
    range_spacing_m, wavelength_m : float
        Secondary (or master-consistent) radar geometry constants.

    Returns
    -------
    numpy.ndarray
        Phase in radians, same shape as ``range_offset_px`` (or scalar array).

    """
    scale = phase_per_range_pixel(
        range_spacing_m=range_spacing_m,
        wavelength_m=wavelength_m,
    )
    off = np.asarray(range_offset_px, dtype=np.float64)
    return (scale * off).astype(np.float32, copy=False)


def apply_geometric_phase_from_range_offset(
    complex_samples: np.ndarray,
    range_offset_px: np.ndarray | float,
    *,
    range_spacing_m: float,
    wavelength_m: float,
    already_removed: bool = False,
) -> tuple[np.ndarray, GeometricPhaseMode]:
    """Multiply complex samples by ``exp(-j · φ_geom(range_offset))``.

    Parameters
    ----------
    complex_samples : numpy.ndarray
        Complex SLC or IFG samples.
    range_offset_px : array or float
        Range offset field aligned with samples (or broadcastable).
    range_spacing_m, wavelength_m : float
        Geometry constants for the phase scale.
    already_removed : bool, optional
        If True, return samples unchanged and mode ``"none"``.

    Returns
    -------
    samples : numpy.ndarray
        Phase-adjusted complex array.
    mode : {"topo", "flat", "none"}
        Contract flag. Callers that used DEM-backed offsets should record
        ``"topo"``; ellipsoid-only offsets should record ``"flat"``. This
        helper cannot see the DEM and defaults to ``"topo"`` when applied
        (callers may override the returned mode).

    """
    if already_removed:
        return complex_samples, "none"
    if not np.iscomplexobj(complex_samples):
        reject_invalid_state("geometric phase requires a complex array")
    phase = geometric_phase_from_range_offset(
        range_offset_px,
        range_spacing_m=range_spacing_m,
        wavelength_m=wavelength_m,
    )
    out = remove_topographic_phase(complex_samples, phase)
    logger.debug(
        "Applied geometric phase from range offset shape=%s",
        getattr(phase, "shape", ()),
    )
    return out, "topo"


def stage_topo(*_args: object, **_kwargs: object) -> None:
    """Removed API (PROPOSAL-0017). Geometric phase lives in coreg only."""
    message = (
        "stage_topo was removed (PROPOSAL-0017). Geometric phase is applied "
        "during coregistration via apply_geometric_phase_from_range_offset; "
        "interferogram formation must not re-apply flat/topo."
    )
    logger.error(message)
    raise RuntimeError(message)

"""Shared Enhanced Spectral Diversity (ESD) result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ESDResult:
    """Azimuth shift and quality from spectral-diversity estimation."""

    azimuth_shift_px: float
    """Residual azimuth shift in pixels (positive = secondary lags reference)."""
    coherence: float
    """Mean coherence of the differential interferogram (0..1)."""
    phase_rad: float
    """Estimated differential phase in radians."""

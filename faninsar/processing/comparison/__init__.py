"""Reference comparison harness for FanInSAR pair products."""

from __future__ import annotations

from faninsar.processing.comparison.metrics import (
    CoherenceComparisonResult,
    GeolocationResidualResult,
    PairComparisonError,
    PairComparisonReport,
    PhaseComparisonResult,
    RewrapResidualResult,
    coherence_absolute_error,
    compare_pair_products,
    geolocation_residual,
    phase_circular_rmse,
    rewrap_residual_stats,
)

__all__ = [
    "CoherenceComparisonResult",
    "GeolocationResidualResult",
    "PairComparisonError",
    "PairComparisonReport",
    "PhaseComparisonResult",
    "RewrapResidualResult",
    "coherence_absolute_error",
    "compare_pair_products",
    "geolocation_residual",
    "phase_circular_rmse",
    "rewrap_residual_stats",
]

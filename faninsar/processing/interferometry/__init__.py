"""Complex interferometry operators."""

from __future__ import annotations

from .flatten import (
    apply_residual_phase_screen_to_products,
    compute_topographic_phase,
    copernicus_glo30_dem,
    estimate_residual_azimuth_ramp,
    estimate_residual_height_poly_screen,
    estimate_residual_phase_screen,
    estimate_residual_phase_screen_from_unwrapped,
    estimate_residual_topographic_scale,
    estimate_tiled_residual_height_screen,
    remove_azimuth_phase_ramp,
    remove_residual_phase_screen,
    remove_topographic_phase,
)
from .pair import (
    InterferogramProduct,
    form_interferogram,
    goldstein_filter,
    mask_invalid_looks,
    validate_coherence_window,
)
from .phase_filter import (
    BoxcarFilter,
    FilterProvenance,
    GaussianFilter,
    GoldsteinWerner,
    PhaseFilter,
    PhaseFilterResult,
)

__all__ = [
    "BoxcarFilter",
    "FilterProvenance",
    "GaussianFilter",
    "GoldsteinWerner",
    "InterferogramProduct",
    "PhaseFilter",
    "PhaseFilterResult",
    "apply_residual_phase_screen_to_products",
    "compute_topographic_phase",
    "copernicus_glo30_dem",
    "estimate_residual_azimuth_ramp",
    "estimate_residual_height_poly_screen",
    "estimate_residual_phase_screen",
    "estimate_residual_phase_screen_from_unwrapped",
    "estimate_residual_topographic_scale",
    "estimate_tiled_residual_height_screen",
    "form_interferogram",
    "goldstein_filter",
    "mask_invalid_looks",
    "remove_azimuth_phase_ramp",
    "remove_residual_phase_screen",
    "remove_topographic_phase",
    "validate_coherence_window",
]

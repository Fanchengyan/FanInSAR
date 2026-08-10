"""Coregistration offsets and complex resampling."""

from __future__ import annotations

from .dense_geometry import dense_geometry_offsets, geometry_offset_window_extent
from .esd import ESDResult, estimate_azimuth_shift_esd
from .geometric_phase import (
    apply_geometric_phase_from_range_offset,
    geometric_phase_from_range_offset,
    phase_per_range_pixel,
    stage_topo,
)
from .geometry_coreg import (
    build_offset_field,
    combine_offset_fields,
    geometry_coarse_shift,
    refine_shift_with_correlation,
)
from .misreg_network import DateMisreg, MisregArc, invert_pair_misregistration
from .offsets import (
    OffsetFieldResult,
    PatchAmplitudeShiftResult,
    estimate_global_shift,
    estimate_patch_amplitude_shift,
    geometry_shift_offsets,
    refine_peak_subpixel,
    resample_complex,
    resample_complex_deramped_reramp,
)

__all__ = [
    "DateMisreg",
    "ESDResult",
    "MisregArc",
    "OffsetFieldResult",
    "PatchAmplitudeShiftResult",
    "apply_geometric_phase_from_range_offset",
    "build_offset_field",
    "combine_offset_fields",
    "dense_geometry_offsets",
    "estimate_azimuth_shift_esd",
    "estimate_global_shift",
    "estimate_patch_amplitude_shift",
    "geometric_phase_from_range_offset",
    "geometry_coarse_shift",
    "geometry_offset_window_extent",
    "geometry_shift_offsets",
    "invert_pair_misregistration",
    "phase_per_range_pixel",
    "refine_peak_subpixel",
    "refine_shift_with_correlation",
    "resample_complex",
    "resample_complex_deramped_reramp",
    "stage_topo",
]

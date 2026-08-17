"""Coregistration offsets and complex resampling."""

from __future__ import annotations

from .ampcor_backend import (
    AmpcorBackendRegistry,
    AmpcorCandidateError,
    AmpcorEnergyCandidate,
    AmpcorNccCandidate,
    AmpcorNccCandidateError,
    AmpcorNccRuntimeProfile,
    ampcor_ncc_postprocess_reference,
    eager_ampcor_candidate,
    native_ncc_workspace_bytes,
    native_workspace_bytes,
    prepare_ampcor_compile,
    prepare_ampcor_native,
    prepare_ampcor_ncc_native,
    torch_integral_energy,
)
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
    resolve_ampcor_policy,
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
    "AmpcorBackendRegistry",
    "AmpcorCandidateError",
    "AmpcorEnergyCandidate",
    "AmpcorNccCandidate",
    "AmpcorNccCandidateError",
    "AmpcorNccRuntimeProfile",
    "DateMisreg",
    "ESDResult",
    "MisregArc",
    "OffsetFieldResult",
    "PatchAmplitudeShiftResult",
    "ampcor_ncc_postprocess_reference",
    "apply_geometric_phase_from_range_offset",
    "build_offset_field",
    "combine_offset_fields",
    "dense_geometry_offsets",
    "eager_ampcor_candidate",
    "estimate_azimuth_shift_esd",
    "estimate_global_shift",
    "estimate_patch_amplitude_shift",
    "geometric_phase_from_range_offset",
    "geometry_coarse_shift",
    "geometry_offset_window_extent",
    "geometry_shift_offsets",
    "invert_pair_misregistration",
    "native_ncc_workspace_bytes",
    "native_workspace_bytes",
    "phase_per_range_pixel",
    "prepare_ampcor_compile",
    "prepare_ampcor_native",
    "prepare_ampcor_ncc_native",
    "refine_peak_subpixel",
    "refine_shift_with_correlation",
    "resample_complex",
    "resample_complex_deramped_reramp",
    "resolve_ampcor_policy",
    "stage_topo",
    "torch_integral_energy",
]

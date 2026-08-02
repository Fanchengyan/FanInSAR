"""Plain callable production stages (no missions imports)."""

from __future__ import annotations

from typing import Any

from faninsar.core.physical import PhysicalType

# Re-export production stages under stable stage_* names for Workflow wiring.
# Implementations live in processing.pipeline.production (rehomed gradually).


def stage_read(state: Any, **kwargs: Any) -> Any:
    """Load reference/secondary scene metadata (entry stage)."""
    return state


stage_read.input_type = None  # type: ignore[attr-defined]
stage_read.output_type = PhysicalType.SLC_RAW  # type: ignore[attr-defined]


def stage_deramp(state: Any, **kwargs: Any) -> Any:
    """Deramp TOPS carrier on a pair state."""
    from faninsar.processing.pipeline.production import stage_deramp as _impl

    return _impl(state)


stage_deramp.input_type = PhysicalType.SLC_RAW  # type: ignore[attr-defined]
stage_deramp.output_type = PhysicalType.SLC_DERAMPED  # type: ignore[attr-defined]


def stage_coreg(state: Any, **kwargs: Any) -> Any:
    """Coregister secondary to reference (radar or geo grid)."""
    from faninsar.processing.pipeline.production import stage_coregister as _impl

    return _impl(state)


stage_coreg.input_type = PhysicalType.SLC_DERAMPED  # type: ignore[attr-defined]
stage_coreg.output_type = PhysicalType.SLC_COREG  # type: ignore[attr-defined]


def stage_ifg(state: Any, **kwargs: Any) -> Any:
    """Form complex interferogram + multilook + Goldstein."""
    from faninsar.processing.pipeline.production import stage_interferogram as _impl

    return _impl(state)


stage_ifg.input_type = PhysicalType.SLC_COREG  # type: ignore[attr-defined]
stage_ifg.output_type = PhysicalType.IFG_COMPLEX  # type: ignore[attr-defined]


def stage_flatten(state: Any, **kwargs: Any) -> Any:
    """Remove topographic reference phase."""
    from faninsar.processing.pipeline.production import stage_flatten as _impl

    return _impl(state)


stage_flatten.input_type = PhysicalType.IFG_COMPLEX  # type: ignore[attr-defined]
stage_flatten.output_type = PhysicalType.IFG_FLATTENED  # type: ignore[attr-defined]


def stage_unwrap(state: Any, **kwargs: Any) -> Any:
    """Spatial phase unwrapping (IRLS or snaphu)."""
    from faninsar.processing.pipeline.production import stage_unwrap as _impl

    return _impl(state)


stage_unwrap.input_type = PhysicalType.IFG_FLATTENED  # type: ignore[attr-defined]
stage_unwrap.output_type = PhysicalType.PHASE_UNWRAPPED  # type: ignore[attr-defined]


def stage_geocode(state: Any, **kwargs: Any) -> Any:
    """Geocode radar products when still on the radar grid."""
    from faninsar.processing.pipeline.production import stage_geocode as _impl

    return _impl(state)


stage_geocode.input_type = PhysicalType.PHASE_UNWRAPPED  # type: ignore[attr-defined]
stage_geocode.output_type = PhysicalType.PHASE_UNWRAPPED  # type: ignore[attr-defined]


def stage_write(state: Any, **kwargs: Any) -> Any:
    """Write Zarr + STAC pair products."""
    from faninsar.processing.pipeline.production import stage_write as _impl

    return _impl(state)


__all__ = [
    "stage_coreg",
    "stage_deramp",
    "stage_flatten",
    "stage_geocode",
    "stage_ifg",
    "stage_read",
    "stage_unwrap",
    "stage_write",
]

"""Consistency between Network asset kinds and the geocode kernel table."""

from __future__ import annotations

from faninsar.network.products import (
    ASSET_KIND_GEOCODE_KERNELS,
    AssetKind,
)
from faninsar.network.readers.metadata import (
    CONTINUOUS_ASSETS,
    PHASE_ASSETS,
)


def test_kernel_table_covers_every_asset_kind_member() -> None:
    """Every enumerated AssetKind member has exactly one geocode kernel."""
    members = list(AssetKind)
    assert set(ASSET_KIND_GEOCODE_KERNELS) == set(members)
    assert len(members) == 8


def test_kernel_values_are_stable() -> None:
    """The AssetKind to kernel assignment matches the confirmed design."""
    expected = {
        AssetKind.COMPLEX_INTERFEROGRAM: "complex",
        AssetKind.WRAPPED_PHASE: "phase",
        AssetKind.UNWRAPPED_PHASE: "continuous",
        AssetKind.DISPLACEMENT: "continuous",
        AssetKind.LOS: "continuous",
        AssetKind.AMPLITUDE: "continuous",
        AssetKind.COHERENCE: "continuous",
        AssetKind.LABEL: "label",
    }
    assert dict(ASSET_KIND_GEOCODE_KERNELS) == expected


def test_kernel_table_is_exported() -> None:
    """The kernel table is part of the module's public surface."""
    import faninsar.network.products as products_module

    assert "ASSET_KIND_GEOCODE_KERNELS" in products_module.__all__


def test_unwrapped_phase_stays_continuous_and_wrapped_is_phase() -> None:
    """Wrapped phase is cyclic (phase kernel); unwrapped stays continuous.

    Unwrapped phase resampled with bilinear is locally unreliable across
    connectivity-component boundaries (2*pi jumps); this is a documented
    limitation, not a membership change.  The reader metadata sets mirror
    the kernel assignment for the two phase-bearing asset kinds.
    """
    assert ASSET_KIND_GEOCODE_KERNELS[AssetKind.WRAPPED_PHASE] == "phase"
    assert ASSET_KIND_GEOCODE_KERNELS[AssetKind.UNWRAPPED_PHASE] == "continuous"
    assert "wrapped_phase" in PHASE_ASSETS
    assert "wrapped_phase" not in CONTINUOUS_ASSETS
    assert "unw_phase" in CONTINUOUS_ASSETS
    assert "unw_phase" not in PHASE_ASSETS

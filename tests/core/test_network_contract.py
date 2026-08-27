"""Tests for the immutable Network identity and phase contracts."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from faninsar.core import Pair
from faninsar.core.network import (
    AcquisitionKey,
    AssetKind,
    AssetTransform,
    AssetTransformOperation,
    Network,
    NetworkProduct,
    NetworkProductIndex,
    NetworkProductKey,
    PhaseConvention,
)
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.stack.session import Stack


def _key(acquisition_id: str, *, swath: str = "IW1") -> AcquisitionKey:
    """Build a representative Sentinel-1 acquisition key."""
    return AcquisitionKey(acquisition_id, "frame-1", swath, "S1", "VV")


def _product(
    *,
    phase_convention: PhaseConvention = PhaseConvention.PRIMARY_MINUS_SECONDARY,
    swath: str = "IW1",
) -> NetworkProduct:
    """Build a representative immutable product record."""
    primary = _key("20200101", swath=swath)
    secondary = _key("20200113", swath=swath)
    kind = AssetKind.COMPLEX_INTERFEROGRAM
    return NetworkProduct(
        key=NetworkProductKey(primary, secondary, kind),
        asset_location="generation-1/ifg/20200101_20200113.zarr",
        geometry_identity="geo-grid-a",
        source_software="isce3",
        phase_convention=phase_convention,
        asset_transform=AssetTransform.for_convention(kind, phase_convention),
    )


def test_acquisition_key_disambiguates_same_date_products() -> None:
    """Frame/swath/channel/polarization are part of stable identity."""
    first = _key("20200101", swath="IW1")
    second = _key("20200101", swath="IW2")
    assert first != second
    assert first.cohort != second.cohort
    assert first.canonical == "20200101|frame-1|IW1|S1|VV"


@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        (AssetKind.COMPLEX_INTERFEROGRAM, AssetTransformOperation.COMPLEX_CONJUGATE),
        (
            AssetKind.WRAPPED_PHASE,
            AssetTransformOperation.WRAPPED_PHASE_NEGATE_MODULO,
        ),
        (AssetKind.UNWRAPPED_PHASE, AssetTransformOperation.UNWRAPPED_PHASE_NEGATE),
        (AssetKind.DISPLACEMENT, AssetTransformOperation.SCALAR_NEGATE),
    ],
)
def test_reverse_phase_convention_has_asset_specific_transform(
    kind: AssetKind,
    operation: AssetTransformOperation,
) -> None:
    """Reverse phase orientation never relies on a filename convention."""
    transform = AssetTransform.for_convention(
        kind,
        PhaseConvention.SECONDARY_MINUS_PRIMARY,
    )
    assert transform.operation is operation


def test_phase_transforms_apply_expected_math() -> None:
    """Complex, wrapped, and unwrapped assets normalize with distinct rules."""
    complex_asset = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    assert np.allclose(
        AssetTransform.for_convention(
            AssetKind.COMPLEX,
            PhaseConvention.SECONDARY_MINUS_PRIMARY,
        ).apply(complex_asset),
        np.conjugate(complex_asset),
    )
    wrapped = np.asarray([np.pi * 0.75, -np.pi * 0.75])
    expected_wrapped = np.angle(np.exp(-1j * wrapped))
    actual_wrapped = AssetTransform.for_convention(
        AssetKind.WRAPPED_PHASE,
        PhaseConvention.SECONDARY_MINUS_PRIMARY,
    ).apply(wrapped)
    assert np.allclose(actual_wrapped, expected_wrapped)
    unwrapped = np.asarray([-4.0, 2.0])
    assert np.array_equal(
        AssetTransform.for_convention(
            AssetKind.UNWRAPPED_PHASE,
            PhaseConvention.SECONDARY_MINUS_PRIMARY,
        ).apply(unwrapped),
        -unwrapped,
    )


def test_transform_and_product_convention_must_agree() -> None:
    """A stale transform declaration is rejected before indexing."""
    with pytest.raises(ValueError, match="phase convention"):
        NetworkProduct(
            key=NetworkProductKey(
                _key("20200101"), _key("20200113"), AssetKind.COMPLEX
            ),
            asset_location="ifg.zarr",
            geometry_identity="geo-grid-a",
            source_software="isce3",
            phase_convention=PhaseConvention.PRIMARY_MINUS_SECONDARY,
            asset_transform=AssetTransform.for_convention(
                AssetKind.COMPLEX,
                PhaseConvention.SECONDARY_MINUS_PRIMARY,
            ),
        )


def test_product_index_rejects_duplicates_and_mixed_cohorts() -> None:
    """A Network index cannot silently mix or overwrite products."""
    product = _product()
    with pytest.raises(ValueError, match="unique"):
        NetworkProductIndex((product, product))

    mixed = _product(swath="IW2")
    index = NetworkProductIndex((product, mixed))
    with pytest.raises(ValueError, match="homogeneous"):
        index.homogeneous()


def test_pair_retains_chronological_primary_secondary_roles() -> None:
    """Network identity additions do not alter FanInSAR Pair ordering."""
    pair = Pair((pd.Timestamp("2020-01-13"), pd.Timestamp("2020-01-01")))
    assert pair.primary == pd.Timestamp("2020-01-01")
    assert pair.secondary == pd.Timestamp("2020-01-13")
    assert pair.name == "20200101_20200113"


def test_unknown_phase_convention_and_transform_operation_fail_closed() -> None:
    """Free-form phase labels and mismatched operations are not accepted."""
    with pytest.raises(ValueError, match="unknown phase convention"):
        AssetTransform.for_convention(AssetKind.COMPLEX, "filename_order")
    with pytest.raises(ValueError, match="not valid"):
        AssetTransform(
            AssetKind.COMPLEX,
            PhaseConvention.PRIMARY_MINUS_SECONDARY,
            AssetTransformOperation.COMPLEX_CONJUGATE,
        )


def test_network_analysis_is_fail_closed_until_generation_refresh() -> None:
    """The analysis seam cannot run before a complete generation is admitted."""

    class ConcreteNetwork(Network):
        """Test implementation recording the generation passed to analysis."""

        def _analyze_network_products(
            self,
            products: NetworkProductIndex,
            *,
            generation_id: str,
        ) -> tuple[str, int]:
            return generation_id, len(products.products)

    network = ConcreteNetwork()
    assert not network.analysis_ready
    with pytest.raises(RuntimeError, match="before"):
        network.analyze_time_series()

    product = _product()
    network.refresh_generation("generation-a", (product,))
    assert network.analysis_ready
    assert network.analyze_time_series() == ("generation-a", 1)


def test_invalid_refresh_does_not_replace_previous_generation() -> None:
    """Generation replacement is atomic with respect to cohort validation."""
    network = Network()
    network.refresh_generation("generation-a", (_product(),))
    with pytest.raises(ValueError, match="homogeneous"):
        network.refresh_generation("generation-b", (_product(), _product(swath="IW2")))
    assert network.network_generation_id == "generation-a"
    assert network.analysis_ready


def test_stack_analysis_delegates_through_network_base(monkeypatch) -> None:
    """Stack uses inherited Network admission before its existing inversion."""
    stack = object.__new__(Stack)
    stack.refresh_generation("generation-a", (_product(),))
    stack.unwrap_result = object()
    monkeypatch.setattr(
        stack,
        "invert_timeseries",
        lambda **_: "inversion-result",
    )
    assert stack.analyze_time_series() == "inversion-result"


def test_stack_analysis_keeps_ifg_leases_until_solver_returns(monkeypatch) -> None:
    """Stack pins, heartbeats, and releases IFG stores around the solver."""
    stack = object.__new__(Stack)
    stack.refresh_generation("generation-a", (_product(),))
    stack.unwrap_result = object()
    stack.config = SimpleNamespace(multilook=(1, 1))
    stack.pairs = object()
    events: list[str] = []

    class Lease:
        """Minimal lease double recording the analysis lifecycle."""

        def heartbeat(self) -> None:
            events.append("heartbeat")

    class Store:
        """Minimal artifact store double with one durable lease."""

        _lease = Lease()
        manifest_digest = "manifest-digest"

        def close(self) -> None:
            events.append("close")

    store = Store()
    stack._network_generation_id = hashlib.sha256(
        store.manifest_digest.encode()
    ).hexdigest()
    monkeypatch.setattr(
        stack,
        "_pair_artifact_stores",
        lambda **_: [store],
    )

    def solve(**kwargs: object) -> str:
        assert kwargs["_artifact_stores"] == [store]
        events.append("solver")
        assert events == ["heartbeat", "solver"]
        return "inversion-result"

    monkeypatch.setattr(stack, "invert_timeseries", solve)
    assert stack.analyze_time_series() == "inversion-result"
    assert events == ["heartbeat", "solver", "heartbeat", "close"]


def test_stack_analysis_closes_ifg_leases_when_heartbeat_fails(monkeypatch) -> None:
    """A failed lease heartbeat aborts analysis and still releases stores."""
    stack = object.__new__(Stack)
    stack.refresh_generation("generation-a", (_product(),))
    stack.unwrap_result = object()
    stack.config = SimpleNamespace(multilook=(1, 1))
    stack.pairs = object()
    events: list[str] = []

    class Lease:
        """Lease double that fails closed on admission heartbeat."""

        def heartbeat(self) -> None:
            events.append("heartbeat")
            raise InvalidProcessingStateError("lease expired")

    class Store:
        """Artifact store double used to verify cleanup after failure."""

        _lease = Lease()
        manifest_digest = "manifest-digest"

        def close(self) -> None:
            events.append("close")

    store = Store()
    stack._network_generation_id = hashlib.sha256(
        store.manifest_digest.encode()
    ).hexdigest()
    monkeypatch.setattr(stack, "_pair_artifact_stores", lambda **_: [store])
    monkeypatch.setattr(
        stack,
        "invert_timeseries",
        lambda **_: pytest.fail("solver must not run after lease failure"),
    )
    with pytest.raises(InvalidProcessingStateError, match="expired"):
        stack.analyze_time_series()
    assert events == ["heartbeat", "close"]

"""Public merge_bursts midpoint selection and VALID fallback (PROPOSAL-0074)."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.mosaicking import (
    BurstWindow,
    MergeLayoutWindow,
    MergeScope,
    merge_bursts,
)

SCOPE = MergeScope(
    domain="radar",
    track_id="T100_A",
    channel="vv",
    reference_id="ref-20240101",
    grid_identity="a" * 64,
    polygon_identity="poly-0",
)


def _scope(**changes: object) -> MergeScope:
    payload = {
        "domain": SCOPE.domain,
        "track_id": SCOPE.track_id,
        "channel": SCOPE.channel,
        "reference_id": SCOPE.reference_id,
        "grid_identity": SCOPE.grid_identity,
        "polygon_identity": SCOPE.polygon_identity,
        **changes,
    }
    return MergeScope(**payload)  # type: ignore[arg-type]


def _burst(
    burst_id: str,
    window: tuple[int, int, int, int],
    values: np.ndarray,
    valid: np.ndarray,
    *,
    acquisitions: tuple[str, ...] = ("20240101", "20240113"),
    phase_convention: str = "reramped",
    coherence: np.ndarray | None = None,
    primary_power: np.ndarray | None = None,
    secondary_power: np.ndarray | None = None,
    scope: MergeScope = SCOPE,
) -> BurstWindow:
    return BurstWindow(
        scope=scope,
        burst_id=burst_id,
        acquisitions=acquisitions,
        write_ids=(f"{burst_id}-w1", f"{burst_id}-w2"),
        phase_convention=phase_convention,
        window=window,
        complex=np.asarray(values, dtype=np.complex64),
        valid_mask=np.asarray(valid, dtype=bool),
        coherence=coherence,
        primary_power=primary_power,
        secondary_power=secondary_power,
    )


def _two_burst_azimuth_layout() -> tuple[list[BurstWindow], MergeLayoutWindow]:
    """Two same-swath bursts overlapping in azimuth on a 6x4 ROI.

    Burst A occupies rows ``[0, 4)``, burst B rows ``[2, 6)``. Overlap
    ``[2, 4)`` cuts at ``2 + 2 // 2 = 3``: rows ``< 3`` prefer A, rows
    ``>= 3`` prefer B.
    """
    window_a = (0, 4, 0, 4)
    window_b = (2, 6, 0, 4)
    layout_window = (0, 6, 0, 4)
    values_a = np.full((4, 4), 1 + 0j, dtype=np.complex64)
    values_b = np.full((4, 4), 2 + 0j, dtype=np.complex64)
    valid_a = np.ones((4, 4), dtype=bool)
    valid_b = np.ones((4, 4), dtype=bool)
    products = [
        _burst("iw1-b1", window_a, values_a, valid_a),
        _burst("iw1-b2", window_b, values_b, valid_b),
    ]
    owner = np.zeros((6, 4), dtype=np.uint16)
    owner[0:3] = 1
    owner[3:6] = 2
    support = np.ones((6, 4), dtype=bool)
    layout = MergeLayoutWindow(
        scope=SCOPE,
        window=layout_window,
        source_ids=("iw1-b1", "iw1-b2"),
        owner=owner,
        scientific_support=support,
    )
    return products, layout


def test_midpoint_owner_selects_earlier_then_later_burst() -> None:
    """Preferred midpoint labels copy native samples without averaging."""
    products, layout = _two_burst_azimuth_layout()
    merged = merge_bursts(products, layout=layout)
    assert merged.window == layout.window
    assert merged.complex.dtype == np.complex64
    np.testing.assert_array_equal(merged.complex[:3], 1 + 0j)
    np.testing.assert_array_equal(merged.complex[3:], 2 + 0j)
    np.testing.assert_array_equal(merged.valid_mask, True)
    np.testing.assert_array_equal(merged.source[:3], 1)
    np.testing.assert_array_equal(merged.source[3:], 2)


def test_input_order_does_not_change_midpoint_output() -> None:
    """Selection keys on layout source_ids, not caller list order."""
    products, layout = _two_burst_azimuth_layout()
    forward = merge_bursts(products, layout=layout)
    reversed_products = list(reversed(products))
    backward = merge_bursts(reversed_products, layout=layout)
    np.testing.assert_array_equal(forward.complex, backward.complex)
    np.testing.assert_array_equal(forward.source, backward.source)
    np.testing.assert_array_equal(forward.valid_mask, backward.valid_mask)


def test_valid_fallback_uses_next_covering_burst() -> None:
    """When the preferred owner is invalid, the next VALID source fills."""
    products, layout = _two_burst_azimuth_layout()
    preferred = products[0]
    hole = np.array(preferred.valid_mask, copy=True)
    hole[2, 1] = False
    products[0] = _burst(
        preferred.burst_id,
        preferred.window,
        preferred.complex,
        hole,
    )
    merged = merge_bursts(products, layout=layout)
    assert merged.complex[2, 1] == pytest.approx(2 + 0j)
    assert bool(merged.valid_mask[2, 1])
    assert merged.source[2, 1] == 2
    assert merged.complex[2, 0] == pytest.approx(1 + 0j)
    assert merged.source[2, 0] == 1


def test_supported_zero_complex_is_not_treated_as_nodata() -> None:
    """Explicit masks keep a valid 0j sample; amplitude is not the support test."""
    products, layout = _two_burst_azimuth_layout()
    values = np.array(products[0].complex, copy=True)
    values[0, 0] = 0 + 0j
    products[0] = _burst(
        products[0].burst_id,
        products[0].window,
        values,
        products[0].valid_mask,
    )
    merged = merge_bursts(products, layout=layout)
    assert merged.complex[0, 0] == pytest.approx(0 + 0j)
    assert bool(merged.valid_mask[0, 0])
    assert merged.source[0, 0] == 1


def test_optional_fields_follow_the_same_selected_candidate() -> None:
    """Coherence and endpoint powers are copied from the chosen burst only."""
    products, layout = _two_burst_azimuth_layout()
    for index, product in enumerate(products, start=1):
        coh = np.full(product.complex.shape, float(index) * 0.1, dtype=np.float32)
        primary = np.full(product.complex.shape, float(index) * 3.0, dtype=np.float32)
        secondary = np.full(product.complex.shape, float(index) * 5.0, dtype=np.float32)
        products[index - 1] = _burst(
            product.burst_id,
            product.window,
            product.complex,
            product.valid_mask,
            coherence=coh,
            primary_power=primary,
            secondary_power=secondary,
        )
    hole = np.array(products[0].valid_mask, copy=True)
    hole[2, 1] = False
    first = products[0]
    products[0] = _burst(
        first.burst_id,
        first.window,
        first.complex,
        hole,
        coherence=first.coherence,
        primary_power=first.primary_power,
        secondary_power=first.secondary_power,
    )
    merged = merge_bursts(products, layout=layout)
    assert merged.coherence is not None
    assert merged.primary_power is not None
    assert merged.secondary_power is not None
    assert merged.coherence[0, 0] == pytest.approx(0.1)
    assert merged.primary_power[0, 0] == pytest.approx(3.0)
    assert merged.coherence[2, 1] == pytest.approx(0.2)
    assert merged.primary_power[2, 1] == pytest.approx(6.0)
    assert merged.secondary_power[2, 1] == pytest.approx(10.0)


def test_uncovered_preferred_pixel_without_fallback_is_invalid() -> None:
    """No VALID covering burst leaves nodata, not a manufactured sample."""
    products, layout = _two_burst_azimuth_layout()
    hole_a = np.array(products[0].valid_mask, copy=True)
    hole_a[0, 0] = False
    products[0] = _burst(
        products[0].burst_id,
        products[0].window,
        products[0].complex,
        hole_a,
    )
    merged = merge_bursts(products, layout=layout)
    assert not bool(merged.valid_mask[0, 0])
    assert merged.source[0, 0] == 0
    assert merged.complex[0, 0] == pytest.approx(0 + 0j)


@pytest.mark.parametrize(
    "changes",
    [
        {"domain": "geo"},
        {"track_id": "T101_A"},
        {"channel": "vh"},
        {"reference_id": "other-ref"},
        {"grid_identity": "b" * 64},
        {"polygon_identity": "poly-1"},
    ],
)
def test_incompatible_scope_is_rejected(changes: dict[str, str]) -> None:
    """Every MergeScope field is part of the admission key."""
    products, layout = _two_burst_azimuth_layout()
    products[0] = _burst(
        products[0].burst_id,
        products[0].window,
        products[0].complex,
        products[0].valid_mask,
        scope=_scope(**changes),
    )
    with pytest.raises(InvalidProcessingStateError, match="scope"):
        merge_bursts(products, layout=layout)


def test_mismatched_acquisitions_and_conventions_are_rejected() -> None:
    """IFG pairing requires matching endpoints and phase convention."""
    products, layout = _two_burst_azimuth_layout()
    products[1] = _burst(
        products[1].burst_id,
        products[1].window,
        products[1].complex,
        products[1].valid_mask,
        acquisitions=("20240101", "20240125"),
    )
    with pytest.raises(InvalidProcessingStateError, match="acquisition"):
        merge_bursts(products, layout=layout)
    products, layout = _two_burst_azimuth_layout()
    products[1] = _burst(
        products[1].burst_id,
        products[1].window,
        products[1].complex,
        products[1].valid_mask,
        phase_convention="deramped",
    )
    with pytest.raises(InvalidProcessingStateError, match="phase_convention"):
        merge_bursts(products, layout=layout)


def test_unwrapped_and_merged_inputs_are_rejected() -> None:
    """The kernel only accepts original complex burst windows."""
    products, layout = _two_burst_azimuth_layout()
    with pytest.raises(InvalidProcessingStateError, match="phase_domain"):
        BurstWindow(
            scope=SCOPE,
            burst_id="iw1-b1",
            acquisitions=("20240101",),
            write_ids=("w1",),
            phase_convention="reramped",
            window=(0, 1, 0, 1),
            complex=np.ones((1, 1), dtype=np.complex64),
            valid_mask=np.ones((1, 1), dtype=bool),
            phase_domain="unwrapped",
        )
    with pytest.raises(InvalidProcessingStateError, match="representation"):
        merge_bursts(
            [
                BurstWindow(
                    scope=SCOPE,
                    burst_id="iw1-b1",
                    acquisitions=products[0].acquisitions,
                    write_ids=products[0].write_ids,
                    phase_convention=products[0].phase_convention,
                    window=products[0].window,
                    complex=products[0].complex,
                    valid_mask=products[0].valid_mask,
                    representation="merged",
                )
            ],
            layout=layout,
        )

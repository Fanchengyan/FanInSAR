"""Cross-satellite common-burst pairing by along-track position.

Sentinel-1 products clip the burst grid to different start lines: a given
ground strip may sit at burst 7 of a Sentinel-1C swath but burst 5 of the
matching Sentinel-1D swath.  ``_common_burst_pairs`` must recover that whole
index offset (ISCE2 ``getBurstOffset`` model) so the same ground is paired.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1.processing import (
    _common_burst_indices,
    _common_burst_pairs,
)
from faninsar.missions.s1.types import S1Burst, S1Swath

from .test_measure import _LINES, _SAMPLES

_BURST_STEP_S = 2.756
# One burst's azimuth extent equals the burst step, matching real TOPS
# products where consecutive bursts are separated by ~1 lines_per_burst.
_ATI_S = _BURST_STEP_S / _LINES


def _offset_swath(
    count: int,
    *,
    first_burst: int,
    azimuth_origin: datetime,
    anx_origin: float,
) -> S1Swath:
    """Build a swath whose bursts step ``_BURST_STEP_S`` apart on the orbit.

    ``first_burst`` shifts where burst index 0 lands on the shared orbit grid
    (in burst steps); ``anx_origin`` supplies distinct along-track times so
    the ANX axis is exercised (real products carry annotation-derived ANX
    times).
    """
    bursts = []
    for index in range(count):
        orbit_burst = first_burst + index
        start_line = orbit_burst * _LINES
        azimuth_time = azimuth_origin + timedelta(seconds=start_line * _ATI_S)
        bursts.append(
            S1Burst(
                index=index,
                azimuth_time=azimuth_time,
                sensing_time=azimuth_time,
                azimuth_anx_time_s=anx_origin + _BURST_STEP_S * index,
                byte_offset=index * _LINES * _SAMPLES * 4,
                lines=_LINES,
                samples=_SAMPLES,
                first_valid_sample=tuple([0] * _LINES),
                last_valid_sample=tuple([_SAMPLES - 1] * _LINES),
            )
        )
    start = bursts[0].azimuth_time
    stop = bursts[-1].azimuth_time + timedelta(seconds=(_LINES - 1) * _ATI_S)
    return S1Swath(
        swath="IW2",
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path="memory://measurement.tiff",
        lines=_LINES * count,
        samples=_SAMPLES,
        lines_per_burst=_LINES,
        samples_per_burst=_SAMPLES,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=_ATI_S,
        slant_range_time_s=0.005,
        range_sampling_rate_hz=64_000_000.0,
        radar_frequency_hz=5.405e9,
        sensing_start=start,
        sensing_stop=stop,
        bursts=tuple(bursts),
        orbit=OrbitMetadata(
            reference_frame="ITRF",
            source="test",
            vectors=(
                OrbitStateVector(
                    time=start,
                    position_m=(7_000_000.0, 0.0, 0.0),
                    velocity_m_s=(0.0, 7_500.0, 0.0),
                ),
            ),
        ),
        doppler_centroid=(
            DopplerCentroidPolynomial(
                coefficients_hz=(100.0, -0.01),
                reference_range_m=800_000.0,
                reference_time_s=0.0,
            ),
        ),
        azimuth_fm_rate=(),
        azimuth_steering_rate_rad_s=0.0,
    )


def test_common_burst_pairs_recovers_whole_index_offset() -> None:
    """S1C(9 bursts) vs S1D(10 bursts) offset by -2 pairs the same ground."""
    s1c = _offset_swath(
        9,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 1, 23, 26, 6, tzinfo=UTC),
        anx_origin=1_000.0,
    )
    # S1D starts two orbit-grid burst steps later: its burst *i* covers the
    # ground of S1C burst *i + 2*.
    s1d = _offset_swath(
        10,
        first_burst=2,
        azimuth_origin=datetime(2026, 7, 15, 23, 26, 27, tzinfo=UTC),
        anx_origin=1_000.0 + 2 * _BURST_STEP_S,
    )
    pairs = _common_burst_pairs(s1c, s1d)
    assert pairs[7] == 5
    assert pairs[2] == 0
    assert _common_burst_indices(s1c, s1d) == list(range(2, 9))


def test_common_burst_pairs_same_grid_is_identity() -> None:
    """Same-satellite products keep the index-for-index mapping."""
    ref = _offset_swath(
        6,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
        anx_origin=500.0,
    )
    sec = _offset_swath(
        6,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC),
        anx_origin=500.0,
    )
    pairs = _common_burst_pairs(ref, sec)
    assert pairs == {index: index for index in range(6)}


def test_common_burst_pairs_drops_unmatched_ground() -> None:
    """Bursts with no secondary counterpart on the same strip are omitted."""
    ref = _offset_swath(
        8,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
        anx_origin=200.0,
    )
    # Secondary only covers the last three bursts' worth of ground.
    sec = _offset_swath(
        3,
        first_burst=5,
        azimuth_origin=datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC),
        anx_origin=200.0 + 5 * _BURST_STEP_S,
    )
    pairs = _common_burst_pairs(ref, sec)
    assert set(pairs.values()) == {0, 1, 2}
    assert set(pairs) == {5, 6, 7}


# ---------------------------------------------------------------------------
# Stage 4: exact absolute_burst_id pairing + cross-product swath lookup
# ---------------------------------------------------------------------------


def _with_absolute_ids(swath: S1Swath, first_id: int) -> S1Swath:
    """Return ``swath`` with ``absolute_burst_id`` = first_id + index."""
    return replace(
        swath,
        bursts=tuple(
            replace(burst, absolute_burst_id=first_id + i)
            for i, burst in enumerate(swath.bursts)
        ),
    )


def test_absolute_id_pairs_across_shifted_grids() -> None:
    """Same ids pair exactly even when ANX positions disagree."""
    primary = _with_absolute_ids(
        _offset_swath(
            4,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
            anx_origin=100.0,
        ),
        5000,
    )
    # Secondary bursts carry the same ids but sit two grid steps later on
    # the orbit axis — ANX alone would offset the pairing; ids resolve it.
    secondary = _with_absolute_ids(
        _offset_swath(
            4,
            first_burst=2,
            azimuth_origin=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            anx_origin=100.0 + 2 * _BURST_STEP_S,
        ),
        5000,
    )
    # primary[i] and secondary[i] share id 5000+i → identity pairing by id.
    assert _common_burst_pairs(primary, secondary) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_absolute_id_leaves_unmatched_primary_unpaired() -> None:
    """An id-carrying primary without the same id is never guessed."""
    primary = _with_absolute_ids(
        _offset_swath(
            3,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
            anx_origin=0.0,
        ),
        9000,
    )
    secondary = _with_absolute_ids(
        _offset_swath(
            3,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            anx_origin=0.0,
        ),
        7000,
    )
    assert _common_burst_pairs(primary, secondary) == {}


def test_mixed_id_presence_falls_back_per_burst() -> None:
    """Id-less primaries still resolve through ANX beside exact id pairs."""
    primary = _with_absolute_ids(
        _offset_swath(
            3,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
            anx_origin=300.0,
        ),
        6000,
    )
    # Strip the id off primary burst 2 → it must pair through ANX.
    primary = replace(
        primary,
        bursts=tuple(
            replace(burst, absolute_burst_id=None) if i == 2 else burst
            for i, burst in enumerate(primary.bursts)
        ),
    )
    secondary = _with_absolute_ids(
        _offset_swath(
            3,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            anx_origin=300.0,
        ),
        6000,
    )
    assert _common_burst_pairs(primary, secondary) == {0: 0, 1: 1, 2: 2}


def test_id_primary_against_idless_secondary_falls_back() -> None:
    """Backward compat: an id-carrying primary still pairs with an
    annotation-only secondary through ANX — legacy products carry no ids."""
    primary = _with_absolute_ids(
        _offset_swath(
            3,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
            anx_origin=400.0,
        ),
        8000,
    )
    secondary = _offset_swath(
        3,
        first_burst=0,
        azimuth_origin=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
        anx_origin=400.0,
    )
    assert _common_burst_pairs(primary, secondary) == {0: 0, 1: 1, 2: 2}


def _unit_for(
    frame_index: int,
    swath_name: str,
    burst_index: int,
    first_id: int,
) -> object:
    """Minimal S1ReferenceUnit for ``_secondary_burst_match`` tests."""
    from unittest.mock import MagicMock

    from faninsar.missions.s1.coregister import S1ReferenceUnit

    scene = MagicMock()
    scene.swath = _with_absolute_ids(
        _offset_swath(
            2,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
            anx_origin=0.0,
        ),
        first_id,
    )
    return S1ReferenceUnit(
        tag=f"f{frame_index}_{swath_name}_b{burst_index}",
        frame_index=frame_index,
        swath=swath_name,
        burst_index=burst_index,
        scene=scene,
        native_shape=(_LINES, _SAMPLES),
    )


def _product_for(name: str, swaths: tuple[S1Swath, ...]) -> object:
    """Minimal S1Product over ``swaths``."""
    from faninsar.missions.s1.types import S1Product

    return S1Product(
        product_id=name,
        source_path=Path(name),
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=("VV",),
        swaths=swaths,
    )


def test_secondary_burst_match_finds_swath_in_sibling_product() -> None:
    """Stage-3 layout: frame-slot product lacks the swath; sibling has it."""
    from faninsar.missions.s1.coregister import _secondary_burst_match

    iw1 = _with_absolute_ids(
        _offset_swath(
            1,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            anx_origin=0.0,
        ),
        77,
    )
    iw1 = replace(iw1, swath="IW1")
    iw3 = _with_absolute_ids(
        _offset_swath(
            1,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            anx_origin=_BURST_STEP_S,
        ),
        1001,
    )
    iw3 = replace(iw3, swath="IW3")
    products = (
        _product_for("frame0.SAFE", (iw1,)),
        _product_for("frame1.SAFE", (iw3,)),
    )
    unit = _unit_for(0, "IW3", 1, 1000)
    match = _secondary_burst_match(unit, products, unit.tag, "20260903")
    assert match is not None
    burst_index, product = match
    assert product is products[1]
    assert burst_index == 0


def test_secondary_burst_match_missing_swath_returns_none() -> None:
    """A swath absent from every secondary product yields ``None``."""
    from faninsar.missions.s1.coregister import _secondary_burst_match

    iw1 = _with_absolute_ids(
        _offset_swath(
            1,
            first_burst=0,
            azimuth_origin=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            anx_origin=0.0,
        ),
        77,
    )
    iw1 = replace(iw1, swath="IW1")
    products = (_product_for("frame0.SAFE", (iw1,)),)
    unit = _unit_for(0, "IW3", 0, 1000)
    assert _secondary_burst_match(unit, products, unit.tag, "20260903") is None

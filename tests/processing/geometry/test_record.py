"""Reference radar geometry record round trip (PROPOSAL-0059)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.record import (
    ORBIT_TRIM_EXTRA_VECTORS,
    radar_geometry_from_record,
    radar_geometry_record,
    radar_grid_from_record,
)

START = datetime(2021, 3, 4, 5, 6, 7, 123456, tzinfo=UTC)


def _orbit(count: int = 60, step_s: float = 10.0) -> OrbitMetadata:
    first = START - timedelta(seconds=step_s * count / 2)
    return OrbitMetadata(
        reference_frame="ITRF",
        source="POEORB",
        vectors=tuple(
            OrbitStateVector(
                time=first + timedelta(seconds=step_s * index),
                position_m=(7.0e6, 7.5e3 * step_s * index, 1.0 + index),
                velocity_m_s=(0.1 * index, 7.5e3, -0.5),
            )
            for index in range(count)
        ),
    )


def _grid() -> RadarGrid:
    return RadarGrid(
        shape=(1500, 2000),
        starting_slant_range_m=800_123.456789,
        range_spacing_m=2.329562114715323,
        sensing_start=START,
        azimuth_time_interval_s=0.002055556299999998,
        wavelength_m=0.05546576,
        look_direction="right",
    )


def test_record_round_trips_grid_and_geometry_exactly() -> None:
    """Strict JSON preserves every float and the timezone-aware start."""
    record = radar_geometry_record(_grid(), _orbit(), swath_column_offsets={"IW1": 0})
    decoded = json.loads(json.dumps(record, allow_nan=False))
    assert radar_grid_from_record(decoded) == _grid()
    geometry = radar_geometry_from_record(decoded)
    assert geometry.sensing_start == START
    assert geometry.starting_slant_range_m == _grid().starting_slant_range_m
    assert geometry.azimuth_time_interval_s == _grid().azimuth_time_interval_s
    assert decoded["swath_column_offsets"] == {"IW1": 0}


def test_orbit_is_trimmed_to_span_with_margin_and_extra_vectors() -> None:
    """Vectors outside span +- 60 s are dropped except two guard vectors."""
    orbit = _orbit()
    record = radar_geometry_record(_grid(), orbit)
    times = [datetime.fromisoformat(v["time"]) for v in record["orbit"]["vectors"]]
    span_stop = START + timedelta(seconds=1500 * _grid().azimuth_time_interval_s)
    inside = [
        vector.time
        for vector in orbit.vectors
        if START - timedelta(seconds=60)
        <= vector.time
        <= span_stop + timedelta(seconds=60)
    ]
    assert len(times) == len(inside) + 2 * ORBIT_TRIM_EXTRA_VECTORS
    assert times[ORBIT_TRIM_EXTRA_VECTORS] == inside[0]
    assert times[-ORBIT_TRIM_EXTRA_VECTORS - 1] == inside[-1]
    assert len(times) < len(orbit.vectors)


def test_incomplete_record_is_rejected() -> None:
    """A record missing a field fails closed instead of guessing."""
    record = radar_geometry_record(_grid(), _orbit())
    del record["range_spacing_m"]
    with pytest.raises(InvalidProcessingStateError, match="incomplete"):
        radar_geometry_from_record(record)


def test_trimmed_orbit_interpolates_like_the_full_orbit_inside_the_span() -> None:
    """Hermite interpolation is local, so trimming does not move the orbit."""
    import numpy as np

    from faninsar.processing.geometry.orbit import OrbitInterpolator

    orbit = _orbit()
    record = radar_geometry_record(_grid(), orbit)
    full = OrbitInterpolator.from_orbit(orbit)
    trimmed = radar_geometry_from_record(record).orbit
    span_s = 1500 * _grid().azimuth_time_interval_s
    for fraction in np.linspace(-1.0, 1.0, 41):
        when = START + timedelta(seconds=fraction * (span_s / 2 + 60.0) + span_s / 2)
        expected = full.evaluate(when)
        actual = trimmed.evaluate(when)
        np.testing.assert_allclose(actual.position_m, expected.position_m, atol=1e-6)
        np.testing.assert_allclose(
            actual.velocity_m_s, expected.velocity_m_s, atol=1e-9
        )

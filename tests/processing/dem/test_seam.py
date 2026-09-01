# ruff: noqa: D103, INP001
"""Automatic antimeridian source-window tests."""

from __future__ import annotations

import pytest

from faninsar.processing.dem.seam import (
    ExplicitAntimeridianError,
    SeamAwareSourceSampler,
    plan_query_windows,
)


def test_automatic_crossing_has_at_most_two_deterministic_windows() -> None:
    windows = plan_query_windows((170.0, -2.0, -170.0, 2.0))
    assert windows == ((170.0, -2.0, 180.0, 2.0), (-180.0, -2.0, -170.0, 2.0))


def test_explicit_crossing_fails_before_source_callback() -> None:
    with pytest.raises(ExplicitAntimeridianError):
        plan_query_windows((170.0, -2.0, -170.0, 2.0), explicit=True)


def test_logical_sampler_evaluates_each_target_once_across_seam() -> None:
    calls: list[tuple[float, float]] = []

    def sample(longitude: float, latitude: float) -> float:
        calls.append((longitude, latitude))
        return longitude

    sampler = SeamAwareSourceSampler(sample, target_center_longitude=180.0)
    values = sampler.sample([179.9, -179.9], [0.0, 0.0])
    assert values == [179.9, 180.1]
    assert len(calls) == 2

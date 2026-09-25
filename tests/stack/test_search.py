"""Focused tests for the generic Stack search seam."""

# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

from typing import Any

import pytest

from faninsar.stack import MissionSearchProfile, SelectionRequest, Stack


class _SearchStack(Stack):
    """Small mission adapter used to test generic dispatch."""

    @classmethod
    def _search_profile(cls) -> MissionSearchProfile:
        return MissionSearchProfile(
            mission="test",
            catalog="test-catalog",
            collections=("collection",),
            default_filters={"platform": "TEST"},
            selector=lambda items, request: list(items)[: request.limit],
        )


def test_stack_search_passes_profile_and_common_fields(monkeypatch: Any) -> None:
    """Profile defaults and common fields reach one remote search call."""
    calls: list[dict[str, object]] = []
    expected = object()

    def fake_search(spatial: object, **kwargs: object) -> list[object]:
        calls.append({"spatial": spatial, **kwargs})
        return [expected]

    monkeypatch.setattr("faninsar.remote.search", fake_search)
    result = _SearchStack.search(
        "roi",
        flight_direction="ASCENDING",
        filters={"beamMode": "IW"},
        limit=3,
    )

    assert result == [expected]
    assert calls == [
        {
            "spatial": "roi",
            "catalog": "test-catalog",
            "collections": ("collection",),
            "datetime_range": None,
            "flight_direction": "ASCENDING",
            "filters": {"platform": "TEST", "beamMode": "IW"},
            "auth_profile": None,
            "limit": 3,
            "budget": None,
        }
    ]


def test_selection_request_is_immutable() -> None:
    """Nested provider filters cannot be mutated after construction."""
    request = SelectionRequest(filters={"beamMode": "IW"})

    assert request.filters["beamMode"] == "IW"
    with pytest.raises(TypeError):
        request.filters["beamMode"] = "EW"  # type: ignore[index]

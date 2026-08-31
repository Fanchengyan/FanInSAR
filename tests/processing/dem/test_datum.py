"""Tests for the canonical geoid conversion graph."""

from __future__ import annotations

from faninsar.processing.dem.datum import conversion_models, fetch_required


def test_same_datum_is_a_noop() -> None:
    """No geoid is fetched when source and target already agree."""
    assert conversion_models("egm2008", "egm2008") == ()


def test_single_geoid_paths() -> None:
    """Each conversion to or from ellipsoidal uses one model."""
    assert conversion_models("egm2008", "ellipsoidal") == ("egm2008",)
    assert conversion_models("ellipsoidal", "egm96") == ("egm96",)


def test_cross_geoid_path_uses_both_in_order() -> None:
    """Cross-model conversion passes through ellipsoidal height."""
    assert conversion_models("egm96", "egm2008") == ("egm96", "egm2008")


def test_fetch_required_preserves_graph_order() -> None:
    """The conversion graph controls lazy fetch call order."""
    calls: list[str] = []
    result = fetch_required("egm96", "egm2008", lambda model: calls.append(model) or model)
    assert result == ("egm96", "egm2008")
    assert calls == ["egm96", "egm2008"]

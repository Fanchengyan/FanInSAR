"""Burst footprint plot labels (PROPOSAL-0084)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from shapely.geometry import Polygon

from faninsar.network.readers.scene_bursts import (
    SceneBurstCollection,
    SceneBurstObservation,
)
from faninsar.plotting.burst_footprints import plot_burst_footprints



def _poly() -> Polygon:
    return Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def _gdf(*, official: int | None = 12345, geometry: object = ...) -> object:
    import geopandas as gpd

    geom = _poly() if geometry is ... else geometry
    return gpd.GeoDataFrame(
        {
            "date": pd.Series(["20240101"], dtype="string"),
            "track": pd.Series(["88A"], dtype="string"),
            "burst": pd.Series(["f0_IW1_b0"], dtype="string"),
            "absolute_burst_id": pd.array([official], dtype="Int64"),
            "geometry": [geom],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _poly_at(offset: float) -> Polygon:
    return Polygon(
        [
            (offset, 0.0),
            (offset + 1.0, 0.0),
            (offset + 1.0, 1.0),
            (offset, 1.0),
        ]
    )


def _two_date_gdf(*, n_bursts: int = 5, shared_official: bool = False) -> object:
    """Two dates, matching burst tags, identical per-burst polygons."""
    import geopandas as gpd

    dates: list[str] = []
    tracks: list[str] = []
    bursts: list[str] = []
    ids: list[int | None] = []
    geoms: list[Polygon] = []
    for date in ("20240101", "20240113"):
        for index in range(n_bursts):
            dates.append(date)
            tracks.append("128D")
            bursts.append(f"f0_IW1_b{index}")
            if shared_official or date == "20240101":
                ids.append(12345 + index)
            else:
                ids.append(None)
            geoms.append(_poly_at(float(index) * 1.2))
    return gpd.GeoDataFrame(
        {
            "date": pd.Series(dates, dtype="string"),
            "track": pd.Series(tracks, dtype="string"),
            "burst": pd.Series(bursts, dtype="string"),
            "absolute_burst_id": pd.array(ids, dtype="Int64"),
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _burst_label_texts(ax: object) -> list[str]:
    return [
        str(text.get_text())
        for text in ax.texts
        if str(text.get_text()).startswith("Burst:")
    ]


def _burst_label_artists(ax: object) -> list[object]:
    return [
        text
        for text in ax.texts
        if str(text.get_text()).startswith("Burst:")
    ]


def _assert_identity_bboxes_disjoint(ax: object) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        artist.get_window_extent(renderer=renderer)
        for artist in _burst_label_artists(ax)
    ]
    for i, left in enumerate(boxes):
        for right in boxes[i + 1 :]:
            assert not left.overlaps(right)


def _texts(ax: object) -> str:
    parts = [str(text.get_text()) for text in ax.texts]
    parts.extend(
        str(child.get_text())
        for child in ax.get_children()
        if hasattr(child, "get_text")
    )
    return "\n".join(parts)


def test_collection_plot_returns_axes_with_exact_two_line_labels() -> None:
    """Plot labels are Burst: <tag> and Absolute burst ID: <int>."""
    ax = plot_burst_footprints(_gdf(official=12345))
    blob = _texts(ax)
    assert "Burst: f0_IW1_b0" in blob
    assert "Absolute burst ID: 12345" in blob
    assert "12345.0" not in blob
    assert "<NA>" not in blob


def test_individual_plot_renders_none_for_missing_official_id() -> None:
    """Missing official IDs render as the literal None."""
    ax = plot_burst_footprints(_gdf(official=None))
    blob = _texts(ax)
    assert "Burst: f0_IW1_b0" in blob
    assert "Absolute burst ID: None" in blob


def test_null_geometries_are_omitted_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Null geometry rows are omitted and warned, not drawn as rectangles."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {
            "date": pd.Series(["20240101", "20240113"], dtype="string"),
            "track": pd.Series(["88A", "88A"], dtype="string"),
            "burst": pd.Series(["f0_IW1_b0", "f0_IW1_b1"], dtype="string"),
            "absolute_burst_id": pd.array([1, 2], dtype="Int64"),
            "geometry": [_poly(), None],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    with caplog.at_level("WARNING"):
        ax = plot_burst_footprints(gdf)
    blob = _texts(ax)
    assert "Burst: f0_IW1_b0" in blob
    assert "Burst: f0_IW1_b1" not in blob
    assert any(
        "omit" in rec.message.lower() or "null" in rec.message.lower()
        for rec in caplog.records
    )


def test_empty_or_all_null_plot_raises_logged_valueerror(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty and all-null tables fail closed with a logged ValueError."""
    import geopandas as gpd

    empty = gpd.GeoDataFrame(
        {
            "date": pd.Series(dtype="string"),
            "track": pd.Series(dtype="string"),
            "burst": pd.Series(dtype="string"),
            "absolute_burst_id": pd.Series(dtype="Int64"),
            "geometry": gpd.GeoSeries(dtype="geometry"),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    with caplog.at_level("ERROR"), pytest.raises(ValueError, match="plottable"):
        plot_burst_footprints(empty)
    nulls = _gdf(geometry=None)
    with caplog.at_level("ERROR"), pytest.raises(ValueError, match="plottable"):
        plot_burst_footprints(nulls)


def test_plot_does_not_open_annotation_or_prepare_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection.plot uses resolved footprints and does not parse annotation."""
    from faninsar.missions.s1 import annotation
    from tests.network.test_scene_collection import _stack_with_scenes

    calls = {"n": 0}
    original = annotation.parse_annotation_xml

    def wrapped(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(annotation, "parse_annotation_xml", wrapped)
    stack = _stack_with_scenes(tmp_path)
    stack._sensor_registry_key = "sentinel1"
    bursts = stack.scenes.bursts
    monkeypatch.setattr(
        SceneBurstCollection,
        "_resolve_footprints",
        lambda self: _gdf(official=9),  # noqa: ARG005
    )
    ax = bursts.plot()
    assert calls["n"] == 0
    assert "Burst: f0_IW1_b0" in _texts(ax)


def test_observation_plot_delegates_to_same_labels() -> None:
    """An individual observation plots its own two-line label."""
    collection = SceneBurstCollection(stack=object(), observations=())
    item = SceneBurstObservation(
        date="20240101",
        burst="f0_IW1_b0",
        track="88A",
        absolute_burst_id=12345,
        _collection=collection,
    )
    collection._observations = (item,)
    collection._footprints_cache = _gdf(official=12345)
    ax = item.plot()
    blob = _texts(ax)
    assert "Burst: f0_IW1_b0" in blob
    assert "Absolute burst ID: 12345" in blob


def test_identical_identity_labels_are_not_repeated_per_date() -> None:
    """Same geometry and the same two-line annotation draw one display burst."""
    ax = plot_burst_footprints(_two_date_gdf(n_bursts=5, shared_official=True))
    labels = _burst_label_texts(ax)
    assert len(labels) == 5
    assert ax.get_legend() is None
    for index in range(5):
        assert any(f"Burst: f0_IW1_b{index}" in text for text in labels)
    for text in labels:
        assert "Absolute burst ID:" in text
        assert "\n" in text
        assert "20240101" not in text
        assert "20240113" not in text
    _assert_identity_bboxes_disjoint(ax)
    dest = (
        Path(__file__).resolve().parents[2]
        / "tmp"
        / "p0084"
        / "synthetic_two_dates_repair2.png"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(dest, dpi=120, bbox_inches="tight")


def test_differing_official_ids_keep_separate_identity_labels() -> None:
    """Differing official IDs are not collapsed even when geometry matches."""
    ax = plot_burst_footprints(_two_date_gdf(n_bursts=5, shared_official=False))
    labels = _burst_label_texts(ax)
    assert len(labels) == 10
    assert ax.get_legend() is None
    blob = "\n".join(labels)
    assert "Absolute burst ID: 12345" in blob
    assert "Absolute burst ID: None" in blob
    _assert_identity_bboxes_disjoint(ax)

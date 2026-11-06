"""Stage-5 sel() surface: track/direction/pol/burst + polygon→track rules.

Exercises the mission-owned track metadata resolver
(:mod:`faninsar.missions.s1.track_map`), the Stack-side selector plumbing
(:meth:`Stack.track_groups`/:meth:`Stack.polygon_tracks`/
``_catalog_dates_for_track``), the collection ``sel()`` dims on
:class:`StackInterferogramCollection` and :class:`StackSceneCollection`,
and the cross-track pairing constraint in :meth:`S1Stack._coregister_one_scene`.
"""

from __future__ import annotations

from tests.support.local_stack import LocalStack

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from shapely.geometry import box

from faninsar.core import Pairs
from faninsar.missions.s1.stack import S1Stack
from faninsar.missions.s1.track_map import (
    TrackAcquisitionInfo,
    disambiguate_track_letters,
    polygon_track_assignment,
    resolve_track_acquisition,
    select_dates,
)
from faninsar.network.readers.stack import StackInterferogramCollection
from faninsar.stack import StackConfig
from faninsar.stack.catalog import SceneCatalog
from tests.network.test_stack_reader_isolation import _write_pair_product

if TYPE_CHECKING:
    from faninsar.stack.session import Stack

# ---------------------------------------------------------------------------
# track_map unit coverage
# ---------------------------------------------------------------------------


def _info(
    track: str | None,
    *,
    direction: str | None = None,
    orbit: int | None = None,
    mission: str = "S1A",
    pols: tuple[str, ...] = ("VV",),
    absids: frozenset[int] = frozenset(),
) -> TrackAcquisitionInfo:
    return TrackAcquisitionInfo(
        track_key=track,
        direction=direction,
        relative_orbit=orbit,
        mission_id=mission,
        polarizations=pols,
        absolute_burst_ids=absids,
    )


def test_track_key_from_tracks_layout(tmp_path: Path) -> None:
    """A staged path under tracks/<key>/acquisitions resolves its key."""
    safe = (
        tmp_path / "tracks" / "88A" / "acquisitions"
        / "S1A_IW_SLC__1SDV_20240101T000000.SAFE"
    )
    safe.mkdir(parents=True)
    info = resolve_track_acquisition([safe], open_products=False)
    assert info.relative_orbit == 88
    assert info.direction == "ASCENDING"
    assert info.track_key == "88A"


def test_track_key_from_safe_name_field(tmp_path: Path) -> None:
    """The path-SAFE product id's ``_<orbit><letter>_`` field resolves."""
    safe = (
        tmp_path / "anything"
        / "S1A_IW_SLC__1SDV_20240101T000000_20240101T000100_00088A_000000_ABCD.SAFE"
    )
    safe.mkdir(parents=True)
    info = resolve_track_acquisition([safe], open_products=False)
    assert info.relative_orbit == 88
    assert info.direction == "ASCENDING"
    assert info.track_key == "88A"


def test_track_key_unresolvable_without_layout_or_name(tmp_path: Path) -> None:
    """A bare SAFE directory carries no orbit-group identity."""
    safe = tmp_path / "S1A_IW_SLC__1SDV_20240101T000000.SAFE"
    safe.mkdir()
    info = resolve_track_acquisition([safe], open_products=False)
    assert info.relative_orbit is None
    assert info.direction is None
    assert info.track_key is None


def test_select_dates_by_track_direction_pol() -> None:
    """select_dates composes track, direction, and pol correctly."""
    infos = {
        "d1": _info("88A", direction="ASCENDING", orbit=88, pols=("VV", "VH")),
        "d2": _info("88D", direction="DESCENDING", orbit=88),
        "d3": _info("12A", direction="ASCENDING", orbit=12),
    }
    assert select_dates(infos, track="88A") == ("d1",)
    assert select_dates(infos, track="88") == ("d1", "d2")
    assert select_dates(infos, direction="ASC") == ("d1", "d3")
    assert select_dates(infos, direction="DESCENDING") == ("d2",)
    assert select_dates(infos, pol="VH") == ("d1",)
    assert select_dates(infos, pol=["VV", "VH"]) == ("d1",)


def test_select_dates_rejects_bad_pol() -> None:
    """pol= outside the Sentinel-1 channel set fails."""
    infos = {"d1": _info("88A", direction="ASCENDING", orbit=88)}
    with pytest.raises(ValueError, match="polarization"):
        select_dates(infos, pol="XX")


def test_disambiguate_track_letters_single_direction() -> None:
    """A ``?``-letter track borrows the catalog's only same-orbit direction."""
    infos = {
        "s1a": _info("88A", direction="ASCENDING", orbit=88, mission="S1A"),
        "s1c": _info("88?", direction=None, orbit=88, mission="S1C"),
    }
    resolved = disambiguate_track_letters(infos)
    assert resolved["s1c"].track_key == "88A"
    assert resolved["s1c"].direction == "ASCENDING"
    assert resolved["s1a"] is infos["s1a"]


def test_disambiguate_track_letters_mixed_direction_keeps_q() -> None:
    """Same-orbit catalogs mixing A and D never guess a direction."""
    infos = {
        "a": _info("88A", direction="ASCENDING", orbit=88),
        "d": _info("88D", direction="DESCENDING", orbit=88),
        "q": _info("88?", direction=None, orbit=88, mission="S1C"),
    }
    resolved = disambiguate_track_letters(infos)
    assert resolved["q"].track_key == "88?"
    assert resolved["q"].direction is None


def test_polygon_track_assignment_picks_covering_track() -> None:
    """Each polygon lands on the track whose footprints cover it best."""
    footprints = {
        "d1": (box(0.0, 0.0, 2.0, 2.0),),
        "d2": (box(10.0, 10.0, 12.0, 12.0),),
    }
    date_track = {"d1": "88A", "d2": "12D"}
    polygons = [box(0.5, 0.5, 1.0, 1.0), box(10.5, 10.5, 11.0, 11.0)]
    assert polygon_track_assignment(polygons, footprints, date_track) == {
        0: "88A",
        1: "12D",
    }


# ---------------------------------------------------------------------------
# Stack-level plumbing with injected track metadata
# ---------------------------------------------------------------------------

_TRACK_DATES = ("20240101", "20240113", "20240125")


def _tracked_stack(tmp_path: Path) -> S1Stack:
    """Build a planned S1Stack whose catalog carries injected track metadata.

    Two dates share track ``88A`` and one sits on ``12D`` so every
    selector sees both an on-track and a cross-track member; a third pair
    crosses the tracks to prove the single-track rule drops it. The
    persisted ``track_metadata`` extra makes the mission hook decode the
    recorded map without opening the (empty) SAFE directories.
    """
    paths = []
    for date_id in _TRACK_DATES:
        safe = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        safe.mkdir()
        paths.append(safe)
    stack = S1Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(work_dir=tmp_path / "work", multilook=(1, 1)),
        pairs=Pairs.from_names(
            ["20240101_20240113", "20240113_20240125", "20240101_20240125"]
        ),
        misreg_pairs=Pairs.from_names(["20240101_20240113"]),
        reference="20240101",
    )
    stack.config.extra["track_metadata"] = {
        "dates": {
            "20240101": {
                "track_key": "88A",
                "direction": "ASCENDING",
                "relative_orbit": 88,
                "mission_id": "S1A",
                "polarizations": ("VV",),
                "absolute_burst_ids": (3001, 3002),
            },
            "20240113": {
                "track_key": "88A",
                "direction": "ASCENDING",
                "relative_orbit": 88,
                "mission_id": "S1A",
                "polarizations": ("VV",),
                "absolute_burst_ids": (3001, 3002),
            },
            "20240125": {
                "track_key": "12D",
                "direction": "DESCENDING",
                "relative_orbit": 12,
                "mission_id": "S1A",
                "polarizations": ("VV",),
                "absolute_burst_ids": (5001,),
            },
        }
    }
    return stack


def test_stack_track_groups_property(tmp_path: Path) -> None:
    """track_groups enumerates the distinct labels of the catalog."""
    stack = _tracked_stack(tmp_path)
    assert stack.track_groups == ("12D", "88A")


def test_catalog_dates_for_track(tmp_path: Path) -> None:
    """_catalog_dates_for_track honours track/direction/pol selectors."""
    stack = _tracked_stack(tmp_path)
    assert stack._catalog_dates_for_track(track="88A") == (
        "20240101",
        "20240113",
    )
    assert stack._catalog_dates_for_track(direction="DESCENDING") == ("20240125",)
    assert stack._catalog_dates_for_track(pol="VV") == _TRACK_DATES
    assert stack._catalog_dates_for_track(pol="VH") == ()


def test_select_pairs_for_track_drops_cross_track(tmp_path: Path) -> None:
    """The single-track rule keeps only pairs with both endpoints on-track."""
    stack = _tracked_stack(tmp_path)
    dates = stack._catalog_dates_for_track(track="88A")
    selected = stack._select_pairs_for_track(dates)
    assert selected.to_names().tolist() == ["20240101_20240113"]


# ---------------------------------------------------------------------------
# Collection sel() coverage
# ---------------------------------------------------------------------------


def _tracked_collection(tmp_path: Path) -> tuple[Stack, StackInterferogramCollection]:
    stack = _tracked_stack(tmp_path)
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    for pair_id in ("20240101_20240113", "20240113_20240125", "20240101_20240125"):
        _write_pair_product(stack, pair_id, phase=phase, looks=(1, 1))
    return stack, StackInterferogramCollection(stack)


def test_sel_track_narrows_pairs(tmp_path: Path) -> None:
    """sel(track=) keeps only pairs whose endpoints share the track."""
    _stack, collection = _tracked_collection(tmp_path)
    view = collection.sel(track="88A")
    assert view.pairs.to_names().tolist() == ["20240101_20240113"]
    # The cross-track pair is invisible.
    with pytest.raises(KeyError):
        view.open("20240113_20240125", "complex")
    np.testing.assert_allclose(
        view.open("20240101_20240113", "complex").values,
        np.exp(1j * np.full((3, 4), 0.2)).astype(np.complex64),
    )


def test_sel_track_numeric_orbit(tmp_path: Path) -> None:
    """A bare orbit number resolves to its only direction letter."""
    _stack, collection = _tracked_collection(tmp_path)
    view = collection.sel(track="88")
    assert view.pairs.to_names().tolist() == ["20240101_20240113"]


def test_sel_track_unknown_rejected(tmp_path: Path) -> None:
    """A track outside the catalog's set fails with the known list."""
    _stack, collection = _tracked_collection(tmp_path)
    with pytest.raises(ValueError, match="tracks"):
        collection.sel(track="99A")


def test_sel_direction_narrows_pairs(tmp_path: Path) -> None:
    """sel(direction=) keeps same-direction pairs only."""
    _stack, collection = _tracked_collection(tmp_path)
    view = collection.sel(direction="ASC")
    assert view.pairs.to_names().tolist() == ["20240101_20240113"]


def test_sel_pol_narrows_pairs(tmp_path: Path) -> None:
    """sel(pol=) drops pairs whose dates lack the channel."""
    stack = _tracked_stack(tmp_path)
    # Give two dates an extra channel; only the pair they form survives.
    meta = stack.config.extra["track_metadata"]["dates"]
    meta["20240113"]["polarizations"] = ("VV", "VH")
    meta["20240125"]["polarizations"] = ("VV", "VH")
    stack._track_metadata_cache = None
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    for pair_id in ("20240101_20240113", "20240113_20240125", "20240101_20240125"):
        _write_pair_product(stack, pair_id, phase=phase, looks=(1, 1))
    collection = StackInterferogramCollection(stack)
    view = collection.sel(pol="VH")
    assert view.pairs.to_names().tolist() == ["20240113_20240125"]


def test_sel_pol_no_pairs_raises(tmp_path: Path) -> None:
    """sel(pol=) on a channel no pair fully covers fails closed."""
    stack = _tracked_stack(tmp_path)
    # Only one date carries VH — no planned pair can satisfy pol="VH".
    meta = stack.config.extra["track_metadata"]["dates"]
    meta["20240125"]["polarizations"] = ("VV", "VH")
    stack._track_metadata_cache = None
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    _write_pair_product(stack, "20240113_20240125", phase=phase, looks=(1, 1))
    collection = StackInterferogramCollection(stack)
    with pytest.raises(ValueError, match="selects no pairs"):
        collection.sel(pol="VH")


def test_sel_pol_invalid(tmp_path: Path) -> None:
    """pol= outside the Sentinel-1 channel set fails."""
    _stack, collection = _tracked_collection(tmp_path)
    with pytest.raises(ValueError, match="pol"):
        collection.sel(pol="XX")


def test_sel_burst_narrows_pairs(tmp_path: Path) -> None:
    """sel(burst=) keeps only pairs whose dates cover the id."""
    _stack, collection = _tracked_collection(tmp_path)
    view = collection.sel(burst=3001)
    assert view.pairs.to_names().tolist() == ["20240101_20240113"]
    with pytest.raises(KeyError):
        view.open("20240113_20240125", "complex")


def test_sel_burst_unknown_id(tmp_path: Path) -> None:
    """sel(burst=) on an absent absolute id fails naming it."""
    _stack, collection = _tracked_collection(tmp_path)
    with pytest.raises(ValueError, match="absolute burst"):
        collection.sel(burst=9999)


def test_sel_burst_no_metadata_fails_closed(tmp_path: Path) -> None:
    """Without absolute-burst metadata, sel(burst=) refuses to guess."""
    stack = _tracked_stack(tmp_path)
    for record in stack.config.extra["track_metadata"]["dates"].values():
        record["absolute_burst_ids"] = ()
    stack._track_metadata_cache = None
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    _write_pair_product(stack, "20240101_20240113", phase=phase, looks=(1, 1))
    collection = StackInterferogramCollection(stack)
    with pytest.raises(ValueError, match="absolute-burst"):
        collection.sel(burst=3001)


def test_sel_pairs_cannot_smuggle_cross_track(tmp_path: Path) -> None:
    """sel(track=).sel(pairs=) cannot widen past the track selection."""
    _stack, collection = _tracked_collection(tmp_path)
    view = collection.sel(track="88A")
    with pytest.raises(ValueError, match="selected network"):
        view.sel(pairs=Pairs.from_names(["20240113_20240125"]))


def test_sel_polygon_wrong_track_names_correct_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sel(polygon=N) on the wrong track errors naming the right one."""
    stack = _tracked_stack(tmp_path)
    monkeypatch.setattr(type(stack), "roi_polygon_count", property(lambda _s: 2))
    monkeypatch.setattr(
        type(stack), "polygon_tracks", property(lambda _s: {0: "88A", 1: "12D"})
    )
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    _write_pair_product(stack, "20240101_20240113", phase=phase, looks=(1, 1))
    collection = StackInterferogramCollection(stack)
    view = collection.sel(track="88A")
    with pytest.raises(ValueError, match="12D"):
        view.sel(polygon=1)


def test_sel_polygon_on_track_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sel(polygon=N) on the matching track returns the subtree view."""
    stack = _tracked_stack(tmp_path)
    monkeypatch.setattr(type(stack), "roi_polygon_count", property(lambda _s: 2))
    monkeypatch.setattr(
        type(stack), "polygon_tracks", property(lambda _s: {0: "88A", 1: "12D"})
    )
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    _write_pair_product(stack, "20240101_20240113", phase=phase, looks=(1, 1))
    collection = StackInterferogramCollection(stack)
    view = collection.sel(track="88A").sel(polygon=0)
    assert view._selection.polygon == 0


# ---------------------------------------------------------------------------
# StackSceneCollection sel()
# ---------------------------------------------------------------------------


def test_scenes_sel_track(tmp_path: Path) -> None:
    """scenes.sel(track=) keeps committed dates on the selected track."""
    stack = _tracked_stack(tmp_path)
    # Commit two scenes so the dict has members to filter.
    from faninsar.io.storage.scene_store import write_scene

    for date_id in ("20240101", "20240113", "20240125"):
        write_scene(
            stack.config.work_dir / "coreg" / date_id,
            date_id=date_id,
            reference_id="20240101",
            domain="radar",
            grid_shape=(2, 2),
            units=[
                {
                    "tag": "f0_IW1_b0",
                    "payload": np.ones((2, 2), dtype=np.complex64),
                    "row_origin": 0,
                    "col_origin": 0,
                }
            ],
        )
    scenes = stack.scenes.sel(track="88A")
    assert scenes.dates == ["20240101", "20240113"]


# ---------------------------------------------------------------------------
# Cross-track pairing constraint
# ---------------------------------------------------------------------------


def test_coregister_rejects_cross_track_secondary(tmp_path: Path) -> None:
    """_coregister_one_scene fails closed across two tracks."""
    stack = _tracked_stack(tmp_path)
    stack._prepared = True
    with pytest.raises(Exception, match="different tracks"):
        stack._coregister_one_scene(
            "20240125",
            output_dir=tmp_path / "work" / "coreg" / "20240125",
            misreg_az_px=0.0,
            misreg_rg_px=0.0,
            overwrite=True,
        )


def test_reference_scene_polygon_wrong_track_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_prepare_reference_scene on an off-track polygon reports no units."""
    stack = _tracked_stack(tmp_path)
    stack._prepared = True
    monkeypatch.setattr(
        type(stack), "polygon_tracks", property(lambda _s: {0: "88A", 1: "12D"})
    )
    monkeypatch.setattr(
        stack, "_roi_polygon_rois", lambda: [box(0, 0, 1, 1), box(2, 2, 3, 3)]
    )
    with pytest.raises(Exception, match="no scene units"):
        stack._prepare_reference_scene(overwrite=False, polygon=1)


# ---------------------------------------------------------------------------
# track_map record decoding
# ---------------------------------------------------------------------------


def test_track_metadata_round_trip_via_extra(tmp_path: Path) -> None:
    """Persisted ``track_metadata`` extra decodes into TrackAcquisitionInfo."""
    stack = _tracked_stack(tmp_path)
    infos = stack._catalog_track_infos()
    assert infos["20240101"].track_key == "88A"
    assert infos["20240125"].absolute_burst_ids == frozenset({5001})
    assert infos["20240113"].direction == "ASCENDING"


def test_s1stack_date_track_key(tmp_path: Path) -> None:
    """S1Stack._date_track_key reads the persisted per-date label."""
    stack = _tracked_stack(tmp_path)
    assert stack._date_track_key("20240101") == "88A"
    assert stack._date_track_key("20240125") == "12D"

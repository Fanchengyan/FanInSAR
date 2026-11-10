"""CDSE rediscovery must use ASF parent SAFE identity, not child burst start.

ASF SLC-BURST hrefs encode the parent frame (``.../<SAFE>/IW1/VV/8.zip``).
``remote.search`` keeps items by ``acquisition.start`` only, so a window
opened at the child burst start drops the parent.  These tests drive the
actual selectors, search postfilter, stack rediscovery, and burst-index
resolver — not a reimplemented predicate.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from shapely.geometry import box, mapping

from faninsar import remote
from faninsar.data.query import BoundingBox
from faninsar.missions.s1.acquisition import (
    _apply_asf_burst_hints,
    _asf_burst_groups,
    _resolve_burst_indices,
    asf_burst_selectors,
    asf_frame_selectors,
    materialize_cdse_burst_safes,
)
from faninsar.missions.s1.s1search import S1Search
from faninsar.missions.s1.stack import S1Stack
from faninsar.remote.catalog import _ADAPTERS

ROI = BoundingBox(100.0, 40.0, 100.4, 40.4, crs="EPSG:4326")
FOOTPRINT = mapping(box(100.0, 40.0, 100.4, 40.4))

PARENT = "S1B_IW_SLC__1SDV_20200110T114920_20200110T114950_019754_02559E_5E5F"
PARENT_START = datetime(2020, 1, 10, 11, 49, 20, tzinfo=UTC)
PARENT_STOP = datetime(2020, 1, 10, 11, 49, 50, tzinfo=UTC)
# Parent IW1 burstList index 8 sensingTime; matches ASF acquisition.start.
BURST8_START = datetime(2020, 1, 10, 11, 49, 44, 39352, tzinfo=UTC)

NEIGHBOR = "S1B_IW_SLC__1SDV_20200110T114945_20200110T115015_019754_02559E_AAAA"
NEIGHBOR_START = datetime(2020, 1, 10, 11, 49, 45, tzinfo=UTC)

PREVIOUS = "S1B_IW_SLC__1SDV_20200110T114855_20200110T114925_019754_02559E_BBBB"
PREVIOUS_START = datetime(2020, 1, 10, 11, 48, 55, tzinfo=UTC)

FRAME_B = "S1B_IW_SLC__1SDV_20200110T114920_20200110T114950_019754_02559E_FAAA"
DATE2 = "S1B_IW_SLC__1SDV_20200122T114918_20200122T114948_019929_025ABC_1111"
DATE2_STOP = datetime(2020, 1, 22, 11, 49, 48, tzinfo=UTC)
DATE2_BURST = datetime(2020, 1, 22, 11, 49, 42, tzinfo=UTC)


def _asset(href: str, item_id: str = "burst") -> remote.RemoteAsset:
    return remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="SLC-BURST",
        item_id=item_id,
        key="data",
        href=href,
        media_type="application/zip",
        representation="complete",
    )


def _item(
    href: str,
    *,
    item_id: str,
    start: datetime | None,
    end: datetime | None = None,
    properties: dict[str, object] | None = None,
    collection: str = "SLC-BURST",
) -> remote.CatalogItem:
    acquisition = remote.AcquisitionMetadata(
        acquisition_id=item_id,
        start=start,
        end=end if end is not None else start,
        geometry=FOOTPRINT,
        properties=properties or {},
    )
    return remote.CatalogItem(
        provider="ASF",
        catalog="sentinel1-asf",
        collection=collection,
        item_id=item_id,
        acquisition=acquisition,
        assets={"data": _asset(href, item_id)},
        raw_metadata={},
    )


def _burst_item(
    *,
    parent: str = PARENT,
    swath: str = "IW1",
    pol: str = "VV",
    ordinal: int = 8,
    start: datetime = BURST8_START,
    absolute: int = 159_008,
    relative_burst: int = 8,
    mission: str = "S1B",
    sensing_start: str | None = None,
    item_id: str | None = None,
) -> remote.CatalogItem:
    href = (
        f"https://sentinel1-burst.asf.alaska.edu/{parent}/{swath}/{pol}/{ordinal}.zip"
    )
    return _item(
        href,
        item_id=item_id or f"{parent}-{swath}-{ordinal}",
        start=start,
        end=start + timedelta(seconds=3),
        properties={
            "absoluteBurstID": absolute,
            "relativeBurstID": relative_burst,
            "relativeOrbit": 88,
            "orbitDirection": "ASCENDING",
            "mission": mission,
            "sensingStart": sensing_start,
        },
    )


def _frame_item(
    product_id: str = FRAME_B,
    *,
    start: datetime = PARENT_START,
    end: datetime = PARENT_STOP,
) -> remote.CatalogItem:
    return _item(
        f"https://datapool.asf.alaska.edu/SLC/SB/{product_id}.zip",
        item_id=product_id,
        start=start,
        end=end,
        properties={
            "relativeOrbit": 88,
            "orbitDirection": "ASCENDING",
            "mission": "S1B",
        },
        collection="S1-SLC",
    )


def _parent_swath() -> SimpleNamespace:
    """Ten-burst parent whose index 8 sensingTime is the ASF child start."""
    t0 = PARENT_START
    bursts = []
    for index in range(10):
        sensing = BURST8_START if index == 8 else t0 + timedelta(seconds=3 * index)
        bursts.append(SimpleNamespace(index=index, sensing_time=sensing))
    return SimpleNamespace(
        bursts=tuple(bursts),
        lines_per_burst=1500,
        azimuth_time_interval_s=0.0020555563,
    )


def _cdse_binding(
    product_id: str,
    *,
    orbit: int = 88,
    swath: str = "IW1",
    polarization: str = "VV",
) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "relative_orbit": orbit,
        "orbit_direction": "ASCENDING",
        "mission_id": "S1B",
        "polarization": polarization,
        "swath": swath,
        "measurement_url": (
            f"https://catalogue.dataspace.copernicus.eu/{product_id}/{swath}/{polarization}"
        ),
    }


def _asf_search(items: list[remote.CatalogItem]) -> S1Search:
    return S1Search(
        items=tuple(items),
        catalog="sentinel1-asf",
        spatial=ROI,
        query={},
    )


class _Records:
    provider = "cdse-bridge-fixture"
    origins = ("https://fixture.invalid",)
    path_prefixes = ("/",)
    redirect_origins = ()
    profiles = ("anonymous",)

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def items(self, **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return list(self.records)


def _cdse_record(product_id: str, start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "id": product_id,
        "geometry": FOOTPRINT,
        "acquisition": {"id": product_id, "start": start, "end": end},
        "assets": {},
    }


def test_burst_href_window_covers_parent_start_not_child_burst() -> None:
    """SLC-BURST child start 11:49:44 must not become the rediscovery window."""
    result = asf_frame_selectors([_burst_item()])
    window = result[0]
    assert window is not None
    assert window[0] == PARENT_START
    assert window[0] != BURST8_START
    assert window[1] == PARENT_STOP
    assert "88A" in result[1]
    parents = result[2] if len(result) > 2 else None
    assert parents is not None
    assert PARENT in parents


def test_remote_search_postfilter_keeps_parent_drops_previous_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual start-only postfilter keeps parent 11:49:20; drops 11:48:55."""
    adapter = _Records(
        [
            _cdse_record(
                PREVIOUS, PREVIOUS_START, PREVIOUS_START + timedelta(seconds=30)
            ),
            _cdse_record(PARENT, PARENT_START, PARENT_STOP),
            _cdse_record(
                NEIGHBOR, NEIGHBOR_START, NEIGHBOR_START + timedelta(seconds=30)
            ),
        ]
    )
    monkeypatch.setitem(_ADAPTERS, "cdse-bridge-fixture", adapter)
    result = asf_frame_selectors([_burst_item()])
    window = result[0]
    parents = result[2] if len(result) > 2 else None
    hits = remote.search(ROI, catalog="cdse-bridge-fixture", datetime_range=window)
    ids = [item.item_id for item in hits]
    assert PARENT in ids
    assert PREVIOUS not in ids
    # Overlapping next frame starts inside the parent window; id filter must
    # drop it later.  The postfilter itself correctly keeps it.
    assert NEIGHBOR in ids
    assert parents is not None
    assert NEIGHBOR not in parents
    assert PARENT in parents


def test_known_parent_filter_drops_same_track_neighbor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adjacent same-track CDSE frame must not leak into materialization."""
    captured: dict[str, Any] = {}

    def fake_discover(roi, datetime_range=None, polarizations=None, **kw):
        captured["datetime_range"] = datetime_range
        return [
            _cdse_binding(PARENT),
            _cdse_binding(PARENT, swath="IW2"),
            _cdse_binding(NEIGHBOR),
            _cdse_binding("OTHER_TRACK", orbit=117),
        ]

    def fake_materialize(roi, **kwargs):
        captured["items"] = kwargs.get("items")
        captured["absolute_burst_ids"] = kwargs.get("absolute_burst_ids")
        captured["catalog_sensing_times"] = kwargs.get("catalog_sensing_times")
        return [tmp_path / "X.SAFE"]

    monkeypatch.setattr(
        "faninsar.missions.s1.stack.discover_cdse_slc_items", fake_discover
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.stack.materialize_s1_acquisitions", fake_materialize
    )
    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(lambda _cls, _paths, **_kw: SimpleNamespace(prepare=lambda: None)),
    )
    S1Stack._materialize_from_search(
        _asf_search([_burst_item(), _burst_item(pol="VH", item_id="vh")]),
        work_dir=tmp_path,
        provider="cdse",
        granularity="burst",
        polarizations=("VV",),
        coreg_mode="geometry",
        orbit_group=None,
        orbit_paths=None,
        aux_cal_paths=None,
        effective_roi=ROI,
        kwargs={"extra": {"orbit_auto_fetch": False}},
    )
    assert captured["datetime_range"] == (PARENT_START, PARENT_STOP)
    assert [(b["product_id"], b["swath"]) for b in captured["items"]] == [
        (PARENT, "IW1")
    ]
    assert captured["absolute_burst_ids"] is None
    assert captured.get("catalog_sensing_times") is None
    assert captured["items"][0]["_asf_burst_hints"] == [(159_008, BURST8_START)]


def test_mixed_frame_and_burst_keeps_explicit_frame_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Known-parent filter must not drop an explicit whole-frame selection."""
    captured: dict[str, Any] = {}

    def fake_discover(roi, datetime_range=None, polarizations=None, **kw):
        return [
            _cdse_binding(PARENT),
            _cdse_binding(PARENT, swath="IW2"),
            _cdse_binding(FRAME_B),
            _cdse_binding(FRAME_B, swath="IW2"),
            _cdse_binding(NEIGHBOR),
        ]

    def fake_materialize(roi, **kwargs):
        captured["items"] = kwargs.get("items")
        captured["absolute_burst_ids"] = kwargs.get("absolute_burst_ids")
        return [tmp_path / "X.SAFE"]

    monkeypatch.setattr(
        "faninsar.missions.s1.stack.discover_cdse_slc_items", fake_discover
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.stack.materialize_s1_acquisitions", fake_materialize
    )
    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(lambda _cls, _paths, **_kw: SimpleNamespace(prepare=lambda: None)),
    )
    S1Stack._materialize_from_search(
        _asf_search([_burst_item(), _frame_item()]),
        work_dir=tmp_path,
        provider="cdse",
        granularity="burst",
        polarizations=("VV",),
        coreg_mode="geometry",
        orbit_group=None,
        orbit_paths=None,
        aux_cal_paths=None,
        effective_roi=ROI,
        kwargs={"extra": {"orbit_auto_fetch": False}},
    )
    keys = [(b["product_id"], b["swath"]) for b in captured["items"]]
    assert (PARENT, "IW1") in keys
    assert (PARENT, "IW2") not in keys
    assert (FRAME_B, "IW1") in keys
    assert (FRAME_B, "IW2") in keys
    assert NEIGHBOR not in [b["product_id"] for b in captured["items"]]
    assert captured["absolute_burst_ids"] is None
    hinted = next(b for b in captured["items"] if b["product_id"] == PARENT)
    assert hinted["_asf_burst_hints"] == [(159_008, BURST8_START)]
    for binding in captured["items"]:
        if binding["product_id"] == FRAME_B:
            assert "_asf_burst_hints" not in binding


def test_frame_item_uses_catalog_acquisition_window() -> None:
    result = asf_frame_selectors([_frame_item()])
    assert result[0] == (PARENT_START, PARENT_STOP)
    parents = result[2] if len(result) > 2 else None
    assert parents is not None
    assert FRAME_B in parents


@pytest.mark.parametrize(
    "href,item_id",
    [
        (
            "https://sentinel1-burst.asf.alaska.edu/opaque-token.zip",
            "S1_159008_IW1_VV_8-BURST",
        ),
        (
            "https://sentinel1-burst.asf.alaska.edu/not-a-safe/IW1/VV/8.zip",
            "orphan-burst",
        ),
    ],
)
def test_unknown_burst_parent_raises(href: str, item_id: str) -> None:
    item = _item(
        href,
        item_id=item_id,
        start=BURST8_START,
        collection="SLC-BURST",
        properties={
            "absoluteBurstID": 1,
            "relativeOrbit": 88,
            "orbitDirection": "ASCENDING",
            "mission": "S1B",
        },
    )
    with pytest.raises(ValueError, match="parent"):
        asf_frame_selectors([item])


def test_sensing_start_precedes_acquisition_start_fallback() -> None:
    hinted = datetime(2020, 1, 10, 11, 49, 44, tzinfo=UTC)
    item = _burst_item(sensing_start=hinted.isoformat())
    item = replace(
        item,
        acquisition=replace(
            item.acquisition,
            start=datetime(2020, 1, 10, 11, 49, 40, tzinfo=UTC),
        ),
    )
    ids, times = asf_burst_selectors([item])
    assert ids == [159_008]
    assert times == [hinted]


def test_sensing_start_none_falls_back_to_acquisition_start() -> None:
    ids, times = asf_burst_selectors([_burst_item(sensing_start=None)])
    assert ids == [159_008]
    assert times == [BURST8_START]


def test_shuffled_multi_swath_id_time_pairs_stay_aligned() -> None:
    iw2 = _burst_item(
        swath="IW2",
        ordinal=3,
        start=datetime(2020, 1, 10, 11, 49, 30, tzinfo=UTC),
        absolute=159_203,
        relative_burst=3,
        item_id="iw2",
    )
    iw1 = _burst_item()
    ids, times = asf_burst_selectors([iw2, iw1])
    assert list(zip(ids, times, strict=True)) == [
        (159_203, iw2.acquisition.start),
        (159_008, BURST8_START),
    ]


def test_parent_index_eight_resolves_by_time_not_local_zero() -> None:
    """Extracted canvas slot 0 is not parent burstList index 8."""
    swath = _parent_swath()
    ids, times = asf_burst_selectors([_burst_item()])
    resolved = _resolve_burst_indices(
        swath,
        absolute_burst_ids=ids,
        catalog_sensing_times=times,
    )
    assert resolved == (8,)
    assert resolved != (0,)


def test_same_relative_burst_id_across_dates_keeps_both_parents() -> None:
    first = _burst_item()
    second = _burst_item(
        parent=DATE2,
        start=DATE2_BURST,
        absolute=160_008,
        item_id="date2",
    )
    result = asf_frame_selectors([first, second])
    assert result[0] == (PARENT_START, DATE2_STOP)
    parents = result[2] if len(result) > 2 else None
    assert parents == frozenset({PARENT, DATE2})
    ids, times = asf_burst_selectors([first, second])
    assert ids == [159_008, 160_008]
    assert times == [BURST8_START, DATE2_BURST]


def test_identified_burst_without_sensing_time_raises() -> None:
    item = _burst_item()
    item = replace(item, acquisition=replace(item.acquisition, start=None, end=None))
    with pytest.raises(ValueError, match="sensing"):
        asf_burst_selectors([item])


def test_same_parent_frame_and_burst_keeps_full_swath_unhinted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit whole-frame union must not be narrowed by a same-parent burst."""
    captured: dict[str, Any] = {}

    def fake_discover(roi, datetime_range=None, polarizations=None, **kw):
        return [
            _cdse_binding(PARENT),
            _cdse_binding(PARENT, swath="IW2"),
            _cdse_binding(NEIGHBOR),
        ]

    def fake_materialize(roi, **kwargs):
        captured["items"] = kwargs.get("items")
        captured["absolute_burst_ids"] = kwargs.get("absolute_burst_ids")
        return [tmp_path / "X.SAFE"]

    monkeypatch.setattr(
        "faninsar.missions.s1.stack.discover_cdse_slc_items", fake_discover
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.stack.materialize_s1_acquisitions", fake_materialize
    )
    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(lambda _cls, _paths, **_kw: SimpleNamespace(prepare=lambda: None)),
    )
    S1Stack._materialize_from_search(
        _asf_search([_burst_item(), _frame_item(PARENT)]),
        work_dir=tmp_path,
        provider="cdse",
        granularity="burst",
        polarizations=("VV",),
        coreg_mode="geometry",
        orbit_group=None,
        orbit_paths=None,
        aux_cal_paths=None,
        effective_roi=ROI,
        kwargs={"extra": {"orbit_auto_fetch": False}},
    )
    keys = [(b["product_id"], b["swath"]) for b in captured["items"]]
    assert keys == [(PARENT, "IW1"), (PARENT, "IW2")]
    assert captured["absolute_burst_ids"] is None
    for binding in captured["items"]:
        assert "_asf_burst_hints" not in binding


def test_missing_selected_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "faninsar.missions.s1.stack.discover_cdse_slc_items",
        lambda *a, **k: [_cdse_binding(PARENT, swath="IW2")],
    )
    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(lambda _cls, _paths, **_kw: SimpleNamespace(prepare=lambda: None)),
    )
    with pytest.raises(ValueError, match="missing"):
        S1Stack._materialize_from_search(
            _asf_search([_burst_item()]),
            work_dir=tmp_path,
            provider="cdse",
            granularity="burst",
            polarizations=("VV",),
            coreg_mode="geometry",
            orbit_group=None,
            orbit_paths=None,
            aux_cal_paths=None,
            effective_roi=ROI,
            kwargs={"extra": {"orbit_auto_fetch": False}},
        )


def _swath_with_target(t0: datetime, index: int, sensing: datetime) -> SimpleNamespace:
    bursts = []
    for i in range(10):
        bursts.append(
            SimpleNamespace(
                index=i,
                sensing_time=sensing if i == index else t0 + timedelta(seconds=3 * i),
                absolute_burst_id=None,
            )
        )
    return SimpleNamespace(
        bursts=tuple(bursts),
        lines_per_burst=1500,
        azimuth_time_interval_s=0.0020555563,
        samples_per_burst=2,
    )


def test_per_binding_hints_resolve_two_swaths_two_dates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Global id/time lists would extra-match across IW and dates; hints must not."""
    import faninsar.missions.s1.acquisition as acq

    iw2_d1 = datetime(2020, 1, 10, 11, 49, 30, tzinfo=UTC)
    iw2_d2 = datetime(2020, 1, 22, 11, 49, 28, tzinfo=UTC)
    date2_start = datetime(2020, 1, 22, 11, 49, 18, tzinfo=UTC)
    asf_items = [
        _burst_item(),
        _burst_item(swath="IW2", ordinal=3, start=iw2_d1, absolute=159_203),
        _burst_item(parent=DATE2, start=DATE2_BURST, absolute=160_008, item_id="d2iw1"),
        _burst_item(
            parent=DATE2,
            swath="IW2",
            ordinal=3,
            start=iw2_d2,
            absolute=160_203,
            item_id="d2iw2",
        ),
    ]
    bindings = [
        _cdse_binding(PARENT),
        _cdse_binding(PARENT, swath="IW2"),
        _cdse_binding(DATE2),
        _cdse_binding(DATE2, swath="IW2"),
        _cdse_binding(NEIGHBOR),
        _cdse_binding(PARENT, swath="IW3"),
    ]
    parents = {PARENT, DATE2}
    bindings = [b for b in bindings if b["product_id"] in parents]
    hinted = _apply_asf_burst_hints(bindings, _asf_burst_groups(asf_items))
    assert {(b["product_id"], b["swath"]) for b in hinted} == {
        (PARENT, "IW1"),
        (PARENT, "IW2"),
        (DATE2, "IW1"),
        (DATE2, "IW2"),
    }

    swaths = {
        (PARENT, "IW1"): _swath_with_target(PARENT_START, 8, BURST8_START),
        (PARENT, "IW2"): _swath_with_target(PARENT_START, 3, iw2_d1),
        (DATE2, "IW1"): _swath_with_target(date2_start, 8, DATE2_BURST),
        (DATE2, "IW2"): _swath_with_target(date2_start, 3, iw2_d2),
    }
    monkeypatch.setattr(
        acq,
        "_fetch_annotation_xml",
        lambda item, fetch_context=None: f"{item['product_id']}|{item['swath']}",
    )
    monkeypatch.setattr(
        acq,
        "parse_annotation_xml",
        lambda xml, **kw: swaths[tuple(xml.split("|"))],
    )
    resolved: list[tuple[str, str, int]] = []

    def fake_block(item, swath, index, fetch_context=None):
        resolved.append((item["product_id"], item["swath"], int(index)))
        return object(), ((0, 1),)

    monkeypatch.setattr(acq, "_fetch_burst_block", fake_block)
    written = {"n": 0}

    def fake_write(swath, payloads, track_key, out_dir, **kw):
        written["n"] += 1
        return SimpleNamespace(safe_root=Path(out_dir) / f"{written['n']}.SAFE")

    monkeypatch.setattr(
        "faninsar.missions.s1.io.path_safe.write_path_safe", fake_write
    )

    paths = materialize_cdse_burst_safes(
        None, items=hinted, work_dir=tmp_path, polarizations=("VV",)
    )
    assert len(paths) == 4
    assert set(resolved) == {
        (PARENT, "IW1", 8),
        (PARENT, "IW2", 3),
        (DATE2, "IW1", 8),
        (DATE2, "IW2", 3),
    }

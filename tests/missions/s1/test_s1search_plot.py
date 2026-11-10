"""Plotting of ASF burst search results."""

from __future__ import annotations

from dataclasses import replace

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from shapely.geometry import box, mapping

from faninsar.missions.s1.s1search import S1Search
from faninsar.remote.records import AcquisitionMetadata, CatalogItem
from faninsar.remote.catalog import _normalize_record
from faninsar.remote.protocols import _CallLedger
from faninsar.remote.providers.asf_search import ASFSearchAdapter
from faninsar.remote.records import RemoteResourceBudget


def _item(name, properties, *, raw=None, geometry=True):
    return CatalogItem(
        provider="asf",
        catalog="sentinel1-asf",
        collection=None,
        item_id=name,
        acquisition=AcquisitionMetadata(
            acquisition_id=name,
            platform="S1A",
            geometry=mapping(box(0, 0, 2, 1)) if geometry else None,
            properties=properties,
        ),
        assets={},
        raw_metadata=raw or {},
    )


def test_plot_annotations_group_actual_burst_ids_and_counts():
    items = (
        _item("a", {"absoluteBurstID": 11515915}),
        _item("b", {"absoluteBurstID": 11515915}),
        _item(
            "c", {}, raw={"properties": {"burst": {"fullBurstID": "071_151229_IW2"}}}
        ),
        _item("d", {"absoluteBurstID": 11515915}, geometry=False),
        _item("unknown", {}),
    )
    fig, ax = plt.subplots()
    try:
        assert S1Search(items).plot(ax=ax) is ax
        assert [text.get_text() for text in ax.texts] == [
            "11515915 (1)",
            "11515915 (1)",
            "071_151229_IW2 (1)",
        ]
    finally:
        plt.close(fig)


class _Options:
    def __init__(self, **kwargs):
        self.values = kwargs


class _Session:
    def __init__(self):
        self.hooks = {"response": []}

    def post(self, **kwargs):
        raise AssertionError("No network request is expected")

    def close(self):
        pass


class _Package:
    __version__ = "13.0.0"
    ASFSearchOptions = _Options

    def __init__(self, products):
        self.products = products

    def search_generator(self, *, opts):
        yield self.products


def test_adapter_multi_date_bursts_group_by_track_cycle_and_swath():
    # IDs and dates are from ASF's live 071_151229_IW2 search; variants
    # exercise the track and subswath boundaries with the same cycle number.
    names = [
        ("S1_151229_IW2_20261107T135203_VV_8229-BURST", 71, 11515915),
        ("S1_151229_IW2_20261101T135150_VH_F578-BURST", 71, 21787306),
        ("S1_151229_IW1_20261101T135150_VV_F578-BURST", 71, 21787307),
        ("S1_151229_IW2_20261101T135150_VV_F578-BURST", 72, 21787308),
        ("unrecognized", 71, 21787309),
        ("S1_151230_IW2_20261101T135150_VV_F578-BURST", 71, 21787310),
    ]
    products = []
    for name, track, absolute in names:
        products.append(
            {
                "type": "Feature",
                "id": name,
                "geometry": mapping(box(0, 0, 1, 1)),
                "properties": {
                    "sceneName": name,
                    "pathNumber": track,
                    "burst": {"absoluteBurstID": absolute, "relativeBurstID": 151229},
                    "url": "https://datapool.asf.alaska.edu/SLC/fixture.zip",
                },
            }
        )
    adapter = ASFSearchAdapter(
        collection="sentinel-1", package=_Package(products), session_factory=_Session
    )
    records = list(adapter.items(ledger=_CallLedger(RemoteResourceBudget())))
    items = tuple(
        _normalize_record(r, "sentinel1-asf", adapter, "anonymous") for r in records
    )
    assert all("relativeBurstID" in item.acquisition.properties for item in items)
    fig, ax = plt.subplots()
    try:
        S1Search(items).plot(ax=ax)
        assert {text.get_text() for text in ax.texts} == {
            "071_151229_IW2 (2)",
            "071_151229_IW1 (1)",
            "072_151229_IW2 (1)",
            "21787309 (1)",
            "21787310 (1)",
        }
    finally:
        plt.close(fig)


def test_real_asf_s1burst_product_umm_adapter_multi_date_plot():
    asf = pytest.importorskip("asf_search")
    samples = [
        ("S1_151229_IW2_20261107T135203_VV_8229-BURST", 71, 11515915),
        ("S1_151229_IW2_20261101T135150_VH_F578-BURST", 71, 21787306),
        ("S1_151229_IW1_20261101T135150_VV_F578-BURST", 71, 21787307),
        ("S1_151229_IW2_20261101T135150_VV_F578-BURST", 72, 21787308),
        ("S1_151230_IW2_20261101T135150_VV_F578-BURST", 71, 21787309),
        ("S1_151230_IW2_20261107T135203_VV_8229-BURST", 71, 21787309),
    ]
    products = []
    for name, track, absolute in samples:
        base = "https://sentinel1-burst.asf.alaska.edu/fixture/IW2/VV/1"
        umm = {
            "GranuleUR": name,
            "Provider": "ASF",
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2026-11-01T13:51:51Z",
                    "EndingDateTime": "2026-11-01T13:51:54Z",
                }
            },
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "BoundingRect": {
                            "WestBoundingCoordinate": 0,
                            "EastBoundingCoordinate": 1,
                            "SouthBoundingCoordinate": 0,
                            "NorthBoundingCoordinate": 1,
                        }
                    }
                }
            },
            "Platforms": [{"ShortName": "SENTINEL-1C"}],
            "AdditionalAttributes": [
                {"Name": "BURST_ID_ABSOLUTE", "Values": [str(absolute)]},
                {"Name": "BURST_ID_RELATIVE", "Values": ["151229"]},
                {"Name": "PATH_NUMBER", "Values": [str(track)]},
                {"Name": "SUBSWATH_NAME", "Values": ["IW2"]},
                {"Name": "BURST_ID_FULL", "Values": ["071_151229_IW2"]},
            ],
            "RelatedUrls": [
                {"Type": "USE SERVICE API", "Format": "GeoTIFF", "URL": base + ".tiff"},
                {"Type": "USE SERVICE API", "Format": "XML", "URL": base + ".xml"},
            ],
        }
        products.append(asf.S1BurstProduct({"umm": umm, "meta": {}}))
    adapter = ASFSearchAdapter(
        collection="sentinel-1", package=_Package(products), session_factory=_Session
    )
    records = list(adapter.items(ledger=_CallLedger(RemoteResourceBudget())))
    items = tuple(
        _normalize_record(r, "sentinel1-asf", adapter, "anonymous") for r in records
    )
    assert [item.acquisition.properties["absoluteBurstID"] for item in items] == [
        value for _, _, value in samples
    ]
    assert [item.acquisition.properties["relativeOrbit"] for item in items] == [
        value for _, value, _ in samples
    ]
    assert all(
        item.acquisition.properties["relativeBurstID"] == 151229 for item in items
    )
    assert all("fullBurstID" not in item.acquisition.properties for item in items)
    # Search record without geometry remains part of the count, not a date count.
    no_geometry = replace(
        items[0], acquisition=replace(items[0].acquisition, geometry=None)
    )
    fig, ax = plt.subplots()
    try:
        S1Search((*items, no_geometry)).plot(ax=ax)
        assert [text.get_text() for text in ax.texts] == [
            "071_151229_IW2 (3)",
            "071_151229_IW1 (1)",
            "072_151229_IW2 (1)",
            "21787309 (1)",
            "21787309 (1)",
        ]
    finally:
        plt.close(fig)


def test_plot_without_burst_id_preserves_roi_and_no_annotations():
    fig, ax = plt.subplots()
    try:
        result = S1Search((_item("a", {}),)).plot(roi=box(0, 0, 1, 1), ax=ax)
        assert result is ax
        assert not ax.texts
        assert len(ax.collections) == 2
        assert ax.get_title() == "S1Search (1 items)"
    finally:
        plt.close(fig)

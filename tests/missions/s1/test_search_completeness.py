"""Sentinel-1 complete-search behavior under PROPOSAL-0073."""

from __future__ import annotations

import pytest
from shapely.geometry import box, mapping

from faninsar import remote
from faninsar.data.query import BoundingBox
from faninsar.missions.s1 import S1Stack
from faninsar.remote.catalog import _ADAPTERS

ROI = BoundingBox(0, 0, 1, 1, crs=4326)


class Bursts:
    provider = "asf"
    origins = ("https://fixture.invalid",)
    path_prefixes = ("/",)
    redirect_origins = ()
    profiles = ("anonymous", "earthdata-asf")

    def __init__(self, count, channels=("VV",)):
        from faninsar.remote.providers.asf_search import ASFSearchAdapter

        expected = ASFSearchAdapter(collection="sentinel-1")
        for field in (
            "provider",
            "collection",
            "query_collection",
            "engine",
            "endpoint",
            "profiles",
            "data_origins",
            "path_prefixes",
            "redirect_origins",
        ):
            setattr(self, field, getattr(expected, field))
        self.count = count
        self.channels = channels
        self.visited = 0

    def prepare_query(self, **kwargs):
        return {**kwargs, "requires_complete_traversal": True}

    def items(self, *, limit=None, **kwargs):
        for i in range(self.count if limit is None else min(limit, self.count)):
            self.visited += 1
            yield {
                "id": f"burst-{i:05d}",
                "geometry": mapping(box(0, 0, 1, 1)),
                "assets": {},
                "Polarization": self.channels[i % len(self.channels)],
            }


def test_default_s1_search_has_no_ten_thousand_candidate_cap(monkeypatch):
    adapter = Bursts(10005)
    monkeypatch.setitem(_ADAPTERS, "sentinel1-asf", adapter)
    result = S1Stack.search(ROI)
    assert len(result) == 10005
    assert result[-1].item_id == "burst-10004"
    assert adapter.visited == 10005


def test_local_polarization_finishing_precedes_explicit_limit(monkeypatch):
    adapter = Bursts(225, ("HH", "VV", "VH"))
    monkeypatch.setitem(_ADAPTERS, "sentinel1-asf", adapter)
    result = S1Stack.search(ROI, polarizations=("VV", "VH"), limit=150)
    assert len(result) == 150
    assert result[0].item_id == "burst-00001"
    assert result[-1].item_id == "burst-00224"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_s1_invalid_limit_rejected(monkeypatch, limit):
    adapter = Bursts(5)
    monkeypatch.setitem(_ADAPTERS, "sentinel1-asf", adapter)
    with pytest.raises(remote.RemoteQueryError, match="invalid_limit"):
        S1Stack.search(ROI, limit=limit)
    assert adapter.visited == 0

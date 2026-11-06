"""Public complete-search contracts for PROPOSAL-0073."""

from __future__ import annotations

import pytest
from shapely.geometry import box, mapping

from faninsar import remote
from faninsar.data.query import BoundingBox
from faninsar.remote.catalog import _ADAPTERS
from faninsar.remote.errors import RemoteLimitError, RemoteQueryError

ROI = BoundingBox(0, 0, 1, 1, crs=4326)


def record(index):
    return {"id": str(index), "geometry": mapping(box(0, 0, 1, 1)), "assets": {}}


class Pages:
    provider = "complete-fixture"
    origins = ("https://fixture.invalid",)
    path_prefixes = ("/",)
    redirect_origins = ()
    profiles = ("anonymous",)

    def __init__(self, fail=False):
        self.visited = []
        self.fail = fail

    def items(self, *, limit=None, ledger=None, **kwargs):
        emitted = 0
        for page in range(3):
            ledger.request()
            self.visited.append(page)
            if self.fail and page == 2:
                raise RemoteQueryError("incomplete_results")
            for index in [*range(page * 75, (page + 1) * 75), page * 75]:
                if limit is not None and emitted >= limit:
                    return
                yield record(index)
                emitted += 1


@pytest.fixture
def pages(monkeypatch):
    adapter = Pages()
    monkeypatch.setitem(_ADAPTERS, "complete-fixture", adapter)
    return adapter


def test_default_search_exhausts_three_pages_and_deduplicates(pages):
    result = remote.search(ROI, catalog="complete-fixture")
    assert [item.item_id for item in result] == [str(i) for i in range(225)]
    assert pages.visited == [0, 1, 2]


def test_explicit_limit_counts_matching_unique_records(pages):
    result = remote.search(ROI, catalog="complete-fixture", limit=150)
    assert [item.item_id for item in result] == [str(i) for i in range(150)]


def test_explicit_limit_stops_before_unneeded_provider_page(pages):
    pages.fail = True
    result = remote.search(ROI, catalog="complete-fixture", limit=75)
    assert [item.item_id for item in result] == [str(i) for i in range(75)]
    assert pages.visited == [0]


def test_exact_filtering_does_not_consume_result_limit(pages, monkeypatch):
    original = record

    def with_outside_footprints(index):
        value = original(index)
        if index < 100:
            value["geometry"] = mapping(box(20, 20, 21, 21))
        return value

    monkeypatch.setattr(
        __import__(__name__, fromlist=["record"]), "record", with_outside_footprints
    )
    result = remote.search(ROI, catalog="complete-fixture", limit=100)
    assert [item.item_id for item in result] == [str(i) for i in range(100, 200)]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_limits_fail_before_provider_work(pages, limit):
    with pytest.raises(RemoteQueryError, match="invalid_limit"):
        remote.search(ROI, catalog="complete-fixture", limit=limit)
    assert pages.visited == []


def test_explicit_item_guard_reports_incomplete_query(pages):
    with pytest.raises(RemoteLimitError, match="max_items"):
        remote.search(
            ROI,
            catalog="complete-fixture",
            budget=remote.RemoteResourceBudget(max_items=100),
        )


def test_provider_failure_does_not_publish_partial_success(pages):
    pages.fail = True
    with pytest.raises(RemoteQueryError, match="incomplete_results"):
        remote.search(ROI, catalog="complete-fixture")


def test_request_guard_does_not_publish_partial_success(pages):
    with pytest.raises(RemoteLimitError, match="max_requests"):
        remote.search(
            ROI,
            catalog="complete-fixture",
            budget=remote.RemoteResourceBudget(max_requests=2),
        )

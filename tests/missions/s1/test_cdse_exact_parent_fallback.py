"""Missing-only exact-parent CDSE OData fallback at the discovery seam.

The CDSE STAC index can omit a parent that ASF selected but that still exists
in the catalogue OData Products/Nodes endpoints.  These tests exercise
``discover_cdse_slc_items`` with scripted search and lookup seams, including
IW2/IW3 assets, ambiguity, absent products, ROI bounds, and the shared
preparation ledger.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from shapely.geometry import box, mapping

from faninsar import remote
from faninsar.data.query import BoundingBox
from faninsar.missions.s1.acquisition import (
    _preparation_remote_policy,
    discover_cdse_slc_items,
)
from faninsar.remote.catalog import _ADAPTERS, _REGISTRY_LOCK
from faninsar.remote.errors import RemoteAccessError
from faninsar.remote.protocols import _CallLedger
from faninsar.remote.records import RemoteResourceBudget

PARENT = "S1A_IW_SLC__1SDV_20150428T171437_20150428T171510_005687_0074BD_87A1"
PRODUCT_UUID = "ab0e1fd5-6011-574f-ab49-08119ba6c91c"
ROI = BoundingBox(10.0, 45.0, 11.0, 46.0, crs="EPSG:4326")
FOOTPRINT = mapping(box(7.4, 45.0, 11.2, 47.4))
START = datetime(2015, 4, 28, 17, 14, 37, 931000, tzinfo=UTC)


@pytest.fixture
def isolated_cdse_catalog() -> Any:
    """Remove ``sentinel1-cdse`` for the test and restore afterwards."""
    with _REGISTRY_LOCK:
        previous = _ADAPTERS.pop("sentinel1-cdse", None)
    try:
        yield
    finally:
        with _REGISTRY_LOCK:
            _ADAPTERS.pop("sentinel1-cdse", None)
            if previous is not None:
                _ADAPTERS["sentinel1-cdse"] = previous


def _catalog_item(
    parent: str = PARENT,
    *,
    swaths: tuple[str, ...] = ("iw1", "iw2", "iw3"),
    polarization: str = "vv",
    geometry: Any = None,
) -> remote.CatalogItem:
    """Return a normalized CDSE item with OData range assets."""
    assets: dict[str, remote.RemoteAsset] = {}
    for swath in swaths:
        suffix = f"{swath}-{polarization}"
        stem = (
            f"s1a-{swath}-slc-{polarization}-"
            "20150428t171437-20150428t171508-005687-0074bd-004"
        )
        base = (
            "https://download.dataspace.copernicus.eu/odata/v1/"
            f"Products({PRODUCT_UUID})/Nodes({parent}.SAFE)"
        )
        properties = {
            "swath": swath.upper(),
            "polarization": polarization.upper(),
            "file:size": 1000,
        }
        assets[f"measurement-{suffix}"] = remote.RemoteAsset(
            provider="cdse",
            catalog="sentinel1-cdse",
            collection="sentinel-1-slc",
            item_id=parent,
            key=f"measurement-{suffix}",
            href=f"{base}/Nodes(measurement)/Nodes({stem}.tiff)/$value",
            media_type="image/tiff",
            size_bytes=1000,
            properties=properties,
        )
        assets[f"annotation-{suffix}"] = remote.RemoteAsset(
            provider="cdse",
            catalog="sentinel1-cdse",
            collection="sentinel-1-slc",
            item_id=parent,
            key=f"annotation-{suffix}",
            href=f"{base}/Nodes(annotation)/Nodes({stem}.xml)/$value",
            media_type="application/xml",
            size_bytes=100,
            properties=properties,
        )
    acquisition = remote.AcquisitionMetadata(
        acquisition_id=parent,
        start=START,
        end=START + timedelta(seconds=33),
        geometry=FOOTPRINT if geometry is None else geometry,
        properties={"relative_orbit": 15, "orbit_direction": "ASCENDING"},
    )
    return remote.CatalogItem(
        provider="cdse",
        catalog="sentinel1-cdse",
        collection="sentinel-1-slc",
        item_id=parent,
        acquisition=acquisition,
        assets=assets,
        raw_metadata={"relative_orbit": 15, "orbit_direction": "ASCENDING"},
    )


def _patch_search(
    monkeypatch: pytest.MonkeyPatch, items: list[remote.CatalogItem]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_search(roi: Any, **kwargs: Any) -> list[remote.CatalogItem]:
        calls.append({"roi": roi, **kwargs})
        return list(items)

    monkeypatch.setattr("faninsar.remote.search", fake_search)
    return calls


def _patch_lookup(monkeypatch: pytest.MonkeyPatch, lookup: Any) -> None:
    from faninsar.remote.providers import cdse

    monkeypatch.setattr(cdse, "lookup_cdse_safe_record", lookup)


def test_missing_parent_binds_every_exact_iw_swath(
    monkeypatch: pytest.MonkeyPatch, isolated_cdse_catalog: None
) -> None:
    """Absent STAC parent yields IW1/IW2/IW3 bindings from exact OData."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [])
    captured: dict[str, Any] = {}

    def fake_lookup(parent: str, *, ledger: _CallLedger) -> remote.CatalogItem:
        captured["parent"] = parent
        captured["ledger"] = ledger
        return _catalog_item(parent)

    _patch_lookup(monkeypatch, fake_lookup)
    bindings = discover_cdse_slc_items(ROI, parent_ids=(PARENT,), polarizations=("VV",))

    assert captured["parent"] == PARENT
    assert {
        (binding["product_id"], binding["swath"], binding["polarization"])
        for binding in bindings
    } == {
        (PARENT, "IW1", "VV"),
        (PARENT, "IW2", "VV"),
        (PARENT, "IW3", "VV"),
    }
    assert all(binding["measurement_size"] == 1000 for binding in bindings)
    assert all(binding["annotation_size"] == 100 for binding in bindings)
    assert all(binding["relative_orbit"] == 15 for binding in bindings)
    assert all(binding["orbit_direction"] == "ASCENDING" for binding in bindings)
    assert all(
        str(binding["measurement_url"]).startswith(
            "https://download.dataspace.copernicus.eu/"
        )
        for binding in bindings
    )
    assert any(
        "s1a-iw2-slc-vv-" in str(binding["measurement_url"]) for binding in bindings
    )
    assert any(
        "s1a-iw3-slc-vv-" in str(binding["measurement_url"]) for binding in bindings
    )


def test_present_stac_parent_never_calls_odata(
    monkeypatch: pytest.MonkeyPatch, isolated_cdse_catalog: None
) -> None:
    """A STAC-satisfied parent keeps the existing path and zero lookup calls."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [_catalog_item(PARENT)])
    calls: list[str] = []

    def unexpected_lookup(parent: str, *, ledger: _CallLedger) -> None:
        del parent, ledger
        calls.append("lookup")
        msg = "OData fallback must not run for a present parent"
        raise AssertionError(msg)

    _patch_lookup(monkeypatch, unexpected_lookup)
    bindings = discover_cdse_slc_items(ROI, parent_ids=(PARENT,), polarizations=("VV",))
    assert calls == []
    assert len(bindings) == 3


def test_no_parent_ids_disables_the_fallback(
    monkeypatch: pytest.MonkeyPatch, isolated_cdse_catalog: None
) -> None:
    """Untrusted/absent parent identity means no exact-name lookup at all."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [])
    calls: list[str] = []

    def unexpected_lookup(parent: str, *, ledger: _CallLedger) -> None:
        del parent, ledger
        calls.append("lookup")

    _patch_lookup(monkeypatch, unexpected_lookup)
    assert discover_cdse_slc_items(ROI, polarizations=("VV",)) == []
    assert calls == []


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (
            RemoteAccessError("ambiguous_product", "two exact products"),
            "ambiguous_product",
        ),
        (None, "product_not_found"),
    ],
)
def test_missing_parent_failures_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cdse_catalog: None,
    outcome: Any,
    reason: str,
) -> None:
    """Ambiguous or absent exact products raise instead of fabricating a match."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [])

    def fake_lookup(parent: str, *, ledger: _CallLedger) -> Any:
        del parent, ledger
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    _patch_lookup(monkeypatch, fake_lookup)
    with pytest.raises(RemoteAccessError) as error:
        discover_cdse_slc_items(ROI, parent_ids=(PARENT,), polarizations=("VV",))
    assert error.value.reason == reason


def test_fallback_uses_the_caller_budget(
    monkeypatch: pytest.MonkeyPatch, isolated_cdse_catalog: None
) -> None:
    """A direct call meters the lookup with the caller's budget, not defaults."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [])
    budget = RemoteResourceBudget(max_requests=7)
    captured: dict[str, Any] = {}

    def fake_lookup(parent: str, *, ledger: _CallLedger) -> remote.CatalogItem:
        captured["ledger"] = ledger
        return _catalog_item(parent)

    _patch_lookup(monkeypatch, fake_lookup)
    discover_cdse_slc_items(
        ROI,
        parent_ids=(PARENT,),
        polarizations=("VV",),
        budget=budget,
    )
    assert captured["ledger"].budget is budget


def test_fallback_reuses_the_preparation_ledger(
    monkeypatch: pytest.MonkeyPatch, isolated_cdse_catalog: None
) -> None:
    """During preparation, exact-name requests share the operation ledger."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [])
    budget = RemoteResourceBudget(max_requests=7)
    policy_ledger = _CallLedger(budget)
    captured: dict[str, Any] = {}

    def fake_lookup(parent: str, *, ledger: _CallLedger) -> remote.CatalogItem:
        captured["ledger"] = ledger
        return _catalog_item(parent)

    _patch_lookup(monkeypatch, fake_lookup)
    token = _preparation_remote_policy.set((budget, policy_ledger))
    try:
        discover_cdse_slc_items(
            ROI,
            parent_ids=(PARENT,),
            polarizations=("VV",),
            budget=RemoteResourceBudget(max_requests=1),
        )
    finally:
        _preparation_remote_policy.reset(token)
    assert captured["ledger"] is policy_ledger


def test_exact_parent_outside_roi_fails_closed(
    monkeypatch: pytest.MonkeyPatch, isolated_cdse_catalog: None
) -> None:
    """An exact parent that does not intersect the ROI is never substituted."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [])

    def fake_lookup(parent: str, *, ledger: _CallLedger) -> remote.CatalogItem:
        del ledger
        return _catalog_item(parent, geometry=mapping(box(0.0, 0.0, 1.0, 1.0)))

    _patch_lookup(monkeypatch, fake_lookup)
    with pytest.raises(RemoteAccessError) as error:
        discover_cdse_slc_items(ROI, parent_ids=(PARENT,), polarizations=("VV",))
    assert error.value.reason == "product_not_found"


def test_missing_requested_polarization_fails_closed(
    monkeypatch: pytest.MonkeyPatch, isolated_cdse_catalog: None
) -> None:
    """An exact parent without the requested channel is not replaced."""
    del isolated_cdse_catalog
    _patch_search(monkeypatch, [])

    def fake_lookup(parent: str, *, ledger: _CallLedger) -> remote.CatalogItem:
        del ledger
        return _catalog_item(parent, polarization="vv")

    _patch_lookup(monkeypatch, fake_lookup)
    with pytest.raises(RemoteAccessError) as error:
        discover_cdse_slc_items(ROI, parent_ids=(PARENT,), polarizations=("VH",))
    assert error.value.reason == "invalid_product"

"""Explicit DEM resolver contracts for PROPOSAL-0045 Task C."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from faninsar.processing.geometry.dem_sources import (
    AuthenticatedGranuleSource,
    DeferredGranulePlan,
    DemSourceUnavailableError,
    get_dem_source,
)


def test_nasadem_lpdaac_resolver_is_zip_and_vsizip() -> None:
    """NASADEM resolves to the authoritative LPDAAC zip collection."""
    source = get_dem_source("nasadem:earthdata")
    assert isinstance(source, AuthenticatedGranuleSource)
    assert source.cmr_collection == "C2763264762-LPCLOUD"
    assert source.asset_pattern == "*.zip"
    assert source.mosaic_recipe().gdal_open == "/vsizip/{path}/{member}"


def test_nasadem_discovery_registers_live_lcloud_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NASADEM uses CMR's live LPCLOUD provider identity."""
    from faninsar import remote
    from faninsar.processing.geometry import dem_sources

    source = get_dem_source("nasadem:earthdata")
    plan = DeferredGranulePlan(
        allowed_hosts=("cmr.earthdata.nasa.gov", source.data_host),
        credential_ref="earthdata",
        source_name=source.name,
        bounds=(-121.0, 36.0, -120.0, 37.0),
        cmr_collection=source.cmr_collection,
        data_host=source.data_host,
        asset_pattern=source.asset_pattern,
    )
    captured: dict[str, object] = {}

    def register_adapter(_name: str, adapter: object) -> None:
        captured["adapter"] = adapter

    def search(*_args: object, **_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr(remote, "_register_adapter", register_adapter)
    monkeypatch.setattr(remote, "search", search)
    dem_sources._CMR_ADAPTERS.pop("faninsar-dem-cmr-nasadem@earthdata", None)
    with monkeypatch.context() as context:
        context.setattr(
            "faninsar.processing.geometry.dem_transport.resolve_credentials",
            lambda _: object(),
        )
        with contextlib.suppress(DemSourceUnavailableError):
            source.discover(plan)
    adapter = captured["adapter"]
    assert adapter.provider == "LPCLOUD"


def test_nisar_resolver_selects_wgs84_cog_without_vsizip() -> None:
    """NISAR DEM resolves to EPSG4326 COG assets, not archive members."""
    source = get_dem_source("nisar-glo30")
    assert isinstance(source, AuthenticatedGranuleSource)
    assert source.cmr_collection == "C3803703055-ASF"
    assert source.asset_pattern == "*.tif"
    assert source.title_filter_required is True
    recipe = source.mosaic_recipe()
    assert recipe.gdal_open == "{path}"
    assert recipe.source_crs == "EPSG:4326"

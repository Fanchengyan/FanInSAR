"""Explicit DEM resolver contracts for PROPOSAL-0045 Task C."""

from __future__ import annotations

from faninsar.processing.geometry.dem_sources import (
    AuthenticatedGranuleSource,
    get_dem_source,
)


def test_nasadem_lpdaac_resolver_is_zip_and_vsizip() -> None:
    """NASADEM resolves to the authoritative LPDAAC zip collection."""
    source = get_dem_source("nasadem:earthdata")
    assert isinstance(source, AuthenticatedGranuleSource)
    assert source.cmr_collection == "C2763264762-LPCLOUD"
    assert source.asset_pattern == "*.zip"
    assert source.mosaic_recipe().gdal_open == "/vsizip/{path}/{member}"


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

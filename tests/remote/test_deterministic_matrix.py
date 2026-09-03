"""Deterministic public provider and STAC profile matrix tests."""

from __future__ import annotations

import pytest

from faninsar import remote
from faninsar.processing.geometry.dem_sources import dem_catalog, get_dem_source


@pytest.mark.parametrize(
    ("selection", "product", "provider", "datum"),
    [
        ("glo30", "glo30", "aws", "egm2008"),
        ("glo30:pc", "glo30", "pc", "egm2008"),
        ("glo90:pc", "glo90", "pc", "egm2008"),
        ("nasadem", "nasadem", "pc", "egm96"),
        ("nasadem:earthdata", "nasadem", "earthdata", "egm96"),
        ("nisar-glo30", "nisar-glo30", "earthdata", "ellipsoidal"),
        ("auto", "glo30", "aws", "egm2008"),
    ],
)
def test_dem_selection_matrix_is_stable(
    selection: str, product: str, provider: str, datum: str
) -> None:
    """Every supported selection resolves to its exact registry identity."""
    source = get_dem_source(selection)
    assert (source.product, source.provider, source.vertical_datum) == (
        product,
        provider,
        datum,
    )
    assert source.wired is True


def test_dem_catalog_is_offline_and_contains_provider_axis() -> None:
    """Catalog inspection exposes product/provider metadata without I/O."""
    catalog = dem_catalog()
    assert catalog["glo30"]["default"] == "aws"
    assert catalog["glo30"]["providers"]["pc"] == {
        "wired": True,
        "auth": "none",
    }
    assert catalog["auto"]["providers"] == {"aws": {"wired": True, "auth": "none"}}


def test_public_stac_profile_mapping_is_deterministic() -> None:
    """STAC profile normalization retains only the pinned profile names."""
    item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/sar/v1.3.2/schema.json"],
        "id": "scene-1",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, -1]]],
        },
        "properties": {"datetime": "2024-01-01T00:00:00Z"},
        "assets": {
            "data": {
                "href": "https://example.invalid/data/scene-1.tif",
                "type": "image/tiff; application=geotiff",
            }
        },
    }
    record = remote.normalize_stac_item(item, provider="fixture", catalog="stac")
    assert record["stac_profiles"] == ("sar",)
    assert record["assets"]["data"]["href"].endswith("scene-1.tif")

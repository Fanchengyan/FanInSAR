"""Fixture-first tests for the pinned remote STAC profile."""

from __future__ import annotations

from typing import Any

import pytest

from faninsar.remote.standards import (
    MalformedSTACItemError,
    STACProfileError,
    UnknownSTACProfileError,
    normalize_stac_item,
    validate_stac_item,
)


def _item(**overrides: Any) -> dict[str, Any]:
    """Build a valid STAC 1.1 fixture item."""
    item: dict[str, Any] = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/sar/v1.3.2/schema.json",
            "https://stac-extensions.github.io/sat/v1.2.0/schema.json",
            "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
            "https://stac-extensions.github.io/file/v2.1.0/schema.json",
        ],
        "id": "scene-1",
        "collection": "registered-scenes",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
        "properties": {
            "datetime": "2024-01-02T03:04:05Z",
            "platform": "sentinel-1a",
            "instruments": ["sar"],
            "sar:instrument_mode": "IW",
            "sat:relative_orbit": 12,
            "proj:epsg": 4326,
        },
        "assets": {
            "data": {
                "href": "https://example.invalid/data/scene-1.tif",
                "type": "image/tiff; application=geotiff",
                "file:size": 4,
                "file:checksum": (
                    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                ),
            }
        },
    }
    item.update(overrides)
    return item


def test_profile_maps_common_sar_satellite_projection_and_file_fields() -> None:
    """The mapper emits exact geometry-ready records without custom fields."""
    record = normalize_stac_item(
        _item(),
        provider="fixture-provider",
        catalog="fixture-catalog",
        collection="registered-scenes",
    )

    assert record["id"] == "scene-1"
    assert record["collection"] == "registered-scenes"
    assert record["geometry"]["type"] == "Polygon"
    assert record["acquisition"]["platform"] == "sentinel-1a"
    assert record["acquisition"]["mode"] == "IW"
    assert record["acquisition"]["orbit"] == 12
    assert record["assets"]["data"]["size"] == 4
    assert not any(str(key).startswith("faninsar:") for key in record["properties"])


def test_live_planetary_computer_stac_1_0_projection_input_is_accepted() -> None:
    """The live PC profile is accepted without changing FanInSAR output policy."""
    record = normalize_stac_item(
        _item(
            stac_version="1.0.0",
            stac_extensions=[
                "https://stac-extensions.github.io/projection/v1.0.0/schema.json"
            ],
        ),
        provider="planetary-computer",
        catalog="planetary-computer",
    )

    assert record["stac_profiles"] == ("projection",)
    assert record["properties"]["proj:code"] == "EPSG:4326"
    assert "proj:epsg" not in record["properties"]


def test_unknown_legacy_extension_is_still_rejected() -> None:
    """Input compatibility does not become an arbitrary-extension escape hatch."""
    item = _item(
        stac_version="1.0.0",
        stac_extensions=["https://stac-extensions.github.io/eo/v1.0.0/schema.json"],
    )
    with pytest.raises(UnknownSTACProfileError):
        validate_stac_item(item)


def test_unknown_extension_fails_closed() -> None:
    """A provider cannot opt into an unregistered extension by name."""
    item = _item(
        stac_extensions=["https://example.invalid/extensions/custom/v1/schema.json"]
    )
    with pytest.raises(UnknownSTACProfileError) as error:
        validate_stac_item(item)
    assert error.value.reason == "unknown_stac_extension"


def test_extension_schema_uri_requires_the_approved_https_host() -> None:
    """A matching path on another host cannot impersonate a STAC schema."""
    item = _item(
        stac_extensions=[
            "https://attacker.invalid/sar/v1.3.2/schema.json",
        ]
    )
    with pytest.raises(UnknownSTACProfileError) as error:
        validate_stac_item(item)
    assert error.value.reason == "unknown_stac_extension"


def test_normalized_stac_records_scrub_azure_sas_material() -> None:
    """Direct STAC normalization does not persist signed URL query fields."""
    item = _item(
        properties={
            "datetime": "2024-01-02T03:04:05Z",
            "provider_href": "https://blob.invalid/meta.json?sv=2023&sig=secret&x=1",
            "sp": "r",
        },
        assets={
            "data": {
                "href": "https://blob.invalid/data.tif?sp=r&sv=2023&sig=secret&x=1"
            }
        },
    )
    record = normalize_stac_item(item, provider="fixture", catalog="catalog")

    assert record["assets"]["data"]["href"] == "https://blob.invalid/data.tif"
    assert "sp" not in record["properties"]
    assert record["properties"]["provider_href"] == "https://blob.invalid/meta.json"


def test_normalized_properties_scrub_all_azure_sas_keys() -> None:
    """New Azure SAS fields are removed from nested properties as well."""
    item = _item(
        properties={
            "datetime": "2024-01-02T03:04:05Z",
            "ss": "b",
            "srt": "o",
            "sdd": "2024-01-01",
            "nested": {"ss": "b", "safe": "value"},
        }
    )
    record = normalize_stac_item(item, provider="fixture", catalog="catalog")

    assert "ss" not in record["properties"]
    assert "srt" not in record["properties"]
    assert "sdd" not in record["properties"]
    assert record["properties"]["nested"] == {"safe": "value"}


def test_malformed_geometry_and_unqualified_insar_fields_are_typed() -> None:
    """Geometry and InSAR policy violations use typed profile errors."""
    with pytest.raises(MalformedSTACItemError) as error:
        validate_stac_item(_item(geometry=None))
    assert error.value.reason == "missing_geometry"

    item = _item(properties={"datetime": "2024-01-02T03:04:05Z", "insar:units": "m"})
    with pytest.raises(STACProfileError) as error:
        validate_stac_item(item)
    assert error.value.reason == "unregistered_insar_field"

"""Tests for the pure georeferenced Zarr view descriptors (P0089)."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from faninsar.io.stac.georef import (
    GeoreferencedZarrError,
    GeoreferencedZarrView,
    build_georeferenced_zarr_view,
)
from faninsar.io.stac.record import StacRecord, TemporalCoverage

_CRS_URL = "http://www.opengis.net/def/crs/EPSG/0/32719"
_TRANSFORM = (8.0, 0.0, 288728.0, 0.0, -200.0, 6066040.0)


def _record(
    *,
    kind: str = "ifg",
    domain: str | None = "geo",
    transform: tuple[float, ...] = _TRANSFORM,
    shape: tuple[int, int] = (3, 4),
    properties: dict[str, object] | None = None,
    assets: dict[str, dict[str, object]] | None = None,
) -> StacRecord:
    """Build one metadata-only normalized record for descriptor tests."""
    item_properties: dict[str, object] = {
        "faninsar:shape": list(shape),
        "proj:code": "EPSG:32719",
        "proj:transform": list(transform),
        "proj:shape": list(shape),
    }
    if domain is not None:
        item_properties["faninsar:domain"] = domain
    item_properties.update(properties or {})
    if assets is None:
        assets = {"complex": {"href": "complex.zarr"}}
    asset_paths = {
        key: Path("/work") / "ifg" / str(asset["href"]) for key, asset in assets.items()
    }
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    return StacRecord(
        public_id="ifg-test",
        write_id="0" * 26,
        kind=kind,
        family="interferograms",
        source_item_path=Path("/work") / "ifg" / "item.json",
        source_relative_path="ifg",
        collection_id=None,
        temporal=TemporalCoverage(moment, moment, "timestamp"),
        bbox=None,
        item={
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "ifg-test",
            "properties": item_properties,
            "assets": assets,
        },
        asset_paths=asset_paths,
    )


def _metadata(
    *,
    shape: tuple[int, int] = (3, 4),
    data_type: str = "float32",
    chunk_shape: tuple[int, int] | None = None,
    encoding: dict[str, object] | None = None,
    attributes: dict[str, object] | None = None,
    codecs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one bounded native Zarr v3 array metadata document."""
    if chunk_shape is None:
        chunk_shape = shape
    if encoding is None:
        encoding = {"name": "default", "configuration": {"separator": "/"}}
    if codecs is None:
        codecs = [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ]
    return {
        "shape": list(shape),
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunk_shape)},
        },
        "chunk_key_encoding": encoding,
        "fill_value": 0.0,
        "codecs": codecs,
        "attributes": dict(attributes or {}),
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def _view(
    metadata: dict[str, object] | None = None, **record_kwargs: object
) -> GeoreferencedZarrView:
    """Build a view for the default geo IFG fixture."""
    return build_georeferenced_zarr_view(
        _record(**record_kwargs), "complex", metadata or _metadata()
    )


def _decoded(data: bytes) -> np.ndarray:
    """Decode one generated little-endian float64 coordinate chunk."""
    return np.frombuffer(data, dtype="<f8")


def test_builds_georeferenced_view_for_geo_complex_ifg() -> None:
    """The descriptor preserves native data metadata and adds view metadata."""
    metadata = _metadata(
        data_type="complex64",
        chunk_shape=(2, 3),
        attributes={"faninsar:format": "stac-zarr", "scientific:note": "keep"},
    )
    original = copy.deepcopy(metadata)

    view = build_georeferenced_zarr_view(_record(kind="ifg"), "complex", metadata)

    assert view.source_asset_key == "complex"
    assert view.asset_key == "complex_georef"
    assert view.shape == (3, 4)
    assert view.transform == _TRANSFORM
    assert view.crs == "EPSG:32719"
    assert view.data_type == "complex64"

    values = view.metadata("values/zarr.json")
    assert values is not None
    assert values["shape"] == [3, 4]
    assert values["data_type"] == "complex64"
    assert values["chunk_grid"] == metadata["chunk_grid"]
    assert values["chunk_key_encoding"] == metadata["chunk_key_encoding"]
    assert values["fill_value"] == metadata["fill_value"]
    assert values["codecs"] == metadata["codecs"]
    assert values["attributes"] == {
        "faninsar:format": "stac-zarr",
        "scientific:note": "keep",
        "_CRS": {"url": _CRS_URL},
    }
    assert values["dimension_names"] == ["y", "x"]

    x_metadata = view.metadata("x/zarr.json")
    y_metadata = view.metadata("y/zarr.json")
    assert x_metadata is not None
    assert y_metadata is not None
    assert x_metadata["data_type"] == "float64"
    assert x_metadata["attributes"] == {
        "standard_name": "projection_x_coordinate",
        "axis": "X",
        "units": "m",
    }
    assert y_metadata["attributes"] == {
        "standard_name": "projection_y_coordinate",
        "axis": "Y",
        "units": "m",
    }
    assert x_metadata["codecs"] == [
        {"name": "bytes", "configuration": {"endian": "little"}}
    ]
    assert x_metadata["chunk_grid"]["configuration"]["chunk_shape"] == [4]
    assert y_metadata["chunk_grid"]["configuration"]["chunk_shape"] == [3]

    root = view.metadata("zarr.json")
    assert root is not None
    assert root["node_type"] == "group"
    assert root["consolidated_metadata"] == {
        "kind": "inline",
        "must_understand": False,
        "metadata": {"values": values, "x": x_metadata, "y": y_metadata},
    }

    assert metadata == original
    for relative in (
        "values/c/0/0",
        "/zarr.json",
        "../zarr.json",
        "x/../zarr.json",
        "",
        "nope",
    ):
        assert view.metadata(relative) is None


def test_geographic_crs_uses_degree_units_and_epsg_url() -> None:
    """EPSG geographic output boxes use degrees_east and degrees_north."""
    metadata = _metadata()
    record = _record(properties={"proj:code": "EPSG:4326"})

    view = build_georeferenced_zarr_view(record, "complex", metadata)

    assert view.crs == "EPSG:4326"
    values = view.metadata("values/zarr.json")
    assert values is not None
    assert values["attributes"]["_CRS"] == {
        "url": "http://www.opengis.net/def/crs/EPSG/0/4326"
    }
    x_metadata = view.metadata("x/zarr.json")
    y_metadata = view.metadata("y/zarr.json")
    assert x_metadata is not None
    assert y_metadata is not None
    assert x_metadata["attributes"]["standard_name"] == "longitude"
    assert x_metadata["attributes"]["units"] == "degrees_east"
    assert y_metadata["attributes"]["standard_name"] == "latitude"
    assert y_metadata["attributes"]["units"] == "degrees_north"


def test_non_epsg_crs_uses_pyproj_wkt2() -> None:
    """A non-EPSG CRS is emitted as WKT2 rather than an EPSG URL."""
    metadata = _metadata()
    record = _record(properties={"proj:code": "OGC:CRS84"})

    view = build_georeferenced_zarr_view(record, "complex", metadata)

    values = view.metadata("values/zarr.json")
    assert values is not None
    payload = values["attributes"]["_CRS"]
    assert "wkt" in payload
    assert "url" not in payload
    assert payload["wkt"].startswith("GEOGCRS")


@pytest.mark.parametrize(
    "transform",
    [
        (8.0, 1.0, 0.0, 0.0, -200.0, 0.0),
        (8.0, 0.0, 0.0, 1.0, -200.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, -200.0, 0.0),
        (8.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0, -200.0, 0.0),
    ],
)
def test_rotated_degenerate_and_nonfinite_transforms_rejected(
    transform: tuple[float, ...],
) -> None:
    """Only finite, unrotated affines with nonzero steps are accepted."""
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(
            _record(transform=transform), "complex", _metadata()
        )
    assert error.value.reason == "invalid-transform"


def test_radar_and_unknown_domains_rejected() -> None:
    """Radar and missing-domain products get no geographic view."""
    for record in (_record(domain="radar"), _record(domain=None), _record(domain="x")):
        with pytest.raises(GeoreferencedZarrError) as error:
            build_georeferenced_zarr_view(record, "complex", _metadata())
        assert error.value.reason == "unsupported-domain"


def test_geocoded_kind_without_domain_uses_product_grid() -> None:
    """The geocoded kind is inherently geographic when no domain is stored."""
    view = _view(kind="geocoded", domain=None)
    assert view.transform == _TRANSFORM
    assert view.crs == "EPSG:32719"


def test_non_v3_non_array_and_rank_one_metadata_rejected() -> None:
    """Only native Zarr v3 rank-2 arrays are accepted."""
    v2 = _metadata()
    v2["zarr_format"] = 2
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(), "complex", v2)
    assert error.value.reason == "unsupported-format"

    group = _metadata()
    group["node_type"] = "group"
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(), "complex", group)
    assert error.value.reason == "unsupported-node-type"

    rank_one = _metadata()
    rank_one["shape"] = [4]
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(), "complex", rank_one)
    assert error.value.reason == "unsupported-rank"

    length_one = _metadata(shape=(1, 4))
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(shape=(1, 4)), "complex", length_one)
    assert error.value.reason == "grid-too-small"


@pytest.mark.parametrize("data_type", ["bool", "int32", "uint8"])
def test_unsupported_data_types_rejected_without_casting(data_type: str) -> None:
    """Types outside the proven float/complex set stay native-only."""
    metadata = _metadata(data_type=data_type)
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(), "complex", metadata)
    assert error.value.reason == "unsupported-data-type"


def test_unsupported_chunk_grid_and_encoding_rejected() -> None:
    """Only regular grids and configured default/v2 chunk encodings are mapped."""
    metadata = _metadata()
    metadata["chunk_grid"] = {"name": "rectilinear", "configuration": {}}
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(), "complex", metadata)
    assert error.value.reason == "unsupported-chunk-grid"

    metadata = _metadata(encoding={"name": "nested"})
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(), "complex", metadata)
    assert error.value.reason == "unsupported-chunk-key-encoding"


def test_non_scene_shape_must_match_product_shape() -> None:
    """Non-scene assets must be on the product grid."""
    metadata = _metadata(shape=(2, 4))
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(), "complex", metadata)
    assert error.value.reason == "grid-shape-mismatch"


def test_ifg_does_not_apply_looks_or_item_origins_twice() -> None:
    """IFG transforms are already product-windowed; origin fields are ignored."""
    record = _record(
        properties={
            "faninsar:looks": [2, 2],
            "faninsar:row_origin": 4,
            "faninsar:col_origin": 5,
        }
    )
    view = build_georeferenced_zarr_view(record, "complex", _metadata())
    assert view.transform == _TRANSFORM


def test_scene_slc_shifts_parent_affine_once() -> None:
    """A scene SLC uses its unit shape and half-open row/column origin once."""
    assets = {
        "IW1_b0": {
            "href": "IW1_b0.slc.zarr",
            "faninsar:role": "slc",
            "faninsar:rows": 3,
            "faninsar:cols": 4,
            "faninsar:row_origin": 2,
            "faninsar:col_origin": 5,
        }
    }
    record = _record(
        kind="scene",
        shape=(10, 12),
        assets=assets,
        properties={"faninsar:row_origin": 7, "faninsar:col_origin": 9},
    )
    view = build_georeferenced_zarr_view(record, "IW1_b0", _metadata())

    assert view.asset_key == "IW1_b0_georef"
    assert view.shape == (3, 4)
    assert view.transform == (
        8.0,
        0.0,
        288728.0 + 8.0 * 5,
        0.0,
        -200.0,
        6066040.0 - 200.0 * 2,
    )


def test_scene_slc_shape_must_match_metadata() -> None:
    """The native array shape is authoritative for a scene SLC."""
    assets = {
        "IW1_b0": {
            "href": "IW1_b0.slc.zarr",
            "faninsar:rows": 3,
            "faninsar:cols": 5,
            "faninsar:row_origin": 2,
            "faninsar:col_origin": 5,
        }
    }
    record = _record(kind="scene", shape=(10, 12), assets=assets)
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(record, "IW1_b0", _metadata())
    assert error.value.reason == "unit-shape-mismatch"


def test_scene_valid_and_phase_companions_inherit_parent_unit() -> None:
    """Companions use the exact base SLC unit and its shifted affine."""
    assets = {
        "IW1_b0": {
            "href": "IW1_b0.slc.zarr",
            "faninsar:role": "slc",
            "faninsar:rows": 3,
            "faninsar:cols": 4,
            "faninsar:row_origin": 2,
            "faninsar:col_origin": 5,
        },
        "IW1_b0.valid": {"href": "IW1_b0.valid.zarr"},
        "IW1_b0.phase": {"href": "IW1_b0.phase.zarr"},
    }
    record = _record(kind="scene", shape=(10, 12), assets=assets)
    expected = (8.0, 0.0, 288768.0, 0.0, -200.0, 6065640.0)

    for key in ("IW1_b0.valid", "IW1_b0.phase"):
        view = build_georeferenced_zarr_view(record, key, _metadata())
        assert view.asset_key == f"{key}_georef"
        assert view.transform == expected


def test_scene_companion_without_unambiguous_parent_rejected() -> None:
    """A companion with no matching SLC parent is not georeferenced."""
    assets = {"IW1_b0.valid": {"href": "IW1_b0.valid.zarr"}}
    record = _record(kind="scene", shape=(10, 12), assets=assets)
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(record, "IW1_b0.valid", _metadata())
    assert error.value.reason == "companion-unresolved"

    parent = {
        "IW1_b0": {
            "href": "IW1_b0.slc.zarr",
            "faninsar:rows": 3,
            "faninsar:cols": 5,
            "faninsar:row_origin": 2,
            "faninsar:col_origin": 5,
        },
        "IW1_b0.valid": {"href": "IW1_b0.valid.zarr"},
    }
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(
            _record(kind="scene", shape=(10, 12), assets=parent),
            "IW1_b0.valid",
            _metadata(),
        )
    assert error.value.reason == "unit-shape-mismatch"


def test_explicit_complete_asset_projection_is_preferred_without_second_shift() -> None:
    """Complete consistent per-asset projection wins over parent-grid shifting."""
    shifted = (8.0, 0.0, 288768.0, 0.0, -200.0, 6065640.0)
    assets = {
        "IW1_b0": {
            "href": "IW1_b0.slc.zarr",
            "faninsar:rows": 3,
            "faninsar:cols": 4,
            "faninsar:row_origin": 2,
            "faninsar:col_origin": 5,
            "proj:code": "EPSG:32719",
            "proj:transform": list(shifted),
            "proj:shape": [3, 4],
        }
    }
    record = _record(kind="scene", shape=(10, 12), assets=assets)
    view = build_georeferenced_zarr_view(record, "IW1_b0", _metadata())
    assert view.transform == shifted


def test_incomplete_explicit_projection_falls_back_to_product_grid() -> None:
    """Incomplete per-asset projection metadata is ignored for the parent grid."""
    assets = {
        "IW1_b0": {
            "href": "IW1_b0.slc.zarr",
            "faninsar:rows": 3,
            "faninsar:cols": 4,
            "faninsar:row_origin": 2,
            "faninsar:col_origin": 5,
            "proj:code": "EPSG:32719",
        }
    }
    record = _record(kind="scene", shape=(10, 12), assets=assets)
    view = build_georeferenced_zarr_view(record, "IW1_b0", _metadata())
    assert view.transform == (8.0, 0.0, 288768.0, 0.0, -200.0, 6065640.0)


def test_inconsistent_explicit_projection_rejected() -> None:
    """Complete projection metadata must agree with the native shape and grid."""
    metadata = _metadata()
    assets = {
        "complex": {
            "href": "complex.zarr",
            "proj:code": "EPSG:32719",
            "proj:transform": list(_TRANSFORM),
            "proj:shape": [3, 5],
        }
    }
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(assets=assets), "complex", metadata)
    assert error.value.reason == "asset-projection-mismatch"

    assets["complex"]["proj:shape"] = [3, 4]
    assets["complex"]["proj:transform"] = [8.0, 0.0, 0.0, 0.0, -200.0, 0.0]
    with pytest.raises(GeoreferencedZarrError) as error:
        build_georeferenced_zarr_view(_record(assets=assets), "complex", metadata)
    assert error.value.reason == "asset-projection-mismatch"


def test_coordinate_chunks_use_pixel_centres_and_zero_pad_edges() -> None:
    """Coordinate chunks are bounded float64 pixel centres with full padding."""
    shape = (1500, 2500)
    view = build_georeferenced_zarr_view(
        _record(shape=shape), "complex", _metadata(shape=shape)
    )

    x0 = view.coordinate_chunk("x/c/0")
    assert x0 is not None
    assert len(x0) == 1024 * 8
    expected_x0 = 288728.0 + 8.0 * (np.arange(1024, dtype=np.float64) + 0.5)
    np.testing.assert_array_equal(_decoded(x0), expected_x0)

    x2 = view.coordinate_chunk("x/c/2")
    assert x2 is not None
    assert len(x2) == 1024 * 8
    positions = 2048 + np.arange(452, dtype=np.float64)
    expected_x2 = 288728.0 + 8.0 * (positions + 0.5)
    decoded_x2 = _decoded(x2)
    np.testing.assert_array_equal(decoded_x2[:452], expected_x2)
    assert np.all(decoded_x2[452:] == 0.0)

    y0 = view.coordinate_chunk("y/c/0")
    assert y0 is not None
    expected_y0 = 6066040.0 - 200.0 * (np.arange(1024, dtype=np.float64) + 0.5)
    np.testing.assert_array_equal(_decoded(y0), expected_y0)

    y1 = view.coordinate_chunk("y/c/1")
    assert y1 is not None
    positions = 1024 + np.arange(476, dtype=np.float64)
    expected_y1 = 6066040.0 - 200.0 * (positions + 0.5)
    decoded_y1 = _decoded(y1)
    np.testing.assert_array_equal(decoded_y1[:476], expected_y1)
    assert np.all(decoded_y1[476:] == 0.0)

    for relative in (
        "x/c/3",
        "y/c/2",
        "x/c/-1",
        "x/c/01",
        "x/c/0/0",
        "x/c",
        "x/../c/0",
        "x/c/" + "9" * 40,
    ):
        assert view.coordinate_chunk(relative) is None


def test_source_chunk_maps_native_default_and_v2_encodings() -> None:
    """Source chunk requests follow the native encoding and chunk grid."""
    metadata = _metadata(chunk_shape=(2, 3))
    view = build_georeferenced_zarr_view(_record(), "complex", metadata)

    assert view.source_chunk("values/c/0/0") == "c/0/0"
    assert view.source_chunk("values/c/1/1") == "c/1/1"
    for relative in (
        "values/c/2/0",
        "values/c/0/2",
        "values/0.0",
        "values/zarr.json",
        "values/c/0",
        "values/c/0/0/0",
        "values/../c/0/0",
        "values//c/0/0",
        "values/c/00/0",
        "values",
    ):
        assert view.source_chunk(relative) is None

    dotted = build_georeferenced_zarr_view(
        _record(),
        "complex",
        _metadata(
            chunk_shape=(2, 3),
            encoding={"name": "default", "configuration": {"separator": "."}},
        ),
    )
    assert dotted.source_chunk("values/0.0") == "0.0"
    assert dotted.source_chunk("values/1.1") == "1.1"
    assert dotted.source_chunk("values/c/0/0") is None

    native_v2 = build_georeferenced_zarr_view(
        _record(),
        "complex",
        _metadata(chunk_shape=(2, 3), encoding={"name": "v2"}),
    )
    assert native_v2.source_chunk("values/0.0") == "0.0"
    assert native_v2.source_chunk("values/c/0/0") is None


def test_view_is_frozen_and_returned_metadata_is_detached() -> None:
    """The descriptor cannot be mutated and metadata copies are independent."""
    view = _view()
    with pytest.raises(FrozenInstanceError):
        view.shape = (1, 1)  # type: ignore[misc]
    assert isinstance(view._values_metadata, MappingProxyType)

    first = view.metadata("values/zarr.json")
    assert first is not None
    first["shape"].append(99)
    first["attributes"]["_CRS"]["url"] = "changed"
    second = view.metadata("values/zarr.json")
    assert second is not None
    assert second["shape"] == [3, 4]
    assert second["attributes"]["_CRS"] == {"url": _CRS_URL}


def test_tiny_view_roundtrips_with_zarr_python(tmp_path: Path) -> None:
    """The generated metadata and chunks open as a standard Zarr v3 group."""
    import zarr

    source = tmp_path / "source.zarr"
    array = zarr.open_array(
        store=str(source),
        mode="w",
        shape=(3, 4),
        chunks=(3, 4),
        dtype="float32",
    )
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    array[...] = original
    metadata = json.loads((source / "zarr.json").read_text(encoding="utf-8"))

    view = build_georeferenced_zarr_view(_record(), "complex", metadata)
    root = tmp_path / "view.zarr"
    for relative in (
        "zarr.json",
        "values/zarr.json",
        "x/zarr.json",
        "y/zarr.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(view.metadata(relative), sort_keys=True), encoding="utf-8"
        )

    chunk = root / "values" / "c" / "0" / "0"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "c" / "0" / "0", chunk)
    for axis in ("x", "y"):
        data = view.coordinate_chunk(f"{axis}/c/0")
        assert data is not None
        target = root / axis / "c" / "0"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    group = zarr.open_group(str(root), mode="r")
    assert set(group.array_keys()) == {"values", "x", "y"}
    values = group["values"]
    assert values.shape == (3, 4)
    assert values.dtype == np.dtype("float32")
    assert values.metadata.dimension_names == ("y", "x")
    assert values.attrs.asdict()["_CRS"] == {"url": _CRS_URL}
    np.testing.assert_array_equal(values[:], original)
    np.testing.assert_array_equal(
        group["x"][:],
        288728.0 + 8.0 * (np.arange(4, dtype=np.float64) + 0.5),
    )
    np.testing.assert_array_equal(
        group["y"][:],
        6066040.0 - 200.0 * (np.arange(3, dtype=np.float64) + 0.5),
    )

"""Normalization from committed products to public STAC Items (PROPOSAL-0089)."""

# ruff: noqa: INP001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pystac
import pytest

from faninsar.io.storage.product import Product, publish_product
from faninsar.io.storage._zarr_stac import write_zarr_array
from faninsar.io.stac.normalize import normalize_product
from faninsar.io.stac.record import (
    PROJECTION_EXTENSION,
    StacNormalizationError,
    StacRecord,
    public_item_id,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _write_id(number: int) -> str:
    """Return a deterministic 26-character ULID-shaped write id."""
    return f"{number:026d}"


def _direct_product(
    work_dir: Path,
    relative: str,
    *,
    kind: str,
    write_id: str,
    properties: Mapping[str, object] | None = None,
    inputs: Mapping[str, str] | None = None,
    assets: Mapping[str, Mapping[str, object]] | None = None,
) -> Product:
    """Create a metadata-only Product under ``work_dir`` for local tests."""
    path = work_dir / relative
    path.mkdir(parents=True)
    item = {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": f"{kind}-{path.name}",
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": None,
            **dict(properties or {}),
            "faninsar:kind": kind,
            "faninsar:write_id": write_id,
            "faninsar:inputs": dict(inputs or {}),
        },
        "assets": {name: dict(value) for name, value in (assets or {}).items()},
        "links": [],
    }
    (path / "item.json").write_text(json.dumps(item), encoding="utf-8")
    return Product(path=path, item=item)


def _commit_product(
    work_dir: Path,
    relative: str,
    *,
    kind: str,
    properties: Mapping[str, object],
    arrays: Mapping[str, np.ndarray] | None = None,
    inputs: Mapping[str, object] | None = None,
):
    """Commit a real product with tiny Zarr arrays into ``work_dir``."""

    def write_payloads(staging: Path) -> dict[str, dict[str, object]]:
        assets: dict[str, dict[str, object]] = {}
        for name, value in (arrays or {"data": np.arange(6).reshape(2, 3)}).items():
            write_zarr_array(staging / f"{name}.zarr", np.asarray(value))
            assets[name] = {"href": f"{name}.zarr"}
        return assets

    return publish_product(
        work_dir / relative,
        kind=kind,
        properties=properties,
        write_payloads=write_payloads,
        inputs=inputs,
    )


def test_committed_scene_normalizes_to_one_public_feature(tmp_path: Path) -> None:
    """A real committed radar scene becomes one normalized STAC Feature."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _commit_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 3],
            "faninsar:grid_identity": "radar-grid-identity",
        },
    )
    canonical_item = (product.path / "item.json").read_bytes()

    record = normalize_product(product, work_dir=work_dir)

    assert isinstance(record, StacRecord)
    assert record.public_id == (f"scene-coreg-20200101-{product.write_id}")
    assert record.write_id == product.write_id
    assert record.kind == "scene"
    assert record.family == "scenes"
    assert record.source_relative_path == "coreg/20200101"
    assert record.source_item_path == product.path / "item.json"
    assert record.collection_id is None
    assert record.bbox is None
    assert record.temporal.precision == "day"
    assert record.asset_paths["data"] == product.path / "data.zarr"

    item = record.as_item()
    assert item["type"] == "Feature"
    assert item["stac_version"] == "1.0.0"
    assert item["id"] == record.public_id
    assert item["geometry"] is None
    assert "bbox" not in item
    assert item["assets"]["data"]["href"] == "coreg/20200101/data.zarr"
    assert item["properties"]["faninsar:family"] == "scenes"
    assert item["properties"]["faninsar:source_item"] == ("coreg/20200101/item.json")
    assert item["properties"]["faninsar:source_item_id"] == "scene-20200101"
    assert item["properties"]["faninsar:temporal_precision"] == "day"
    assert item["properties"]["start_datetime"] == "2020-01-01T00:00:00.000000Z"
    assert item["properties"]["end_datetime"] == "2020-01-01T23:59:59.999999Z"
    assert item["properties"]["datetime"] is None
    assert "proj:code" not in item["properties"]
    assert "proj:transform" not in item["properties"]
    json.dumps(item, allow_nan=False)

    with pytest.raises(TypeError):
        record.item["id"] = "replacement"  # type: ignore[index]

    copied = record.as_item()
    copied["properties"]["faninsar:family"] = "changed"
    assert record.as_item()["properties"]["faninsar:family"] == "scenes"
    assert (product.path / "item.json").read_bytes() == canonical_item


def test_committed_zarr_asset_gains_zarr_media_type(tmp_path: Path) -> None:
    """A real committed Zarr directory asset is typed for STAC clients."""
    import zarr

    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _commit_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 3],
            "faninsar:grid_identity": "radar-grid-identity",
        },
    )
    canonical = json.loads((product.path / "item.json").read_text(encoding="utf-8"))

    record = normalize_product(product, work_dir=work_dir)
    asset = record.as_item()["assets"]["data"]

    assert "type" not in canonical["assets"]["data"]
    assert record.asset_paths["data"].is_dir()
    assert asset["type"] == "application/vnd+zarr"
    assert asset["href"] == "coreg/20200101/data.zarr"
    array = zarr.open_array(store=str(record.asset_paths["data"]), mode="r")
    assert array.shape == (2, 3)


def test_non_zarr_directory_asset_is_not_typed_as_zarr(tmp_path: Path) -> None:
    """A plain asset directory without Zarr metadata keeps no invented type."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "scene/plain",
        kind="scene",
        write_id=_write_id(7),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 3],
            "faninsar:grid_identity": "radar-grid-identity",
        },
        assets={"data": {"href": "data"}},
    )
    (product.path / "data").mkdir()
    (product.path / "data" / "note.txt").write_text("not zarr", encoding="utf-8")

    asset = normalize_product(product, work_dir=work_dir).as_item()["assets"]["data"]

    assert "type" not in asset


def test_geo_product_publishes_true_geometry_and_projection_v2(
    tmp_path: Path,
) -> None:
    """A geo Item gets a real WGS84 ring, numeric bbox, and projection fields."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _commit_product(
        work_dir,
        "ifg/20200101_20200102",
        kind="ifg",
        properties={
            "faninsar:pair": ["20200101", "20200102"],
            "faninsar:domain": "geo",
            "proj:code": "EPSG:4326",
            "proj:transform": [0.1, 0.0, -10.0, 0.0, -0.1, 10.0],
            "faninsar:shape": [5, 6],
            "faninsar:grid_identity": "geo-grid",
        },
        arrays={"phase": np.zeros((5, 6), dtype=np.float32)},
    )

    record = normalize_product(product, work_dir=work_dir)
    item = record.as_item()

    assert record.family == "interferograms"
    assert record.bbox == pytest.approx((-10.0, 9.5, -9.4, 10.0))
    assert item["bbox"] == pytest.approx([-10.0, 9.5, -9.4, 10.0])
    assert item["geometry"] == {
        "type": "Polygon",
        "coordinates": [
            [
                [-10.0, 10.0],
                [-9.4, 10.0],
                [-9.4, 9.5],
                [-10.0, 9.5],
                [-10.0, 10.0],
            ]
        ],
    }
    properties = item["properties"]
    assert properties["proj:code"] == "EPSG:4326"
    assert properties["proj:shape"] == [5, 6]
    assert properties["proj:transform"] == [
        0.1,
        0.0,
        -10.0,
        0.0,
        -0.1,
        10.0,
        0.0,
        0.0,
        1.0,
    ]
    assert PROJECTION_EXTENSION in item["stac_extensions"]
    assert properties["start_datetime"] == "2020-01-01T00:00:00.000000Z"
    assert properties["end_datetime"] == "2020-01-02T23:59:59.999999Z"
    pystac.Item.from_dict(item)


def test_geocoded_inherits_source_family_time_and_pair_not_reference(
    tmp_path: Path,
) -> None:
    """A geocoded child uses its source observation and its own true grid."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    source = _direct_product(
        work_dir,
        "ifg/20200101_20200103",
        kind="ifg",
        write_id=_write_id(1),
        properties={
            "faninsar:pair": ["20200101", "20200103"],
            "faninsar:domain": "radar",
            "faninsar:shape": [4, 4],
            "faninsar:grid_identity": "radar-grid",
        },
    )
    reference = _direct_product(
        work_dir,
        "coreg/20200105",
        kind="scene",
        write_id=_write_id(2),
        properties={
            "faninsar:date": "20200105",
            "faninsar:domain": "radar",
            "faninsar:shape": [4, 4],
            "faninsar:grid_identity": "radar-grid",
        },
    )
    geocoded = _direct_product(
        work_dir,
        "ifg/20200101_20200103/geocoded/coherence",
        kind="geocoded",
        write_id=_write_id(3),
        properties={
            "faninsar:name": "coherence",
            "faninsar:geocode": {"kernel": "nearest"},
            "faninsar:grid": {
                "crs": "EPSG:4326",
                "transform": [0.2, 0.0, -5.0, 0.0, -0.2, 5.0],
                "shape": [2, 3],
            },
        },
        inputs={"source": source.write_id, "reference": reference.write_id},
        assets={"values": {"href": "values.tif", "type": "image/tiff"}},
    )
    products = {product.write_id: product for product in (source, reference, geocoded)}

    record = normalize_product(
        geocoded, work_dir=work_dir, products_by_write_id=products
    )
    item = record.as_item()

    assert record.family == "interferograms"
    assert record.temporal.precision == "day"
    assert item["properties"]["start_datetime"] == ("2020-01-01T00:00:00.000000Z")
    assert item["properties"]["end_datetime"] == ("2020-01-03T23:59:59.999999Z")
    assert item["properties"]["faninsar:pair"] == ["20200101", "20200103"]
    assert item["properties"]["faninsar:inputs"] == {
        "source": source.write_id,
        "reference": reference.write_id,
    }
    assert item["properties"]["faninsar:geocode"] == {"kernel": "nearest"}
    assert item["properties"]["faninsar:source_item_id"] == ("geocoded-coherence")
    assert item["assets"]["values"]["type"] == "image/tiff"
    assert item["assets"]["values"]["href"] == (
        "ifg/20200101_20200103/geocoded/coherence/values.tif"
    )
    assert record.bbox == pytest.approx((-5.0, 4.6, -4.4, 5.0))
    assert item["geometry"] is not None


def test_geocoded_missing_source_and_cycle_are_excluded(tmp_path: Path) -> None:
    """A geocoded child never guesses a missing or cyclic source."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    geocoded = _direct_product(
        work_dir,
        "geocoded/missing",
        kind="geocoded",
        write_id=_write_id(10),
        properties={
            "faninsar:grid": {
                "crs": "EPSG:4326",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                "height": 1,
                "width": 1,
            }
        },
        inputs={"source": _write_id(11)},
    )
    with pytest.raises(StacNormalizationError) as missing:
        normalize_product(
            geocoded,
            work_dir=work_dir,
            products_by_write_id={geocoded.write_id: geocoded},
        )
    assert missing.value.reason == "missing-source"
    assert _write_id(11) in missing.value.detail

    first = _direct_product(
        work_dir,
        "geocoded/first",
        kind="geocoded",
        write_id=_write_id(20),
        properties={
            "faninsar:grid": {
                "crs": "EPSG:4326",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                "shape": [1, 1],
            }
        },
        inputs={"source": _write_id(21)},
    )
    second = _direct_product(
        work_dir,
        "geocoded/second",
        kind="geocoded",
        write_id=_write_id(21),
        properties={
            "faninsar:grid": {
                "crs": "EPSG:4326",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                "shape": [1, 1],
            }
        },
        inputs={"source": _write_id(20)},
    )
    with pytest.raises(StacNormalizationError) as cyclic:
        normalize_product(
            first,
            work_dir=work_dir,
            products_by_write_id={
                first.write_id: first,
                second.write_id: second,
            },
        )
    assert cyclic.value.reason == "cyclic-inheritance"


def test_unwrap_inherits_temporal_union_from_committed_inputs(
    tmp_path: Path,
) -> None:
    """A dependent without its own dates inherits the union of input coverage."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    day_product = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(30),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
            "faninsar:grid_identity": "radar-grid",
        },
    )
    precise_product = _direct_product(
        work_dir,
        "coreg/20200102",
        kind="scene",
        write_id=_write_id(31),
        properties={
            "datetime": "2020-01-02T12:00:00Z",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
            "faninsar:grid_identity": "radar-grid",
        },
    )
    dependent = _direct_product(
        work_dir,
        "timeseries/result",
        kind="timeseries",
        write_id=_write_id(32),
        properties={"faninsar:domain": "radar", "faninsar:shape": [2, 2]},
        inputs={
            "first": day_product.write_id,
            "second": precise_product.write_id,
        },
    )
    products = {
        product.write_id: product
        for product in (day_product, precise_product, dependent)
    }

    record = normalize_product(
        dependent, work_dir=work_dir, products_by_write_id=products
    )

    assert record.temporal.precision == "timestamp"
    assert record.as_item()["properties"]["start_datetime"] == ("2020-01-01T00:00:00Z")
    assert record.as_item()["properties"]["end_datetime"] == ("2020-01-02T12:00:00Z")
    assert record.as_item()["properties"]["datetime"] is None


@pytest.mark.parametrize(
    ("properties", "expected_precision", "expected_start", "expected_end"),
    [
        (
            {"faninsar:dates": ["20200103", "20200101"]},
            "day",
            "2020-01-01T00:00:00.000000Z",
            "2020-01-03T23:59:59.999999Z",
        ),
        (
            {
                "faninsar:pair": ["20200101", "20200102"],
                "faninsar:date": "20191231",
            },
            "day",
            "2019-12-31T00:00:00.000000Z",
            "2020-01-02T23:59:59.999999Z",
        ),
        (
            {
                "datetime": "2020-01-02T12:34:56Z",
                "faninsar:date": "20200101",
            },
            "timestamp",
            "2020-01-02T12:34:56Z",
            "2020-01-02T12:34:56Z",
        ),
        (
            {
                "datetime": "2020-01-01T00:00:00Z",
                "faninsar:date": "20200101",
            },
            "day",
            "2020-01-01T00:00:00.000000Z",
            "2020-01-01T23:59:59.999999Z",
        ),
        (
            {"datetime": "2020-01-01T00:00:00Z"},
            "timestamp",
            "2020-01-01T00:00:00Z",
            "2020-01-01T00:00:00Z",
        ),
        (
            {
                "start_datetime": "2020-01-01T00:00:00Z",
                "end_datetime": "2020-01-02T00:00:00Z",
            },
            "timestamp",
            "2020-01-01T00:00:00Z",
            "2020-01-02T00:00:00Z",
        ),
    ],
)
def test_temporal_variants_use_persisted_evidence(
    tmp_path: Path,
    properties: Mapping[str, object],
    expected_precision: str,
    expected_start: str,
    expected_end: str,
) -> None:
    """Date, dates, pair, and precise timestamp metadata are not conflated."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(40),
        properties={
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
            "faninsar:grid_identity": "radar-grid",
            **properties,
        },
    )

    record = normalize_product(product, work_dir=work_dir)
    item_properties = record.as_item()["properties"]

    assert record.temporal.precision == expected_precision
    assert item_properties["faninsar:temporal_precision"] == (expected_precision)
    if expected_precision == "day":
        assert item_properties["datetime"] is None
        assert item_properties["start_datetime"] == expected_start
        assert item_properties["end_datetime"] == expected_end
    elif expected_start == expected_end:
        assert item_properties["datetime"] == expected_start
        assert "start_datetime" not in item_properties
        assert "end_datetime" not in item_properties
    else:
        assert item_properties["datetime"] is None
        assert item_properties["start_datetime"] == expected_start
        assert item_properties["end_datetime"] == expected_end


def test_pair_dates_resolve_through_primary_and_secondary_scene_inputs(
    tmp_path: Path,
) -> None:
    """A non-date pair id inherits committed scene dates through its roles."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    primary = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(50),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
            "faninsar:grid_identity": "radar-grid",
        },
    )
    secondary = _direct_product(
        work_dir,
        "coreg/20200107",
        kind="scene",
        write_id=_write_id(51),
        properties={
            "faninsar:date": "20200107",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
            "faninsar:grid_identity": "radar-grid",
        },
    )
    product = _direct_product(
        work_dir,
        "ifg/pair",
        kind="ifg",
        write_id=_write_id(52),
        properties={
            "faninsar:pair": ["PRIMARY", "SECONDARY"],
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
            "faninsar:grid_identity": "radar-grid",
        },
        inputs={"primary": primary.write_id, "secondary": secondary.write_id},
    )
    products = {product.write_id: product for product in (primary, secondary, product)}

    record = normalize_product(
        product, work_dir=work_dir, products_by_write_id=products
    )

    assert record.temporal.precision == "day"
    assert record.as_item()["properties"]["start_datetime"] == (
        "2020-01-01T00:00:00.000000Z"
    )
    assert record.as_item()["properties"]["end_datetime"] == (
        "2020-01-07T23:59:59.999999Z"
    )
    assert record.as_item()["properties"]["faninsar:pair"] == [
        "PRIMARY",
        "SECONDARY",
    ]


def test_mask_domain_and_both_grid_shape_spellings(tmp_path: Path) -> None:
    """A geo mask resolves its own grid and keeps the masks family."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "masks/water",
        kind="mask",
        write_id=_write_id(60),
        properties={
            "faninsar:date": "20200101",
            "faninsar:mask_domain": "geo",
            "faninsar:grid": {
                "crs": "EPSG:4326",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
                "shape": [2, 2],
                "height": 2,
                "width": 2,
            },
        },
    )

    record = normalize_product(product, work_dir=work_dir)

    assert record.family == "masks"
    assert record.as_item()["properties"]["proj:shape"] == [2, 2]
    assert record.bbox == pytest.approx((0.0, 0.0, 2.0, 2.0))


def test_unknown_geometry_stays_unplaced_without_projection_fields(
    tmp_path: Path,
) -> None:
    """A product with no domain and no complete grid remains unplaced."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(61),
        properties={
            "faninsar:date": "20200101",
            "faninsar:shape": [2, 2],
            "proj:code": "EPSG:4326",
            "proj:transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        },
    )

    record = normalize_product(product, work_dir=work_dir)
    item = record.as_item()

    assert item["geometry"] is None
    assert "bbox" not in item
    assert not any(key.startswith("proj:") for key in item["properties"])


def test_unsupported_kind_and_missing_temporal_are_excluded(
    tmp_path: Path,
) -> None:
    """Internal kinds and undated products fail with precise reasons."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    internal = _direct_product(
        work_dir,
        "layout/merge",
        kind="merge_layout",
        write_id=_write_id(70),
        properties={"faninsar:domain": "radar"},
    )
    with pytest.raises(StacNormalizationError) as unsupported:
        normalize_product(internal, work_dir=work_dir)
    assert unsupported.value.reason == "unsupported-kind"

    undated = _direct_product(
        work_dir,
        "coreg/unknown",
        kind="scene",
        write_id=_write_id(71),
        properties={"faninsar:domain": "radar", "faninsar:shape": [2, 2]},
    )
    with pytest.raises(StacNormalizationError) as missing:
        normalize_product(undated, work_dir=work_dir)
    assert missing.value.reason == "missing-temporal"


def test_public_ids_distinguish_equal_basenames_and_long_slugs(
    tmp_path: Path,
) -> None:
    """Kind, locator, and write id make public identity workspace-unique."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    first = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(80),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )
    second = _direct_product(
        work_dir,
        "merged/coreg/20200101",
        kind="scene",
        write_id=_write_id(81),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )

    first_record = normalize_product(first, work_dir=work_dir)
    second_record = normalize_product(second, work_dir=work_dir)

    assert first_record.public_id != second_record.public_id
    assert first_record.public_id.endswith(_write_id(80))
    assert second_record.public_id.endswith(_write_id(81))
    long_id = public_item_id(
        kind="scene",
        source_relative_path="a-very-long-directory-name/" * 6,
        write_id=_write_id(82),
    )
    assert long_id.endswith(_write_id(82))
    assert long_id == public_item_id(
        kind="scene",
        source_relative_path="a-very-long-directory-name/" * 6,
        write_id=_write_id(82),
    )
    assert (
        public_item_id(kind="scene", source_relative_path="", write_id=_write_id(83))
        == f"scene-root-{_write_id(83)}"
    )


def test_unsafe_assets_and_outside_products_are_excluded(
    tmp_path: Path,
) -> None:
    """Remote, traversing, absolute, and symlinked assets never register."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    for number, href in enumerate(
        ("../escape.zarr", "https://example.test/data.zarr", "/etc/passwd")
    ):
        product = _direct_product(
            work_dir,
            f"coreg/unsafe-{number}",
            kind="scene",
            write_id=_write_id(90 + number),
            properties={
                "faninsar:date": "20200101",
                "faninsar:domain": "radar",
                "faninsar:shape": [2, 2],
            },
            assets={"data": {"href": href}},
        )
        with pytest.raises(StacNormalizationError) as unsafe:
            normalize_product(product, work_dir=work_dir)
        assert unsafe.value.reason == "unsafe-asset"

    linked = _direct_product(
        work_dir,
        "coreg/unsafe-link",
        kind="scene",
        write_id=_write_id(99),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
        assets={"data": {"href": "data.zarr"}},
    )
    (linked.path / "data.zarr").symlink_to(tmp_path / "outside.zarr")
    with pytest.raises(StacNormalizationError) as symlinked:
        normalize_product(linked, work_dir=work_dir)
    assert symlinked.value.reason == "unsafe-asset"

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_product = _direct_product(
        outside,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(100),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )
    with pytest.raises(StacNormalizationError) as escaped:
        normalize_product(outside_product, work_dir=work_dir)
    assert escaped.value.reason == "outside-work-dir"


def test_normalization_never_reads_payloads_or_rewrites_canonical_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata-only normalization leaves canonical bytes and readers untouched."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _commit_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 3],
            "faninsar:grid_identity": "radar-grid",
        },
        arrays={"data": np.arange(6).reshape(2, 3)},
    )
    canonical_item = (product.path / "item.json").read_bytes()
    payload = (product.path / "data.zarr").joinpath("zarr.json").read_bytes()

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("normalization must not read a payload")

    monkeypatch.setattr(Product, "read", forbidden)
    monkeypatch.setattr("faninsar.io.storage._zarr_stac.read_zarr_array", forbidden)
    monkeypatch.setattr(
        "faninsar.io.storage._zarr_stac.inspect_zarr_array_metadata", forbidden
    )

    record = normalize_product(product, work_dir=work_dir)

    assert record.kind == "scene"
    assert (product.path / "item.json").read_bytes() == canonical_item
    assert (product.path / "data.zarr" / "zarr.json").read_bytes() == payload


def test_geocoded_requires_its_own_complete_geographic_grid(
    tmp_path: Path,
) -> None:
    """A geocoded child without a complete faninsar:grid is invalid."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    source = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(110),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )
    geocoded = _direct_product(
        work_dir,
        "geocoded/incomplete",
        kind="geocoded",
        write_id=_write_id(111),
        properties={"faninsar:grid": {"crs": "EPSG:4326", "shape": [1, 1]}},
        inputs={"source": source.write_id},
    )
    with pytest.raises(StacNormalizationError) as invalid:
        normalize_product(
            geocoded,
            work_dir=work_dir,
            products_by_write_id={
                source.write_id: source,
                geocoded.write_id: geocoded,
            },
        )
    assert invalid.value.reason == "invalid-item"


def test_malformed_pair_and_geo_grid_are_invalid_items(tmp_path: Path) -> None:
    """Malformed pair entries and incomplete geo grids are never guessed."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    malformed = _direct_product(
        work_dir,
        "ifg/malformed",
        kind="ifg",
        write_id=_write_id(120),
        properties={
            "faninsar:date": "20200101",
            "faninsar:pair": ["", "20200102"],
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )
    with pytest.raises(StacNormalizationError) as bad_pair:
        normalize_product(malformed, work_dir=work_dir)
    assert bad_pair.value.reason == "invalid-item"

    missing_geo = _direct_product(
        work_dir,
        "ifg/missing-geo",
        kind="ifg",
        write_id=_write_id(121),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "geo",
            "faninsar:shape": [2, 2],
        },
    )
    with pytest.raises(StacNormalizationError) as bad_geo:
        normalize_product(missing_geo, work_dir=work_dir)
    assert bad_geo.value.reason == "invalid-item"

    disagreeing = _direct_product(
        work_dir,
        "masks/disagree",
        kind="mask",
        write_id=_write_id(122),
        properties={
            "faninsar:date": "20200101",
            "faninsar:mask_domain": "geo",
            "faninsar:grid": {
                "crs": "EPSG:4326",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                "shape": [2, 2],
                "height": 3,
                "width": 2,
            },
        },
    )
    with pytest.raises(StacNormalizationError) as mismatch:
        normalize_product(disagreeing, work_dir=work_dir)
    assert mismatch.value.reason == "invalid-item"


def test_canonical_checksums_and_unknown_properties_are_preserved(
    tmp_path: Path,
) -> None:
    """Normalization copies checksums and unknown scientific metadata verbatim."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(130),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
            "faninsar:custom": {"nested": [1, 2, 3]},
        },
        assets={
            "data": {
                "href": "data.zarr",
                "file:checksum": "1220deadbeef",
                "faninsar:role": "slc",
            }
        },
    )

    record = normalize_product(product, work_dir=work_dir)
    item = record.as_item()

    assert item["assets"]["data"]["file:checksum"] == "1220deadbeef"
    assert item["assets"]["data"]["faninsar:role"] == "slc"
    assert item["assets"]["data"]["href"] == "coreg/20200101/data.zarr"
    assert item["properties"]["faninsar:custom"] == {"nested": [1, 2, 3]}


def test_nine_element_projection_transform_is_normalized(tmp_path: Path) -> None:
    """A v2 3x3 projection transform is accepted and re-emitted as nine values."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "ifg/nine",
        kind="ifg",
        write_id=_write_id(140),
        properties={
            "faninsar:dates": ["20200101", "20200102"],
            "faninsar:domain": "geo",
            "proj:code": "EPSG:4326",
            "proj:transform": [
                1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                2.0,
                0.0,
                0.0,
                1.0,
            ],
            "faninsar:shape": [2, 2],
        },
    )

    record = normalize_product(product, work_dir=work_dir)

    assert record.as_item()["properties"]["proj:transform"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        2.0,
        0.0,
        0.0,
        1.0,
    ]
    assert record.bbox == pytest.approx((0.0, 0.0, 2.0, 2.0))


@pytest.mark.parametrize(
    ("kind", "family"),
    [
        ("scene", "scenes"),
        ("ifg", "interferograms"),
        ("unwrap", "interferograms"),
        ("complex_ion", "interferograms"),
        ("unwrap_ion", "interferograms"),
        ("ion", "interferograms"),
        ("ion_corrected", "interferograms"),
        ("timeseries", "time-series"),
        ("mask", "masks"),
    ],
)
def test_known_kinds_map_to_their_logical_family(
    tmp_path: Path, kind: str, family: str
) -> None:
    """Every public kind maps to the fixed logical family table."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    properties: dict[str, object] = {"faninsar:date": "20200101"}
    if kind == "mask":
        properties["faninsar:mask_domain"] = "radar"
    product = _direct_product(
        work_dir,
        f"products/{kind}",
        kind=kind,
        write_id=_write_id(150),
        properties=properties,
    )

    record = normalize_product(product, work_dir=work_dir)

    assert record.family == family
    assert record.as_item()["properties"]["faninsar:family"] == family


def test_precise_timestamp_takes_precedence_over_malformed_date_field(
    tmp_path: Path,
) -> None:
    """A trusted non-midnight timestamp is not invalidated by a date label."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(160),
        properties={
            "datetime": "2020-01-02T12:34:56Z",
            "faninsar:date": "not-a-calendar-day",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )

    record = normalize_product(product, work_dir=work_dir)

    assert record.temporal.precision == "timestamp"
    assert record.as_item()["properties"]["datetime"] == "2020-01-02T12:34:56Z"


def test_date_only_datetime_and_bad_extensions_are_invalid(tmp_path: Path) -> None:
    """RFC 3339 requires a time separator and extensions must be a list."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    date_only = _direct_product(
        work_dir,
        "coreg/date-only",
        kind="scene",
        write_id=_write_id(161),
        properties={
            "datetime": "2020-01-01",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )
    with pytest.raises(StacNormalizationError) as bad_time:
        normalize_product(date_only, work_dir=work_dir)
    assert bad_time.value.reason == "invalid-item"

    bad_extensions = _direct_product(
        work_dir,
        "coreg/bad-extensions",
        kind="scene",
        write_id=_write_id(162),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )
    bad_extensions.item["stac_extensions"] = "not-a-list"  # type: ignore[assignment]
    with pytest.raises(StacNormalizationError) as bad_list:
        normalize_product(bad_extensions, work_dir=work_dir)
    assert bad_list.value.reason == "invalid-item"


def test_product_behind_a_symlinked_directory_is_excluded(tmp_path: Path) -> None:
    """A product reached through a symlinked directory is outside the workspace."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    real = work_dir / "real"
    real.mkdir()
    (work_dir / "alias").symlink_to(real, target_is_directory=True)
    product = _direct_product(
        work_dir,
        "alias/coreg/20200101",
        kind="scene",
        write_id=_write_id(163),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )

    with pytest.raises(StacNormalizationError) as linked:
        normalize_product(product, work_dir=work_dir)
    assert linked.value.reason == "outside-work-dir"


def test_disabled_inheritance_does_not_resolve_pair_inputs(tmp_path: Path) -> None:
    """``products_by_write_id=None`` disables pair-date inheritance entirely."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    primary = _direct_product(
        work_dir,
        "coreg/20200101",
        kind="scene",
        write_id=_write_id(170),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
    )
    product = _direct_product(
        work_dir,
        "ifg/pair",
        kind="ifg",
        write_id=_write_id(171),
        properties={
            "faninsar:pair": ["PRIMARY", "SECONDARY"],
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
        inputs={"primary": primary.write_id},
    )

    with pytest.raises(StacNormalizationError) as missing:
        normalize_product(product, work_dir=work_dir)

    assert missing.value.reason == "missing-temporal"


def test_null_byte_asset_href_is_unsafe(tmp_path: Path) -> None:
    """A null byte in an asset href is rejected as unsafe before any lstat."""
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    product = _direct_product(
        work_dir,
        "coreg/null",
        kind="scene",
        write_id=_write_id(172),
        properties={
            "faninsar:date": "20200101",
            "faninsar:domain": "radar",
            "faninsar:shape": [2, 2],
        },
        assets={"data": {"href": "data\x00.zarr"}},
    )

    with pytest.raises(StacNormalizationError) as unsafe:
        normalize_product(product, work_dir=work_dir)

    assert unsafe.value.reason == "unsafe-asset"

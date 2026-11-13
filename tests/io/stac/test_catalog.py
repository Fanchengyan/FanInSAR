"""Index discovery, shared rendering, and portable persistence (P0089)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import faninsar.io.stac.catalog as catalog_module
import faninsar.io.stac.index as index_module
from faninsar.io.stac.catalog import write_portable_catalog
from faninsar.io.stac.index import (
    IndexExclusion,
    StacCollection,
    StacIndex,
    build_index,
)
from faninsar.io.stac.record import (
    StacNormalizationError,
    StacRecord,
    TemporalCoverage,
)
from faninsar.io.stac.render import (
    LinkTargets,
    render_collection,
    render_family_catalog,
    render_item,
    render_root_catalog,
)
from faninsar.io.storage.product import Product, new_write_id

_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 1, 1, 23, 59, 59, 999999, tzinfo=UTC)
_SQUARE = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
}


def _product_item(
    kind: str,
    write_id: str,
    *,
    date: str | None = "20240101",
    inputs: dict[str, str] | None = None,
    assets: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build one canonical product Item for the injection seam."""
    properties: dict[str, object] = {
        "datetime": None,
        "faninsar:kind": kind,
        "faninsar:write_id": write_id,
    }
    if date is not None:
        properties["faninsar:date"] = date
    if inputs:
        properties["faninsar:inputs"] = dict(inputs)
    return {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": f"{kind}-{write_id}",
        "geometry": None,
        "bbox": None,
        "properties": properties,
        "assets": {
            name: {"href": href, "roles": ["data"]}
            for name, href in (assets or {"data": "data.bin"}).items()
        },
        "links": [],
    }


def _write_item(path: Path, item: dict[str, object]) -> None:
    """Write one canonical Item into a candidate directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "item.json").write_text(
        json.dumps(item, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _record(
    work_dir: Path,
    public_id: str,
    *,
    family: str = "scenes",
    kind: str = "scene",
    bbox: tuple[float, float, float, float] | None = None,
    geometry: dict[str, object] | None = None,
    collection_id: str | None = None,
    assets: dict[str, str] | None = None,
    write_id: str | None = None,
    start: datetime = _START,
    end: datetime = _END,
) -> StacRecord:
    """Build one synthetic normalized record with real asset files."""
    asset_paths: dict[str, Path] = {}
    item_assets: dict[str, dict[str, object]] = {}
    for key, relative in (assets or {"data": f"{public_id}/data.bin"}).items():
        path = work_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
        asset_paths[key] = path
        item_assets[key] = {
            "href": relative,
            "type": "application/octet-stream",
            "roles": ["data"],
            "checksum:multihash": "kept",
        }
    item: dict[str, object] = {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": f"{kind}-{public_id}",
        "geometry": geometry,
        "properties": {
            "datetime": None,
            "faninsar:kind": kind,
            "faninsar:family": family,
            "faninsar:write_id": write_id or public_id,
            "faninsar:source_item": f"{public_id}/item.json",
            "custom:note": "preserved",
        },
        "assets": item_assets,
        "links": [],
    }
    if bbox is not None:
        item["bbox"] = list(bbox)
    return StacRecord(
        public_id=public_id,
        write_id=write_id or public_id,
        kind=kind,
        family=family,
        source_item_path=work_dir / public_id / "item.json",
        source_relative_path=public_id,
        collection_id=collection_id,
        temporal=TemporalCoverage(start=start, end=end, precision="timestamp"),
        bbox=bbox,
        item=item,
        asset_paths=asset_paths,
    )


def _collection(
    family: str,
    records: tuple[StacRecord, ...],
) -> StacCollection:
    """Build the geographic Collection from geographic members."""
    members = [record for record in records if record.bbox is not None]
    return StacCollection(
        collection_id=f"{family}-geographic",
        family=family,
        item_ids=tuple(sorted(record.public_id for record in members)),
        bbox=(
            min(record.bbox[0] for record in members if record.bbox is not None),
            min(record.bbox[1] for record in members if record.bbox is not None),
            max(record.bbox[2] for record in members if record.bbox is not None),
            max(record.bbox[3] for record in members if record.bbox is not None),
        ),
        interval=(
            min(record.temporal.start for record in members),
            max(record.temporal.end for record in members),
        ),
    )


def _index(
    work_dir: Path,
    records: list[StacRecord],
    *,
    generation: str = "generation-1",
    collections: tuple[StacCollection, ...] | None = None,
    exclusions: tuple[IndexExclusion, ...] = (),
) -> StacIndex:
    """Build one synthetic index for persistence and rendering tests."""
    ordered = tuple(sorted(records, key=lambda record: record.public_id))
    if collections is None:
        families = sorted({record.family for record in ordered})
        collections = tuple(
            _collection(family, ordered)
            for family in families
            if any(
                record.bbox is not None and record.family == family
                for record in ordered
            )
        )
    collection_by_family = {
        collection.family: collection.collection_id for collection in collections
    }
    ordered = tuple(
        StacRecord(
            public_id=record.public_id,
            write_id=record.write_id,
            kind=record.kind,
            family=record.family,
            source_item_path=record.source_item_path,
            source_relative_path=record.source_relative_path,
            collection_id=collection_by_family.get(record.family)
            if record.bbox is not None
            else None,
            temporal=record.temporal,
            bbox=record.bbox,
            item=record.item,
            asset_paths=record.asset_paths,
        )
        for record in ordered
    )
    return StacIndex(
        generation=generation,
        created_at="2024-01-01T00:00:00.000000Z",
        work_dir=work_dir,
        records=ordered,
        collections=collections,
        exclusions=exclusions,
    )


def test_build_index_assembles_families_and_reports_exclusions(
    tmp_path: Path,
) -> None:
    """Injected products become records or exact exclusion reasons."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    scene_id = new_write_id()
    missing_time_id = new_write_id()
    layout_id = new_write_id()
    stale_id = new_write_id()
    dependent_id = new_write_id()
    products = [
        Product(
            path=work_dir / "scene" / "a",
            item=_product_item("scene", scene_id),
        ),
        Product(
            path=work_dir / "scene" / "b",
            item=_product_item("scene", missing_time_id, date=None),
        ),
        Product(
            path=work_dir / "layout" / "a",
            item=_product_item("merge_layout", layout_id),
        ),
        Product(
            path=work_dir / "scene" / "c",
            item=_product_item(
                "scene",
                stale_id,
                inputs={"layout": new_write_id()},
            ),
        ),
        Product(
            path=work_dir / "ifg" / "a",
            item=_product_item(
                "ifg",
                dependent_id,
                inputs={"layout": layout_id},
            ),
        ),
    ]
    index = build_index(work_dir, products=products)
    assert sorted(record.write_id for record in index.records) == sorted(
        [scene_id, dependent_id]
    )
    assert index.families() == ("interferograms", "scenes")
    assert index.collections == ()
    reasons = {
        exclusion.source_relative_path: exclusion.reason
        for exclusion in index.exclusions
    }
    assert reasons["scene/b"] == "missing-temporal"
    assert reasons["layout/a"] == "unsupported-kind"
    assert reasons["scene/c"] == "stale"
    report = index.report()
    assert report["item_count"] == 2
    assert report["family_count"] == 2
    assert report["families"] == {"interferograms": 1, "scenes": 1}
    assert report["collections"] == []
    assert len(report["exclusions"]) == 3
    assert json.dumps(report, allow_nan=False)


def test_build_index_prunes_derived_cache_tmp_and_hidden_dirs(
    tmp_path: Path,
) -> None:
    """Filesystem discovery ignores derived, cache, tmp, hidden, and symlink dirs."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    write_id = new_write_id()
    _write_item(work_dir / "scene" / "real", _product_item("scene", write_id))
    for relative in (
        "stac/items/fake",
        "cache/fake",
        "tmp/fake",
        ".staging/fake",
        ".aside/fake",
    ):
        _write_item(
            work_dir / relative,
            _product_item("scene", new_write_id()),
        )
    (work_dir / "link").symlink_to(work_dir / "scene" / "real", target_is_directory=True)
    index = build_index(work_dir)
    assert [record.write_id for record in index.records] == [write_id]
    assert index.exclusions == ()


def test_build_index_does_not_descend_registered_asset_directories(
    tmp_path: Path,
) -> None:
    """Discovery prunes payload trees but still finds committed child products."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    parent_id = new_write_id()
    parent = work_dir / "ifg" / "pair"
    _write_item(
        parent,
        _product_item(
            "ifg",
            parent_id,
            assets={"data": "data.bin", "zarr": "zarr"},
        ),
    )
    _write_item(
        parent / "zarr" / "0" / "0",
        _product_item("scene", new_write_id()),
    )
    unwrap_id = new_write_id()
    _write_item(
        parent / "unwrap",
        _product_item("unwrap", unwrap_id, inputs={"ifg": parent_id}),
    )
    index = build_index(work_dir)
    assert {record.write_id for record in index.records} == {parent_id, unwrap_id}
    assert index.exclusions == ()


def test_duplicate_unsafe_outside_and_symlink_exclusions(tmp_path: Path) -> None:
    """Duplicate ids, unsafe assets, and unsafe locations are rejected."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    duplicate = new_write_id()
    products = [
        Product(
            path=work_dir / "scene" / "a",
            item=_product_item("scene", duplicate),
        ),
        Product(
            path=work_dir / "scene" / "b",
            item=_product_item("scene", duplicate),
        ),
        Product(
            path=work_dir / "scene" / "unsafe",
            item=_product_item(
                "scene",
                new_write_id(),
                assets={"data": "../escape.bin"},
            ),
        ),
    ]
    outside = tmp_path / "outside"
    outside.mkdir()
    products.append(
        Product(path=outside, item=_product_item("scene", new_write_id()))
    )
    real = work_dir / "scene" / "real"
    _write_item(real, _product_item("scene", new_write_id()))
    link = work_dir / "link"
    link.symlink_to(real, target_is_directory=True)
    products.append(
        Product(path=link, item=_product_item("scene", new_write_id()))
    )
    index = build_index(work_dir, products=products)
    assert index.records == ()
    reasons = sorted(exclusion.reason for exclusion in index.exclusions)
    assert reasons.count("duplicate-write-id") == 2
    assert reasons.count("unsafe-asset") == 1
    assert reasons.count("outside-work-dir") == 2


def test_collections_exist_only_for_geographic_family_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assembly assigns membership and union extents from actual geo members."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    geo_id = new_write_id()
    radar_id = new_write_id()

    def fake_normalize(
        product: Product,
        *,
        work_dir: Path,
        products_by_write_id: object = None,
    ) -> StacRecord:
        del products_by_write_id
        write_id = product.write_id
        is_geo = write_id == geo_id
        return StacRecord(
            public_id=f"scene-{write_id}",
            write_id=write_id,
            kind="scene",
            family="scenes",
            source_item_path=product.path / "item.json",
            source_relative_path=product.path.relative_to(work_dir).as_posix(),
            collection_id=None,
            temporal=TemporalCoverage(
                start=_START,
                end=_END,
                precision="day",
            ),
            bbox=(0.0, 0.0, 2.0, 2.0) if is_geo else None,
            item={
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": f"scene-{write_id}",
                "geometry": _SQUARE if is_geo else None,
                "properties": {},
                "assets": {},
                "links": [],
            },
            asset_paths={},
        )

    monkeypatch.setattr(index_module, "normalize_product", fake_normalize)
    products = [
        Product(
            path=work_dir / "scene" / "geo",
            item=_product_item("scene", geo_id),
        ),
        Product(
            path=work_dir / "scene" / "radar",
            item=_product_item("scene", radar_id),
        ),
    ]
    index = build_index(work_dir, products=products)
    assert len(index.records) == 2
    collection = index.collections[0]
    assert collection.collection_id == "scenes-geographic"
    assert collection.item_ids == (f"scene-{geo_id}",)
    assert collection.bbox == (0.0, 0.0, 2.0, 2.0)
    geo = next(record for record in index.records if record.write_id == geo_id)
    radar = next(record for record in index.records if record.write_id == radar_id)
    assert geo.collection_id == "scenes-geographic"
    assert radar.collection_id is None


def test_renderers_are_deterministic_and_schema_complete(tmp_path: Path) -> None:
    """Catalogs, Collections, and Items carry stable required fields."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    radar = _record(work_dir, "scene-radar")
    geo = _record(
        work_dir,
        "scene-geo",
        bbox=(0.0, 0.0, 2.0, 2.0),
        geometry=_SQUARE,
        assets={"data": "scene-geo/data.cog"},
    )
    index = _index(work_dir, [radar, geo])
    targets = LinkTargets.for_http(
        base_url="http://127.0.0.1:8000",
        view_path="/catalog.json",
    )
    root_first = render_root_catalog(index, targets)
    root_second = render_root_catalog(index, targets)
    assert root_first == root_second
    assert root_first["type"] == "Catalog"
    assert root_first["id"].startswith("stack-")
    assert root_first["description"] == root_second["description"]
    assert str(tmp_path) not in root_first["description"]
    assert "http" not in root_first["description"]
    assert {"type", "stac_version", "id", "description", "links"} <= set(root_first)
    child_rels = [link["href"] for link in root_first["links"] if link["rel"] == "child"]
    assert any(href.endswith("/catalogs/scenes.json") for href in child_rels)
    assert any(href.endswith("/collections/scenes-geographic") for href in child_rels)

    family = render_family_catalog(index, "scenes", targets)
    assert family["id"] == "scenes"
    assert {"type", "stac_version", "id", "description", "links"} <= set(family)
    assert any(link["rel"] == "item" for link in family["links"])
    assert any(link["rel"] == "child" for link in family["links"])

    collection = render_collection(index.collections[0], index, targets)
    assert collection["type"] == "Collection"
    assert collection["license"] == "proprietary"
    assert {"id", "description", "license", "extent", "links"} <= set(collection)
    assert collection["extent"]["spatial"]["bbox"] == [[0.0, 0.0, 2.0, 2.0]]
    assert collection["extent"]["temporal"]["interval"][0][0].endswith("Z")
    assert any(link["rel"] == "item" for link in collection["links"])

    indexed_geo = index.record(geo.public_id)
    assert indexed_geo is not None
    item = render_item(indexed_geo, targets)
    assert item["type"] == "Feature"
    assert item["properties"]["custom:note"] == "preserved"
    assert item["assets"]["data"]["checksum:multihash"] == "kept"
    assert item["collection"] == "scenes-geographic"
    item_rels = {link["rel"] for link in item["links"]}
    assert {"root", "self", "parent", "collection"} <= item_rels
    assert item["assets"]["data"]["href"] == (
        f"http://127.0.0.1:8000/assets/{geo.public_id}/data"
    )
    radar_item = render_item(radar, targets)
    assert "collection" not in radar_item
    assert "bbox" not in radar_item


def test_disk_links_are_relative_and_resolve_to_registered_assets(
    tmp_path: Path,
) -> None:
    """Disk asset hrefs resolve from an Item view to the registered source."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    geo = _record(
        work_dir,
        "scene-geo",
        bbox=(0.0, 0.0, 2.0, 2.0),
        geometry=_SQUARE,
        assets={"data": "scene-geo/data.cog"},
    )
    index = _index(work_dir, [geo])
    view = work_dir / "stac" / "items" / f"{geo.public_id}.json"
    targets = LinkTargets.for_disk(work_dir=work_dir, view_path=view)
    rendered = render_item(geo, targets)
    href = rendered["assets"]["data"]["href"]
    assert href == "../../scene-geo/data.cog"
    assert (view.parent / href).resolve() == geo.asset_paths["data"].resolve()
    assert targets.item(geo.public_id) == f"{geo.public_id}.json"
    assert targets.root == "../../catalog.json"


def test_write_portable_catalog_builds_complete_tree(tmp_path: Path) -> None:
    """A successful write stages, swaps, and replaces the root catalog last."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    radar = _record(work_dir, "scene-radar")
    geo = _record(
        work_dir,
        "scene-geo",
        bbox=(0.0, 0.0, 2.0, 2.0),
        geometry=_SQUARE,
        assets={"data": "scene-geo/data.cog"},
    )
    index = _index(work_dir, [radar, geo])
    source_item = work_dir / "scene-radar" / "item.json"
    source_item.parent.mkdir(parents=True, exist_ok=True)
    source_item.write_bytes(b"unchanged canonical item")

    report = write_portable_catalog(index, work_dir=work_dir)
    assert report.requested is True
    assert report.persisted is True
    assert report.error is None
    assert report.generation == index.generation
    assert report.catalog_path == work_dir / "catalog.json"

    assert (work_dir / "catalog.json").is_file()
    assert (work_dir / "stac" / ".faninsar-stac.json").is_file()
    assert (work_dir / "stac" / "items" / f"{radar.public_id}.json").is_file()
    assert (work_dir / "stac" / "items" / f"{geo.public_id}.json").is_file()
    assert (work_dir / "stac" / "catalogs" / "scenes.json").is_file()
    assert (
        work_dir / "stac" / "collections" / "scenes-geographic.json"
    ).is_file()
    assert not list(work_dir.glob(".stac-staging-*"))
    assert not list(work_dir.glob(".stac-previous-*"))
    assert source_item.read_bytes() == b"unchanged canonical item"

    root = json.loads((work_dir / "catalog.json").read_text(encoding="utf-8"))
    assert root["faninsar:generated_by"] == "faninsar.stac_serve"
    assert root["type"] == "Catalog"
    assert not any(link["rel"] == "item" for link in root["links"])
    assert any(link["rel"] == "child" for link in root["links"])
    marker = json.loads(
        (work_dir / "stac" / ".faninsar-stac.json").read_text(encoding="utf-8")
    )
    assert marker["generation"] == index.generation
    family = json.loads(
        (work_dir / "stac" / "catalogs" / "scenes.json").read_text(encoding="utf-8")
    )
    assert any(link["rel"] == "item" for link in family["links"])
    collection = json.loads(
        (work_dir / "stac" / "collections" / "scenes-geographic.json").read_text(
            encoding="utf-8"
        )
    )
    assert collection["extent"]["spatial"]["bbox"] == [[0.0, 0.0, 2.0, 2.0]]
    item = json.loads(
        (work_dir / "stac" / "items" / f"{geo.public_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert item["assets"]["data"]["href"] == "../../scene-geo/data.cog"
    assert (
        work_dir / "stac" / "items" / item["assets"]["data"]["href"]
    ).resolve() == geo.asset_paths["data"].resolve()


def test_write_portable_catalog_empty_index(tmp_path: Path) -> None:
    """An empty generation writes a valid root Catalog with no child views."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    index = _index(work_dir, [])
    report = write_portable_catalog(index, work_dir=work_dir)
    assert report.persisted is True
    root = json.loads((work_dir / "catalog.json").read_text(encoding="utf-8"))
    assert root["links"] == []
    assert root["faninsar:generated_by"] == "faninsar.stac_serve"
    assert (work_dir / "stac" / ".faninsar-stac.json").is_file()
    assert not (work_dir / "stac" / "items").exists()
    assert not (work_dir / "stac" / "catalogs").exists()
    assert not (work_dir / "stac" / "collections").exists()


def test_write_portable_catalog_protects_unowned_artifacts(tmp_path: Path) -> None:
    """Unowned catalogs, stac trees, and symlinks are never overwritten."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    catalog = work_dir / "catalog.json"
    catalog.write_text('{"id": "user-catalog"}', encoding="utf-8")
    index = _index(work_dir, [])
    report = write_portable_catalog(index, work_dir=work_dir)
    assert report.persisted is False
    assert report.error is not None
    assert catalog.read_text(encoding="utf-8") == '{"id": "user-catalog"}'
    assert not (work_dir / "stac").exists()

    second = tmp_path / "stack2"
    second.mkdir()
    (second / "stac").mkdir()
    (second / "stac" / "mine.txt").write_text("mine", encoding="utf-8")
    report = write_portable_catalog(_index(second, []), work_dir=second)
    assert report.persisted is False
    assert not (second / "catalog.json").exists()
    assert (second / "stac" / "mine.txt").read_text(encoding="utf-8") == "mine"

    third = tmp_path / "stack3"
    third.mkdir()
    target = third / "external.json"
    target.write_text("external", encoding="utf-8")
    (third / "catalog.json").symlink_to(target)
    report = write_portable_catalog(_index(third, []), work_dir=third)
    assert report.persisted is False
    assert target.read_text(encoding="utf-8") == "external"


def test_write_portable_catalog_rolls_back_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected failure restores the previous complete tree and root catalog."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    first = _index(
        work_dir,
        [_record(work_dir, "scene-first")],
        generation="generation-1",
    )
    assert write_portable_catalog(first, work_dir=work_dir).persisted is True
    catalog_before = (work_dir / "catalog.json").read_bytes()
    marker_before = (work_dir / "stac" / ".faninsar-stac.json").read_bytes()

    second = _index(
        work_dir,
        [_record(work_dir, "scene-second")],
        generation="generation-2",
    )

    def guarded_write(path: Path, document: dict[str, object]) -> None:
        if path.name.startswith(".stac-catalog-"):
            message = "injected root catalog failure"
            raise OSError(message)
        real_write(path, document)  # type: ignore[arg-type]

    real_write = catalog_module._write_document

    monkeypatch.setattr(catalog_module, "_write_document", guarded_write)
    report = write_portable_catalog(second, work_dir=work_dir)
    assert report.persisted is False
    assert (work_dir / "catalog.json").read_bytes() == catalog_before
    assert (work_dir / "stac" / ".faninsar-stac.json").read_bytes() == marker_before
    assert (work_dir / "stac" / "items" / "scene-first.json").is_file()
    assert not (work_dir / "stac" / "items" / "scene-second.json").exists()
    assert not list(work_dir.glob(".stac-staging-*"))
    assert not list(work_dir.glob(".stac-previous-*"))


def test_write_portable_catalog_replaces_generation(tmp_path: Path) -> None:
    """A later successful generation replaces stale views without leftovers."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    first = _index(
        work_dir,
        [_record(work_dir, "scene-first")],
        generation="generation-1",
    )
    assert write_portable_catalog(first, work_dir=work_dir).persisted is True
    second = _index(
        work_dir,
        [_record(work_dir, "scene-second")],
        generation="generation-2",
    )
    report = write_portable_catalog(second, work_dir=work_dir)
    assert report.persisted is True
    assert (work_dir / "stac" / "items" / "scene-second.json").is_file()
    assert not (work_dir / "stac" / "items" / "scene-first.json").exists()
    marker = json.loads(
        (work_dir / "stac" / ".faninsar-stac.json").read_text(encoding="utf-8")
    )
    assert marker["generation"] == "generation-2"


def test_build_index_failure_in_normalization_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normalization failure becomes an exclusion with its reason and detail."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    write_id = new_write_id()

    def fail_normalize(
        product: Product,
        *,
        work_dir: Path,
        products_by_write_id: object = None,
    ) -> StacRecord:
        del product, work_dir, products_by_write_id
        raise StacNormalizationError("invalid-item", "synthetic failure")

    monkeypatch.setattr(index_module, "normalize_product", fail_normalize)
    index = build_index(
        work_dir,
        products=[
            Product(
                path=work_dir / "scene" / "a",
                item=_product_item("scene", write_id),
            )
        ],
    )
    assert index.records == ()
    assert index.exclusions[0].reason == "invalid-item"
    assert index.exclusions[0].detail == "synthetic failure"


def test_normalizer_receives_physical_map_without_duplicate_or_stale_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate and stale write ids are unavailable for inheritance."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    good_id = new_write_id()
    layout_id = new_write_id()
    stale_id = new_write_id()
    duplicate_id = new_write_id()
    captured: dict[str, Product] = {}

    def record_for(product: Product) -> StacRecord:
        return StacRecord(
            public_id=f"scene-{product.write_id}",
            write_id=product.write_id,
            kind="scene",
            family="scenes",
            source_item_path=product.path / "item.json",
            source_relative_path=product.path.relative_to(work_dir).as_posix(),
            collection_id=None,
            temporal=TemporalCoverage(start=_START, end=_END, precision="day"),
            bbox=None,
            item={
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": f"scene-{product.write_id}",
                "geometry": None,
                "properties": {},
                "assets": {},
                "links": [],
            },
            asset_paths={},
        )

    def fake_normalize(
        product: Product,
        *,
        work_dir: Path,
        products_by_write_id: object = None,
    ) -> StacRecord:
        del work_dir
        captured.update(products_by_write_id or {})  # type: ignore[arg-type]
        return record_for(product)

    monkeypatch.setattr(index_module, "normalize_product", fake_normalize)
    products = [
        Product(
            path=work_dir / "scene" / "good",
            item=_product_item("scene", good_id),
        ),
        Product(
            path=work_dir / "layout" / "good",
            item=_product_item("merge_layout", layout_id),
        ),
        Product(
            path=work_dir / "scene" / "stale",
            item=_product_item(
                "scene",
                stale_id,
                inputs={"layout": new_write_id()},
            ),
        ),
        Product(
            path=work_dir / "scene" / "dup-a",
            item=_product_item("scene", duplicate_id),
        ),
        Product(
            path=work_dir / "scene" / "dup-b",
            item=_product_item("scene", duplicate_id),
        ),
    ]
    index = build_index(work_dir, products=products)
    assert set(captured) == {good_id, layout_id}
    assert len(index.records) == 2
    reasons = sorted(exclusion.reason for exclusion in index.exclusions)
    assert reasons.count("duplicate-write-id") == 2
    assert reasons.count("stale") == 1


def test_dependent_of_a_stale_source_is_missing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale source is excluded from inheritance and its child fails."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    source_id = new_write_id()
    geocoded_id = new_write_id()

    def fake_normalize(
        product: Product,
        *,
        work_dir: Path,
        products_by_write_id: object = None,
    ) -> StacRecord:
        del work_dir
        available = products_by_write_id or {}
        if product.kind == "geocoded":
            source = available.get(product.inputs["source"])  # type: ignore[union-attr]
            if source is None:
                raise StacNormalizationError(
                    "missing-source",
                    f"source {product.inputs['source']} is unavailable",
                )
        return StacRecord(
            public_id=f"{product.kind}-{product.write_id}",
            write_id=product.write_id,
            kind=product.kind,
            family="scenes",
            source_item_path=product.path / "item.json",
            source_relative_path=product.path.relative_to(work_dir).as_posix(),
            collection_id=None,
            temporal=TemporalCoverage(start=_START, end=_END, precision="day"),
            bbox=None,
            item={
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": f"{product.kind}-{product.write_id}",
                "geometry": None,
                "properties": {},
                "assets": {},
                "links": [],
            },
            asset_paths={},
        )

    monkeypatch.setattr(index_module, "normalize_product", fake_normalize)
    products = [
        Product(
            path=work_dir / "scene" / "source",
            item=_product_item(
                "scene",
                source_id,
                inputs={"layout": new_write_id()},
            ),
        ),
        Product(
            path=work_dir / "geocoded" / "child",
            item=_product_item(
                "geocoded",
                geocoded_id,
                inputs={"source": source_id},
            ),
        ),
    ]
    index = build_index(work_dir, products=products)
    assert index.records == ()
    reasons = {exclusion.source_relative_path: exclusion.reason for exclusion in index.exclusions}
    assert reasons["scene/source"] == "stale"
    assert reasons["geocoded/child"] == "missing-source"


def test_generation_is_stable_for_identical_candidates(tmp_path: Path) -> None:
    """An identical rebuild keeps the same generation digest."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    products = [
        Product(
            path=work_dir / "scene" / "a",
            item=_product_item("scene", new_write_id()),
        )
    ]
    first = build_index(work_dir, products=products)
    second = build_index(work_dir, products=products)
    assert first.generation == second.generation
    assert first.records == second.records


def test_malformed_and_oversized_items_are_excluded(tmp_path: Path) -> None:
    """Invalid strict JSON, non-finite values, and over-bound files are rejected."""
    work_dir = tmp_path / "stack"
    work_dir.mkdir()
    broken = work_dir / "broken"
    broken.mkdir()
    (broken / "item.json").write_text("{", encoding="utf-8")
    nonfinite = work_dir / "nonfinite"
    nonfinite.mkdir()
    (nonfinite / "item.json").write_text(
        '{"stac_version": "1.0.0", "type": "Feature", "id": "x", '
        '"properties": {"faninsar:write_id": "01M7A50HSMW2M5G3H6V9YD6HV2", '
        '"x": NaN}, "assets": {}}',
        encoding="utf-8",
    )
    oversized = work_dir / "oversized"
    oversized.mkdir()
    (oversized / "item.json").write_bytes(b"{" + b" " * (4 * 1024 * 1024) + b"}")
    index = build_index(work_dir)
    assert index.records == ()
    assert sorted(exclusion.reason for exclusion in index.exclusions) == [
        "invalid-item",
        "invalid-item",
        "invalid-item",
    ]


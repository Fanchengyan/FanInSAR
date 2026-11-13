# ruff: noqa: INP001
"""Offline STAC 1.0.0 schema qualification for served documents (PROPOSAL-0089).

The vendored schemas under ``tests/fixtures/stac_v1_0`` are pinned copies of
official upstream documents, used only by development tests:

- ``item-spec``, ``catalog-spec``, ``collection-spec``:
  https://github.com/radiantearth/stac-spec/tree/v1.0.0 (Apache-2.0,
  https://github.com/radiantearth/stac-spec/blob/v1.0.0/LICENSE)
- ``projection/v2.0.0/schema.json``:
  https://github.com/stac-extensions/projection/tree/v2.0.0 (Apache-2.0,
  https://github.com/stac-extensions/projection/blob/v2.0.0/LICENSE)
- ``geojson``:
  https://github.com/geojson/schema (MIT,
  https://github.com/geojson/schema/blob/main/LICENSE)
- ``projjson/v0.7/projjson.schema.json``:
  https://proj.org/schemas/v0.7/projjson.schema.json (PROJ license,
  https://github.com/OSGeo/PROJ/blob/9.6.0/COPYING)

Every external ``$ref`` resolves through a local ``referencing`` registry.
Validation performs no network access; a missing schema is a hard failure.
"""

from __future__ import annotations

import copy
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
import pystac
import pytest
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from faninsar.io.storage.product import Product, publish_product

if TYPE_CHECKING:
    from faninsar.stack._stac_service import StackStacService

_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "stac_v1_0"
_ITEM_SCHEMA = "item-spec/json-schema/item.json"
_CATALOG_SCHEMA = "catalog-spec/json-schema/catalog.json"
_COLLECTION_SCHEMA = "collection-spec/json-schema/collection.json"
_FEATURE_COLLECTION_SCHEMA = "geojson/FeatureCollection.json"
_PROJECTION_SCHEMA = "projection/v2.0.0/schema.json"
_PROJECTION_URI = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
_DRAFT7_URI = "http://json-schema.org/draft-07/schema"

# directory prefix -> (pinned source tree, license)
_SOURCE_MANIFEST = {
    "item-spec/json-schema": (
        "https://github.com/radiantearth/stac-spec/tree/v1.0.0/item-spec/json-schema",
        "https://github.com/radiantearth/stac-spec/blob/v1.0.0/LICENSE",
    ),
    "catalog-spec/json-schema": (
        "https://github.com/radiantearth/stac-spec/tree/v1.0.0/catalog-spec/json-schema",
        "https://github.com/radiantearth/stac-spec/blob/v1.0.0/LICENSE",
    ),
    "collection-spec/json-schema": (
        "https://github.com/radiantearth/stac-spec/tree/v1.0.0/collection-spec/json-schema",
        "https://github.com/radiantearth/stac-spec/blob/v1.0.0/LICENSE",
    ),
    "projection/v2.0.0": (
        "https://github.com/stac-extensions/projection/tree/v2.0.0/json-schema",
        "https://github.com/stac-extensions/projection/blob/v2.0.0/LICENSE",
    ),
    "geojson": (
        "https://github.com/geojson/schema",
        "https://github.com/geojson/schema/blob/main/LICENSE",
    ),
    "projjson/v0.7": (
        "https://proj.org/schemas/v0.7/projjson.schema.json",
        "https://github.com/OSGeo/PROJ/blob/9.6.0/COPYING",
    ),
}


def _load_schemas() -> dict[str, dict[str, Any]]:
    """Load every vendored schema, keyed by its path below the fixture root."""
    schemas: dict[str, dict[str, Any]] = {}
    for path in sorted(_FIXTURE_ROOT.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "$id" not in document:
            message = f"vendored schema lacks an $id: {path}"
            raise AssertionError(message)
        schemas[path.relative_to(_FIXTURE_ROOT).as_posix()] = document
    if not schemas:
        message = f"no vendored STAC schemas under {_FIXTURE_ROOT}"
        raise AssertionError(message)
    return schemas


_SCHEMAS = _load_schemas()


def _registry_resources() -> tuple[Registry, frozenset[str]]:
    """Register canonical ``$id`` URIs for every vendored schema."""
    entries: dict[str, Resource[Any]] = {}
    for document in _SCHEMAS.values():
        resource = Resource.from_contents(document, default_specification=DRAFT7)
        identifier = str(document["$id"])
        entries[identifier] = resource
        entries[identifier.rstrip("#")] = resource
    meta_schema = Resource.from_contents(
        jsonschema.Draft7Validator.META_SCHEMA, default_specification=DRAFT7
    )
    entries[_DRAFT7_URI] = meta_schema
    entries[f"{_DRAFT7_URI}#"] = meta_schema
    return Registry().with_resources(entries.items()), frozenset(entries)


_REGISTRY, _REGISTRY_KEYS = _registry_resources()


@pytest.fixture(autouse=True)
def _pin_pystac_write_version():
    """Write the pinned STAC 1.0.0 version in PySTAC round trips.

    PySTAC writes its own supported STAC version by default, which is newer
    than the vendored 1.0.0 schemas under test. Pinning the write version keeps
    the round-trip assertion about our documents rather than the installed
    library default, and restores the previous value afterwards.
    """
    previous = pystac.get_stac_version()
    pystac.set_stac_version("1.0.0")
    try:
        yield
    finally:
        pystac.set_stac_version(previous)


def _external_refs(document: Any) -> list[str]:
    """Return every non-internal ``$ref`` in a schema document."""
    refs: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str) and not reference.startswith("#"):
                refs.append(reference)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    return refs


def _instant(value: str) -> datetime:
    """Parse one RFC 3339 instant for semantic timestamp comparison.

    ``pystac`` reserializes timestamps and drops a zero microsecond part, so
    round-trip comparisons use parsed instants while the emitted document keeps
    the exact STAC 1.0.0 string asserted before the round trip.
    """
    return datetime.fromisoformat(value)


def _validate(relative: str, document: dict[str, Any]) -> None:
    """Validate one document against a vendored schema, failing loudly."""
    validator = jsonschema.Draft7Validator(_SCHEMAS[relative], registry=_REGISTRY)
    errors = sorted(validator.iter_errors(document), key=lambda error: error.json_path)
    if errors:
        details = "\n".join(f"{error.json_path}: {error.message}" for error in errors)
        message = f"{relative} validation failed:\n{details}\n{document}"
        raise AssertionError(message)


def _assert_catalog(document: dict[str, Any]) -> None:
    """Validate a Catalog and require its STAC required fields explicitly."""
    _validate(_CATALOG_SCHEMA, document)
    assert document["type"] == "Catalog"
    assert document["stac_version"] == "1.0.0"
    assert isinstance(document["id"], str)
    assert document["id"]
    assert isinstance(document["description"], str)
    assert document["description"]
    assert isinstance(document["links"], list)
    roundtrip = pystac.Catalog.from_dict(copy.deepcopy(document)).to_dict(
        transform_hrefs=False
    )
    assert roundtrip["id"] == document["id"]
    assert roundtrip["type"] == "Catalog"
    assert roundtrip["stac_version"] == "1.0.0"
    assert roundtrip["description"] == document["description"]
    assert isinstance(roundtrip["links"], list)


def _assert_collection(document: dict[str, Any]) -> None:
    """Validate a Collection and assert its extent survives a round trip."""
    _validate(_COLLECTION_SCHEMA, document)
    assert document["type"] == "Collection"
    assert document["stac_version"] == "1.0.0"
    assert isinstance(document["id"], str)
    assert document["id"]
    assert isinstance(document["description"], str)
    assert document["description"]
    assert isinstance(document["license"], str)
    assert document["license"]
    assert isinstance(document["links"], list)
    spatial = document["extent"]["spatial"]["bbox"]
    temporal = document["extent"]["temporal"]["interval"]
    assert spatial
    assert all(isinstance(value, (int, float)) for value in spatial[0])
    assert temporal
    assert len(temporal[0]) == 2
    roundtrip = pystac.Collection.from_dict(copy.deepcopy(document)).to_dict(
        transform_hrefs=False
    )
    assert roundtrip["id"] == document["id"]
    assert roundtrip["type"] == "Collection"
    assert roundtrip["stac_version"] == "1.0.0"
    assert roundtrip["description"] == document["description"]
    assert roundtrip["extent"]["spatial"]["bbox"] == spatial
    roundtrip_temporal = roundtrip["extent"]["temporal"]["interval"]
    assert len(roundtrip_temporal) == 1
    assert [_instant(value) for value in roundtrip_temporal[0]] == [
        _instant(value) for value in temporal[0]
    ]
    assert isinstance(roundtrip["links"], list)


def _assert_item(document: dict[str, Any]) -> None:
    """Validate an Item, extension fields, and required temporal metadata."""
    _validate(_ITEM_SCHEMA, document)
    if _PROJECTION_URI in document.get("stac_extensions", ()):
        _validate(_PROJECTION_SCHEMA, document)
    assert document["type"] == "Feature"
    assert document["stac_version"] == "1.0.0"
    assert isinstance(document["id"], str)
    assert document["id"]
    assert isinstance(document["links"], list)
    properties = document["properties"]
    has_datetime = properties.get("datetime") is not None or (
        properties.get("start_datetime") is not None
        and properties.get("end_datetime") is not None
    )
    assert has_datetime, "an Item must carry a real timestamp or temporal interval"
    roundtrip = pystac.Item.from_dict(copy.deepcopy(document)).to_dict(
        transform_hrefs=False
    )
    assert roundtrip["id"] == document["id"]
    assert roundtrip["type"] == "Feature"
    assert roundtrip["stac_version"] == "1.0.0"
    assert isinstance(roundtrip["links"], list)
    roundtrip_properties = roundtrip["properties"]
    has_roundtrip_datetime = roundtrip_properties["datetime"] is not None or (
        roundtrip_properties["start_datetime"] is not None
        and roundtrip_properties["end_datetime"] is not None
    )
    assert has_roundtrip_datetime


def _assert_feature_collection(document: dict[str, Any]) -> None:
    """Validate a search response and every Item it contains."""
    _validate(_FEATURE_COLLECTION_SCHEMA, document)
    assert document["type"] == "FeatureCollection"
    assert isinstance(document["features"], list)
    for feature in document["features"]:
        _assert_item(feature)


def _asset_writer(staging: Path) -> dict[str, dict[str, Any]]:
    """Write one deterministic payload used by every fixture product."""
    (staging / "data.bin").write_bytes(b"faninsar-stac-schema")
    return {
        "data": {
            "href": "data.bin",
            "type": "application/octet-stream",
            "roles": ["data"],
        }
    }


def _publish(
    target: Path,
    *,
    kind: str,
    properties: dict[str, Any],
    inputs: dict[str, Product] | None = None,
) -> Product:
    """Publish one committed fixture product."""
    return publish_product(
        target,
        kind=kind,
        properties=properties,
        write_payloads=_asset_writer,
        inputs=inputs,
    )


def _build_workspace(root: Path) -> Path:
    """Create a workspace with radar, geo, IFG, and geocoded products."""
    root.mkdir(parents=True, exist_ok=True)
    radar_scene = _publish(
        root / "coreg" / "20240101",
        kind="scene",
        properties={
            "faninsar:date": "20240101",
            "faninsar:domain": "radar",
        },
    )
    geo_scene = _publish(
        root / "coreg" / "20240102",
        kind="scene",
        properties={
            "faninsar:date": "20240102",
            "faninsar:domain": "geo",
            "faninsar:crs": "EPSG:4326",
            "faninsar:transform": [0.01, 0.0, 10.0, 0.0, -0.01, 20.0],
            "faninsar:shape": [4, 5],
        },
    )
    _publish(
        root / "ifg" / "20240101_20240102",
        kind="ifg",
        properties={
            "faninsar:pair": ["20240101", "20240102"],
        },
        inputs={"primary": radar_scene, "secondary": geo_scene},
    )
    _publish(
        root / "geo" / "20240101_20240102",
        kind="geocoded",
        properties={
            "faninsar:name": "geocoded-ifg",
            "faninsar:grid": {
                "crs": "EPSG:4326",
                "transform": [0.01, 0.0, 10.0, 0.0, -0.01, 20.0],
                "shape": [4, 5],
            },
        },
        inputs={"source": geo_scene},
    )
    return root


def _start_service(work_dir: Path, *, persist_catalog: bool) -> StackStacService:
    """Start the agreed serving factory on an ephemeral loopback port."""
    from faninsar.stack._stac_service import start_stac_service

    return start_stac_service(
        work_dir,
        host="127.0.0.1",
        port=0,
        persist_catalog=persist_catalog,
    )


def _fetch_json(url: str) -> dict[str, Any]:
    """GET one local service URL and decode its JSON body."""
    with urllib.request.urlopen(url, timeout=10) as response:
        assert response.status == 200
        payload = json.loads(response.read().decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _disk_documents(work_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return the persisted derived catalog documents of one workspace."""
    paths = [work_dir / "catalog.json"]
    paths.extend(
        path
        for path in sorted((work_dir / "stac").rglob("*.json"))
        if not path.name.startswith(".")
    )
    documents = []
    for path in paths:
        assert path.is_file(), f"missing derived catalog document: {path}"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict)
        documents.append((path, document))
    return documents


def test_vendored_schema_registry_is_offline_complete() -> None:
    """Every external schema ref resolves locally and every file is attributed."""
    for relative, document in _SCHEMAS.items():
        base = str(document["$id"])
        for reference in _external_refs(document):
            target = urllib.parse.urldefrag(urllib.parse.urljoin(base, reference))[0]
            assert target in _REGISTRY_KEYS, (
                f"{relative} references an unvendored schema: {target}"
            )
        assert any(relative.startswith(prefix) for prefix in _SOURCE_MANIFEST), (
            f"missing source attribution for {relative}"
        )
    for source_url, license_url in _SOURCE_MANIFEST.values():
        assert source_url.startswith("https://")
        assert license_url.startswith("https://")


def test_served_documents_validate_against_vendored_stac_schemas(
    tmp_path: Path,
) -> None:
    """Every HTTP and persisted Catalog, Collection, and Item is valid STAC."""
    work_dir = _build_workspace(tmp_path / "schema-work")

    with _start_service(work_dir, persist_catalog=True) as service:
        assert service.base_url
        assert not service.base_url.endswith("/")

        landing = _fetch_json(f"{service.base_url}/")
        _assert_catalog(landing)

        root = _fetch_json(service.catalog_url)
        _assert_catalog(root)
        assert root["id"] == "stack-schema-work"

        family_ids = set()
        for family in ("scenes", "interferograms"):
            family_catalog = _fetch_json(f"{service.base_url}/catalogs/{family}.json")
            _assert_catalog(family_catalog)
            assert family_catalog["id"] == family
            assert any(link["rel"] == "item" for link in family_catalog["links"])
            family_ids.add(family_catalog["id"])
        assert family_ids == {"scenes", "interferograms"}

        collections = _fetch_json(f"{service.base_url}/collections")
        assert collections["collections"], "geo members must create a Collection"
        for collection in collections["collections"]:
            _assert_collection(collection)
            assert collection["id"].endswith("-geographic")
            routed = _fetch_json(f"{service.base_url}/collections/{collection['id']}")
            _assert_collection(routed)
            assert routed["id"] == collection["id"]

        search = _fetch_json(f"{service.base_url}/search?limit=100")
        _assert_feature_collection(search)
        features = search["features"]
        assert len(features) == 4
        assert len({feature["id"] for feature in features}) == len(features)
        families = {feature["properties"]["faninsar:family"] for feature in features}
        assert families == {"scenes", "interferograms"}
        for feature in features:
            if feature["id"].startswith("geocoded-"):
                assert feature["geometry"] is not None
                assert "bbox" in feature
                assert _PROJECTION_URI in feature["stac_extensions"]
                assert {"proj:code", "proj:transform", "proj:shape"} <= set(
                    feature["properties"]
                )
            else:
                routed = _fetch_json(f"{service.base_url}/items/{feature['id']}")
                _assert_item(routed)
        radar_items = [
            feature for feature in features if feature["id"].startswith("ifg-")
        ]
        assert radar_items
        for item in radar_items:
            assert item["geometry"] is None
            assert "bbox" not in item

    for path, document in _disk_documents(work_dir):
        kind = document.get("type")
        if kind == "Catalog":
            _assert_catalog(document)
        elif kind == "Collection":
            _assert_collection(document)
        elif kind == "Feature":
            _assert_item(document)
        else:
            message = f"unexpected derived document type at {path}"
            raise AssertionError(message)


def test_empty_workspace_serves_valid_empty_catalog(tmp_path: Path) -> None:
    """An empty workspace emits a valid Catalog with no child or item links."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with _start_service(empty, persist_catalog=False) as service:
        landing = _fetch_json(f"{service.base_url}/")
        _assert_catalog(landing)
        root = _fetch_json(service.catalog_url)
        _assert_catalog(root)
        assert root["id"] == "stack-empty"
        assert not any(link["rel"] in {"child", "item"} for link in root["links"])

        collections = _fetch_json(f"{service.base_url}/collections")
        assert collections["collections"] == []

        search = _fetch_json(f"{service.base_url}/search")
        _assert_feature_collection(search)
        assert search["features"] == []
        assert not any(empty.iterdir()), "persist_catalog=False wrote a file"


def test_catalog_descriptions_are_deterministic_and_location_free(
    tmp_path: Path,
) -> None:
    """Descriptions repeat across builds and name no host, port, path, or time."""
    work_dir = _build_workspace(tmp_path / "description-work")

    def descriptions() -> dict[str, str]:
        catalog = json.loads((work_dir / "catalog.json").read_text(encoding="utf-8"))
        values = {"catalog": catalog["description"]}
        for path in sorted((work_dir / "stac" / "catalogs").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            values[path.stem] = document["description"]
        return values

    with _start_service(work_dir, persist_catalog=True) as first:
        first_port = urllib.parse.urlsplit(first.base_url).port
        first_descriptions = descriptions()
    with _start_service(work_dir, persist_catalog=True):
        second_descriptions = descriptions()

    assert first_descriptions == second_descriptions
    for description in first_descriptions.values():
        assert str(tmp_path) not in description
        assert "127.0.0.1" not in description
        assert f":{first_port}" not in description
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", description)

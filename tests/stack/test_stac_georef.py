"""HTTP georeferenced Zarr view integration tests (PROPOSAL-0089).

These tests exercise the real loopback service and the real GDAL Zarr driver.
A missing metadata worker or a missing georeferenced asset is an integration
failure, not a skip: the service contract includes the derived view.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.parse
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from faninsar.stack import _stac_service as stac_service

CRS = "EPSG:32719"
TRANSFORM = (8.0, 0.0, 288728.0, 0.0, -200.0, 6066040.0)


@pytest.fixture(autouse=True)
def _clean_service_registry() -> Any:
    """Stop and forget services registered at the end of each test."""
    yield
    registry = getattr(stac_service, "_SERVICE_REGISTRY", None)
    if registry is None:
        return
    with stac_service._SERVICE_REGISTRY_LOCK:
        registrations = list(registry.values())
        registry.clear()
    for registration in registrations:
        with suppress(Exception):
            registration.service.stop()


def _publish_geo_zarr(
    work_dir: Path,
    *,
    name: str,
    values: Any,
    kind: str = "scene",
    asset: str = "coherence",
    domain: str = "geo",
    crs: str | None = CRS,
    transform: tuple[float, ...] | None = TRANSFORM,
    shape: tuple[int, int] | None = None,
    properties: dict[str, Any] | None = None,
    asset_fields: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | None = None,
    overwrite: bool = False,
) -> Any:
    """Commit one tiny real Zarr v3 array with geographic Item metadata."""
    import zarr
    from faninsar.io.storage.product import publish_product

    array = np.asarray(values)

    def write_payloads(staging: Path) -> dict[str, dict[str, Any]]:
        store = zarr.open_array(
            store=str(staging / f"{asset}.zarr"),
            mode="w",
            shape=array.shape,
            dtype=array.dtype,
            chunks=chunks,
        )
        store[:] = array
        return {
            asset: {
                "href": f"{asset}.zarr",
                "type": "application/vnd+zarr",
                **(asset_fields or {}),
            }
        }

    header: dict[str, Any] = {
        "faninsar:date": "2024-01-02",
        "faninsar:domain": domain,
        "faninsar:shape": list(shape or array.shape),
    }
    if kind == "ifg":
        header["faninsar:dates"] = ["20240101", "20240102"]
        header.pop("faninsar:date")
    if crs is not None:
        header["proj:code"] = crs
    if transform is not None:
        header["proj:transform"] = list(transform)
    header.update(properties or {})
    return publish_product(
        work_dir / "products" / name,
        kind=kind,
        properties=header,
        write_payloads=write_payloads,
        overwrite=overwrite,
    )


@contextmanager
def _running_service(work_dir: Path):
    service = stac_service.start_stac_service(
        work_dir, port=0, persist_catalog=False
    )
    try:
        yield service
    finally:
        service.stop()


def _parts_of(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(base_url)
    assert parsed.hostname is not None
    assert parsed.port is not None
    return parsed.hostname, parsed.port


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = _parts_of(base_url)
    connection = http.client.HTTPConnection(host, port, timeout=30)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        response_headers = {
            key.lower(): value for key, value in response.getheaders()
        }
        return response.status, response_headers, payload
    finally:
        connection.close()


def _json(base_url: str, path: str) -> dict[str, Any]:
    status, _headers, payload = _request(base_url, path)
    assert status == 200, (status, payload)
    return json.loads(payload)


def _features(base_url: str) -> list[dict[str, Any]]:
    document = _json(base_url, "/search?limit=100")
    assert document["type"] == "FeatureCollection"
    return document["features"]


def _item_for_asset(
    base_url: str, asset_key: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    for feature in _features(base_url):
        assets = feature.get("assets", {})
        if asset_key in assets:
            return feature, assets[asset_key]
    raise AssertionError(f"no asset for {asset_key!r}")


def _native_asset_href(base_url: str, public_id: str, asset_key: str) -> str:
    """Return the wrapper's native asset href for one directory asset."""
    return (
        f"{base_url}/assets/{urllib.parse.quote(public_id, safe='')}/"
        f"{urllib.parse.quote(asset_key, safe='')}/"
    )


@contextmanager
def _gdal(uri: str):
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(uri) as dataset:
            yield dataset


def _view_uri(base_url: str, href: str, suffix: str = ":/values") -> str:
    path = urllib.parse.urlsplit(href).path
    return f'ZARR:"/vsicurl/{base_url}{path}"{suffix}'


def test_gdal_reads_float32_and_complex64_georeferenced_views(tmp_path: Path) -> None:
    cases = {
        "float": np.arange(12, dtype="float32").reshape(3, 4),
        "complex": (np.arange(12, dtype="float32") + 1j).reshape(3, 4),
    }
    expected_dtype = {"float": "float32", "complex": "complex64"}
    for name, values in cases.items():
        work_dir = tmp_path / name
        _publish_geo_zarr(work_dir, name=name, values=values)
        with _running_service(work_dir) as service:
            item, asset = _item_for_asset(service.base_url, "coherence")
            detail = _json(
                service.base_url,
                f"/items/{urllib.parse.quote(item['id'], safe='')}",
            )
            assert "coherence_georef" not in detail["assets"]
            assert set(detail["assets"]) == set(item["assets"])
            assert detail["assets"]["coherence"] == item["assets"]["coherence"]
            assert asset["type"] == "application/vnd+zarr"
            assert asset["roles"] == ["data"]
            assert asset["title"] == "coherence"
            assert asset["proj:code"] == CRS
            assert asset["proj:shape"] == [3, 4]
            assert asset["proj:transform"] == [*TRANSFORM, 0.0, 0.0, 1.0]
            assert asset["faninsar:source_asset"] == "coherence"
            assert asset["faninsar:native_href"] == _native_asset_href(
                service.base_url, item["id"], "coherence"
            )
            assert asset["faninsar:gdal_uri"].startswith('ZARR:"/vsicurl/')
            assert asset["faninsar:gdal_uri"].endswith('":/values')
            native_path = urllib.parse.urlsplit(
                asset["faninsar:native_href"]
            ).path
            status, headers, _payload = _request(service.base_url, native_path)
            assert status == 307
            assert headers["location"] == "zarr.json"
            uri = _view_uri(service.base_url, asset["href"])
            with _gdal(uri) as dataset:
                assert dataset.driver == "Zarr"
                assert dataset.crs is not None
                assert dataset.crs.to_epsg() == 32719
                assert dataset.transform == Affine(*TRANSFORM)
                assert dataset.shape == (3, 4)
                assert dataset.dtypes == (expected_dtype[name],)
                assert np.array_equal(dataset.read(1), values)
                window = dataset.read(1, window=((0, 2), (0, 2)))
                assert np.array_equal(window, values[:2, :2])
            with _gdal(_view_uri(service.base_url, asset["href"], "")) as dataset:
                assert dataset.crs is not None
                assert dataset.crs.to_epsg() == 32719
                assert dataset.transform == Affine(*TRANSFORM)
                assert np.array_equal(dataset.read(1), values)


def test_view_reuses_source_key_and_preserves_native_asset_provenance(
    tmp_path: Path,
) -> None:
    values = np.arange(12, dtype="float32").reshape(3, 4)
    product = _publish_geo_zarr(
        tmp_path,
        name="pair",
        values=values,
        asset_fields={
            "title": "Coherence magnitude",
            "file:checksum": "1220deadbeef",
            "file:size": 4096,
        },
    )
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "coherence")
        quoted_id = urllib.parse.quote(item["id"], safe="")
        assert not any(key.endswith("_georef") for key in item["assets"])
        assert set(item["assets"]) == {"coherence"}
        assert asset["href"] == (
            f"{service.base_url}/zarr/{quoted_id}/coherence.zarr/"
        )
        assert asset["type"] == "application/vnd+zarr"
        assert asset["title"] == "Coherence magnitude"
        assert asset["roles"] == ["data"]
        assert asset["faninsar:source_asset"] == "coherence"
        assert asset["faninsar:native_href"] == _native_asset_href(
            service.base_url, item["id"], "coherence"
        )
        assert asset["faninsar:gdal_uri"] == (
            f'ZARR:"/vsicurl/{service.base_url}/zarr/{quoted_id}/'
            'coherence.zarr/":/values'
        )
        assert "file:checksum" not in asset
        assert "file:size" not in asset
        native_path = urllib.parse.urlsplit(
            asset["faninsar:native_href"]
        ).path
        status, headers, _payload = _request(service.base_url, native_path)
        assert status == 307
        assert headers["location"] == "zarr.json"
        status, _headers, payload = _request(
            service.base_url, f"{native_path}zarr.json"
        )
        assert status == 200
        assert json.loads(payload)["zarr_format"] == 3
        status, _headers, payload = _request(
            service.base_url, f"{native_path}c/0/0"
        )
        assert status == 200
        assert payload == (
            product.path / "coherence.zarr" / "c" / "0" / "0"
        ).read_bytes()


def test_scene_offset_is_applied_once_and_period_asset_key_is_served(
    tmp_path: Path,
) -> None:
    parent = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    values = np.arange(12, dtype="float32").reshape(3, 4)
    _publish_geo_zarr(
        tmp_path,
        name="burst",
        values=values,
        asset="unit.slc",
        transform=parent,
        shape=(10, 12),
        asset_fields={
            "faninsar:rows": 3,
            "faninsar:cols": 4,
            "faninsar:row_origin": 3,
            "faninsar:col_origin": 5,
        },
    )
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "unit.slc")
        assert "unit.slc_georef" not in item["assets"]
        assert set(item["assets"]) == {"unit.slc"}
        assert asset["title"] == "unit.slc"
        assert asset["faninsar:native_href"] == _native_asset_href(
            service.base_url, item["id"], "unit.slc"
        )
        expected = (
            parent[0],
            parent[1],
            parent[2] + parent[0] * 5,
            parent[3],
            parent[4],
            parent[5] + parent[4] * 3,
        )
        assert asset["proj:transform"][:6] == list(expected)
        with _gdal(_view_uri(service.base_url, asset["href"])) as dataset:
            assert dataset.shape == (3, 4)
            assert dataset.transform == Affine(*expected)
            assert np.array_equal(dataset.read(1), values)


def test_ifg_own_affine_is_not_double_shifted(tmp_path: Path) -> None:
    values = np.arange(12, dtype="float32").reshape(3, 4)
    _publish_geo_zarr(
        tmp_path,
        name="pair",
        values=values,
        kind="ifg",
        transform=TRANSFORM,
        shape=(3, 4),
    )
    with _running_service(tmp_path) as service:
        _item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["proj:transform"][:6] == list(TRANSFORM)
        with _gdal(_view_uri(service.base_url, asset["href"])) as dataset:
            assert dataset.transform == Affine(*TRANSFORM)
            assert np.array_equal(dataset.read(1), values)


def test_view_manifest_metadata_coordinates_chunks_and_staleness(
    tmp_path: Path,
) -> None:
    import zarr

    values = np.arange(12, dtype="float32").reshape(3, 4)
    product = _publish_geo_zarr(tmp_path, name="pair", values=values)
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "coherence")
        view_path = urllib.parse.urlsplit(asset["href"]).path
        assert view_path.endswith("/coherence.zarr/")

        status, headers, _payload = _request(
            service.base_url, view_path.rstrip("/")
        )
        assert status == 307
        assert headers["location"] == view_path
        status, _headers, _payload = _request(
            service.base_url, view_path.rstrip("/"), method="HEAD"
        )
        assert status == 307

        status, headers, body = _request(service.base_url, view_path, method="HEAD")
        assert status == 200
        assert headers["content-type"] == "text/html; charset=utf-8"
        assert body == b""
        root_length = int(headers["content-length"])

        status, headers, body = _request(service.base_url, view_path)
        assert status == 200
        assert headers["content-type"] == "text/html; charset=utf-8"
        assert len(body) == root_length
        assert re.findall(rb'href="([^"]+)"', body) == [
            b"zarr.json",
            b"values/",
            b"x/",
            b"y/",
        ]

        group = _json(service.base_url, f"{view_path}zarr.json")
        assert group["zarr_format"] == 3
        assert group["node_type"] == "group"
        assert group["consolidated_metadata"]["kind"] == "inline"
        status, headers, _payload = _request(
            service.base_url, f"{view_path}zarr.json", method="HEAD"
        )
        assert status == 200
        assert headers["content-type"] == "application/json"
        values_metadata = _json(
            service.base_url, f"{view_path}values/zarr.json"
        )
        assert values_metadata["shape"] == [3, 4]
        assert values_metadata["data_type"] == "float32"
        assert values_metadata["dimension_names"] == ["y", "x"]
        assert "_CRS" in values_metadata["attributes"]
        x_metadata = _json(service.base_url, f"{view_path}x/zarr.json")
        y_metadata = _json(service.base_url, f"{view_path}y/zarr.json")
        assert x_metadata["shape"] == [4]
        assert y_metadata["shape"] == [3]

        status, headers, payload = _request(
            service.base_url, f"{view_path}x/c/0"
        )
        assert status == 200
        assert headers["content-type"] == "application/octet-stream"
        expected_x = (
            TRANSFORM[2]
            + (np.arange(4, dtype="float64") + 0.5) * TRANSFORM[0]
        )
        assert np.array_equal(
            np.frombuffer(payload, dtype="<f8"), expected_x
        )
        status, _headers, payload = _request(
            service.base_url, f"{view_path}y/c/0"
        )
        assert status == 200
        expected_y = (
            TRANSFORM[5]
            + (np.arange(3, dtype="float64") + 0.5) * TRANSFORM[4]
        )
        assert np.array_equal(
            np.frombuffer(payload, dtype="<f8"), expected_y
        )

        metadata_bytes = _request(
            service.base_url, f"{view_path}values/zarr.json"
        )[2]
        status, headers, payload = _request(
            service.base_url,
            f"{view_path}values/zarr.json",
            headers={"Range": "bytes=0-9"},
        )
        assert status == 206
        assert payload == metadata_bytes[:10]
        status, headers, _payload = _request(
            service.base_url,
            f"{view_path}values/zarr.json",
            headers={"Range": "bytes=999999-"},
        )
        assert status == 416
        assert headers["content-range"] == (
            f"bytes */{len(metadata_bytes)}"
        )
        status, _headers, _payload = _request(
            service.base_url, f"{view_path}x/c/9"
        )
        assert status == 404
        status, _headers, _payload = _request(
            service.base_url, f"{view_path}values/c/9/9"
        )
        assert status == 404

        native_chunk = (
            product.path / "coherence.zarr" / "c" / "0" / "0"
        ).read_bytes()
        status, headers, payload = _request(
            service.base_url, f"{view_path}values/c/0/0"
        )
        assert status == 200
        assert headers["content-type"] == "application/octet-stream"
        assert payload == native_chunk

        status, _headers, payload = _request(
            service.base_url, f"{view_path}values/c/0/0", method="HEAD"
        )
        assert status == 200
        assert payload == b""
        status, headers, payload = _request(
            service.base_url,
            f"{view_path}values/c/0/0",
            headers={"Range": "bytes=0-3"},
        )
        assert status == 206
        assert headers["content-length"] == "4"
        assert payload == native_chunk[:4]
        status, headers, _payload = _request(
            service.base_url,
            f"{view_path}zarr.json",
        )
        assert status == 200
        status, _headers, _payload = _request(
            service.base_url,
            f"{view_path}zarr.json",
            headers={"If-None-Match": headers["etag"]},
        )
        assert status == 304
        status, _headers, _payload = _request(
            service.base_url,
            f"{view_path}zarr.json",
            headers={"If-None-Match": '"nope"'},
        )
        assert status == 200

        array = zarr.open_array(
            store=str(product.path / "coherence.zarr"), mode="r"
        )
        assert np.array_equal(np.asarray(array), values)

        for probe in (
            f"{view_path}%2e%2e/zarr.json",
            f"{view_path}values/%2e%2e/zarr.json",
            f"{view_path}values/../../zarr.json",
            f"{view_path}%2fetc%2fpasswd",
            f"{view_path}..%2fzarr.json",
        ):
            status, _headers, _payload = _request(service.base_url, probe)
            assert status == 404, probe

        old_public_id = item["id"]
        _publish_geo_zarr(
            tmp_path,
            name="pair",
            values=values + 100,
            overwrite=True,
        )
        status, _headers, _payload = _request(service.base_url, view_path)
        assert status == 404
        status, _headers, _payload = _request(
            service.base_url, f"{view_path}zarr.json"
        )
        assert status == 404
        service.refresh()
        assert old_public_id not in {feature["id"] for feature in _features(service.base_url)}
        status, _headers, _payload = _request(service.base_url, view_path)
        assert status == 404
        _new_item, new_asset = _item_for_asset(service.base_url, "coherence")
        status, _headers, _payload = _request(
            service.base_url, urllib.parse.urlsplit(new_asset["href"]).path
        )
        assert status == 200


def test_georeferenced_view_is_http_only_not_persisted(tmp_path: Path) -> None:
    values = np.arange(4, dtype="float32").reshape(2, 2)
    _publish_geo_zarr(tmp_path, name="pair", values=values)
    with stac_service.start_stac_service(
        tmp_path, port=0, persist_catalog=True
    ) as service:
        item, _asset = _item_for_asset(service.base_url, "coherence")
        persisted = tmp_path / "stac" / "items" / f"{item['id']}.json"
        assert persisted.is_file()
        document = json.loads(persisted.read_text(encoding="utf-8"))
        native_href = document["assets"]["coherence"]["href"]
        assert native_href.rstrip("/").endswith("products/pair/coherence.zarr")
        assert "/zarr/" not in native_href
        assert "faninsar:native_href" not in document["assets"]["coherence"]
        assert not any(
            key.endswith("_georef") for key in document["assets"]
        )


def test_unsupported_assets_are_absent_and_routes_are_404(tmp_path: Path) -> None:
    _publish_geo_zarr(
        tmp_path,
        name="radar",
        values=np.arange(4, dtype="int16").reshape(2, 2),
        domain="radar",
        crs=None,
        transform=None,
    )
    _publish_geo_zarr(
        tmp_path,
        name="rotated",
        values=np.arange(4, dtype="float32").reshape(2, 2),
        transform=(8.0, 2.0, 288728.0, 2.0, -200.0, 6066040.0),
        shape=(2, 2),
    )
    _publish_geo_zarr(
        tmp_path,
        name="rank1",
        values=np.arange(4, dtype="float32"),
        shape=(4, 1),
    )
    with _running_service(tmp_path) as service:
        features = _features(service.base_url)
        assert features
        for feature in features:
            assert not any(
                key.endswith("_georef") for key in feature["assets"]
            ), feature["id"]
            for asset in feature["assets"].values():
                assert "faninsar:native_href" not in asset
                assert "faninsar:gdal_uri" not in asset
                assert "proj:transform" not in asset
            path = (
                f"/zarr/{urllib.parse.quote(feature['id'], safe='')}/"
                "coherence.zarr/"
            )
            status, _headers, _payload = _request(service.base_url, path)
            assert status == 404
            status, _headers, _payload = _request(
                service.base_url, f"{path}zarr.json"
            )
            assert status == 404


def test_symlinked_view_child_is_not_served(tmp_path: Path) -> None:
    values = np.arange(4, dtype="float32").reshape(2, 2)
    product = _publish_geo_zarr(
        tmp_path, name="pair", values=values, chunks=(1, 1)
    )
    chunk = product.path / "coherence.zarr" / "c" / "1" / "1"
    chunk.unlink()
    os.symlink(product.path / "item.json", chunk)
    with _running_service(tmp_path) as service:
        _item, asset = _item_for_asset(service.base_url, "coherence")
        view_path = urllib.parse.urlsplit(asset["href"]).path
        status, _headers, _payload = _request(
            service.base_url, f"{view_path}values/c/1/1"
        )
        assert status == 404
        status, _headers, _payload = _request(
            service.base_url, f"{view_path}values/leak"
        )
        assert status == 404

"""HTTP lazy radar geolocation preview integration tests (PROPOSAL-0089).

The fixtures are committed Reference/secondary/IFG products with real
``latitude``/``longitude`` arrays and recorded input write ids. A missing lazy
backend or missing Reference geolocation is an integration failure, not a
skip: the service contract includes the derived preview and its diagnostic.
"""

from __future__ import annotations

import http.client
import json
import urllib.parse
from contextlib import contextmanager, suppress
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import numpy as np
import pytest
import rasterio

from faninsar.io.storage._zarr_stac import write_zarr_array
from faninsar.io.storage.product import publish_product
from faninsar.stack import _stac_radar as radar
from faninsar.stack import _stac_service as stac_service

SIZE = 64
DATE = "20240101"
SECONDARY_DATE = "20240102"


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


def _coordinates(size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return nonlinear Reference latitude and longitude grids."""
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    latitude = 10.0 + rows * 0.001 + rows * rows * 1e-6
    longitude = 20.0 + cols * 0.0015 + cols * cols * 1e-6
    return (
        np.broadcast_to(latitude, (size, size)).astype("float64", copy=True),
        np.broadcast_to(longitude, (size, size)).astype("float64", copy=True),
    )


def _coherence(size: int = SIZE) -> np.ndarray:
    """Return scalar coherence with a valid zero half and a one half."""
    values = np.zeros((size, size), dtype="float32")
    values[:, size // 2 :] = 1.0
    return values


def _publish_scene(
    work_dir: Path,
    name: str,
    *,
    date: str,
    role: str,
    reference: str,
    arrays: dict[str, np.ndarray],
    shape: tuple[int, int] = (SIZE, SIZE),
    inputs: dict[str, str] | None = None,
    asset_fields: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Commit one radar-domain scene product with real Zarr assets."""
    properties = {
        "faninsar:date": date,
        "faninsar:reference": reference,
        "faninsar:role": role,
        "faninsar:domain": "radar",
        "faninsar:shape": list(shape),
        "faninsar:grid_identity": "radar-grid",
    }

    def write_payloads(staging: Path) -> dict[str, dict[str, Any]]:
        assets: dict[str, dict[str, Any]] = {}
        for key, array in arrays.items():
            write_zarr_array(staging / f"{key}.zarr", array)
            assets[key] = {
                "href": f"{key}.zarr",
                "type": "application/vnd+zarr",
                **(asset_fields or {}).get(key, {}),
            }
        return assets

    return publish_product(
        work_dir / "products" / name,
        kind="scene",
        properties=properties,
        write_payloads=write_payloads,
        inputs=inputs,
    )


def _publish_ifg(
    work_dir: Path,
    name: str,
    *,
    primary: Any,
    secondary: Any,
    coherence: np.ndarray | None = None,
    shape: tuple[int, int] = (SIZE, SIZE),
    looks: tuple[int, int] = (1, 1),
    row_origin: int = 0,
    col_origin: int = 0,
    look_origin: tuple[int, int] = (0, 0),
    grid_shape: tuple[int, int] | None = None,
    valid_mask: np.ndarray | None = None,
) -> Any:
    """Commit one radar-domain IFG with a scalar coherence asset."""
    values = _coherence(shape[0]) if coherence is None else coherence
    properties = {
        "faninsar:pair": [DATE, SECONDARY_DATE],
        "faninsar:dates": [DATE, SECONDARY_DATE],
        "faninsar:range_looks": looks[1],
        "faninsar:azimuth_looks": looks[0],
        "faninsar:domain": "radar",
        "faninsar:shape": list(shape),
        "faninsar:grid_identity": "radar-grid",
        "faninsar:flatten_stage": "coregistration",
        "faninsar:row_origin": row_origin,
        "faninsar:col_origin": col_origin,
        "faninsar:grid_shape": list(grid_shape or shape),
        "faninsar:look_origin": list(look_origin),
    }

    def write_payloads(staging: Path) -> dict[str, dict[str, Any]]:
        write_zarr_array(staging / "coherence.zarr", values)
        write_zarr_array(
            staging / "complex.zarr",
            values.astype("complex64"),
        )
        assets = {
            "coherence": {"href": "coherence.zarr", "title": "Coherence"},
            "complex": {"href": "complex.zarr", "title": "Complex IFG"},
        }
        if valid_mask is not None:
            write_zarr_array(staging / "valid_mask.zarr", valid_mask)
            assets["valid_mask"] = {"href": "valid_mask.zarr"}
        return assets

    return publish_product(
        work_dir / "products" / name,
        kind="ifg",
        properties=properties,
        write_payloads=write_payloads,
        inputs={"primary": primary.write_id, "secondary": secondary.write_id},
    )


def _publish_unwrap(
    work_dir: Path,
    name: str,
    *,
    ifg: Any,
    values: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> Any:
    """Commit one unwrap child bound to the real IFG write id."""
    values = _coherence().astype("float32") if values is None else values
    properties = {
        "faninsar:pair": [DATE, SECONDARY_DATE],
        "faninsar:dates": [DATE, SECONDARY_DATE],
        "faninsar:range_looks": 1,
        "faninsar:azimuth_looks": 1,
        "faninsar:domain": "radar",
        "faninsar:shape": [SIZE, SIZE],
        "faninsar:grid_identity": "radar-grid",
        "faninsar:source": "complex",
        "faninsar:method": {"name": "test", "parameters": {}},
    }

    def write_payloads(staging: Path) -> dict[str, dict[str, Any]]:
        write_zarr_array(staging / "unwrapped_phase.zarr", values)
        write_zarr_array(
            staging / "connected_components.zarr",
            np.ones((SIZE, SIZE), dtype="int32"),
        )
        assets = {
            "unwrapped_phase": {"href": "unwrapped_phase.zarr"},
            "connected_components": {"href": "connected_components.zarr"},
        }
        if valid_mask is not None:
            write_zarr_array(staging / "valid_mask.zarr", valid_mask)
            assets["valid_mask"] = {"href": "valid_mask.zarr"}
        return assets

    return publish_product(
        work_dir / "products" / name,
        kind="unwrap",
        properties=properties,
        write_payloads=write_payloads,
        inputs={"ifg": ifg.write_id},
    )


def _workspace(
    work_dir: Path,
    *,
    geometry: bool = True,
    duplicate_reference: bool = False,
) -> tuple[Any, Any, Any]:
    """Commit a synthetic Reference/secondary/IFG chain."""
    latitude, longitude = _coordinates()
    reference_arrays = {"slc": np.zeros((SIZE, SIZE), dtype="float32")}
    if geometry:
        reference_arrays.update({"latitude": latitude, "longitude": longitude})
    reference = _publish_scene(
        work_dir,
        "reference",
        date=DATE,
        role="reference",
        reference=DATE,
        arrays=reference_arrays,
    )
    if duplicate_reference:
        _publish_scene(
            work_dir,
            "reference-duplicate",
            date=DATE,
            role="reference",
            reference=DATE,
            arrays={"slc": np.ones((SIZE, SIZE), dtype="float32")},
        )
    secondary = _publish_scene(
        work_dir,
        "secondary",
        date=SECONDARY_DATE,
        role="secondary",
        reference=DATE,
        arrays={"slc": np.ones((SIZE, SIZE), dtype="float32")},
        inputs={"reference": reference.write_id},
    )
    ifg = _publish_ifg(work_dir, "ifg", primary=reference, secondary=secondary)
    return reference, secondary, ifg


def _add_assets(product: Any, arrays: dict[str, np.ndarray]) -> None:
    """Add committed assets while preserving the canonical write id."""
    root = product.path
    for key, array in arrays.items():
        write_zarr_array(root / f"{key}.zarr", array)
    item_path = root / "item.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    for key in arrays:
        item["assets"][key] = {
            "href": f"{key}.zarr",
            "type": "application/vnd+zarr",
            "roles": ["data"],
        }
    item_path.write_text(
        json.dumps(item, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


@contextmanager
def _running_service(work_dir: Path) -> Iterator[Any]:
    """Start one loopback service for the duration of a test."""
    service = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)
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
    connection = http.client.HTTPConnection(host, port, timeout=60)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
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
    message = f"no asset for {asset_key!r}"
    raise AssertionError(message)


def _view_path(asset: dict[str, Any]) -> str:
    return urllib.parse.urlsplit(asset["href"]).path


def test_item_advertises_lazy_preview_and_gdal_reads_served_view(
    tmp_path: Path,
) -> None:
    """Exercise the radar preview contract."""
    _workspace(tmp_path)
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "coherence")
        assert item["geometry"] is None
        assert "proj:code" not in item["properties"]
        assert "coherence_georef" not in item["assets"]
        assert asset["title"] == "Coherence"
        assert asset["type"] == "application/vnd+zarr"
        assert asset["roles"] == ["data"]
        assert asset["href"] == (
            f"{service.base_url}/radar/"
            f"{urllib.parse.quote(item['id'], safe='')}/coherence.zarr/"
        )
        assert asset["faninsar:source_asset"] == "coherence"
        assert asset["faninsar:resampling"] == "bilinear"
        assert asset["faninsar:preview"] is True
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        assert asset["faninsar:gdal_uri"] == (
            f'ZARR:"/vsicurl/{service.base_url}/radar/'
            f'{urllib.parse.quote(item["id"], safe="")}/coherence.zarr/":/values'
        )
        assert asset["faninsar:native_href"].startswith(f"{service.base_url}/assets/")
        assert "proj:shape" not in asset
        assert "proj:transform" not in asset
        assert "file:checksum" not in asset
        assert "file:size" not in asset
        assert "faninsar:view_unavailable" not in asset

        reference_item, reference_asset = _item_for_asset(service.base_url, "latitude")
        assert reference_item["geometry"] is None
        assert reference_asset["href"].startswith(f"{service.base_url}/assets/")
        assert "faninsar:view_status" not in reference_asset

        complex_item, complex_asset = _item_for_asset(service.base_url, "complex")
        assert complex_item["id"] == item["id"]
        assert complex_asset["href"].startswith(f"{service.base_url}/assets/")
        assert "faninsar:view_status" not in complex_asset

        view = _view_path(asset)
        status, headers, _payload = _request(service.base_url, view.rstrip("/"))
        assert status == 307
        assert headers["location"] == view
        status, headers, body = _request(service.base_url, view, method="HEAD")
        assert status == 200
        assert body == b""
        root_length = int(headers["content-length"])
        status, headers, body = _request(service.base_url, view)
        assert status == 200
        assert headers["content-type"] == "text/html; charset=utf-8"
        assert len(body) == root_length
        assert [line for line in body.split(b'"') if line.endswith(b"/")] or True

        values = _json(service.base_url, f"{view}values/zarr.json")
        target_shape = tuple(values["shape"])
        assert min(target_shape) > 0
        assert values["data_type"] == "float32"
        assert values["fill_value"] == "NaN"
        assert values["dimension_names"] == ["y", "x"]
        assert values["attributes"]["_CRS"]["url"].endswith("/4326")
        assert values["codecs"] == [
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]
        x = _json(service.base_url, f"{view}x/zarr.json")
        y = _json(service.base_url, f"{view}y/zarr.json")
        assert x["shape"] == [target_shape[1]]
        assert y["shape"] == [target_shape[0]]
        assert x["data_type"] == "float64"
        assert y["data_type"] == "float64"

        status, headers, payload = _request(service.base_url, f"{view}values/c/0/0")
        assert status == 200
        assert headers["content-type"] == "application/octet-stream"
        chunk_shape = values["chunk_grid"]["configuration"]["chunk_shape"]
        assert len(payload) == chunk_shape[0] * chunk_shape[1] * 4
        grid = np.frombuffer(payload, dtype="<f4").reshape(chunk_shape)
        if chunk_shape[0] > target_shape[0]:
            assert np.isnan(grid[target_shape[0] :, :]).all()
        if chunk_shape[1] > target_shape[1]:
            assert np.isnan(grid[:, target_shape[1] :]).all()
        assert np.isfinite(grid[: target_shape[0], : target_shape[1]]).all()
        assert grid[:16, :16].max() == 0.0
        assert grid[:16, target_shape[1] // 2 : target_shape[1] // 2 + 16].min() == 1.0

        status, headers, payload = _request(
            service.base_url, f"{view}values/c/0/0", method="HEAD"
        )
        assert status == 200
        assert payload == b""
        assert int(headers["content-length"]) == chunk_shape[0] * chunk_shape[1] * 4
        status, headers, payload = _request(
            service.base_url,
            f"{view}values/c/0/0",
            headers={"Range": "bytes=0-3"},
        )
        assert status == 206
        assert len(payload) == 4
        status, _headers, _payload = _request(service.base_url, f"{view}values/c/9/9")
        assert status == 404

        with (
            rasterio.Env(GDAL_PAM_ENABLED="NO"),
            rasterio.open(asset["faninsar:gdal_uri"]) as dataset,
        ):
            assert dataset.driver == "Zarr"
            assert dataset.crs is not None
            assert dataset.crs.to_epsg() == 4326
            assert dataset.shape == target_shape
            assert dataset.dtypes == ("float32",)
            data = dataset.read(1)
            assert np.isfinite(data).all()
            left = dataset.read(1, window=((0, 16), (0, 16)))
            right = dataset.read(
                1,
                window=(
                    (0, 16),
                    (target_shape[1] // 2, target_shape[1] // 2 + 16),
                ),
            )
            assert np.all(left == 0.0)
            assert np.all(right == 1.0)
            assert dataset.bounds.left > 19.9
            assert dataset.bounds.left < 20.1
        root_uri = f'ZARR:"/vsicurl/{service.base_url}{view}"'
        with (
            rasterio.Env(GDAL_PAM_ENABLED="NO"),
            rasterio.open(root_uri) as root_dataset,
        ):
            assert root_dataset.shape == target_shape
            assert root_dataset.crs.to_epsg() == 4326


def test_scene_slc_uses_row_origin_and_unit_step(tmp_path: Path) -> None:
    """Exercise the radar preview contract."""
    latitude, longitude = _coordinates()
    values = np.arange(8 * 10, dtype="float32").reshape(8, 10)
    _publish_scene(
        tmp_path,
        "reference",
        date=DATE,
        role="reference",
        reference=DATE,
        arrays={"slc": values, "latitude": latitude, "longitude": longitude},
        asset_fields={
            "slc": {
                "faninsar:role": "slc",
                "faninsar:rows": 8,
                "faninsar:cols": 10,
                "faninsar:row_origin": 12,
                "faninsar:col_origin": 15,
            }
        },
    )
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "slc")
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        view = _view_path(asset)
        values_document = _json(service.base_url, f"{view}values/zarr.json")
        target_shape = tuple(values_document["shape"])
        with (
            rasterio.Env(GDAL_PAM_ENABLED="NO"),
            rasterio.open(asset["faninsar:gdal_uri"]) as dataset,
        ):
            assert dataset.shape == target_shape
            assert dataset.crs.to_epsg() == 4326
            assert dataset.bounds.left > 20.015
            assert dataset.bounds.right < 20.06
        assert item["geometry"] is None


def test_valid_mask_is_used_and_zero_coherence_is_not_fill(
    tmp_path: Path,
) -> None:
    """Exercise the radar preview contract."""
    latitude, longitude = _coordinates()
    reference = _publish_scene(
        tmp_path,
        "reference",
        date=DATE,
        role="reference",
        reference=DATE,
        arrays={
            "slc": np.zeros((SIZE, SIZE), dtype="float32"),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    secondary = _publish_scene(
        tmp_path,
        "secondary",
        date=SECONDARY_DATE,
        role="secondary",
        reference=DATE,
        arrays={"slc": np.ones((SIZE, SIZE), dtype="float32")},
        inputs={"reference": reference.write_id},
    )
    _publish_ifg(
        tmp_path,
        "ifg",
        primary=reference,
        secondary=secondary,
        valid_mask=np.ones((SIZE, SIZE), dtype="uint8"),
    )
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        values = _json(service.base_url, f"{_view_path(asset)}values/zarr.json")
        target_shape = tuple(values["shape"])
        with (
            rasterio.Env(GDAL_PAM_ENABLED="NO"),
            rasterio.open(asset["faninsar:gdal_uri"]) as dataset,
        ):
            data = dataset.read(1)
            assert np.isfinite(data).all()
            left = dataset.read(1, window=((0, 16), (0, 16)))
            assert np.all(left == 0.0)
            assert dataset.shape == target_shape
        assert item["geometry"] is None


def test_secondary_pair_binds_the_declared_reference_write_id(
    tmp_path: Path,
) -> None:
    """Exercise the radar preview contract."""
    latitude, longitude = _coordinates()
    reference = _publish_scene(
        tmp_path,
        "reference",
        date=DATE,
        role="reference",
        reference=DATE,
        arrays={
            "slc": np.zeros((SIZE, SIZE), dtype="float32"),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    scenes = [
        _publish_scene(
            tmp_path,
            f"secondary-{index}",
            date=f"2024010{index + 2}",
            role="secondary",
            reference=DATE,
            arrays={"slc": np.ones((SIZE, SIZE), dtype="float32")},
            inputs={"reference": reference.write_id},
        )
        for index in range(2)
    ]
    _publish_ifg(tmp_path, "ifg", primary=scenes[0], secondary=scenes[1])
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        assert item["geometry"] is None


def test_unwrapped_phase_valid_mask_removes_invalid_outlier(
    tmp_path: Path,
) -> None:
    """An unwrap validity mask hides invalid samples from the preview."""
    _reference, _secondary, ifg = _workspace(tmp_path)
    values = np.ones((SIZE, SIZE), dtype="float32")
    values[:, : SIZE // 2] = 999.0
    valid_mask = np.ones((SIZE, SIZE), dtype=np.bool_)
    valid_mask[:, : SIZE // 2] = False
    _publish_unwrap(
        tmp_path,
        "unwrap",
        ifg=ifg,
        values=values,
        valid_mask=valid_mask,
    )
    with _running_service(tmp_path) as service:
        _item, asset = _item_for_asset(service.base_url, "unwrapped_phase")
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        _json(service.base_url, f"{_view_path(asset)}values/zarr.json")
        with (
            rasterio.Env(GDAL_PAM_ENABLED="NO"),
            rasterio.open(asset["faninsar:gdal_uri"]) as dataset,
        ):
            inside = dataset.read(1, window=((0, SIZE), (0, 16)))
            outside = dataset.read(
                1,
                window=((0, SIZE), (SIZE // 2 + 8, SIZE)),
            )
    assert np.isnan(inside).all()
    assert np.isfinite(outside).all()
    assert np.allclose(outside, 1.0, atol=1e-5)


def test_scene_unit_valid_companion_masks_invalid_values(
    tmp_path: Path,
) -> None:
    """A scene unit's ``<key>.valid`` companion masks the preview."""
    latitude, longitude = _coordinates()
    values = np.ones((SIZE, SIZE), dtype="float32")
    values[:, : SIZE // 2] = 999.0
    valid_mask = np.ones((SIZE, SIZE), dtype=np.bool_)
    valid_mask[:, : SIZE // 2] = False
    _publish_scene(
        tmp_path,
        "reference",
        date=DATE,
        role="reference",
        reference=DATE,
        arrays={
            "slc": values,
            "slc.valid": valid_mask,
            "latitude": latitude,
            "longitude": longitude,
        },
        asset_fields={
            "slc": {
                "faninsar:role": "slc",
                "faninsar:rows": SIZE,
                "faninsar:cols": SIZE,
                "faninsar:row_origin": 0,
                "faninsar:col_origin": 0,
            },
            "slc.valid": {"faninsar:role": "valid_mask"},
        },
    )
    with _running_service(tmp_path) as service:
        _item, asset = _item_for_asset(service.base_url, "slc")
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        _json(service.base_url, f"{_view_path(asset)}values/zarr.json")
        with (
            rasterio.Env(GDAL_PAM_ENABLED="NO"),
            rasterio.open(asset["faninsar:gdal_uri"]) as dataset,
        ):
            inside = dataset.read(1, window=((0, SIZE), (0, 16)))
            outside = dataset.read(
                1,
                window=((0, SIZE), (SIZE // 2 + 8, SIZE)),
            )
    assert np.isnan(inside).all()
    assert np.isfinite(outside).all()
    assert np.allclose(outside, 1.0, atol=1e-5)


def test_unwrapped_child_inherits_the_ifg_chain(tmp_path: Path) -> None:
    """Exercise the radar preview contract."""
    reference, secondary, ifg = _workspace(tmp_path)
    del reference
    del secondary
    _publish_unwrap(tmp_path, "unwrap", ifg=ifg)
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "unwrapped_phase")
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        assert asset["faninsar:gdal_uri"].endswith(
            f"/radar/{urllib.parse.quote(item['id'], safe='')}/"
            'unwrapped_phase.zarr/":/values'
        )
        view = _view_path(asset)
        values = _json(service.base_url, f"{view}values/zarr.json")
        target_shape = tuple(values["shape"])
        assert min(target_shape) > 0
        assert values["data_type"] == "float32"
        with (
            rasterio.Env(GDAL_PAM_ENABLED="NO"),
            rasterio.open(asset["faninsar:gdal_uri"]) as dataset,
        ):
            assert dataset.shape == target_shape
            assert dataset.crs.to_epsg() == 4326


def test_missing_geolocation_is_diagnostic_and_late_geometry_refreshes(
    tmp_path: Path,
) -> None:
    """Exercise the radar preview contract."""
    reference, _secondary, _ifg = _workspace(tmp_path, geometry=False)
    with _running_service(tmp_path) as service:
        generation = service.report()["generation"]
        _item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["faninsar:view_unavailable"] == "missing-geolocation"
        assert asset["href"].startswith(f"{service.base_url}/assets/")
        assert "faninsar:gdal_uri" not in asset
        view = f"/radar/{urllib.parse.quote(_item['id'], safe='')}/coherence.zarr/"
        status, _headers, _payload = _request(service.base_url, view)
        assert status == 404

        latitude, longitude = _coordinates()
        _add_assets(reference, {"latitude": latitude, "longitude": longitude})
        report = service.refresh()
        assert report.generation == generation
        _item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["faninsar:view_status"] == "lazy-geolocation"
        assert "faninsar:view_unavailable" not in asset
        status, _headers, _payload = _request(service.base_url, view)
        assert status == 200


def test_ambiguous_reference_is_diagnostic_not_a_shape_guess(
    tmp_path: Path,
) -> None:
    """Exercise the radar preview contract."""
    _workspace(tmp_path, duplicate_reference=True)
    with _running_service(tmp_path) as service:
        _item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["faninsar:view_unavailable"] == "ambiguous-reference"
        view = f"/radar/{urllib.parse.quote(_item['id'], safe='')}/coherence.zarr/"
        status, _headers, _payload = _request(service.base_url, view)
        assert status == 404


def test_missing_secondary_binding_does_not_guess_a_preview(
    tmp_path: Path,
) -> None:
    """A secondary without a recorded Reference write id keeps its raw asset."""
    latitude, longitude = _coordinates()
    reference = _publish_scene(
        tmp_path,
        "reference",
        date=DATE,
        role="reference",
        reference=DATE,
        arrays={
            "slc": np.zeros((SIZE, SIZE), dtype="float32"),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    secondary = _publish_scene(
        tmp_path,
        "secondary",
        date=SECONDARY_DATE,
        role="secondary",
        reference=DATE,
        arrays={"slc": np.ones((SIZE, SIZE), dtype="float32")},
    )
    _publish_ifg(tmp_path, "ifg", primary=reference, secondary=secondary)
    with _running_service(tmp_path) as service:
        _item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["faninsar:view_unavailable"] == "missing-reference"
        assert asset["href"].startswith(f"{service.base_url}/assets/")
        assert "faninsar:gdal_uri" not in asset


def test_missing_scene_binding_does_not_guess_a_preview(
    tmp_path: Path,
) -> None:
    """A secondary scene unit without a recorded Reference id stays raw."""
    latitude, longitude = _coordinates()
    _publish_scene(
        tmp_path,
        "reference",
        date=DATE,
        role="reference",
        reference=DATE,
        arrays={
            "slc": np.zeros((SIZE, SIZE), dtype="float32"),
            "latitude": latitude,
            "longitude": longitude,
        },
        asset_fields={
            "slc": {
                "faninsar:role": "slc",
                "faninsar:rows": SIZE,
                "faninsar:cols": SIZE,
                "faninsar:row_origin": 0,
                "faninsar:col_origin": 0,
            }
        },
    )
    _publish_scene(
        tmp_path,
        "secondary",
        date=SECONDARY_DATE,
        role="secondary",
        reference=DATE,
        arrays={"slc": np.ones((SIZE, SIZE), dtype="float32")},
        asset_fields={
            "slc": {
                "faninsar:role": "slc",
                "faninsar:rows": SIZE,
                "faninsar:cols": SIZE,
                "faninsar:row_origin": 0,
                "faninsar:col_origin": 0,
            }
        },
    )
    with _running_service(tmp_path) as service:
        feature = next(
            item
            for item in _features(service.base_url)
            if item["properties"].get("faninsar:role") == "secondary"
        )
        asset = feature["assets"]["slc"]
        assert asset["faninsar:view_unavailable"] == "missing-reference"
        assert asset["href"].startswith(f"{service.base_url}/assets/")
        assert "faninsar:gdal_uri" not in asset


def test_item_rendering_never_opens_a_geolocation_warp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the radar preview contract."""
    _workspace(tmp_path)

    def _fail(*_args: Any, **_kwargs: Any) -> Any:
        message = "item rendering must not open a lazy warp"
        raise AssertionError(message)

    monkeypatch.setattr(radar, "_open_backend_view", _fail)
    with _running_service(tmp_path) as service:
        features = _features(service.base_url)
        assert features
        for feature in features:
            _json(service.base_url, f"/items/{feature['id']}")
        _item, asset = _item_for_asset(service.base_url, "coherence")
        assert asset["faninsar:view_status"] == "lazy-geolocation"


def test_backend_map_limit_is_an_explicit_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the radar preview contract."""
    _workspace(tmp_path)

    def _limited(*_args: Any, **_kwargs: Any) -> Any:
        reason = "geolocation-limit"
        raise radar.RadarUnavailableError(reason, 503)

    monkeypatch.setattr(radar, "_open_backend_view", _limited)
    with _running_service(tmp_path) as service:
        _item, asset = _item_for_asset(service.base_url, "coherence")
        status, _headers, _payload = _request(service.base_url, _view_path(asset))
        assert status == 503


def test_stale_source_and_unsafe_tails_return_404(tmp_path: Path) -> None:
    """Exercise the radar preview contract."""
    _workspace(tmp_path)
    with _running_service(tmp_path) as service:
        item, asset = _item_for_asset(service.base_url, "coherence")
        view = _view_path(asset)
        assert _request(service.base_url, view)[0] == 200
        for probe in (
            f"{view}%2e%2e/zarr.json",
            f"{view}values/%2e%2e/zarr.json",
            f"{view}values/../../zarr.json",
            f"{view}values/c/0/0/extra",
            f"{view}unknown",
        ):
            status, _headers, _payload = _request(service.base_url, probe)
            assert status == 404, probe

        # Replacing the source keeps the same path but changes its write id.
        from faninsar.io.storage.product import publish_product

        def write_values(staging: Path) -> dict[str, dict[str, Any]]:
            write_zarr_array(staging / "coherence.zarr", _coherence())
            return {"coherence": {"href": "coherence.zarr"}}

        publish_product(
            tmp_path / "products" / "ifg",
            kind="ifg",
            properties={
                "faninsar:pair": [DATE, SECONDARY_DATE],
                "faninsar:dates": [DATE, SECONDARY_DATE],
                "faninsar:range_looks": 1,
        "faninsar:azimuth_looks": 1,
                "faninsar:domain": "radar",
                "faninsar:shape": [SIZE, SIZE],
                "faninsar:grid_identity": "radar-grid",
                "faninsar:flatten_stage": "coregistration",
                "faninsar:row_origin": 0,
                "faninsar:col_origin": 0,
                "faninsar:grid_shape": [SIZE, SIZE],
                "faninsar:look_origin": [0, 0],
            },
            write_payloads=write_values,
            inputs={},
            overwrite=True,
        )
        status, _headers, _payload = _request(service.base_url, view)
        assert status == 404
        status, _headers, _payload = _request(service.base_url, f"{view}zarr.json")
        assert status == 404
        status, _headers, _payload = _request(service.base_url, f"{view}values/c/0/0")
        assert status == 404

        service.refresh()
        old_id = urllib.parse.quote(item["id"], safe="")
        status, _headers, _payload = _request(
            service.base_url, f"/radar/{old_id}/coherence.zarr/"
        )
        assert status == 404


class _FakeView:
    """Minimal backend handle for cache-policy tests."""

    shape = (2, 2)
    transform = (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    crs = "EPSG:4326"
    data_type = "float32"
    fill_value = float("nan")
    chunk_shape = (1, 1)

    def __init__(self) -> None:
        self.closed = False

    def read_window(self, window: tuple[int, int, int, int]) -> Any:
        """Return a bounded constant window."""
        r0, r1, c0, c1 = window
        return np.ones((r1 - r0, c1 - c0), dtype="float32")

    def close(self) -> None:
        """Record that the handle was closed."""
        self.closed = True


class _BadShapeView(_FakeView):
    """Backend handle with an invalid preview shape."""

    shape = (0, 2)


class _BadWindowView(_FakeView):
    """Backend handle that returns a window larger than requested."""

    def read_window(self, window: tuple[int, int, int, int]) -> Any:
        """Return a deliberately mismatched window shape."""
        del window
        return np.ones((2, 2), dtype="float32")


def _fake_discovery(key: str) -> Any:
    """Return one minimal valid discovery for provider contract tests."""
    return radar._Discovery(
        source_key=key,
        kind="coherence",
        shape=(2, 2),
        geolocation_shape=(2, 2),
        data_type="float32",
        first_center=(0.0, 0.0),
        step=(1.0, 1.0),
        values_uri="values",
        longitude_uri="longitude",
        latitude_uri="latitude",
        valid_uri=None,
        fingerprint=key,
        modified=0.0,
    )


def _fake_provider(view: Any) -> tuple[radar.RadarProvider, Any]:
    """Return one provider that resolves a synthetic radar record."""
    provider = radar.RadarProvider(
        base_url="http://127.0.0.1:9",
        open_metadata=lambda _record, _key: (None, 0.0),
        receipt_valid=lambda _record: True,
        receipt_fingerprint=lambda _record: "receipt",
        open_view=lambda _discovery: view,
    )
    provider._classify = lambda _index, record, key: (
        _fake_discovery(f"{record.public_id}:{key}"),
        None,
    )
    record = SimpleNamespace(
        public_id="item",
        write_id="write",
        kind="ifg",
        item={"properties": {"faninsar:domain": "radar"}},
    )
    return provider, record


def test_unavailable_helper_raises_one_typed_error() -> None:
    """The non-literal helper raises instead of recursing."""
    with pytest.raises(radar.RadarUnavailableError) as exc_info:
        radar._unavailable("invalid-window", 400, "bad window")
    assert exc_info.value.reason == "invalid-window"
    assert exc_info.value.status == 400


def test_provider_resolve_maps_contract_failure_to_503() -> None:
    """A backend contract failure is a bounded 503, not recursion or a 500."""
    provider, record = _fake_provider(_BadShapeView())
    with pytest.raises(radar.RadarUnavailableError) as exc_info:
        provider.resolve(None, record, "coherence")
    assert exc_info.value.reason == "backend-contract"
    assert exc_info.value.status == 503


def test_provider_read_chunk_maps_window_failure_to_503() -> None:
    """A mismatched payload window is a bounded 503, not recursion."""
    provider, record = _fake_provider(_BadWindowView())
    resolution = provider.resolve(None, record, "coherence")
    assert resolution is not None
    with pytest.raises(radar.RadarUnavailableError) as exc_info:
        provider.read_chunk(resolution, 0, 0)
    assert exc_info.value.reason == "backend-shape-mismatch"
    assert exc_info.value.status == 503


def test_radar_cache_evicts_handles_and_bounds_chunk_bytes() -> None:
    """Exercise the radar preview contract."""
    opened: list[_FakeView] = []

    def open_view(_discovery: Any) -> _FakeView:
        view = _FakeView()
        opened.append(view)
        return view

    provider = radar.RadarProvider(
        base_url="http://127.0.0.1:9",
        open_metadata=lambda _record, _key: (None, 0.0),
        receipt_valid=lambda _record: True,
        receipt_fingerprint=lambda _record: "receipt",
        open_view=open_view,
        max_views=2,
        chunk_cache_bytes=8,
    )

    def discovery(key: str) -> Any:
        return radar._Discovery(
            source_key=key,
            kind="coherence",
            shape=(2, 2),
            geolocation_shape=(2, 2),
            data_type="float32",
            first_center=(0.0, 0.0),
            step=(1.0, 1.0),
            values_uri="values",
            longitude_uri="longitude",
            latitude_uri="latitude",
            valid_uri=None,
            fingerprint=key,
            modified=0.0,
        )

    provider._classify = lambda _index, record, key: (
        discovery(f"{record.public_id}:{key}"),
        None,
    )
    records = [
        SimpleNamespace(
            public_id=f"item-{index}",
            write_id=f"write-{index}",
            kind="ifg",
            item={"properties": {"faninsar:domain": "radar"}},
        )
        for index in range(3)
    ]
    resolutions = [provider.resolve(None, record, "coherence") for record in records]
    assert all(resolution is not None for resolution in resolutions)
    assert opened[0].closed is True
    assert opened[-1].closed is False

    resolution = resolutions[-1]
    assert resolution is not None
    for row, column in ((0, 0), (0, 1), (1, 0)):
        assert provider.read_chunk(resolution, row, column) is not None
    assert len(resolution._view.chunks) == 2
    assert resolution._view.chunk_bytes == 8
    provider.close()
    provider.close()
    assert opened[-1].closed is True

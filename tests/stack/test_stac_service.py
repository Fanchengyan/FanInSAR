"""Read-only STAC service tests (PROPOSAL-0089 slice C).

Pure helper tests cover request parsing, and the localhost tests exercise the
real service over real sockets against the integrated ``faninsar.io.stac``
index. A missing serving dependency is an integration failure, not a skip.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import threading
import time
import urllib.parse
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest

from faninsar.stack import _stac_service as stac_service


@pytest.fixture(autouse=True)
def _clean_service_registry() -> Any:
    """Stop and forget services still registered at the end of each test."""
    yield
    registry = getattr(stac_service, "_SERVICE_REGISTRY", None)
    if registry is None:
        return
    lock = stac_service._SERVICE_REGISTRY_LOCK
    with lock:
        registrations = list(registry.values())
        registry.clear()
    for registration in registrations:
        with suppress(Exception):
            registration.service.stop()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_safe_asset_parts_accept_relative_chunk_paths() -> None:
    assert stac_service._safe_asset_parts("") == ()
    assert stac_service._safe_asset_parts("zarr.json") == ("zarr.json",)
    assert stac_service._safe_asset_parts("c/0/0") == ("c", "0", "0")
    assert stac_service._safe_asset_parts("%30/%31") == ("0", "1")


@pytest.mark.parametrize(
    "value",
    [
        "../x",
        "a/../../x",
        "a/./b",
        "a//b",
        "/etc/passwd",
        "\\windows\\x",
        "a\x00b",
        ".",
        "..",
        "%2e%2e/x",
        "..%2f..%2fetc",
        "%2fetc",
    ],
)
def test_safe_asset_parts_reject_traversal(value: str) -> None:
    with pytest.raises(stac_service._UnsafeAssetPathError):
        stac_service._safe_asset_parts(value)


def test_parse_byte_range_closed_open_suffix_and_clamping() -> None:
    assert stac_service._parse_byte_range("bytes=0-3", 10) == (0, 3)
    assert stac_service._parse_byte_range("bytes=5-", 10) == (5, 9)
    assert stac_service._parse_byte_range("bytes=-4", 10) == (6, 9)
    assert stac_service._parse_byte_range("bytes=8-20", 10) == (8, 9)
    assert stac_service._parse_byte_range("bytes=-20", 10) == (0, 9)


@pytest.mark.parametrize(
    "value",
    [
        "bytes=10-",
        "bytes=11-12",
        "bytes=5-4",
        "bytes=-0",
        "bytes=0-1,4-5",
        "chars=0-1",
    ],
)
def test_parse_byte_range_unsatisfiable(value: str) -> None:
    with pytest.raises(stac_service._UnsatisfiableRangeError):
        stac_service._parse_byte_range(value, 10)


@pytest.mark.parametrize(
    "value", ["bytes=abc", "bytes=", "bytes=-", "bytes=1-2-3", "bytes=--2"]
)
def test_parse_byte_range_malformed(value: str) -> None:
    with pytest.raises(stac_service._InvalidRangeError):
        stac_service._parse_byte_range(value, 10)


def test_asset_etag_is_strong_and_write_scoped() -> None:
    etag = stac_service._asset_etag("W1", "data", "c/0/0")
    assert etag.startswith('"') and etag.endswith('"')
    assert etag == stac_service._asset_etag("W1", "data", "c/0/0")
    assert etag != stac_service._asset_etag("W2", "data", "c/0/0")
    assert etag != stac_service._asset_etag("W1", "other", "c/0/0")
    assert etag != stac_service._asset_etag("W1", "data", "c/0/1")


def test_if_none_match_uses_weak_comparison() -> None:
    assert stac_service._if_none_match_matches('"abc"', '"abc"')
    assert stac_service._if_none_match_matches('W/"abc"', '"abc"')
    assert stac_service._if_none_match_matches("*", '"abc"')
    assert not stac_service._if_none_match_matches('"other"', '"abc"')


def test_if_range_requires_strong_etag_or_fresh_date() -> None:
    assert stac_service._if_range_matches('"abc"', '"abc"', 0.0)
    assert not stac_service._if_range_matches('W/"abc"', '"abc"', 0.0)
    assert stac_service._if_range_matches("Thu, 01 Jan 1970 00:00:10 GMT", '"abc"', 0.0)
    assert not stac_service._if_range_matches(
        "Thu, 01 Jan 1970 00:00:00 GMT", '"abc"', 30.0
    )
    assert not stac_service._if_range_matches("garbage", '"abc"', 0.0)


def test_base_url_brackets_ipv6_and_loopback_detection() -> None:
    assert stac_service._base_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"
    assert stac_service._base_url("::1", 8001) == "http://[::1]:8001"
    assert stac_service._is_loopback("localhost")
    assert stac_service._is_loopback("127.0.0.1")
    assert stac_service._is_loopback("::1")
    assert not stac_service._is_loopback("0.0.0.0")


# ---------------------------------------------------------------------------
# Localhost integration helpers
# ---------------------------------------------------------------------------


def _publish_scene(
    work_dir: Path,
    name: str,
    payload: bytes,
    *,
    date: str = "2024-01-02",
    overwrite: bool = False,
) -> Any:
    """Commit one tiny radar scene product with a single file asset."""
    from faninsar.io.storage.product import publish_product

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        (staging / "data.bin").write_bytes(payload)
        return {
            "data": {
                "href": "data.bin",
                "type": "application/octet-stream",
            }
        }

    return publish_product(
        work_dir / "scenes" / name,
        kind="scene",
        properties={"faninsar:date": date, "faninsar:domain": "radar"},
        write_payloads=write_payloads,
        overwrite=overwrite,
    )


def _publish_zarr(
    work_dir: Path,
    name: str,
    *,
    values: Any = None,
    overwrite: bool = False,
) -> Any:
    """Commit one tiny real Zarr v3 array asset with a known 2x2 value grid."""
    import numpy as np
    import zarr
    from faninsar.io.storage.product import publish_product

    grid = (
        np.arange(4, dtype="i4").reshape(2, 2)
        if values is None
        else np.asarray(values, dtype="i4")
    )

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        array = zarr.open_array(
            store=str(staging / "arr"),
            mode="w",
            shape=grid.shape,
            dtype=grid.dtype,
        )
        array[:] = grid
        return {"data": {"href": "arr", "type": "application/octet-stream"}}

    return publish_product(
        work_dir / "scenes" / name,
        kind="scene",
        properties={"faninsar:date": "2024-01-02", "faninsar:domain": "radar"},
        write_payloads=write_payloads,
        overwrite=overwrite,
    )


def _publish_zarr_v2(work_dir: Path, name: str) -> Any:
    """Commit one tiny real Zarr v2 array asset with ``.zarray`` metadata."""
    import zarr
    from faninsar.io.storage.product import publish_product

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        array = zarr.open_array(
            store=str(staging / "arr"),
            mode="w",
            shape=(2, 2),
            dtype="i4",
            zarr_format=2,
        )
        array[:] = [[1, 2], [3, 4]]
        return {"data": {"href": "arr", "type": "application/octet-stream"}}

    return publish_product(
        work_dir / "scenes" / name,
        kind="scene",
        properties={"faninsar:date": "2024-01-02", "faninsar:domain": "radar"},
        write_payloads=write_payloads,
    )


def _publish_zarr_attrs_only(work_dir: Path, name: str) -> Any:
    """Commit one directory asset with only ``.zattrs`` child metadata."""
    from faninsar.io.storage.product import publish_product

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        root = staging / "arr"
        root.mkdir(parents=True)
        (root / ".zattrs").write_bytes(b'{"unit": "metre"}\n')
        (root / "0.0").write_bytes(b"\x00\x01\x02\x03")
        return {"data": {"href": "arr", "type": "application/octet-stream"}}

    return publish_product(
        work_dir / "scenes" / name,
        kind="scene",
        properties={"faninsar:date": "2024-01-02", "faninsar:domain": "radar"},
        write_payloads=write_payloads,
    )


def _publish_plain_directory(work_dir: Path, name: str) -> Any:
    """Commit one directory asset that contains no Zarr metadata."""
    from faninsar.io.storage.product import publish_product

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        root = staging / "arr"
        root.mkdir(parents=True)
        (root / "plain.txt").write_bytes(b"not-a-zarr-store")
        return {"data": {"href": "arr", "type": "application/octet-stream"}}

    return publish_product(
        work_dir / "scenes" / name,
        kind="scene",
        properties={"faninsar:date": "2024-01-02", "faninsar:domain": "radar"},
        write_payloads=write_payloads,
    )


@contextmanager
def _running_service(work_dir: Path, *, persist_catalog: bool = False):
    service = stac_service.start_stac_service(
        work_dir, port=0, persist_catalog=persist_catalog
    )
    try:
        yield service
    finally:
        service.stop()


def _parts_of(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(base_url)
    assert parsed.scheme == "http"
    assert parsed.hostname is not None
    assert parsed.port is not None
    return parsed.hostname, parsed.port


def _registration_keys() -> set[tuple[str, int]]:
    """Return the currently published endpoint keys under the registry lock."""
    with stac_service._SERVICE_REGISTRY_LOCK:
        return set(stac_service._SERVICE_REGISTRY)


def _free_port() -> int:
    """Reserve and release one loopback port for a deterministic bind race."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _resolved_path(request_path: str, location: str) -> str:
    """Resolve a relative redirect Location against the request URL path."""
    joined = urllib.parse.urljoin(f"http://unit.test{request_path}", location)
    return urllib.parse.urlsplit(joined).path


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = _parts_of(base_url)
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, response_headers, payload
    finally:
        connection.close()


def _json(base_url: str, path: str) -> tuple[int, dict[str, str], dict]:
    status, headers, payload = _request(base_url, path)
    assert status == 200, (status, payload)
    return status, headers, json.loads(payload)


def _search_ids(base_url: str) -> list[str]:
    _, _, document = _json(base_url, "/search")
    assert document["type"] == "FeatureCollection"
    return [feature["id"] for feature in document["features"]]


def _tree_snapshot(work_dir: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(work_dir.rglob("*")):
        relative = path.relative_to(work_dir).as_posix()
        if path.is_symlink():
            snapshot[relative] = b"<symlink>"
        elif path.is_file():
            snapshot[relative] = path.read_bytes()
        elif path.is_dir():
            snapshot[relative] = b"<dir>"
    return snapshot


# ---------------------------------------------------------------------------
# Actual localhost HTTP
# ---------------------------------------------------------------------------


def test_printed_urls_serve_metadata_on_the_assigned_port(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "stack"
    product = _publish_scene(work_dir, "a", b"payload-a")

    with _running_service(work_dir) as service:
        printed = capsys.readouterr().out
        assert service.catalog_url in printed
        assert service.api_url in printed
        host, port = _parts_of(service.base_url)
        assert host == "127.0.0.1"
        assert port != 0

        _, headers, landing = _json(service.base_url, "/")
        assert headers["content-type"].startswith("application/json")
        assert landing["type"] == "Catalog"
        assert landing["stac_version"] == "1.0.0"
        assert landing["id"].startswith("stack-")
        assert landing["description"]
        assert landing["conformsTo"] == [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/item-search",
            "https://api.stacspec.org/v1.0.0/collections",
        ]
        relations = {
            (link["rel"], link.get("method", "GET")) for link in landing["links"]
        }
        assert ("self", "GET") in relations
        assert ("root", "GET") in relations
        assert ("conformance", "GET") in relations
        assert ("data", "GET") in relations
        assert ("service-desc", "GET") in relations
        assert ("search", "GET") in relations
        assert ("search", "POST") in relations

        _, _, root_catalog = _json(service.base_url, "/catalog.json")
        assert root_catalog["type"] == "Catalog"
        assert root_catalog["id"] == landing["id"]

        _, _, conformance = _json(service.base_url, "/conformance")
        assert conformance["conformsTo"] == landing["conformsTo"]

        status, api_headers, payload = _request(service.base_url, "/api")
        assert status == 200
        assert api_headers["content-type"].startswith(
            "application/vnd.oai.openapi+json;version=3.0"
        )
        api = json.loads(payload)
        assert api["openapi"].startswith("3.0")
        assert "/search" in api["paths"]
        assert "/assets/{publicId}/{assetKey}/{relative}" in api["paths"]

        _, _, collections = _json(service.base_url, "/collections")
        assert collections["collections"] == []

        ids = _search_ids(service.base_url)
        assert len(ids) == 1
        assert ids[0].endswith(product.write_id)

        status, _, _ = _request(
            service.base_url,
            "/search",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"limit": 10}).encode(),
        )
        assert status == 200

        status, _, _ = _request(
            service.base_url,
            "/search",
            method="POST",
            headers={"Content-Type": "text/plain"},
            body=b"{}",
        )
        assert status == 415

        status, _, _ = _request(service.base_url, "/search?unknown=1")
        assert status == 400
        status, _, _ = _request(
            service.base_url,
            "/search?bbox=0,0,1,1&intersects=%7B%22type%22%3A%22Point%22%2C"
            "%22coordinates%22%3A%5B0%2C0%5D%7D",
        )
        assert status == 400

        status, _, item = _json(service.base_url, f"/items/{ids[0]}")
        assert status == 200
        assert item["id"] == ids[0]
        assert item["type"] == "Feature"
        assert item["geometry"] is None
        assert "bbox" not in item

        status, headers, payload = _request(
            service.base_url, f"/items/{ids[0]}", method="HEAD"
        )
        assert status == 200
        assert payload == b""
        assert headers["content-type"].startswith("application/geo+json")

        status, headers, payload = _request(service.base_url, "/", method="OPTIONS")
        assert status == 204
        assert payload == b""
        assert headers["access-control-allow-origin"] == "*"
        assert "range" in headers["access-control-allow-headers"].lower()

        status, _, _ = _request(service.base_url, "/items/does-not-exist")
        assert status == 404
        status, _, _ = _request(service.base_url, "/collections/does-not-exist")
        assert status == 404


def test_asset_get_head_range_conditionals_and_traversal(tmp_path: Path) -> None:
    work_dir = tmp_path / "stack"
    payload = bytes(range(256)) * 4
    _publish_scene(work_dir, "a", payload)

    with _running_service(work_dir) as service:
        (public_id,) = _search_ids(service.base_url)
        asset_url = f"/assets/{public_id}/data"
        asset_dir_url = f"/assets/{public_id}/data/"

        status, headers, body = _request(service.base_url, asset_url)
        assert status == 200
        assert body == payload
        assert headers["content-length"] == str(len(payload))
        assert headers["content-type"] == "application/octet-stream"
        assert headers["accept-ranges"] == "bytes"
        assert headers["etag"].startswith('"')
        assert headers["cache-control"] == "no-cache"
        etag = headers["etag"]

        status, headers, body = _request(service.base_url, asset_dir_url)
        assert status == 200
        assert body == payload

        status, head_headers, head_body = _request(
            service.base_url, asset_url, method="HEAD"
        )
        assert status == 200
        assert head_body == b""
        assert head_headers["content-length"] == str(len(payload))
        assert head_headers["etag"] == etag

        status, headers, body = _request(
            service.base_url, asset_url, headers={"Range": "bytes=10-19"}
        )
        assert status == 206
        assert body == payload[10:20]
        assert headers["content-range"] == f"bytes 10-19/{len(payload)}"
        assert headers["content-length"] == "10"

        status, headers, body = _request(
            service.base_url, asset_url, headers={"Range": "bytes=1000-"}
        )
        assert status == 206
        assert body == payload[1000:]
        assert (
            headers["content-range"] == f"bytes 1000-{len(payload) - 1}/{len(payload)}"
        )

        status, headers, body = _request(
            service.base_url, asset_url, headers={"Range": "bytes=-10"}
        )
        assert status == 206
        assert body == payload[-10:]
        assert (
            headers["content-range"]
            == f"bytes {len(payload) - 10}-{len(payload) - 1}/{len(payload)}"
        )

        status, headers, body = _request(
            service.base_url, asset_url, headers={"Range": "bytes=99999-"}
        )
        assert status == 416
        assert body == b""
        assert headers["content-range"] == f"bytes */{len(payload)}"

        status, _, _ = _request(
            service.base_url, asset_url, headers={"Range": "bytes=0-1,4-5"}
        )
        assert status == 416
        status, _, _ = _request(
            service.base_url, asset_url, headers={"Range": "bytes=abc"}
        )
        assert status == 400

        status, headers, body = _request(
            service.base_url, asset_url, headers={"If-None-Match": etag}
        )
        assert status == 304
        assert body == b""
        assert headers["etag"] == etag

        status, headers, body = _request(
            service.base_url,
            asset_url,
            headers={"Range": "bytes=0-3", "If-Range": '"stale"'},
        )
        assert status == 200
        assert body == payload

        status, _, _ = _request(
            service.base_url,
            f"/assets/{public_id}/data/%2e%2e/%2e%2e/item.json",
        )
        assert status == 404


def test_zarr_directory_root_redirects_to_verified_metadata_and_serves_bytes(
    tmp_path: Path,
) -> None:
    """Legacy and slash roots 307 to the real metadata, while children still serve."""
    work_dir = tmp_path / "stack"
    product = _publish_zarr(work_dir, "a")
    local_metadata = (product.path / "arr" / "zarr.json").read_bytes()

    with _running_service(work_dir) as service:
        (public_id,) = _search_ids(service.base_url)
        legacy_root = f"/assets/{public_id}/data"
        slash_root = f"{legacy_root}/"

        locations: dict[str, str] = {}
        for root in (legacy_root, slash_root):
            status, headers, body = _request(service.base_url, root)
            assert status == 307
            assert body == b""
            location = headers["location"]
            assert "\r" not in location
            assert "\n" not in location
            assert _resolved_path(root, location) == f"{legacy_root}/zarr.json"
            locations[root] = location

        status, headers, body = _request(service.base_url, legacy_root, method="HEAD")
        assert status == 307
        assert body == b""
        assert headers["location"] == locations[legacy_root]

        status, headers, body = _request(
            service.base_url,
            f"{legacy_root}/zarr.json",
        )
        assert status == 200
        assert body == local_metadata
        assert headers["content-type"] == "application/json"

        status, headers, body = _request(
            service.base_url,
            f"{legacy_root}/c/0/0",
            headers={"Range": "bytes=0-3"},
        )
        assert status == 206
        assert len(body) == 4
        assert headers["content-range"].startswith("bytes 0-3/")

        import numpy as np
        import zarr
        from zarr.storage import FsspecStore

        store = FsspecStore.from_url(f"{service.base_url}{slash_root}", read_only=True)
        array = zarr.open_array(store=store, mode="r")
        assert array.shape == (2, 2)
        assert np.array_equal(array[:], np.arange(4, dtype="i4").reshape(2, 2))


def test_zarr_v2_root_redirects_to_verified_zarray_metadata(tmp_path: Path) -> None:
    """A v2 store root redirects to ``.zarray`` and keeps its chunk ranged."""
    work_dir = tmp_path / "stack"
    product = _publish_zarr_v2(work_dir, "a")
    local_metadata = (product.path / "arr" / ".zarray").read_bytes()

    with _running_service(work_dir) as service:
        (public_id,) = _search_ids(service.base_url)
        root = f"/assets/{public_id}/data"
        status, headers, body = _request(service.base_url, root)
        assert status == 307
        assert body == b""
        assert _resolved_path(root, headers["location"]) == f"{root}/.zarray"

        status, headers, body = _request(service.base_url, f"{root}/.zarray")
        assert status == 200
        assert body == local_metadata
        assert headers["content-type"] == "application/json"

        status, headers, body = _request(
            service.base_url,
            f"{root}/0.0",
            headers={"Range": "bytes=4-7"},
        )
        assert status == 206
        assert len(body) == 4
        assert headers["content-range"].startswith("bytes 4-7/")


def test_zattrs_only_directory_root_is_404_but_child_is_served(
    tmp_path: Path,
) -> None:
    """``.zattrs`` alone is child JSON, not a store root, and must not redirect."""
    work_dir = tmp_path / "stack"
    product = _publish_zarr_attrs_only(work_dir, "a")
    local_attrs = (product.path / "arr" / ".zattrs").read_bytes()

    with _running_service(work_dir) as service:
        (public_id,) = _search_ids(service.base_url)
        root = f"/assets/{public_id}/data"

        for path in (root, f"{root}/"):
            status, headers, body = _request(service.base_url, path)
            assert status == 404
            assert body
            assert "location" not in headers

        status, headers, body = _request(service.base_url, f"{root}/.zattrs")
        assert status == 200
        assert body == local_attrs
        assert headers["content-type"] == "application/json"

        status, _, body = _request(service.base_url, f"{root}/0.0")
        assert status == 200
        assert body == b"\x00\x01\x02\x03"


def test_plain_directory_asset_root_stays_404_without_zarr_metadata(
    tmp_path: Path,
) -> None:
    """An arbitrary directory root is never listed or redirected."""
    work_dir = tmp_path / "stack"
    _publish_plain_directory(work_dir, "a")

    with _running_service(work_dir) as service:
        (public_id,) = _search_ids(service.base_url)
        root = f"/assets/{public_id}/data"
        for path in (root, f"{root}/"):
            status, _, body = _request(service.base_url, path)
            assert status == 404
            assert body

        status, _, body = _request(service.base_url, f"{root}/plain.txt")
        assert status == 200
        assert body == b"not-a-zarr-store"


def test_overwritten_zarr_root_redirect_is_404_before_refresh(tmp_path: Path) -> None:
    """A stale root URL cannot redirect to replacement metadata before refresh."""
    work_dir = tmp_path / "stack"
    _publish_zarr(work_dir, "a")

    with _running_service(work_dir) as service:
        (old_id,) = _search_ids(service.base_url)
        old_root = f"/assets/{old_id}/data"
        status, _, _ = _request(service.base_url, old_root)
        assert status == 307

        _publish_zarr(work_dir, "a", values=[[9, 8], [7, 6]], overwrite=True)

        for path in (old_root, f"{old_root}/", f"{old_root}/zarr.json"):
            status, headers, _ = _request(service.base_url, path)
            assert status == 404
            assert "location" not in headers

        service.refresh()
        (new_id,) = _search_ids(service.base_url)
        assert new_id != old_id
        status, headers, _ = _request(service.base_url, f"/assets/{new_id}/data")
        assert status == 307
        status, _, body = _request(
            service.base_url,
            _resolved_path(f"/assets/{new_id}/data", headers["location"]),
        )
        assert status == 200
        assert b'"shape"' in body


def test_overwrite_before_refresh_old_asset_is_404_then_refresh_serves_new(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"generation-one")

    with _running_service(work_dir) as service:
        (old_id,) = _search_ids(service.base_url)
        old_asset = f"/assets/{old_id}/data"
        status, _, body = _request(service.base_url, old_asset)
        assert status == 200
        assert body == b"generation-one"

        _publish_scene(work_dir, "a", b"generation-two", overwrite=True)

        status, _, body = _request(service.base_url, old_asset)
        assert status == 404
        assert b"generation-two" not in body

        report = service.refresh()
        assert report.generation
        assert report.item_count == 1

        (new_id,) = _search_ids(service.base_url)
        assert new_id != old_id
        status, _, _ = _request(service.base_url, old_asset)
        assert status == 404
        status, _, body = _request(service.base_url, f"/assets/{new_id}/data")
        assert status == 200
        assert body == b"generation-two"

        document = service.report()
        json.dumps(document, allow_nan=False)
        assert document["persistence"]["persisted"] is False


def test_occupied_port_fails_without_workspace_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    before = _tree_snapshot(work_dir)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        with pytest.raises(OSError):
            stac_service.start_stac_service(
                work_dir, host="127.0.0.1", port=port, persist_catalog=True
            )
    finally:
        listener.close()

    assert capsys.readouterr().out == ""
    assert _tree_snapshot(work_dir) == before
    assert not (work_dir / "catalog.json").exists()
    assert not (work_dir / "stac").exists()
    assert not list(work_dir.glob(".stac-staging-*"))


def test_persist_catalog_writes_only_after_bind_and_stop_is_idempotent(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    assert not (work_dir / "catalog.json").exists()

    service = stac_service.start_stac_service(work_dir, port=0, persist_catalog=True)
    try:
        assert (work_dir / "catalog.json").is_file()
        assert (work_dir / "stac" / ".faninsar-stac.json").is_file()
        root = json.loads((work_dir / "catalog.json").read_bytes())
        assert root["type"] == "Catalog"
        assert root["id"].startswith("stack-")

        status, _, _ = _request(service.base_url, "/catalog.json")
        assert status == 200
    finally:
        service.stop()
        service.stop()

    service.wait(timeout=5)
    with pytest.raises(OSError):
        connection = socket.create_connection(_parts_of(service.base_url), timeout=1)
        connection.close()
    report = service.report()
    assert report["persistence"]["persisted"] is True
    assert report["service"]["lifecycle"] == "stopped"
    assert report["service"]["persist_catalog"] is True


# ---------------------------------------------------------------------------
# Process-local endpoint reuse
# ---------------------------------------------------------------------------


def test_repeated_start_same_workspace_and_actual_port_returns_same_handle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A nonzero-port request for a live endpoint returns the exact same handle."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    first = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)
    try:
        first_output = capsys.readouterr().out
        assert first.catalog_url in first_output

        second = stac_service.start_stac_service(
            work_dir / ".", port=first.port, persist_catalog=False
        )

        assert second is first
        assert second.base_url == first.base_url
        assert second.port == first.port
        assert second.host == first.host
        assert capsys.readouterr().out == ""
        assert len(_search_ids(second.base_url)) == 1
        report = second.report()
        assert report["item_count"] == 1
        assert "exclusions" in report
        assert report["service"] == {
            "work_dir": str(work_dir.resolve()),
            "host": "127.0.0.1",
            "port": first.port,
            "persist_catalog": False,
            "lifecycle": "running",
        }
    finally:
        first.stop()


def test_reuse_does_not_rebuild_index_or_load_serving_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse is a handle return, not an implicit refresh or reindex."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    first = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)

    def fail() -> Any:
        message = "reuse must not rebuild the index or serving runtime"
        raise AssertionError(message)

    try:
        monkeypatch.setattr(stac_service, "_load_web", fail)
        monkeypatch.setattr(stac_service, "_load_io", fail)
        second = stac_service.start_stac_service(work_dir, port=first.port)
        assert second is first
    finally:
        first.stop()


def test_reuse_preserves_the_original_persist_mode_without_hidden_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """A different persist flag reuses the running mode and says so explicitly."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    first = stac_service.start_stac_service(work_dir, port=0, persist_catalog=True)
    try:
        assert (work_dir / "catalog.json").is_file()
        capsys.readouterr()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=stac_service.logger.name):
            second = stac_service.start_stac_service(
                work_dir, port=first.port, persist_catalog=False
            )
        assert second is first
        assert second.report()["persistence"]["requested"] is True
        assert first._persist_catalog is True
        assert capsys.readouterr().out == ""
        messages = [record.getMessage() for record in caplog.records]
        assert any("persist_catalog" in message for message in messages)
        assert any("True" in message for message in messages)
    finally:
        first.stop()


def test_port_zero_always_starts_new_listener_and_actual_port_can_be_reacquired(
    tmp_path: Path,
) -> None:
    """Ephemeral requests never reuse, while the assigned port is addressable."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    first = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)
    second = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)
    try:
        assert first is not second
        assert first.port != second.port
        assert stac_service.start_stac_service(work_dir, port=first.port) is first
        assert stac_service.start_stac_service(work_dir, port=second.port) is second
    finally:
        first.stop()
        second.stop()


def test_different_workspace_on_a_busy_endpoint_raises_and_preserves_active(
    tmp_path: Path,
) -> None:
    """A busy registered endpoint never gets attached to a different workspace."""
    active_dir = tmp_path / "active"
    other_dir = tmp_path / "other"
    _publish_scene(active_dir, "a", b"payload")
    _publish_scene(other_dir, "b", b"payload")
    active = stac_service.start_stac_service(active_dir, port=0, persist_catalog=False)
    try:
        with pytest.raises(OSError, match="another workspace") as excinfo:
            stac_service.start_stac_service(
                other_dir, port=active.port, persist_catalog=False
            )
        message = str(excinfo.value)
        assert str(active_dir.resolve()) in message
        assert "another" in message.lower()
        assert _registration_keys() == {
            stac_service._service_key(active.host, active.port)
        }
        status, _, _ = _request(active.base_url, "/")
        assert status == 200
        assert stac_service.start_stac_service(active_dir, port=active.port) is active
    finally:
        active.stop()


def test_concurrent_starts_on_one_port_publish_exactly_one_handle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two racing callers serialize on the registry and share one service."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    port = _free_port()
    barrier = threading.Barrier(2)
    results: list[Any] = []
    errors: list[BaseException] = []

    def start() -> None:
        barrier.wait()
        try:
            results.append(
                stac_service.start_stac_service(
                    work_dir, port=port, persist_catalog=False
                )
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert errors == []
    assert len(results) == 2
    assert results[0] is results[1]
    service = results[0]
    try:
        assert service.port == port
        assert _registration_keys() == {stac_service._service_key("127.0.0.1", port)}
        assert capsys.readouterr().out.count("FanInSAR STAC Catalog:") == 1
    finally:
        service.stop()


def test_stop_removes_only_its_registration_and_restart_returns_new_handle(
    tmp_path: Path,
) -> None:
    """Stopping is idempotent, clears the entry, and frees the endpoint."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    first = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)
    port = first.port
    key = stac_service._service_key(first.host, port)
    first.stop()
    first.stop()
    assert key not in _registration_keys()

    second = stac_service.start_stac_service(work_dir, port=port, persist_catalog=False)
    try:
        assert second is not first
        assert second.port == port
        assert key in _registration_keys()
    finally:
        second.stop()


def test_dead_service_is_purged_and_failed_start_publishes_no_entry(
    tmp_path: Path,
) -> None:
    """A dead runtime is never returned and a bind failure leaves no registration."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    first = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)
    port = first.port
    key = stac_service._service_key(first.host, port)

    # Simulate an unexpectedly stopped runtime whose strong registration remains.
    first._runtime.stop()
    assert key in _registration_keys()

    replacement = stac_service.start_stac_service(
        work_dir, port=port, persist_catalog=False
    )
    try:
        assert replacement is not first
        assert replacement.port == port
        assert key in _registration_keys()
    finally:
        replacement.stop()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    blocked_port = int(listener.getsockname()[1])
    try:
        with pytest.raises(OSError, match="in use"):
            stac_service.start_stac_service(
                work_dir, port=blocked_port, persist_catalog=False
            )
    finally:
        listener.close()

    assert all(
        existing_port != blocked_port for _, existing_port in _registration_keys()
    )


def test_start_build_does_not_hold_the_registry_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second endpoint can start while a peer is still building its index."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _publish_scene(first_dir, "a", b"payload-a")
    _publish_scene(second_dir, "b", b"payload-b")
    original_load_io = stac_service._load_io
    entered_build = threading.Event()
    release_build = threading.Event()
    calls = {"count": 0}

    def slow_load_io() -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            entered_build.set()
            if not release_build.wait(15):
                message = "slow build gate timed out"
                raise RuntimeError(message)
        return original_load_io()

    monkeypatch.setattr(stac_service, "_load_io", slow_load_io)
    results: list[Any] = []
    errors: list[BaseException] = []

    def start_first() -> None:
        try:
            results.append(
                stac_service.start_stac_service(
                    first_dir, port=0, persist_catalog=False
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=start_first)
    thread.start()
    second: Any = None
    try:
        assert entered_build.wait(10)
        second = stac_service.start_stac_service(
            second_dir, port=0, persist_catalog=False
        )
        assert second._runtime.is_alive()
    finally:
        release_build.set()
        thread.join(30)
        if second is not None:
            second.stop()
        for service in results:
            service.stop()

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert not results[0]._runtime.is_alive()


def test_start_after_stop_begins_waits_and_returns_a_new_live_handle(
    tmp_path: Path,
) -> None:
    """A stop in progress is waited out, never returned as a live handle."""
    work_dir = tmp_path / "stack"
    _publish_scene(work_dir, "a", b"payload")
    first = stac_service.start_stac_service(work_dir, port=0, persist_catalog=False)
    port = first.port
    stop_thread = threading.Thread(target=first.stop)
    stop_thread.start()
    deadline = time.monotonic() + 10
    while not first._stopping.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert first._stopping.is_set()

    second = stac_service.start_stac_service(work_dir, port=port, persist_catalog=False)
    stop_thread.join(15)
    try:
        assert not stop_thread.is_alive()
        assert second is not first
        assert second.port == port
        assert second._runtime.is_alive()
        assert not second._stopping.is_set()
    finally:
        second.stop()

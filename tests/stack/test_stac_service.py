"""Read-only STAC service tests (PROPOSAL-0089 slice C).

Pure helper tests cover request parsing, and the localhost tests exercise the
real service over real sockets against the integrated ``faninsar.io.stac``
index. A missing serving dependency is an integration failure, not a skip.
"""

from __future__ import annotations

import http.client
import json
import socket
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from faninsar.stack import _stac_service as stac_service


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


def _publish_zarr(work_dir: Path, name: str, *, payload: bytes, chunk: bytes) -> Any:
    """Commit one tiny directory asset that contains Zarr metadata and a chunk."""
    from faninsar.io.storage.product import publish_product

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        root = staging / "arr"
        (root / "c" / "0").mkdir(parents=True)
        (root / "zarr.json").write_bytes(payload)
        (root / "c" / "0" / "0").write_bytes(chunk)
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


def test_zarr_directory_asset_serves_metadata_and_one_chunk(tmp_path: Path) -> None:
    work_dir = tmp_path / "stack"
    metadata = b'{"zarr_format": 3}\n'
    chunk = b"chunk-bytes"
    _publish_zarr(work_dir, "a", payload=metadata, chunk=chunk)

    with _running_service(work_dir) as service:
        (public_id,) = _search_ids(service.base_url)
        status, headers, body = _request(
            service.base_url, f"/assets/{public_id}/data/zarr.json"
        )
        assert status == 200
        assert body == metadata
        assert headers["content-type"] == "application/json"

        status, headers, body = _request(
            service.base_url, f"/assets/{public_id}/data/c/0/0"
        )
        assert status == 200
        assert body == chunk

        status, _, _ = _request(service.base_url, f"/assets/{public_id}/data")
        assert status == 404


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

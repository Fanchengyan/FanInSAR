"""Contract tests for the minimal remote boundary."""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from faninsar import remote
from faninsar.data.query import Points, Polygons

if TYPE_CHECKING:
    from pathlib import Path


def _register(name: str = "contract") -> str:
    """Register a deterministic fixture catalog for one test."""
    payload = b"fixture bytes"
    remote._register_fixture(
        [
            {
                "id": "item-1",
                "collection": "dem",
                "geometry": Polygon(((-1, -1), (1, -1), (1, 1), (-1, 1))),
                "properties": {"token": "secret", "safe": "yes"},
                "assets": {
                    "data": {
                        "href": "https://fixture.invalid/assets/data",
                        "data": payload,
                        "size": len(payload),
                        "version": "immutable-1",
                        "checksum": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                    }
                },
            }
        ],
        name=name,
    )
    return name


def test_points_are_exact_and_metadata_is_safe() -> None:
    """Points do not become a bounding box and secrets are removed."""
    name = _register("points-contract")
    items = remote.search(Points([(0, 0), (10, 10)], crs=4326), catalog=name)

    assert len(items) == 1
    assert items[0].matched_point_indices == (0,)
    assert "token" not in items[0].raw_metadata
    with pytest.raises(TypeError):
        items[0].assets["data"].properties["x"] = "y"  # type: ignore[index]


def test_polygons_subtract_undesired_region() -> None:
    """A candidate wholly inside an undesired region is excluded."""
    name = _register("polygon-contract")
    gdf = Polygons(
        gpd.GeoDataFrame(
            geometry=[
                Polygon(((-2, -2), (2, -2), (2, 2), (-2, 2))),
                Polygon(((-1.1, -1.1), (1.1, -1.1), (1.1, 1.1), (-1.1, 1.1))),
            ],
            crs=4326,
        ),
        types=["desired", "undesired"],
        crs=4326,
    )
    assert remote.search(gdf, catalog=name) == []


def test_download_is_atomic_and_reuses_verified_destination(tmp_path: Path) -> None:
    """A qualified manifest permits reuse while a different asset conflicts."""
    name = _register("download-contract")
    asset = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]
    destination = tmp_path / "asset.bin"

    assert remote.download(asset, destination) == destination
    assert remote.download(asset, destination) == destination
    destination.write_bytes(b"changed")
    with pytest.raises(remote.RemoteIntegrityError, match="destination_conflict"):
        remote.download(asset, destination)


def test_unqualified_download_requires_explicit_overwrite(tmp_path: Path) -> None:
    """An unqualified asset is fresh but cannot silently replace a destination."""
    name = "download-unqualified"
    remote._register_fixture(
        [
            {
                "id": "item-1",
                "geometry": Polygon(((-1, -1), (1, -1), (1, 1), (-1, 1))),
                "assets": {
                    "data": {
                        "href": "https://fixture.invalid/assets/data",
                        "data": b"fresh bytes",
                    }
                },
            }
        ],
        name=name,
    )
    asset = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]
    destination = tmp_path / "asset.bin"

    destination.write_bytes(b"stale bytes")
    with pytest.raises(remote.RemoteIntegrityError, match="destination_conflict"):
        remote.download(asset, destination)
    assert destination.read_bytes() == b"stale bytes"
    assert (
        remote.download(asset, destination, overwrite=True).read_bytes()
        == b"fresh bytes"
    )
    destination.write_bytes(b"changed bytes")
    with pytest.raises(remote.RemoteIntegrityError, match="destination_conflict"):
        remote.download(asset, destination)


def test_invalid_spatial_and_datetime_inputs_are_query_errors() -> None:
    """Malformed CRS and datetime values use the public query error type."""
    name = _register("invalid-inputs")
    points = Points([(0, 0)], crs=4326)
    points._crs = "not-a-crs"  # type: ignore[assignment]
    with pytest.raises(remote.RemoteQueryError) as error:
        remote.search(points, catalog=name)
    assert error.value.reason == "invalid_crs"
    with pytest.raises(remote.RemoteQueryError) as error:
        remote.search(
            Points([(0, 0)], crs=4326),
            catalog=name,
            datetime_range=("not-a-date", "2024-01-01T00:00:00+00:00"),  # type: ignore[arg-type]
        )
    assert error.value.reason == "invalid_datetime"


class _RetryAdapter:
    """Small adapter fixture for transfer-budget regression tests."""

    provider = "retry-provider"
    origins = ("https://source.invalid",)
    path_prefixes = ("/assets",)
    redirect_origins = ("https://cdn.invalid",)
    profiles = ("anonymous",)

    def __init__(self, results: list[object]) -> None:
        self.results = iter(results)

    def items(self) -> list[object]:
        """Return no catalog records."""
        return []

    def fetch(self, asset: object, budget: object) -> object:
        """Return the next configured transfer result."""
        del asset, budget
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


def _retry_asset(catalog: str) -> remote.RemoteAsset:
    """Build an asset bound to a test adapter."""
    return remote.RemoteAsset(
        provider="retry-provider",
        catalog=catalog,
        collection=None,
        item_id="item-1",
        key="data",
        href="https://source.invalid/assets/data",
    )


def test_operation_bytes_accumulate_across_failed_retries(tmp_path: Path) -> None:
    """Bytes read before each retry count against one operation budget."""

    def failing_stream() -> object:
        """Yield bytes and then fail during streaming."""
        yield b"abc"
        message = "interrupted"
        raise OSError(message)

    name = "operation-budget"
    remote._register_adapter(
        name,
        _RetryAdapter(
            [
                failing_stream(),
                failing_stream(),
                [b"abc", b"ok"],
            ]
        ),
    )
    with pytest.raises(remote.RemoteLimitError, match="max_operation_bytes"):
        remote.download(
            _retry_asset(name),
            tmp_path / "asset.bin",
            budget=remote.RemoteResourceBudget(
                max_operation_bytes=5, max_retries=2, max_requests=3
            ),
        )


def test_elapsed_limit_applies_when_retry_attempt_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed attempt that crosses the deadline returns a typed limit error."""
    name = "elapsed-budget"
    remote._register_adapter(name, _RetryAdapter([OSError("unavailable")]))
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(remote.time, "monotonic", lambda: next(clock))

    with pytest.raises(remote.RemoteLimitError, match="max_elapsed_seconds"):
        remote.download(
            _retry_asset(name),
            tmp_path / "asset.bin",
            budget=remote.RemoteResourceBudget(
                max_elapsed_seconds=1, max_retries=1, max_requests=2
            ),
        )


def test_redirect_handler_validates_allowlist_and_counts_hops() -> None:
    """Each redirect is checked against registered origins and the hop budget."""
    adapter = _RetryAdapter([])
    handler = remote._RedirectHandler(
        adapter, remote.RemoteResourceBudget(max_redirects=4)
    )
    request = remote.urllib.request.Request("https://source.invalid/assets/data")
    response = io.BytesIO()
    response.headers = {"Location": "https://cdn.invalid/assets/data"}  # type: ignore[attr-defined]
    redirected = handler.redirect_request(
        request, response, 302, "Found", response.headers, response.headers["Location"]
    )
    assert redirected is not None
    assert redirected.full_url == "https://cdn.invalid/assets/data"

    response.headers["Location"] = "https://evil.invalid/assets/data"
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        handler.redirect_request(
            redirected,
            response,
            302,
            "Found",
            response.headers,
            response.headers["Location"],
        )

    response.headers["Location"] = "https://cdn.invalid/other/data"
    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        handler.redirect_request(
            redirected,
            response,
            302,
            "Found",
            response.headers,
            response.headers["Location"],
        )

    limit_handler = remote._RedirectHandler(
        adapter, remote.RemoteResourceBudget(max_redirects=1)
    )
    response.headers["Location"] = "https://cdn.invalid/assets/data"
    assert (
        limit_handler.redirect_request(
            request,
            response,
            302,
            "Found",
            response.headers,
            response.headers["Location"],
        )
        is not None
    )
    with pytest.raises(remote.RemoteLimitError, match="max_redirects"):
        limit_handler.redirect_request(
            request,
            response,
            302,
            "Found",
            response.headers,
            response.headers["Location"],
        )

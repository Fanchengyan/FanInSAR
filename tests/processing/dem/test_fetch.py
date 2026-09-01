"""Tests for the PROPOSAL-0041 geoid fetch/cache seam."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from faninsar.processing.dem.cache import (
    CachePathError,
    cache_artifact_path,
    resolve_cache_root,
)
from faninsar.processing.dem.fetch import (
    EGM2008_2_5,
    Fetch,
    GeoidArtifactError,
    GeoidOfflineError,
    GeoidResource,
)


def _resource(payload: bytes) -> GeoidResource:
    return GeoidResource(
        name="fixture",
        filename="fixture.bin",
        url="https://example.invalid/fixture.bin",
        expected_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        version="fixture-v1",
        vertical_crs="EPSG:5773",
    )


def test_fetch_cache_miss_downloads_then_hit_does_not_transport(tmp_path: Path) -> None:
    """A valid miss is downloaded once and a later hit is local-only."""
    payload = b"known geoid fixture"
    calls: list[str] = []

    def transport(url: str):
        calls.append(url)
        yield payload

    resolver = Fetch(
        tmp_path,
        transport=transport,
        resources={"fixture": _resource(payload)},
    )
    first = resolver.fetch("fixture")
    second = resolver.fetch("fixture")
    assert first == second
    assert first.read_bytes() == payload
    assert calls == ["https://example.invalid/fixture.bin"]


def test_corrupt_cache_is_removed_and_refetched(tmp_path: Path) -> None:
    """A corrupt entry never becomes a usable cache hit."""
    payload = b"correct"
    resource = _resource(payload)
    target = tmp_path / resource.name / resource.filename
    target.parent.mkdir()
    target.write_bytes(b"corrupt")
    calls: list[str] = []

    def transport(url: str):
        calls.append(url)
        yield payload

    resolved = Fetch(tmp_path, transport=transport, resources={"fixture": resource})
    assert resolved.fetch("fixture").read_bytes() == payload
    assert len(calls) == 1


def test_bad_download_never_remains_in_cache(tmp_path: Path) -> None:
    """Mismatched bytes are rejected and removed from the cache."""
    payload = b"correct"
    resource = _resource(payload)
    resolver = Fetch(
        tmp_path,
        transport=lambda _url: iter([b"wrong"]),
        resources={"fixture": resource},
    )
    with pytest.raises(GeoidArtifactError):
        resolver.fetch("fixture")
    assert not (tmp_path / resource.name / resource.filename).exists()


def test_offline_missing_resource_fails_without_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROJ_NETWORK=OFF blocks a missing required model."""
    monkeypatch.setenv("PROJ_NETWORK", "OFF")
    calls: list[str] = []
    resource = _resource(b"offline")
    resolver = Fetch(
        tmp_path,
        transport=lambda url: calls.append(url) or iter([b"offline"]),
        resources={"fixture": resource},
    )
    with pytest.raises(GeoidOfflineError):
        resolver.fetch("fixture")
    assert calls == []


def test_cache_root_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The environment override is expanded and normalized."""
    monkeypatch.setenv("FANINSAR_GEOID_CACHE", str(tmp_path / "cache"))
    assert resolve_cache_root() == (tmp_path / "cache").resolve()


def test_cache_path_rejects_traversal(tmp_path: Path) -> None:
    """Cache identity components cannot escape the configured root."""
    with pytest.raises(CachePathError):
        cache_artifact_path(tmp_path, "../escape", "file")


def test_egm2008_identity_is_pinned() -> None:
    """The accepted PROPOSAL-0041 EGM2008 identity remains exact."""
    assert EGM2008_2_5.version == "egm2008-2_5"
    assert EGM2008_2_5.expected_size == 80_585_622
    assert EGM2008_2_5.sha256 == (
        "4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a"
    )

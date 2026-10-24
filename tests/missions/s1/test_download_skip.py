"""Mission-layer skip semantics for failed Sentinel-1 download verification.

PROPOSAL-0055 slice 4: a download whose bytes fail provider checksum
verification (``RemoteIntegrityError("checksum_mismatch")``) excludes that
acquisition from the Stack — warned, recorded in the prepared-state record,
and excluded from pair formation — while other acquisitions continue.  Every
other download failure keeps the fail-closed construction boundary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from faninsar import remote
from faninsar.missions.s1.stack import (
    S1Stack,
    _excluded_acquisition_provenance,
    _input_verification_provenance,
)

if TYPE_CHECKING:
    from pathlib import Path


def _item(*assets: remote.RemoteAsset) -> remote.CatalogItem:
    """Build one normalized catalog item for executor seam tests."""
    acquisition = remote.AcquisitionMetadata(
        acquisition_id="S1_TEST",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
    )
    return remote.CatalogItem(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="S1-SLC",
        item_id="S1_TEST",
        acquisition=acquisition,
        assets={asset.key: asset for asset in assets},
        raw_metadata={},
    )


def _asset(key: str = "data") -> remote.RemoteAsset:
    """Build one complete remote SAFE asset."""
    return remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="S1-SLC",
        item_id="S1_TEST",
        key=key,
        href="https://datapool.asf.alaska.edu/S1.zip",
        media_type="application/zip",
        representation="complete",
    )


class _FakeStack:
    """Minimal prepared-stack double for the construction boundary."""

    def prepare(self) -> None:
        """Satisfy the post-construction preparation call."""


def test_checksum_mismatch_skips_acquisition_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A verification failure warns, records the exclusion, and keeps others."""
    failed = replace(_item(_asset("failed")), item_id="S1_FAILED")
    good = replace(_item(_asset("good")), item_id="S1_GOOD")
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [failed, good])
    )
    failed_destination = tmp_path / "inputs" / "S1_FAILED.zip"
    captured: list[tuple[tuple[Path, ...], dict[str, Any]]] = []

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del kwargs
        if asset.key == "failed":
            reason = "checksum_mismatch"
            message = "provider checksum mismatch"
            raise remote.RemoteIntegrityError(reason, message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"SAFE")
        return destination

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> _FakeStack:
        del cls
        captured.append((tuple(paths), kwargs))
        return _FakeStack()

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with caplog.at_level(logging.WARNING, logger="faninsar.missions.s1.stack"):
        S1Stack.from_search(object(), work_dir=tmp_path, product_type="scene")

    assert len(captured) == 1
    paths, kwargs = captured[0]
    # The failed acquisition never reaches pair formation.
    assert paths == (tmp_path / "inputs" / "S1_GOOD.zip",)
    # The exclusion is recorded in the prepared-state record, not only warned.
    assert kwargs["extra"]["excluded_acquisitions"] == (
        {
            "date": "2024-01-01T00:00:00+00:00",
            "path": str(failed_destination),
            "status": "failed",
            "reason": "checksum_mismatch",
        },
    )
    # The warning names the file and advises re-download.
    assert str(failed_destination) in caplog.text
    assert "re-download" in caplog.text


def test_other_integrity_errors_still_fail_the_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-verification integrity errors keep the fail-closed contract."""
    first = replace(_item(_asset("first")), item_id="S1_FIRST")
    second = replace(_item(_asset("second")), item_id="S1_SECOND")
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [first, second])
    )
    constructed = False

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del kwargs
        if asset.key == "second":
            reason = "content_length_mismatch"
            message = "truncated transfer"
            raise remote.RemoteIntegrityError(reason, message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"SAFE")
        return destination

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> Any:
        del cls, paths, kwargs
        nonlocal constructed
        constructed = True
        message = "from_safes must not run after a fail-closed download error"
        raise AssertionError(message)

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with pytest.raises(remote.RemoteIntegrityError) as integrity_error:
        S1Stack.from_search(object(), work_dir=tmp_path, product_type="scene")
    assert integrity_error.value.reason == "content_length_mismatch"
    assert not constructed


def test_network_transfer_failure_still_fails_the_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transport-level download errors keep the fail-closed contract."""
    item = _item(_asset())
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [item])
    )

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del asset, destination, kwargs
        message = "connection reset"
        raise OSError(message)

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> Any:
        del cls, paths, kwargs
        message = "from_safes must not run after a transport failure"
        raise AssertionError(message)

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with pytest.raises(OSError, match="connection reset"):
        S1Stack.from_search(object(), work_dir=tmp_path, product_type="scene")


def test_all_acquisitions_failed_verification_raises_without_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No surviving acquisition fails closed with a clear error."""
    only = replace(_item(_asset("only")), item_id="S1_ONLY")
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [only])
    )

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del asset, destination, kwargs
        reason = "checksum_mismatch"
        message = "provider checksum mismatch"
        raise remote.RemoteIntegrityError(reason, message)

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> Any:
        del cls, paths, kwargs
        message = "from_safes must not run without surviving acquisitions"
        raise AssertionError(message)

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with pytest.raises(ValueError, match="download verification"):
        S1Stack.from_search(object(), work_dir=tmp_path, product_type="scene")


def test_burst_poll_path_skips_verification_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The burst probe/download loop applies the same skip-and-record contract."""
    from faninsar.remote.transfer import _RemotePollObservation

    failed = replace(_item(_asset("burst-failed")), item_id="S1_BFAILED")
    good = replace(_item(_asset("burst-good")), item_id="S1_BGOOD")
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [failed, good])
    )

    def probe(asset: Any, **kwargs: Any) -> Any:
        del kwargs
        return _RemotePollObservation("ready", asset.href)

    monkeypatch.setattr(remote, "probe", probe)

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del kwargs
        if asset.key == "burst-failed":
            reason = "checksum_mismatch"
            message = "provider checksum mismatch"
            raise remote.RemoteIntegrityError(reason, message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"SAFE")
        return destination

    monkeypatch.setattr(remote, "download", download)
    captured: list[tuple[Path, ...]] = []

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> _FakeStack:
        del cls, kwargs
        captured.append(tuple(paths))
        return _FakeStack()

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    S1Stack.from_search(object(), work_dir=tmp_path)

    assert captured == [(tmp_path / "inputs" / "S1_BGOOD.zip",)]


def test_exclusion_provenance_serialization_format() -> None:
    """The persisted provenance mirror serializes records deterministically."""
    record = {
        "date": "2024-01-01T00:00:00+00:00",
        "path": "/work/inputs/S1_FAILED.zip",
        "status": "failed",
        "reason": "checksum_mismatch",
    }
    assert _excluded_acquisition_provenance((record,)) == {
        "excluded_acquisitions": json.dumps([record], sort_keys=True)
    }
    assert _excluded_acquisition_provenance(()) == {}


def test_input_verification_provenance_surfaces_per_input_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each downloaded input's verification record is surfaced in provenance.

    PROPOSAL-0060 goal: ``verification.status`` is a consumed provenance fact.
    The helper reads the download-time verification record of each input and
    serializes ``{status, algorithm, size}`` per input into one deterministic
    entry, so the run summary reports which inputs were verified, failed, or
    unverifiable. An input with no record carries ``status: unverifiable``.
    """
    # verified input
    verified = tmp_path / "S1_VERIFIED.zip"
    verified.write_bytes(b"verified-bytes")
    manifest = verified.with_name(f"{verified.name}.faninsar.remote.abc.json")
    manifest.write_text(
        json.dumps(
            {
                "identity": "abc",
                "verification": {
                    "algorithm": "md5",
                    "checksum": "d41d8cd98f00b204e9800998ecf8427e",
                    "status": "verified",
                    "size": 14,
                },
            }
        )
    )
    # unverifiable input (no manifest)
    unverifiable = tmp_path / "S1_NOVER.zip"
    unverifiable.write_bytes(b"plain")

    result = _input_verification_provenance((verified, unverifiable))

    entries = json.loads(result["input_verification"])
    by_path = {entry["path"]: entry for entry in entries}
    assert by_path[str(verified)] == {
        "path": str(verified),
        "status": "verified",
        "algorithm": "md5",
        "size": 14,
    }
    assert by_path[str(unverifiable)] == {
        "path": str(unverifiable),
        "status": "unverifiable",
        "algorithm": None,
        "size": 5,
    }

"""Mission-layer skip semantics for failed Sentinel-1 download verification.

PROPOSAL-0055 slice 4: a download whose bytes fail provider checksum
verification (``RemoteIntegrityError("checksum_mismatch")``) excludes that
acquisition from the Stack — warned, recorded in the prepared-state record,
and excluded from pair formation — while other acquisitions continue.  Every
other download failure keeps the fail-closed construction boundary.
"""

from __future__ import annotations

from tests.missions.s1.test_stack_search import _materialize_selected

import json
import logging
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from faninsar import remote
from faninsar.data.query import BoundingBox
from faninsar.missions.s1.s1search import S1Search
from faninsar.missions.s1.stack import (
    S1Stack,
    _excluded_acquisition_provenance,
    _input_verification_provenance,
)
from faninsar.stack.layout import staged_acquisition_path

if TYPE_CHECKING:
    from pathlib import Path

ROI = BoundingBox(100.0, 40.0, 100.4, 40.4, crs="EPSG:4326")

_LINES_PER_BURST = 8
_SAMPLES_PER_BURST = 16


def _asf_search(items: list[remote.CatalogItem]) -> S1Search:
    """Wrap ASF catalog items in an ``S1Search`` for ``from_search`` tests."""
    return S1Search(
        items=tuple(items),
        catalog="sentinel1-asf",
        spatial=ROI,
        query={},
    )


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


def _burst_xml(sensing_second: int) -> str:
    """Minimal one-burst IW1 annotation for a reassembled ASF product."""
    first = " ".join(["0"] * _LINES_PER_BURST)
    last = " ".join([str(_SAMPLES_PER_BURST - 1)] * _LINES_PER_BURST)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1A</missionId>
    <productType>SLC</productType>
    <polarisation>VV</polarisation>
    <mode>IW</mode>
    <swath>IW1</swath>
    <startTime>2016-12-07T11:18:{sensing_second:02d}.000000</startTime>
    <stopTime>2016-12-07T11:18:{sensing_second + 1:02d}.000000</stopTime>
  </adsHeader>
  <generalAnnotation>
    <productInformation>
      <rangeSamplingRate>64345238.12571428</rangeSamplingRate>
      <radarFrequency>5405000454.33435</radarFrequency>
      <azimuthSteeringRate>1.397440</azimuthSteeringRate>
    </productInformation>
    <orbitList count="2">
      <orbit>
        <time>2016-12-07T11:17:45.950212</time>
        <frame>Earth Fixed</frame>
        <position><x>-572093.4</x><y>5939124.4</y><z>3795662.9</z></position>
        <velocity><x>2075.1</x><y>-3800.2</y><z>6239.7</z></velocity>
      </orbit>
      <orbit>
        <time>2016-12-07T11:17:55.950213</time>
        <frame>Earth Fixed</frame>
        <position><x>-551342.0</x><y>5901124.0</y><z>3858060.0</z></position>
        <velocity><x>2075.0</x><y>-3800.0</y><z>6240.0</z></velocity>
      </orbit>
    </orbitList>
    <azimuthFmRateList count="1">
      <azimuthFmRate>
        <azimuthTime>2016-12-07T11:18:52.178090</azimuthTime>
        <t0>0.0053317936</t0>
        <azimuthFmRatePolynomial>-2330.5 451066.6</azimuthFmRatePolynomial>
      </azimuthFmRate>
    </azimuthFmRateList>
  </generalAnnotation>
  <imageAnnotation>
    <imageInformation>
      <slantRangeTime>0.0053317936</slantRangeTime>
      <rangePixelSpacing>2.329562</rangePixelSpacing>
      <azimuthPixelSpacing>13.96706</azimuthPixelSpacing>
      <azimuthTimeInterval>0.0020555563</azimuthTimeInterval>
      <numberOfSamples>{_SAMPLES_PER_BURST}</numberOfSamples>
      <numberOfLines>{_LINES_PER_BURST}</numberOfLines>
    </imageInformation>
  </imageAnnotation>
  <dopplerCentroid>
    <dcEstimateList count="1">
      <dcEstimate>
        <azimuthTime>2016-12-07T11:18:53.140830</azimuthTime>
        <t0>0.0053422084</t0>
        <geometryDcPolynomial>-3.992857 -287.3943 134566.6</geometryDcPolynomial>
      </dcEstimate>
    </dcEstimateList>
  </dopplerCentroid>
  <swathTiming>
    <linesPerBurst>{_LINES_PER_BURST}</linesPerBurst>
    <samplesPerBurst>{_SAMPLES_PER_BURST}</samplesPerBurst>
    <burstList count="1">
      <burst>
        <azimuthTime>2016-12-07T11:18:{sensing_second:02d}.426276</azimuthTime>
        <azimuthAnxTime>607.4725</azimuthAnxTime>
        <sensingTime>2016-12-07T11:18:{sensing_second:02d}.543845</sensingTime>
        <byteOffset>0</byteOffset>
        <firstValidSample>{first}</firstValidSample>
        <lastValidSample>{last}</lastValidSample>
      </burst>
    </burstList>
  </swathTiming>
  <geolocationGrid>
    <geolocationGridPointList count="2">
      <geolocationGridPoint><line>0</line><pixel>0</pixel><latitude>40.0</latitude><longitude>100.0</longitude></geolocationGridPoint>
      <geolocationGridPoint><line>{_LINES_PER_BURST - 1}</line><pixel>0</pixel><latitude>40.1</latitude><longitude>100.0</longitude></geolocationGridPoint>
    </geolocationGridPointList>
  </geolocationGrid>
</product>"""


def _fake_burst_download(asset: Any, destination: Path, **kwargs: Any) -> Path:
    """Download hook that stages a real one-burst SAFE zip."""
    del asset, kwargs
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    stem = "s1a-iw1-slc-vv-20161207t111853"
    data = (
        np.arange(_LINES_PER_BURST * _SAMPLES_PER_BURST, dtype=np.float32).reshape(
            _LINES_PER_BURST, _SAMPLES_PER_BURST
        )
        + 1j
    ).astype(np.complex64)
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=_LINES_PER_BURST,
        width=_SAMPLES_PER_BURST,
        count=1,
        dtype="complex64",
        crs=None,
        transform=Affine.identity(),
    ) as dataset:
        dataset.write(data, 1)
    tiff_bytes = memfile.read()
    memfile.close()
    manifest = '<?xml version="1.0"?><xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"/>'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("X.SAFE/manifest.safe", manifest)
        archive.writestr(f"X.SAFE/annotation/{stem}.xml", _burst_xml(53))
        archive.writestr(f"X.SAFE/measurement/{stem}.tiff", tiff_bytes)
    return destination


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
    failed_destination = staged_acquisition_path(tmp_path / "inputs", "S1_FAILED.zip")
    captured: list[tuple[tuple[Path, ...], dict[str, Any]]] = []

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        if asset.key == "failed":
            reason = "checksum_mismatch"
            message = "provider checksum mismatch"
            raise remote.RemoteIntegrityError(reason, message)
        return _fake_burst_download(asset, destination, **kwargs)

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> _FakeStack:
        del cls
        captured.append((tuple(paths), kwargs))
        return _FakeStack()

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with caplog.at_level(logging.WARNING, logger="faninsar.missions.s1.acquisition"):
        _materialize_selected(
            _asf_search([failed, good]), work_dir=tmp_path, provider="asf"
        )

    assert len(captured) == 1
    paths, kwargs = captured[0]
    # The failed acquisition never reaches pair formation: only the
    # surviving burst was reassembled into a path-SAFE for from_safes.
    assert len(paths) == 1
    assert paths[0].suffix == ".SAFE"
    assert "S1_FAILED" not in str(paths[0])
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
    constructed = False

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        if asset.key == "second":
            reason = "content_length_mismatch"
            message = "truncated transfer"
            raise remote.RemoteIntegrityError(reason, message)
        return _fake_burst_download(asset, destination, **kwargs)

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> Any:
        del cls, paths, kwargs
        nonlocal constructed
        constructed = True
        message = "from_safes must not run after a fail-closed download error"
        raise AssertionError(message)

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with pytest.raises(remote.RemoteIntegrityError) as integrity_error:
        _materialize_selected(
            _asf_search([first, second]), work_dir=tmp_path, provider="asf"
        )
    assert integrity_error.value.reason == "content_length_mismatch"
    assert not constructed


def test_network_transfer_failure_still_fails_the_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transport-level download errors keep the fail-closed contract."""
    item = _item(_asset())

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
        _materialize_selected(_asf_search([item]), work_dir=tmp_path, provider="asf")


def test_all_acquisitions_failed_verification_raises_without_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No surviving acquisition fails closed with a clear error."""
    only = replace(_item(_asset("only")), item_id="S1_ONLY")

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
        _materialize_selected(_asf_search([only]), work_dir=tmp_path, provider="asf")


def test_burst_path_skips_verification_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ASF burst download loop applies the same skip-and-record contract."""
    failed = replace(_item(_asset("burst-failed")), item_id="S1_BFAILED")
    good = replace(_item(_asset("burst-good")), item_id="S1_BGOOD")

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        if asset.key == "burst-failed":
            reason = "checksum_mismatch"
            message = "provider checksum mismatch"
            raise remote.RemoteIntegrityError(reason, message)
        return _fake_burst_download(asset, destination, **kwargs)

    monkeypatch.setattr(remote, "download", download)
    captured: list[tuple[Path, ...]] = []

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> _FakeStack:
        del cls, kwargs
        captured.append(tuple(paths))
        return _FakeStack()

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    _materialize_selected(_asf_search([failed, good]), work_dir=tmp_path, provider="asf")

    # Only the surviving burst was reassembled into a path-SAFE.
    assert len(captured) == 1
    assert len(captured[0]) == 1
    assert captured[0][0].suffix == ".SAFE"


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
    tmp_path: Path,
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

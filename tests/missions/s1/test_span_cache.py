"""PROPOSAL-0070: CDSE canvas manifest (S4) and span cache (S3) tests.

A completed item records ``.canvas_manifest.json`` with a planner
fingerprint; a restart skips intact items and refetches only unfinished
byte spans through ``.span-cache/<fingerprint>/`` part files.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pytest

from faninsar.data.query import BoundingBox
from faninsar.missions.s1 import acquisition as acquisition_mod
from faninsar.remote.fetch import RangeSpan

ROI = BoundingBox(100.0, 44.0, 100.4, 44.4, crs="EPSG:4326")

N_BURSTS = 6
LINES_PER_BURST = 10
SAMPLES_PER_BURST = 100

ANNOTATION_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/"
    "Products(S1A_TEST)/Nodes(annotation.xml)/$value"
)
MEASUREMENT_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/"
    "Products(S1A_TEST)/Nodes(measurement.tiff)/$value"
)
ANNOTATION_URL_B = ANNOTATION_URL.replace("S1A_TEST", "S1B_TEST").replace(
    "annotation.xml", "annotation-b.xml"
)
MEASUREMENT_URL_B = MEASUREMENT_URL.replace("S1A_TEST", "S1B_TEST").replace(
    "measurement.tiff", "measurement-b.tiff"
)


def _parent_xml() -> str:
    """Return annotation XML for a 6-burst synthetic IW1 parent swath."""
    burst_els: list[str] = []
    grid_els: list[str] = []
    for k in range(N_BURSTS):
        first = " ".join(["0"] * LINES_PER_BURST)
        last = " ".join([str(SAMPLES_PER_BURST - 1)] * LINES_PER_BURST)
        burst_els.append(
            f"""
      <burst>
        <azimuthTime>2016-12-07T11:18:{53 + k:02d}.426276</azimuthTime>
        <azimuthAnxTime>{607.4725 + 2.5731 * k}</azimuthAnxTime>
        <sensingTime>2016-12-07T11:18:{54 + k:02d}.543845</sensingTime>
        <byteOffset>{k * LINES_PER_BURST * SAMPLES_PER_BURST * 4}</byteOffset>
        <firstValidSample>{first}</firstValidSample>
        <lastValidSample>{last}</lastValidSample>
      </burst>"""
        )
        grid_els.extend(
            "        <geolocationGridPoint>"
            f"<line>{k * LINES_PER_BURST + local}</line>"
            f"<pixel>{pixel}</pixel>"
            f"<latitude>{40.0 + k + 0.01 * local:.2f}</latitude>"
            f"<longitude>{100.0 + 0.4 * pixel:.2f}</longitude>"
            "</geolocationGridPoint>"
            for local in (0, LINES_PER_BURST - 1)
            for pixel in (0, SAMPLES_PER_BURST - 1)
        )
    points = "\n".join(grid_els)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1A</missionId>
    <productType>SLC</productType>
    <polarisation>VV</polarisation>
    <mode>IW</mode>
    <swath>IW1</swath>
    <startTime>2016-12-07T11:18:53.426275</startTime>
    <stopTime>2016-12-07T11:19:18.571896</stopTime>
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
      <numberOfSamples>{SAMPLES_PER_BURST}</numberOfSamples>
      <numberOfLines>{N_BURSTS * LINES_PER_BURST}</numberOfLines>
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
    <linesPerBurst>{LINES_PER_BURST}</linesPerBurst>
    <samplesPerBurst>{SAMPLES_PER_BURST}</samplesPerBurst>
    <burstList count="{N_BURSTS}">{"".join(burst_els)}
    </burstList>
  </swathTiming>
  <geolocationGrid>
    <geolocationGridPointList count="{len(grid_els)}">
{points}
    </geolocationGridPointList>
  </geolocationGrid>
</product>"""


def _parent_raster() -> np.ndarray:
    """Return the synthetic parent SLC: real part = line, imag part = column."""
    lines = np.arange(N_BURSTS * LINES_PER_BURST, dtype=np.float32)[:, None]
    cols = np.arange(SAMPLES_PER_BURST, dtype=np.float32)[None, :]
    return (lines + 1j * cols).astype(np.complex64)


def _complex_int16_bytes(samples: np.ndarray) -> bytes:
    """Pack a complex64 array as interleaved little-endian int16 pairs."""
    real = np.asarray(samples.real, dtype=np.int16)
    imag = np.asarray(samples.imag, dtype=np.int16)
    interleaved = np.empty((*samples.shape, 2), dtype="<i2")
    interleaved[..., 0] = real
    interleaved[..., 1] = imag
    return interleaved.tobytes()


def _measurement_blob(data: np.ndarray) -> bytes:
    """Build a classic LE strip TIFF: header, int16 data rows, trailing IFD."""
    rows = data.shape[0]
    row_stride = data.shape[1] * 4
    data_start = 16
    offsets = tuple(data_start + i * row_stride for i in range(rows))
    counts = (row_stride,) * rows
    payload = _complex_int16_bytes(data)
    ifd_offset = data_start + len(payload)
    entries = [
        (259, 3, 1, (1,)),
        (278, 3, 1, (1,)),
        (273, 4, len(offsets), offsets),
        (279, 4, len(counts), counts),
    ]
    n = len(entries)
    cursor = ifd_offset + 2 + n * 12 + 4
    packed: list[tuple[int, int, int, int]] = []
    extras = bytearray()
    for tag, typ, count, values in entries:
        type_size = 2 if typ == 3 else 4
        payload_size = type_size * count
        if payload_size <= 4:
            pack = "<H" if typ == 3 else "<I"
            raw = b"".join(struct.pack(pack, v) for v in values)
            field = int.from_bytes(raw.ljust(4, b"\x00"), "little")
            packed.append((tag, typ, count, field))
        else:
            packed.append((tag, typ, count, cursor))
            extras.extend(b"".join(struct.pack("<I", v) for v in values))
            cursor += 4 * count
    ifd = bytearray(struct.pack("<H", n))
    for tag, typ, count, field in packed:
        ifd.extend(struct.pack("<HHII", tag, typ, count, field))
    ifd.extend(struct.pack("<I", 0))
    blob = bytearray(b"II" + struct.pack("<HI", 42, ifd_offset))
    blob = blob.ljust(data_start, b"\x00")
    blob += payload
    blob += ifd + extras
    return bytes(blob)


def _binding(
    product_id: str,
    annotation_url: str,
    measurement_url: str,
    *,
    measurement_size: int,
    annotation_size: int,
) -> dict[str, Any]:
    """Return one pre-bound CDSE item dict for ``materialize_cdse_roi_safes``."""
    return {
        "product_id": product_id,
        "annotation_url": annotation_url,
        "measurement_url": measurement_url,
        "measurement_size": measurement_size,
        "annotation_size": annotation_size,
        "polarization": "VV",
        "swath": "IW1",
    }


class _FakeRangeFetcher:
    """Mimics ``RangeFetcher.fetch_into``: resumes from the sink's file size.

    Class-level ``requests`` records ``(url, start, end, received, label)``
    per call; ``received`` is the byte offset the request resumed at, so a
    span restarted after an abort shows ``start + received``.
    """

    requests: ClassVar[list[tuple[str, int, int, int, str]]] = []
    abort_after_bytes: ClassVar[int | None] = None
    aborted: ClassVar[bool] = False
    blob: ClassVar[bytes] = b""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def fetch_into(self, span: RangeSpan, sink: Any) -> int:
        """Append ``body[received:]`` like the real resume contract."""
        path = Path(sink)
        received = path.stat().st_size if path.exists() else 0
        type(self).requests.append(
            (span.url, span.start, span.end, received, span.label)
        )
        body = type(self).blob[span.start : span.end + 1]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            abort_at = type(self).abort_after_bytes
            if abort_at is not None and not type(self).aborted:
                stream.write(body[received : received + abort_at])
                type(self).aborted = True
                message = "simulated mid-span abort"
                raise RuntimeError(message)
            stream.write(body[received:])
        return len(body)


@pytest.fixture
def mission_env(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch the fetch seams for materialize-level tests.

    Annotation and IFD probes stay on the ``fetch_ranges_async`` seam;
    measurement spans exercise the span-cache ``fetch_into`` path through
    a fake ``RangeFetcher``.
    """
    xml_text = _parent_xml()
    blob = _measurement_blob(_parent_raster())

    _FakeRangeFetcher.blob = blob
    _FakeRangeFetcher.requests = []
    _FakeRangeFetcher.abort_after_bytes = None
    _FakeRangeFetcher.aborted = False
    monkeypatch.setattr(acquisition_mod, "RangeFetcher", _FakeRangeFetcher)

    async def fake_fetch_ranges_async(
        spans: Any, *, fetcher: Any = None
    ) -> dict[str, bytes]:
        del fetcher
        bodies: dict[str, bytes] = {}
        for span in spans:
            if span.url in (ANNOTATION_URL, ANNOTATION_URL_B):
                bodies[span.label] = xml_text.encode("utf-8")
            elif span.url in (MEASUREMENT_URL, MEASUREMENT_URL_B):
                bodies[span.label] = blob[span.start : span.end + 1]
            else:
                bodies[span.label] = b"\x00" * (span.end - span.start + 1)
        return bodies

    monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch_ranges_async)
    return {
        "xml_text": xml_text,
        "blob": blob,
        "fetcher": _FakeRangeFetcher,
    }


def _materialize_items(
    tmp_path: Path,
    items: list[dict[str, Any]],
    **kwargs: Any,
) -> list[Path]:
    """Run ``materialize_cdse_roi_safes`` with pre-bound items."""
    return acquisition_mod.materialize_cdse_roi_safes(
        ROI,
        work_dir=tmp_path,
        granularity="roi",
        items=items,
        **kwargs,
    )


class TestSpanCacheFetchInto:
    """S3: per-span .part files + manifest under .span-cache/<F>/."""

    def test_completed_span_cache_deletes_parts_keeps_manifest(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """After the item completes, .part files are gone; manifest stays."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        cache_root = safe_root / ".span-cache"
        assert cache_root.is_dir()
        manifests = list(cache_root.glob("*/manifest.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        assert manifest["fingerprint"]
        assert manifest["spans"]
        for record in manifest["spans"]:
            assert record["state"] == "complete"
            assert record["part_size"] == record["end"] - record["start"] + 1
            assert record["sha1"]
            assert "?" not in record["url"]  # OData $value hrefs carry no query
        # Post-completion the .part payloads are deleted; manifest.json stays.
        assert not list(cache_root.glob("*/*.part"))

    def test_abort_mid_span_resumes_from_received_prefix(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """A mid-span abort keeps the partial .part; rerun resumes the tail."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        fetcher = mission_env["fetcher"]
        fetcher.abort_after_bytes = 7
        with pytest.raises(RuntimeError, match="abort"):
            _materialize_items(tmp_path, [item])
        first_requests = list(fetcher.requests)
        assert len(first_requests) == 1
        _url, start, end, _received, _label = first_requests[0]
        safe_root = tmp_path / "inputs" / "S1A_TEST-IW1-VV.SAFE"
        part_files = list((safe_root / ".span-cache").glob("*/*.part"))
        assert len(part_files) == 1
        assert part_files[0].stat().st_size == 7

        fetcher.requests.clear()
        fetcher.abort_after_bytes = None
        paths = _materialize_items(tmp_path, [item])
        assert paths[0] == safe_root
        resumed = [r for r in fetcher.requests if r[4].startswith(("row-", "col-"))]
        assert resumed
        # The aborted span resumed at start+received; other spans started at 0.
        resumed_first = [r for r in resumed if (r[1], r[2]) == (start, end)]
        assert resumed_first
        assert resumed_first[0][3] == 7

    def test_complete_record_with_size_drift_refetches_from_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """state=complete is honored only when part_size == actual == length."""
        cache = acquisition_mod._SpanCache(tmp_path / ".span-cache", "F" * 16)
        span = RangeSpan(MEASUREMENT_URL, 1000, 1999, "row-0-10")
        fetcher = _FakeRangeFetcher
        fetcher.blob = b"\xab" * 10_000
        fetcher.requests = []
        monkeypatch.setattr(acquisition_mod, "RangeFetcher", fetcher)
        cache.fetch((span,))
        manifest_path = cache.dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = manifest["spans"][0]
        assert record["state"] == "complete"

        # Truncate the .part: recorded part_size no longer matches actual.
        part = cache.dir / f"{record['key']}.part"
        part.write_bytes(part.read_bytes()[:400])
        fetcher.requests.clear()
        cache2 = acquisition_mod._SpanCache(tmp_path / ".span-cache", "F" * 16)
        bodies = cache2.fetch((span,))
        assert fetcher.requests[0][3] == 0  # refetched from byte 0
        assert len(bodies["row-0-10"]) == 1000

    def test_complete_record_with_matching_size_reads_part(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid complete record serves bytes from disk with no request."""
        cache = acquisition_mod._SpanCache(tmp_path / ".span-cache", "F" * 16)
        span = RangeSpan(MEASUREMENT_URL, 500, 599, "row-0-1")
        fetcher = _FakeRangeFetcher
        fetcher.blob = b"\xcd" * 1000
        fetcher.requests = []
        monkeypatch.setattr(acquisition_mod, "RangeFetcher", fetcher)
        cache.fetch((span,))
        fetcher.requests.clear()
        cache2 = acquisition_mod._SpanCache(tmp_path / ".span-cache", "F" * 16)
        bodies = cache2.fetch((span,))
        assert fetcher.requests == []
        assert bodies["row-0-1"] == b"\xcd" * 100

    def test_corrupt_same_size_part_detected_by_sha1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A same-size byte-corrupted .part fails the recorded sha1 -> refetch."""
        cache = acquisition_mod._SpanCache(tmp_path / ".span-cache", "F" * 16)
        span = RangeSpan(MEASUREMENT_URL, 0, 99, "row-0-1")
        fetcher = _FakeRangeFetcher
        fetcher.blob = b"\x77" * 200
        fetcher.requests = []
        monkeypatch.setattr(acquisition_mod, "RangeFetcher", fetcher)
        cache.fetch((span,))
        manifest = json.loads((cache.dir / "manifest.json").read_text())
        record = manifest["spans"][0]
        part = cache.dir / f"{record['key']}.part"
        corrupted = bytearray(part.read_bytes())
        corrupted[0] ^= 0xFF  # same length, wrong bytes
        part.write_bytes(bytes(corrupted))
        fetcher.requests.clear()
        cache2 = acquisition_mod._SpanCache(tmp_path / ".span-cache", "F" * 16)
        bodies = cache2.fetch((span,))
        assert fetcher.requests  # refetched, not read back
        assert bodies["row-0-1"] == b"\x77" * 100

    def test_span_manifest_url_carries_no_query(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """manifest.json url fields stay OData-canonical (no signed query)."""
        cache = acquisition_mod._SpanCache(tmp_path / ".span-cache", "F" * 16)
        span = RangeSpan(MEASUREMENT_URL + "?sig=abc", 0, 49, "row-0-1")
        fetcher = _FakeRangeFetcher
        fetcher.blob = b"\x55" * 100
        fetcher.requests = []
        monkeypatch.setattr(acquisition_mod, "RangeFetcher", fetcher)
        cache.fetch((span,))
        manifest = json.loads((cache.dir / "manifest.json").read_text())
        record = manifest["spans"][0]
        assert "?" not in record["url"]
        assert "sig" not in record["url"]

    def test_orphan_gc_keeps_overlapping_pending_partials(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Orphan <F'> dirs are GC'd unless a pending partial overlaps the plan."""
        root = tmp_path / ".span-cache"
        current = root / ("b" * 40)
        overlapping = root / "overlap-orphan"
        stale = root / "stale-orphan"
        for orphan in (overlapping, stale):
            orphan.mkdir(parents=True)
        plan_span = RangeSpan(MEASUREMENT_URL, 1000, 1999, "row-0-10")
        key = acquisition_mod._SpanCache.span_key(plan_span)
        # Overlapping orphan: pending record + partial .part for a planned span.
        (overlapping / f"{key}.part").write_bytes(b"\x00" * 100)
        (overlapping / "manifest.json").write_text(
            json.dumps(
                {
                    "fingerprint": "overlap-orphan",
                    "spans": [
                        {
                            "key": key,
                            "url": MEASUREMENT_URL,
                            "start": 1000,
                            "end": 1999,
                            "label": "row-0-10",
                            "state": "pending",
                            "part_size": 100,
                        }
                    ],
                }
            )
        )
        # Stale orphan: pending partial on a span outside the current plan.
        (stale / "dead.part").write_bytes(b"\x00" * 50)
        (stale / "manifest.json").write_text(
            json.dumps(
                {
                    "fingerprint": "stale-orphan",
                    "spans": [
                        {
                            "key": "dead",
                            "url": MEASUREMENT_URL,
                            "start": 9000,
                            "end": 9999,
                            "label": "row-9-10",
                            "state": "pending",
                            "part_size": 50,
                        }
                    ],
                }
            )
        )
        fetcher = _FakeRangeFetcher
        fetcher.blob = b"\x99" * 5000
        fetcher.requests = []
        monkeypatch.setattr(acquisition_mod, "RangeFetcher", fetcher)
        cache = acquisition_mod._SpanCache(root, "b" * 40)
        cache.fetch((plan_span,))
        assert overlapping.is_dir()  # pending partial overlapping the plan kept
        assert not stale.exists()  # zero-overlap orphan removed
        assert current.is_dir()


class TestCanvasManifest:
    """S4: item-level .canvas_manifest.json skip + fingerprint coverage."""

    def test_second_run_skips_completed_item(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """A complete manifest + intact SAFE skips all fetches on rerun."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        manifest_path = safe_root / ".canvas_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["complete"] is True
        assert manifest["fingerprint"]
        assert manifest["item"]["product"] == "S1A_TEST"
        fetcher = mission_env["fetcher"]
        fetcher.requests.clear()
        rerun = _materialize_items(tmp_path, [item])
        assert rerun == paths
        assert fetcher.requests == []  # no span fetches at all

    def test_two_item_stack_refetches_only_deleted_manifest_item(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """Deleting one item's manifest re-materializes only that item."""
        blob = mission_env["blob"]
        xml_text = mission_env["xml_text"]
        items = [
            _binding(
                "S1A_TEST",
                ANNOTATION_URL,
                MEASUREMENT_URL,
                measurement_size=len(blob),
                annotation_size=len(xml_text),
            ),
            _binding(
                "S1B_TEST",
                ANNOTATION_URL_B,
                MEASUREMENT_URL_B,
                measurement_size=len(blob),
                annotation_size=len(xml_text),
            ),
        ]
        paths = _materialize_items(tmp_path, items)
        assert len(paths) == 2
        (paths[0] / ".canvas_manifest.json").unlink()
        fetcher = mission_env["fetcher"]
        fetcher.requests.clear()
        rerun = _materialize_items(tmp_path, items)
        assert rerun == paths
        refetched_urls = {request[0] for request in fetcher.requests}
        assert refetched_urls == {MEASUREMENT_URL}
        assert MEASUREMENT_URL_B not in refetched_urls

    def test_complete_manifest_with_torn_safe_falls_through(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """complete:true + a truncated measurement TIFF re-materializes."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        measurement = next(safe_root.joinpath("measurement").glob("*.tiff"))
        # Truncate the measurement TIFF: present on disk, broken on open.
        payload = measurement.read_bytes()
        measurement.write_bytes(payload[: len(payload) // 3])
        fetcher = mission_env["fetcher"]
        fetcher.requests.clear()
        rerun = _materialize_items(tmp_path, [item])
        assert rerun == paths
        assert fetcher.requests  # re-materialized, not skipped

    def test_fingerprint_covers_planner_inputs(
        self,
        mission_env: dict[str, Any],
    ) -> None:
        """Same inputs -> same F; changed measurement_size/planner_version -> new F."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        fingerprint1 = acquisition_mod._canvas_fingerprint(
            item,
            ROI,
            coreg_mode="geometry",
            granularity="roi",
            dem_id="dem:none",
        )
        fingerprint2 = acquisition_mod._canvas_fingerprint(
            item,
            ROI,
            coreg_mode="geometry",
            granularity="roi",
            dem_id="dem:none",
        )
        assert fingerprint1 == fingerprint2
        changed_item = dict(item, measurement_size=item["measurement_size"] + 1)
        changed_size = acquisition_mod._canvas_fingerprint(
            changed_item,
            ROI,
            coreg_mode="geometry",
            granularity="roi",
            dem_id="dem:none",
        )
        assert changed_size != fingerprint1
        changed_dem = acquisition_mod._canvas_fingerprint(
            item,
            ROI,
            coreg_mode="geometry",
            granularity="roi",
            dem_id="dem:file:/x:abc",
        )
        assert changed_dem != fingerprint1

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            acquisition_mod, "_PLANNER_VERSION", "different-version-token"
        )
        changed_version = acquisition_mod._canvas_fingerprint(
            item,
            ROI,
            coreg_mode="geometry",
            granularity="roi",
            dem_id="dem:none",
        )
        monkeypatch.undo()
        assert changed_version != fingerprint1

    def test_changed_measurement_size_forces_rematerialization(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """A catalog-size change produces a different F and refetches the item."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        _materialize_items(tmp_path, [item])
        fetcher = mission_env["fetcher"]
        fetcher.requests.clear()
        changed = dict(item, measurement_size=item["measurement_size"] + 4096)
        _materialize_items(tmp_path, [changed])
        assert fetcher.requests  # fingerprint mismatch -> refetch


class TestSpanCacheOrphanGCIntegration:
    """S3 GC at item fetch start keeps only plan-overlapping pending partials."""

    def test_materialize_gc_removes_stale_orphan_keeps_overlap(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """End-to-end: orphan dir seeded under the item SAFE is GC'd on rerun."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        cache_root = safe_root / ".span-cache"
        # Adopt one real planned span from the finished manifest.
        current_manifest = json.loads(
            next(cache_root.glob("*/manifest.json")).read_text()
        )
        adopted = dict(current_manifest["spans"][0])
        adopted_key = adopted["key"]

        overlap_orphan = cache_root / "orphan-overlap"
        stale_orphan = cache_root / "orphan-stale"
        for orphan in (overlap_orphan, stale_orphan):
            orphan.mkdir()
        (overlap_orphan / f"{adopted_key}.part").write_bytes(b"\x00" * 32)
        adopted["state"] = "pending"
        adopted["part_size"] = 32
        (overlap_orphan / "manifest.json").write_text(
            json.dumps({"fingerprint": "orphan-overlap", "spans": [adopted]})
        )
        stale = dict(current_manifest["spans"][0])
        stale["key"] = "stalekey"
        stale["url"] = MEASUREMENT_URL
        stale["start"] = 999_000_000
        stale["end"] = 999_000_999
        stale["state"] = "pending"
        stale["part_size"] = 10
        (stale_orphan / "stalekey.part").write_bytes(b"\x00" * 10)
        (stale_orphan / "manifest.json").write_text(
            json.dumps({"fingerprint": "orphan-stale", "spans": [stale]})
        )

        # Force re-materialization so the item's span cache runs GC.
        (safe_root / ".canvas_manifest.json").unlink()
        _materialize_items(tmp_path, [item])
        assert overlap_orphan.is_dir()
        assert not stale_orphan.exists()

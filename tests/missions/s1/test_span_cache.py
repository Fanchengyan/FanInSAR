"""PROPOSAL-0070: CDSE canvas coverage and span-cache tests.

A completed item records ``coverage.json`` with a planner fingerprint;
a restart skips intact items.  Unfinished byte spans resume from
``tmp/s1-transfer/<canvas>/<fingerprint>/`` part files, which are
deleted when the canvas completes.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pytest

from faninsar.data.query import BoundingBox
from faninsar.missions.s1 import acquisition as acquisition_mod
from faninsar.remote.fetch import RangeSpan
from faninsar.stack.layout import s1_transfer_root, staged_acquisition_path

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
ANNOTATION_URL_C = ANNOTATION_URL.replace("S1A_TEST", "S1C_TEST").replace(
    "annotation.xml", "annotation-c.xml"
)
MEASUREMENT_URL_C = MEASUREMENT_URL.replace("S1A_TEST", "S1C_TEST").replace(
    "measurement.tiff", "measurement-c.tiff"
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

    async def fetch(self, spans: Any) -> dict[str, bytes]:
        """Return in-memory bodies so XML/IFD probes can reuse this session."""
        return {
            span.label: type(self).blob[span.start : span.end + 1] for span in spans
        }

    async def fetch_into(
        self, span: RangeSpan, sink: Any, *, on_bytes: Any = None
    ) -> int:
        """Append ``body[received:]`` like the real resume contract."""
        del on_bytes
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
    tmp_path_factory: pytest.TempPathFactory,
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

    # The roi branch now inverts every ROI through terrain-aware geo2rdr,
    # which needs a real DEM and a geophysically valid swath.  These tests
    # exercise the span-cache/manifest byte accounting on a synthetic
    # annotation, so the inversion is stubbed to the geolocation-grid
    # envelope the synthetic XML actually supports; ``_windows_for_polygons``
    # (the code under test) still plans the real byte ranges.
    def fake_roi_envelopes(roi, swath, dem):
        del swath, dem
        return [acquisition_mod.invert_geolocation_grid(xml_text, roi)]

    monkeypatch.setattr(acquisition_mod, "_roi_envelopes", fake_roi_envelopes)
    # ``materialize_cdse_roi_safes`` auto-fetches a DEM for the roi branch;
    # the stubbed inversion ignores its heights, but the canvas fingerprint
    # stats its file, so return a real local RasterDEM without any fetch.
    from faninsar.processing.geometry.dem import RasterDEM

    dem_path = tmp_path_factory.mktemp("dem") / "dem.tif"
    dem_path.write_bytes(b"stub-dem")
    monkeypatch.setattr(
        acquisition_mod,
        "_fetch_dem_for_roi",
        lambda *_args, **_kwargs: RasterDEM(path=dem_path),
    )
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


def _transfer_root(work_dir: Path, safe: Path) -> Path:
    """Return the per-canvas transfer scratch root."""
    return s1_transfer_root(work_dir, safe.name)


def _transfer_parts(work_dir: Path, safe: Path) -> list[Path]:
    """Return in-flight ``.part`` files for one canvas."""
    return list(_transfer_root(work_dir, safe).glob("*/*.part"))


def _progress_file(work_dir: Path, safe: Path) -> Path | None:
    """Return the first in-flight progress.json, if any."""
    matches = list(_transfer_root(work_dir, safe).glob("*/progress.json"))
    return matches[0] if matches else None


def _coverage(safe: Path) -> dict[str, Any]:
    """Load the published canvas coverage record."""
    return json.loads((safe / "coverage.json").read_text(encoding="utf-8"))


def test_global_roi_pool_overlaps_tracks_and_keeps_owner_bytes(
    tmp_path: Path,
    mission_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different tracks share one loop/session without colliding row labels."""
    import rasterio

    xml_text = mission_env["xml_text"]
    blobs = {
        MEASUREMENT_URL: mission_env["blob"],
        MEASUREMENT_URL_B: _measurement_blob(_parent_raster() + 1000),
    }
    items = [
        dict(
            _binding(
                product,
                annotation,
                measurement,
                measurement_size=len(blobs[measurement]),
                annotation_size=len(xml_text),
            ),
            mission_id="S1A",
            relative_orbit=orbit,
            orbit_direction="ASCENDING",
        )
        for product, annotation, measurement, orbit in (
            ("S1A_TEST", ANNOTATION_URL, MEASUREMENT_URL, 88),
            ("S1B_TEST", ANNOTATION_URL_B, MEASUREMENT_URL_B, 117),
        )
    ]
    metadata_calls: list[str] = []
    transfers: list[tuple[str, str, int, int]] = []
    opened: list[object] = []

    async def metadata(spans: Any, **kwargs: Any) -> dict[str, bytes]:
        del kwargs
        result = {}
        for span in spans:
            metadata_calls.append(span.url)
            body = (
                xml_text.encode()
                if span.url in (ANNOTATION_URL, ANNOTATION_URL_B)
                else blobs[span.url]
            )
            result[span.label] = body[span.start : span.end + 1]
        return result

    class ConcurrentFetcher:
        async def __aenter__(self) -> Self:
            opened.append(self)
            self.arrived: set[str] = set()
            self.both = asyncio.Event()
            return self

        async def __aexit__(self, *args: Any) -> None:
            del args

        async def fetch(self, spans: Any) -> dict[str, bytes]:
            return await metadata(spans)

        async def fetch_into(self, span: RangeSpan, sink: Any, **kwargs: Any) -> int:
            del kwargs
            assert span.url in metadata_calls
            self.arrived.add(span.url)
            if len(self.arrived) == 2:
                self.both.set()
            await asyncio.wait_for(self.both.wait(), timeout=0.5)
            transfers.append(
                (
                    span.url,
                    span.label,
                    id(asyncio.get_running_loop()),
                    threading.get_ident(),
                )
            )
            path = Path(sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blobs[span.url][span.start : span.end + 1])
            return span.end - span.start + 1

    monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", metadata)
    monkeypatch.setattr(acquisition_mod, "RangeFetcher", ConcurrentFetcher)
    paths = acquisition_mod.materialize_s1_acquisitions(
        ROI,
        provider="cdse",
        granularity="roi",
        work_dir=tmp_path,
        items=items,
        orbit_group=None,
    )
    assert len(opened) == 1
    assert {url for url, *_ in transfers} == set(blobs)
    assert len({loop for _, _, loop, _ in transfers}) == 1
    assert len({thread for _, _, _, thread in transfers}) == 1
    assert {path.parents[3].name for path in paths} == {"88A", "117A"}
    for path, expected_offset in zip(paths, (0, 1000), strict=True):
        with rasterio.open(next((path / "measurement").glob("*.tiff"))) as dataset:
            fetched = dataset.read(1)
        assert fetched.real.max() >= 40 + expected_offset
        assert fetched.real.max() < 60 + expected_offset
    transfers.clear()
    metadata_calls.clear()
    opened.clear()
    assert (
        acquisition_mod.materialize_s1_acquisitions(
            ROI,
            provider="cdse",
            granularity="roi",
            work_dir=tmp_path,
            items=items,
            orbit_group=None,
        )
        == paths
    )
    assert not transfers and not metadata_calls and not opened


def test_first_xml_starts_measurement_before_later_xml_finishes(
    tmp_path: Path,
    mission_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One finished annotation is planned and queued without waiting for the rest."""
    xml_text = mission_env["xml_text"]
    blobs = {
        MEASUREMENT_URL: mission_env["blob"],
        MEASUREMENT_URL_B: _measurement_blob(_parent_raster() + 1000),
    }
    items = [
        dict(
            _binding(
                product,
                annotation,
                measurement,
                measurement_size=len(blobs[measurement]),
                annotation_size=len(xml_text),
            ),
            mission_id="S1A",
            relative_orbit=orbit,
            orbit_direction="ASCENDING",
        )
        for product, annotation, measurement, orbit in (
            ("S1A_TEST", ANNOTATION_URL, MEASUREMENT_URL, 88),
            ("S1B_TEST", ANNOTATION_URL_B, MEASUREMENT_URL_B, 117),
        )
    ]
    order: list[str] = []
    state: dict[str, asyncio.Event] = {}

    async def metadata(spans: Any, **kwargs: Any) -> dict[str, bytes]:
        del kwargs
        result: dict[str, bytes] = {}
        for span in spans:
            if span.url == ANNOTATION_URL_B:
                order.append("xml-b-wait")
                gate = state.setdefault("a_meas", asyncio.Event())
                await asyncio.wait_for(gate.wait(), timeout=5)
                order.append("xml-b-done")
                result[span.label] = xml_text.encode()[span.start : span.end + 1]
            elif span.url == ANNOTATION_URL:
                order.append("xml-a-done")
                result[span.label] = xml_text.encode()[span.start : span.end + 1]
            else:
                result[span.label] = blobs[span.url][span.start : span.end + 1]
        return result

    class StreamingFetcher:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def fetch(self, spans: Any) -> dict[str, bytes]:
            return await metadata(spans)

        async def fetch_into(self, span: RangeSpan, sink: Any, **kwargs: Any) -> int:
            del kwargs
            state.setdefault("a_meas", asyncio.Event()).set()
            order.append(f"meas:{span.url}")
            path = Path(sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blobs[span.url][span.start : span.end + 1])
            return span.end - span.start + 1

    monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", metadata)
    monkeypatch.setattr(acquisition_mod, "RangeFetcher", StreamingFetcher)
    paths = acquisition_mod.materialize_s1_acquisitions(
        ROI,
        provider="cdse",
        granularity="roi",
        work_dir=tmp_path,
        items=items,
        orbit_group=None,
    )
    assert len(paths) == 2
    xml_b_done = order.index("xml-b-done")
    assert order.index("xml-a-done") < xml_b_done
    assert any(entry.startswith("meas:") for entry in order[:xml_b_done])


def test_xml_and_measurement_share_pool_one_to_one(
    tmp_path: Path,
    mission_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XML and measurement share one slot pool and grow 1:1 when both wait.

    A two-slot pool with three owners must start a measurement before the
    last XML (so remaining annotations do not drain first) and must still
    fetch that last XML while the measurement occupies a slot (so bulk
    transfer does not starve planning).
    """
    from faninsar.missions.s1 import roi_transfer as roi_transfer_mod

    xml_text = mission_env["xml_text"]
    blobs = {
        MEASUREMENT_URL: mission_env["blob"],
        MEASUREMENT_URL_B: mission_env["blob"],
        MEASUREMENT_URL_C: mission_env["blob"],
    }
    items = [
        dict(
            _binding(
                product,
                annotation,
                measurement,
                measurement_size=len(blobs[measurement]),
                annotation_size=len(xml_text),
            ),
            mission_id="S1A",
            relative_orbit=orbit,
            orbit_direction="ASCENDING",
        )
        for product, annotation, measurement, orbit in (
            ("S1A_TEST", ANNOTATION_URL, MEASUREMENT_URL, 88),
            ("S1B_TEST", ANNOTATION_URL_B, MEASUREMENT_URL_B, 117),
            ("S1C_TEST", ANNOTATION_URL_C, MEASUREMENT_URL_C, 66),
        )
    ]
    order: list[str] = []
    first_meas = asyncio.Event()
    xml_c_started = asyncio.Event()

    async def metadata(spans: Any, **kwargs: Any) -> dict[str, bytes]:
        del kwargs
        result: dict[str, bytes] = {}
        for span in spans:
            if span.url == ANNOTATION_URL_B:
                order.append("xml-b")
                await asyncio.wait_for(first_meas.wait(), timeout=5)
                result[span.label] = xml_text.encode()[span.start : span.end + 1]
            elif span.url == ANNOTATION_URL_C:
                order.append("xml-c")
                xml_c_started.set()
                result[span.label] = xml_text.encode()[span.start : span.end + 1]
            elif span.url == ANNOTATION_URL:
                order.append("xml-a")
                result[span.label] = xml_text.encode()[span.start : span.end + 1]
            else:
                result[span.label] = blobs[span.url][span.start : span.end + 1]
        return result

    class FairFetcher:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def fetch(self, spans: Any) -> dict[str, bytes]:
            return await metadata(spans)

        async def fetch_into(self, span: RangeSpan, sink: Any, **kwargs: Any) -> int:
            del kwargs
            order.append("meas-start")
            first_meas.set()
            await asyncio.wait_for(xml_c_started.wait(), timeout=5)
            order.append("meas-end")
            path = Path(sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blobs[span.url][span.start : span.end + 1])
            return span.end - span.start + 1

    monkeypatch.setattr(roi_transfer_mod, "_MAX_TRANSFER_TASKS", 2)
    monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", metadata)
    monkeypatch.setattr(acquisition_mod, "RangeFetcher", FairFetcher)
    paths = acquisition_mod.materialize_s1_acquisitions(
        ROI,
        provider="cdse",
        granularity="roi",
        work_dir=tmp_path,
        items=items,
        orbit_group=None,
    )
    assert len(paths) == 3
    assert order.index("meas-start") < order.index("xml-c")
    assert order.index("xml-c") < order.index("meas-end")


@pytest.mark.parametrize("corrupt", (False, True), ids=("resume", "corrupt-burst"))
def test_ready_burst_is_written_and_released_before_sibling_finishes(
    tmp_path: Path,
    mission_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    corrupt: bool,
) -> None:
    """A durable burst checkpoint replaces raw bytes before whole-SAFE completion."""
    import rasterio
    from rasterio.windows import Window

    item = _binding(
        "S1A_TEST",
        ANNOTATION_URL,
        MEASUREMENT_URL,
        measurement_size=len(mission_env["blob"]),
        annotation_size=len(mission_env["xml_text"]),
    )
    safe = staged_acquisition_path(tmp_path / "inputs", "S1A_TEST-IW1-VV.SAFE")
    observed: list[dict[str, Any]] = []
    fail = True

    class SlowSibling(_FakeRangeFetcher):
        async def fetch_into(self, span: RangeSpan, sink: Any, **kwargs: Any) -> int:
            if span.start >= 16 + 50 * SAMPLES_PER_BURST * 4 and fail:

                async def wait_for_burst() -> None:
                    while True:
                        progress = _progress_file(tmp_path, safe)
                        if progress is None:
                            await asyncio.sleep(0.001)
                            continue
                        try:
                            checkpoint = json.loads(progress.read_text())
                        except (OSError, ValueError):
                            await asyncio.sleep(0.001)
                            continue
                        if "4" in checkpoint.get("bursts", {}):
                            observed.append(checkpoint)
                            return
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_for_burst(), timeout=2)
                assert not (safe / "coverage.json").exists()
                assert not _transfer_parts(tmp_path, safe)
                raise RuntimeError("unfinished sibling")
            return await super().fetch_into(span, sink, **kwargs)

    monkeypatch.setattr(acquisition_mod, "RangeFetcher", SlowSibling)
    with pytest.raises(RuntimeError, match="unfinished sibling"):
        _materialize_items(tmp_path, [item])
    assert observed
    fail = False
    if corrupt:
        measurement = next((safe / "measurement").glob("*.tiff"))
        with rasterio.open(measurement, "r+") as dataset:
            dataset.write(
                np.array([[999 + 0j]], dtype=np.complex64), 1, window=Window(0, 0, 1, 1)
            )
    SlowSibling.requests.clear()
    assert _materialize_items(tmp_path, [item]) == [safe]
    assert SlowSibling.requests
    starts = [request[1] for request in SlowSibling.requests]
    boundary = 16 + 50 * SAMPLES_PER_BURST * 4
    if corrupt:
        assert any(start < boundary for start in starts)
    else:
        assert all(start >= boundary for start in starts)
    assert _progress_file(tmp_path, safe) is None
    assert not _transfer_parts(tmp_path, safe)
    assert _coverage(safe)["complete"]


def test_pool_stops_sibling_writers_before_closing_session(
    tmp_path: Path,
    mission_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed download drains every sibling before its session exits."""
    items = [
        _binding(
            product,
            annotation,
            measurement,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        for product, annotation, measurement in (
            ("S1A_TEST", ANNOTATION_URL, MEASUREMENT_URL),
            ("S1B_TEST", ANNOTATION_URL_B, MEASUREMENT_URL_B),
        )
    ]
    events: list[str] = []

    class FailingFetcher:
        async def __aenter__(self) -> Self:
            self.sibling_started = asyncio.Event()
            return self

        async def __aexit__(self, *args: Any) -> None:
            del args
            assert events and all(event == "sibling-stopped" for event in events)
            events.append("session-closed")

        async def fetch_into(self, span: RangeSpan, sink: Any, **kwargs: Any) -> int:
            del sink, kwargs
            if span.url == MEASUREMENT_URL:
                await self.sibling_started.wait()
                raise RuntimeError("download failed")
            self.sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                events.append("sibling-stopped")
            return 0

    monkeypatch.setattr(acquisition_mod, "RangeFetcher", FailingFetcher)
    with pytest.raises(RuntimeError, match="download failed"):
        _materialize_items(tmp_path, items)
    assert events[-1] == "session-closed"
    assert events[:-1] and all(event == "sibling-stopped" for event in events[:-1])
    assert not list(tmp_path.rglob("coverage.json"))


def test_roi_and_nesd_share_spatial_row_union_without_duplicate_requests(
    tmp_path: Path,
    mission_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping ROI/NESD rows fetch once, retaining both physical burst copies."""
    from faninsar.missions.s1.overlap import BurstOverlapWindow

    monkeypatch.setattr(
        acquisition_mod,
        "_roi_envelopes",
        lambda *args: [(43, 48, 20, 60), (46, 49, 30, 70)],
    )
    overlap = BurstOverlapWindow(
        swath="IW1",
        top_index=4,
        bottom_index=5,
        top_row0=7,
        bottom_row0=0,
        col0=0,
        n_lines=3,
        n_samples=100,
    )
    monkeypatch.setattr(
        acquisition_mod,
        "overlap_windows_for_canvas",
        lambda *args, **kwargs: (overlap,),
    )
    item = _binding(
        "S1A_TEST",
        ANNOTATION_URL,
        MEASUREMENT_URL,
        measurement_size=len(mission_env["blob"]),
        annotation_size=len(mission_env["xml_text"]),
    )
    _materialize_items(tmp_path, [item], coreg_mode="nesd")
    requested = [
        (start, end, label)
        for _, start, end, _, label in mission_env["fetcher"].requests
    ]
    stride = SAMPLES_PER_BURST * 4
    assert requested == [
        (16 + 43 * stride, 16 + 50 * stride - 1, "row-43-50"),
        (16 + 50 * stride, 16 + 53 * stride - 1, "row-50-53"),
    ]


def test_canvas_sync_failure_keeps_raw_spans_and_publishes_no_checkpoint(
    tmp_path: Path,
    mission_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disk I/O failure cannot authorize deleting the only durable raw bytes."""
    import errno
    from faninsar.missions.s1.io import canvas as canvas_mod

    real_fsync = canvas_mod.os.fsync
    original_sync = canvas_mod._sync_canvas_file
    syncing_canvas = False

    def fail_sync(descriptor: int) -> None:
        if syncing_canvas:
            raise OSError(errno.EIO, "simulated disk I/O error")
        real_fsync(descriptor)

    def canvas_sync(path: Path) -> None:
        nonlocal syncing_canvas
        syncing_canvas = True
        try:
            original_sync(path)
        finally:
            syncing_canvas = False

    monkeypatch.setattr(canvas_mod.os, "fsync", fail_sync)
    monkeypatch.setattr(canvas_mod, "_sync_canvas_file", canvas_sync)
    item = _binding(
        "S1A_TEST",
        ANNOTATION_URL,
        MEASUREMENT_URL,
        measurement_size=len(mission_env["blob"]),
        annotation_size=len(mission_env["xml_text"]),
    )
    with pytest.raises(OSError, match="disk I/O"):
        _materialize_items(tmp_path, [item])
    safe = staged_acquisition_path(tmp_path / "inputs", "S1A_TEST-IW1-VV.SAFE")
    assert _transfer_parts(tmp_path, safe)
    assert not (safe / "coverage.json").exists()
    assert not (safe / ".span-cache").exists()
    assert _progress_file(tmp_path, safe) is None


def test_executed_window_change_invalidates_canvas_with_same_dem_stat(
    tmp_path: Path, mission_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume identity follows executed placement/validity, including DEM effects."""
    item = _binding("S1A_TEST", ANNOTATION_URL, MEASUREMENT_URL,
                    measurement_size=len(mission_env["blob"]),
                    annotation_size=len(mission_env["xml_text"]))
    envelopes = [(43, 48, 20, 60)]
    planned = []

    def plan(*args):
        planned.append(tuple(envelopes))
        return envelopes

    monkeypatch.setattr(acquisition_mod, "_roi_envelopes", plan)
    safe = _materialize_items(tmp_path, [item])[0]
    original = _coverage(safe)["fingerprint"]
    mission_env["fetcher"].requests.clear()
    assert _materialize_items(tmp_path, [item]) == [safe]
    assert not mission_env["fetcher"].requests
    assert len(planned) == 2  # each attempt plans once, including the payload skip
    envelopes[:] = [(44, 48, 21, 60)]
    _materialize_items(tmp_path, [item])
    assert mission_env["fetcher"].requests
    assert _coverage(safe)["fingerprint"] != original
    assert len(planned) == 3


def test_dem_stat_change_with_identical_executed_plan_reuses_canvas(
    tmp_path: Path, mission_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """File metadata is not the scientific identity of the sampled plan."""
    from faninsar.processing.geometry.dem import RasterDEM

    dem_path = tmp_path / "dem.tif"
    dem_path.write_bytes(b"a")
    dem = RasterDEM(path=dem_path)
    item = _binding("S1A_TEST", ANNOTATION_URL, MEASUREMENT_URL,
                    measurement_size=len(mission_env["blob"]),
                    annotation_size=len(mission_env["xml_text"]))
    safe = _materialize_items(tmp_path, [item], dem=dem)[0]
    fingerprint = _coverage(safe)["fingerprint"]
    mission_env["fetcher"].requests.clear()
    dem_path.write_bytes(b"a changed file whose executed plan is identical")
    assert _materialize_items(tmp_path, [item], dem=dem) == [safe]
    assert not mission_env["fetcher"].requests
    assert _coverage(safe)["fingerprint"] == fingerprint


class TestSpanCacheFetchInto:
    """S3: per-span .part files under tmp/s1-transfer/<canvas>/<F>/."""

    def test_completed_span_cache_deletes_parts_keeps_manifest(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """After the item completes, transfer scratch is gone; coverage stays."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        coverage = _coverage(safe_root)
        assert coverage["complete"] is True
        assert coverage["fingerprint"]
        assert coverage["parent"] == "S1A_TEST"
        assert coverage["swath"] == "IW1"
        assert coverage["polarization"] == "VV"
        assert coverage["bursts"]
        assert all("index" in burst for burst in coverage["bursts"])
        assert not (safe_root / ".span-cache").exists()
        assert not (safe_root / ".canvas_manifest.json").exists()
        # Raw transfer bytes are fully reclaimed; only the small durable
        # plan-metadata caches (original annotation, TIFF strip tables)
        # may remain to support a zero-network rerun.
        scratch = _transfer_root(tmp_path, safe_root)
        if scratch.exists():
            assert {path.name for path in scratch.iterdir()} <= {
                "annotations",
                "layouts",
            }
        assert not _transfer_parts(tmp_path, safe_root)

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
        # Concurrent fetches may issue sibling requests before the abort
        # cancels them; what matters is the partial .part is kept and the
        # aborted span recorded its 7 received bytes.
        first_requests = list(fetcher.requests)
        aborted = [r for r in first_requests if r[3] == 0 and r[4].startswith("row-")]
        assert aborted
        _url, start, end, _received, _label = aborted[0]
        safe_root = staged_acquisition_path(tmp_path / "inputs", "S1A_TEST-IW1-VV.SAFE")
        part_files = _transfer_parts(tmp_path, safe_root)
        assert part_files
        assert any(p.stat().st_size == 7 for p in part_files)

        fetcher.requests.clear()
        fetcher.abort_after_bytes = None
        paths = _materialize_items(tmp_path, [item])
        assert paths[0] == safe_root
        resumed = [
            r
            for r in fetcher.requests
            if r[4] == _label and (r[1], r[2]) == (start, end)
        ]
        # The aborted span resumed at start+received (received=7).
        assert resumed
        assert resumed[0][3] == 7

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
    """S4: item-level coverage.json skip + fingerprint coverage."""

    def test_second_run_skips_completed_item(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """A complete coverage record + intact SAFE skips all fetches on rerun."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        coverage = _coverage(safe_root)
        assert coverage["complete"] is True
        assert coverage["fingerprint"]
        assert coverage["parent"] == "S1A_TEST"
        fetcher = mission_env["fetcher"]
        fetcher.requests.clear()
        rerun = _materialize_items(tmp_path, [item])
        assert rerun == paths
        assert fetcher.requests == []  # no span fetches at all

    def test_legacy_hidden_manifest_is_promoted_on_skip(
        self,
        tmp_path: Path,
        mission_env: dict[str, Any],
    ) -> None:
        """A complete ``.canvas_manifest.json`` is rewritten as ``coverage.json``."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        coverage = _coverage(safe_root)
        (safe_root / "coverage.json").unlink()
        (safe_root / ".canvas_manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "created": coverage["created"],
                    "fingerprint": coverage["fingerprint"],
                    "item": {
                        "n_bursts": len(coverage["bursts"]),
                        "pol": "VV",
                        "product": "S1A_TEST",
                        "swath": "IW1",
                    },
                }
            ),
            encoding="utf-8",
        )
        (safe_root / ".span-cache" / "old").mkdir(parents=True)
        fetcher = mission_env["fetcher"]
        fetcher.requests.clear()
        rerun = _materialize_items(tmp_path, [item])
        assert rerun == paths
        assert fetcher.requests == []
        promoted = _coverage(safe_root)
        assert promoted["complete"] is True
        assert promoted["parent"] == "S1A_TEST"
        assert promoted["bursts"]
        assert not (safe_root / ".canvas_manifest.json").exists()
        assert not (safe_root / ".span-cache").exists()

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
        (paths[0] / "coverage.json").unlink()
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

    def test_dem_identity_follows_planner_dem_use(self, tmp_path: Path) -> None:
        """Every ROI-granularity plan consumes the DEM, so F must see it."""
        from faninsar.processing.geometry.dem import RasterDEM

        dem_a_path = tmp_path / "dem_a.tif"
        dem_b_path = tmp_path / "dem_b.tif"
        dem_a_path.write_bytes(b"heights-a")
        dem_b_path.write_bytes(b"heights-bb")
        dem_a = RasterDEM(path=dem_a_path)
        dem_b = RasterDEM(path=dem_b_path)

        id_a = acquisition_mod._canvas_dem_id(ROI, granularity="roi", dem=dem_a)
        id_b = acquisition_mod._canvas_dem_id(ROI, granularity="roi", dem=dem_b)
        assert id_a != "dem:none"
        assert id_a != id_b
        assert (
            acquisition_mod._canvas_dem_id(
                ROI, granularity="roi", dem=RasterDEM(path=dem_a_path)
            )
            == id_a
        )
        # Burst granularity plans from the geolocation grid without a DEM.
        assert (
            acquisition_mod._canvas_dem_id(ROI, granularity="burst", dem=dem_a)
            == "dem:none"
        )

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
        """End-to-end: orphan dir seeded under transfer scratch is GC'd on rerun."""
        item = _binding(
            "S1A_TEST",
            ANNOTATION_URL,
            MEASUREMENT_URL,
            measurement_size=len(mission_env["blob"]),
            annotation_size=len(mission_env["xml_text"]),
        )
        paths = _materialize_items(tmp_path, [item])
        safe_root = paths[0]
        (safe_root / "coverage.json").unlink()
        cache_root = _transfer_root(tmp_path, safe_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        blob_end = len(mission_env["blob"]) - 1
        overlap_key = acquisition_mod._SpanCache.span_key(
            RangeSpan(MEASUREMENT_URL, 0, blob_end, "overlap")
        )
        overlap_orphan = cache_root / "orphan-overlap"
        stale_orphan = cache_root / "orphan-stale"
        for orphan in (overlap_orphan, stale_orphan):
            orphan.mkdir()
        (overlap_orphan / f"{overlap_key}.part").write_bytes(b"\x00" * 32)
        (overlap_orphan / "manifest.json").write_text(
            json.dumps(
                {
                    "fingerprint": "orphan-overlap",
                    "spans": [
                        {
                            "key": overlap_key,
                            "url": MEASUREMENT_URL,
                            "start": 0,
                            "end": blob_end,
                            "label": "overlap",
                            "state": "pending",
                            "part_size": 32,
                        }
                    ],
                }
            )
        )
        (stale_orphan / "stalekey.part").write_bytes(b"\x00" * 10)
        (stale_orphan / "manifest.json").write_text(
            json.dumps(
                {
                    "fingerprint": "orphan-stale",
                    "spans": [
                        {
                            "key": "stalekey",
                            "url": MEASUREMENT_URL,
                            "start": 999_000_000,
                            "end": 999_000_999,
                            "label": "stale",
                            "state": "pending",
                            "part_size": 10,
                        }
                    ],
                }
            )
        )

        _materialize_items(tmp_path, [item])
        assert overlap_orphan.is_dir()
        assert not stale_orphan.exists()

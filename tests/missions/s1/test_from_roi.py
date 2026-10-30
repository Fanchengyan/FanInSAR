"""Tests for S1Stack.from_search (PROPOSAL-0062).

``from_search`` is the single acquisition entry: it takes a pre-filtered
:class:`~faninsar.missions.s1.s1search.S1Search` and dispatches on
``search.catalog`` — ``sentinel1-cdse`` materializes full-burst-height
canvases before ``from_safes``; ``sentinel1-asf`` downloads complete
burst/scene products.  ``from_roi`` is deleted.
"""

from __future__ import annotations

import ast
import inspect
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from faninsar import remote
from faninsar.data.query import BoundingBox, Points
from faninsar.missions.s1 import acquisition as acquisition_mod
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.missions.s1.io.canvas import BurstFetchedWindow, RoiSafeCanvas
from faninsar.missions.s1.overlap import BurstOverlapWindow
from faninsar.missions.s1.s1search import S1Search
from faninsar.missions.s1.stack import S1Stack
from faninsar.remote.errors import RemoteAccessError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from faninsar.missions.s1.types import S1Swath

ROI = BoundingBox(100.0, 44.0, 100.4, 44.4, crs="EPSG:4326")

LEGAL_COMBOS: tuple[tuple[str | None, str | None, str, str], ...] = (
    (None, None, "roi", "cdse"),
    ("roi", None, "roi", "cdse"),
    ("roi", "cdse", "roi", "cdse"),
    ("burst", None, "burst", "cdse"),
    ("burst", "cdse", "burst", "cdse"),
    ("burst", "asf", "burst", "asf"),
    ("scene", None, "scene", "asf"),
    ("scene", "asf", "scene", "asf"),
)
ILLEGAL_COMBOS: tuple[tuple[str, str], ...] = (
    ("roi", "asf"),
    ("scene", "cdse"),
)

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


def _cdse_asset(
    key: str,
    href: str,
    *,
    media_type: str | None = None,
    size_bytes: int | None = None,
    properties: dict[str, object] | None = None,
) -> remote.RemoteAsset:
    """Build one CDSE catalog asset for bind tests."""
    return remote.RemoteAsset(
        provider="CDSE",
        catalog="sentinel1-cdse",
        collection="SENTINEL-1",
        item_id="S1A_TEST",
        key=key,
        href=href,
        media_type=media_type,
        size_bytes=size_bytes,
        properties=properties or {},
    )


def _cdse_item(*assets: remote.RemoteAsset) -> remote.CatalogItem:
    """Build one CDSE catalog item wrapping the given assets."""
    return remote.CatalogItem(
        provider="CDSE",
        catalog="sentinel1-cdse",
        collection="SENTINEL-1",
        item_id="S1A_TEST",
        acquisition=remote.AcquisitionMetadata(
            acquisition_id="S1A_TEST",
            start=datetime(2016, 12, 7, tzinfo=UTC),
            end=datetime(2016, 12, 7, tzinfo=UTC),
        ),
        assets={asset.key: asset for asset in assets},
        raw_metadata={"polarization": "VV"},
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


@pytest.fixture
def parent_swath() -> Iterator[tuple[S1Swath, MemoryFile, str, np.ndarray]]:
    """Yield a parsed synthetic parent swath plus its raster and XML."""
    xml_text = _parent_xml()
    data = _parent_raster()
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="complex64",
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(data, 1)
    swath = parse_annotation_xml(
        xml_text,
        annotation_path="memory://parent-annotation.xml",
        measurement_path=memfile.name,
    )
    yield swath, memfile, xml_text, data
    memfile.close()


def _resolve_roi_acquisition(**kwargs: Any) -> tuple[str, str]:
    """Call the mission resolver; fail as an assertion if it is missing."""
    resolver = getattr(acquisition_mod, "resolve_roi_acquisition", None)
    assert resolver is not None, (
        "resolve_roi_acquisition is the granularity/provider table resolver "
        "consumed by the materialization layer"
    )
    return resolver(**kwargs)


def _cdse_search(item: remote.CatalogItem | None = None) -> S1Search:
    """Wrap one CDSE ``CatalogItem`` (or the default) in an ``S1Search``."""
    return S1Search(
        items=(
            item
            if item is not None
            else _cdse_item(
                _cdse_asset(
                    "annotation",
                    ANNOTATION_URL,
                    media_type="application/xml",
                    size_bytes=845_602,
                ),
                _cdse_asset(
                    "measurement",
                    MEASUREMENT_URL,
                    media_type="image/tiff",
                    size_bytes=1_135_276_112,
                    properties={"file:size": 1_135_276_112, "swath": "IW1"},
                ),
            ),
        ),
        catalog="sentinel1-cdse",
        spatial=ROI,
        query={},
    )


def _from_search(search: S1Search, **kwargs: Any) -> Any:
    """Call S1Stack.from_search; fail as an assertion if it is missing."""
    constructor = getattr(S1Stack, "from_search", None)
    assert constructor is not None, "S1Stack.from_search is required by PROPOSAL-0062"
    return constructor(search, **kwargs)


class TestResolutionMatrix:
    """The granularity/provider table still resolves; illegal combos fail fast."""

    @pytest.mark.parametrize(
        ("granularity", "provider", "expected_granularity", "expected_provider"),
        LEGAL_COMBOS,
    )
    def test_legal_combo_infers_provider(
        self,
        granularity: str | None,
        provider: str | None,
        expected_granularity: str,
        expected_provider: str,
    ) -> None:
        """Legal combinations resolve to the documented provider."""
        kwargs: dict[str, str] = {}
        if granularity is not None:
            kwargs["granularity"] = granularity
        if provider is not None:
            kwargs["provider"] = provider
        assert _resolve_roi_acquisition(**kwargs) == (
            expected_granularity,
            expected_provider,
        )

    @pytest.mark.parametrize(("granularity", "provider"), ILLEGAL_COMBOS)
    def test_illegal_combo_raises_citing_table(
        self,
        granularity: str,
        provider: str,
    ) -> None:
        """Illegal combinations raise ValueError that cites the matrix table."""
        with pytest.raises(ValueError, match="table") as resolved:
            _resolve_roi_acquisition(granularity=granularity, provider=provider)
        message = str(resolved.value)
        assert granularity in message
        assert provider in message


def test_odata_annotation_stem_uses_nodes_filename() -> None:
    """Canvas filenames come from ``Nodes(name.xml)``, not the ``$value`` leaf."""
    from faninsar.missions.s1.io.export import _annotation_stem

    url = (
        "https://zipper.dataspace.copernicus.eu/odata/v1/"
        "Products(078a8ae1-47f1-40ed-87f6-99a3e62ca665)/"
        "Nodes(S1C_IW_SLC.SAFE)/Nodes(annotation)/"
        "Nodes(s1c-iw1-slc-vv-20251120t111823-005097-004.xml)/$value"
    )
    assert _annotation_stem(url) == "s1c-iw1-slc-vv-20251120t111823-005097-004"
    assert _annotation_stem("memory://parent-annotation.xml") == "parent-annotation"


def test_from_roi_is_gone() -> None:
    """``from_roi`` is deleted with no deprecation shim (PROPOSAL-0062)."""
    assert not hasattr(S1Stack, "from_roi")
    with pytest.raises(AttributeError):
        S1Stack.from_roi  # noqa: B018
    with pytest.raises(AttributeError, match="from_roi"):
        S1Stack.from_roi(object(), work_dir="unused")  # type: ignore[misc]


def test_from_search_rejects_non_s1search() -> None:
    """``from_search`` requires an S1Search, not a raw spatial query."""
    with pytest.raises(TypeError, match="S1Search"):
        S1Stack.from_search(ROI, work_dir="unused")


def test_from_search_rejects_empty_result() -> None:
    """An empty S1Search fails fast with a clear error."""
    empty = S1Search(items=(), catalog="sentinel1-cdse", spatial=ROI, query={})
    with pytest.raises(ValueError, match="empty"):
        S1Stack.from_search(empty, work_dir="unused")


def test_from_search_signature_matches_proposal() -> None:
    """The public signature matches the PROPOSAL-0062 constructor contract."""
    constructor = getattr(S1Stack, "from_search", None)
    assert constructor is not None, "S1Stack.from_search is required by PROPOSAL-0062"
    parameters = inspect.signature(constructor).parameters
    assert next(iter(parameters)) == "search"
    for name in (
        "roi",
        "polarizations",
        "work_dir",
        "granularity",
        "product_type",
        "coreg_mode",
        "orbit_paths",
        "calibration_paths",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["roi"].default is None
    assert parameters["polarizations"].default == ("VV",)
    assert parameters["granularity"].default is None
    assert parameters["product_type"].default is None
    assert parameters["coreg_mode"].default == "geometry"


class TestCdseRoiCanvasOrder:
    """CDSE roi path: canvas then overlap splice then from_safes then prepare."""

    def test_write_canvas_then_from_safes_sees_full_burst_height(
        self,
        parent_swath: tuple[S1Swath, MemoryFile, str, np.ndarray],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CDSE from_search binds the item, writes canvas, then from_safes."""
        swath, _memfile, xml_text, data = parent_swath
        order: list[str] = []
        captured_paths: list[tuple[Path, ...]] = []
        prepared: list[bool] = []
        windows = [
            BurstFetchedWindow(
                burst_index=4,
                row0=3,
                col0=20,
                samples=np.ascontiguousarray(data[43:48, 20:60]),
            ),
            BurstFetchedWindow(
                burst_index=5,
                row0=0,
                col0=10,
                samples=np.ascontiguousarray(data[50:55, 10:50]),
            ),
        ]

        from faninsar.missions.s1 import acquisition as acquisition_mod
        from faninsar.missions.s1.io import canvas as canvas_mod

        async def fake_fetch_ranges_async(
            spans: Any,
            *,
            fetcher: Any = None,
        ) -> dict[str, bytes]:
            del fetcher
            order.append("fetch_ranges")
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.url == ANNOTATION_URL:
                    bodies[span.label] = xml_text.encode("utf-8")
                else:
                    bodies[span.label] = b"\x00" * (span.end - span.start + 1)
            return bodies

        original_write = canvas_mod.write_roi_safe_canvas
        original_splice = canvas_mod.splice_overlap_into_canvas

        def tracking_write(*args: Any, **kwargs: Any) -> RoiSafeCanvas:
            order.append("write_canvas")
            handle = original_write(*args, **kwargs)
            with rasterio.open(handle.measurement_path) as dataset:
                assert dataset.height == N_BURSTS * LINES_PER_BURST
                assert dataset.width == SAMPLES_PER_BURST
            return handle

        def tracking_splice(*args: Any, **kwargs: Any) -> None:
            order.append("splice_overlap")
            original_splice(*args, **kwargs)

        class FakeStack:
            def prepare(self) -> None:
                order.append("prepare")
                prepared.append(True)

        def fake_from_safes(cls: Any, paths: Any, **kwargs: Any) -> FakeStack:
            del cls, kwargs
            order.append("from_safes")
            captured_paths.append(tuple(paths))
            for path in paths:
                measurement = next(Path(path).joinpath("measurement").glob("*.tiff"))
                with rasterio.open(measurement) as dataset:
                    assert dataset.height == N_BURSTS * LINES_PER_BURST
                    assert dataset.width == SAMPLES_PER_BURST
            return FakeStack()

        monkeypatch.setattr(
            "faninsar.remote.fetch.fetch_ranges_async",
            fake_fetch_ranges_async,
        )
        monkeypatch.setattr(
            "faninsar.remote.fetch.ranges.fetch_ranges_async",
            fake_fetch_ranges_async,
        )
        monkeypatch.setattr(
            acquisition_mod,
            "fetch_ranges_async",
            fake_fetch_ranges_async,
        )
        monkeypatch.setattr(canvas_mod, "write_roi_safe_canvas", tracking_write)
        monkeypatch.setattr(
            "faninsar.missions.s1.acquisition.write_roi_safe_canvas",
            tracking_write,
        )
        monkeypatch.setattr(canvas_mod, "splice_overlap_into_canvas", tracking_splice)
        monkeypatch.setattr(
            "faninsar.missions.s1.acquisition.splice_overlap_into_canvas",
            tracking_splice,
        )

        def fake_windows(*_args: Any, **_kwargs: Any) -> Any:
            return swath, windows, xml_text

        def fake_overlaps(
            *_args: Any, **_kwargs: Any
        ) -> tuple[BurstOverlapWindow, ...]:
            return (
                BurstOverlapWindow(
                    swath="IW1",
                    top_index=4,
                    bottom_index=5,
                    top_row0=4,
                    bottom_row0=0,
                    col0=20,
                    n_lines=3,
                    n_samples=30,
                ),
            )

        monkeypatch.setattr(acquisition_mod, "windows_from_cdse_fetch", fake_windows)
        monkeypatch.setattr(
            acquisition_mod,
            "overlap_windows_for_canvas",
            fake_overlaps,
        )
        monkeypatch.setattr(S1Stack, "from_safes", classmethod(fake_from_safes))

        result = _from_search(
            _cdse_search(),
            work_dir=tmp_path,
            granularity="roi",
            coreg_mode="nesd",
        )
        assert isinstance(result, FakeStack)
        assert prepared == [True]
        assert captured_paths
        assert all(Path(path).name.endswith(".SAFE") for path in captured_paths[0])
        write_at = order.index("write_canvas")
        splice_at = order.index("splice_overlap")
        safes_at = order.index("from_safes")
        prepare_at = order.index("prepare")
        assert write_at < splice_at < safes_at < prepare_at
        assert "fetch_ranges" in order


def test_no_loop_bridge_uses_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no running loop the mission bridge uses ``asyncio.run``."""
    import asyncio

    from faninsar.missions.s1 import acquisition as acquisition_mod

    ran: list[str] = []
    original_run = asyncio.run

    def tracking_run(coro: Any, *args: Any, **kwargs: Any) -> Any:
        ran.append("asyncio.run")
        return original_run(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "run", tracking_run)
    monkeypatch.setattr(acquisition_mod, "asyncio", asyncio)

    async def fake_coro() -> str:
        return "ok"

    runner = getattr(acquisition_mod, "run_sync", None)
    assert runner is not None, "run_sync is the mission/stack async bridge"
    result = runner(fake_coro())
    assert result == "ok"
    assert ran == ["asyncio.run"]


def test_module_does_not_import_nest_asyncio() -> None:
    """Neither the stack nor the acquisition module may depend on nest_asyncio."""
    import faninsar.missions.s1.acquisition as acquisition_mod
    import faninsar.missions.s1.stack as stack_mod

    for module in (acquisition_mod, stack_mod):
        tree = ast.parse(inspect.getsource(module))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        assert "nest_asyncio" not in imported
        assert not hasattr(module, "nest_asyncio")


# ---------------------------------------------------------------------------
# C1/C2/C3/C4: strip-table decode, fail-closed fetch, OData bind, overlap splice
# ---------------------------------------------------------------------------

_TAG_COMPRESSION = 259
_TAG_STRIP_OFFSETS = 273
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTE_COUNTS = 279
_TAG_TILE_WIDTH = 322
_TAG_TILE_LENGTH = 323
_TYPE_SHORT = 3
_TYPE_LONG = 4

# Literal strip tables used by the IFD codec tests (not derived from the parser).
_DECODE_STRIP_OFFSETS = (1000, 1016, 1032)
_DECODE_STRIP_BYTE_COUNTS = (16, 16, 16)
_DECODE_ROWS_PER_STRIP = 2
_DECODE_SAMPLES = 2  # 2 samples * 4 bytes * 2 rows = 16 bytes per strip


def _classic_le_strip_tiff(
    *,
    compression: int = 1,
    strip_offsets: tuple[int, ...] = _DECODE_STRIP_OFFSETS,
    strip_byte_counts: tuple[int, ...] = _DECODE_STRIP_BYTE_COUNTS,
    rows_per_strip: int = _DECODE_ROWS_PER_STRIP,
    tiled: bool = False,
    ifd_offset: int = 16,
) -> bytes:
    """Return a classic little-endian TIFF whose IFD carries the given tables."""
    entries: list[tuple[int, int, int, tuple[int, ...]]] = [
        (_TAG_COMPRESSION, _TYPE_SHORT, 1, (compression,)),
        (_TAG_ROWS_PER_STRIP, _TYPE_SHORT, 1, (rows_per_strip,)),
        (_TAG_STRIP_OFFSETS, _TYPE_LONG, len(strip_offsets), strip_offsets),
        (_TAG_STRIP_BYTE_COUNTS, _TYPE_LONG, len(strip_byte_counts), strip_byte_counts),
    ]
    if tiled:
        entries.extend(
            (
                (_TAG_TILE_WIDTH, _TYPE_SHORT, 1, (16,)),
                (_TAG_TILE_LENGTH, _TYPE_SHORT, 1, (16,)),
            )
        )

    n_entries = len(entries)
    ifd_size = 2 + n_entries * 12 + 4
    cursor = ifd_offset + ifd_size
    packed: list[tuple[int, int, int, int]] = []
    extras = bytearray()
    for tag, typ, count, values in entries:
        type_size = 2 if typ == _TYPE_SHORT else 4
        payload_size = type_size * count
        if payload_size <= 4:
            pack = "<H" if typ == _TYPE_SHORT else "<I"
            raw = b"".join(struct.pack(pack, value) for value in values)
            value_field = int.from_bytes(raw.ljust(4, b"\x00"), "little")
            packed.append((tag, typ, count, value_field))
        else:
            packed.append((tag, typ, count, cursor))
            raw = b"".join(struct.pack("<I", value) for value in values)
            extras.extend(raw)
            cursor += len(raw)

    ifd = bytearray(struct.pack("<H", n_entries))
    for tag, typ, count, value_field in packed:
        ifd.extend(struct.pack("<HHII", tag, typ, count, value_field))
    ifd.extend(struct.pack("<I", 0))

    header = b"II" + struct.pack("<HI", 42, ifd_offset)
    blob = bytearray(header.ljust(ifd_offset, b"\x00"))
    blob.extend(ifd)
    blob.extend(extras)
    return bytes(blob)


def _probe_bodies_from_tiff(blob: bytes, *, ifd_offset: int = 16) -> dict[str, bytes]:
    """Slice a synthetic TIFF into the labels ``plan_ifd_probe_spans`` emits."""
    return {
        "tiff-header": blob[:16],
        f"ifd-{ifd_offset}": blob[ifd_offset:],
    }


def _complex_int16_bytes(samples: np.ndarray) -> bytes:
    """Pack a complex64 array as interleaved little-endian int16 pairs."""
    real = np.asarray(samples.real, dtype=np.int16)
    imag = np.asarray(samples.imag, dtype=np.int16)
    interleaved = np.empty((*samples.shape, 2), dtype="<i2")
    interleaved[..., 0] = real
    interleaved[..., 1] = imag
    return interleaved.tobytes()


class TestStripTableDecode:
    """C1: probe bodies parse into literal strip tables; compressed/tiled fail."""

    def test_uncompressed_strip_ifd_yields_literal_tables(self) -> None:
        """A classic uncompressed strip IFD returns the chosen offsets and counts."""
        blob = _classic_le_strip_tiff()
        bodies = _probe_bodies_from_tiff(blob)
        tables = acquisition_mod._strip_tables_from_probes(
            bodies, samples=_DECODE_SAMPLES
        )
        offsets, counts, rows_per_strip = tables
        assert tuple(offsets) == _DECODE_STRIP_OFFSETS
        assert tuple(counts) == _DECODE_STRIP_BYTE_COUNTS
        assert rows_per_strip == _DECODE_ROWS_PER_STRIP

    def test_compressed_ifd_raises_not_zeros(self) -> None:
        """Compression != 1 is a product error, not an empty canvas."""
        blob = _classic_le_strip_tiff(compression=8)
        bodies = _probe_bodies_from_tiff(blob)
        with pytest.raises(Sentinel1ProductError, match="compressed"):
            acquisition_mod._strip_tables_from_probes(bodies, samples=_DECODE_SAMPLES)

    def test_tiled_ifd_raises_not_zeros(self) -> None:
        """Tile tags are a product error, not an empty canvas."""
        blob = _classic_le_strip_tiff(tiled=True)
        bodies = _probe_bodies_from_tiff(blob)
        with pytest.raises(Sentinel1ProductError, match="tiled"):
            acquisition_mod._strip_tables_from_probes(bodies, samples=_DECODE_SAMPLES)


class TestAssembleWindowSamples:
    """C1: row-labeled span bodies assemble into a known complex64 rectangle."""

    def test_packed_row_span_round_trips_literal_int16_pairs(self) -> None:
        """``row-{line0}-{line1}`` bytes of interleaved int16 become complex64."""
        expected = np.array(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
            dtype=np.complex64,
        )
        bodies = {"row-10-12": _complex_int16_bytes(expected)}
        assembled = acquisition_mod._assemble_window_samples(
            bodies,
            parent_line0=10,
            n_lines=2,
            col0=0,
            n_cols=2,
            samples_per_burst=2,
        )
        np.testing.assert_array_equal(assembled, expected)

    def test_whole_row_span_slices_requested_columns(self) -> None:
        """A full-row body contributes only the requested column slice."""
        full = np.array(
            [[1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j]],
            dtype=np.complex64,
        )
        bodies = {"row-4-5": _complex_int16_bytes(full)}
        assembled = acquisition_mod._assemble_window_samples(
            bodies,
            parent_line0=4,
            n_lines=1,
            col0=1,
            n_cols=2,
            samples_per_burst=4,
        )
        np.testing.assert_array_equal(assembled, full[:, 1:3])

    def test_whole_row_block_slices_requested_columns(self) -> None:
        """A multi-row whole-row body is sliced to the ROI columns after download."""
        samples_per_burst = 6
        col0, n_cols = 1, 2
        full = np.arange(3 * samples_per_burst, dtype=np.float32).reshape(
            3, samples_per_burst
        ) + 1j * np.arange(3 * samples_per_burst, dtype=np.float32).reshape(
            3, samples_per_burst
        )
        bodies = {"row-10-13": _complex_int16_bytes(full)}
        assembled = acquisition_mod._assemble_window_samples(
            bodies,
            parent_line0=10,
            n_lines=3,
            col0=col0,
            n_cols=n_cols,
            samples_per_burst=samples_per_burst,
        )
        np.testing.assert_array_equal(assembled, full[:, col0 : col0 + n_cols])


class TestFailClosedMeasurementFetch:
    """C2: compressed/tiled probes raise; they are not swallowed into zeros."""

    def test_fetch_measurement_bodies_propagates_compressed_product_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``Sentinel1ProductError`` from a compressed IFD is not caught."""
        blob = _classic_le_strip_tiff(compression=5)
        ifd_offset = 16
        file_size = ifd_offset + acquisition_mod._IFD_PROBE_BYTES

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.label == "tiff-header":
                    bodies[span.label] = blob[:16]
                else:
                    bodies[span.label] = blob[ifd_offset:]
            return bodies

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        with pytest.raises(Sentinel1ProductError, match="compressed"):
            acquisition_mod._fetch_measurement_bodies(
                MEASUREMENT_URL,
                file_size=file_size,
                line0=0,
                line1=2,
                col0=0,
                col1=2,
                samples=_DECODE_SAMPLES,
            )

    def test_fetch_measurement_bodies_follows_tiff_header_ifd_pointer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IFD probes start at the header pointer, not ``file_size - 16 KiB``."""
        ifd_offset = 200_000
        blob = _classic_le_strip_tiff(ifd_offset=ifd_offset)
        file_size = len(blob)
        eof_guess = max(
            file_size - acquisition_mod._IFD_PROBE_BYTES,
            acquisition_mod.HEADER_PROBE_BYTES,
        )
        assert eof_guess != ifd_offset
        requested_ifd: list[int] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.label == "tiff-header":
                    bodies[span.label] = blob[:16]
                elif span.label.startswith("ifd-"):
                    requested_ifd.append(span.start)
                    bodies[span.label] = blob[ifd_offset:]
                else:
                    bodies[span.label] = b"\x00" * (span.end - span.start + 1)
            return bodies

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        acquisition_mod._fetch_measurement_bodies(
            MEASUREMENT_URL,
            file_size=file_size,
            line0=0,
            line1=2,
            col0=0,
            col1=2,
            samples=_DECODE_SAMPLES,
        )
        assert requested_ifd
        assert requested_ifd[0] == ifd_offset

    def test_many_tight_column_spans_are_fetched_as_whole_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A long narrow ROI is fetched as whole-row blocks without a 32-span fuse."""
        n_rows = 200
        samples = 21_118
        row_stride = samples * 4
        offsets = [1000 + index * row_stride for index in range(n_rows)]
        counts = [row_stride] * n_rows
        header = b"II" + struct.pack("<HI", 42, 16)
        captured_row_spans: list[Any] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.label == "tiff-header":
                    bodies[span.label] = header
                elif span.label.startswith("row-"):
                    captured_row_spans.append(span)
                    bodies[span.label] = b"\x00" * (span.end - span.start + 1)
                else:
                    bodies[span.label] = b"\x00"
            return bodies

        def fake_tables(
            _bodies: Any, *, samples: int
        ) -> tuple[list[int], list[int], int]:
            del samples
            return list(offsets), list(counts), 1

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        monkeypatch.setattr(acquisition_mod, "_strip_tables_from_probes", fake_tables)
        acquisition_mod._fetch_measurement_bodies(
            MEASUREMENT_URL,
            file_size=offsets[-1] + row_stride,
            line0=0,
            line1=n_rows,
            col0=100,
            col1=2000,
            samples=samples,
        )
        assert captured_row_spans
        assert len(captured_row_spans) < 32
        for span in captured_row_spans:
            assert (span.end - span.start + 1) % row_stride == 0


class TestCdseCatalogAndODataBind:
    """C3/I2: CDSE auto-registers; non-OData measurement hrefs are rejected."""

    def test_discover_auto_registers_catalog_and_binds_odata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CDSE discovery registers ``sentinel1-cdse`` and binds OData hrefs."""
        registered: list[str] = []

        def fake_ensure(name: str | None = None) -> str:
            del name
            registered.append("sentinel1-cdse")
            return "sentinel1-cdse"

        item = _cdse_item(
            _cdse_asset(
                "annotation",
                ANNOTATION_URL,
                media_type="application/xml",
                size_bytes=845_602,
            ),
            _cdse_asset(
                "measurement",
                MEASUREMENT_URL,
                media_type="image/tiff",
                size_bytes=1_135_276_112,
                properties={"file:size": 1_135_276_112, "swath": "IW1"},
            ),
        )
        monkeypatch.setattr(
            "faninsar.remote.providers.cdse.ensure_cdse_catalog",
            fake_ensure,
        )
        monkeypatch.setattr(remote, "search", lambda *_args, **_kwargs: [item])
        items = acquisition_mod.discover_cdse_slc_items(ROI)
        assert registered == ["sentinel1-cdse"]
        assert items[0]["measurement_url"] == MEASUREMENT_URL
        assert items[0]["measurement_size"] == 1_135_276_112

    def test_bind_rejects_s3_eodata_measurement_href(self) -> None:
        """S3 ``eodata`` measurement URLs are not OData range URLs."""
        item = _cdse_item(
            _cdse_asset("annotation", ANNOTATION_URL, media_type="application/xml"),
            _cdse_asset(
                "measurement",
                "s3://eodata/Sentinel-1/measurement.tiff",
                media_type="image/tiff",
            ),
        )
        with pytest.raises(Sentinel1ProductError, match="OData"):
            acquisition_mod._bind_cdse_item(item)

    def test_bind_accepts_odata_value_hrefs(self) -> None:
        """HTTPS OData ``Nodes(...)/$value`` hrefs bind to measurement URLs."""
        item = _cdse_item(
            _cdse_asset(
                "annotation",
                ANNOTATION_URL,
                media_type="application/xml",
                size_bytes=2048,
            ),
            _cdse_asset(
                "measurement",
                MEASUREMENT_URL,
                media_type="image/tiff",
                size_bytes=16_000,
                properties={"file:size": 16_000, "swath": "IW1"},
            ),
        )
        bindings = acquisition_mod._bind_cdse_item(item)
        assert len(bindings) == 1
        binding = bindings[0]
        assert binding["measurement_url"] == MEASUREMENT_URL
        assert binding["annotation_url"] == ANNOTATION_URL
        assert binding["measurement_size"] == 16_000

    def test_bind_emits_one_row_per_swath_pair(self) -> None:
        """Each measurement/annotation OData pair becomes its own canvas binding."""
        iw2_measurement, iw2_annotation = (
            MEASUREMENT_URL.replace("measurement.tiff", "iw2-vv.tiff"),
            ANNOTATION_URL.replace("annotation.xml", "iw2-vv.xml"),
        )
        item = _cdse_item(
            _cdse_asset(
                "measurement-iw1-vv",
                MEASUREMENT_URL,
                media_type="image/tiff",
                size_bytes=16_000,
                properties={"file:size": 16_000, "swath": "IW1", "polarization": "VV"},
            ),
            _cdse_asset(
                "annotation-iw1-vv",
                ANNOTATION_URL,
                media_type="application/xml",
                size_bytes=2048,
            ),
            _cdse_asset(
                "measurement-iw2-vv",
                iw2_measurement,
                media_type="image/tiff",
                size_bytes=17_000,
                properties={"file:size": 17_000, "swath": "IW2", "polarization": "VV"},
            ),
            _cdse_asset(
                "annotation-iw2-vv",
                iw2_annotation,
                media_type="application/xml",
                size_bytes=2048,
            ),
        )
        bindings = acquisition_mod._bind_cdse_item(item)
        assert {(row["swath"], row["polarization"]) for row in bindings} == {
            ("IW1", "VV"),
            ("IW2", "VV"),
        }
        by_swath = {row["swath"]: row for row in bindings}
        assert by_swath["IW2"]["measurement_url"] == iw2_measurement
        assert by_swath["IW2"]["measurement_size"] == 17_000


class TestOverlapSpliceFromFetchedWindows:
    """C4: overlap roles splice fetched rectangles, never np.zeros marked valid."""

    def test_materialize_splices_window_samples_not_zeros(
        self,
        parent_swath: tuple[Any, MemoryFile, str, np.ndarray],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Covered overlap rectangles keep the fetched burst samples."""
        _swath, _memfile, xml_text, data = parent_swath
        windows = [
            BurstFetchedWindow(
                burst_index=4,
                row0=0,
                col0=0,
                samples=np.ascontiguousarray(data[40:50, :]),
            ),
            BurstFetchedWindow(
                burst_index=5,
                row0=0,
                col0=0,
                samples=np.ascontiguousarray(data[50:60, :]),
            ),
        ]
        overlap = BurstOverlapWindow(
            swath="IW1",
            top_index=4,
            bottom_index=5,
            top_row0=7,
            bottom_row0=0,
            col0=10,
            n_lines=3,
            n_samples=40,
        )

        def fake_discover(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "product_id": "S1A_TEST",
                    "annotation_url": ANNOTATION_URL,
                    "measurement_url": MEASUREMENT_URL,
                    "measurement_size": 16_000,
                    "polarization": "VV",
                    "swath": "IW1",
                }
            ]

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.url == ANNOTATION_URL:
                    bodies[span.label] = xml_text.encode("utf-8")
                    continue
                message = (
                    "overlap rectangle was fully covered; no extra "
                    f"measurement fetch expected, got {span.label}"
                )
                raise AssertionError(message)
            return bodies

        def fake_windows(*_args: Any, **_kwargs: Any) -> Any:
            return _swath, windows, xml_text

        def fake_overlaps(
            *_args: Any, **_kwargs: Any
        ) -> tuple[BurstOverlapWindow, ...]:
            return (overlap,)

        monkeypatch.setattr(acquisition_mod, "discover_cdse_slc_items", fake_discover)
        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        monkeypatch.setattr(acquisition_mod, "windows_from_cdse_fetch", fake_windows)
        monkeypatch.setattr(
            acquisition_mod, "overlap_windows_for_canvas", fake_overlaps
        )

        paths = acquisition_mod.materialize_cdse_roi_safes(
            ROI,
            work_dir=tmp_path,
            granularity="roi",
            coreg_mode="nesd",
        )
        measurement = next(paths[0].joinpath("measurement").glob("*.tiff"))
        with rasterio.open(measurement) as dataset:
            raster = dataset.read(1)

        top = raster[47:50, 10:50]
        bottom = raster[50:53, 10:50]
        np.testing.assert_array_equal(top, data[47:50, 10:50])
        np.testing.assert_array_equal(bottom, data[50:53, 10:50])
        assert np.any(top)
        assert np.any(bottom)

    def test_geometry_materialize_skips_overlap_splice(
        self,
        parent_swath: tuple[Any, MemoryFile, str, np.ndarray],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default geometry canvases list intersecting bursts and skip splice."""
        _swath, _memfile, xml_text, data = parent_swath
        windows = [
            BurstFetchedWindow(
                burst_index=4,
                row0=0,
                col0=0,
                samples=np.ascontiguousarray(data[40:50, :]),
            ),
            BurstFetchedWindow(
                burst_index=5,
                row0=0,
                col0=0,
                samples=np.ascontiguousarray(data[50:60, :]),
            ),
        ]

        def fake_discover(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "product_id": "S1A_TEST",
                    "annotation_url": ANNOTATION_URL,
                    "measurement_url": MEASUREMENT_URL,
                    "measurement_size": 16_000,
                    "polarization": "VV",
                    "swath": "IW1",
                }
            ]

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.url == ANNOTATION_URL:
                    bodies[span.label] = xml_text.encode("utf-8")
                    continue
                message = f"geometry must not fetch overlap spans, got {span.label}"
                raise AssertionError(message)
            return bodies

        def fake_windows(*_args: Any, **_kwargs: Any) -> Any:
            return _swath, windows, xml_text

        def boom(*_args: Any, **_kwargs: Any) -> None:
            message = "geometry must not splice overlap"
            raise AssertionError(message)

        monkeypatch.setattr(acquisition_mod, "discover_cdse_slc_items", fake_discover)
        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        monkeypatch.setattr(acquisition_mod, "windows_from_cdse_fetch", fake_windows)
        monkeypatch.setattr(acquisition_mod, "overlap_windows_for_canvas", boom)
        monkeypatch.setattr(acquisition_mod, "splice_overlap_into_canvas", boom)

        paths = acquisition_mod.materialize_cdse_roi_safes(
            ROI,
            work_dir=tmp_path,
            granularity="roi",
        )
        measurement = next(paths[0].joinpath("measurement").glob("*.tiff"))
        with rasterio.open(measurement) as dataset:
            assert dataset.height == 2 * LINES_PER_BURST
            raster = dataset.read(1)
        np.testing.assert_array_equal(raster[0:10], data[40:50])
        np.testing.assert_array_equal(raster[10:20], data[50:60])


def _sparse_burst_geolocation_xml() -> str:
    """Return a one-line-per-burst geolocation grid like a CDSE IW annotation."""
    lines = (0, 1493, 2986)
    pixels = (0, 10_000, 20_000)
    body = "".join(
        "<geolocationGridPoint>"
        f"<line>{line}</line><pixel>{pixel}</pixel>"
        f"<latitude>{40.0 + 0.001 * line:.6f}</latitude>"
        f"<longitude>{100.0 + 0.00005 * pixel:.6f}</longitude>"
        "</geolocationGridPoint>"
        for line in lines
        for pixel in pixels
    )
    return f"<product><geolocationGrid>{body}</geolocationGrid></product>"


def _dense_geolocation_xml() -> str:
    """Return a ~2 km geolocation grid around lon 100, lat 40."""
    points = [
        (0, 0, 39.99, 99.99),
        (0, 50, 39.99, 100.00),
        (0, 99, 39.99, 100.01),
        (10, 0, 40.01, 99.99),
        (10, 50, 40.01, 100.00),
        (10, 99, 40.01, 100.01),
    ]
    body = "".join(
        "<geolocationGridPoint>"
        f"<line>{line}</line><pixel>{pixel}</pixel>"
        f"<latitude>{lat}</latitude><longitude>{lon}</longitude>"
        "</geolocationGridPoint>"
        for line, pixel, lat, lon in points
    )
    return f"<product><geolocationGrid>{body}</geolocationGrid></product>"


class TestRoiWgs84AndPointEnvelope:
    """ROI must be WGS84; a single point is buffered before inversion."""

    def test_missing_crs_is_rejected(self) -> None:
        """A BoundingBox without CRS is not treated as lon/lat."""
        with pytest.raises(ValueError, match="CRS"):
            acquisition_mod.normalize_roi_wgs84(BoundingBox(100.0, 40.0, 100.4, 40.4))

    def test_utm_bbox_inverts_like_wgs84_bbox(self) -> None:
        """A projected bbox is transformed to EPSG:4326 before inversion."""
        xml_text = _parent_xml()
        wgs84 = BoundingBox(100.0, 40.0, 100.4, 40.4, crs="EPSG:4326")
        utm = BoundingBox(
            585360.4618427712,
            4428236.06463309,
            618805.2791234051,
            4473095.679410577,
            crs="EPSG:32647",
        )
        assert acquisition_mod.invert_geolocation_grid(
            xml_text, utm
        ) == acquisition_mod.invert_geolocation_grid(xml_text, wgs84)

    def test_single_point_envelope_is_two_kilometres_across(self) -> None:
        """A GPS point expands to a ~1 km ground square, not a degenerate box."""
        from pyproj import Geod

        point = Points([[100.0, 40.0]], crs="EPSG:4326")
        west, south, east, north = acquisition_mod.inversion_envelope(point)
        assert east > west
        assert north > south
        geod = Geod(ellps="WGS84")
        _az12, _az21, width_m = geod.inv(west, 40.0, east, 40.0)
        _az12, _az21, height_m = geod.inv(100.0, south, 100.0, north)
        assert 1900 < abs(width_m) < 2100
        assert 1900 < abs(height_m) < 2100

    def test_single_point_inversion_covers_more_than_one_pixel(self) -> None:
        """Buffered point inversion is not a 1x1 radar window."""
        point = Points([[100.0, 40.0]], crs="EPSG:4326")
        line0, line1, col0, col1 = acquisition_mod.invert_geolocation_grid(
            _dense_geolocation_xml(),
            point,
        )
        assert line1 - line0 > 1
        assert col1 - col0 > 1

    def test_sparse_burst_grid_interior_roi_covers_more_than_one_line(self) -> None:
        """A one-line-per-burst grid does not collapse a small interior ROI to 1 row."""
        roi = BoundingBox(100.4, 40.4, 100.6, 40.6, crs="EPSG:4326")
        line0, line1, col0, col1 = acquisition_mod.invert_geolocation_grid(
            _sparse_burst_geolocation_xml(),
            roi,
        )
        assert line1 - line0 > 1
        assert col1 - col0 > 1
        assert line1 > 1

    def test_normalize_keeps_points_type(self) -> None:
        """Catalog search still sees a point, not a bounding box."""
        point = Points([[585360.4618427712, 4428236.06463309]], crs="EPSG:32647")
        normalized = acquisition_mod.normalize_roi_wgs84(point)
        assert isinstance(normalized, Points)
        assert normalized.crs is not None
        assert normalized.crs.to_epsg() == 4326
        lon, lat = float(normalized.values[0, 0]), float(normalized.values[0, 1])
        assert abs(lon - 100.0) < 1e-5
        assert abs(lat - 40.0) < 1e-5


class TestNarrowRoiRule:
    """Probe-based narrow-ROI rule for exact-column per-row span planning."""

    def test_rule_requires_context(self) -> None:
        """Without a fetch context the narrow rule never engages."""
        assert (
            acquisition_mod._use_exact_columns(None, n_cols=10, samples=1000) is False
        )

    def test_rule_requires_measured_bandwidth(self) -> None:
        """A context with no probe bytes yet leaves whole-row planning on."""
        context = acquisition_mod._RoiFetchContext()
        assert context.bandwidth_bps is None
        assert (
            acquisition_mod._use_exact_columns(context, n_cols=10, samples=1000)
            is False
        )

    def test_rule_engages_below_both_thresholds(self) -> None:
        """Slow link plus <30% width enables exact-column planning."""
        context = acquisition_mod._RoiFetchContext()
        context.probed_bytes = 1000
        context.probed_seconds = 1.0  # 1 KB/s << 1.3 MiB/s
        assert (
            acquisition_mod._use_exact_columns(context, n_cols=299, samples=1000)
            is True
        )

    def test_rule_stays_off_at_30_percent_width(self) -> None:
        """An ROI exactly 30% of the row width keeps whole-row spans."""
        context = acquisition_mod._RoiFetchContext()
        context.probed_bytes = 1000
        context.probed_seconds = 1.0
        assert (
            acquisition_mod._use_exact_columns(context, n_cols=300, samples=1000)
            is False
        )

    def test_rule_stays_off_above_bandwidth_ceiling(self) -> None:
        """A fast link keeps whole-row spans even for a narrow ROI."""
        context = acquisition_mod._RoiFetchContext()
        context.probed_bytes = 10 * 1024 * 1024
        context.probed_seconds = 1.0  # 10 MiB/s > 1.3 MiB/s
        assert (
            acquisition_mod._use_exact_columns(context, n_cols=10, samples=1000)
            is False
        )


class TestRoiFetchContextDedup:
    """Shared context dedups probes and identical byte ranges."""

    def test_deduped_fetch_collapses_identical_ranges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two spans on one byte range produce a single request."""
        requested: list[tuple[int, int]] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                requested.append((span.start, span.end))
                bodies[span.label] = b"\x01" * (span.end - span.start + 1)
            return bodies

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        context = acquisition_mod._RoiFetchContext()
        spans = (
            remote.fetch.RangeSpan(MEASUREMENT_URL, 100, 199, "row-0-1"),
            remote.fetch.RangeSpan(MEASUREMENT_URL, 100, 199, "alias"),
            remote.fetch.RangeSpan(MEASUREMENT_URL, 200, 299, "row-1-2"),
        )
        bodies = acquisition_mod._fetch_deduped(spans, context)
        assert requested == [(100, 199), (200, 299)]
        assert bodies["row-0-1"] == bodies["alias"] == b"\x01" * 100
        assert bodies["row-1-2"] == b"\x01" * 100

    def test_deduped_fetch_reuses_cached_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A range already in the context is never requested again."""
        requested: list[tuple[int, int]] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            requested.extend((span.start, span.end) for span in spans)
            return {span.label: b"\x02" * (span.end - span.start + 1) for span in spans}

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        context = acquisition_mod._RoiFetchContext()
        first = (remote.fetch.RangeSpan(MEASUREMENT_URL, 0, 99, "row-0-1"),)
        second = (remote.fetch.RangeSpan(MEASUREMENT_URL, 0, 99, "overlap-0-1"),)
        acquisition_mod._fetch_deduped(first, context)
        bodies = acquisition_mod._fetch_deduped(second, context)
        assert requested == [(0, 99)]
        assert bodies["overlap-0-1"] == b"\x02" * 100

    def test_deduped_fetch_without_context_requests_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No context means no dedup: every span is dispatched as before."""
        requested: list[tuple[int, int]] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            requested.extend((span.start, span.end) for span in spans)
            return {span.label: b"\x00" for span in spans}

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        spans = (
            remote.fetch.RangeSpan(MEASUREMENT_URL, 0, 99, "a"),
            remote.fetch.RangeSpan(MEASUREMENT_URL, 0, 99, "b"),
        )
        acquisition_mod._fetch_deduped(spans, None)
        assert requested == [(0, 99), (0, 99)]

    def test_measurement_bodies_reuse_strip_tables_and_spans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shared context probes once and never re-fetches identical spans."""
        n_rows = 4
        samples = 100
        row_stride = samples * 4
        offsets = [1000 + index * row_stride for index in range(n_rows)]
        counts = [row_stride] * n_rows
        header = b"II" + struct.pack("<HI", 42, 16)
        probe_requests: list[str] = []
        row_requests: list[tuple[int, int]] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.label == "tiff-header":
                    probe_requests.append(span.label)
                    bodies[span.label] = header
                elif span.label.startswith("row-"):
                    row_requests.append((span.start, span.end))
                    bodies[span.label] = b"\x00" * (span.end - span.start + 1)
                else:
                    probe_requests.append(span.label)
                    bodies[span.label] = b"\x00" * (span.end - span.start + 1)
            return bodies

        def fake_tables(
            _bodies: Any, *, samples: int
        ) -> tuple[list[int], list[int], int]:
            del samples
            return list(offsets), list(counts), 1

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        monkeypatch.setattr(acquisition_mod, "_strip_tables_from_probes", fake_tables)
        context = acquisition_mod._RoiFetchContext()
        file_size = offsets[-1] + row_stride
        first = acquisition_mod._fetch_measurement_bodies(
            MEASUREMENT_URL,
            file_size=file_size,
            line0=0,
            line1=2,
            col0=0,
            col1=samples,
            samples=samples,
            fetch_context=context,
        )
        # Second call plans the identical byte range (same rows/columns), as
        # happens when an ROI span and an overlap span cover the same rows.
        second = acquisition_mod._fetch_measurement_bodies(
            MEASUREMENT_URL,
            file_size=file_size,
            line0=0,
            line1=2,
            col0=0,
            col1=samples,
            samples=samples,
            fetch_context=context,
        )
        # Header + IFD probed exactly once; the identical row range was not
        # requested again.
        assert probe_requests == ["tiff-header", "ifd-16"]
        assert row_requests == [(1000, 1000 + 2 * row_stride - 1)]
        assert second["row-0-2"] == first["row-0-2"]

    def test_exact_column_spans_decode_through_assembly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The narrow-ROI path fetches col spans and assembles them correctly."""
        n_rows = 4
        samples = 1000
        col0, col1 = 100, 250  # 15% of the row -> narrow
        row_stride = samples * 4
        offsets = [500 + index * row_stride for index in range(n_rows)]
        counts = [row_stride] * n_rows
        header = b"II" + struct.pack("<HI", 42, 16)
        expected = (
            np.arange(n_rows * (col1 - col0), dtype=np.float32).reshape(
                n_rows, col1 - col0
            )
            + 1j
        )
        captured: list[Any] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.label == "tiff-header":
                    bodies[span.label] = header
                elif span.label.startswith("col-"):
                    captured.append(span)
                    row = int(span.label.split("-")[1])
                    bodies[span.label] = _complex_int16_bytes(expected[row])
                else:
                    bodies[span.label] = b"\x00" * (span.end - span.start + 1)
            return bodies

        def fake_tables(
            _bodies: Any, *, samples: int
        ) -> tuple[list[int], list[int], int]:
            del samples
            return list(offsets), list(counts), 1

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        monkeypatch.setattr(acquisition_mod, "_strip_tables_from_probes", fake_tables)
        context = acquisition_mod._RoiFetchContext()
        # Seed a slow observed bandwidth so the narrow rule engages.
        context.probed_bytes = 1000
        context.probed_seconds = 1.0
        bodies = acquisition_mod._fetch_measurement_bodies(
            MEASUREMENT_URL,
            file_size=offsets[-1] + row_stride,
            line0=0,
            line1=n_rows,
            col0=col0,
            col1=col1,
            samples=samples,
            fetch_context=context,
        )
        assert len(captured) == n_rows
        for line, span in enumerate(captured):
            row_start = offsets[line]
            assert (span.start, span.end) == (
                row_start + col0 * 4,
                row_start + col1 * 4 - 1,
            )
            assert span.label == f"col-{line}-{col0}-{col1}"
        assembled = acquisition_mod._assemble_window_samples(
            bodies,
            parent_line0=0,
            n_lines=n_rows,
            col0=col0,
            n_cols=col1 - col0,
            samples_per_burst=samples,
        )
        np.testing.assert_array_equal(assembled, expected)

    def test_column_span_mismatch_fails_closed(self) -> None:
        """A col- span whose columns differ from the request is rejected."""
        payload = _complex_int16_bytes(np.zeros(10, dtype=np.complex64))
        with pytest.raises(Sentinel1ProductError):
            acquisition_mod._assemble_window_samples(
                {"col-0-0-10": payload},
                parent_line0=0,
                n_lines=1,
                col0=5,
                n_cols=5,
                samples_per_burst=100,
            )


class TestBurstSubSpans:
    """Large CDSE full-burst selections split into parallel 4 MiB sub-spans."""

    def test_full_burst_selection_splits_into_capped_sub_spans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A >4 MiB burst emits multiple row-aligned spans in one batch."""
        samples = 262144  # 1 MiB rows
        n_rows = 6  # 6 MiB > 4 MiB cap
        row_stride = samples * 4
        offsets = [index * row_stride for index in range(n_rows)]
        counts = [row_stride] * n_rows
        header = b"II" + struct.pack("<HI", 42, 16)
        captured: list[Any] = []

        async def fake_fetch(spans: Any, *, fetcher: Any = None) -> dict[str, bytes]:
            del fetcher
            bodies: dict[str, bytes] = {}
            for span in spans:
                if span.label == "tiff-header":
                    bodies[span.label] = header
                else:
                    captured.append(span)
                    bodies[span.label] = b"\x00" * (span.end - span.start + 1)
            return bodies

        def fake_tables(
            _bodies: Any, *, samples: int
        ) -> tuple[list[int], list[int], int]:
            del samples
            return list(offsets), list(counts), 1

        monkeypatch.setattr(acquisition_mod, "fetch_ranges_async", fake_fetch)
        monkeypatch.setattr(acquisition_mod, "_strip_tables_from_probes", fake_tables)
        acquisition_mod._fetch_measurement_bodies(
            MEASUREMENT_URL,
            file_size=offsets[-1] + row_stride,
            line0=0,
            line1=n_rows,
            col0=0,
            col1=samples,
            samples=samples,
        )
        row_spans = [span for span in captured if span.label.startswith("row-")]
        assert len(row_spans) == 2  # 4 MiB + 2 MiB
        for span in row_spans:
            assert span.end - span.start + 1 <= acquisition_mod.MAX_SPAN_BYTES
            assert (span.end + 1) % row_stride == 0
        # Spans stay byte-adjacent so the parallel batch reassembles cleanly.
        assert row_spans[1].start == row_spans[0].end + 1

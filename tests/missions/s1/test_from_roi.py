"""Tests for S1Stack.from_roi (PROPOSAL-0061).

The constructor is the single top-level acquisition entry: granularity is the
scientific axis, provider is derived, and CDSE ROI materializes a full-burst-
height canvas before ``from_safes``.  ``from_search`` is deleted.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from faninsar.data.query import BoundingBox
from faninsar.missions.s1 import acquisition as acquisition_mod
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.io.canvas import BurstFetchedWindow, RoiSafeCanvas
from faninsar.missions.s1.overlap import BurstOverlapWindow
from faninsar.missions.s1.stack import S1Stack

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
            f"<longitude>{100.0 + k + 0.4 * pixel:.2f}</longitude>"
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
    assert resolver is not None, "resolve_roi_acquisition is required by from_roi"
    return resolver(**kwargs)


def _from_roi(*args: Any, **kwargs: Any) -> Any:
    """Call S1Stack.from_roi; fail as an assertion if it is missing."""
    constructor = getattr(S1Stack, "from_roi", None)
    assert constructor is not None, "S1Stack.from_roi is required by PROPOSAL-0061"
    return constructor(*args, **kwargs)


class TestResolutionMatrix:
    """Every legal combo infers the right provider; illegal combos fail fast."""

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
        tmp_path: Path,
    ) -> None:
        """Illegal combinations raise ValueError that cites the matrix table."""
        with pytest.raises(ValueError, match="table") as resolved:
            _resolve_roi_acquisition(granularity=granularity, provider=provider)
        message = str(resolved.value)
        assert granularity in message
        assert provider in message

        with pytest.raises(ValueError, match="table") as constructed:
            _from_roi(
                ROI,
                work_dir=tmp_path,
                granularity=granularity,
                provider=provider,
            )
        message = str(constructed.value)
        assert granularity in message
        assert provider in message


def test_from_search_is_gone() -> None:
    """``from_search`` is deleted with no deprecation shim."""
    assert not hasattr(S1Stack, "from_search")
    with pytest.raises(AttributeError):
        S1Stack.from_search  # noqa: B018
    with pytest.raises(AttributeError, match="from_search"):
        S1Stack.from_search(object(), work_dir="unused")  # type: ignore[misc]


def test_from_roi_docstring_documents_matrix_without_fixed_fraction() -> None:
    """The constructor docstring carries the table and no fixed byte contract."""
    constructor = getattr(S1Stack, "from_roi", None)
    assert constructor is not None, "S1Stack.from_roi is required by PROPOSAL-0061"
    doc = inspect.getdoc(constructor)
    assert doc is not None
    lowered = doc.lower()
    for token in (
        "granularity",
        "providers",
        "inferred provider",
        "roi",
        "burst",
        "scene",
        "cdse",
        "asf",
        "table",
    ):
        assert token in lowered
    assert "2.9%" not in doc
    assert "3%" not in doc
    assert "roi shape" in lowered or "shape" in lowered


def test_from_roi_signature_matches_proposal() -> None:
    """The public signature matches the PROPOSAL-0061 constructor contract."""
    constructor = getattr(S1Stack, "from_roi", None)
    assert constructor is not None, "S1Stack.from_roi is required by PROPOSAL-0061"
    parameters = inspect.signature(constructor).parameters
    assert list(parameters)[:2] == ["roi", "datetime_range"]
    assert parameters["datetime_range"].default is None
    assert parameters["datetime_range"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in (
        "granularity",
        "provider",
        "polarizations",
        "work_dir",
        "orbit_paths",
        "calibration_paths",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["granularity"].default == "roi"
    assert parameters["provider"].default is None
    assert parameters["polarizations"].default == ("VV",)


class TestCdseRoiCanvasOrder:
    """CDSE roi path: canvas then overlap splice then from_safes then prepare."""

    def test_write_canvas_then_from_safes_sees_full_burst_height(
        self,
        parent_swath: tuple[S1Swath, MemoryFile, str, np.ndarray],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mocked catalog + range fetch still writes a full-burst-height canvas."""
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

        def fake_discover(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            del args, kwargs
            order.append("discover")
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
                assert dataset.height == 2 * LINES_PER_BURST
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
                    assert dataset.height == 2 * LINES_PER_BURST
                    assert dataset.width == SAMPLES_PER_BURST
            return FakeStack()

        monkeypatch.setattr(acquisition_mod, "discover_cdse_slc_items", fake_discover)
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
                    top_row0=7,
                    bottom_row0=0,
                    col0=10,
                    n_lines=3,
                    n_samples=40,
                ),
            )

        monkeypatch.setattr(acquisition_mod, "windows_from_cdse_fetch", fake_windows)
        monkeypatch.setattr(
            acquisition_mod,
            "overlap_windows_for_canvas",
            fake_overlaps,
        )
        monkeypatch.setattr(S1Stack, "from_safes", classmethod(fake_from_safes))

        result = _from_roi(ROI, work_dir=tmp_path, granularity="roi")
        assert isinstance(result, FakeStack)
        assert prepared == [True]
        assert captured_paths
        assert all(Path(path).name.endswith(".SAFE") for path in captured_paths[0])
        write_at = order.index("write_canvas")
        splice_at = order.index("splice_overlap")
        safes_at = order.index("from_safes")
        prepare_at = order.index("prepare")
        assert write_at < splice_at < safes_at < prepare_at
        assert "discover" in order
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

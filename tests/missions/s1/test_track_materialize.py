"""Stage-3 provider-decoupled track materialization tests.

The CDSE burst path resolves ``absoluteBurstID`` selections through the
sensing-time axis and range-fetches whole bursts into a path-SAFE under
``tracks/<trackKey>/acquisitions/<parentBox>``.  The ASF path reassembles downloaded
one-burst SAFE zips into the identical structure, so ``provider`` never
leaks above the materialize seam.
"""

from __future__ import annotations

import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio

from faninsar.data.query import BoundingBox
from faninsar.missions.s1.acquisition import (
    _resolve_burst_indices,
    materialize_asf_burst_safes,
    materialize_cdse_burst_safes,
    materialize_s1_acquisitions,
)
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.missions.s1.safe import open_safe_product

if TYPE_CHECKING:
    from pathlib import Path

N_BURSTS = 4
LINES_PER_BURST = 8
SAMPLES_PER_BURST = 64


def _parent_xml() -> str:
    """Annotation for a 4-burst synthetic IW1 parent, with geolocation."""
    burst_els: list[str] = []
    grid_els: list[str] = []
    first = " ".join(["0"] * LINES_PER_BURST)
    last = " ".join([str(SAMPLES_PER_BURST - 1)] * LINES_PER_BURST)
    for k in range(N_BURSTS):
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
        for local in (0, LINES_PER_BURST - 1):
            grid_els.append(
                "        <geolocationGridPoint>"
                f"<line>{k * LINES_PER_BURST + local}</line><pixel>0</pixel>"
                f"<latitude>{40.0 + k + 0.01 * local:.4f}</latitude>"
                f"<longitude>{100.0 + 0.01 * local:.4f}</longitude>"
                "</geolocationGridPoint>"
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
    <stopTime>2016-12-07T11:18:58.571896</stopTime>
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


def _swath():
    """Parse the synthetic parent swath."""
    return parse_annotation_xml(
        _parent_xml(),
        annotation_path="memory://parent-annotation.xml",
        measurement_path="memory://parent-measurement.tiff",
    )


def _burst_raster(index: int) -> np.ndarray:
    """Deterministic complex64 block for one burst (line=i*100+row)."""
    rows = np.arange(LINES_PER_BURST, dtype=np.float32)[:, None]
    cols = np.arange(SAMPLES_PER_BURST, dtype=np.float32)[None, :]
    return (index * 1000 + rows + 1j * cols).astype(np.complex64)


class TestResolveBurstIndices:
    """absoluteBurstID selection resolves through the sensing-time axis."""

    def test_all_bursts_when_no_ids(self) -> None:
        swath = _swath()
        assert _resolve_burst_indices(swath, absolute_burst_ids=None) == (0, 1, 2, 3)

    def test_sensing_time_hint_picks_nearest_burst(self) -> None:
        swath = _swath()
        # sensingTime of burst 2 is 11:18:56.543845; hint inside half of
        # the synthetic burst window (lpb*ATI = 16.4 ms).
        hint = datetime(2016, 12, 7, 11, 18, 56, 543845, tzinfo=UTC)
        resolved = _resolve_burst_indices(
            swath,
            absolute_burst_ids=[999],
            catalog_sensing_times=[hint],
        )
        assert resolved == (2,)

    def test_unmatchable_hint_rejects(self) -> None:
        swath = _swath()
        with pytest.raises(Sentinel1ProductError, match="no parent burst matches"):
            _resolve_burst_indices(
                swath,
                absolute_burst_ids=[1],
                catalog_sensing_times=[datetime(2016, 12, 7, 0, 0, 0, tzinfo=UTC)],
            )

    def test_catalog_ids_associate_by_sensing_time_not_list_order(self) -> None:
        """Official IDs bind to the matching burst, never zip(ids, indices)."""
        from faninsar.missions.s1.acquisition import _associate_catalog_absolute_ids

        swath = _swath()
        mapping = _associate_catalog_absolute_ids(
            swath,
            absolute_burst_ids=[41, 42],
            catalog_sensing_times=[
                datetime(2016, 12, 7, 11, 18, 56, 543845, tzinfo=UTC),
                datetime(2016, 12, 7, 11, 18, 54, 543845, tzinfo=UTC),
            ],
        )
        assert mapping == {2: 41, 0: 42}

    def test_catalog_ids_without_sensing_times_are_not_assigned_by_order(self) -> None:
        """Missing sensing hints leave official IDs unset rather than list-order."""
        from faninsar.missions.s1.acquisition import _associate_catalog_absolute_ids

        swath = _swath()
        assert (
            _associate_catalog_absolute_ids(
                swath,
                absolute_burst_ids=[41, 42],
                catalog_sensing_times=None,
            )
            == {}
        )


class _FakeCatalogAsset:
    key = "product"
    href = "https://example.test/product.zip"
    media_type = "application/zip"
    representation = "complete"


class _FakeAcquisition:
    def __init__(self, props: dict, start: datetime) -> None:
        self.properties = props
        self.start = start
        self.platform = "Sentinel-1A"
        self.beam_mode = "IW1"
        self.polarization = "VV"


class _FakeCatalogItem:
    """Minimal CatalogItem stand-in for the ASF burst path."""

    def __init__(
        self,
        item_id: str,
        *,
        absolute_burst_id: int,
        relative_orbit: int = 88,
        direction: str = "ASCENDING",
        sensing: datetime,
    ) -> None:
        self.item_id = item_id
        self.provider = "asf"
        self.catalog = "sentinel1-asf"
        self.raw_metadata: dict = {}
        self.acquisition = _FakeAcquisition(
            {
                "absoluteBurstID": absolute_burst_id,
                "relativeOrbit": relative_orbit,
                "orbitDirection": direction,
                "mission": "S1A",
                "sensingStart": sensing.isoformat(),
            },
            sensing,
        )
        self.assets = {"product": _FakeCatalogAsset()}


def _one_burst_xml(sensing_second: int) -> str:
    """One-burst annotation derived from the parent XML's first burst."""
    first = " ".join(["0"] * LINES_PER_BURST)
    last = " ".join([str(SAMPLES_PER_BURST - 1)] * LINES_PER_BURST)
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
      <numberOfSamples>{SAMPLES_PER_BURST}</numberOfSamples>
      <numberOfLines>{LINES_PER_BURST}</numberOfLines>
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
    <burstList count="1">
      <burst>
        <azimuthTime>2016-12-07T11:18:{sensing_second:02d}.426276</azimuthTime>
        <azimuthAnxTime>{607.4725}</azimuthAnxTime>
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
      <geolocationGridPoint><line>{LINES_PER_BURST - 1}</line><pixel>0</pixel><latitude>40.1</latitude><longitude>100.0</longitude></geolocationGridPoint>
    </geolocationGridPointList>
  </geolocationGrid>
</product>"""


def _write_burst_zip(dest: Path, burst_index: int, sensing_second: int) -> None:
    """Write a minimal one-burst SAFE zip (annotation + measurement)."""
    stem = f"s1a-iw1-slc-vv-20161207t1118{sensing_second:02d}"
    data = _burst_raster(burst_index)
    tiff_bytes = _tiff_bytes(data)
    manifest = '<?xml version="1.0"?><xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"/>'
    with zipfile.ZipFile(dest, "w") as archive:
        archive.writestr("X.SAFE/manifest.safe", manifest)
        archive.writestr(
            f"X.SAFE/annotation/{stem}.xml", _one_burst_xml(sensing_second)
        )
        archive.writestr(f"X.SAFE/measurement/{stem}.tiff", tiff_bytes)


def _tiff_bytes(data: np.ndarray) -> bytes:
    """Return a small GTiff payload for ``data``."""
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="complex64",
        crs=None,
        transform=Affine.identity(),
    ) as dst:
        dst.write(data, 1)
    payload = memfile.read()
    memfile.close()
    return payload


class TestPathSafeOfficialIdStamp:
    """Catalog IDs are written into the materialized annotation copy only."""

    def test_write_path_safe_stamps_catalog_id_when_annotation_lacks_it(
        self, tmp_path: Path
    ) -> None:
        from faninsar.missions.s1.io.path_safe import PathBurstPayload, write_path_safe

        swath = _swath()
        result = write_path_safe(
            swath,
            [
                PathBurstPayload(
                    burst_index=2,
                    samples=_burst_raster(2),
                    absolute_burst_id=4242,
                )
            ],
            "88A",
            tmp_path,
            annotation_xml=_parent_xml(),
        )
        product = open_safe_product(result.safe_root)
        assert product.swaths[0].bursts[0].absolute_burst_id == 4242

    def test_write_path_safe_rejects_conflicting_annotation_id(
        self, tmp_path: Path
    ) -> None:
        from faninsar.missions.s1.io.path_safe import PathBurstPayload, write_path_safe

        xml_text = _parent_xml().replace(
            "<sensingTime>2016-12-07T11:18:54.543845</sensingTime>",
            "<sensingTime>2016-12-07T11:18:54.543845</sensingTime>\n"
            "        <absoluteBurstID>111</absoluteBurstID>",
            1,
        )
        swath = parse_annotation_xml(
            xml_text,
            annotation_path="memory://parent-annotation.xml",
            measurement_path="memory://parent-measurement.tiff",
        )
        with pytest.raises(Sentinel1ProductError, match="conflicting"):
            write_path_safe(
                swath,
                [
                    PathBurstPayload(
                        burst_index=0,
                        samples=_burst_raster(0),
                        absolute_burst_id=222,
                    )
                ],
                "88A",
                tmp_path,
                annotation_xml=xml_text,
            )


class TestAsfBurstPath:
    """ASF burst zips reassemble into a track path-SAFE."""

    def test_two_burst_zips_merge_into_one_path_safe(self, tmp_path: Path) -> None:
        items = [
            _FakeCatalogItem(
                "S1A_BURST_1",
                absolute_burst_id=3001,
                sensing=datetime(2016, 12, 7, 11, 18, 53, tzinfo=UTC),
            ),
            _FakeCatalogItem(
                "S1A_BURST_2",
                absolute_burst_id=3002,
                sensing=datetime(2016, 12, 7, 11, 18, 54, tzinfo=UTC),
            ),
        ]

        def fake_download(asset, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            idx = 0 if "BURST_1" in str(destination) else 1
            _write_burst_zip(destination, idx, 53 + idx)
            return destination

        paths = materialize_asf_burst_safes(
            items,
            work_dir=tmp_path,
            download=fake_download,
        )
        assert len(paths) == 1
        safe_root = paths[0]
        # Staged under tracks/88A/acquisitions/<date>/<parentBox>/.
        assert (
            safe_root.parent.parent.parent
            == tmp_path / "tracks" / "88A" / "acquisitions"
        )
        product = open_safe_product(safe_root)
        assert len(product.swaths) == 1
        assert len(product.swaths[0].bursts) == 2
        assert (
            len({s.annotation_path for s in product.swaths[0].burst_source_swaths}) == 2
        )
        burst_ids = sorted(
            burst.absolute_burst_id
            for swath in product.swaths
            for burst in swath.bursts
        )
        assert burst_ids == [3001, 3002]


class TestCdseBurstPath:
    """CDSE range fetch emits path-SAFEs under tracks/<key>/acquisitions."""

    def test_burst_path_writes_path_safe(self, tmp_path: Path, monkeypatch) -> None:
        import faninsar.missions.s1.acquisition as acq

        xml_text = _parent_xml()

        monkeypatch.setattr(
            acq, "_fetch_annotation_xml", lambda item, fetch_context=None: xml_text
        )

        def fake_bodies(
            url,
            *,
            file_size,
            line0,
            line1,
            col0,
            col1,
            samples,
            fetch_context=None,
            max_span_bytes=None,
        ):
            assert line0 == 2 * LINES_PER_BURST
            assert line1 == 3 * LINES_PER_BURST
            data = _burst_raster(2)
            pairs = np.stack(
                [data.real.astype("<i2"), data.imag.astype("<i2")], axis=-1
            )
            return {f"row-{line0}-{line1}": pairs.tobytes()}

        monkeypatch.setattr(acq, "_fetch_measurement_bodies", fake_bodies)
        # ROI covering burst 2's latitude only.
        roi = BoundingBox(99.9, 41.9, 100.1, 42.1, crs="EPSG:4326")
        items = [
            {
                "product_id": "S1A_IW_SLC__1SDV_TEST",
                "annotation_url": "cdse://annotation.xml",
                "measurement_url": "cdse://measurement.tiff",
                "measurement_size": 0,
                "annotation_size": 0,
                "polarization": "VV",
                "swath": "IW1",
                "mission_id": "A",
                "relative_orbit": 88,
                "orbit_direction": "ASCENDING",
            }
        ]
        paths = materialize_cdse_burst_safes(
            roi,
            items=items,
            work_dir=tmp_path,
            absolute_burst_ids=[42],
            catalog_sensing_times=[
                datetime(2016, 12, 7, 11, 18, 56, 543845, tzinfo=UTC)
            ],
        )
        assert len(paths) == 1
        safe_root = paths[0]
        assert (
            safe_root.parent.parent.parent
            == tmp_path / "tracks" / "88A" / "acquisitions"
        )
        product = open_safe_product(safe_root)
        assert len(product.swaths) == 1
        swath = product.swaths[0]
        assert len(swath.bursts) == 1
        with rasterio.open(swath.measurement_path) as ds:
            assert ds.height == LINES_PER_BURST
            assert ds.width == SAMPLES_PER_BURST

    def test_multi_track_items_split_into_own_track_dirs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Each item lands under its own track's acquisitions directory.

        Regression: a multi-track ``orbit_group=None`` materialize used to
        resolve ``out_dir``/``track_key`` once from ``items[0]`` and stack
        every product into that single track directory.
        """
        import faninsar.missions.s1.acquisition as acq

        xml_text = _parent_xml()
        monkeypatch.setattr(
            acq, "_fetch_annotation_xml", lambda item, fetch_context=None: xml_text
        )

        def fake_bodies(
            url,
            *,
            file_size,
            line0,
            line1,
            col0,
            col1,
            samples,
            fetch_context=None,
            max_span_bytes=None,
        ):
            data = _burst_raster(2)
            pairs = np.stack(
                [data.real.astype("<i2"), data.imag.astype("<i2")], axis=-1
            )
            return {f"row-{line0}-{line1}": pairs.tobytes()}

        monkeypatch.setattr(acq, "_fetch_measurement_bodies", fake_bodies)
        roi = BoundingBox(99.9, 41.9, 100.1, 42.1, crs="EPSG:4326")

        def _item(track: tuple[str, int, str]) -> dict:
            mission, rel_orbit, direction = track
            return {
                "product_id": f"S1{mission}_IW_SLC__1SDV_TEST_{rel_orbit}",
                "annotation_url": "cdse://annotation.xml",
                "measurement_url": "cdse://measurement.tiff",
                "measurement_size": 0,
                "annotation_size": 0,
                "polarization": "VV",
                "swath": "IW1",
                "mission_id": mission,
                "relative_orbit": rel_orbit,
                "orbit_direction": direction,
            }

        items = [
            _item(("A", 88, "ASCENDING")),
            _item(("D", 117, "ASCENDING")),
        ]
        paths = materialize_cdse_burst_safes(
            roi,
            items=items,
            work_dir=tmp_path,
            absolute_burst_ids=[42],
            catalog_sensing_times=[
                datetime(2016, 12, 7, 11, 18, 56, 543845, tzinfo=UTC)
            ],
        )
        parents = {p.parent.parent.parent for p in paths}
        assert parents == {
            tmp_path / "tracks" / "88A" / "acquisitions",
            tmp_path / "tracks" / "117A" / "acquisitions",
        }

    def test_dispatch_defaults_to_cdse(self, tmp_path: Path, monkeypatch) -> None:
        import faninsar.missions.s1.acquisition as acq

        called = {}
        monkeypatch.setattr(
            acq,
            "materialize_cdse_burst_safes",
            lambda *a, **k: called.setdefault("cdse", True) or [],
        )
        roi = BoundingBox(99.9, 41.9, 100.1, 42.1, crs="EPSG:4326")
        materialize_s1_acquisitions(roi, work_dir=tmp_path)
        assert called.get("cdse") is True

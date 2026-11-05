"""Focused tests for the Sentinel-1 Stack search seam."""

from __future__ import annotations

import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest

from faninsar import remote
from faninsar.data.query import BoundingBox
from faninsar.missions.s1.s1search import S1Search
from faninsar.missions.s1.search import (
    catalog_for_product,
    complete_safe_asset,
    filter_burst_polarizations,
    normalize_polarizations,
    product_filters,
    provider_polarization_filter,
    select_orbit_bindings,
)
from faninsar.missions.s1.stack import S1Stack
from faninsar.stack.layout import staged_acquisition_path

if TYPE_CHECKING:
    from pathlib import Path

ROI = BoundingBox(100.0, 40.0, 100.4, 40.4, crs="EPSG:4326")

# Minimal one-burst geometry used by the from_search ASF-path fakes: the
# downloaded zip is a real one-burst SAFE the extractor can open and
# reassemble into a path-SAFE.
_LINES_PER_BURST = 8
_SAMPLES_PER_BURST = 16


def _asf_search(items: list[remote.CatalogItem], spatial: Any = ROI) -> S1Search:
    """Wrap ASF catalog items in an ``S1Search`` for ``from_search`` tests."""
    return S1Search(
        items=tuple(items),
        catalog="sentinel1-asf",
        spatial=spatial,
        query={},
    )


def _item(
    *assets: remote.RemoteAsset,
    raw_metadata: dict[str, object] | None = None,
    properties: dict[str, object] | None = None,
) -> remote.CatalogItem:
    """Build one normalized catalog item for seam tests."""
    acquisition = remote.AcquisitionMetadata(
        acquisition_id="S1_TEST",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
        properties=properties or {},
    )
    return remote.CatalogItem(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="S1-SLC",
        item_id="S1_TEST",
        acquisition=acquisition,
        assets={asset.key: asset for asset in assets},
        raw_metadata=raw_metadata or {},
    )


def _asset(
    key: str = "data", href: str = "https://datapool.asf.alaska.edu/S1.zip"
) -> remote.RemoteAsset:
    """Build one complete remote SAFE asset."""
    return remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="S1-SLC",
        item_id="S1_TEST",
        key=key,
        href=href,
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


def _write_burst_zip(dest: Path, sensing_second: int = 53) -> None:
    """Write a minimal one-burst SAFE zip readable by ``open_safe_product``."""
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    stem = f"s1a-iw1-slc-vv-20161207t1118{sensing_second:02d}"
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
    with zipfile.ZipFile(dest, "w") as archive:
        archive.writestr("X.SAFE/manifest.safe", manifest)
        archive.writestr(f"X.SAFE/annotation/{stem}.xml", _burst_xml(sensing_second))
        archive.writestr(f"X.SAFE/measurement/{stem}.tiff", tiff_bytes)


def _fake_burst_download(asset: Any, destination: Path, **kwargs: Any) -> Path:
    """Download hook that stages a real one-burst SAFE zip."""
    del asset, kwargs
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_burst_zip(destination)
    return destination


def test_s1_polarization_defaults_and_dual_modes() -> None:
    """Single, dual, and all-channel requests map to local channels."""
    assert normalize_polarizations(None) == ("VV",)
    assert normalize_polarizations("VV+VH") == ("VV", "VH")
    assert normalize_polarizations(("HH", "HV")) == ("HH", "HV")
    assert normalize_polarizations("all") == "all"
    assert provider_polarization_filter(("VV", "VH")) is None
    assert provider_polarization_filter(("VV", "VH"), allow_dual=True) == "VV+VH"
    assert provider_polarization_filter("all") is None


def test_s1_product_type_maps_to_asf_dataset_and_level() -> None:
    """Burst is the default and scene selects full-scene SLC."""
    assert product_filters("burst", ("VV",)) == {
        "dataset": "SLC-BURST",
        "processingLevel": "BURST",
        "polarization": "VV",
    }
    assert "polarization" not in product_filters("burst", ("VV", "VH"))
    assert product_filters("scene", ("VV", "VH"))["polarization"] == "VV+VH"
    assert product_filters("scene", "all") == {
        "dataset": "SENTINEL-1",
        "processingLevel": "SLC",
    }


def test_complete_safe_asset_rejects_bare_burst_tiff() -> None:
    """A burst TIFF cannot cross the SAFE reader boundary as a complete file."""
    tiff = _asset(href="https://sentinel1.asf.alaska.edu/burst.tiff")
    with pytest.raises(ValueError, match="SAFE ZIP"):
        complete_safe_asset(_item(tiff))


def test_product_families_share_stable_asf_registration() -> None:
    """Dataset filters distinguish burst and scene under one S1 catalog."""
    assert catalog_for_product("burst") == "sentinel1-asf"
    assert catalog_for_product("scene") == "sentinel1-asf"


def test_asf_scene_registration_uses_canonical_collection_concept() -> None:
    """The scientific scene name maps to ASF's Sentinel-1 concept ID."""
    from faninsar.missions.s1.search import ensure_asf_catalog
    from faninsar.remote.catalog import _ADAPTERS

    name = ensure_asf_catalog(product_type="scene")
    assert _ADAPTERS[name].query_collection == "C1214470488-ASF"


def test_asf_burst_search_queries_burst_collection_not_s1a_slc() -> None:
    """Burst discovery must not pin CMR to the S1A SLC concept ID."""
    from faninsar.missions.s1.search import ensure_asf_catalog
    from faninsar.remote.catalog import _ADAPTERS
    from faninsar.remote.providers.asf_search import _query_parameters

    ensure_asf_catalog(product_type="burst")
    adapter = _ADAPTERS[catalog_for_product("burst")]
    parameters = _query_parameters(
        adapter.query_collection,
        spatial=None,
        datetime_range=None,
        filters=product_filters("burst", ("VV",)),
    )
    assert parameters["collections"] == "C2709161906-ASF"
    assert parameters["dataset"] == "SLC-BURST"


def test_asf_registration_rejects_different_redirect_policy() -> None:
    """A stable catalog name cannot hide a different redirect policy."""
    from faninsar.missions.s1.search import ensure_asf_catalog
    from faninsar.remote.catalog import _ADAPTERS

    name = ensure_asf_catalog(product_type="burst")
    adapter = _ADAPTERS[name]
    original = adapter.redirect_origins
    adapter.redirect_origins = (*original, "https://unexpected.example")
    try:
        with pytest.raises(ValueError, match="incompatible ASF registration"):
            ensure_asf_catalog(product_type="scene")
    finally:
        adapter.redirect_origins = original


def test_orbit_selection_prefers_covering_poeorb(tmp_path: Path) -> None:
    """Official S1 priority selects POEORB before a covering RESORB."""
    body = """<Earth_Explorer_File><Data_Block><List_of_OSVs>
    <OSV><UTC>UTC=2023-12-31T23:00:00</UTC><X>1</X><Y>2</Y><Z>3</Z>
    <VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>
    <OSV><UTC>UTC=2024-01-01T01:00:00</UTC><X>1</X><Y>2</Y><Z>3</Z>
    <VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>
    </List_of_OSVs></Data_Block></Earth_Explorer_File>"""
    precise = tmp_path / "S1A_OPER_AUX_POEORB_TEST.EOF"
    restituted = tmp_path / "S1A_OPER_AUX_RESORB_TEST.EOF"
    precise.write_text(body)
    restituted.write_text(body)
    selected = select_orbit_bindings(
        [restituted, precise],
        {
            "20240101": (
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            )
        },
    )
    assert selected == {"20240101": precise}


def test_ipf_002_36_is_detected_from_safe_manifest() -> None:
    """ISCE2's early-product condition is read from manifest.safe."""
    from faninsar.missions.s1.safe import parse_ipf_version

    manifest = (
        '<safe:processing xmlns:safe="http://www.esa.int/safe/sentinel-1.0">'
        '<safe:facility><safe:software name="Sentinel-1 IPF" version="002.36"/>'
        "</safe:facility></safe:processing>"
    )
    assert parse_ipf_version(manifest) == "002.36"
    assert parse_ipf_version("<not-xml") is None


def test_pre_240_annotation_fm_rate_coefficients_parse() -> None:
    """IPF < 2.40 stores azimuth FM-rate coefficients as c0, c1, ... elements."""
    from pathlib import Path

    from faninsar.missions.s1.annotation import parse_annotation_xml

    fixture = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "sentinel1"
        / "minimal_annotation_iw1.xml"
    )
    xml_text = fixture.read_text(encoding="utf-8")
    xml_text = xml_text.replace(
        "<azimuthFmRatePolynomial>-2330.570588164277 451066.6805858936"
        " -79121617.77766916</azimuthFmRatePolynomial>",
        "<c0>-2.329526977539063e+03</c0>\n"
        "        <c1>4.496041718043527e+05</c1>\n"
        "        <c2>-7.886473300662930e+07</c2>",
    )
    swath = parse_annotation_xml(
        xml_text,
        annotation_path=str(fixture),
        measurement_path="memory://synthetic.tiff",
    )
    assert swath.azimuth_fm_rate[0][2] == (
        -2.329526977539063e03,
        4.496041718043527e05,
        -7.886473300662930e07,
    )


def test_burst_multi_polarization_filters_catalog_metadata() -> None:
    """Multi-channel burst searches retain only requested channels."""
    dual = _item(_asset(), raw_metadata={"Polarization": "VV+VH"})
    single = _item(_asset(), raw_metadata={"Polarization": "VV"})
    hh = _item(_asset(), raw_metadata={"Polarization": "HH"})
    assert filter_burst_polarizations([dual, single, hh], ("VV", "VH")) == [
        dual,
        single,
    ]
    with pytest.raises(ValueError, match="unambiguous polarization"):
        filter_burst_polarizations([_item(_asset())], ("VV", "VH"))


def test_orbit_selection_prefers_poeorb_then_resorb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Valid covering POEORB wins deterministically over RESORB."""
    poe = tmp_path / "S1_POEORB.EOF"
    res = tmp_path / "S1_RESORB.EOF"
    poe.touch()
    res.touch()
    vectors = [
        type("Vector", (), {"time": datetime(2024, 1, 1, tzinfo=UTC)})(),
        type("Vector", (), {"time": datetime(2024, 1, 2, tzinfo=UTC)})(),
    ]
    monkeypatch.setattr(
        "faninsar.missions.s1.orbit.read_eof_orbit",
        lambda _path: type("Orbit", (), {"vectors": vectors})(),
    )
    interval = {"20240101": (vectors[0].time, vectors[1].time)}
    assert select_orbit_bindings([res, poe], interval) == {"20240101": poe}
    assert select_orbit_bindings([res], interval) == {"20240101": res}


def test_orbit_selection_uses_embedded_vectors_without_candidates() -> None:
    """No explicit candidates preserve the SAFE embedded orbit vectors."""
    assert (
        select_orbit_bindings(
            None,
            {
                "20240101": (
                    datetime.min.replace(tzinfo=UTC),
                    datetime.max.replace(tzinfo=UTC),
                )
            },
        )
        == {}
    )


def test_orbit_selection_rejects_configured_gap(tmp_path: Path) -> None:
    """Configured orbit lookup failure is visible instead of silently embedded."""
    candidate = tmp_path / "S1_RESORB.EOF"
    candidate.touch()
    with pytest.raises(ValueError, match="do not cover"):
        select_orbit_bindings(
            [candidate],
            {
                "20240101": (
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                )
            },
        )


def test_aux_cal_bindings_require_existing_local_files(tmp_path: Path) -> None:
    """Explicit AUX_CAL bindings are validated and selected per date."""
    from faninsar.missions.s1.aux_cal import select_aux_cal_bindings

    aux = tmp_path / "S1A_AUX_CAL_V20140908T000000_G20240327T101157.SAFE.zip"
    aux.touch()
    sensing = datetime(2015, 1, 1, tzinfo=UTC)
    assert select_aux_cal_bindings([aux], {"20150101": ("S1A", sensing)}) == {
        "20150101": aux
    }
    with pytest.raises(FileNotFoundError):
        select_aux_cal_bindings(
            [tmp_path / "missing.zip"], {"20150101": ("S1A", sensing)}
        )


def test_from_search_downloads_one_selected_safe_and_prepares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The constructor downloads selected complete assets before preparation."""
    item = _item(_asset())
    downloaded: list[Path] = []
    prepared: list[bool] = []

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        downloaded.append(destination)
        return _fake_burst_download(asset, destination, **kwargs)

    monkeypatch.setattr(remote, "download", download)

    class FakeStack:
        def prepare(self) -> None:
            prepared.append(True)

    captured: dict[str, object] = {}

    def from_safes(_cls: object, _paths: object, **kwargs: object) -> FakeStack:
        captured.update(kwargs)
        return FakeStack()

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    result = S1Stack.from_search(_asf_search([item]), work_dir=tmp_path, provider="asf")
    assert isinstance(result, FakeStack)
    assert downloaded == [staged_acquisition_path(tmp_path / "inputs", "S1_TEST.zip")]
    assert prepared == [True]
    assert captured.get("roi") == ROI


def test_from_search_downloads_safes_in_selection_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ASF burst items stage their zips then reassemble into path-SAFEs."""
    slow = replace(_item(_asset("slow")), item_id="A_SLOW")
    fast = replace(_item(_asset("fast")), item_id="B_FAST")
    downloaded: list[str] = []
    captured: list[tuple[Path, ...]] = []

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        downloaded.append(destination.name)
        return _fake_burst_download(asset, destination, **kwargs)

    monkeypatch.setattr(remote, "download", download)

    class FakeStack:
        def prepare(self) -> None:
            return None

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> FakeStack:
        del cls, kwargs
        captured.append(tuple(paths))
        return FakeStack()

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    S1Stack.from_search(_asf_search([slow, fast]), work_dir=tmp_path, provider="asf")

    # Both burst zips are staged, then reassembled into one path-SAFE that
    # from_safes opens (not the raw staged zips).
    assert downloaded == ["A_SLOW.zip", "B_FAST.zip"]
    assert len(captured) == 1
    assert len(captured[0]) == 1
    safe_root = captured[0][0]
    assert safe_root.suffix == ".SAFE"
    assert safe_root.is_dir()


def test_from_search_downloads_every_selected_burst_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each ASF burst item is staged and reassembled into the path-SAFE."""
    items = [
        _item(replace(_asset(f"burst-{index}"), item_id=f"S1_{index}"))
        for index in range(3)
    ]
    downloaded: list[str] = []

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        downloaded.append(asset.item_id)
        return _fake_burst_download(asset, destination, **kwargs)

    monkeypatch.setattr(remote, "download", download)

    class FakeStack:
        def prepare(self) -> None:
            return None

    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(lambda _cls, _paths, **_kwargs: FakeStack()),
    )
    S1Stack.from_search(_asf_search(items), work_dir=tmp_path, provider="asf")

    assert downloaded == ["S1_0", "S1_1", "S1_2"]


def test_from_search_does_not_construct_stack_after_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One failed SAFE cancels the construction boundary for the whole search."""
    first = replace(_item(_asset("first")), item_id="S1_FIRST")
    second = replace(_item(_asset("second")), item_id="S1_SECOND")
    stack_constructed = False

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del kwargs
        if asset.key == "second":
            message = "burst generation failed"
            raise RuntimeError(message)
        return destination

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> NoReturn:
        del cls, paths, kwargs
        nonlocal stack_constructed
        stack_constructed = True
        message = "from_safes must not run after download failure"
        raise AssertionError(message)

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with pytest.raises(RuntimeError, match="burst generation failed"):
        S1Stack.from_search(
            _asf_search([first, second]), work_dir=tmp_path, provider="asf"
        )
    assert not stack_constructed


def test_from_search_rejects_tiff_xml_without_burst_extractor_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TIFF/XML components never cross the complete SAFE input boundary."""
    measurement = remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="SLC-BURST",
        item_id="S1_TEST",
        key="measurement",
        href="https://sentinel1.asf.alaska.edu/burst.tiff",
        media_type="image/tiff",
        representation="component",
    )
    annotation = remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="SLC-BURST",
        item_id="S1_TEST",
        key="annotation",
        href="https://sentinel1.asf.alaska.edu/burst.xml",
        media_type="application/xml",
        representation="component",
    )
    item = _item(measurement, annotation)

    downloads: list[object] = []
    monkeypatch.setattr(
        remote,
        "download",
        lambda *args, **kwargs: downloads.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="exactly one complete SAFE ZIP"):
        S1Stack.from_search(_asf_search([item]), work_dir=tmp_path, provider="asf")

    assert downloads == []


def test_from_search_rejects_ambiguous_safe_zips_before_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Extra TIFF/XML assets cannot hide two competing complete SAFE ZIPs."""
    item = _item(
        _asset("safe-a", "https://sentinel1.asf.alaska.edu/a.zip"),
        _asset("safe-b", "https://sentinel1.asf.alaska.edu/b.zip"),
        _asset("measurement", "https://sentinel1.asf.alaska.edu/burst.tiff"),
        _asset("annotation", "https://sentinel1.asf.alaska.edu/burst.xml"),
    )
    downloads: list[object] = []
    monkeypatch.setattr(
        remote,
        "download",
        lambda *args, **kwargs: downloads.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="found 2"):
        S1Stack.from_search(_asf_search([item]), work_dir=tmp_path, provider="asf")

    assert downloads == []


def test_from_search_asf_items_rediscover_on_cdse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """provider="cdse" re-discovers ASF-selected frames inside CDSE.

    ASF items carry no CDSE OData assets, so ``from_search`` re-discovers
    them on CDSE narrowed to the ASF sensing window, keeps only the
    searched tracks, and forwards the burst selector so the same bursts
    are picked inside the CDSE parent frames.
    """
    science = {
        "absoluteBurstID": 3001,
        "relativeOrbit": 88,
        "orbitDirection": "ASCENDING",
        "mission": "S1A",
        "sensingStart": "2024-01-01T00:00:10",
    }
    item = _item(_asset(), properties=science)
    item = replace(
        item,
        acquisition=replace(
            item.acquisition,
            start=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        ),
    )
    discover_calls: dict[str, Any] = {}
    # Two CDSE bindings on different tracks; only the searched one (88A)
    # must be materialized.
    cdse_88 = {
        "product_id": "P88",
        "relative_orbit": 88,
        "orbit_direction": "ASCENDING",
        "mission_id": "S1A",
        "polarization": "VV",
        "swath": "IW1",
    }
    cdse_117 = {
        "product_id": "P117",
        "relative_orbit": 117,
        "orbit_direction": "ASCENDING",
        "mission_id": "S1A",
        "polarization": "VV",
        "swath": "IW1",
    }

    def fake_discover(roi, datetime_range=None, polarizations=None, **kw):
        discover_calls["datetime_range"] = datetime_range
        return [cdse_88, cdse_117]

    captured: dict[str, Any] = {}

    def fake_materialize(roi, **kwargs):
        captured.update(kwargs)
        return [tmp_path / "tracks" / "88A" / "acquisitions" / "X.SAFE"]

    monkeypatch.setattr(
        "faninsar.missions.s1.stack.discover_cdse_slc_items", fake_discover
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.stack.materialize_s1_acquisitions",
        fake_materialize,
    )
    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(
            lambda _cls, _paths, **_kw: type("S", (), {"prepare": lambda self: None})()
        ),
    )
    S1Stack.from_search(
        _asf_search([item]),
        work_dir=tmp_path,
        provider="cdse",
        extra={"orbit_auto_fetch": False},
    )

    # Re-discovery is narrowed to the ASF item's sensing window.
    assert discover_calls["datetime_range"] == (
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
    )
    # Only the searched-track binding is materialized; burst selector
    # forwards the ASF absoluteBurstID/sensingStart.
    assert captured["items"] == [cdse_88]
    assert captured["absolute_burst_ids"] == [3001]
    assert [str(t) for t in captured["catalog_sensing_times"]] == [
        "2024-01-01 00:00:10+00:00"
    ]
    assert captured["provider"] == "cdse"


# -- single search implementation (PROPOSAL-0072 §2) -------------------------


def _search_item(
    item_id: str,
    day: int,
    *,
    href: str | None = None,
    polarization: str = "VV",
) -> remote.CatalogItem:
    """Build one ASF burst record with a distinct start and polarization."""
    base = _item(
        _asset(href=href or f"https://datapool.asf.alaska.edu/{item_id}.zip"),
        raw_metadata={"Polarization": polarization},
    )
    start = datetime(2024, 1, day, tzinfo=UTC)
    return replace(
        base,
        item_id=item_id,
        acquisition=replace(base.acquisition, start=start, end=start),
    )


def _fake_remote_search(
    monkeypatch: pytest.MonkeyPatch, items: list[remote.CatalogItem]
) -> list[dict[str, Any]]:
    """Replace the provider with one returning ``items`` in the given order."""
    calls: list[dict[str, Any]] = []

    def search(roi: Any, **kwargs: Any) -> list[remote.CatalogItem]:
        del roi
        calls.append(kwargs)
        return list(items)

    monkeypatch.setattr(remote, "search", search)
    return calls


def test_stack_search_and_module_search_are_one_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both public entry points return equal ``S1Search`` values."""
    from faninsar.missions.s1 import search_acquisitions

    items = [_search_item("B", 2), _search_item("A", 1)]
    calls = _fake_remote_search(monkeypatch, items)
    kwargs: dict[str, Any] = {
        "datetime_range": ("2024-01-01", "2024-02-01"),
        "track": 88,
        "flight_direction": "ASCENDING",
    }
    via_class = S1Stack.search(ROI, **kwargs)
    via_module = search_acquisitions(ROI, **kwargs)

    assert S1Stack.search is search_acquisitions
    assert isinstance(via_class, S1Search)
    assert via_class == via_module
    assert calls[0] == calls[1]
    assert calls[0]["filters"] == {
        "dataset": "SLC-BURST",
        "processingLevel": "BURST",
        "relativeOrbit": 88,
        "polarization": "VV",
    }
    assert calls[0]["flight_direction"] == "ASCENDING"


def test_search_orders_canonically_regardless_of_provider_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start, item ID, then asset hrefs decide order; ``limit`` truncates last."""
    late = _search_item("A_LATE", 3)
    tie_b = _search_item("B_TIE", 2)
    tie_a2 = _search_item("A_TIE", 2, href="https://datapool.asf.alaska.edu/z.zip")
    tie_a1 = _search_item("A_TIE", 2, href="https://datapool.asf.alaska.edu/a.zip")
    early = _search_item("Z_EARLY", 1)
    expected = [early, tie_a1, tie_a2, tie_b, late]

    for provider_order in (
        [late, tie_b, tie_a2, tie_a1, early],
        [tie_a1, early, late, tie_a2, tie_b],
    ):
        _fake_remote_search(monkeypatch, provider_order)
        assert list(S1Stack.search(ROI)) == expected

    _fake_remote_search(monkeypatch, [late, tie_b, tie_a2, tie_a1, early])
    assert list(S1Stack.search(ROI, limit=2)) == expected[:2]


def test_search_defaults_to_vv_and_named_values_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile defaults, then aliases, then named values (incl. the VV default)."""
    calls = _fake_remote_search(monkeypatch, [])
    result = S1Stack.search(
        ROI,
        track=88,
        relativeOrbit=117,
        platform="Sentinel-1A",
        polarization="VH",
        beamMode="IW",
    )
    assert result.query["polarizations"] == ("VV",)
    assert calls[0]["filters"] == {
        "dataset": "SLC-BURST",
        "processingLevel": "BURST",
        "relativeOrbit": 88,
        "platform": "Sentinel-1A",
        "polarization": "VV",
        "beamMode": "IW",
    }


def test_search_uses_one_direction_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``flightDirection`` alias moves to the dedicated slot; named wins."""
    calls = _fake_remote_search(monkeypatch, [])
    S1Stack.search(ROI, flightDirection="DESCENDING")
    S1Stack.search(ROI, flight_direction="ASCENDING", flightDirection="DESCENDING")
    assert [call["flight_direction"] for call in calls] == ["DESCENDING", "ASCENDING"]
    assert all("flightDirection" not in call["filters"] for call in calls)


def test_search_all_returns_every_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """``'all'`` sends no polarization filter and keeps every channel."""
    items = [
        _search_item("VV", 1, polarization="VV"),
        _search_item("VH", 2, polarization="VH"),
        _search_item("HH", 3, polarization="HH"),
    ]
    calls = _fake_remote_search(monkeypatch, items)
    result = S1Stack.search(ROI, polarizations="all", polarization="VV")
    assert "polarization" not in calls[0]["filters"]
    assert [item.item_id for item in result] == ["VV", "VH", "HH"]


def test_search_multi_channel_filters_client_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dual request drops a single-channel alias and filters metadata."""
    items = [
        _search_item("VV", 1, polarization="VV"),
        _search_item("VH", 2, polarization="VH"),
        _search_item("HH", 3, polarization="HH"),
    ]
    calls = _fake_remote_search(monkeypatch, items)
    result = S1Stack.search(ROI, polarizations="VV+VH", polarization="VV")
    assert "polarization" not in calls[0]["filters"]
    assert [item.item_id for item in result] == ["VV", "VH"]


def test_search_rejects_none_polarizations(monkeypatch: pytest.MonkeyPatch) -> None:
    """All channels must be requested explicitly with ``'all'``."""
    calls = _fake_remote_search(monkeypatch, [])
    with pytest.raises(ValueError, match="polarizations='all'"):
        S1Stack.search(ROI, polarizations=None)  # type: ignore[arg-type]
    assert calls == []


def test_from_search_accepts_stack_search_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``S1Stack.from_search(S1Stack.search(...))`` composes directly."""
    _fake_remote_search(monkeypatch, [_item(_asset())])
    monkeypatch.setattr(remote, "download", _fake_burst_download)

    class FakeStack:
        def prepare(self) -> None:
            return None

    monkeypatch.setattr(
        S1Stack, "from_safes", classmethod(lambda _cls, _paths, **_kw: FakeStack())
    )
    result = S1Stack.from_search(S1Stack.search(ROI), work_dir=tmp_path, provider="asf")
    assert isinstance(result, FakeStack)


@pytest.mark.parametrize("limit", [0, -1])
@pytest.mark.parametrize("explicit_budget", [False, True])
def test_search_invalid_limit_has_consistent_query_error(
    limit: int, explicit_budget: bool
) -> None:
    """A derived resource budget must not replace public query validation."""
    from faninsar.remote.errors import RemoteQueryError

    budget = remote.RemoteResourceBudget() if explicit_budget else None
    with pytest.raises(RemoteQueryError, match="invalid_limit"):
        S1Stack.search(ROI, limit=limit, budget=budget)

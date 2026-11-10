"""Parent-frame burst number persistence (PROPOSAL-0086)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from faninsar.missions.s1 import acquisition as s1_acquisition
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.missions.s1.io.canvas import (
    BurstFetchedWindow,
    _RoiCanvasWriter,
    write_roi_safe_canvas,
)
from faninsar.missions.s1.io.export import export_swath_bursts
from faninsar.missions.s1.io.path_safe import PathBurstPayload, write_path_safe
from faninsar.missions.s1.io.read import read_full_burst
from faninsar.missions.s1.roi_transfer import _coverage_bursts
from faninsar.missions.s1.safe import open_safe_acquisition, open_safe_product
from faninsar.missions.s1.types import S1Burst, S1Swath
from tests.missions.s1.test_track_materialize import (
    _FakeCatalogItem,
    _write_burst_zip,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sentinel1"
    / "minimal_annotation_iw1.xml"
)
NS = "https://faninsar.org/ns/s1"
N_BURSTS = 10
LINES_PER_BURST = 10
SAMPLES_PER_BURST = 100


def _burst_times(index: int) -> tuple[str, str]:
    """Return valid ISO azimuth/sensing timestamps for parent burst ``index``."""
    origin = datetime(2016, 12, 7, 11, 18, 10, 426276, tzinfo=UTC)
    azimuth = origin + timedelta(seconds=index)
    sensing = origin + timedelta(seconds=index, milliseconds=117)
    return (
        azimuth.replace(tzinfo=None).isoformat(),
        sensing.replace(tzinfo=None).isoformat(),
    )


def _parent_xml(n_bursts: int = N_BURSTS) -> str:
    """Return annotation XML for a synthetic IW1 parent with ``n_bursts`` bursts."""
    burst_els: list[str] = []
    first = " ".join(["0"] * LINES_PER_BURST)
    last = " ".join([str(SAMPLES_PER_BURST - 1)] * LINES_PER_BURST)
    for k in range(n_bursts):
        azimuth, sensing = _burst_times(k)
        burst_els.append(
            f"""
      <burst>
        <azimuthTime>{azimuth}</azimuthTime>
        <azimuthAnxTime>{600.0 + k}</azimuthAnxTime>
        <sensingTime>{sensing}</sensingTime>
        <byteOffset>{k * LINES_PER_BURST * SAMPLES_PER_BURST * 4}</byteOffset>
        <firstValidSample>{first}</firstValidSample>
        <lastValidSample>{last}</lastValidSample>
      </burst>"""
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1A</missionId>
    <productType>SLC</productType>
    <polarisation>VV</polarisation>
    <mode>IW</mode>
    <swath>IW1</swath>
    <startTime>2016-12-07T11:18:50.426275</startTime>
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
      <numberOfLines>{n_bursts * LINES_PER_BURST}</numberOfLines>
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
    <burstList count="{n_bursts}">{"".join(burst_els)}
    </burstList>
  </swathTiming>
</product>"""


def _parse(
    xml_text: str,
    *,
    original_parent: bool = False,
    path: str = "memory://ann.xml",
    measurement_path: str = "memory://meas.tiff",
) -> S1Swath:
    """Parse annotation XML, optionally as a trusted complete parent."""
    kwargs = {
        "annotation_path": path,
        "measurement_path": measurement_path,
    }
    if original_parent:
        kwargs["_original_parent"] = True
    return parse_annotation_xml(xml_text, **kwargs)


def _inject_number(xml_text: str, *, local_index: int, number: str) -> str:
    """Insert a namespaced parent ordinal on the local burst at ``local_index``."""
    stamp = f'<number xmlns="{NS}">{number}</number>'
    parts = xml_text.split("</burst>")
    assert 0 <= local_index < len(parts) - 1
    parts[local_index] = parts[local_index] + stamp
    return "</burst>".join(parts)


def _window(burst_index: int) -> BurstFetchedWindow:
    """Return a 2x4 fetched rectangle inside one parent burst."""
    return BurstFetchedWindow(
        burst_index=burst_index,
        row0=1,
        col0=2,
        samples=np.ones((2, 4), dtype=np.complex64),
    )


def _payload(
    swath: S1Swath,
    burst_index: int,
    *,
    number: int | None = None,
    xml: str | None = None,
) -> PathBurstPayload:
    """Return a full-burst path-SAFE payload of zeros."""
    samples = np.zeros(
        (int(swath.lines_per_burst), int(swath.samples_per_burst)),
        dtype=np.complex64,
    )
    return PathBurstPayload(
        burst_index=burst_index,
        samples=samples,
        number=number,
        annotation_xml=xml,
    )


def test_generic_annotation_xml_leaves_number_none(tmp_path: Path) -> None:
    """Unknown-provenance XML must not invent parent numbers from local order."""
    swath = _parse(FIXTURE.read_text(encoding="utf-8"), path=str(FIXTURE))
    assert [burst.index for burst in swath.bursts] == [0, 1]
    assert [burst.number for burst in swath.bursts] == [None, None]
    xml_text = _parent_xml()
    parent = _parse(xml_text)
    canvas = write_roi_safe_canvas(
        parent,
        [_window(8)],
        tmp_path / "generic",
        annotation_xml=xml_text,
        product_id="GENERIC",
    )
    live = open_safe_product(canvas.safe_root)
    assert live.swaths[0].bursts[0].index == 0
    assert live.swaths[0].bursts[0].number is None


def test_s1burst_docstring_distinguishes_index_and_number() -> None:
    """Public docs must spell the singleton 0/8 example and reject absolute IDs."""
    doc = S1Burst.__doc__ or ""
    assert "index 0" in doc
    assert "number 8" in doc
    assert "absoluteBurstID" in doc or "absolute_burst_id" in doc
    assert "must not be guessed" in doc or "unknown" in doc


def test_namespaced_stamp_singleton_index0_number8() -> None:
    """A stamped singleton of parent burst 8 reopens as local index 0."""
    xml_text = _inject_number(_parent_xml(), local_index=8, number="8")
    parent = _parse(xml_text, original_parent=True)
    kept = parent.bursts[8]
    singleton = _parent_xml(n_bursts=1)
    singleton = singleton.replace(
        parent.bursts[0].sensing_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        kept.sensing_time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        1,
    )
    singleton = _inject_number(singleton, local_index=0, number="8")
    swath = _parse(singleton)
    assert len(swath.bursts) == 1
    assert swath.bursts[0].index == 0
    assert swath.bursts[0].number == 8
    assert swath.bursts[0].absolute_burst_id is None


def test_trusted_parent_enumerates_without_guessing_on_generic_parse() -> None:
    """Only an explicit original-parent parse may assign enumerate ordinals."""
    xml_text = _parent_xml()
    generic = _parse(xml_text)
    assert generic.bursts[8].index == 8
    assert generic.bursts[8].number is None
    trusted = _parse(xml_text, original_parent=True)
    assert [burst.number for burst in trusted.bursts] == list(range(N_BURSTS))
    assert trusted.bursts[8].index == 8
    assert trusted.bursts[8].number == 8


def test_canvas_keep_8_and_2_8_roundtrip(tmp_path: Path) -> None:
    """Canvas keep-8 and keep-[2,8] persist original numbers through reopen."""
    xml_text = _parent_xml()
    parent = _parse(xml_text, original_parent=True, path="memory://parent.xml")
    single = write_roi_safe_canvas(
        parent,
        [_window(8)],
        tmp_path / "one",
        annotation_xml=xml_text,
        product_id="KEEP8",
    )
    one = _parse(
        single.annotation_path.read_text(encoding="utf-8"),
        path=str(single.annotation_path),
        measurement_path=str(single.measurement_path),
    )
    assert [burst.index for burst in one.bursts] == [0]
    assert [burst.number for burst in one.bursts] == [8]
    live = open_safe_product(single.safe_root)
    assert live.swaths[0].bursts[0].index == 0
    assert live.swaths[0].bursts[0].number == 8
    assert NS in single.annotation_path.read_text(encoding="utf-8")
    samples = read_full_burst(one, burst_index=0)
    assert samples.row0 == 0
    assert samples.samples.shape == (LINES_PER_BURST, 4)
    np.testing.assert_array_equal(
        samples.samples[1:3, :],
        np.ones((2, 4), dtype=np.complex64),
    )

    pair = write_roi_safe_canvas(
        parent,
        [_window(2), _window(8)],
        tmp_path / "pair",
        annotation_xml=xml_text,
        product_id="KEEP28",
    )
    two = _parse(
        pair.annotation_path.read_text(encoding="utf-8"),
        path=str(pair.annotation_path),
    )
    assert [burst.index for burst in two.bursts] == [0, 1]
    assert [burst.number for burst in two.bursts] == [2, 8]


def test_repeat_subset_does_not_rebase_number(tmp_path: Path) -> None:
    """A second trim of local index 0 must keep original number 8."""
    xml_text = _parent_xml()
    parent = _parse(xml_text, original_parent=True, path="memory://parent.xml")
    first = write_roi_safe_canvas(
        parent,
        [_window(8)],
        tmp_path / "first",
        annotation_xml=xml_text,
        product_id="FIRST8",
    )
    subset = _parse(
        first.annotation_path.read_text(encoding="utf-8"),
        path=str(first.annotation_path),
    )
    second = write_roi_safe_canvas(
        subset,
        [_window(0)],
        tmp_path / "second",
        annotation_xml=first.annotation_path.read_text(encoding="utf-8"),
        product_id="SECOND8",
    )
    again = _parse(
        second.annotation_path.read_text(encoding="utf-8"),
        path=str(second.annotation_path),
    )
    assert again.bursts[0].index == 0
    assert again.bursts[0].number == 8


def test_path_safe_and_asf_href_8_zip_not_inner_index_0(tmp_path: Path) -> None:
    """ASF ``8.zip`` is the parent ordinal; inner one-burst XML stays index 0."""
    href = (
        "https://example.test/"
        "S1A_IW_SLC__1SSV_20161207T111852_20161207T111919_014273_01716D_67BC.SAFE/"
        "IW1/VV/8.zip"
    )
    assert s1_acquisition._asf_href_burst_number(href) == 8
    assert (
        s1_acquisition._asf_href_burst_number("https://example.test/product.zip")
        is None
    )

    xml_text = _parent_xml()
    parent = _parse(xml_text, original_parent=True, path="memory://parent.xml")
    shared = write_path_safe(
        parent,
        [_payload(parent, 8)],
        "88A",
        tmp_path / "shared",
        annotation_xml=xml_text,
        product_id="S1A_IW_SLC__1SSV_20161207T111858_20161207T111858_014273_01716D_0001",
    )
    shared_live = open_safe_product(shared.safe_root)
    assert shared_live.swaths[0].bursts[0].index == 0
    assert shared_live.swaths[0].bursts[0].number == 8

    item = _FakeCatalogItem(
        "S1A_BURST_8",
        absolute_burst_id=8888,
        sensing=datetime(2016, 12, 7, 11, 18, 10, tzinfo=UTC),
    )
    item.assets = {
        "thumbnail": type(
            "Thumb",
            (),
            {
                "key": "thumbnail",
                "href": "https://example.test/thumb.png",
                "media_type": "image/png",
                "representation": "preview",
            },
        )(),
        "data": type(
            "Data",
            (),
            {
                "key": "data",
                "href": href,
                "media_type": "application/zip",
                "representation": "complete",
            },
        )(),
    }

    def fake_download(_asset: object, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_burst_zip(destination, 0, 10)
        return destination

    paths = s1_acquisition.materialize_asf_burst_safes(
        [item],
        work_dir=tmp_path / "asf",
        download=fake_download,
    )
    asf_live = open_safe_product(paths[0])
    assert asf_live.swaths[0].bursts[0].index == 0
    assert asf_live.swaths[0].bursts[0].number == 8


def test_merge_recomputes_index_preserves_number(tmp_path: Path) -> None:
    """Merged one-burst extracts get local 0,1 while numbers stay 8 then 9."""
    xml_text = _parent_xml()
    parent = _parse(xml_text, original_parent=True, path="memory://parent.xml")
    roots: list[Path] = []
    for burst_index, tag in ((8, "0008"), (9, "0009")):
        result = write_path_safe(
            parent,
            [_payload(parent, burst_index)],
            "88A",
            tmp_path / tag,
            annotation_xml=xml_text,
            product_id=(
                f"S1A_IW_SLC__1SSV_20161207T111850_20161207T111850_014273_01716D_{tag}"
            ),
        )
        roots.append(result.safe_root)
    merged = open_safe_acquisition(roots)
    bursts = merged.swath("IW1").bursts
    assert [burst.index for burst in bursts] == [0, 1]
    assert [burst.number for burst in bursts] == [8, 9]


def test_malformed_or_negative_stamp_rejects(tmp_path: Path) -> None:
    """Corrupt explicit ordinals must not become guessed numbers."""
    xml_text = _inject_number(_parent_xml(n_bursts=1), local_index=0, number="eight")
    with pytest.raises(Sentinel1ProductError, match="number"):
        _parse(xml_text)
    xml_text = _inject_number(_parent_xml(n_bursts=1), local_index=0, number="-1")
    with pytest.raises(Sentinel1ProductError, match="number"):
        _parse(xml_text)
    stamped = _inject_number(_parent_xml(n_bursts=1), local_index=0, number="8")
    parent = _parse(stamped)
    product_id = "S1A_IW_SLC__1SSV_20161207T111850_20161207T111850_014273_01716D_00C1"
    with pytest.raises(Sentinel1ProductError, match="contradict"):
        write_path_safe(
            parent,
            [_payload(parent, 0, number=7, xml=stamped)],
            "88A",
            tmp_path / "conflict",
            annotation_xml=stamped,
            product_id=product_id,
        )
    inner = _parent_xml(n_bursts=1)
    inner_swath = _parse(inner)
    with pytest.raises(Sentinel1ProductError, match="number"):
        write_path_safe(
            inner_swath,
            [_payload(inner_swath, 0, number=1.5, xml=inner)],  # type: ignore[arg-type]
            "88A",
            tmp_path / "float",
            annotation_xml=inner,
            product_id=product_id.replace("00C1", "00F1"),
        )
    parent_xml = _parent_xml()
    parent = _parse(parent_xml, original_parent=True)
    canvas = write_roi_safe_canvas(
        parent,
        [_window(8)],
        tmp_path / "resume",
        annotation_xml=parent_xml,
        product_id="RESUME",
    )
    tree = ET.parse(canvas.annotation_path)
    stamped = list(tree.iter(f"{{{NS}}}number"))
    assert stamped
    stamped[0].text = "7"
    tree.write(canvas.annotation_path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(Sentinel1ProductError, match="contradict"):
        _RoiCanvasWriter(
            parent,
            tmp_path / "resume",
            annotation_xml=parent_xml,
            product_id="RESUME",
            listed_indices=(8,),
            resume=True,
        )


def test_export_trim_preserves_number(tmp_path: Path) -> None:
    """export_swath_bursts keeps stamped numbers on the trimmed annotation."""
    xml_text = _parent_xml()
    parent = _parse(xml_text, original_parent=True, path="memory://parent.xml")
    canvas = write_roi_safe_canvas(
        parent,
        [_window(2), _window(8)],
        tmp_path / "src",
        annotation_xml=xml_text,
        product_id="EXPORTSRC",
    )
    swath = replace(
        _parse(
            canvas.annotation_path.read_text(encoding="utf-8"),
            path=str(canvas.annotation_path),
        ),
        measurement_path=str(canvas.measurement_path),
    )
    exported = export_swath_bursts(
        swath,
        [1],
        tmp_path / "export",
        product_id="EXPORTED",
    )
    out = _parse(
        next((exported / "annotation").glob("*.xml")).read_text(encoding="utf-8")
    )
    assert [burst.index for burst in out.bursts] == [0]
    assert [burst.number for burst in out.bursts] == [8]


def test_coverage_bursts_copies_optional_number() -> None:
    """Committed coverage JSON must copy number when set and omit it when unknown."""
    xml_text = _parent_xml()
    numbered = _coverage_bursts(
        SimpleNamespace(  # type: ignore[arg-type]
            swath=_parse(xml_text, original_parent=True),
            listed=(0, 8),
        )
    )
    assert [(row["index"], row.get("number")) for row in numbered] == [
        (0, 0),
        (8, 8),
    ]
    unknown = _coverage_bursts(
        SimpleNamespace(  # type: ignore[arg-type]
            swath=_parse(xml_text),
            listed=(8,),
        )
    )
    assert unknown[0]["index"] == 8
    assert "number" not in unknown[0]

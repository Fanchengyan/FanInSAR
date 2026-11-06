"""Tests for AUX_CAL parsing, selection, fetch, and the IPF 002.36 EAP correction.

The EAP phase vector is pinned against ISCE2
``Sentinel1.computeElevationAntennaPatternCorrection`` evaluated on the
committed IW1 annotation and a trimmed real S1A AUX_CAL product.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from faninsar.missions.s1.annotation import load_annotation_file
from faninsar.missions.s1.aux_cal import (
    correct_eap_phase,
    eap_phase_vector,
    nominal_roll_deg,
    parse_aux_cal_name,
    read_elevation_antenna_pattern,
    requires_eap_correction,
    select_aux_cal_bindings,
)
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.missions.s1.io.read import BurstArray
from faninsar.missions.s1.safe import parse_ipf_version
from faninsar.missions.s1.types import S1Product

DATA = Path(__file__).resolve().parent / "data"
ANNOTATION = DATA / "iw1-annotation-20161207-f121.xml"
AUX_CAL_NAME = "S1A_AUX_CAL_V20140908T000000_G20240327T101157"
AUX_CAL_ZIP = DATA / f"{AUX_CAL_NAME}.SAFE.zip"
ISCE2_REFERENCE = DATA / "isce2_eap_phase_iw1_vv.npz"


@pytest.fixture(scope="module")
def swath() -> Any:
    """Return the committed IW1 annotation as a parsed swath."""
    return load_annotation_file(ANNOTATION, measurement_path="memory://iw1.tiff")


def _product(swath: Any, ipf_version: str | None) -> S1Product:
    return S1Product(
        product_id="S1A_IW_SLC__1SSV_TEST",
        source_path=Path("S1A_IW_SLC__1SSV_TEST.SAFE"),
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=("VV",),
        swaths=(swath,),
        ipf_version=ipf_version,
    )


def _burst_array(*, col0: int, width: int) -> BurstArray:
    rng = np.random.default_rng(7)
    shape = (8, width)
    samples = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    valid = np.ones(shape, dtype=bool)
    valid[:, :3] = False
    samples[~valid] = 0
    return BurstArray(
        samples=samples, row0=0, col0=col0, burst_index=2, valid_mask=valid
    )


def test_annotation_exposes_ascending_node_and_elevation_records(swath: Any) -> None:
    """Annotation parsing keeps the fields the EAP correction needs."""
    assert swath.ascending_node_time == datetime(
        2016, 12, 7, 11, 8, 45, 953764, tzinfo=UTC
    )
    assert len(swath.elevation_angle_records) == 10
    record = swath.elevation_angle_records[0]
    assert len(record.slant_range_time_s) == len(record.elevation_angle_deg) == 671


def test_aux_cal_name_parsing() -> None:
    """AUX_CAL validity and generation times are parsed from any path part."""
    parsed = parse_aux_cal_name(f"/x/{AUX_CAL_NAME}.SAFE/data/s1a-aux-cal.xml")
    assert parsed is not None
    assert parsed.mission_id == "S1A"
    assert parsed.validity_start == datetime(2014, 9, 8, tzinfo=UTC)
    assert parsed.generation == datetime(2024, 3, 27, 10, 11, 57, tzinfo=UTC)
    assert parse_aux_cal_name("S1A_OPER_AUX_POEORB_OPOD_x.EOF") is None


def test_selection_prefers_latest_validity_then_generation(tmp_path: Path) -> None:
    """The newest calibration valid at sensing time is bound per date."""
    names = (
        "S1A_AUX_CAL_V20140406T133000_G20190626T100036",
        "S1A_AUX_CAL_V20140908T000000_G20190626T100201",
        "S1A_AUX_CAL_V20140908T000000_G20240327T101157",
        "S1A_AUX_CAL_V20150301T000000_G20240327T101157",
        "S1B_AUX_CAL_V20140908T000000_G20250101T000000",
    )
    files = []
    for name in names:
        path = tmp_path / f"{name}.SAFE.zip"
        path.touch()
        files.append(path)
    sensing = datetime(2015, 1, 15, tzinfo=UTC)
    selected = select_aux_cal_bindings(files, {"20150115": ("S1A", sensing)})
    assert selected["20150115"].name == f"{names[2]}.SAFE.zip"
    early = datetime(2014, 5, 1, tzinfo=UTC)
    assert select_aux_cal_bindings(files, {"d": ("S1A", early)})["d"] == files[0]
    assert (
        select_aux_cal_bindings(files, {"d": ("S1A", datetime(2014, 1, 1, tzinfo=UTC))})
        == {}
    )
    with pytest.raises(ValueError, match="covers acquisition"):
        select_aux_cal_bindings(
            files,
            {"d": ("S1A", datetime(2014, 1, 1, tzinfo=UTC))},
            require_coverage=True,
        )


def test_pattern_parse_deinterleaves_iq() -> None:
    """The IW1/VV pattern is read as 601 complex samples."""
    pattern = read_elevation_antenna_pattern(
        AUX_CAL_ZIP, swath="IW1", polarization="VV"
    )
    assert pattern.elevation_angle_increment_deg == pytest.approx(0.05)
    assert pattern.gain.shape == (601,)
    assert np.iscomplexobj(pattern.gain)
    with pytest.raises(Sentinel1ProductError, match="IW3/VV"):
        read_elevation_antenna_pattern(AUX_CAL_ZIP, swath="IW3", polarization="VV")


# Values printed by ISCE2 ``isceobj.Sensor.TOPS.Sentinel1.s1_anx2roll``.
@pytest.mark.parametrize(
    ("delta_anx", "expected"),
    [(0.0, 30.18637329), (1500.0, 29.64153763), (4000.0, 28.95191772)],
)
def test_nominal_roll_matches_isce2(delta_anx: float, expected: float) -> None:
    """The roll law reproduces ISCE2 ``s1_anx2roll``."""
    assert nominal_roll_deg(delta_anx) == pytest.approx(expected, abs=1e-6)


def test_eap_phase_vector_matches_isce2_reference(swath: Any) -> None:
    """Every burst phasor matches ISCE2 to machine precision."""
    reference = np.load(ISCE2_REFERENCE)
    pattern = read_elevation_antenna_pattern(
        AUX_CAL_ZIP, swath="IW1", polarization="VV"
    )
    columns = reference["columns"]
    for burst in swath.bursts:
        phase = np.angle(eap_phase_vector(swath, burst, pattern))[columns]
        diff = np.angle(np.exp(1j * (phase - reference["phase"][burst.index])))
        assert np.max(np.abs(diff)) < 1e-9


def test_correction_is_phase_only_and_layout_independent(swath: Any) -> None:
    """Amplitudes and zeros are preserved for any range window origin."""
    product = _product(swath, "002.36")
    pattern = read_elevation_antenna_pattern(
        AUX_CAL_ZIP, swath="IW1", polarization="VV"
    )
    expected = eap_phase_vector(swath, swath.bursts[2], pattern)
    for col0 in (0, 1234):
        array = _burst_array(col0=col0, width=500)
        (out,) = correct_eap_phase(product, swath, [array], aux_cal_path=AUX_CAL_ZIP)
        assert out.samples.dtype == np.complex64
        np.testing.assert_allclose(
            np.abs(out.samples), np.abs(array.samples), rtol=1e-6, atol=1e-7
        )
        assert np.all(out.samples[~array.valid_mask] == 0)
        ratio = out.samples[:, 3:] / array.samples[:, 3:]
        np.testing.assert_allclose(
            ratio,
            np.broadcast_to(1 / expected[col0 + 3 : col0 + 500], ratio.shape),
            atol=1e-5,
        )


def test_other_ipf_versions_are_untouched_and_002_36_requires_aux_cal(
    swath: Any,
) -> None:
    """Later IPF products pass through; 002.36 without AUX_CAL fails closed."""
    array = _burst_array(col0=0, width=64)
    later = _product(swath, "002.72")
    assert not requires_eap_correction(later)
    (out,) = correct_eap_phase(later, swath, [array], aux_cal_path=None)
    assert out is array
    with pytest.raises(Sentinel1ProductError, match="aux_cal_paths"):
        correct_eap_phase(_product(swath, "002.36"), swath, [array], aux_cal_path=None)


def test_missing_annotation_records_fail_closed(swath: Any) -> None:
    """Missing antennaPattern records never silently skip the correction."""
    pattern = read_elevation_antenna_pattern(
        AUX_CAL_ZIP, swath="IW1", polarization="VV"
    )
    stripped = replace(swath, elevation_angle_records=())
    with pytest.raises(Sentinel1ProductError, match="antennaPattern"):
        eap_phase_vector(stripped, stripped.bursts[0], pattern)


def test_path_safe_manifest_keeps_ipf_version(tmp_path: Path) -> None:
    """Materialized SAFEs keep the source IPF version."""
    from faninsar.missions.s1.io.path_safe import write_minimal_manifest

    write_minimal_manifest(tmp_path, "S1A_TEST", ["a.xml"], ipf_version="002.36")
    assert parse_ipf_version((tmp_path / "manifest.safe").read_bytes()) == "002.36"
    write_minimal_manifest(tmp_path, "S1A_TEST", ["a.xml"])
    assert parse_ipf_version((tmp_path / "manifest.safe").read_bytes()) is None


def test_fetch_uses_cache_then_sar_mpc(tmp_path: Path, monkeypatch: Any) -> None:
    """One SAR-MPC download serves every date; reuse never touches the network."""
    from faninsar import remote
    from faninsar.missions.s1 import aux_cal_fetch

    sensing = datetime(2015, 1, 15, tzinfo=UTC)
    calls: list[str] = []
    record = {
        "uuid": "f1430536-86b5-4045-ab8e-3d49eaed87cb",
        "product_name": AUX_CAL_NAME,
        "hash": None,
        "size": AUX_CAL_ZIP.stat().st_size,
        "validity_start": "2014-09-08T00:00:00",
        "creation_date": "2024-03-27T10:11:57",
    }

    class _Adapter:
        def list_records(self, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(f"list:{kwargs['mission_id']}")
            return [record]

    def _download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del kwargs
        calls.append(f"get:{asset.href}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(AUX_CAL_ZIP.read_bytes())
        return destination

    monkeypatch.setattr(aux_cal_fetch, "get_sar_mpc_adapter", _Adapter)
    monkeypatch.setattr(remote, "download", _download)
    cache = tmp_path / "aux_cal"
    acquisitions = {"20150115": ("S1A", sensing), "20150127": ("S1A", sensing)}
    bound = aux_cal_fetch.ensure_aux_cal_paths(
        acquisitions=acquisitions, cache_dir=cache
    )
    assert set(bound) == set(acquisitions)
    assert bound["20150115"].name == "s1a-aux-cal.xml"
    assert [c for c in calls if c.startswith("get:")] == [
        "get:https://sar-mpc.eu/download/f1430536-86b5-4045-ab8e-3d49eaed87cb/"
    ]
    calls.clear()
    again = aux_cal_fetch.ensure_aux_cal_paths(
        acquisitions=acquisitions, cache_dir=cache, auto_fetch=False
    )
    assert again == bound
    assert calls == []
    with pytest.raises(OSError, match="automatic download is disabled"):
        aux_cal_fetch.ensure_aux_cal_paths(
            acquisitions=acquisitions, cache_dir=tmp_path / "empty", auto_fetch=False
        )


def test_sar_mpc_listing_parses_json(monkeypatch: Any) -> None:
    """The listing query asks for active records on the registered API."""
    from faninsar.remote.providers import sar_mpc
    from faninsar.remote.records import RemoteResourceBudget

    payload = {"count": 1, "next": None, "results": [{"product_name": AUX_CAL_NAME}]}
    seen: list[str] = []

    def _get(self: Any, url: str, budget: Any, ledger: Any, *, accept: str) -> bytes:
        del self, budget, ledger
        seen.append(url)
        assert accept == "application/json"
        return json.dumps(payload).encode()

    monkeypatch.setattr(sar_mpc.SarMpcAdapter, "_get", _get)
    records = sar_mpc.SarMpcAdapter().list_records(
        product_type="AUX_CAL",
        mission_id="S1A",
        validity_start_lte="2015-01-15T00:00:00",
        budget=RemoteResourceBudget(),
    )
    assert records == payload["results"]
    assert "adf__active=1" in seen[0]
    assert seen[0].startswith("https://sar-mpc.eu/api/v1/?")


def test_every_production_burst_read_goes_through_eap_correction() -> None:
    """Loaders that feed interferometry must not bypass the EAP seam."""
    root = Path(__file__).resolve().parents[3] / "faninsar" / "missions" / "s1"
    for name in ("processing.py", "measure.py"):
        text = (root / name).read_text(encoding="utf-8")
        reads = text.count("read_full_burst(") + text.count("read_swath_bursts(")
        assert reads >= 1
        assert text.count("correct_eap_phase(") >= reads - (
            text.count("read_swath_bursts(") if name == "processing.py" else 0
        ), name


def test_zip_member_lookup_rejects_archives_without_data_xml(tmp_path: Path) -> None:
    """An AUX_CAL zip without its data XML is rejected."""
    bad = tmp_path / f"{AUX_CAL_NAME}.SAFE.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{AUX_CAL_NAME}.SAFE/manifest.safe", "<x/>")
    bad.write_bytes(buffer.getvalue())
    with pytest.raises(Sentinel1ProductError, match="no single data XML"):
        read_elevation_antenna_pattern(bad, swath="IW1", polarization="VV")

"""Developer-stage overlap NESD measurer for Sentinel-1 pairs."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1.measure import measure_s1_pair_misreg
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.coregistration.esd_overlap import OverlapESDRecord
from faninsar.processing.coregistration.misreg_network import MisregArc
from faninsar.processing.coregistration.offsets import PatchAmplitudeShiftResult
from faninsar.processing.geometry import ConstantDEM

_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)
_DT_S = 1.0
_LINES = 100
_SAMPLES = 80


def _valid_window(lines: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return a full-width valid-sample mask."""
    last = _SAMPLES - 1
    return tuple([0] * lines), tuple([last] * lines)


def _burst(index: int, *, start_line: int) -> S1Burst:
    """Build one burst whose line 0 is ``start_line`` intervals after origin."""
    first_valid, last_valid = _valid_window(_LINES)
    azimuth_time = _ORIGIN + timedelta(seconds=start_line * _DT_S)
    return S1Burst(
        index=index,
        azimuth_time=azimuth_time,
        sensing_time=azimuth_time,
        azimuth_anx_time_s=0.0,
        byte_offset=index * _LINES * _SAMPLES * 4,
        lines=_LINES,
        samples=_SAMPLES,
        first_valid_sample=first_valid,
        last_valid_sample=last_valid,
    )


def _swath(*bursts: S1Burst) -> S1Swath:
    """Build a minimal IW1 swath around the given bursts."""
    start = bursts[0].azimuth_time
    stop = bursts[-1].azimuth_time + timedelta(seconds=(_LINES - 1) * _DT_S)
    return S1Swath(
        swath="IW1",
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path="memory://measurement.tiff",
        lines=_LINES * len(bursts),
        samples=_SAMPLES,
        lines_per_burst=_LINES,
        samples_per_burst=_SAMPLES,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=_DT_S,
        slant_range_time_s=0.005,
        range_sampling_rate_hz=64_000_000.0,
        radar_frequency_hz=5.405e9,
        sensing_start=start,
        sensing_stop=stop,
        bursts=bursts,
        orbit=OrbitMetadata(
            reference_frame="ITRF",
            source="test",
            vectors=(
                OrbitStateVector(
                    time=start,
                    position_m=(7_000_000.0, 0.0, 0.0),
                    velocity_m_s=(0.0, 7_500.0, 0.0),
                ),
            ),
        ),
        doppler_centroid=(
            DopplerCentroidPolynomial(
                coefficients_hz=(100.0, -0.01),
                reference_range_m=800_000.0,
                reference_time_s=0.0,
            ),
        ),
        azimuth_fm_rate=(),
        azimuth_steering_rate_rad_s=0.0,
    )


def _product(path: Path, *bursts: S1Burst) -> S1Product:
    """Build a one-swath product whose date id is taken from ``path``."""
    return S1Product(
        product_id=path.stem,
        source_path=path,
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=1,
        polarizations=("VV",),
        swaths=(_swath(*bursts),),
    )


def test_measure_s1_pair_misreg_is_exported() -> None:
    """The measurer is part of the public Sentinel-1 mission surface."""
    from faninsar.missions.s1 import measure_s1_pair_misreg as exported

    assert exported is measure_s1_pair_misreg


def test_measure_s1_pair_misreg_public_signature() -> None:
    """The pair measurer exposes the PROPOSAL-0056 developer-stage signature."""
    parameters = inspect.signature(measure_s1_pair_misreg).parameters
    assert list(parameters) == [
        "primary_path",
        "secondary_path",
        "dem",
        "swaths",
        "bursts",
        "coherence_threshold",
        "ampcor_snr_threshold",
        "primary_orbit_path",
        "secondary_orbit_path",
        "device",
    ]
    assert parameters["dem"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["swaths"].default is None
    assert parameters["bursts"].default is None
    assert parameters["coherence_threshold"].default == 0.85
    assert parameters["ampcor_snr_threshold"].default == 6.0
    assert parameters["primary_orbit_path"].default is None
    assert parameters["secondary_orbit_path"].default is None
    assert parameters["device"].default == "auto"


def test_measure_s1_pair_misreg_no_windows_omits_both_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-burst selection has no overlap windows, so both axes are None."""
    primary = Path("S1A_IW_SLC__1SDV_20200101T000000.SAFE")
    secondary = Path("S1A_IW_SLC__1SDV_20200113T000000.SAFE")
    products = {
        primary: _product(primary, _burst(0, start_line=0)),
        secondary: _product(secondary, _burst(0, start_line=0)),
    }
    monkeypatch.setattr(
        "faninsar.missions.s1.measure.open_safe_product",
        lambda path, **_kwargs: products[Path(path)],
    )

    arc = measure_s1_pair_misreg(primary, secondary, dem=ConstantDEM(0.0))

    assert isinstance(arc, MisregArc)
    assert arc.primary == "20200101"
    assert arc.secondary == "20200113"
    assert arc.azimuth_shift_px is None
    assert arc.range_shift_px is None
    assert arc.method == "overlap"
    assert arc.n_valid == 0


def test_measure_s1_pair_misreg_empty_ampcor_omits_range_keeps_azimuth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ampcor n_valid=0 is not a range observation; a valid ESD azimuth is kept."""
    primary = Path("S1A_IW_SLC__1SDV_20200101T000000.SAFE")
    secondary = Path("S1A_IW_SLC__1SDV_20200113T000000.SAFE")
    products = {
        primary: _product(
            primary,
            _burst(0, start_line=0),
            _burst(1, start_line=80),
        ),
        secondary: _product(
            secondary,
            _burst(0, start_line=0),
            _burst(1, start_line=80),
        ),
    }
    monkeypatch.setattr(
        "faninsar.missions.s1.measure.open_safe_product",
        lambda path, **_kwargs: products[Path(path)],
    )

    def fake_observe(*_args: object, **_kwargs: object) -> tuple[object, object]:
        return (
            OverlapESDRecord(
                top_index=0,
                bottom_index=1,
                azimuth_shift_px=0.02,
                coherence=0.91,
                n_valid=12,
                response_rb_rad_px=48.0,
                response_consistency=0.01,
            ),
            PatchAmplitudeShiftResult(
                range_shift_px=0.0,
                azimuth_shift_px=0.0,
                n_valid=0,
                snr_median=0.0,
                n_attempted=0,
            ),
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.measure._observe_overlap_window",
        fake_observe,
        raising=False,
    )

    arc = measure_s1_pair_misreg(primary, secondary, dem=ConstantDEM(0.0))

    assert arc.primary == "20200101"
    assert arc.secondary == "20200113"
    assert arc.azimuth_shift_px == pytest.approx(0.02)
    assert arc.range_shift_px is None
    assert arc.method == "overlap"
    assert arc.n_valid == 12

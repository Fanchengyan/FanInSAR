"""Geographic complex sampling must use stencil-completeness validity."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coregistration.offsets import OffsetFieldResult
from faninsar.processing.coregistration.tops.deramp import TOPSCarrierModel
from faninsar.processing.geocoding.coordinates import RadarCoordinates
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geocoding.geo_modes import (
    coregister_geocoded_slcs,
    geocode_secondary_slc,
)
from faninsar.processing.geocoding.geo_resample import (
    resample_complex_at_coordinates,
    sample,
)

torch = pytest.importorskip("torch")

HEIGHT = WIDTH = 32
STRIPE = slice(16, 20)


def _carrier() -> TOPSCarrierModel:
    """Return a minimal TOPS carrier for geo reramp tests."""
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.005,
        range_sampling_rate_hz=32.0e6,
        azimuth_time_interval_s=0.002,
        doppler_centroid_hz=(20.0, 0.0, 0.0),
        doppler_t0_s=0.005,
        fm_rate_hz_s=(-2200.0, 0.0, 0.0),
        fm_t0_s=0.005,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.005,
        azimuth_steering_rate_hz_s=800.0,
    )


def test_sample_complex_invalidates_incomplete_source_stencil() -> None:
    """A masked zero-filled stripe must NaN outputs whose taps meet it."""
    source = np.ones((HEIGHT, WIDTH), dtype=np.complex64)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    valid[:, STRIPE] = False
    source[:, STRIPE] = 0
    azimuth, range_index = np.meshgrid(
        np.arange(HEIGHT, dtype=np.float64),
        np.arange(WIDTH, dtype=np.float64),
        indexing="ij",
    )
    coords = RadarCoordinates(
        azimuth + 0.5,
        range_index + 0.5,
        np.ones((HEIGHT, WIDTH), dtype=bool),
    )
    values, mask = sample(
        source,
        coords,
        kind="complex",
        device="cpu",
        source_valid=valid,
    )
    assert not mask[16, 18]
    assert np.isnan(values[16, 18].real) and np.isnan(values[16, 18].imag)
    complete, complete_mask = sample(
        np.ones((HEIGHT, WIDTH), dtype=np.complex64),
        coords,
        kind="complex",
        device="cpu",
        source_valid=np.ones((HEIGHT, WIDTH), dtype=bool),
    )
    assert complete_mask[16, 10]
    assert np.isfinite(complete[16, 10])


def test_opencv_executor_rejects_source_valid_before_cv2_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mask-aware OpenCV sampling is unsupported, including an all-true mask."""
    import builtins

    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "cv2" or name.startswith("cv2."):
            raise AssertionError("cv2 must not be imported when source_valid is set")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    source = np.ones((8, 8), dtype=np.complex64)
    azimuth = np.zeros((4, 4), dtype=np.float64)
    range_index = np.zeros((4, 4), dtype=np.float64)
    all_true = np.ones(source.shape, dtype=bool)
    with pytest.raises(NotImplementedError, match="torch"):
        resample_complex_at_coordinates(
            source,
            azimuth,
            range_index,
            source_valid=all_true,
            executor="opencv",
        )
    hole = all_true.copy()
    hole[0, 0] = False
    with pytest.raises(NotImplementedError, match="torch"):
        resample_complex_at_coordinates(
            source,
            azimuth,
            range_index,
            source_valid=hole,
            executor="opencv",
        )


def test_opencv_without_source_valid_keeps_optional_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting source_valid preserves the pre-existing OpenCV optional dependency."""
    import builtins

    real_import = builtins.__import__

    def no_cv2(name: str, *args: object, **kwargs: object) -> object:
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError("simulated missing opencv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_cv2)
    with pytest.raises(ImportError, match="opencv"):
        resample_complex_at_coordinates(
            np.ones((4, 4), dtype=np.complex64),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            executor="opencv",
        )


def test_geocode_secondary_does_not_inherit_reference_echo() -> None:
    """Secondary geo publication stays valid where only Reference echo is missing."""
    shape = (8, 10)
    azimuth, range_index = np.meshgrid(
        np.arange(1.0, 7.0),
        np.arange(1.0, 9.0),
        indexing="ij",
    )
    lut = Geo2RdrLUT(
        az_full=azimuth,
        rg_full=range_index,
        valid=np.ones(azimuth.shape, dtype=bool),
        full_radar_shape=shape,
        height_m=0.0,
    )
    offsets = OffsetFieldResult(
        range_offset_px=np.zeros(shape, dtype=np.float32),
        azimuth_offset_px=np.zeros(shape, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )
    secondary = np.ones(shape, dtype=np.complex64)
    reference_echo = np.ones(lut.shape, dtype=bool)
    reference_echo[:, 3] = False
    geo, valid = geocode_secondary_slc(
        secondary,
        secondary_carrier=_carrier(),
        reference_lut=lut,
        reference_valid=reference_echo,
        offsets=offsets,
        device="cpu",
        source_valid=np.ones(shape, dtype=bool),
    )
    assert valid[:, 3].all()
    assert np.isfinite(geo[:, 3]).all()

    reference_source_valid = np.ones(shape, dtype=bool)
    # Destination column j samples native range j+1 on this LUT.
    reference_source_valid[:, 4] = False
    _, pair_secondary, pair_valid = coregister_geocoded_slcs(
        np.ones(shape, dtype=np.complex64),
        secondary,
        reference_carrier=_carrier(),
        secondary_carrier=_carrier(),
        reference_lut=lut,
        offsets=offsets,
        reference_source_valid=reference_source_valid,
        secondary_source_valid=np.ones(shape, dtype=bool),
    )
    assert not pair_valid[:, 3].all()
    assert np.isnan(pair_secondary.real[:, 3]).all()

"""End-to-end tests for Stack ionosphere wiring (PROPOSAL-0036)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.atmosphere import IonosphereEstimationConfig
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.unwrap import SpatialIRLS
from faninsar.stack import Stack
from faninsar.stack.ion_store import IonosphereArtifactStore
from faninsar.stack.scene_store import write_scene_unit

F0 = 1257.5e6
BANDWIDTH = 28.0e6
FS = 32.0e6
GRID = (4, 16)
DATES = ("20240101", "20240113", "20240125")
DATE_ION = {"20240101": 0.0, "20240113": 0.05, "20240125": -0.1}


def _fine_mode_config() -> IonosphereEstimationConfig:
    return IonosphereEstimationConfig(
        f0=F0,
        freq_low=F0 - BANDWIDTH / 3,
        freq_high=F0 + BANDWIDTH / 3,
    )


def _slc_with_subband_phases(ion_rad: float) -> np.ndarray:
    """Craft an SLC whose low/high thirds carry the dispersive screen.

    One range bin per subband keeps each subband SLC a pure complex
    exponential (strictly positive amplitude, constant phase), and the
    distinct subband magnitudes keep the full-band SLC away from zero as
    well. Bin phases follow the ISCE3 solve convention: the dispersive
    screen at subband carrier ``f`` scales as ``ion * (f0 / f)``.
    """
    config = _fine_mode_config()
    spectrum = np.zeros(GRID, dtype=np.complex128)
    offsets_mhz = np.fft.fftfreq(GRID[1], d=1.0 / (FS / 1e6))
    low_bin = int(np.argmin(np.abs(offsets_mhz + 10.0)))
    high_bin = int(np.argmin(np.abs(offsets_mhz - 10.0)))
    spectrum[:, low_bin] = np.exp(1j * ion_rad * (F0 / config.freq_low))
    spectrum[:, high_bin] = 0.6 * np.exp(1j * ion_rad * (F0 / config.freq_high))
    slc = np.fft.ifft(spectrum, axis=-1) * 100.0
    return slc.astype(np.complex64)


def _stack_with_scene_units(tmp_path: Path) -> Stack:
    from faninsar import Pairs

    paths = []
    for date_id in DATES:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        pairs=Pairs.from_names(
            ["20240101_20240113", "20240113_20240125", "20240101_20240125"]
        ),
        multilook=(1, 1),
    ).prepare_scenes()
    reference = stack.reference
    for date_id in DATES:
        slc = _slc_with_subband_phases(DATE_ION[date_id])
        ones = np.ones(GRID, dtype=np.complex64)
        root = stack.config.work_dir / "coreg" / date_id / "scenes"
        write_scene_unit(
            root,
            date_id=date_id,
            reference_id=reference,
            domain="radar",
            tag="f0_IW1_b0",
            primary=slc if date_id == reference else ones,
            secondary=ones if date_id == reference else slc,
            row_origin=0,
            col_origin=0,
            grid_shape=GRID,
            wavelength_m=0.236,
        )
        stack.coreg_paths[date_id] = root.parent
    return stack


def test_estimate_apply_and_invert_end_to_end(tmp_path: Path) -> None:
    stack = _stack_with_scene_units(tmp_path)
    config = _fine_mode_config()
    stores = stack.estimate_ionosphere(
        config=config,
        range_sampling_rate_hz=FS,
        ion_multilook=(1, 1),
        filter_sigma=(0.0, 0.0),
        min_cluster_pixels=1,
        unwrap_method="irls",
        device="cpu",
    )
    assert len(stores) == 3
    expected_pair_screens = {
        "20240101_20240113": DATE_ION["20240101"] - DATE_ION["20240113"],
        "20240113_20240125": DATE_ION["20240113"] - DATE_ION["20240125"],
        "20240101_20240125": DATE_ION["20240101"] - DATE_ION["20240125"],
    }
    for store in stores:
        try:
            pair_id = f"{store.pair[0]}_{store.pair[1]}"
            artifact = store.read()
            np.testing.assert_allclose(
                artifact.ionosphere_phase,
                expected_pair_screens[pair_id],
                atol=2e-4,
            )
            assert np.isfinite(artifact.ionosphere_phase).all()
        finally:
            store.close()

    # Full IFG + unwrap generation on the same network.
    stack.form_interferograms(multilook=(1, 1))
    stack.unwrap(SpatialIRLS())
    # Correction consumes the durable root generation, including after the
    # in-memory generation lease has been refreshed from ``UNWRAP_CURRENT``.
    stack.refresh_unwrap_generation()

    corrected = stack.apply_ionosphere_correction(ion_multilook=(1, 1))
    assert set(corrected) == set(expected_pair_screens)
    unwrapped_stack = stack.network.interferograms.open_stack("unw_phase")
    for store in stores:
        try:
            pair_id = f"{store.pair[0]}_{store.pair[1]}"
            ifg_store_path = stack.config.work_dir / "ifg" / "ml_1x1" / pair_id
            from faninsar.stack.ion_store import (
                read_ion_correction_artifact,
            )

            unwrapped = unwrapped_stack.sel(pair=pair_id).values
            ion_screen = (
                IonosphereArtifactStore.open(
                    stack.config.work_dir / "ion" / "ml_1x1" / pair_id
                )
                .read()
                .ionosphere_phase
            )
            np.testing.assert_allclose(
                read_ion_correction_artifact(ifg_store_path),
                unwrapped - ion_screen,
                atol=1e-6,
            )
        finally:
            store.close()

    result = stack.invert_ionosphere_dates(ion_multilook=(1, 1))
    assert result.dates == DATES
    reference = DATE_ION[result.reference_date]
    expected = np.array([DATE_ION[date] - reference for date in DATES])
    np.testing.assert_allclose(result.screens[:, 0, 0], expected, atol=2e-4)


def test_estimate_rejects_multi_unit_scene_generations(tmp_path: Path) -> None:
    stack = _stack_with_scene_units(tmp_path)
    second = stack.config.work_dir / "coreg" / stack.reference / "scenes"
    write_scene_unit(
        second,
        date_id=stack.reference,
        reference_id=stack.reference,
        domain="radar",
        tag="f0_IW1_b1",
        primary=np.ones(GRID, dtype=np.complex64),
        secondary=np.ones(GRID, dtype=np.complex64),
        row_origin=0,
        col_origin=0,
        grid_shape=GRID,
        wavelength_m=0.236,
    )
    with pytest.raises(InvalidProcessingStateError, match="single-unit"):
        stack.estimate_ionosphere(
            config=_fine_mode_config(),
            range_sampling_rate_hz=FS,
            ion_multilook=(1, 1),
            unwrap_method="irls",
            device="cpu",
        )


def test_apply_refuses_degraded_generations_without_explicit_admission(
    tmp_path: Path,
) -> None:
    stack = _stack_with_scene_units(tmp_path)
    stack.estimate_ionosphere(
        config=_fine_mode_config(),
        range_sampling_rate_hz=FS,
        ion_multilook=(1, 1),
        filter_sigma=(0.0, 0.0),
        min_cluster_pixels=1,
        unwrap_method="irls",
        device="cpu",
        overwrite=True,
        degraded=True,
        degradation_reason="synthetic narrow-band mode",
    )
    stack.form_interferograms(multilook=(1, 1))
    stack.unwrap(SpatialIRLS())
    with pytest.raises(InvalidProcessingStateError, match="allow_degraded"):
        stack.apply_ionosphere_correction(ion_multilook=(1, 1))
    published = stack.apply_ionosphere_correction(
        ion_multilook=(1, 1), allow_degraded=True
    )
    assert published

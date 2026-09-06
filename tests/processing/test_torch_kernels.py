"""Parity tests for the Torch GPU kernels against the NumPy references."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coreg.esd import estimate_azimuth_shift_esd
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.interferometry.flatten import remove_topographic_phase
from faninsar.processing.interferometry.pair import (
    _block_reduce,
    form_interferogram,
    goldstein_filter,
)
from faninsar.processing.tops import TOPSCarrierModel, deramp, reramp
from faninsar.processing.torch_kernels import (
    CUDA_DTYPE_CHOICE,
    carrier_multiply_torch,
    carrier_phase_at_points_torch,
    esd_azimuth_shift_torch,
    goldstein_filter_torch,
    multilook_interferogram_torch,
    multilook_real_torch,
    remove_topographic_phase_torch,
    resolve_torch_device,
    tops_carrier_multiply_torch,
)


def _carrier_model() -> TOPSCarrierModel:
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.00533,
        range_sampling_rate_hz=6.434e7,
        azimuth_time_interval_s=0.002055,
        doppler_centroid_hz=(100.0, -50.0, 10.0),
        doppler_t0_s=0.00534,
        fm_rate_hz_s=(-2300.0, 4.5e5, -7.9e7),
        fm_t0_s=0.00533,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.00533,
        azimuth_steering_rate_hz_s=6500.0,
    )


def _random_slc(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


def test_resolve_torch_device_auto_falls_back_to_cpu() -> None:
    """Auto device resolution never hard-fails when CUDA is absent."""
    import torch

    device = resolve_torch_device("auto")
    assert isinstance(device, torch.device)
    if not torch.cuda.is_available():
        assert device.type == "cpu"
        with pytest.raises(RuntimeError, match="CUDA requested"):
            resolve_torch_device("cuda")


def test_cpu_carrier_multiply_preserves_complex128() -> None:
    """CPU kernels never truncate complex128 inputs."""
    samples = _random_slc((12, 16), seed=7).astype(np.complex128)
    samples += np.complex128(2.0**-40 + 1j * 2.0**-42)
    phase = np.linspace(-1.0e5, 1.0e5, samples.size).reshape(samples.shape)
    actual = carrier_multiply_torch(samples, phase, sign=-1.0, device="cpu")
    expected = samples * np.exp(-1j * phase)
    assert actual.dtype == np.complex128
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_carrier_precision_choice_is_pre_registered() -> None:
    """Large TOPS carrier phases use the qualified float64 CUDA path."""
    assert CUDA_DTYPE_CHOICE["carrier_phase"] == "float64"
    assert CUDA_DTYPE_CHOICE["carrier_multiply"] == "float64"


def test_explicit_float64_precision_override() -> None:
    """A per-kernel override selects the float64 fallback dtype."""
    import torch

    import faninsar.processing.torch_kernels as kernels

    dtype = kernels._complex_dtype(
        torch.device("cuda"),
        "multilook_interferogram",
        "float64",
    )
    assert dtype == torch.complex128


def test_unqualified_auto_float32_falls_back_to_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapping entry cannot enable float32 without qualification evidence."""
    import torch

    import faninsar.processing.torch_kernels as kernels

    monkeypatch.setitem(
        kernels.CUDA_DTYPE_CHOICE,
        "multilook_interferogram",
        "float32",
    )
    dtype = kernels._complex_dtype(
        torch.device("cuda"),
        "multilook_interferogram",
        "auto",
    )
    assert dtype == torch.complex128


def test_unqualified_explicit_float32_falls_back_to_float64() -> None:
    """An explicit float32 override cannot bypass CUDA qualification."""
    import torch

    import faninsar.processing.torch_kernels as kernels

    dtype = kernels._complex_dtype(
        torch.device("cuda"),
        "multilook_interferogram",
        "float32",
    )
    assert dtype == torch.complex128


def test_cleanup_runs_when_kernel_validation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel wrappers never call empty_cache, including on exception paths."""
    import inspect

    import torch

    import faninsar.processing.torch_kernels as kernels

    empty_calls: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_calls.append("cuda"),
    )
    if hasattr(torch, "mps"):
        monkeypatch.setattr(
            torch.mps,
            "empty_cache",
            lambda: empty_calls.append("mps"),
            raising=False,
        )
    with pytest.raises(RuntimeError, match="2-D complex array"):
        kernels.carrier_multiply_torch(
            np.ones(8, dtype=np.complex64),
            np.zeros(8),
            sign=-1.0,
            device="cpu",
        )
    assert empty_calls == []
    assert "empty_cache" not in inspect.getsource(kernels.cleanup_device)
    assert "empty_cache" not in inspect.getsource(kernels._cleanup_after_kernel)


def test_kernel_tiles_do_not_empty_cache_under_eager_8gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FINDING-c03ef278: mocked 8 GiB eager still has zero kernel empty_cache."""
    import inspect

    import torch

    import faninsar.processing.torch_kernels as kernels
    from faninsar.processing.runtime.device import reclaim_checkpoint

    empty_calls: list[str] = []

    def record_empty() -> None:
        empty_calls.append("cuda")

    monkeypatch.setattr(torch.cuda, "empty_cache", record_empty)
    samples = _random_slc((32, 48), seed=3)
    model = _carrier_model()
    kernels.tops_carrier_multiply_torch(
        samples,
        model,
        sign=-1.0,
        device="cpu",
    )
    assert empty_calls == []
    from faninsar.processing import resampling_torch

    assert "empty_cache" not in inspect.getsource(resampling_torch._cleanup_device)
    eight_gib = 8 * 1024**3
    reclaim_checkpoint(
        torch.device("cuda"),
        "eager",
        kind="persist",
        total_bytes=eight_gib,
        reserved_bytes=eight_gib,
    )
    assert empty_calls == ["cuda"]
    empty_calls.clear()
    kernels.tops_carrier_multiply_torch(
        samples,
        model,
        sign=-1.0,
        device="cpu",
    )
    reclaim_checkpoint(
        torch.device("cuda"),
        "adaptive",
        kind="stage",
        total_bytes=eight_gib,
        reserved_bytes=eight_gib,
    )
    assert empty_calls == ["cuda"]


def test_multilook_interferogram_torch_matches_numpy() -> None:
    """Torch multilook interferogram matches the NumPy reference."""
    primary = _random_slc((48, 96), seed=11)
    secondary = _random_slc((48, 96), seed=12)
    reference = form_interferogram(primary, secondary, multilook=(3, 4))
    product = multilook_interferogram_torch(
        primary,
        secondary,
        multilook=(3, 4),
        device="cpu",
    )
    np.testing.assert_allclose(
        product.complex_ifg,
        reference.complex_ifg,
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        product.coherence,
        reference.coherence,
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        product.wrapped_phase,
        reference.wrapped_phase,
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        product.amplitude,
        reference.amplitude,
        rtol=1e-4,
        atol=1e-5,
    )


def test_multilook_interferogram_torch_dead_pixel_mask_matches_numpy() -> None:
    """Dead-pixel weighted multilook matches the NumPy path."""
    primary = _random_slc((40, 64), seed=21)
    secondary = _random_slc((40, 64), seed=22)
    primary[:, 10] = np.complex64(0.5 + 0.0j)
    secondary[:, 10] = np.complex64(0.25 + 0.1j)
    reference = form_interferogram(
        primary,
        secondary,
        multilook=(2, 4),
        dead_pixel_amp_threshold=1.0,
    )
    product = multilook_interferogram_torch(
        primary,
        secondary,
        multilook=(2, 4),
        dead_pixel_amp_threshold=1.0,
        device="cpu",
    )
    np.testing.assert_allclose(
        product.complex_ifg,
        reference.complex_ifg,
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        product.coherence,
        reference.coherence,
        rtol=1e-4,
        atol=1e-5,
    )


def test_multilook_interferogram_torch_identical_slcs_unit_coherence() -> None:
    """Identical SLCs keep unit coherence through the Torch kernel."""
    slc = _random_slc((32, 32), seed=3)
    product = multilook_interferogram_torch(
        slc,
        slc,
        multilook=(2, 2),
        device="cpu",
    )
    assert float(np.nanmean(product.coherence)) > 0.99
    assert float(np.nanmax(np.abs(product.wrapped_phase))) < 1e-5


def test_multilook_interferogram_torch_singleton_keeps_ifg_but_not_coherence() -> None:
    """A one-sample look remains valid complex support without MLE coherence."""
    primary = np.full((2, 2), np.nan + 1j * np.nan, dtype=np.complex64)
    secondary = np.full((2, 2), np.nan + 1j * np.nan, dtype=np.complex64)
    primary[0, 0] = 2.0 + 0.0j
    secondary[0, 0] = 1.0 + 0.0j

    product = multilook_interferogram_torch(
        primary,
        secondary,
        multilook=(2, 2),
        device="cpu",
    )

    assert product.valid_mask is not None
    assert bool(product.valid_mask[0, 0])
    assert product.complex_ifg[0, 0] == pytest.approx(2.0 + 0.0j)
    assert np.isnan(product.coherence[0, 0])


def test_goldstein_filter_torch_matches_numpy() -> None:
    """Torch Goldstein filter matches the NumPy ISCE2-port within tolerance."""
    ifg = _random_slc((64, 64), seed=31)
    reference = goldstein_filter(ifg, alpha=0.5, window=16)
    filtered = goldstein_filter_torch(ifg, alpha=0.5, window=16, device="cpu")
    assert filtered.shape == ifg.shape
    assert np.iscomplexobj(filtered)
    np.testing.assert_allclose(filtered, reference, rtol=1e-3, atol=1e-3)


def test_goldstein_filter_torch_chunked_matches_full() -> None:
    """Chunked Goldstein scheduling reproduces the full-array result."""
    ifg = _random_slc((70, 60), seed=33)
    full = goldstein_filter_torch(ifg, alpha=0.5, window=16, device="cpu")
    window = 16
    halo = window - 1
    chunks = []
    for row_start in range(0, ifg.shape[0], 24):
        row_stop = min(row_start + 24, ifg.shape[0])
        slice0 = max(0, row_start - halo)
        slice1 = min(ifg.shape[0], row_stop + halo)
        chunk = goldstein_filter_torch(
            ifg[slice0:slice1],
            alpha=0.5,
            window=window,
            device="cpu",
            output_row0=row_start,
            output_rows=row_stop - row_start,
            input_row0=slice0,
        )
        chunks.append(chunk)
    chunked = np.concatenate(chunks, axis=0)
    np.testing.assert_array_equal(chunked, full)


def test_carrier_multiply_torch_matches_deramp_and_reramp() -> None:
    """Torch carrier multiply reproduces NumPy deramp/reramp closely."""
    model = _carrier_model()
    samples = _random_slc((32, 64), seed=41)
    deramped = tops_carrier_multiply_torch(
        samples,
        model,
        sign=-1.0,
        device="cpu",
    )
    np.testing.assert_allclose(deramped, deramp(samples, model), atol=1e-6)
    restored = tops_carrier_multiply_torch(
        deramped,
        model,
        sign=1.0,
        device="cpu",
    )
    np.testing.assert_allclose(restored, reramp(deramped, model), atol=1e-6)


def test_carrier_phase_at_points_torch_matches_numpy() -> None:
    """Analytical carrier phase matches the NumPy polynomial evaluation."""
    from faninsar.processing.tops import tops_carrier_phase
    from faninsar.processing.tops.deramp import carrier_phase_at_points

    model = _carrier_model()
    rows = np.arange(32, dtype=np.float64)[:, None]
    cols = np.arange(64, dtype=np.float64)[None, :]
    expected = tops_carrier_phase(model, 32, 64, dtype=np.float64)
    actual = carrier_phase_at_points_torch(
        model,
        np.broadcast_to(rows, (32, 64)),
        np.broadcast_to(cols, (32, 64)),
        centre_row=16.0,
        dtype=np.float64,
        device="cpu",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    scalar = carrier_phase_at_points(
        model,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        centre_row=16.0,
    )
    assert np.isfinite(float(np.asarray(scalar).flat[0]))


def test_remove_topographic_phase_torch_matches_numpy() -> None:
    """Torch flatten multiply matches the NumPy reference."""
    ifg = _random_slc((40, 50), seed=51)
    topo = np.linspace(-2.0, 3.0, 40 * 50).reshape(40, 50)
    expected = remove_topographic_phase(ifg, topo)
    actual = remove_topographic_phase_torch(ifg, topo, device="cpu")
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_multilook_real_torch_matches_numpy() -> None:
    """Torch real-field block reduce matches the NumPy reference."""
    rng = np.random.default_rng(61)
    array = rng.normal(size=(48, 80)).astype(np.float32)
    actual = multilook_real_torch(array, 3, 5, device="cpu")
    expected = _block_reduce(array, 3, 5)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def _dual_tone_reference(
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    n_az, n_rg = shape
    i = np.arange(n_az)[:, None]
    tone_lower = np.exp(2j * np.pi * -0.225 * i)
    tone_upper = np.exp(2j * np.pi * 0.225 * i)
    noise = 0.01 * (rng.normal(size=(n_az, n_rg)) + 1j * rng.normal(size=(n_az, n_rg)))
    return ((tone_lower + tone_upper) + noise).astype(np.complex64)


def test_esd_azimuth_shift_torch_recovers_shift() -> None:
    """Torch ESD recovers an injected azimuth shift like the NumPy ESD."""
    rng = np.random.default_rng(42)
    n_az, n_rg = 256, 64
    reference = _dual_tone_reference((n_az, n_rg), rng)
    injected = 0.25
    f_az = np.fft.fftfreq(n_az, d=1.0)[:, None]
    phase_ramp = np.exp(-2j * np.pi * f_az * injected)
    secondary = np.fft.ifft(
        np.fft.fft(reference, axis=0) * phase_ramp,
        axis=0,
    ).astype(np.complex64)
    expected = estimate_azimuth_shift_esd(reference, secondary)
    result = esd_azimuth_shift_torch(reference, secondary, device="cpu")
    assert result.azimuth_shift_px == pytest.approx(injected, abs=0.05)
    assert result.azimuth_shift_px == pytest.approx(
        expected.azimuth_shift_px,
        abs=0.01,
    )


def test_esd_cpu_complex128_is_not_truncated() -> None:
    """CPU ESD preserves sub-complex64 perturbations at the tensor boundary."""
    rng = np.random.default_rng(43)
    reference = _dual_tone_reference((128, 32), rng).astype(np.complex128)
    reference += np.complex128(2.0**-40 + 1j * 2.0**-42)
    secondary = reference * np.exp(1j * np.float64(1.0e-8))
    expected = estimate_azimuth_shift_esd(reference, secondary)
    actual = esd_azimuth_shift_torch(reference, secondary, device="cpu")
    assert actual.azimuth_shift_px == pytest.approx(
        expected.azimuth_shift_px,
        abs=1e-10,
    )


def test_esd_range_chunking_matches_whole_range() -> None:
    """Range-column chunking preserves the ESD estimate."""
    rng = np.random.default_rng(44)
    reference = _dual_tone_reference((128, 37), rng)
    secondary = reference * np.exp(1j * np.float32(0.03))
    whole = esd_azimuth_shift_torch(
        reference,
        secondary,
        device="cpu",
        range_chunk_size=128,
    )
    chunked = esd_azimuth_shift_torch(
        reference,
        secondary,
        device="cpu",
        range_chunk_size=5,
    )
    assert chunked.azimuth_shift_px == pytest.approx(whole.azimuth_shift_px, abs=1e-10)
    assert chunked.coherence == pytest.approx(whole.coherence, abs=1e-10)
    assert chunked.phase_rad == pytest.approx(whole.phase_rad, abs=1e-10)


def test_esd_rejects_invalid_range_chunk_size() -> None:
    """ESD rejects non-positive internal chunk sizes."""
    with pytest.raises(InvalidProcessingStateError, match="range_chunk_size"):
        esd_azimuth_shift_torch(
            np.ones((8, 4), dtype=np.complex64),
            np.ones((8, 4), dtype=np.complex64),
            device="cpu",
            range_chunk_size=0,
        )


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA qualification requires a visible CUDA device",
)
def test_cuda_float32_qualification_records_parity() -> None:
    """Measure a qualified float32 path against the CPU float64 reference."""
    import torch

    import faninsar.processing.torch_kernels as kernels

    samples = _random_slc((64, 96), seed=45)
    phase = np.linspace(-10.0, 10.0, samples.size).reshape(samples.shape)
    expected = carrier_multiply_torch(samples, phase, sign=-1.0, device="cpu")
    original = kernels.CUDA_FLOAT32_QUALIFIED
    kernels.CUDA_FLOAT32_QUALIFIED = frozenset({"carrier_multiply"})
    try:
        actual = carrier_multiply_torch(
            samples,
            phase,
            sign=-1.0,
            device="cuda",
            precision="float32",
        )
    finally:
        kernels.CUDA_FLOAT32_QUALIFIED = original
    assert np.max(np.abs(actual - expected)) <= 1e-3
    assert torch.cuda.is_available()

"""Tests for coregistration offsets and complex resampling."""

from __future__ import annotations

import gc
import subprocess
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.processing.coreg import (
    combine_offset_fields,
    estimate_global_shift,
    estimate_patch_amplitude_shift,
    geometry_shift_offsets,
    refine_peak_subpixel,
    resample_complex,
)
from faninsar.processing.errors import InvalidProcessingStateError


def test_estimate_global_shift_recovers_injected_offset() -> None:
    """Cross-correlation recovers an injected integer pixel shift."""
    rng = np.random.default_rng(1)
    reference = (rng.normal(size=(48, 48)) + 1j * rng.normal(size=(48, 48))).astype(
        np.complex64
    )
    # secondary is reference shifted by +3 range, -2 azimuth
    secondary = np.roll(np.roll(reference, shift=3, axis=1), shift=-2, axis=0)
    rg_shift, az_shift = estimate_global_shift(reference, secondary, max_shift=8)
    assert rg_shift == pytest.approx(3.0, abs=1e-3)
    assert az_shift == pytest.approx(-2.0, abs=1e-3)


def test_estimate_global_shift_subpixel_recovers_fractional_shift() -> None:
    """Subpixel cross-correlation recovers a fractional pixel shift."""
    from scipy.ndimage import shift

    # Use a Gaussian blob so the cross-correlation peak is sharp
    y, x = np.mgrid[-32:32, -32:32]
    blob = np.exp(-(x**2 + y**2) / 200.0)
    reference = (blob + 1j * blob).astype(np.complex64)
    # Apply a real fractional shift to both real and imaginary parts
    secondary = (
        shift(reference.real, (0.0, 0.4), order=3)
        + 1j * shift(reference.imag, (0.0, 0.4), order=3)
    ).astype(np.complex64)
    rg_shift, az_shift = estimate_global_shift(
        reference, secondary, max_shift=8, subpixel=True
    )
    assert rg_shift == pytest.approx(0.4, abs=0.05)
    assert az_shift == pytest.approx(0.0, abs=0.05)


def _sar_like_amplitude(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Band-limited random amplitude with sparse bright scatterers (SAR-like)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=shape), sigma=1.5)
    # Sparse bright peaks improve local uniqueness for Ampcor
    n_peaks = max(20, shape[0] * shape[1] // 2000)
    for _ in range(n_peaks):
        az = int(rng.integers(10, shape[0] - 10))
        rg = int(rng.integers(10, shape[1] - 10))
        field[az - 1 : az + 2, rg - 1 : rg + 2] += float(rng.uniform(3.0, 8.0))
    return np.abs(field).astype(np.float32)


def test_ampcor_magnitude_tile_cyclic_shift_matches_rolled_source() -> None:
    """Lazy cyclic indexing matches the legacy full-image roll semantics."""
    from faninsar.processing.coreg import offsets as offsets_mod

    source = np.arange(8 * 11, dtype=np.float32).reshape(8, 11).astype(np.complex64)
    shifted = np.roll(np.roll(source, shift=3, axis=0), shift=-4, axis=1)
    expected = np.abs(shifted[1:6, 2:9]).astype(np.float32)
    actual = offsets_mod._ampcor_magnitude_tile(
        source,
        1,
        6,
        2,
        9,
        cyclic_shift=(3, -4),
    )
    np.testing.assert_array_equal(actual, expected)


def test_estimate_patch_amplitude_shift_recovers_injected_offset() -> None:
    """Multi-window Ampcor recovers an injected integer residual."""
    texture = _sar_like_amplitude((256, 512), seed=2)
    reference = texture.astype(np.complex64)
    secondary = np.roll(np.roll(reference, shift=2, axis=1), shift=-1, axis=0)
    result = estimate_patch_amplitude_shift(
        reference,
        secondary,
        window_az=32,
        window_rg=64,
        search_az=8,
        search_rg=8,
        n_az=8,
        n_rg=12,
        snr_threshold=3.0,
        max_abs_residual=4.0,
        margin_rg=40,
        margin_az=40,
    )
    assert result.n_valid > 0
    assert result.range_shift_px == pytest.approx(2.0, abs=0.15)
    assert result.azimuth_shift_px == pytest.approx(-1.0, abs=0.15)


def test_estimate_patch_lazy_prealignment_matches_full_roll() -> None:
    """Production lazy pre-alignment preserves the rolled-source result."""
    texture = _sar_like_amplitude((160, 320), seed=22)
    reference = texture.astype(np.complex64)
    source = np.roll(np.roll(reference, shift=2, axis=1), shift=-1, axis=0)
    kwargs = {
        "window_az": 24,
        "window_rg": 40,
        "search_az": 5,
        "search_rg": 5,
        "n_az": 5,
        "n_rg": 7,
        "snr_threshold": 3.0,
        "max_abs_residual": 4.0,
        "margin_rg": 30,
        "margin_az": 30,
    }
    rolled = estimate_patch_amplitude_shift(reference, source, **kwargs)
    lazy = estimate_patch_amplitude_shift(
        reference,
        reference,
        secondary_shift=(-1, 2),
        **kwargs,
    )
    assert lazy.n_attempted == rolled.n_attempted
    assert lazy.n_valid == rolled.n_valid
    assert lazy.range_shift_px == pytest.approx(rolled.range_shift_px, abs=1e-6)
    assert lazy.azimuth_shift_px == pytest.approx(rolled.azimuth_shift_px, abs=1e-6)


def test_estimate_patch_amplitude_shift_zero_when_aligned() -> None:
    """Aligned scenes yield near-zero residual after SNR cull."""
    texture = _sar_like_amplitude((200, 400), seed=3)
    reference = texture.astype(np.complex64)
    result = estimate_patch_amplitude_shift(
        reference,
        reference.copy(),
        window_az=32,
        window_rg=48,
        search_az=6,
        search_rg=6,
        n_az=6,
        n_rg=8,
        snr_threshold=3.0,
        max_abs_residual=1.2,
        margin_rg=40,
        margin_az=40,
    )
    assert result.n_valid > 0
    assert abs(result.range_shift_px) < 0.15
    assert abs(result.azimuth_shift_px) < 0.15


def test_estimate_patch_amplitude_shift_torch_matches_numpy() -> None:
    """Bounded eager Torch Ampcor preserves NumPy cull and shift results."""
    pytest.importorskip("torch")
    texture = _sar_like_amplitude((160, 320), seed=23)
    reference = texture.astype(np.complex64)
    secondary = np.roll(np.roll(reference, shift=2, axis=1), shift=-1, axis=0)
    kwargs = {
        "window_az": 32,
        "window_rg": 48,
        "search_az": 8,
        "search_rg": 8,
        "n_az": 6,
        "n_rg": 10,
        "snr_threshold": 3.0,
        "max_abs_residual": 4.0,
        "margin_rg": 32,
        "margin_az": 32,
        "subpixel": True,
    }
    expected = estimate_patch_amplitude_shift(reference, secondary, **kwargs)
    actual = estimate_patch_amplitude_shift(
        reference,
        secondary,
        executor="torch",
        batch_size=3,
        device="cpu",
        **kwargs,
    )
    assert actual.n_attempted == expected.n_attempted
    assert actual.n_valid == expected.n_valid
    np.testing.assert_allclose(
        (actual.range_shift_px, actual.azimuth_shift_px),
        (expected.range_shift_px, expected.azimuth_shift_px),
        atol=2e-5,
    )


def test_torch_ampcor_even_median_stays_in_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch aggregation averages even middle values without NumPy."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    calls = 0

    def fake_ncc(
        ref_windows: object,
        _sec_searches: object,
        **_kwargs: object,
    ) -> tuple[object, object, object]:
        nonlocal calls
        values = (
            ([1.0, 3.0], [2.0, 4.0], [10.0, 20.0]),
            ([5.0, 9.0], [6.0, 10.0], [30.0, 40.0]),
        )[calls]
        calls += 1
        count = ref_windows.shape[0]
        return tuple(
            torch.tensor(values[index][:count], dtype=torch.float64)
            for index in range(3)
        )

    def fail_numpy_median(*_args: object, **_kwargs: object) -> object:
        pytest.fail("Torch Ampcor aggregation called NumPy median")

    monkeypatch.setattr(offsets_mod, "_torch_patch_ncc_batch", fake_ncc)
    monkeypatch.setattr(offsets_mod.np, "median", fail_numpy_median)
    samples = np.ones((64, 96), dtype=np.complex64)
    result = estimate_patch_amplitude_shift(
        samples,
        samples,
        window_az=8,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=4,
        snr_threshold=0.0,
        max_abs_residual=100.0,
        margin_rg=16,
        margin_az=8,
        executor="torch",
        batch_size=2,
        device="cpu",
    )

    assert result.n_valid == 4
    assert result.range_shift_px == 4.0
    assert result.azimuth_shift_px == 5.0
    assert result.snr_median == 25.0
    assert calls == 2


@pytest.mark.parametrize(
    ("dtype", "value", "expected_magnitude"),
    [
        (np.dtype(np.complex64), 3.0 + 4.0j, 5.0),
        (np.dtype(np.float32), 3.0, 3.0),
    ],
)
def test_torch_ampcor_allowed_dtypes_convert_to_float32_magnitude(
    monkeypatch: pytest.MonkeyPatch,
    dtype: np.dtype,
    value: complex,
    expected_magnitude: float,
) -> None:
    """Allowed Torch inputs become contiguous float32 magnitude batches."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    captured: dict[str, object] = {}

    def fake_ncc(
        ref_windows: object,
        sec_searches: object,
        **_kwargs: object,
    ) -> tuple[object, object, object]:
        captured["reference"] = ref_windows.detach().cpu()
        captured["secondary"] = sec_searches.detach().cpu()
        count = ref_windows.shape[0]
        return (
            torch.zeros(count, dtype=torch.float64),
            torch.zeros(count, dtype=torch.float64),
            torch.full((count,), 10.0, dtype=torch.float64),
        )

    monkeypatch.setattr(offsets_mod, "_torch_patch_ncc_batch", fake_ncc)
    samples = np.full((64, 96), value, dtype=dtype)
    result = estimate_patch_amplitude_shift(
        samples,
        samples,
        window_az=8,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        margin_rg=16,
        margin_az=8,
        executor="torch",
        device="cpu",
    )

    assert result.n_valid == 1
    reference_batch = captured["reference"]
    assert reference_batch.dtype == torch.float32
    assert reference_batch.is_contiguous()
    torch.testing.assert_close(
        reference_batch,
        torch.full_like(reference_batch, expected_magnitude),
    )


def test_full_iw1_shape_rejects_overlapping_view() -> None:
    """Malformed overlapping views fail closed before Torch copies them."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg import geometry_coreg

    shape = (1494, 20460)
    backing = np.ones(1, dtype=np.complex64)
    samples = np.lib.stride_tricks.as_strided(
        backing,
        shape=shape,
        strides=(0, 0),
        writeable=False,
    )
    kwargs = {
        "window_az": 32,
        "window_rg": 64,
        "search_az": 16,
        "search_rg": 16,
        "n_az": 1,
        "n_rg": 1,
        "snr_threshold": 5.0,
        "max_abs_residual": 1.2,
        "margin_rg": 1000,
        "margin_az": None,
    }

    with pytest.raises(InvalidProcessingStateError, match="overlapping"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            **kwargs,
        )


def test_ampcor_copies_safe_noncontiguous_torch_view() -> None:
    """Torch admission copies a bounded, non-overlapping positive-stride view."""
    backing = np.ones((256, 2048), dtype=np.complex64)
    view = backing[:, ::2]
    assert view.nbytes > 1_000_000
    assert not view.flags.c_contiguous

    result = estimate_patch_amplitude_shift(
        view,
        view,
        executor="torch",
        device="cpu",
        window_az=16,
        window_rg=32,
        search_az=4,
        search_rg=4,
        n_az=1,
        n_rg=1,
        margin_rg=32,
        margin_az=32,
    )
    assert result.n_attempted == 1


def test_ampcor_conversion_preflight_rejects_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized compatibility copies fail before ``np.array`` allocation."""
    from faninsar.processing.coreg import offsets as offsets_mod

    samples = np.ones((16, 32), dtype=np.complex64)[:, ::2]
    monkeypatch.setattr(offsets_mod, "_TORCH_AMPCOR_CONVERSION_CAP_BYTES", 1)
    monkeypatch.setattr(
        offsets_mod.np,
        "array",
        lambda *_args, **_kwargs: pytest.fail("conversion preflight allocated"),
    )
    with pytest.raises(InvalidProcessingStateError, match="conversion"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            executor="torch",
            device="cpu",
            window_az=8,
            window_rg=16,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_rg=16,
            margin_az=8,
        )


def test_ampcor_public_workspace_limits_conversion_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public workspace limits apply before compatibility copy allocation."""
    from faninsar.processing.coreg import offsets as offsets_mod

    samples = np.ones((16, 32), dtype=np.complex64)[:, ::2]
    monkeypatch.setattr(
        offsets_mod.np,
        "array",
        lambda *_args, **_kwargs: pytest.fail("conversion preflight allocated"),
    )
    with pytest.raises(InvalidProcessingStateError, match="conversion"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            executor="torch",
            device="cpu",
            batch_size=1,
            max_workspace_bytes=1,
            window_az=8,
            window_rg=16,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_rg=16,
            margin_az=8,
        )


def test_estimate_patch_amplitude_shift_rejects_invalid_torch_batch() -> None:
    """Torch Ampcor rejects non-positive batch sizes before execution."""
    with pytest.raises(InvalidProcessingStateError, match="batch_size"):
        estimate_patch_amplitude_shift(
            np.ones((64, 96), dtype=np.complex64),
            np.ones((64, 96), dtype=np.complex64),
            executor="torch",
            device="cpu",
            batch_size=0,
        )


def test_estimate_patch_amplitude_shift_rejects_workspace_overcommit() -> None:
    """Torch Ampcor rejects a batch that exceeds its hard workspace limit."""
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="workspace"):
        estimate_patch_amplitude_shift(
            np.ones((128, 256), dtype=np.complex64),
            np.ones((128, 256), dtype=np.complex64),
            executor="torch",
            device="cpu",
            batch_size=32,
            max_workspace_bytes=1,
        )


def test_estimate_patch_amplitude_shift_rejects_total_work_overcommit() -> None:
    """Ampcor rejects an oversized patch grid before coordinate allocation."""
    with pytest.raises(InvalidProcessingStateError, match="total-work"):
        estimate_patch_amplitude_shift(
            np.ones((64, 96), dtype=np.complex64),
            np.ones((64, 96), dtype=np.complex64),
            executor="torch",
            device="cpu",
            n_az=65,
            n_rg=65,
        )


@pytest.mark.parametrize("executor", ["numpy", "torch"])
def test_estimate_patch_amplitude_shift_checks_total_work_before_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    executor: str,
) -> None:
    """Both executors reject an oversized patch grid before coordinates exist."""
    from faninsar.processing.coreg import offsets as offsets_mod

    monkeypatch.setattr(
        offsets_mod.np,
        "linspace",
        lambda *_args, **_kwargs: pytest.fail("coordinates were constructed first"),
    )
    samples = np.ones((64, 96), dtype=np.complex64)
    with pytest.raises(InvalidProcessingStateError, match="total-work"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            executor=executor,
            device="cpu",
            n_az=65,
            n_rg=65,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("snr_threshold", -1.0), ("max_abs_residual", -1.0), ("margin_rg", -1)],
)
def test_estimate_patch_amplitude_shift_rejects_negative_limits(
    parameter: str, value: float
) -> None:
    """Ampcor rejects negative cull and border limits before execution."""
    with pytest.raises(InvalidProcessingStateError, match=parameter):
        estimate_patch_amplitude_shift(
            np.ones((64, 96), dtype=np.complex64),
            np.ones((64, 96), dtype=np.complex64),
            **{parameter: value},
        )


def test_ampcor_cull_accepts_exact_boundary_and_rejects_near_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch culling keeps inclusive boundaries with deterministic parity."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    def fake_ncc(*_args: object, **_kwargs: object) -> tuple[object, object, object]:
        return (
            torch.tensor([1.2], dtype=torch.float64),
            torch.tensor([-1.2], dtype=torch.float64),
            torch.tensor([5.0], dtype=torch.float64),
        )

    monkeypatch.setattr(
        offsets_mod,
        "_patch_ncc_shift",
        lambda *_args, **_kwargs: (1.2, -1.2, 5.0),
    )
    monkeypatch.setattr(offsets_mod, "_torch_patch_ncc_batch", fake_ncc)
    samples = np.ones((64, 64), dtype=np.complex64)
    exact = estimate_patch_amplitude_shift(
        samples,
        samples,
        window_az=16,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        snr_threshold=5.0,
        max_abs_residual=1.2,
        margin_rg=16,
        margin_az=16,
        executor="torch",
        device="cpu",
    )
    assert exact.n_valid == 1

    snr_below = np.float64(5.0)
    residual_above = np.float64(1.2)
    for _ in range(8):
        snr_below = np.nextafter(snr_below, 0.0)
        residual_above = np.nextafter(residual_above, 2.0)

    def below_boundary(
        *_args: object, **_kwargs: object
    ) -> tuple[object, object, object]:
        return (
            torch.tensor([residual_above], dtype=torch.float64),
            torch.tensor([-1.2], dtype=torch.float64),
            torch.tensor([snr_below], dtype=torch.float64),
        )

    monkeypatch.setattr(offsets_mod, "_torch_patch_ncc_batch", below_boundary)
    monkeypatch.setattr(
        offsets_mod,
        "_patch_ncc_shift",
        lambda *_args, **_kwargs: (
            float(residual_above),
            -1.2,
            float(snr_below),
        ),
    )
    near = estimate_patch_amplitude_shift(
        samples,
        samples,
        window_az=16,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        snr_threshold=5.0,
        max_abs_residual=1.2,
        margin_rg=16,
        margin_az=16,
        executor="torch",
        device="cpu",
    )
    assert near.n_valid == 0


def test_ampcor_workspace_admission_is_released_on_empty_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty and failing Torch batches release their process-local reservation."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    samples = np.ones((64, 64), dtype=np.complex64)
    kwargs = {
        "window_az": 16,
        "window_rg": 16,
        "search_az": 2,
        "search_rg": 2,
        "n_az": 1,
        "n_rg": 1,
        "margin_rg": 16,
        "margin_az": 16,
        "executor": "torch",
        "device": "cpu",
    }

    def no_survivors(
        *_args: object, **_kwargs: object
    ) -> tuple[object, object, object]:
        return (
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        )

    monkeypatch.setattr(offsets_mod, "_torch_patch_ncc_batch", no_survivors)
    result = estimate_patch_amplitude_shift(samples, samples, **kwargs)
    assert result.n_valid == 0
    assert offsets_mod._TORCH_AMPCOR_RESERVED_BYTES == {}

    def failing_ncc(*_args: object, **_kwargs: object) -> tuple[object, object, object]:
        message = "synthetic NCC failure"
        raise RuntimeError(message)

    monkeypatch.setattr(offsets_mod, "_torch_patch_ncc_batch", failing_ncc)
    with pytest.raises(RuntimeError, match="synthetic NCC failure"):
        estimate_patch_amplitude_shift(samples, samples, **kwargs)
    assert offsets_mod._TORCH_AMPCOR_RESERVED_BYTES == {}


def test_ampcor_process_admission_rejects_busy_cuda_process() -> None:
    """A second process cannot admit the same physical CUDA lock."""
    from faninsar.processing.coreg import offsets as offsets_mod

    child_code = (
        "from faninsar.processing.coreg.offsets import "
        "_admit_torch_ampcor_process\n"
        "import time\n"
        "with _admit_torch_ampcor_process('cuda-uuid:test'):\n"
        "    print('ready', flush=True)\n"
        "    time.sleep(2)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with (
            pytest.raises(ValueError, match="admission is busy"),
            offsets_mod._admit_torch_ampcor_process("cuda-uuid:test"),
        ):
            pass
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_ampcor_process_admission_rejects_replaced_lock_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    """A lock symlink cannot redirect admission to an attacker-controlled file."""
    from pathlib import Path

    from faninsar.processing.coreg import offsets as offsets_mod

    lock_root = Path(tmp_path) / "ampcor-lock-root"
    lock_root.mkdir(mode=0o700)
    target = Path(tmp_path) / "outside.lock"
    target.touch(mode=0o600)
    (lock_root / "cuda-uuid-replaced.lock").symlink_to(target)
    monkeypatch.setattr(offsets_mod, "_TORCH_AMPCOR_LOCK_ROOT", lock_root)

    with (
        pytest.raises(OSError, match="symbolic link"),
        offsets_mod._admit_torch_ampcor_process("cuda-uuid-replaced"),
    ):
        pass


def test_ampcor_process_admission_rejects_symlinked_lock_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    """A symlink cannot redirect the private admission namespace."""
    from pathlib import Path

    from faninsar.processing.coreg import offsets as offsets_mod

    target_root = Path(tmp_path) / "target-lock-root"
    target_root.mkdir(mode=0o700)
    lock_root = Path(tmp_path) / "ampcor-lock-root"
    lock_root.symlink_to(target_root, target_is_directory=True)
    monkeypatch.setattr(offsets_mod, "_TORCH_AMPCOR_LOCK_ROOT", lock_root)

    with (
        pytest.raises(OSError, match=r"symbolic link|Too many levels|Not a directory"),
        offsets_mod._admit_torch_ampcor_process("cuda-uuid-root-link"),
    ):
        pass


def test_ampcor_cache_release_runs_when_synchronization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allocator cleanup still runs when device synchronization raises."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    cleanup_devices: list[object] = []

    def failing_synchronize(_device: object) -> None:
        message = "synthetic synchronization failure"
        raise RuntimeError(message)

    def record_cleanup(device: object) -> None:
        cleanup_devices.append(device)

    monkeypatch.setattr(offsets_mod, "_synchronize_torch_device", failing_synchronize)
    monkeypatch.setattr(
        offsets_mod,
        "_release_torch_device_cache",
        record_cleanup,
    )
    samples = np.ones((64, 64), dtype=np.complex64)
    with pytest.raises(RuntimeError, match="synthetic synchronization failure"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            window_az=16,
            window_rg=16,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_rg=16,
            margin_az=16,
            executor="torch",
            device="cpu",
        )
    assert cleanup_devices == [torch.device("cpu")]
    assert offsets_mod._TORCH_AMPCOR_RESERVED_BYTES == {}


def test_ampcor_synchronization_error_is_not_hidden_by_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary synchronization error survives a cleanup failure."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    monkeypatch.setattr(
        offsets_mod,
        "_synchronize_torch_device",
        lambda _device: (_ for _ in ()).throw(RuntimeError("primary sync failure")),
    )
    monkeypatch.setattr(
        offsets_mod,
        "_release_torch_device_cache",
        lambda _device: (_ for _ in ()).throw(RuntimeError("cleanup failure")),
    )
    samples = np.ones((64, 64), dtype=np.complex64)
    with pytest.raises(RuntimeError, match="primary sync failure"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            window_az=16,
            window_rg=16,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_rg=16,
            margin_az=16,
            executor="torch",
            device="cpu",
        )
    assert offsets_mod._TORCH_AMPCOR_RESERVED_BYTES == {}


def test_ampcor_drops_batch_tensors_before_allocator_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup observes no live batch tensors, even on a CPU-only host."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    tensor_refs: list[weakref.ReferenceType[object]] = []
    cleanup_states: list[bool] = []

    def fake_ncc(
        ref_windows: object,
        sec_searches: object,
        **_kwargs: object,
    ) -> tuple[object, object, object]:
        count = ref_windows.shape[0]
        outputs = (
            torch.zeros(count, dtype=torch.float64),
            torch.zeros(count, dtype=torch.float64),
            torch.ones(count, dtype=torch.float64),
        )
        tensor_refs.extend(
            weakref.ref(tensor) for tensor in (ref_windows, sec_searches, *outputs)
        )
        return outputs

    def inspect_cleanup(_device: object) -> None:
        gc.collect()
        cleanup_states.append(all(reference() is None for reference in tensor_refs))

    monkeypatch.setattr(offsets_mod, "_torch_patch_ncc_batch", fake_ncc)
    monkeypatch.setattr(offsets_mod, "_release_torch_device_cache", inspect_cleanup)
    samples = np.ones((64, 64), dtype=np.complex64)
    estimate_patch_amplitude_shift(
        samples,
        samples,
        window_az=16,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        margin_rg=16,
        margin_az=16,
        executor="torch",
        device="cpu",
    )

    assert cleanup_states == [True]
    assert offsets_mod._TORCH_AMPCOR_RESERVED_BYTES == {}


def test_ampcor_rejects_non_array_inputs_before_shape_access() -> None:
    """Invalid direct inputs use the project validation error."""
    with pytest.raises(InvalidProcessingStateError, match="NumPy arrays"):
        estimate_patch_amplitude_shift([[1.0]], [[1.0]])


@pytest.mark.parametrize(
    "samples",
    [
        np.ones((32, 48), dtype=np.int16),
        np.ones((32, 48), dtype=np.float16),
        np.ones((32, 48), dtype=np.bool_),
        np.ones((32, 48), dtype=object),
    ],
)
def test_ampcor_compatibility_casts_numeric_input_dtype(samples: np.ndarray) -> None:
    """The compatibility spelling narrows supported numeric inputs for Torch."""
    kwargs = {
        "window_az": 8,
        "window_rg": 16,
        "search_az": 2,
        "search_rg": 2,
        "n_az": 1,
        "n_rg": 1,
        "margin_rg": 16,
        "margin_az": 8,
        "executor": "torch",
        "device": "cpu",
    }
    if samples.dtype.kind == "O":
        with pytest.raises(InvalidProcessingStateError, match=r"dtype .* unsupported"):
            estimate_patch_amplitude_shift(samples, samples, **kwargs)
        return
    result = estimate_patch_amplitude_shift(samples, samples, **kwargs)
    assert result.n_attempted == 1


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (np.float64, np.finfo(np.float64).max),
        (np.complex128, complex(np.finfo(np.float64).max, 1.0)),
    ],
)
def test_torch_rejects_extreme_high_precision_inputs_after_safe_narrowing(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
    value: complex,
) -> None:
    """Torch rejects values that overflow compatibility narrowing."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    samples = np.full((64, 96), value, dtype=dtype)
    monkeypatch.setattr(
        offsets_mod,
        "_ampcor_magnitude_tile",
        lambda *_args, **_kwargs: pytest.fail("Torch dtype gate ran after tile work"),
    )
    with pytest.raises(InvalidProcessingStateError, match="outside the Torch range"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            window_az=8,
            window_rg=16,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_rg=16,
            margin_az=8,
            executor="torch",
            device="cpu",
        )


def test_ampcor_copies_negative_stride_inputs() -> None:
    """Ampcor copies a safe reversed view before materializing tiles."""
    samples = np.ones((32, 48), dtype=np.complex64)
    reversed_samples = samples[:, ::-1]
    result = estimate_patch_amplitude_shift(
        reversed_samples,
        reversed_samples,
        executor="torch",
        device="cpu",
        window_az=8,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        margin_rg=16,
        margin_az=8,
    )
    assert result.n_attempted == 1


def test_ampcor_numpy_compatibility_spelling_uses_torch_limits() -> None:
    """The NumPy spelling is compatibility-only and uses Torch admission."""
    samples = np.ones((64, 96), dtype=np.complex64)
    with pytest.raises(InvalidProcessingStateError, match="batch_size"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            window_az=8,
            window_rg=16,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_rg=16,
            margin_az=8,
            batch_size=0,
            max_workspace_bytes=0,
        )


@pytest.mark.parametrize(
    "device",
    ["cuda", "cuda:0", "cuda:00", "gpu", "mps", "metal", "tpu", "unknown"],
)
def test_ampcor_numpy_rejects_explicit_cuda_before_tiling(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """The NumPy spelling cannot bypass Torch accelerator admission."""
    from faninsar.processing.coreg import offsets as offsets_mod

    monkeypatch.setattr(
        offsets_mod,
        "_validate_ampcor_inputs",
        lambda *_args, **_kwargs: pytest.fail("input admission ran first"),
    )
    samples = np.ones((64, 96), dtype=np.complex64)
    with pytest.raises(
        (InvalidProcessingStateError, RuntimeError, ImportError),
        match=r"unsupported Ampcor device|qualified Ampcor|CUDA requested|Torch",
    ):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            executor="numpy",
            device=device,
            window_az=8,
            window_rg=16,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_rg=16,
            margin_az=8,
        )


@pytest.mark.parametrize(
    "samples",
    [
        np.ones((64, 96), dtype=np.int16),
        np.ones((64, 96), dtype=np.float64),
        np.ones((64, 96), dtype=np.complex128),
        np.ones((64, 96), dtype=np.complex64)[:, ::-1],
    ],
)
def test_ampcor_numpy_retains_dtype_and_stride_compatibility(
    samples: np.ndarray,
) -> None:
    """The default NumPy lane keeps historical input conversions intact."""
    result = estimate_patch_amplitude_shift(
        samples,
        samples,
        window_az=8,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        margin_rg=16,
        margin_az=8,
    )
    assert result.n_attempted == 1


def test_ampcor_torch_copies_non_owning_c_contiguous_view() -> None:
    """Torch admission copies a safe C-contiguous view once."""
    samples = np.ones((64, 96), dtype=np.complex64)
    view = samples.view()
    assert view.flags.c_contiguous
    assert not view.flags.owndata

    result = estimate_patch_amplitude_shift(
        view,
        view,
        executor="torch",
        device="cpu",
        window_az=8,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        margin_rg=16,
        margin_az=8,
    )
    assert result.n_attempted == 1


def test_ampcor_repeated_shape_calls_release_cache_and_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing valid FFT shapes cannot accumulate Ampcor reservations."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    cleanup_devices: list[str] = []
    original_cleanup = offsets_mod._release_torch_device_cache

    def recording_cleanup(device: object) -> None:
        cleanup_devices.append(str(device))
        original_cleanup(device)

    monkeypatch.setattr(
        offsets_mod,
        "_release_torch_device_cache",
        recording_cleanup,
    )
    cases = (
        ((64, 96), (8, 8, 2, 2, 8)),
        ((80, 112), (12, 16, 3, 2, 12)),
        ((96, 128), (16, 24, 4, 3, 16)),
    )
    for shape, (window_az, window_rg, search_az, search_rg, margin) in cases:
        samples = np.ones(shape, dtype=np.complex64)
        estimate_patch_amplitude_shift(
            samples,
            samples,
            window_az=window_az,
            window_rg=window_rg,
            search_az=search_az,
            search_rg=search_rg,
            n_az=1,
            n_rg=1,
            margin_rg=margin,
            margin_az=margin,
            executor="torch",
            device="cpu",
        )
        assert offsets_mod._TORCH_AMPCOR_RESERVED_BYTES == {}

    assert cleanup_devices == ["cpu"] * len(cases)


def test_torch_ampcor_admission_key_uses_physical_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logical CUDA ordinals sharing one UUID use one admission key."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(uuid="GPU-test"),
    )
    assert (
        offsets_mod._torch_ampcor_admission_key(torch.device("cuda:0"), torch)
        == "cuda-uuid:GPU-test"
    )
    assert (
        offsets_mod._torch_ampcor_admission_key(torch.device("cuda:1"), torch)
        == "cuda-uuid:GPU-test"
    )


def test_torch_ncc_tie_uses_first_flattened_peak() -> None:
    """A fully tied surface has the same deterministic first-peak rule."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    d_rg, d_az, snr = offsets_mod._torch_patch_ncc_batch(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 4, 4), dtype=torch.float32),
        search_az=1,
        search_rg=1,
        subpixel=False,
    )
    assert float(d_rg[0]) == -1.0
    assert float(d_az[0]) == -1.0
    assert np.isnan(float(snr[0]))


def test_torch_ncc_integral_energy_matches_direct_oracle() -> None:
    """Torch NCC uses direct float64 rectangular local-energy sums."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    reference = torch.tensor(
        [[[1.0, -2.0, 0.5], [3.0, 4.0, -1.5]]], dtype=torch.float32
    )
    secondary = torch.tensor(
        [
            [0.5, -1.0, 2.0, 1.5, -0.25],
            [3.0, 0.25, -2.0, 4.0, 1.0],
            [-1.5, 2.5, 0.0, -0.5, 3.5],
            [2.0, -3.0, 1.25, 0.75, -2.5],
        ],
        dtype=torch.float32,
    )[None]
    search_az = 1
    search_rg = 1
    d_rg, d_az, snr = offsets_mod._torch_patch_ncc_batch(
        reference,
        secondary,
        search_az=search_az,
        search_rg=search_rg,
        subpixel=False,
    )

    ref = reference.to(dtype=torch.float64)
    sec = secondary.to(dtype=torch.float64)
    ref = ref - ref.mean(dim=(-2, -1), keepdim=True)
    sec = sec - sec.mean(dim=(-2, -1), keepdim=True)
    ref = ref / torch.linalg.vector_norm(ref, dim=(-2, -1), keepdim=True)
    expected_ncc = torch.empty((3, 3), dtype=torch.float64)
    for az in range(3):
        for rg in range(3):
            window = sec[0, az : az + 2, rg : rg + 3]
            expected_ncc[az, rg] = (ref[0] * window).sum() / torch.sqrt(
                (window * window).sum()
            )
    peak = torch.argmax(expected_ncc.reshape(-1))
    peak_az, peak_rg = np.unravel_index(int(peak), expected_ncc.shape)
    sidelobe = expected_ncc.clone()
    sidelobe[
        max(peak_az - 1, 0) : min(peak_az + 2, 3),
        max(peak_rg - 1, 0) : min(peak_rg + 2, 3),
    ] = torch.nan
    expected_snr = expected_ncc[peak_az, peak_rg] / torch.nanmean(sidelobe.abs())

    assert float(d_rg[0]) == peak_rg - search_rg
    assert float(d_az[0]) == peak_az - search_az
    assert float(snr[0]) == pytest.approx(float(expected_snr), abs=1e-12)


def test_torch_ncc_keeps_only_correlation_fft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safe local energy uses only the two correlation FFT transforms."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    calls = 0
    original_rfft2 = torch.fft.rfft2

    def count_rfft2(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_rfft2(*args, **kwargs)

    monkeypatch.setattr(torch.fft, "rfft2", count_rfft2)
    offsets_mod._torch_patch_ncc_batch(
        torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        torch.arange(20, dtype=torch.float32).reshape(1, 4, 5),
        search_az=1,
        search_rg=1,
        subpixel=False,
    )
    assert calls == 2


def test_torch_ncc_risky_energy_uses_fft_fallback_for_whole_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high-dynamic-range chip uses the legacy FFT energy for every lane."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    reference = torch.tensor(
        [[[1.0, -2.0, 0.5], [3.0, 4.0, -1.5]]], dtype=torch.float32
    )
    secondary = torch.tensor(
        [
            [1.0e8, -1.0e8, 2.0e8, 1.5e8, -2.5e7],
            [3.0e8, 2.5e7, -2.0e8, 4.0e8, 1.0e8],
            [-1.5e8, 2.5e8, 0.0, -5.0e7, 3.5e8],
            [2.0e8, -3.0e8, 1.25e8, 7.5e7, -2.5e8],
        ],
        dtype=torch.float32,
    )[None]

    calls = 0
    original_rfft2 = torch.fft.rfft2

    def count_rfft2(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_rfft2(*args, **kwargs)

    monkeypatch.setattr(torch.fft, "rfft2", count_rfft2)
    actual = offsets_mod._torch_patch_ncc_batch(
        reference,
        secondary,
        search_az=1,
        search_rg=1,
        subpixel=False,
    )
    assert calls == 4
    assert all(torch.isfinite(value).all() for value in actual[:2])

    # Compare against the same-device FFT-energy oracle explicitly.  This
    # also verifies that the batch-level decision does not mix energy paths.
    monkeypatch.setattr(
        offsets_mod,
        "_torch_integral_energy_is_safe",
        lambda *_: None,
    )
    expected = offsets_mod._torch_patch_ncc_batch(
        reference,
        secondary,
        search_az=1,
        search_rg=1,
        subpixel=False,
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, equal_nan=True)


def test_torch_ncc_integral_output_guard_handles_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiny local energy remains stable when large values cancel nearby."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    reference = torch.tensor(
        [[[1.0, -2.0, 0.5], [3.0, 4.0, -1.5]]], dtype=torch.float32
    )
    secondary = torch.full((1, 4, 5), 1.0e-3, dtype=torch.float32)
    secondary[0, 0, 4] = 5.0e5
    secondary[0, 3, 0] = -5.0e5

    calls = 0
    original_rfft2 = torch.fft.rfft2

    def count_rfft2(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_rfft2(*args, **kwargs)

    monkeypatch.setattr(torch.fft, "rfft2", count_rfft2)
    actual = offsets_mod._torch_patch_ncc_batch(
        reference,
        secondary,
        search_az=1,
        search_rg=1,
        subpixel=False,
    )
    assert calls == 4
    assert all(torch.isfinite(value).all() for value in actual[:2])

    monkeypatch.setattr(
        offsets_mod,
        "_torch_integral_energy_is_safe",
        lambda *_: None,
    )
    expected = offsets_mod._torch_patch_ncc_batch(
        reference,
        secondary,
        search_az=1,
        search_rg=1,
        subpixel=False,
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, equal_nan=True)


def test_ampcor_policy_routes_compatibility_spellings_to_torch() -> None:
    """The shared resolver gives every CPU spelling the Torch lane."""
    from faninsar.processing.coreg import resolve_ampcor_policy

    assert resolve_ampcor_policy("torch", "gpu") == ("torch", "cuda")
    assert resolve_ampcor_policy("torch", "cuda:00") == ("torch", "cuda:0")
    assert resolve_ampcor_policy("torch", "auto") == ("torch", "cpu")
    assert resolve_ampcor_policy("torch", "cpu") == ("torch", "cpu")
    assert resolve_ampcor_policy("numpy", "cpu") == ("torch", "cpu")
    assert resolve_ampcor_policy("numpy", "auto") == ("torch", "cpu")
    assert resolve_ampcor_policy("auto", "auto") == ("torch", "cpu")
    assert resolve_ampcor_policy("numpy", "cuda") == ("torch", "cuda")
    with pytest.raises(InvalidProcessingStateError, match="not a qualified Ampcor"):
        resolve_ampcor_policy("torch", "mps")


@pytest.mark.parametrize("executor", [None, [], "jax"])
def test_ampcor_direct_api_rejects_malformed_executor(executor: object) -> None:
    """Malformed executor values fail through the shared resolver."""
    with pytest.raises(
        InvalidProcessingStateError,
        match="unsupported Ampcor executor",
    ):
        estimate_patch_amplitude_shift(
            np.ones((32, 48), dtype=np.complex64),
            np.ones((32, 48), dtype=np.complex64),
            executor=executor,  # type: ignore[arg-type]
        )


def test_ampcor_direct_torch_auto_uses_torch_cpu_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct ``torch/auto`` never selects the host NumPy solver."""
    from faninsar.processing.coreg import offsets as offsets_mod

    torch = pytest.importorskip("torch")

    monkeypatch.setattr(
        offsets_mod,
        "_torch_patch_ncc_batch",
        lambda ref, *_args, **_kwargs: (
            torch.zeros(ref.shape[0], dtype=torch.float64),
            torch.zeros(ref.shape[0], dtype=torch.float64),
            torch.full((ref.shape[0],), 10.0, dtype=torch.float64),
        ),
    )
    monkeypatch.setattr(
        offsets_mod,
        "_patch_ncc_shift",
        lambda *_args, **_kwargs: pytest.fail("Torch auto selected host NumPy"),
    )
    result = estimate_patch_amplitude_shift(
        np.ones((64, 96), dtype=np.complex64),
        np.ones((64, 96), dtype=np.complex64),
        executor="torch",
        device="auto",
        window_az=8,
        window_rg=16,
        search_az=2,
        search_rg=2,
        n_az=1,
        n_rg=1,
        margin_rg=16,
        margin_az=8,
    )
    assert result.n_valid == 1


def test_ampcor_seed_123_boundary_is_parity_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed-123 boundary values keep strict Torch culling semantics."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    rng = np.random.default_rng(123)
    snr_threshold = np.float64(rng.uniform(4.0, 6.0))
    snr_below = snr_threshold
    for _ in range(32):
        snr_below = np.nextafter(snr_below, 0.0)
    samples = np.ones((64, 96), dtype=np.complex64)
    kwargs = {
        "window_az": 8,
        "window_rg": 16,
        "search_az": 2,
        "search_rg": 2,
        "n_az": 1,
        "n_rg": 2,
        "margin_rg": 16,
        "margin_az": 8,
        "snr_threshold": float(snr_threshold),
        "max_abs_residual": 1.2,
    }
    monkeypatch.setattr(
        offsets_mod,
        "_torch_patch_ncc_batch",
        lambda *_args, **_kwargs: (
            torch.tensor([0.25, 0.5], dtype=torch.float64),
            torch.tensor([-0.25, -0.5], dtype=torch.float64),
            torch.tensor([snr_threshold, snr_below], dtype=torch.float64),
        ),
    )
    monkeypatch.setattr(
        offsets_mod,
        "_patch_ncc_shift",
        lambda *_args, **_kwargs: pytest.fail("Torch boundary path used NumPy"),
    )
    result = estimate_patch_amplitude_shift(
        samples,
        samples,
        executor="auto",
        device="auto",
        batch_size=2,
        **kwargs,
    )
    assert result.n_valid == 1
    assert result.range_shift_px == 0.25
    assert result.azimuth_shift_px == -0.25


def test_ampcor_boundary_oracle_stabilizes_backend_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-boundary Torch drift is recomputed by the Torch FFT reference."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    threshold = np.float64(5.0)
    snr_drift = threshold
    for _ in range(8):
        snr_drift = np.nextafter(snr_drift, np.inf)
    samples = np.ones((64, 96), dtype=np.complex64)
    kwargs = {
        "window_az": 8,
        "window_rg": 16,
        "search_az": 2,
        "search_rg": 2,
        "n_az": 1,
        "n_rg": 1,
        "margin_rg": 16,
        "margin_az": 8,
        "snr_threshold": float(threshold),
        "max_abs_residual": 1.2,
    }
    monkeypatch.setattr(
        offsets_mod,
        "_torch_patch_ncc_batch",
        lambda *_args, **kwargs: (
            torch.tensor([0.25], dtype=torch.float64),
            torch.tensor([-0.25], dtype=torch.float64),
            torch.tensor(
                [threshold if kwargs.get("force_fft_energy") else snr_drift],
                dtype=torch.float64,
            ),
        ),
    )
    monkeypatch.setattr(
        offsets_mod,
        "_patch_ncc_shift",
        lambda *_args, **_kwargs: pytest.fail("Torch boundary path used NumPy"),
    )
    result = estimate_patch_amplitude_shift(
        samples,
        samples,
        executor="torch",
        device="cpu",
        **kwargs,
    )
    assert result.n_valid == 1
    assert result.range_shift_px == 0.25
    assert result.azimuth_shift_px == -0.25


def test_ampcor_cuda_rejects_unqualified_fft_shape_before_kernel() -> None:
    """An out-of-window FFT shape cannot bypass the qualified CUDA lane."""
    from faninsar.processing.coreg import offsets as offsets_mod

    with pytest.raises(InvalidProcessingStateError, match="FFT shape"):
        offsets_mod._validate_torch_ampcor_shape(
            window_az=8,
            window_rg=16,
            search_az=2,
            search_rg=2,
            device_type="cuda",
        )
    offsets_mod._validate_torch_ampcor_shape(
        window_az=8,
        window_rg=16,
        search_az=2,
        search_rg=2,
        device_type="cpu",
    )


@pytest.mark.parametrize(
    ("snr_threshold", "max_abs_residual", "values"),
    [
        (
            0.0,
            1.2,
            [
                (0.0, 0.0, np.nextafter(0.0, -1.0)),
                (0.0, 0.0, 0.0),
            ],
        ),
        (
            0.5,
            0.0,
            [
                (np.nextafter(0.0, 1.0), 0.0, 1.0),
                (0.0, 0.0, 1.0),
            ],
        ),
    ],
)
def test_ampcor_zero_boundaries_do_not_widen_cull(
    monkeypatch: pytest.MonkeyPatch,
    snr_threshold: float,
    max_abs_residual: float,
    values: list[tuple[float, float, float]],
) -> None:
    """Zero thresholds retain exact ties and reject nonzero subnormals."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets as offsets_mod

    samples = np.ones((64, 96), dtype=np.complex64)
    kwargs = {
        "window_az": 8,
        "window_rg": 16,
        "search_az": 2,
        "search_rg": 2,
        "n_az": 1,
        "n_rg": 2,
        "margin_rg": 16,
        "margin_az": 8,
        "snr_threshold": snr_threshold,
        "max_abs_residual": max_abs_residual,
    }
    monkeypatch.setattr(
        offsets_mod,
        "_torch_patch_ncc_batch",
        lambda *_args, **_kwargs: tuple(
            torch.tensor([row[index] for row in values], dtype=torch.float64)
            for index in range(3)
        ),
    )
    monkeypatch.setattr(
        offsets_mod,
        "_patch_ncc_shift",
        lambda *_args, **_kwargs: pytest.fail("Torch boundary path used NumPy"),
    )
    torch_result = estimate_patch_amplitude_shift(
        samples,
        samples,
        executor="auto",
        device="auto",
        batch_size=2,
        **kwargs,
    )
    assert torch_result.n_valid == 1


def test_refine_shift_rejects_unqualified_mps_before_secondary_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS is rejected before any input-derived pre-alignment allocation."""
    from faninsar.processing.coreg import geometry_coreg

    monkeypatch.setattr(
        geometry_coreg.np,
        "roll",
        lambda *_args, **_kwargs: pytest.fail("np.roll ran before MPS validation"),
    )
    samples = np.ones((32, 64), dtype=np.complex64)
    with pytest.raises(InvalidProcessingStateError, match="not a qualified Ampcor"):
        geometry_coreg.refine_shift_with_correlation(
            samples,
            samples,
            prior_rg=0.0,
            prior_az=0.0,
            executor="torch",
            device="mps",
        )


def test_refine_shift_rejects_dispatch_before_secondary_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contradictory requests fail before input-derived workspace allocation."""
    from faninsar.processing.coreg import geometry_coreg

    monkeypatch.setattr(
        geometry_coreg.np,
        "roll",
        lambda *_args, **_kwargs: pytest.fail("np.roll ran before policy validation"),
    )
    samples = np.ones((32, 64), dtype=np.complex64)
    with pytest.raises((InvalidProcessingStateError, RuntimeError, ImportError)):
        geometry_coreg.refine_shift_with_correlation(
            samples,
            samples,
            prior_rg=0.0,
            prior_az=0.0,
            executor="numpy",
            device="cuda:0",
        )


def test_refine_shift_rejects_unavailable_cuda_before_secondary_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable explicit CUDA fails before copying a large secondary."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("host has CUDA; unavailable-device branch is not applicable")
    from faninsar.processing.coreg import geometry_coreg

    monkeypatch.setattr(
        geometry_coreg.np,
        "roll",
        lambda *_args, **_kwargs: pytest.fail("np.roll ran before CUDA validation"),
    )
    samples = np.ones((32, 64), dtype=np.complex64)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        geometry_coreg.refine_shift_with_correlation(
            samples,
            samples,
            prior_rg=0.0,
            prior_az=0.0,
            executor="torch",
            device="cuda",
        )


def test_estimate_patch_amplitude_shift_cuda_is_fail_closed() -> None:
    """Explicit CUDA never silently falls back when no CUDA device exists."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("host has CUDA; unavailable-device branch is not applicable")
    with pytest.raises(RuntimeError, match="CUDA requested"):
        estimate_patch_amplitude_shift(
            np.ones((64, 96), dtype=np.complex64),
            np.ones((64, 96), dtype=np.complex64),
            executor="torch",
            device="cuda:0",
        )


def test_estimate_patch_amplitude_shift_rejects_unqualified_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA Ampcor fails closed outside the exact qualified runtime lane."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(name="RTX test", major=8, minor=6, uuid="test"),
    )
    with pytest.raises(InvalidProcessingStateError, match="outside the qualified"):
        estimate_patch_amplitude_shift(
            np.ones((64, 96), dtype=np.complex64),
            np.ones((64, 96), dtype=np.complex64),
            executor="torch",
            device="cuda:0",
        )


def test_refine_shift_with_correlation_resolves_cuda_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CUDA selects the Torch Ampcor executor at the production seam."""
    from faninsar.processing.coreg import geometry_coreg

    captured: dict[str, object] = {}

    def fake_patch(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            range_shift_px=0.0,
            azimuth_shift_px=0.0,
            n_valid=1,
            snr_median=10.0,
        )

    monkeypatch.setattr(geometry_coreg, "estimate_patch_amplitude_shift", fake_patch)
    monkeypatch.setattr(
        geometry_coreg, "_validate_ampcor_accelerator", lambda _device: None
    )
    result = geometry_coreg.refine_shift_with_correlation(
        np.ones((32, 64), dtype=np.complex64),
        np.ones((32, 64), dtype=np.complex64),
        prior_rg=0.0,
        prior_az=0.0,
        executor="torch",
        device="cuda",
    )
    assert result == (0.0, 0.0)
    assert captured["executor"] == "torch"
    assert captured["device"] == "cuda"
    captured.clear()
    geometry_coreg.refine_shift_with_correlation(
        np.ones((32, 64), dtype=np.complex64),
        np.ones((32, 64), dtype=np.complex64),
        prior_rg=0.0,
        prior_az=0.0,
        executor="torch",
        device="auto",
    )
    assert captured["executor"] == "torch"
    assert captured["device"] == "cpu"


def test_refine_peak_subpixel_parabolic_recovery() -> None:
    """Parabolic refinement recovers the true peak on a synthetic parabola."""
    x = np.arange(-2, 3)
    y = -((x - 0.3) ** 2)  # peak at +0.3
    sub = refine_peak_subpixel(y[:, None], (2, 0))[0]
    assert sub == pytest.approx(0.3, abs=1e-3)


def test_resample_complex_applies_constant_offset() -> None:
    """Resampling with a constant offset moves an impulse to the expected pixel."""
    samples = np.zeros((16, 16), dtype=np.complex64)
    samples[8, 8] = 1.0 + 2.0j
    offsets = geometry_shift_offsets(
        samples.shape,
        range_shift_px=2.0,
        azimuth_shift_px=-1.0,
    )
    # source = output - offset => impulse at (8,8) appears at output (7,10)
    out = resample_complex(
        samples,
        range_offset_px=offsets.range_offset_px,
        azimuth_offset_px=offsets.azimuth_offset_px,
        order=0,
    )
    peak = np.unravel_index(int(np.argmax(np.abs(out))), out.shape)
    assert peak == (7, 10)


def test_combine_offset_fields_adds_residuals() -> None:
    """Combined field adds ESD and amplitude residuals to geometry."""
    geometry = geometry_shift_offsets(
        (8, 8),
        range_shift_px=1.0,
        azimuth_shift_px=2.0,
    )
    combined = combine_offset_fields(
        geometry,
        esd_azimuth_shift_px=0.3,
        amplitude_residual_rg=0.1,
        amplitude_residual_az=-0.2,
    )
    assert np.allclose(combined.range_offset_px, 1.1)
    assert np.allclose(combined.azimuth_offset_px, 2.1)
    assert combined.coverage.all()
    assert np.all(combined.uncertainty_px >= geometry.uncertainty_px)

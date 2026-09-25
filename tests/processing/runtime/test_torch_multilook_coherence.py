"""NumPy/Torch parity and edge tests for windowed multilook coherence."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.interferometry.pair import form_interferogram
from faninsar.processing.runtime.torch_kernels import multilook_interferogram_torch


def _scenes(
    shape: tuple[int, int], coherence: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    primary = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    secondary = (coherence * primary + np.sqrt(1.0 - coherence**2) * noise).astype(
        np.complex64
    )
    return primary, secondary


def _torch_product(
    primary: np.ndarray,
    secondary: np.ndarray,
    multilook: tuple[int, int],
    window: tuple[int, int] | None,
    precision: str = "float64",
) -> object:
    return multilook_interferogram_torch(
        primary,
        secondary,
        multilook=multilook,
        coherence_window=window,
        device="cpu",
        precision=precision,
    )


@pytest.mark.parametrize(
    ("multilook", "window", "shape"),
    [
        ((1, 1), (5, 5), (37, 45)),
        ((5, 2), (5, 5), (64, 44)),
        ((3, 4), (3, 7), (41, 53)),
    ],
)
def test_torch_windowed_coherence_matches_numpy_reference(
    multilook: tuple[int, int],
    window: tuple[int, int],
    shape: tuple[int, int],
) -> None:
    """Windowed Torch coherence equals the NumPy reference numerically."""
    primary, secondary = _scenes(shape, coherence=0.6, seed=3)
    primary[5, 7] = np.nan
    secondary[9, 3] = np.inf + 0j
    primary[20:23, 30:33] = 0.0

    reference = form_interferogram(
        primary, secondary, multilook=multilook, coherence_window=window
    )
    product = _torch_product(primary, secondary, multilook, window)

    reference_coh = np.asarray(reference.coherence, dtype=np.float64)
    torch_coh = np.asarray(product.coherence, dtype=np.float64)
    np.testing.assert_allclose(
        torch_coh, reference_coh, rtol=1e-6, atol=1e-9, equal_nan=True
    )
    np.testing.assert_array_equal(np.isnan(torch_coh), np.isnan(reference_coh))
    np.testing.assert_array_equal(product.valid_mask, reference.valid_mask)
    np.testing.assert_allclose(
        product.complex_ifg,
        reference.complex_ifg,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_torch_direct_block_mle_parity_unchanged() -> None:
    """The default (window=None) lane keeps its prior NumPy parity."""
    primary, secondary = _scenes((40, 50), coherence=0.7, seed=11)
    reference = form_interferogram(primary, secondary, multilook=(5, 2))
    product = _torch_product(primary, secondary, (5, 2), None)
    np.testing.assert_allclose(
        product.coherence, reference.coherence, rtol=1e-6, equal_nan=True
    )


def test_single_sample_blocks_have_nan_windowed_coherence() -> None:
    """An isolated powered pixel keeps complex support in both lanes.

    Zero-valued neighbours are finite samples, so the window MLE around the
    single powered pixel estimates exactly 1.0 while every unsupported pixel
    stays NaN; the two implementations agree point for point.
    """
    primary = np.zeros((16, 16), dtype=np.complex64)
    secondary = np.zeros((16, 16), dtype=np.complex64)
    primary[8, 8] = 1 + 1j
    secondary[8, 8] = 1 + 0.5j

    reference = form_interferogram(
        primary, secondary, multilook=(1, 1), coherence_window=(5, 5)
    )
    product = _torch_product(primary, secondary, (1, 1), (5, 5))

    reference_coh = np.asarray(reference.coherence)
    torch_coh = np.asarray(product.coherence)
    np.testing.assert_array_equal(np.isnan(torch_coh), np.isnan(reference_coh))
    finite = np.isfinite(torch_coh)
    np.testing.assert_allclose(torch_coh[finite], reference_coh[finite], rtol=1e-6)
    assert torch_coh[8, 8] == pytest.approx(1.0, abs=1e-6)
    assert np.array_equal(product.valid_mask, reference.valid_mask)


def test_all_invalid_inputs_match_reference() -> None:
    """All-NaN scenes stay all-invalid on both implementations."""
    shape = (12, 14)
    primary = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex64)
    secondary = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex64)

    reference = form_interferogram(
        primary, secondary, multilook=(5, 2), coherence_window=(3, 3)
    )
    product = _torch_product(primary, secondary, (5, 2), (3, 3))

    assert not product.valid_mask.any()
    assert np.isnan(np.asarray(product.coherence)).all()
    assert not reference.valid_mask.any()


def test_window_larger_than_array_matches_reference() -> None:
    """Windows wider than the scene clip identically on both paths."""
    primary, secondary = _scenes((6, 8), coherence=0.8, seed=5)
    reference = form_interferogram(
        primary, secondary, multilook=(2, 2), coherence_window=(9, 9)
    )
    product = _torch_product(primary, secondary, (2, 2), (9, 9))
    np.testing.assert_allclose(
        product.coherence, reference.coherence, rtol=1e-6, equal_nan=True
    )


def test_default_precision_windowed_coherence_close_to_reference() -> None:
    """The default float32 lane stays within tolerance of the reference."""
    primary, secondary = _scenes((40, 50), coherence=0.5, seed=9)
    reference = form_interferogram(
        primary, secondary, multilook=(5, 2), coherence_window=(5, 5)
    )
    product = _torch_product(primary, secondary, (5, 2), (5, 5), precision="auto")
    np.testing.assert_allclose(
        product.coherence,
        reference.coherence,
        rtol=2e-2,
        atol=2e-3,
        equal_nan=True,
    )

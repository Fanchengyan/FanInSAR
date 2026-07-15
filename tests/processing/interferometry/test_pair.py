"""Tests for complex interferogram formation and filtering."""

from __future__ import annotations

import numpy as np

from faninsar.processing.interferometry import form_interferogram, goldstein_filter


def test_identical_slcs_have_unit_coherence_and_zero_phase() -> None:
    """Identical complex SLCs yield coherence ~1 and near-zero phase."""
    rng = np.random.default_rng(2)
    slc = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))).astype(
        np.complex64
    )
    product = form_interferogram(slc, slc, multilook=(2, 2))
    assert float(np.nanmean(product.coherence)) > 0.99
    assert float(np.nanmax(np.abs(product.wrapped_phase))) < 1e-5


def test_goldstein_filter_preserves_shape_and_complex_dtype() -> None:
    """Goldstein filter returns same shape complex output."""
    rng = np.random.default_rng(3)
    ifg = (rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))).astype(
        np.complex64
    )
    filtered = goldstein_filter(ifg, alpha=0.5, window=16)
    assert filtered.shape == ifg.shape
    assert np.iscomplexobj(filtered)

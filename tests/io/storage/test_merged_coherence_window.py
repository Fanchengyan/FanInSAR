"""Regression: merged IFG coherence honours the documented window semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.io.storage.scene_store import form_merged_scene_interferogram
from faninsar.processing.interferometry.pair import _sliding_coherence
from tests.support.scene_stores import open_test_scene, write_test_scene

if TYPE_CHECKING:
    from pathlib import Path


def test_merged_coherence_window_is_local_mle_not_single_sample(
    tmp_path: Path,
) -> None:
    """Under (1, 1) looks the windowed MLE must not saturate at 1.0."""
    rng = np.random.default_rng(7)
    shape = (24, 30)
    reference = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    noise = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    secondary = (reference + 1.5 * noise).astype(np.complex64)
    first = write_test_scene(
        tmp_path,
        date_id="20240101",
        reference_id="20240101",
        payload=reference,
        tag="IW1_b0",
    )
    second = write_test_scene(
        tmp_path,
        date_id="20240113",
        reference_id="20240101",
        payload=secondary,
        reference_payload=reference,
        tag="IW1_b0",
    )
    product = form_merged_scene_interferogram(
        open_test_scene(first),
        open_test_scene(second),
        multilook=(1, 1),
        coherence_window=(5, 5),
    )
    expected = _sliding_coherence(
        reference.astype(np.complex128), secondary.astype(np.complex128), (5, 5)
    )
    got = np.asarray(product.coherence)
    assert got.shape == shape
    assert float(np.nanmax(got)) < 1.0
    mask = np.isfinite(got) & np.isfinite(expected)
    np.testing.assert_allclose(got[mask], expected[mask], rtol=1e-4, atol=1e-4)

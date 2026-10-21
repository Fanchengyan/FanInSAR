"""Stack IFG Dataset seam over committed IFG products."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.data.datasets.ifg import StackInterferogramDataset
from faninsar.io.storage.ifg_store import write_ifg_artifact

if TYPE_CHECKING:
    from pathlib import Path

_SOURCES = {"primary": "a" * 64, "secondary": "b" * 64}


def test_stack_dataset_reads_named_fields_from_product(tmp_path: Path) -> None:
    """Dataset construction reads the named layers of an IFG product."""
    store = write_ifg_artifact(
        tmp_path / "ifg",
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests=_SOURCES,
        complex_ifg=np.ones((2, 3), dtype=np.complex64),
        coherence=np.full((2, 3), 0.5, dtype=np.float32),
        wrapped_phase=np.zeros((2, 3), dtype=np.float32),
        amplitude=np.ones((2, 3), dtype=np.float32),
    )
    dataset = StackInterferogramDataset.from_product(store)

    assert dataset.complex_ifg.shape == (2, 3)
    assert dataset.coherence.shape == (2, 3)
    assert dataset.wrapped_phase.shape == (2, 3)
    assert dataset.amplitude.shape == (2, 3)
    assert dataset.valid_mask.dtype == np.bool_
    np.testing.assert_array_equal(dataset.valid_mask, np.ones((2, 3), dtype=bool))


def test_stack_dataset_preserves_missing_optional_coherence(tmp_path: Path) -> None:
    """An IFG product may intentionally omit the coherence layer."""
    store = write_ifg_artifact(
        tmp_path / "ifg_without_coherence",
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests=_SOURCES,
        complex_ifg=np.ones((2, 3), dtype=np.complex64),
        coherence=None,
        wrapped_phase=np.zeros((2, 3), dtype=np.float32),
        amplitude=np.ones((2, 3), dtype=np.float32),
    )
    dataset = StackInterferogramDataset.from_product(store)
    assert dataset.coherence is None
    np.testing.assert_array_equal(dataset.valid_mask, np.ones((2, 3), dtype=bool))
    assert store.read().coherence is None

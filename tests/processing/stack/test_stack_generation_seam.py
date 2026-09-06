"""Public Stack generation and Dataset seam tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.data.datasets.ifg import StackInterferogramDataset

if TYPE_CHECKING:
    from pathlib import Path
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.ifg_store import (
    InterferogramArtifactStore,
    write_ifg_artifact,
)
from faninsar.stack.stack_generation import (
    open_unwrap_generation,
    publish_unwrap_generation,
)


def _ifg_store(root: Path) -> InterferogramArtifactStore:
    """Publish one small IFG artifact for Dataset seam tests."""
    complex_ifg = np.ones((2, 3), dtype=np.complex64)
    return write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"reference": "a" * 64, "secondary": "b" * 64},
        complex_ifg=complex_ifg,
        coherence=np.full((2, 3), 0.5, dtype=np.float32),
        wrapped_phase=np.zeros((2, 3), dtype=np.float32),
        amplitude=np.ones((2, 3), dtype=np.float32),
    )


def test_stack_dataset_reads_named_fields_from_pinned_generation(
    tmp_path: Path,
) -> None:
    """Dataset construction consumes an already opened IFG generation."""
    store = _ifg_store(tmp_path / "ifg")
    dataset = StackInterferogramDataset.from_generation(store)

    assert dataset.complex_ifg.shape == (2, 3)
    assert dataset.coherence.shape == (2, 3)
    assert dataset.wrapped_phase.shape == (2, 3)
    assert dataset.amplitude.shape == (2, 3)
    assert dataset.valid_mask.dtype == np.bool_
    np.testing.assert_array_equal(dataset.valid_mask, np.ones((2, 3), dtype=bool))
    store.close()


def test_stack_dataset_preserves_missing_optional_coherence(
    tmp_path: Path,
) -> None:
    """A pinned generation may intentionally omit the coherence layer."""
    complex_ifg = np.ones((2, 3), dtype=np.complex64)
    store = write_ifg_artifact(
        tmp_path / "ifg_without_coherence",
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"reference": "a" * 64, "secondary": "b" * 64},
        complex_ifg=complex_ifg,
        coherence=None,
        wrapped_phase=np.zeros((2, 3), dtype=np.float32),
        amplitude=np.ones((2, 3), dtype=np.float32),
    )
    try:
        dataset = StackInterferogramDataset.from_generation(store)
        assert dataset.coherence is None
        np.testing.assert_array_equal(dataset.valid_mask, np.ones((2, 3), dtype=bool))
        assert store.read().coherence is None
    finally:
        store.close()


def test_unwrap_generation_round_trip_is_one_complete_pair_snapshot(
    tmp_path: Path,
) -> None:
    """A root generation round-trips all Pair results atomically."""
    phase = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = publish_unwrap_generation(
        tmp_path / "stack",
        pair_ids=("20240101_20240113",),
        products={"20240101_20240113": {"unwrapped_phase": phase}},
        pair_bindings={
            "20240101_20240113": {
                "ifg_generation_id": "a" * 32,
                "ifg_manifest_digest": "b" * 64,
                "grid_identity": "c" * 64,
            }
        },
    )
    assert result.pair_ids == ("20240101_20240113",)
    result.close()

    with open_unwrap_generation(tmp_path / "stack") as opened:
        assert opened.pair_ids == ("20240101_20240113",)
        assert opened.products["20240101_20240113"]["unwrapped_phase"].shape == (
            2,
            3,
        )
        np.testing.assert_array_equal(
            opened.products["20240101_20240113"]["unwrapped_phase"], phase
        )


def test_unwrap_generation_rejects_missing_pair_result(tmp_path: Path) -> None:
    """The root snapshot cannot publish a partial expected Pair universe."""
    with pytest.raises(InvalidProcessingStateError):
        publish_unwrap_generation(
            tmp_path / "stack",
            pair_ids=("20240101_20240113", "20240113_20240125"),
            products={"20240101_20240113": {"unwrapped_phase": np.zeros((2, 2))}},
            pair_bindings={
                pair_id: {
                    "ifg_generation_id": "a" * 32,
                    "ifg_manifest_digest": "b" * 64,
                    "grid_identity": "c" * 64,
                }
                for pair_id in ("20240101_20240113", "20240113_20240125")
            },
        )


def test_unwrap_generation_requires_exact_source_bindings(tmp_path: Path) -> None:
    """A root unwrap snapshot cannot be published without IFG lineage."""
    with pytest.raises(InvalidProcessingStateError, match="bindings"):
        publish_unwrap_generation(
            tmp_path / "stack",
            pair_ids=("20240101_20240113",),
            products={"20240101_20240113": {"unwrapped_phase": np.zeros((2, 2))}},
            pair_bindings={},
        )

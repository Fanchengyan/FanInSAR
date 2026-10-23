"""Stack IFG Dataset seam over committed IFG products."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.data.datasets.ifg import StackInterferogramDataset
from faninsar.io.storage.ifg_store import write_ifg_artifact

if TYPE_CHECKING:
    from pathlib import Path

_SOURCES = {"primary": "a" * 64, "secondary": "b" * 64}


def _write_store(tmp_path: Path, name: str, **overrides: object) -> object:
    """Commit one small IFG product with persisted layers only."""
    kwargs: dict[str, object] = {
        "pair": ("20240101", "20240113"),
        "looks": (1, 1),
        "filter_name": "none",
        "filter_parameters": {},
        "source_manifest_digests": _SOURCES,
        "complex_ifg": np.ones((2, 3), dtype=np.complex64),
        "coherence": np.full((2, 3), 0.5, dtype=np.float32),
    }
    kwargs.update(overrides)
    return write_ifg_artifact(tmp_path / name, **kwargs)  # type: ignore[arg-type]


def test_stack_dataset_reads_named_fields_from_product(tmp_path: Path) -> None:
    """Dataset construction reads stored layers and derives virtual ones."""
    store = _write_store(tmp_path, "ifg")
    dataset = StackInterferogramDataset.from_product(store)

    assert dataset.complex_ifg.shape == (2, 3)
    assert dataset.coherence.shape == (2, 3)
    np.testing.assert_array_equal(
        dataset.wrapped_phase, np.zeros((2, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        dataset.amplitude, np.ones((2, 3), dtype=np.float32)
    )
    assert dataset.wrapped_phase.dtype == np.float32
    assert dataset.amplitude.dtype == np.float32
    assert dataset.valid_mask.dtype == np.bool_
    np.testing.assert_array_equal(dataset.valid_mask, np.ones((2, 3), dtype=bool))
    # Derived layers never materialize their own stores.
    assert sorted(path.name for path in (tmp_path / "ifg").iterdir()) == [
        "coherence.zarr",
        "complex_ifg.zarr",
        "item.json",
    ]


def test_stack_dataset_preserves_missing_optional_coherence(tmp_path: Path) -> None:
    """An IFG product may intentionally omit the coherence layer."""
    store = _write_store(tmp_path, "ifg_without_coherence", coherence=None)
    dataset = StackInterferogramDataset.from_product(store)
    assert dataset.coherence is None
    np.testing.assert_array_equal(dataset.valid_mask, np.ones((2, 3), dtype=bool))
    assert store.read().coherence is None
    assert sorted(path.name for path in (tmp_path / "ifg_without_coherence").iterdir()) == [
        "complex_ifg.zarr",
        "item.json",
    ]


def test_stack_dataset_uses_persisted_mask_for_derived_nan(tmp_path: Path) -> None:
    """A persisted ``valid_mask`` is authoritative and NaN-masks derived layers."""
    complex_ifg = np.full((2, 3), 1.0 + 1.0j, dtype=np.complex64)
    valid_mask = np.ones((2, 3), dtype=bool)
    valid_mask[0, 1] = False
    store = _write_store(
        tmp_path,
        "ifg_masked",
        complex_ifg=complex_ifg,
        valid_mask=valid_mask,
    )
    dataset = StackInterferogramDataset.from_product(store)

    np.testing.assert_array_equal(dataset.valid_mask, valid_mask)
    assert np.isnan(dataset.wrapped_phase[0, 1])
    assert np.isnan(dataset.amplitude[0, 1])
    assert np.isfinite(dataset.wrapped_phase[0, 0])
    assert np.isfinite(dataset.amplitude[0, 0])
    np.testing.assert_allclose(
        dataset.wrapped_phase[0, 0], np.angle(np.complex64(1.0 + 1.0j))
    )
    np.testing.assert_allclose(
        dataset.amplitude[0, 0], np.abs(np.complex64(1.0 + 1.0j))
    )


def test_stack_dataset_delegates_failures_to_the_store(tmp_path: Path) -> None:
    """Damaged products surface ``ValueError``; non-stores surface ``TypeError``."""
    import zarr

    store = _write_store(tmp_path, "ifg_damaged")
    zarr.open_array(
        store=str(tmp_path / "ifg_damaged" / "coherence.zarr"),
        mode="w",
        shape=(1, 1),
        dtype=np.float32,
    )
    with pytest.raises(ValueError):
        StackInterferogramDataset.from_product(store)
    with pytest.raises(TypeError):
        StackInterferogramDataset.from_product(object())

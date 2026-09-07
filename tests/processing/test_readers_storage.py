"""Tests for mission-neutral SLC readers and storage adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.arrays import (
    ChunkAccessLog,
    ChunkedZarrArrayStore,
    InMemoryArrayStore,
)
from faninsar.missions.protocols import MissingCriticalMetadataError
from faninsar.missions.s1.synthetic_slc import (
    SyntheticSLCReader,
    register_specs,
    write_synthetic_slc,
)

if TYPE_CHECKING:
    from pathlib import Path


def _samples(shape: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Build a deterministic complex array with unique phase per pixel."""
    rows = np.arange(shape[0], dtype=np.float32)[:, None]
    cols = np.arange(shape[1], dtype=np.float32)[None, :]
    return (rows + 1j * cols).astype(np.complex64)


def test_inmemory_and_zarr_readers_return_identical_products(tmp_path: Path) -> None:
    """Return identical typed products from in-memory and chunked Zarr stores."""
    samples = _samples()
    memory = InMemoryArrayStore()
    zarr_store = ChunkedZarrArrayStore(root=tmp_path / "zarr", chunks=(4, 4))

    memory_spec = write_synthetic_slc(
        memory,
        source_path="memory://scene-a",
        acquisition_id="scene-a",
        samples=samples,
    )
    zarr_spec = write_synthetic_slc(
        zarr_store,
        source_path="scene-a",
        acquisition_id="scene-a",
        samples=samples,
    )

    memory_reader = SyntheticSLCReader(
        store=memory,
        specs=register_specs(memory_spec),
    )
    zarr_reader = SyntheticSLCReader(
        store=zarr_store,
        specs=register_specs(zarr_spec),
    )

    memory_result = memory_reader.open("memory://scene-a")
    zarr_result = zarr_reader.open("scene-a")

    assert memory_result.product.acquisition_id == zarr_result.product.acquisition_id
    assert memory_result.product.grid.shape == zarr_result.product.grid.shape
    assert memory_result.doppler_centroid == zarr_result.doppler_centroid
    assert memory_result.valid_samples == zarr_result.valid_samples
    assert memory_result.mission_native == zarr_result.mission_native

    selection = (slice(1, 5), slice(2, 6))
    memory_block = memory.read(memory_result.product.samples, selection)
    zarr_block = zarr_store.read(zarr_result.product.samples, selection)
    np.testing.assert_array_equal(memory_block, zarr_block)
    np.testing.assert_array_equal(memory_block, samples[1:5, 2:6])


def test_zarr_lazy_slice_reads_only_requested_chunks(tmp_path: Path) -> None:
    """Touch only the Zarr chunks that overlap the requested selection."""
    samples = _samples((16, 16))
    access_log = ChunkAccessLog()
    store = ChunkedZarrArrayStore(
        root=tmp_path / "zarr",
        chunks=(4, 4),
        access_log=access_log,
    )
    spec = write_synthetic_slc(
        store,
        source_path="scene-b",
        acquisition_id="scene-b",
        samples=samples,
    )
    reader = SyntheticSLCReader(store=store, specs=register_specs(spec))
    result = reader.open("scene-b")

    access_log.clear()
    selection = (slice(4, 8), slice(8, 12))
    block = store.read(result.product.samples, selection)

    np.testing.assert_array_equal(block, samples[4:8, 8:12])
    assert set(access_log.touches) == {(1, 2)}


def test_missing_doppler_metadata_raises_typed_error() -> None:
    """Reject SLC open when the Doppler centroid polynomial is absent."""
    samples = _samples()
    store = InMemoryArrayStore()
    spec = write_synthetic_slc(
        store,
        source_path="memory://missing-doppler",
        acquisition_id="missing-doppler",
        samples=samples,
        include_doppler=False,
    )
    reader = SyntheticSLCReader(store=store, specs=register_specs(spec))

    with pytest.raises(
        MissingCriticalMetadataError,
        match="doppler_centroid",
    ) as caught:
        reader.open("memory://missing-doppler")

    assert caught.value.source_path == "memory://missing-doppler"
    assert "annotation" in caught.value.remedy


def test_missing_valid_samples_metadata_raises_typed_error() -> None:
    """Reject SLC open when valid-sample windows are absent."""
    samples = _samples()
    store = InMemoryArrayStore()
    spec = write_synthetic_slc(
        store,
        source_path="memory://missing-valid",
        acquisition_id="missing-valid",
        samples=samples,
        include_valid_samples=False,
    )
    reader = SyntheticSLCReader(store=store, specs=register_specs(spec))

    with pytest.raises(MissingCriticalMetadataError, match="valid_samples") as caught:
        reader.open("memory://missing-valid")

    assert caught.value.field == "valid_samples"
    assert "first/last valid sample" in caught.value.remedy

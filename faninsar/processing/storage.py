"""In-memory and chunked Zarr storage adapters for processing arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

from .coordinates import ArrayDescriptor, ArrayRepresentation
from .readers import normalize_selection

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = setup_logger(__name__)


@dataclass(slots=True)
class ChunkAccessLog:
    """Record which chunk indices were read for lazy-slice audits."""

    touches: list[tuple[int, int]] = field(default_factory=list)

    def record(self, row_chunk: int, col_chunk: int) -> None:
        """Append one touched chunk index pair."""
        self.touches.append((row_chunk, col_chunk))

    def clear(self) -> None:
        """Reset the access log."""
        self.touches.clear()


@dataclass(slots=True)
class InMemoryArrayStore:
    """Eager host-memory store implementing ArrayReader and ArrayWriter."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, dict[str, str]] = field(default_factory=dict)

    def write(
        self,
        uri: str,
        value: np.ndarray,
        metadata: Mapping[str, str],
    ) -> ArrayDescriptor:
        """Store a two-dimensional array under a URI.

        Parameters
        ----------
        uri : str
            Stable locator for the written asset.
        value : numpy.ndarray
            Two-dimensional array to persist in memory.
        metadata : Mapping[str, str]
            Small eager metadata retained with the asset.

        Returns
        -------
        ArrayDescriptor
            Immutable descriptor for the stored array.

        """
        if value.ndim != 2:
            message = "InMemoryArrayStore only accepts 2-D arrays"
            logger.error(message)
            raise ValueError(message)
        stored = np.array(value, copy=True)
        self.arrays[uri] = stored
        self.metadata[uri] = dict(metadata)
        return ArrayDescriptor(
            uri=uri,
            shape=(int(stored.shape[0]), int(stored.shape[1])),
            dtype=str(stored.dtype),
            representation=_representation_for(stored.dtype),
        )

    def read(
        self,
        descriptor: ArrayDescriptor,
        selection: tuple[slice, slice],
    ) -> np.ndarray:
        """Read one selection from an in-memory array."""
        array = self.arrays.get(descriptor.uri)
        if array is None:
            message = f"unknown in-memory array URI: {descriptor.uri}"
            logger.error(message)
            raise KeyError(message)
        row_slice, col_slice = normalize_selection(descriptor.shape, selection)
        return np.array(array[row_slice, col_slice], copy=True)


@dataclass(slots=True)
class ChunkedZarrArrayStore:
    """Chunked Zarr store with access logging for lazy-slice proofs."""

    root: Path
    chunks: tuple[int, int]
    access_log: ChunkAccessLog = field(default_factory=ChunkAccessLog)

    def write(
        self,
        uri: str,
        value: np.ndarray,
        metadata: Mapping[str, str],
    ) -> ArrayDescriptor:
        """Write a two-dimensional array to a Zarr group under ``root``.

        Parameters
        ----------
        uri : str
            Relative asset name used as the Zarr array key.
        value : numpy.ndarray
            Two-dimensional array to persist.
        metadata : Mapping[str, str]
            Small eager metadata stored as Zarr attributes.

        Returns
        -------
        ArrayDescriptor
            Immutable descriptor for the written Zarr array.

        """
        import zarr

        if value.ndim != 2:
            message = "ChunkedZarrArrayStore only accepts 2-D arrays"
            logger.error(message)
            raise ValueError(message)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / uri
        if path.exists():
            import shutil

            shutil.rmtree(path)
        array = zarr.open_array(
            store=str(path),
            mode="w",
            shape=value.shape,
            chunks=self.chunks,
            dtype=value.dtype,
        )
        array[...] = value
        for key, item in metadata.items():
            array.attrs[key] = item
        return ArrayDescriptor(
            uri=str(path),
            shape=(int(value.shape[0]), int(value.shape[1])),
            dtype=str(value.dtype),
            representation=_representation_for(value.dtype),
        )

    def read(
        self,
        descriptor: ArrayDescriptor,
        selection: tuple[slice, slice],
    ) -> np.ndarray:
        """Read one selection and record the touched Zarr chunks."""
        import zarr

        array = zarr.open_array(store=descriptor.uri, mode="r")
        row_slice, col_slice = normalize_selection(descriptor.shape, selection)
        self._log_touched_chunks(row_slice, col_slice, array.chunks)
        return np.asarray(array[row_slice, col_slice])

    def _log_touched_chunks(
        self,
        row_slice: slice,
        col_slice: slice,
        chunks: tuple[int, ...] | None,
    ) -> None:
        """Record every chunk index overlapping the normalized selection."""
        if chunks is None or len(chunks) < 2:
            message = "Zarr array must expose two-dimensional chunks"
            logger.error(message)
            raise ValueError(message)
        row_chunk, col_chunk = int(chunks[0]), int(chunks[1])
        row_start = row_slice.start // row_chunk
        row_stop = (row_slice.stop - 1) // row_chunk
        col_start = col_slice.start // col_chunk
        col_stop = (col_slice.stop - 1) // col_chunk
        for row_index in range(row_start, row_stop + 1):
            for col_index in range(col_start, col_stop + 1):
                self.access_log.record(row_index, col_index)


def _representation_for(dtype: np.dtype) -> ArrayRepresentation:
    """Map a NumPy dtype to the processing array representation enum."""
    if np.issubdtype(dtype, np.complexfloating):
        return ArrayRepresentation.COMPLEX
    if np.issubdtype(dtype, np.floating):
        return ArrayRepresentation.PHASE
    return ArrayRepresentation.MASK

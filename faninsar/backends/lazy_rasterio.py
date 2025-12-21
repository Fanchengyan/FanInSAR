"""Lazy loading backend using rasterio + dask + xarray.

This module provides classes to wrap rasterio file reading operations
into dask arrays for lazy loading and parallel computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import dask
import dask.array as da
import rasterio
from rasterio.windows import Window

if TYPE_CHECKING:
    import numpy as np


class LazyRasterioReader:
    """Wrap rasterio reading as dask lazy arrays.

    This class reads raster file metadata immediately but defers actual
    data loading until compute() is called. It uses dask for chunked
    parallel reading.

    Parameters
    ----------
    path : str
        Path to the raster file.
    chunks : tuple[int, int] | dict[str, int] | None, optional
        Chunk size for dask array. Can be a tuple (height, width) or
        a dict with 'y' and 'x' keys. Default is (512, 512).
    lock : bool, optional
        Whether to use locks for thread safety. Rasterio usually doesn't
        need this. Default is False.

    Attributes
    ----------
    path : str
        Path to the raster file.
    shape : tuple[int, int]
        Shape of the raster (height, width).
    dtype : np.dtype
        Data type of the raster.
    crs : CRS
        Coordinate reference system.
    transform : Affine
        Affine transformation matrix.
    nodata : float | None
        No data value.
    count : int
        Number of bands.
    bounds : tuple[float, float, float, float]
        Bounds of the raster (left, bottom, right, top).
    chunks : tuple[int, int]
        Chunk size used for dask arrays.

    Examples
    --------
    >>> reader = LazyRasterioReader("path/to/file.tif", chunks=(512, 512))
    >>> dask_array = reader.to_dask_array(band=1)
    >>> # Data not loaded yet
    >>> print(dask_array)
    dask.array<...>
    >>> # Load data
    >>> numpy_array = dask_array.compute()

    """

    def __init__(
        self,
        path: str,
        chunks: tuple[int, int] | dict[str, int] | None = None,
        lock: bool = False,
    ) -> None:
        """Initialize LazyRasterioReader."""
        self.path = path
        self.lock = lock

        # Normalize chunks
        if chunks is None:
            self.chunks = (512, 512)
        elif isinstance(chunks, dict):
            self.chunks = (chunks.get("y", 512), chunks.get("x", 512))
        else:
            self.chunks = tuple(chunks)

        # Read metadata only (no data loading)
        with rasterio.open(path) as src:
            self.shape = src.shape
            self.dtype = src.dtypes[0]
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata
            self.count = src.count
            self.bounds = src.bounds

    def _make_chunk_reader(
        self, band: int = 1, window: Window | None = None
    ) -> Callable[[tuple[int, ...] | None], np.ndarray]:
        """Create a function to read specific chunks.

        Parameters
        ----------
        band : int, optional
            Band index (1-based). Default is 1.
        window : Window | None, optional
            Global window to read from. Default is None (read entire array).

        Returns
        -------
        read_chunk : Callable
            Function that reads a chunk given block_id.

        """

        def read_chunk(block_id: tuple[int, ...] | None = None) -> np.ndarray:
            """Read a single chunk.

            Parameters
            ----------
            block_id : tuple[int, ...] | None
                Block ID provided by dask, e.g., (0, 0) for first block.
                If None, read entire array.

            Returns
            -------
            chunk_data : np.ndarray
                Data for this chunk.

            """
            if block_id is None:
                # Read entire array
                with rasterio.open(self.path) as src:
                    return src.read(band, window=window, masked=True)

            # Calculate chunk position in global array
            row_idx, col_idx = block_id

            # Calculate chunk window
            row_start = row_idx * self.chunks[0]
            col_start = col_idx * self.chunks[1]

            # Handle boundary cases
            if window is not None:
                # Adjust for global window
                row_start += window.row_off
                col_start += window.col_off
                row_end = min(
                    row_start + self.chunks[0], window.row_off + window.height
                )
                col_end = min(col_start + self.chunks[1], window.col_off + window.width)
            else:
                row_end = min(row_start + self.chunks[0], self.shape[0])
                col_end = min(col_start + self.chunks[1], self.shape[1])

            chunk_window = Window(
                col_off=col_start,
                row_off=row_start,
                width=col_end - col_start,
                height=row_end - row_start,
            )

            # Read using rasterio
            with rasterio.open(self.path) as src:
                return src.read(band, window=chunk_window, masked=True)

        return read_chunk

    def to_dask_array(self, band: int = 1, window: Window | None = None) -> da.Array:
        """Convert to dask array (lazy).

        Parameters
        ----------
        band : int, optional
            Band index (1-based). Default is 1.
        window : Window | None, optional
            Rasterio Window for cropping. Default is None.

        Returns
        -------
        dask_array : da.Array
            Dask array with data not yet loaded.

        Notes
        -----
        The actual data reading happens only when compute() is called
        on the returned dask array.

        """
        # Determine array shape
        shape = (window.height, window.width) if window is not None else self.shape

        # Create chunk reader
        reader = self._make_chunk_reader(band, window)

        # Create delayed read operation
        delayed_read = dask.delayed(reader)(block_id=None)

        # Convert to dask array
        dask_array = da.from_delayed(delayed_read, shape=shape, dtype=self.dtype)

        # Apply chunking
        return dask_array.rechunk(self.chunks)


class LazyMultiFileReader:
    """Lazy loading for multiple raster files.

    This class handles multiple files and stacks them along a new
    dimension (typically the file/time dimension).

    Parameters
    ----------
    paths : list[str]
        List of file paths.
    chunks : dict[str, int] | int | Literal["auto"] | None, optional
        Chunk sizes for dask arrays. Accepts:
        - ``None``: Uses default 512x512 chunks.
        - ``"auto"`` or ``{}``: Auto-detect from first file's block size.
        - ``int``: Use same size for both y and x dimensions.
        - ``dict``: Specify 'y' and 'x' separately, e.g., ``{"y": 256, "x": 512}``.

    Attributes
    ----------
    paths : list[str]
        List of file paths.
    chunks : dict[str, int]
        Resolved chunk sizes.
    readers : list[LazyRasterioReader]
        List of readers for each file.

    Examples
    --------
    >>> multi_reader = LazyMultiFileReader(
    ...     paths=["file1.tif", "file2.tif"], chunks={"y": 512, "x": 512}
    ... )
    >>> stacked = multi_reader.to_stacked_dask_array(band=1)
    >>> print(stacked.shape)
    (2, height, width)

    Using auto chunks (aligned to file block size):

    >>> multi_reader = LazyMultiFileReader(paths=["cog.tif"], chunks="auto")
    >>> print(multi_reader.chunks)  # e.g., {"y": 256, "x": 256} from file

    """

    def __init__(
        self,
        paths: list[str],
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
    ) -> None:
        """Initialize LazyMultiFileReader."""
        self.paths = paths
        self._chunks = self._resolve_chunks(chunks, paths[0] if paths else None)

        # Create reader for each file
        self.readers = [LazyRasterioReader(path, chunks=self._chunks) for path in paths]

    @staticmethod
    def _resolve_chunks(
        chunks: dict[str, int] | int | Literal["auto"] | None,
        reference_path: str | None,
    ) -> dict[str, int]:
        """Resolve chunks to a dict[str, int] format.

        Parameters
        ----------
        chunks : dict | int | Literal["auto"] | None
            User-specified chunk configuration.
        reference_path : str | None
            Path to a reference file for auto-detection.

        Returns
        -------
        dict[str, int]
            Resolved chunks as {"y": ..., "x": ...}.

        """
        # Default chunks
        default_chunks = {"y": 512, "x": 512}

        if chunks is None:
            return default_chunks

        # Auto-detect from file block size
        if chunks == "auto" or chunks == {}:
            if reference_path is None:
                return default_chunks
            try:
                with rasterio.open(reference_path) as src:
                    block_shapes = src.block_shapes
                    if block_shapes and block_shapes[0] != (1, src.width):
                        # Tiled file: use block size
                        block_height, block_width = block_shapes[0]
                        return {"y": block_height, "x": block_width}
                    # Striped file: use default
                    return default_chunks
            except Exception:
                return default_chunks

        # Integer: use same size for both dimensions
        if isinstance(chunks, int):
            return {"y": chunks, "x": chunks}

        # Dict: extract y and x, use defaults for missing
        if isinstance(chunks, dict):
            return {
                "y": chunks.get("y", default_chunks["y"]),
                "x": chunks.get("x", default_chunks["x"]),
            }

        # Fallback
        return default_chunks

    @property
    def chunks(self) -> dict[str, int]:
        """Return resolved chunk sizes."""
        return self._chunks

    def to_stacked_dask_array(
        self, band: int = 1, window: Window | None = None
    ) -> da.Array:
        """Stack files along a new dimension, returning dask array.

        Parameters
        ----------
        band : int, optional
            Band index (1-based). Default is 1.
        window : Window | None, optional
            Window for cropping. Default is None.

        Returns
        -------
        stacked : da.Array
            Dask array with shape (n_files, height, width).

        """
        # Create dask array for each file
        lazy_arrays = []
        for reader in self.readers:
            arr = reader.to_dask_array(band, window)
            lazy_arrays.append(arr)

        # Stack along new dimension (file dimension)
        return da.stack(lazy_arrays, axis=0)


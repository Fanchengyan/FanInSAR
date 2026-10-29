"""TimeSeriesDataset base class extracted from faninsar.datasets.base."""

from __future__ import annotations

import contextlib
from abc import ABC
from typing import TYPE_CHECKING, Literal

import pandas as pd
import rioxarray  # noqa: F401

from faninsar import Acquisition
from faninsar import __version__ as _faninsar_version
from faninsar.query import BoundingBox, GeoQuery, Points, Polygons

from . import manifest as _manifest
from .raster import RasterDataset

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    import numpy as np
    import xarray as xr


class TimeSeriesDataset(RasterDataset, ABC):
    """A base class for time series datasets."""

    _dates: Acquisition

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the dataset and attach acquisition metadata."""
        super().__init__(*args, **kwargs)
        self._manifest_date_strings: list[str] | None = self._manifest_extra("date")
        self._assign_dates_from_paths()

    @property
    def dates(self) -> Acquisition:
        """Return the date for each acquisition in the dataset."""
        return self._dates

    def _manifest_extra(self, key: str) -> list | None:
        """Return the per-file manifest values for ``key``, or None."""
        raw = self._manifest_lookup(self._files.paths.tolist())
        if raw is None:
            return None
        values = [row.get(key) for row in raw]
        if any(v is None for v in values):
            return None
        return values

    def _manifest_write_extras(self, key: str, values: list) -> None:
        """Write ``values`` back into the manifest under ``key``."""
        raw = self._manifest_lookup(self._files.paths.tolist())
        if raw is None:
            return
        for row, value in zip(raw, values, strict=True):
            row[key] = _manifest.encode_file_row({"v": value})["v"]
        with contextlib.suppress(OSError):
            _manifest.write_entry(
                self.root_dir,
                self._manifest_name(),
                self._files.paths.tolist(),
                raw,
                _faninsar_version,
            )

    def _assign_dates_from_paths(self) -> None:
        """Parse acquisition dates from current file list."""
        paths = self._files.paths.tolist()
        if len(paths) == 0:
            self._dates = Acquisition([])
            self._files.loc[:, "date"] = pd.NaT
            self._manifest_write_extras("date", [])
            return

        if self._manifest_date_strings is not None:
            # Manifest restored the parsed dates — instantiate directly,
            # skipping the filename-regex round trip.
            parsed_dates = pd.to_datetime(self._manifest_date_strings, format="%Y%m%d")
            acquisitions = Acquisition(parsed_dates)
        else:
            parsed = self.parse_dates(paths)
            acquisitions = (
                parsed if isinstance(parsed, Acquisition) else Acquisition(parsed)
            )

        if len(acquisitions) != len(self._files):
            msg = (
                "Parsed acquisition dates do not align with scanned files: "
                f"{len(acquisitions)} dates for {len(self._files)} files."
            )
            raise ValueError(msg)

        self._dates = acquisitions
        date_series = pd.Series(acquisitions.values, index=self._files.index)
        self._files.loc[:, "date"] = pd.to_datetime(date_series)
        self._manifest_write_extras(
            "date",
            [
                pd.Timestamp(v).strftime("%Y%m%d") if v is not None else None
                for v in acquisitions.values
            ],
        )

    @classmethod
    def parse_dates(cls, paths: Iterable[str | PathLike]) -> Acquisition:
        """Parse dates from filenames.

        Parameters
        ----------
        paths : list of pathlib.Path
            list of file paths to parse dates

        Returns
        -------
        dates : Acquisition
            dates parsed from filenames

        """
        msg = "parse_dates method must be implemented in subclass"
        raise NotImplementedError(msg)

    @property
    def file_dim_name(self) -> str:
        """Dimension name for time-series stacking."""
        return "date"

    def _file_coords(
        self,
        indexes: np.ndarray,
        paths: list[str],  # noqa: ARG002
        files_df: pd.DataFrame,
    ) -> dict[str, tuple[str, np.ndarray]]:
        """Attach acquisition metadata to stacked coordinates."""
        if "date" in files_df:
            date_values = pd.to_datetime(files_df["date"].to_numpy())
        else:
            date_values = self.dates.take(indexes).to_numpy()
        date_index = pd.DatetimeIndex(date_values)
        coords: dict[str, tuple[str, np.ndarray]] = {
            "date": ("date", date_index.to_numpy())
        }
        return coords

    # New explicit per-shape query methods using dates instead of indexes
    def get_indexes(
        self,
        dates: Acquisition | pd.DatetimeIndex | None = None,
    ) -> np.ndarray:
        """Return file indexes for the given dates.

        Parameters
        ----------
        dates : Acquisition | pd.DatetimeIndex | None, optional
            Dates to select. If None, returns indexes of all valid files.

        Returns
        -------
        indexes : np.ndarray
            Integer array of file indexes matching the given dates.

        """
        files_df = self.files
        mask = files_df.valid.copy()
        if dates is not None:
            target = pd.DatetimeIndex(dates)
            mask = mask & files_df["date"].isin(target)
        return files_df[mask].index.to_numpy(dtype=int)

    def points_query(
        self,
        points: Points,
        dates: Acquisition | pd.DatetimeIndex | None = None,
        verbose: bool | None = None,
    ) -> xr.Dataset:
        """Query points for the given dates subset (no indexes support)."""
        resolved_indexes = self.get_indexes(dates=dates)
        return self._compute_points_ds(points, resolved_indexes, verbose)

    def boxes_query(
        self,
        bbox: BoundingBox | list[BoundingBox],
        dates: Acquisition | pd.DatetimeIndex | None = None,
        verbose: bool | None = None,
        chunks: dict[str, int] | int | Literal["auto", False] | None = None,
    ) -> xr.DataTree:
        """Query bbox/boxes for the given dates subset (no indexes support)."""
        resolved_indexes = self.get_indexes(dates=dates)
        c = self._resolve_chunks(chunks)
        if c is not None:
            return self._compute_bboxes_tree_lazy(bbox, resolved_indexes, c)
        return self._compute_bboxes_tree(bbox, resolved_indexes, verbose)

    def polygons_query(
        self,
        polygons: Polygons,
        dates: Acquisition | pd.DatetimeIndex | None = None,
        verbose: bool | None = None,
        chunks: dict[str, int] | int | Literal["auto", False] | None = None,
    ) -> xr.DataTree:
        """Query polygons for the given dates subset (no indexes support)."""
        resolved_indexes = self.get_indexes(dates=dates)
        c = self._resolve_chunks(chunks)
        if c is not None:
            return self._compute_polygons_tree_lazy(polygons, resolved_indexes, c)
        return self._compute_polygons_tree(polygons, resolved_indexes, verbose)

    def query(
        self,
        query: GeoQuery | Points | BoundingBox | Polygons,
        dates: Acquisition | pd.DatetimeIndex | None = None,
        verbose: bool | None = None,
        chunks: dict[str, int] | int | Literal["auto", False] | None = None,
    ) -> xr.DataTree:
        """Retrieve image values for given query using dates subset only."""
        if isinstance(query, Points):
            query = GeoQuery(points=query)
        if isinstance(query, BoundingBox):
            query = GeoQuery(boxes=query)
        if isinstance(query, Polygons):
            query = GeoQuery(polygons=query)

        resolved_indexes = self.get_indexes(dates=dates)
        paths = self.files.iloc[resolved_indexes].paths.tolist()

        c = self._resolve_chunks(chunks)
        if c is not None:
            return self._sample_files_lazy(paths, query, verbose, c)
        return self._sample_files(paths, query, verbose)

"""Abstract base classes for Atmospheric Phase Screen (APS) datasets."""

from __future__ import annotations

import re
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from faninsar._core.sar.pairs import Pairs
from faninsar.datasets.base import PairDataset, TimeSeriesDataset
from faninsar.logging import setup_logger
from faninsar.query import BoundingBox, Points

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from numpy.typing import NDArray
    from pyproj.crs.crs import CRS
    from xarray import DataArray, Dataset


class ApsDataset(TimeSeriesDataset, ABC):
    """A base class for aps (atmospheric phase screen) datasets."""

    #: This expression is used to find the APS files.
    pattern = "*"

    _pairs = None

    def _requested_dates(
        self,
        dates: Sequence[str] | pd.DatetimeIndex,
    ) -> pd.DatetimeIndex:
        """Return unique, sorted acquisition dates and fail if any are missing."""
        requested = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
        if requested.has_duplicates:
            msg = "dates must be unique"
            logger.error(msg, extra={"dates": requested.strftime("%Y%m%d").tolist()})
            raise ValueError(msg)
        if len(requested) < 2:
            msg = "dates must contain at least two acquisitions"
            logger.error(msg, extra={"n_date": len(requested)})
            raise ValueError(msg)

        available = pd.DatetimeIndex(pd.to_datetime(self.dates))
        missing = requested.difference(available)
        if len(missing) > 0:
            ds_name = self.ds_name or self.__class__.__name__
            msg = (
                f"Following dates are missing in the {ds_name} dataset: "
                f"{missing.strftime('%Y%m%d').tolist()}"
            )
            logger.error(msg, extra={"missing": missing.strftime("%Y%m%d").tolist()})
            raise ValueError(msg)
        return requested

    def _drop_band(self, data: DataArray) -> DataArray:
        """Remove a singleton band dimension when present."""
        if "band" in data.dims:
            return data.isel(band=0)
        return data

    def _values_at_dates(
        self,
        query: Points | BoundingBox,
        dates: pd.DatetimeIndex,
    ) -> NDArray[np.floating]:
        """Sample APS values as a ``(n_date, n_pt)`` array ordered by ``dates``."""
        if isinstance(query, Points):
            sampled: Dataset = self.points_query(query, dates=dates, verbose=False)
            data = self._drop_band(sampled["data"]).sel(date=dates.to_numpy())
            values = np.asarray(data, dtype=np.float64)
        elif isinstance(query, BoundingBox):
            tree = self.boxes_query(query, dates=dates, verbose=False)
            data = self._drop_band(tree.dataset["data"]).sel(date=dates.to_numpy())
            values = np.asarray(data, dtype=np.float64)
        else:
            msg = "query must be Points or BoundingBox"
            logger.error(msg, extra={"query_type": type(query).__name__})
            raise TypeError(msg)

        if values.ndim == 1:
            return values.reshape(values.shape[0], 1)
        if values.ndim > 2:
            return values.reshape(values.shape[0], -1)
        return values

    def increments(
        self,
        dates: Sequence[str] | pd.DatetimeIndex,
        *,
        query: Points | BoundingBox,
        ref_points: Points,
        los_ratio: NDArray[np.floating] | float | None = None,
        insert_ref_zero: bool = False,
    ) -> NDArray[np.floating]:
        """Return adjacent APS increments aligned with NSBAS ``incs``.

        The result is ``G(t_{i+1}) - G(t_i)`` after subtracting the mean
        reference-point difference. The default shape is
        ``(n_date - 1, n_pt)``, matching :meth:`faninsar.NSBAS.NSBASSolver.inverse`.

        Parameters
        ----------
        dates : Sequence[str] or pandas.DatetimeIndex
            Acquisition dates. Must be a unique set of dates present in this
            dataset. They are sorted before differencing.
        query : Points or BoundingBox
            Locations sampled for the APS increments. Points keep one column
            per point; a bounding box is flattened in C order to
            ``(n_date - 1, n_pixel)``.
        ref_points : Points
            Reference points subtracted from every increment. Multiple points
            are averaged.
        los_ratio : array-like or float, optional
            Line-of-sight direction cosine used to convert zenith delay to
            LOS. ``increments / los_ratio`` is applied when provided. For
            HyP3 ``lv_theta`` (look-vector elevation from horizontal), use
            :meth:`faninsar.datasets.ifg.InterferogramDataset.load_los_ratio`
            with the default ``angle_type="look"`` (``sin(θ)``).
        insert_ref_zero : bool, optional
            If True, prepend a zero row so the result has length ``n_date``
            for a cumulative time series. Default is False.

        Returns
        -------
        increments : numpy.ndarray
            Adjacent APS increments in millimeters with shape
            ``(n_date - 1, n_pt)``, or ``(n_date, n_pt)`` when
            ``insert_ref_zero`` is True. GACOS zenith total delay is stored
            in meters; the millimeter conversion matches NSBAS deformation.

        Raises
        ------
        ValueError
            If fewer than two dates are given, dates are duplicated, or any
            requested date is missing from the dataset.
        TypeError
            If ``query`` is not :class:`~faninsar.query.Points` or
            :class:`~faninsar.query.BoundingBox`.

        Examples
        --------
        Subtract the APS increments from NSBAS incremental displacement
        (``gacos`` is a :class:`GACOS` dataset, ``pts`` the sampled
        query points, and ``ref_points`` the reference points)::

            increments = gacos.increments(
                dates=pairs.dates,
                query=pts,
                ref_points=ref_points,
                los_ratio=los_ratio,
            )
            incs_corr = incs - increments

        """
        requested = self._requested_dates(dates)
        values = self._values_at_dates(query, requested)
        ref_values = self._values_at_dates(ref_points, requested)
        increments = np.diff(values, axis=0) - np.diff(ref_values.mean(axis=1))[:, None]
        if los_ratio is not None:
            increments = increments / np.asarray(los_ratio, dtype=np.float64)
        increments = increments * 1000.0
        if insert_ref_zero:
            zeros = np.zeros((1, increments.shape[1]), dtype=increments.dtype)
            increments = np.vstack((zeros, increments))
        return increments

    def correct(
        self,
        incs: NDArray[np.floating],
        dates: Sequence[str] | pd.DatetimeIndex,
        *,
        query: Points | BoundingBox,
        ref_points: Points,
        los_ratio: NDArray[np.floating] | float | None = None,
    ) -> NDArray[np.floating]:
        """Subtract adjacent APS increments from NSBAS incremental displacement.

        Parameters
        ----------
        incs : numpy.ndarray
            Incremental displacement in millimeters from
            :meth:`faninsar.NSBAS.NSBASSolver.inverse`, shape
            ``(n_date - 1, n_pt)``. Do not pass a cumulative series that
            already has a leading zero.
        dates : Sequence[str] or pandas.DatetimeIndex
            Acquisition dates of length ``n_date``.
        query : Points or BoundingBox
            Locations sampled for the APS increments. Must match ``incs``.
        ref_points : Points
            Reference points subtracted from every increment.
        los_ratio : array-like or float, optional
            Line-of-sight direction cosine used to convert zenith delay to
            LOS. For HyP3 ``lv_theta``, use
            :meth:`~faninsar.datasets.ifg.InterferogramDataset.load_los_ratio`
            with the default ``angle_type="look"``.

        Returns
        -------
        incs_corr : numpy.ndarray
            Atmosphere-corrected increments in millimeters with the same
            shape as ``incs``.

        Raises
        ------
        ValueError
            If ``incs`` is not shaped ``(n_date - 1, n_pt)``, including the
            case of a cumulative series with a leading zero.

        Examples
        --------
        Correct the NSBAS increments right after the inversion::

            incs, *_ = solver.inverse()
            incs_corr = gacos.correct(
                incs,
                dates=pairs.dates,
                query=pts,
                ref_points=ref_points,
                los_ratio=los_ratio,
            )

        """
        increments = self.increments(
            dates,
            query=query,
            ref_points=ref_points,
            los_ratio=los_ratio,
        )
        incs_arr = np.asarray(incs, dtype=np.float64)
        if incs_arr.shape[0] != increments.shape[0]:
            msg = (
                f"incs has {incs_arr.shape[0]} rows, APS has {increments.shape[0]}; "
                "incs must be (n_date - 1, n_pt), not cumulative with a leading zero."
            )
            logger.error(
                msg,
                extra={"incs_rows": incs_arr.shape[0], "aps_rows": increments.shape[0]},
            )
            raise ValueError(msg)
        if incs_arr.shape != increments.shape:
            msg = (
                f"incs shape {incs_arr.shape} does not match APS increments "
                f"{increments.shape}"
            )
            logger.error(
                msg,
                extra={"incs_shape": incs_arr.shape, "aps_shape": increments.shape},
            )
            raise ValueError(msg)
        return incs_arr - increments

    def to_pair_files(
        self,
        out_dir: PathLike,
        pairs: Pairs,
        ref_points: Points,
        roi: BoundingBox | None = None,
        overwrite: bool = False,
        prefix: str = "APS",
    ) -> None:
        """Generate aps-pair files for given pairs and reference points.

        Parameters
        ----------
        out_dir : str or Path
            path to the directory to save the aps-pair files
        pairs : Pairs
            pairs to generate aps-pair files
        ref_points : Points
            reference points which values are subtracted for all aps-pair files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.
        overwrite : bool, optional
            if True, overwrite existing files, default: False
        prefix : str, optional
            prefix of the aps-pair files, default: "APS"

        """
        if roi is None:
            roi = self.roi

        profile = self.get_profile(roi)
        profile["count"] = 1
        dates = self.parse_dates(self.files.paths.tolist())

        dates_missing = np.setdiff1d(pairs.dates, dates)
        if len(dates_missing) > 0:
            msg = (
                f"Following dates are missing in the {self.ds_name} "
                f"dataset. \n{dates_missing}"
            )
            logger.warning(msg, stacklevel=2)

        df_paths = pd.Series(self.files.paths.values, index=dates)

        mask = ~np.any(np.isin(pairs.values, dates_missing), axis=1)
        pairs = pairs[mask]

        pairs_names = self._ensure_saving_verbose(
            pairs.to_names().tolist(),
            ds_name=f"{self.ds_name} Pair",
            unit=" pairs",
        )

        for pair_name in pairs_names:
            primary, secondary = pair_name.split("_")
            out_file = Path(out_dir) / f"{prefix}_{pair_name}.tif"
            if out_file.exists() and not overwrite:
                msg = f"{out_file.name} already exists, skipping"
                logger.info(msg)
                continue
            with rasterio.open(out_file, "w", **profile) as dst:
                src_primary = self._load_warp_file(df_paths[primary])
                src_secondary = self._load_warp_file(df_paths[secondary])
                dest_arr = (
                    self._file_query_bbox(roi, src_primary).squeeze(0)
                    - self._file_query_bbox(roi, src_secondary).squeeze(0)
                    - (
                        self._file_query_points(ref_points, src_primary)
                        - self._file_query_points(ref_points, src_secondary)
                    ).mean()
                )

                dst.write(dest_arr, 1)


class ApsPairs(PairDataset, ABC):
    """A dataset manages the data of APS pairs."""

    #: This expression is used to find the GACOSPairs files.
    pattern = "*.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths: Sequence[str] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | None = None,
        roi: BoundingBox | None = None,
        bands: Sequence[str] | None = None,
        cache: bool = True,
        resampling: Resampling = Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "",
        reindex: bool = False,
    ) -> None:
        """Initialize a new ApsPairs instance.

        Parameters
        ----------
        root_dir : str or Path
            root_dir directory where dataset can be found.
        paths : list of str, optional
            list of file paths to use instead of searching for files in ``root_dir``.
            If None, files will be searched for in ``root_dir``.
        crs : CRS, optional
            the output term:`coordinate reference system (CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res : float, optional
            resolution of the output dataset in units of CRS. If None, the resolution
            of the first file found will be used.
        dtype : numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first file
            found will be used.
        nodata : float or int, optional
            no data value of the output dataset. If None, the no data value of
            the first file found will be used. This parameter is useful when the
            no data value is not stored in the file.
        roi : BoundingBox, optional
            region of interest to load from the dataset. If None, the union of all files
            bounds in the dataset will be used.
        bands : list of str, optional
            names of bands to return (defaults to all bands)
        cache : bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling : Resampling, optional
            Resampling algorithm used when reading input files.
            Default: `Resampling.nearest`.
        fill_nodata : bool, optional
            Whether to fill holes in the queried data by interpolating them using
            inverse distance weighting method provided by the
            :func:`rasterio.fill.fillnodata`. Default: False.

            .. note::
                This parameter is only used when sampling data using bounding
                boxes or polygons queries, and will not work for points queries.

        verbose : bool, optional
            if True, print verbose output, default: True
        ds_name : str, optional
            name of the dataset. used for printing verbose output, default: ""
        reindex : bool, optional
            If True, ignore any existing ``faninsar.manifest.json`` entry and
            rescan the files, rewriting the entry. If False (default), reuse a
            previously written manifest entry when it is present.

        Raises
        ------
            FileNotFoundError: if no files are found in ``root_dir``

        """
        super().__init__(
            root_dir=root_dir,
            paths=paths,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands=bands,
            cache=cache,
            resampling=resampling,
            fill_nodata=fill_nodata,
            verbose=verbose,
            ds_name=ds_name,
            reindex=reindex,
        )

    @classmethod
    def parse_pairs(cls, paths: Sequence[str] | Sequence[Path]) -> Pairs:
        """Parse APS pair names from file paths."""
        pair_names = []
        for path in paths:
            stem = Path(path).stem
            matches = re.findall(r"\d{8}", stem)
            if len(matches) >= 2:
                pair_names.append("_".join(matches[:2]))
            else:
                msg = f"Unable to infer APS pair dates from '{stem}'."
                raise ValueError(msg)
        return Pairs.from_names(pair_names)

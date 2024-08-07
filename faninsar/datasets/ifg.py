from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from tqdm import tqdm

from faninsar._core import geo_tools
from faninsar._core.logger import setup_logger
from faninsar._core.pair_tools import Pairs
from faninsar._core.sar_tools import Baselines, PhaseDeformationConverter
from faninsar.datasets.base import ApsPairs, PairDataset, RasterDataset
from faninsar.query import BoundingBox, GeoQuery, Points

logger = setup_logger(
    log_name="FanInSAR.datasets.ifg", log_format="%(levelname)s - %(message)s"
)


class CoherenceDataset(PairDataset):
    """A base class for coherence datasets."""

    _range: tuple[float, float]

    @property
    def range(self) -> tuple[float, float]:
        """Return the range of the coherence."""
        return self._range

    def scale_range(self, arr: np.ndarray) -> np.ndarray:
        """Scale the coherence array to the range of [0, 1]."""
        if self.range[0] != 0 or self.range[1] != 1:
            arr = (arr - self.range[0]) / (self.range[1] - self.range[0])
        return arr

    def trim_extreme(
        self,
        arr: np.ndarray,
        val_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Trim the extreme values of the coherence array.

        Parameters
        ----------
        arr : np.ndarray
            coherence array to be trimmed.
        val_range : tuple[float, float], optional
            value range to clip the array. If None, the range of the dataset will
            be used. Default is None.
        """
        if val_range is None:
            val_range = self.range
        arr = np.clip(arr, val_range[0], val_range[1])
        return arr

    def to_mean(
        self,
        pairs: Optional[Pairs] = None,
        roi: Optional[BoundingBox] = None,
    ) -> np.ndarray:
        """Calculate the mean coherence for given region of interest.

        Parameters
        ----------
        pairs : Pairs, optional
            pairs to calculate the mean coherence. If None, will calculate the
            mean coherence for all pairs.
        roi : BoundingBox, optional
            region of interest to calculate the mean coherence. If None, the roi
            of the dataset will be used.

        Returns
        -------
        mean_coh : np.ndarray
            mean coherence array with value range in the interval of [0, 1].
        """
        if roi is None:
            roi = self.roi
        fill_nodata = self.fill_nodata
        self.fill_nodata = False

        # get files
        m = self.valid
        if pairs is not None:
            m &= self.pairs.where(pairs, return_type="mask")
        files = [self._load_warp_file(f) for f in self.files.paths[m]]

        # load all coherence
        coh = self._bbox_query(roi, files[0]).squeeze(0)
        count = (~coh.mask).astype(int)
        coh_sum = self.trim_extreme(coh.filled(0))
        for f in tqdm(files[1:], desc="Calculating mean coherence", unit="file"):
            coh = self._bbox_query(roi, f).squeeze(0)
            count += (~coh.mask).astype(int)
            coh_sum += self.trim_extreme(coh.filled(0))

        # reset fill_nodata to original value
        self.fill_nodata = fill_nodata

        count = np.ma.array(count, mask=(count == 0))
        coh_sum = np.ma.array(coh_sum, mask=(count == 0))
        # calculate the mean coherence
        mean_coh = coh_sum / count

        return self.scale_range(mean_coh)


class InterferogramDataset(PairDataset):
    """
    A base class for interferogram datasets.

    .. Note::
        1. Only the pairs that **both unwrapped interferograms and coherence files
        are valid will be used**.

        2. The unwrapped interferograms are used to initialize this dataset.
        The ``coherence``, ``dem``, and ``mask`` files can be accessed as attributes
        :attr:`coh_dataset`, :attr:`dem_dataset`, and :attr:`mask_dataset` respectively.
    """

    #: pattern used to find interferogram files.
    pattern_unw = "*"
    #: pattern used to find coherence files.
    pattern_coh = "*"
    #: value range of coherence.
    coh_range: Optional[tuple[float, float]] = [0, 1]

    _ds_coh: RasterDataset
    _ds_dem: Optional[RasterDataset] = None
    _ds_mask: Optional[RasterDataset] = None
    _ds_aps: Optional[RasterDataset] = None

    def __init__(
        self,
        root_dir: str = "data",
        paths_unw: Optional[Sequence[str | Path]] = None,
        paths_coh: Optional[Sequence[str | Path]] = None,
        crs: Optional[CRS] = None,
        res: Optional[float | tuple[float, float]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Any] = None,
        roi: Optional[BoundingBox] = None,
        bands_unw: Optional[Sequence[str]] = None,
        bands_coh: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        fill_nodata: bool = False,
        verbose=True,
        keep_common: bool = True,
    ) -> None:
        """Initialize a new InterferogramDataset instance.

        Parameters
        ----------
        root_dir: str
            root_dir directory where dataset can be found.
        paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root_dir``. If None, files will be searched for in ``root_dir``.
        paths_coh: list of str, optional
            list of coherence file paths to use instead of searching for files in
            ``root_dir``. If None, files will be searched for in ``root_dir``.
        crs: CRS, optional
            the output coordinate reference system term:`(CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res: float, optional
            resolution of the output dataset in units of CRS. If None, the resolution
            of the first file found will be used.
        dtype: numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first
            file found will be used.
        nodata: float or int, optional
            no data value of the output dataset. If None, the no data value of the
            first file found will be used. This parameter is useful when the no
            data value is not stored in the file.
        roi: BoundingBox, optional
            region of interest to load from the dataset. If None, the union of all
            files bounds in the dataset will be used.
        bands_unw: list of str, optional
            names of bands to return (defaults to all bands) for unwrapped interferograms.
        bands_coh: list of str, optional
            names of bands to return (defaults to all bands) for coherence.
        cache: bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling: Resampling, optional
            Resampling algorithm used when reading input files.
            Default: `Resampling.nearest`.
        fill_nodata : bool, optional
            Whether to fill holes in the queried data by interpolating them using
            inverse distance weighting method provided by the
            :func:`rasterio.fill.fillnodata`. Default: False.

            .. note::
                This parameter is only used when sampling data using bounding
                boxes or polygons queries, and will not work for points queries.
        verbose: bool, optional, default: True
            if True, print verbose output.
        keep_common: bool, optional, default: True
            Only used when the number of interferograms and coherence files are
            not equal. If True, keep the common pairs of interferograms and
            coherence files and raise a warning. If False, raise an error.
        """
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.cache = cache
        self.resampling = resampling
        self.verbose = verbose

        if paths_unw is None:
            paths_unw = np.unique(list(root_dir.rglob(self.pattern_unw)))
        if paths_coh is None:
            paths_coh = np.unique(list(root_dir.rglob(self.pattern_coh)))

        # Pairs: ensure there are no duplicate pairs
        # remove duplicate pairs
        paths_unw, pairs_unw = self._deduplicate_pairs(
            paths_unw, "unwrapped interferograms"
        )
        paths_coh, pairs_coh = self._deduplicate_pairs(paths_coh, "coherence files")

        # different number of interferograms and coherence files
        if len(paths_unw) != len(paths_coh):
            mismatch_info = (
                f"Number of interferogram files ({len(paths_unw)}) does not match "
                f"number of coherence files ({len(paths_coh)})."
            )
            if not keep_common:
                raise ValueError(mismatch_info)

            mismatch_info += " Only common pairs will be used."
            warnings.warn(mismatch_info)
            # keep paths only with the common pairs
            pairs = pairs_unw.intersect(pairs_coh)
            paths_unw = paths_unw[pairs_unw.where(pairs)]
            paths_coh = paths_coh[pairs_coh.where(pairs)]

        super().__init__(
            root_dir=root_dir,
            paths=paths_unw,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands=bands_unw,
            cache=cache,
            resampling=resampling,
            fill_nodata=fill_nodata,
            verbose=verbose,
            ds_name="Interferogram",
        )

        self._ds_coh = CoherenceDataset(
            root_dir=root_dir,
            paths=paths_coh,
            crs=self.crs,
            res=self.res,
            dtype=self.dtype,
            nodata=self.nodata,
            roi=self.roi,
            bands=bands_coh,
            cache=cache,
            resampling=resampling,
            fill_nodata=fill_nodata,
            verbose=verbose,
            ds_name="Coherence",
        )
        self._ds_coh._range = self.coh_range

        _valid = self._valid & self._ds_coh.valid

        # remove invalid files for unw and coh
        self._files = self._files[_valid]
        self._valid = self._valid[_valid]
        self._ds_coh._files = self._ds_coh._files[_valid]
        self._ds_coh._valid = self._ds_coh._valid[_valid]

        # remove invalid pairs
        self._pairs = self.parse_pairs(self._files.paths)
        self._ds_coh._pairs = self.parse_pairs(self._ds_coh._files.paths)
        # get the datetime from pairs
        self._datetime = self.parse_datetime(paths_unw[_valid])
        self._ds_coh._datetime = self.parse_datetime(paths_coh[_valid])

    def _deduplicate_pairs(self, paths: list[Path], dataset_name: str) -> list[Path]:
        """Remove duplicate pairs from the list of paths."""
        pairs = self.parse_pairs(paths)
        _, index = pairs.sort(inplace=False)
        if len(index) < len(paths):
            deduplicated = "".join(
                [f"\n\t{i.parent.stem}" for i in set(paths) - set(paths[index])]
            )
            warnings.warn(
                f"Duplicate pairs found in dataset {dataset_name}, keeping only the first occurrence"
                f"\nDeduplicate pairs: {deduplicated}"
            )
        return paths[index], pairs

    def parse_baselines(self, pairs: Pairs | None) -> Baselines:
        """Parse the baseline of the interferogram for given pairs.

        Parameters
        ----------
        pairs : Pairs
            The pairs which the baseline will be parsed. Default is None, which
            means all pairs will be parsed.
        """
        raise NotImplementedError(
            "parse_baseline method must be implemented in subclass"
        )

    @property
    def coh_dataset(self) -> RasterDataset:
        """Return the coherence dataset."""
        return self._ds_coh

    @property
    def aps_dataset(self) -> Optional[RasterDataset]:
        """Return the aps (Atmospheric Phase Screen) dataset. If None, no aps data is used."""
        return self._ds_aps

    @property
    def los_dataset(self) -> Optional[RasterDataset]:
        """Return the theta dataset. If None, no theta data is used."""
        return self._ds_los

    @property
    def dem_dataset(self) -> Optional[RasterDataset]:
        """Return the DEM dataset. If None, no DEM data is used."""
        return self._ds_dem

    @property
    def mask_dataset(self) -> Optional[RasterDataset]:
        """Return the mask dataset. If None, no Mask data is used."""
        return self._ds_mask

    def _ensure_ds_kwargs(self, kwargs: dict) -> dict:
        """Format the kwargs for creating a new RasterDataset object.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments used to create a new RasterDataset object.

        Returns
        -------
        kwargs : dict
            Formatted keyword arguments.
        """
        kwargs.setdefault("crs", self.crs)
        kwargs.setdefault("res", self.res)
        kwargs.setdefault("roi", self.roi)
        kwargs.setdefault("cache", self.cache)
        kwargs.setdefault("resampling", self.resampling)
        kwargs.setdefault("verbose", self.verbose)
        return kwargs

    def _ensure_ds(
        self,
        dataset: Optional[RasterDataset],
        ds_str: str,
        ds_class: RasterDataset = RasterDataset,
        **kwargs,
    ):
        """Ensure the dataset is an instance of ds_class. If dataset is None, a
        new ``ds_class`` object will be created using the kwargs."""
        if dataset is None:
            kwargs = self._ensure_ds_kwargs(kwargs)
            dataset = ds_class(**kwargs)
        elif not isinstance(dataset, ds_class):
            raise TypeError(
                f"{ds_str} must be an instance of {ds_class}, got {type(dataset)}"
            )
        return dataset

    def set_aps_dataset(
        self,
        aps_dataset: Optional[ApsPairs] = None,
        **kwargs: Any,
    ) -> None:
        """Set the aps dataset. If aps_dataset is None, a new ApsPairs object will
        be created using the kwargs.

        Parameters
        ----------
        aps_dataset : ApsPairs, optional
            A ApsPairs object. ApsPairs is used to remove the atmospheric phase
            screen (APS) from the unwrapped interferograms. If None, no APS data
            is used.
        **kwargs : dict, optional
            Keyword arguments used to create a new ApsPairs object if aps_dataset
            is None.
        """
        kwargs.setdefault("ds_name", "ApsPairs")
        self._ds_aps = self._ensure_ds(aps_dataset, "aps_dataset", ApsPairs, **kwargs)

    def set_los_dataset(
        self,
        los_dataset: Optional[RasterDataset] = None,
        **kwargs: Any,
    ) -> None:
        """Set the los dataset. los file could be incidence angle (relative to
        vertical) or look angle (relative to horizontal). This file is used to
        convert differential atmospheric phase from vertical to line-of-sight (LOS)
        direction or convert LOS deformation phase to vertical.

        Parameters
        ----------
        los_dataset : RasterDataset, optional
            A RasterDataset object containing the los files.
        **kwargs : dict, optional
            Keyword arguments used to create a new RasterDataset object if
            ``los_dataset`` is None.
        """
        kwargs.setdefault("ds_name", "LOS")
        self._ds_los = self._ensure_ds(los_dataset, "los_dataset", **kwargs)

    def set_dem_dataset(
        self,
        dem_dataset: Optional[RasterDataset] = None,
        **kwargs: Any,
    ) -> None:
        """Set the dem dataset.

        Parameters
        ----------
        dem_dataset : RasterDataset, optional
            A RasterDataset object containing the dem file.
        **kwargs : dict, optional
            Keyword arguments used to create a new RasterDataset object if
            ``dem_dataset`` is None.
        """
        kwargs.setdefault("ds_name", "DEM")
        self._ds_dem = self._ensure_ds(dem_dataset, "dem_dataset", **kwargs)

    def set_mask_dataset(
        self,
        mask_dataset: Optional[RasterDataset] = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("ds_name", "Mask")
        self._ds_mask = self._ensure_ds(mask_dataset, "mask_dataset", **kwargs)

    def load_los_ratio(
        self,
        roi: Optional[BoundingBox] = None,
        angle_type: Literal["incidence", "look"] = "look",
    ) -> np.ndarray:
        """load and convert los angle map to ratio map for given region of
        interest. The ratio map is used to convert differential atmospheric
        phase from vertical to line-of-sight (LOS) direction or convert LOS
        deformation phase to vertical

        Parameters
        ----------
        roi : BoundingBox, optional
            region of interest to load. If None, the roi of the dataset will be
            used.
        angle_type : Literal['incidence', 'look'], optional
            angle type, one of ['incidence', 'look']. 'incidence' means incidence
            angle (relative to vertical) and 'look' means look angle (relative to
            horizontal). Default is 'look'.
        """
        if self.los_dataset is None:
            return None

        sample_theta = self.los_dataset[roi]
        arr_theta = sample_theta.boxes.data.squeeze((0, 1))
        if angle_type == "incidence":
            los_ratio = np.cos(arr_theta)
        elif angle_type == "look":
            los_ratio = np.sin(arr_theta)
        return los_ratio

    def to_nan_count(
        self,
        pairs: Optional[Pairs] = None,
        roi: Optional[BoundingBox] = None,
    ) -> np.ndarray:
        """Calculate the number of nan values for given region of interest.

        Parameters
        ----------
        pairs : Pairs, optional
            pairs to calculate the number of nan values. If None, will calculate
            the number of nan values for all pairs.
        roi : BoundingBox, optional
            region of interest to calculate the mean coherence. If None, the roi
            of the dataset will be used.
        """
        if roi is None:
            roi = self.roi
        fill_nodata = self.fill_nodata
        self.fill_nodata = False

        # get files
        m = self.valid
        if pairs is not None:
            m &= self.pairs.where(pairs, return_type="mask")
        files = [self._load_warp_file(f) for f in self.files.paths[m]]

        # calculate the number of nan values
        nan_count = (self._bbox_query(roi, files[0]).squeeze(0).mask).astype(int)
        for f in tqdm(files[1:]):
            nan_count += (self._bbox_query(roi, f).squeeze(0).mask).astype(int)

        # reset fill_nodata to original value
        self.fill_nodata = fill_nodata

        return nan_count

    def to_netcdf(
        self,
        filename: str | Path,
        roi: Optional[BoundingBox] = None,
        ref_points: Optional[Points] = None,
    ) -> None:
        """Save the dataset to a netCDF file for given region of interest.

        Parameters
        ----------
        filename : str
            path to the netCDF file to save
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be
            used.
        ref_points : Points, optional, default: None
            reference points to save. If None, will keep the original values.
        """
        if roi is None:
            roi = self.roi

        # TODO: using netcdf4 to save the data to avoid the memory issue
        profile = self.get_profile(roi)
        lat, lon = profile.to_latlon()

        query = GeoQuery(boxes=roi, points=ref_points)

        sample_unw = self[query]
        sample_coh = self.coh_dataset[query]

        if ref_points is None:
            unw = sample_unw.boxes.data[0]
        else:
            ref_mean = np.nanmean(sample_unw.points.data, axis=1)
            unw = sample_unw.boxes.data[0] - ref_mean[:, None, None]

        ds = xr.Dataset(
            {
                "unw": (["pair", "lat", "lon"], unw),
                "coh": (["pair", "lat", "lon"], sample_coh.boxes.data[0]),
            },
            coords={
                "pair": self.pairs.to_names(),
                "lat": lat,
                "lon": lon,
            },
        )

        ds = geo_tools.write_geoinfo_into_ds(
            ds, ["unw", "coh"], crs=self.crs, x_dim="lon", y_dim="lat"
        )
        ds.to_netcdf(filename)

    def to_tiffs(
        self,
        out_dir: str | Path,
        roi: Optional[BoundingBox] = None,
        ref_points: Optional[Points] = None,
        pairs: Optional[Pairs] = None,
        pdc: Optional[PhaseDeformationConverter] = None,
        los_ratio: Optional[np.ndarray] = None,
        names_unw: Optional[list[str]] = None,
        names_coh: Optional[list[str]] = None,
        overwrite: bool = True,
    ) -> None:
        """Save the dataset to files for given region of interest.

        Parameters
        ----------
        out_dir : str
            path to the directory to save the files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be
            used.
        ref_points : Points, optional, default: None
            reference points to save. If None, will keep the original values.
        pairs : Pairs, optional
            pairs to save. If None, will save all pairs.
        pdc : PhaseDeformationConverter, optional
            PhaseDeformationConverter object used to convert the phase to
            deformation. If None, will save the phase.
        los_ratio : np.ndarray, optional
            los angle ratio map used to convert deformation from line-of-sight
            (LOS) direction to vertical. You can use the method :meth:`load_los_ratio`
            to load the los angle ratio map. If None, will save the LOS deformation.
        names_unw : list of str, optional
            names of the unwrapped interferograms to save. If None, original names
            files to save. If None, original names will be used.
            If pairs is not None, names should be with the same length as pairs.
        names_coh : list of str, optional
            names of the files to save. If None, original names will be used.
            If pairs is not None, names should be with the same length as pairs.
        overwrite : bool, optional
            if True, overwrite the existing files. Default is True.
        """
        out_dir = Path(out_dir)
        if roi is None:
            roi = self.roi

        profile = self.get_profile(roi)
        profile["count"] = 1

        if pairs is None:
            pairs = self.pairs

        m_pairs = self.pairs.where(pairs, return_type="mask")
        files_unw = self.files.paths[m_pairs].values
        files_coh = self._ds_coh.files.paths[m_pairs].values

        if self.verbose:
            files_unw = tqdm(files_unw, desc="Saving unwrapped interferograms")
        for i, f_unw in enumerate(files_unw):
            if names_unw is None:
                out_file = out_dir / Path(f_unw).name
            else:
                out_file = out_dir / names_unw[i]

            if out_file.exists() and not overwrite:
                logger.info(f"File {out_file} exists, skip")
                continue

            src = self._load_warp_file(f_unw)
            dest_arr = self._bbox_query(roi, src).squeeze(0)

            if ref_points is not None:
                ref_val = (self._points_query(ref_points, src)).mean()
                dest_arr = dest_arr - ref_val
            if pdc is not None:
                dest_arr = pdc.phase2deformation(dest_arr)
            if los_ratio is not None:
                dest_arr = dest_arr / los_ratio

            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(dest_arr, 1)

        if self.verbose:
            files_coh = tqdm(files_coh, desc="Saving coherence files")
        for i, f_coh in enumerate(files_coh):
            if names_coh is None:
                out_file = out_dir / Path(f_coh).name
            else:
                out_file = out_dir / names_coh[i]

            if out_file.exists() and not overwrite:
                logger.info(f"File {out_file} exists, skip")
                continue

            src = self._load_warp_file(f_coh)
            dest_arr = self._bbox_query(roi, src).squeeze(0)

            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(dest_arr, 1)

"""Module for base classes of interferogram and coherence datasets."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from tqdm import tqdm

from faninsar._core import geo_tools
from faninsar.logging import setup_logger
from faninsar.query import BoundingBox, GeoQuery, Points

from .aps import ApsPairs
from .base import PairDataset, RasterDataset

if TYPE_CHECKING:
    from os import PathLike

    from rasterio.crs import CRS

    from faninsar._core.sar.pairs import Pairs
    from faninsar._core.sar.sar_tools import Baselines, PhaseDeformationConverter

logger = setup_logger(__name__)


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
        return np.clip(arr, val_range[0], val_range[1])

    def get_mean_coh(
        self,
        pairs: Pairs | None = None,
        roi: BoundingBox | None = None,
    ) -> np.ndarray:
        """Calculate the mean coherence for given region of interest.

        Parameters
        ----------
        pairs : Pairs, optional
            pairs to calculate the mean coherence. If None, all pairs will be used.
        roi : BoundingBox, optional
            region of interest to calculate the mean coherence. If None, the roi
            of the dataset will be used.

        Returns
        -------
        mean_coh : np.ndarray
            mean coherence array with values in the interval of [0, 1].

        """
        if roi is None:
            roi = self.roi

        profile = self.get_profile(roi)
        width, height = profile["width"], profile["height"]

        # get files
        m = self.valid
        if pairs is not None:
            m &= self.pairs.where(pairs, return_type="mask")
        files_coh_used = self.files.paths[m]

        coh_sum = np.zeros((height, width))
        count = np.zeros((height, width))
        for index, _ in tqdm(files_coh_used.items(), total=len(files_coh_used)):
            coh_data = self.bbox_query(roi, index).data
            count += np.where(coh_data.mask, 0, 1)
            coh_sum += np.where(coh_data.mask, 0, coh_data.data)

        coh_mean = coh_sum / count
        coh_mean = np.where(count == 0, np.nan, coh_mean)  # mask the no-data

        return self.scale_range(coh_mean)


class InterferogramDataset(PairDataset):
    """A base class for interferogram datasets.

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
    coh_range: tuple[float, float] = (0, 1)

    _ds_coh: CoherenceDataset
    _ds_dem: RasterDataset | None = None
    _ds_mask: RasterDataset | None = None
    _ds_aps: RasterDataset | None = None

    def __init__(
        self,
        root_dir: str | PathLike = "data",
        paths_unw: Iterable[PathLike] | None = None,
        paths_coh: Iterable[PathLike] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | None = None,
        roi: BoundingBox | None = None,
        bands_unw: Iterable[str] | None = None,
        bands_coh: Iterable[str] | None = None,
        cache: bool = True,
        resampling: Resampling = Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        keep_common: bool = True,
        lazy_loading: bool = False,
        chunks: dict[str, int] | None = None,
    ) -> None:
        """Initialize a new InterferogramDataset instance.

        Parameters
        ----------
        root_dir: str or PathLike
            root_dir directory where dataset can be found.
        paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root_dir``. If None, files will be searched for in
            ``root_dir``.
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
            names of bands to return (defaults to all bands) for unwrapped
            interferograms.
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
        lazy_loading: bool, optional, default: False
            Enable lazy loading using dask arrays. When True, data is not loaded
            into memory until compute() is called. Default: False.
        chunks: dict[str, int] | None, optional
            Chunk sizes for dask arrays when lazy_loading is True.
            Example: {'y': 512, 'x': 512}. Default is None, which uses 512x512 chunks.

        """
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.cache = cache
        self.resampling = resampling
        self.verbose = verbose

        if paths_unw is None:
            paths_unw = sorted(set(root_dir.rglob(self.pattern_unw)))
        if paths_coh is None:
            paths_coh = sorted(set(root_dir.rglob(self.pattern_coh)))

        paths_unw = np.array([Path(i) for i in paths_unw], dtype=object)
        paths_coh = np.array([Path(i) for i in paths_coh], dtype=object)

        # Pairs: ensure there are no duplicate pairs
        # remove duplicate pairs
        paths_unw, pairs_unw = self._deduplicate_pairs(
            paths_unw,
            "unwrapped interferograms",
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
            warnings.warn(mismatch_info, stacklevel=2)
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
            lazy_loading=lazy_loading,
            chunks=chunks,
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
            lazy_loading=lazy_loading,
            chunks=chunks,
            pair_parser=self.parse_pairs,
        )
        self._ds_coh._range = self.coh_range

        _valid = self._valid & self._ds_coh.valid

        # remove invalid files for unw and coh
        self._files = self._files[_valid]
        self._valid = self._valid[_valid]
        self._ds_coh._files = self._ds_coh._files[_valid]
        self._ds_coh._valid = self._ds_coh._valid[_valid]

        # remove invalid pairs and refresh metadata
        self._assign_pairs_from_paths()

        self._ds_coh._assign_pairs_from_paths()

    def _deduplicate_pairs(
        self, paths: np.ndarray, dataset_name: str
    ) -> tuple[np.ndarray, Pairs]:
        """Remove duplicate pairs from the list of paths."""
        pairs = self.parse_pairs(paths)
        _, index = pairs.sort(inplace=False)
        if len(index) < len(paths):
            deduplicated = "".join(
                [f"\n\t{i.parent.stem}" for i in set(paths) - set(paths[index])],
            )
            warnings.warn(
                f"Duplicate pairs found in dataset {dataset_name}, "
                "keeping only the first occurrence"
                f"\nDeduplicate pairs: {deduplicated}",
                stacklevel=2,
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
        msg = "parse_baseline method must be implemented in subclass"
        raise NotImplementedError(msg)

    @property
    def coh_dataset(self) -> CoherenceDataset:
        """Return the coherence dataset."""
        return self._ds_coh

    @property
    def aps_dataset(self) -> RasterDataset | None:
        """Return the aps (Atmospheric Phase Screen) dataset.

        If None, no aps data is used.
        """
        return self._ds_aps

    @property
    def los_dataset(self) -> RasterDataset | None:
        """Return the theta dataset. If None, no theta data is used."""
        return self._ds_los

    @property
    def dem_dataset(self) -> RasterDataset | None:
        """Return the DEM dataset. If None, no DEM data is used."""
        return self._ds_dem

    @property
    def mask_dataset(self) -> RasterDataset | None:
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
        dataset: RasterDataset | None,
        ds_str: str,
        ds_class: type[RasterDataset] = RasterDataset,
        **kwargs,
    ) -> RasterDataset:
        """Ensure the dataset is an instance of ds_class.

        If dataset is None, a new ``ds_class`` object will be created using
        the kwargs.
        """
        if dataset is None:
            kwargs = self._ensure_ds_kwargs(kwargs)
            dataset = ds_class(**kwargs)
        elif not isinstance(dataset, ds_class):
            msg = f"{ds_str} must be an instance of {ds_class}, got {type(dataset)}"
            raise TypeError(
                msg,
            )
        return dataset

    def set_aps_dataset(
        self,
        aps_dataset: ApsPairs | None = None,
        **kwargs: dict,
    ) -> None:
        """Set the aps dataset.

        If aps_dataset is None, a new ApsPairs object will be created using the kwargs.

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
        los_dataset: RasterDataset | None = None,
        **kwargs: dict,
    ) -> None:
        """Set the los dataset.

        los file could be incidence angle (relative to vertical) or look angle
        (relative to horizontal). This file is used to convert differential
        atmospheric phase from vertical to line-of-sight (LOS) direction or
        convert LOS deformation phase to vertical.

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
        dem_dataset: RasterDataset | None = None,
        **kwargs: dict,
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
        mask_dataset: RasterDataset | None = None,
        **kwargs,
    ) -> None:
        """Set the mask dataset."""
        kwargs.setdefault("ds_name", "Mask")
        self._ds_mask = self._ensure_ds(mask_dataset, "mask_dataset", **kwargs)

    def load_los_ratio(
        self,
        roi: BoundingBox | None = None,
        angle_type: Literal["incidence", "look"] = "look",
    ) -> np.ndarray:
        """Load and convert los angle map to ratio map for given region of interest.

        The ratio map is used to convert differential atmospheric phase from
        vertical to line-of-sight (LOS) direction or convert LOS deformation
        phase to vertical.

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

    def get_nan_count(
        self,
        pairs: Pairs | None = None,
        roi: BoundingBox | None = None,
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
        nan_count = (self._file_query_bbox(roi, files[0]).squeeze(0).mask).astype(int)
        for f in tqdm(files[1:]):
            nan_count += (self._file_query_bbox(roi, f).squeeze(0).mask).astype(int)

        # reset fill_nodata to original value
        self.fill_nodata = fill_nodata

        return nan_count

    def to_netcdf(
        self,
        filename: PathLike,
        roi: BoundingBox | None = None,
        ref_points: Points | None = None,
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
            ds,
            ["unw", "coh"],
            crs=self.crs,
            x_dim="lon",
            y_dim="lat",
        )
        ds.to_netcdf(filename)

    def to_tiffs(
        self,
        out_dir: PathLike,
        roi: BoundingBox | None = None,
        ref_points: Points | None = None,
        pairs: Pairs | None = None,
        pdc: PhaseDeformationConverter | None = None,
        los_ratio: np.ndarray | None = None,
        names_unw: list[str] | None = None,
        names_coh: list[str] | None = None,
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
                msg = f"File {out_file} exists, skip"
                logger.info(msg)
                continue

            src = self._load_warp_file(f_unw)
            dest_arr = self._file_query_bbox(roi, src).squeeze(0)

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
                msg = f"File {out_file} exists, skip"
                logger.info(msg)
                continue

            src = self._load_warp_file(f_coh)
            dest_arr = self._file_query_bbox(roi, src).squeeze(0)

            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(dest_arr, 1)


class HierarchicalInterferogramDataset(InterferogramDataset):
    """A base class for hierarchical interferogram datasets.

    Hierarchical dataset is a dataset that contains multiple sub-datasets,
    like h5 and nc files.
    """

    #: pattern used to find files containing sub-datasets in the root directory
    pattern_files = "*"

    #: file type of the dataset, one of ['nc', 'h5']
    file_type: Literal["nc", "h5"] = "nc"

    #: group name of the unwrapped interferogram
    group_unw = "/science/grids/data/unwrappedPhase"
    #: group name of the coherence
    group_coh = "/science/grids/data/coherence"
    #: value range of coherence.
    coh_range: tuple[float, float] | None = [0, 1]

    _ds_coh: CoherenceDataset
    _ds_dem: RasterDataset | None = None
    _ds_mask: RasterDataset | None = None
    _ds_aps: RasterDataset | None = None

    def __init__(
        self,
        root_dir: str = "data",
        paths_unw: Iterable[PathLike] | None = None,
        paths_coh: Iterable[PathLike] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | None = None,
        roi: BoundingBox | None = None,
        bands_unw: Iterable[str] | None = None,
        bands_coh: Iterable[str] | None = None,
        cache: bool = True,
        resampling: Resampling = Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        keep_common: bool = True,
        lazy_loading: bool = False,
        chunks: dict[str, int] | None = None,
    ) -> None:
        """Initialize a new InterferogramDataset instance.

        Parameters
        ----------
        root_dir: str
            root_dir directory where dataset can be found.
        paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root_dir``. If None, files will be searched for in
            ``root_dir``.
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
            names of bands to return (defaults to all bands) for unwrapped
            interferograms.
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
        lazy_loading: bool, optional, default: False
            Enable lazy loading using dask arrays. When True, data is not loaded
            into memory until compute() is called. Default: False.
        chunks: dict[str, int] | None, optional
            Chunk sizes for dask arrays when lazy_loading is True.
            Example: {'y': 512, 'x': 512}. Default is None, which uses 512x512 chunks.

        """
        root_dir = Path(root_dir)

        if paths_unw is None and paths_coh is None:
            files = np.unique(list(root_dir.rglob(self.pattern_files)))
            paths_unw = self._get_sub_dataset_paths(files, self.group_unw)
            paths_coh = self._get_sub_dataset_paths(files, self.group_coh)

        super().__init__(
            root_dir=root_dir,
            paths_unw=paths_unw,
            paths_coh=paths_coh,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands_unw=bands_unw,
            bands_coh=bands_coh,
            cache=cache,
            resampling=resampling,
            fill_nodata=fill_nodata,
            verbose=verbose,
            keep_common=keep_common,
            lazy_loading=lazy_loading,
            chunks=chunks,
        )

    def _get_sub_dataset_paths(self, file_paths: list[str], group: str) -> list[str]:
        if self.file_type == "nc":
            paths = [f"netcdf:{path}:{group}" for path in file_paths]
        elif self.file_type == "h5":
            paths = [f"hdf5:{path}:{group}" for path in file_paths]
        else:
            msg = f"Invalid file type: {self.file_type}"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        return paths

"""Datasets for NISAR (NASA-ISRO SAR) products.

This module provides dataset classes for NISAR L2 GUNW (Geocoded Unwrapped
Interferogram) products distributed as HDF5 granules:

- :class:`NISAR` reads the unwrapped interferogram and coherence magnitude
  layers of GUNW granules as an interferogram pair dataset.
- :class:`NISARIonosphere` reads the ionospheric phase screen layers
  estimated by the NISAR processor.
- :meth:`NISAR.load_corrected_unw` reads ionosphere-corrected unwrapped
  interferograms directly. Both the unwrapped phase and the ionospheric
  phase screen are read internally and the correction is applied as
  ``unwrapped_phase - ionosphere_phase_screen``.

.. note::

    NISAR GUNW granules store the phase layers as plain HDF5 datasets with
    sibling ``xCoordinates``, ``yCoordinates`` and ``projection`` datasets.
    GDAL can read the raster layers through its HDF5 driver but does not
    attach the georeferencing automatically. This module generates inline
    GDAL VRT descriptions that reference the HDF5 layers and carry the
    correct geotransform and CRS, so the full :class:`RasterDataset` query
    machinery (points/boxes/polygons queries, warping, lazy loading) can be
    used transparently.

.. warning::

    Reading NISAR granules requires a GDAL build with HDF5 support (e.g.
    rasterio installed from conda-forge). Pip wheels of rasterio are built
    without the HDF5 driver and will not be able to open NISAR granules.

"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from faninsar._core.sar.baseline import Baselines
from faninsar._core.sar.pairs import Pairs
from faninsar.datasets.base import PairDataset
from faninsar.datasets.ifg import (
    CoherenceDataset,
    HierarchicalInterferogramDataset,
)
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from pyproj.crs import CRS
    from xarray import Dataset

    from faninsar.query import BoundingBox

logger = setup_logger(__name__)

__all__ = ["NISAR", "NISARIonosphere", "NisarCoherenceDataset"]

#: root group of the unwrapped interferogram layers in NISAR GUNW granules
NISAR_GUNW_IFG_ROOT = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"

#: mapping from numpy dtype names to GDAL VRT data type names
_NUMPY_TO_GDAL_DTYPE: dict[str, str] = {
    "float16": "Float32",
    "float32": "Float32",
    "float64": "Float64",
    "int8": "Byte",
    "uint8": "Byte",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "complex64": "CFloat32",
    "complex128": "CFloat64",
}


@cache
def _nisar_subdataset_xml(h5_file: str, group: str) -> str | None:
    """Generate an inline GDAL VRT XML string for a NISAR HDF5 layer.

    The VRT references the HDF5 dataset through the GDAL HDF5 driver and
    attaches the geotransform and CRS stored in the sibling coordinate
    datasets (``xCoordinates``, ``yCoordinates``, ``projection``).

    Parameters
    ----------
    h5_file : str
        path to the NISAR HDF5 granule.
    group : str
        full HDF5 path of the raster layer inside the granule.

    Returns
    -------
    str or None
        inline VRT XML string usable directly as a rasterio dataset name,
        or None if the layer cannot be read (e.g. missing georeferencing
        datasets or unsupported dtype).

    """
    try:
        import h5py

        group = group.lstrip("/")
        with h5py.File(h5_file, "r") as f:
            dataset = f[group]
            shape = dataset.shape
            dtype = dataset.dtype
            fill_value = dataset.attrs.get("_FillValue")
            parent = f[group.rsplit("/", 1)[0]]
            x = np.asarray(parent["xCoordinates"][:], dtype=np.float64)
            y = np.asarray(parent["yCoordinates"][:], dtype=np.float64)
            epsg = int(np.asarray(parent["projection"][()]).item())
    except Exception as e:
        msg = f"Unable to read georeferencing of {group!r} from {h5_file}: {e}"
        logger.warning(msg)
        return None

    if len(shape) != 2:
        msg = f"Dataset {group!r} in {h5_file} is not a 2D raster (shape={shape})"
        logger.warning(msg)
        return None
    if x.size < 2 or y.size < 2:
        msg = f"Coordinate arrays of {group!r} in {h5_file} are too short"
        logger.warning(msg)
        return None

    gdal_dtype = _NUMPY_TO_GDAL_DTYPE.get(str(dtype))
    if gdal_dtype is None:
        msg = f"Unsupported dtype {dtype} of {group!r} in {h5_file}"
        logger.warning(msg)
        return None

    height, width = shape
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    geotransform = (float(x[0]) - dx / 2, dx, 0.0, float(y[0]) - dy / 2, 0.0, dy)

    if fill_value is None:
        nodata = "nan"
    else:
        fill = float(np.asarray(fill_value).item())
        nodata = "nan" if np.isnan(fill) else repr(fill)

    return (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'
        f"  <SRS>EPSG:{epsg}</SRS>\n"
        f"  <GeoTransform>{geotransform[0]!r}, {geotransform[1]!r}, "
        f"{geotransform[2]!r}, {geotransform[3]!r}, {geotransform[4]!r}, "
        f"{geotransform[5]!r}</GeoTransform>\n"
        f'  <VRTRasterBand dataType="{gdal_dtype}" band="1">\n'
        f"    <NoDataValue>{nodata}</NoDataValue>\n"
        "    <SimpleSource>\n"
        f'      <SourceFilename relativeToVRT="0">HDF5:"{h5_file}"://{group}'
        "</SourceFilename>\n"
        "      <SourceBand>1</SourceBand>\n"
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
        "    </SimpleSource>\n"
        "  </VRTRasterBand>\n"
        "</VRTDataset>"
    )


class NisarH5Mixin:
    """Mixin adding georeferencing and pair parsing for NISAR HDF5 granules.

    This mixin enables the :class:`RasterDataset` machinery to work with
    NISAR GUNW granules by:

    1. Translating GDAL HDF5 subdataset paths (``hdf5:"file.h5"://group``)
       into inline VRT XML strings carrying the georeferencing stored in
       the granule. This is applied during file scanning
       (:meth:`_extract_single_file_metadata`), file loading
       (:meth:`_load_warp_file`), and lazy query window computation
       (:meth:`_open_query_source`).
    2. Parsing :class:`Pairs <faninsar._core.sar.Pairs>` from the
       reference/secondary zero-Doppler start times embedded in the granule
       file names.

    """

    #: pattern used to find NISAR GUNW granule files in ``root_dir``
    pattern_files: str = "NISAR_L2_PR_GUNW_*.h5"

    #: suffix of auxiliary files (e.g. QA statistics) that are not image granules
    qa_stats_suffix: str = "_QA_STATS.h5"

    #: template of the HDF5 group containing the interferogram layers
    ifg_group: str = NISAR_GUNW_IFG_ROOT + "/{polarization}"

    #: polarization used to locate layers inside the granule
    polarization: str = "HH"

    @staticmethod
    def _split_hdf5_subdataset(path: str | PathLike) -> tuple[str, str] | None:
        """Split a GDAL HDF5 subdataset path into (h5 file, group).

        Parameters
        ----------
        path : str or PathLike
            path to split. Valid HDF5 subdataset paths have the form
            ``hdf5:"file.h5"://group`` (the driver prefix is matched
            case-insensitively).

        Returns
        -------
        tuple[str, str] or None
            (h5 file path, HDF5 group), or None if ``path`` is not an HDF5
            subdataset path.

        """
        match = re.match(r'hdf5:"(.+?)":/(.+)$', str(path), re.IGNORECASE)
        if match is None:
            return None
        return match.group(1), "/" + match.group(2).lstrip("/")

    @staticmethod
    def _granule_name(path: str | PathLike) -> str:
        """Return the granule file name of a path or HDF5 subdataset path."""
        parts = NisarH5Mixin._split_hdf5_subdataset(path)
        source = parts[0] if parts is not None else str(path)
        return Path(source).name

    def _manifest_comparable_path(self, path: str) -> str | list[str]:
        """Return the backing granule path of an HDF5 subdataset URI."""
        parts = self._split_hdf5_subdataset(path)
        if parts is None:
            return str(path)
        return parts[0]

    @classmethod
    def _h5_files_from(cls, paths: Iterable[PathLike]) -> list[Path]:
        """Return the granule file paths of paths or HDF5 subdataset paths."""
        files = []
        for f in paths:
            parts = cls._split_hdf5_subdataset(f)
            files.append(Path(parts[0]) if parts is not None else Path(f))
        return files

    def _scan_granules(self, root_dir: str | PathLike) -> list[Path]:
        """Find NISAR granule files in a directory, excluding auxiliary files.

        Parameters
        ----------
        root_dir : str or PathLike
            directory to search recursively for granule files.

        Returns
        -------
        list[Path]
            sorted paths of image granules. Auxiliary files (e.g.
            ``*_QA_STATS.h5``) are excluded.

        """
        return [
            f
            for f in sorted(set(Path(root_dir).rglob(self.pattern_files)))
            if not f.name.endswith(self.qa_stats_suffix)
        ]

    def subdataset_path(self, h5_file: str | PathLike, layer: str) -> str:
        """Return the GDAL HDF5 subdataset path of a layer in a granule.

        Parameters
        ----------
        h5_file : str or PathLike
            path to the NISAR HDF5 granule.
        layer : str
            name of the layer inside the interferogram group, e.g.
            ``"unwrappedPhase"`` or ``"ionospherePhaseScreen"``.

        Returns
        -------
        str
            GDAL HDF5 subdataset path (``hdf5:"file.h5"://group``).

        """
        group = f"{self.ifg_group.format(polarization=self.polarization)}/{layer}"
        return f'hdf5:"{Path(h5_file).resolve()}"://{group}'

    @classmethod
    def _to_vrt_source(cls, path: str | PathLike) -> str | None:
        """Return the inline VRT XML of an HDF5 subdataset path.

        Parameters
        ----------
        path : str or PathLike
            path to convert. Non-HDF5 subdataset paths are returned
            unchanged.

        Returns
        -------
        str or None
            inline VRT XML string, ``path`` itself for non-HDF5 subdataset
            paths, or None if the georeferencing of the layer cannot be
            read.

        """
        parts = cls._split_hdf5_subdataset(path)
        if parts is None:
            return str(path)
        return _nisar_subdataset_xml(*parts)

    @classmethod
    def _extract_single_file_metadata(
        cls, file_path: str, target_crs: CRS | None = None
    ) -> dict:
        """Extract metadata of a file with NISAR georeferencing attached.

        Parameters
        ----------
        file_path : str
            path to the file (HDF5 subdataset paths are supported).
        target_crs : CRS, optional
            target CRS used to compute transformed bounds and resolution.

        Returns
        -------
        dict
            file metadata, see
            :meth:`RasterDataset._extract_single_file_metadata`.

        """
        vrt_source = cls._to_vrt_source(file_path)
        if vrt_source is None:
            return super()._extract_single_file_metadata(file_path, target_crs)
        metadata = super()._extract_single_file_metadata(vrt_source, target_crs)
        metadata["paths"] = str(file_path)
        return metadata

    def _load_warp_file(self, file_path: str) -> rasterio.DatasetReader:
        """Load a file as a georeferenced (optionally warped) dataset.

        Parameters
        ----------
        file_path : str
            path to the file (HDF5 subdataset paths are supported).

        Returns
        -------
        rasterio.DatasetReader
            file handle of the opened file or warped VRT.

        """
        vrt_source = self._to_vrt_source(file_path)
        if vrt_source is None:
            return super()._load_warp_file(file_path)
        src = rasterio.open(vrt_source)
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        return src

    def _open_query_source(self, path: str) -> rasterio.DatasetReader:
        """Open a georeferenced source used to compute query windows.

        Parameters
        ----------
        path : str
            path to the file (HDF5 subdataset paths are supported).

        Returns
        -------
        rasterio.DatasetReader
            opened dataset carrying the NISAR georeferencing.

        """
        vrt_source = self._to_vrt_source(path)
        if vrt_source is None:
            return super()._open_query_source(path)
        return rasterio.open(vrt_source)

    def _to_read_path(self, path: str) -> str:
        """Return the path used by lazy readers to open a file.

        The inline VRT XML is returned for HDF5 subdataset paths so lazy
        readers (dask workers) can open a georeferenced source directly.
        This is required both to read NISAR layers with valid georeferencing
        and to support lazy queries with reprojection (``warp_crs``). If the
        VRT cannot be generated, the double-slash HDF5 subdataset syntax
        (restored after ``pathlib`` normalization) is returned.

        Parameters
        ----------
        path : str
            path stored in the dataset files metadata.

        Returns
        -------
        str
            path that can be opened directly by rasterio.

        """
        parts = self._split_hdf5_subdataset(path)
        if parts is None:
            return super()._to_read_path(path)
        vrt_source = self._to_vrt_source(path)
        if vrt_source is not None and vrt_source != str(path):
            return vrt_source
        return f'hdf5:"{parts[0]}"://{parts[1].lstrip("/")}'

    @classmethod
    def parse_pairs(cls, paths: Iterable[str | PathLike]) -> Pairs:
        """Parse pairs from NISAR granule file names.

        The reference and secondary dates are taken from the zero-Doppler
        start times embedded in the granule file names
        (``..._SH_YYYYMMDDThhmmss_..._YYYYMMDDThhmmss_...``).

        Parameters
        ----------
        paths : list of str or PathLike
            list of granule file paths or HDF5 subdataset paths.

        Returns
        -------
        pairs : Pairs
            pairs parsed from the file names.

        Raises
        ------
        ValueError
            if the dates cannot be parsed from a file name.

        """
        pair_names = []
        for path in paths:
            name = cls._granule_name(path)
            dates = re.findall(r"(\d{8})T\d{6}", name)
            if len(dates) < 3:
                msg = (
                    f"Unable to parse reference/secondary dates from {name!r}. "
                    "The file name does not contain the NISAR granule "
                    "_SH_YYYYMMDDThhmmss_..._YYYYMMDDThhmmss_... pattern."
                )
                logger.error(msg)
                raise ValueError(msg)
            pair_names.append(f"{dates[0]}_{dates[2]}")
        return Pairs.from_names(pair_names)


class NisarCoherenceDataset(NisarH5Mixin, CoherenceDataset):
    """A dataset manages the coherence magnitude of NISAR GUNW products.

    The dataset reads the coherence magnitude layer
    (``unwrappedInterferogram/<pol>/coherenceMagnitude``) of NISAR L2 GUNW
    granules distributed as HDF5 files. It is used as the coherence dataset
    of :class:`NISAR` and can also be used standalone.
    """

    #: name of the coherence magnitude layer
    layer_coh = "coherenceMagnitude"


class NISAR(NisarH5Mixin, HierarchicalInterferogramDataset):
    """A dataset manages the data of NISAR L2 GUNW products.

    The dataset reads the unwrapped interferogram
    (``unwrappedInterferogram/<pol>/unwrappedPhase``) and coherence magnitude
    (``unwrappedInterferogram/<pol>/coherenceMagnitude``) layers of NISAR L2
    GUNW granules distributed as HDF5 files. The ionospheric phase screen
    (``unwrappedInterferogram/<pol>/ionospherePhaseScreen``) estimated by the
    NISAR processor can be accessed through :attr:`iono_dataset`, and
    ionosphere-corrected unwrapped interferograms can be loaded with
    :meth:`load_corrected_unw`.

    .. note::
        The estimated ionospheric phase screen is included in GUNW products
        but is not applied to the unwrapped phase. The corrected phase is
        therefore computed as ``unwrapped_phase - ionosphere_phase_screen``.

    Examples
    --------
    >>> ds = NISAR("/path/to/nisar_gunw")  # doctest: +SKIP
    >>> ds.pairs  # doctest: +SKIP
    >>> data = ds[ds.roi]  # doctest: +SKIP

    """

    #: type of the files containing sub-datasets, used by parent classes
    file_type: Literal["nc", "h5"] = "h5"

    #: value range of the coherence magnitude
    coh_range: tuple[float, float] = (0, 1)

    #: class used to create the coherence dataset
    _coh_dataset_cls: type[NisarCoherenceDataset] = NisarCoherenceDataset

    #: name of the unwrapped phase layer
    layer_unw = "unwrappedPhase"

    #: name of the coherence magnitude layer
    layer_coh = "coherenceMagnitude"

    #: name of the ionospheric phase screen layer
    layer_iono = "ionospherePhaseScreen"

    _iono_dataset: NISARIonosphere | None = None

    def __init__(
        self,
        root_dir: str | PathLike = "data",
        paths_unw: Iterable[PathLike] | None = None,
        paths_coh: Iterable[PathLike] | None = None,
        polarization: str = "HH",
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
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
        reindex: bool = False,
    ) -> None:
        """Initialize a new NISAR dataset instance.

        Parameters
        ----------
        root_dir : str or PathLike
            root directory where the NISAR GUNW granules can be found.
        paths_unw : list of str, optional
            list of unwrapped phase layer paths (granule paths or HDF5
            subdataset paths) to use instead of searching for granules in
            ``root_dir``. If None, granules are searched for in ``root_dir``.
        paths_coh : list of str, optional
            list of coherence magnitude layer paths to use instead of
            searching for granules in ``root_dir``. If None, granules are
            searched for in ``root_dir``.
        polarization : str, optional
            polarization used to locate layers inside the granules, e.g.
            ``"HH"`` or ``"VV"``. Default is ``"HH"``.
        crs : CRS, optional
            the output coordinate reference system of the dataset. If None,
            the CRS of the first granule found will be used.
        res : float or tuple of float, optional
            resolution of the output dataset in units of CRS. If None, the
            resolution of the first granule found will be used.
        dtype : numpy.dtype, optional
            data type of the output dataset. If None, the data type of the
            first granule found will be used.
        nodata : float or int, optional
            no data value of the output dataset. If None, the no data value
            stored in the granule will be used.
        roi : BoundingBox, optional
            region of interest to load from the dataset. If None, the union
            of all granule bounds in the dataset will be used.
        bands_unw : list of str, optional
            names of bands to return (defaults to all bands) for unwrapped
            interferograms.
        bands_coh : list of str, optional
            names of bands to return (defaults to all bands) for coherence.
        cache : bool, optional
            if True, cache file handle to speed up repeated sampling.
        resampling : Resampling, optional
            resampling algorithm used when reading input files. Default is
            ``Resampling.nearest``.
        fill_nodata : bool, optional
            whether to fill holes in the queried data by interpolating them
            using inverse distance weighting. Default is False.
        verbose : bool, optional
            if True, print verbose output. Default is True.
        keep_common : bool, optional
            only used when the number of unwrapped and coherence layers are
            not equal. If True, keep the common pairs and raise a warning.
            If False, raise an error. Default is True.
        chunks : dict[str, int] | int | Literal["auto"] | None, optional
            chunk sizes for dask arrays. Controls lazy vs eager loading:
            - ``None`` (default): Load data eagerly into memory.
            - ``"auto"`` or ``{}``: Lazy loading with auto-detected block size.
            - ``int``: Lazy loading with same chunk size for y and x.
            - ``dict``: Lazy loading with specified y and x chunk sizes.
        reindex : bool, optional
            If True, ignore any existing ``faninsar.manifest.json`` entry and
            rescan all granules, rewriting the entries. If False (default),
            reuse previously written manifest entries when they are present.

        """
        self.polarization = polarization.upper()

        if paths_unw is None and paths_coh is None:
            h5_files = self._scan_granules(root_dir)
            if len(h5_files) == 0:
                msg = (
                    f"No NISAR GUNW granules (pattern={self.pattern_files!r}) were "
                    f"found in root_dir='{root_dir}'"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)
        else:
            h5_files = self._h5_files_from(paths_unw)

        paths_unw = [self.subdataset_path(f, self.layer_unw) for f in h5_files]
        paths_coh = [self.subdataset_path(f, self.layer_coh) for f in h5_files]

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
            chunks=chunks,
            reindex=reindex,
        )

    @property
    def h5_files(self) -> list[Path]:
        """Return the paths of the granules used by the dataset."""
        return self._h5_files_from(self.files.paths)

    def set_iono_dataset(
        self,
        iono_dataset: NISARIonosphere | None = None,
        **kwargs,
    ) -> None:
        """Set the ionospheric phase screen dataset.

        If ``iono_dataset`` is None, a new :class:`NISARIonosphere` object
        will be created from the granules used by this dataset.

        Parameters
        ----------
        iono_dataset : NISARIonosphere, optional
            dataset containing the ionospheric phase screens. If None, a new
            dataset is created using the granules of this dataset.
        **kwargs : dict, optional
            keyword arguments used to create a new :class:`NISARIonosphere`
            object if ``iono_dataset`` is None.

        """
        kwargs.setdefault("root_dir", self.root_dir)
        kwargs.setdefault(
            "paths", [self.subdataset_path(f, self.layer_iono) for f in self.h5_files]
        )
        kwargs.setdefault("crs", self.crs)
        kwargs.setdefault("res", self.res)
        kwargs.setdefault("roi", self.roi)
        kwargs.setdefault("cache", self.cache)
        kwargs.setdefault("resampling", self.resampling)
        kwargs.setdefault("fill_nodata", self.fill_nodata)
        kwargs.setdefault("verbose", self.verbose)
        kwargs.setdefault("chunks", self._chunks)
        self._iono_dataset = self._ensure_ds(
            iono_dataset, "iono_dataset", NISARIonosphere, **kwargs
        )

    @property
    def iono_dataset(self) -> NISARIonosphere:
        """Return the ionospheric phase screen dataset.

        The dataset is created from the granules used by this dataset the
        first time this property is accessed.
        """
        if self._iono_dataset is None:
            self.set_iono_dataset()
        return self._iono_dataset

    def load_corrected_unw(
        self,
        roi: BoundingBox | None = None,
        pairs: Pairs | None = None,
        chunks: dict[str, int] | int | Literal["auto", False] | None = None,
        verbose: bool | None = None,
        return_components: bool = False,
    ) -> Dataset:
        """Load ionosphere-corrected unwrapped interferograms.

        The unwrapped phase and the ionospheric phase screen are read
        internally for the given region and pairs, and the corrected phase
        is computed as ``unwrapped_phase - ionosphere_phase_screen``.

        Parameters
        ----------
        roi : BoundingBox, optional
            region of interest to load. If None, the roi of the dataset will
            be used.
        pairs : Pairs, optional
            pairs to load. If None, all pairs will be loaded.
        chunks : dict[str, int] | int | Literal["auto", False] | None, optional
            chunk sizes for dask arrays. None uses the dataset setting,
            False forces eager loading. See :meth:`boxes_query`.
        verbose : bool, optional
            override verbose setting for this query. None uses self.verbose.
        return_components : bool, optional
            if True, the returned dataset also contains the original
            unwrapped phase (``"unw"``) and the ionospheric phase screen
            (``"iono"``). Default is False.

        Returns
        -------
        corrected : xarray.Dataset
            dataset containing the corrected unwrapped phase in the
            ``"unw_corrected"`` variable (and additionally ``"unw"`` and
            ``"iono"`` if ``return_components`` is True), with dimensions
            ``(pair, y, x)``.

        Notes
        -----
        The ionospheric phase screen is included in GUNW products but is not
        applied to the unwrapped phase, so the corrected phase is
        ``unwrapped_phase - ionosphere_phase_screen``.

        """
        if roi is None:
            roi = self.roi

        ds_iono = self.iono_dataset
        unw = self.boxes_query(roi, pairs=pairs, chunks=chunks, verbose=verbose)
        iono = ds_iono.boxes_query(roi, pairs=pairs, chunks=chunks, verbose=verbose)

        unw_da = unw.dataset["data"]
        iono_da = iono.dataset["data"]
        unw_da, iono_da = xr.align(unw_da, iono_da, join="inner")
        if len(unw_da.pair) == 0:
            msg = (
                "No common pairs between the unwrapped interferogram dataset "
                "and the ionosphere dataset."
            )
            logger.error(msg)
            raise ValueError(msg)

        corrected = unw_da - iono_da
        corrected.name = "unw_corrected"
        corrected.attrs["units"] = "radians"
        corrected.attrs["long_name"] = (
            "ionosphere-corrected unwrapped interferometric phase"
        )

        data_vars: dict[str, xr.DataArray] = {"unw_corrected": corrected}
        if return_components:
            unw_da.name = "unw"
            iono_da.name = "iono"
            data_vars["unw"] = unw_da
            data_vars["iono"] = iono_da

        ds = xr.Dataset(data_vars)
        for name in data_vars:
            ds[name] = ds[name].fis.set_spatial_ref(
                crs=self.crs,
                x_dim="x",
                y_dim="y",
            )
        return ds

    def parse_baselines(self, pairs: Pairs | None = None) -> Baselines:
        """Parse the temporal baseline of pairs from the acquisition dates.

        NISAR GUNW products do not store perpendicular baselines, so the
        temporal baseline (days relative to the first acquisition) is used.

        Parameters
        ----------
        pairs : Pairs, optional
            pairs which the baseline will be parsed for. If None, all pairs
            will be parsed.

        Returns
        -------
        baselines : Baselines
            baselines with values being the number of days relative to the
            first acquisition date.

        """
        pairs = pairs if pairs is not None else self.pairs
        dates = pd.to_datetime(pairs.dates)
        values = np.asarray((dates - dates[0]) / pd.Timedelta(days=1), dtype=np.float32)
        return Baselines(dates, values)


class NISARIonosphere(NisarH5Mixin, PairDataset):
    """A dataset manages the ionospheric phase screens of NISAR GUNW products.

    The dataset reads the ionospheric phase screen layer
    (``unwrappedInterferogram/<pol>/ionospherePhaseScreen``) estimated by the
    NISAR processor from GUNW granules distributed as HDF5 files. The screen
    is not applied to the unwrapped phase in GUNW products and can be
    subtracted from the unwrapped phase to correct the ionospheric contribution.

    Examples
    --------
    >>> ds = NISARIonosphere("/path/to/nisar_gunw")  # doctest: +SKIP
    >>> ds.pairs  # doctest: +SKIP
    >>> data = ds[ds.roi]  # doctest: +SKIP

    """

    #: name of the ionospheric phase screen layer
    layer_iono = "ionospherePhaseScreen"

    def __init__(
        self,
        root_dir: str | PathLike = "data",
        paths: Iterable[PathLike] | None = None,
        polarization: str = "HH",
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | None = None,
        roi: BoundingBox | None = None,
        bands: Iterable[str] | None = None,
        cache: bool = True,
        resampling: Resampling = Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "NISARIonosphere",
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
        reindex: bool = False,
    ) -> None:
        """Initialize a new NISARIonosphere dataset instance.

        Parameters
        ----------
        root_dir : str or PathLike
            root directory where the NISAR GUNW granules can be found.
        paths : list of str, optional
            list of ionospheric phase screen layer paths (granule paths or
            HDF5 subdataset paths) to use instead of searching for granules
            in ``root_dir``. If None, granules are searched for in
            ``root_dir``.
        polarization : str, optional
            polarization used to locate layers inside the granules, e.g.
            ``"HH"`` or ``"VV"``. Default is ``"HH"``.
        crs : CRS, optional
            the output coordinate reference system of the dataset. If None,
            the CRS of the first granule found will be used.
        res : float or tuple of float, optional
            resolution of the output dataset in units of CRS. If None, the
            resolution of the first granule found will be used.
        dtype : numpy.dtype, optional
            data type of the output dataset. If None, the data type of the
            first granule found will be used.
        nodata : float or int, optional
            no data value of the output dataset. If None, the no data value
            stored in the granule will be used.
        roi : BoundingBox, optional
            region of interest to load from the dataset. If None, the union
            of all granule bounds in the dataset will be used.
        bands : list of str, optional
            names of bands to return (defaults to all bands).
        cache : bool, optional
            if True, cache file handle to speed up repeated sampling.
        resampling : Resampling, optional
            resampling algorithm used when reading input files. Default is
            ``Resampling.nearest``.
        fill_nodata : bool, optional
            whether to fill holes in the queried data by interpolating them
            using inverse distance weighting. Default is False.
        verbose : bool, optional
            if True, print verbose output. Default is True.
        ds_name : str, optional
            name of the dataset used for printing verbose output.
        chunks : dict[str, int] | int | Literal["auto"] | None, optional
            chunk sizes for dask arrays. Controls lazy vs eager loading:
            - ``None`` (default): Load data eagerly into memory.
            - ``"auto"`` or ``{}``: Lazy loading with auto-detected block size.
            - ``int``: Lazy loading with same chunk size for y and x.
            - ``dict``: Lazy loading with specified y and x chunk sizes.
        reindex : bool, optional
            If True, ignore any existing ``faninsar.manifest.json`` entry and
            rescan all granules, rewriting the entries. If False (default),
            reuse previously written manifest entries when they are present.

        Notes
        -----
        The dataset keeps only one granule per interferometric pair. If
        multiple granules share the same reference/secondary date pair
        (e.g. adjacent frames), only the first occurrence is kept and a
        warning is raised.

        """
        self.polarization = polarization.upper()

        if paths is None:
            h5_files = self._scan_granules(root_dir)
            if len(h5_files) == 0:
                msg = (
                    f"No NISAR GUNW granules (pattern={self.pattern_files!r}) were "
                    f"found in root_dir='{root_dir}'"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)
        else:
            h5_files = self._h5_files_from(paths)

        subdataset_paths = [self.subdataset_path(f, self.layer_iono) for f in h5_files]

        # keep only one granule per pair (see Notes)
        pairs = self.parse_pairs(subdataset_paths)
        _, index = pairs.sort(inplace=False)
        if len(index) < len(subdataset_paths):
            removed = "\n\t".join(
                [h5_files[i].name for i in set(range(len(h5_files))) - set(index)]
            )
            msg = (
                "Duplicate pairs found in dataset "
                f"{ds_name}, keeping only the first occurrence"
                f"\nRemoved granules: {removed}"
            )
            logger.warning(msg)
        subdataset_paths = [subdataset_paths[i] for i in index]

        super().__init__(
            root_dir=root_dir,
            paths=subdataset_paths,
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
            chunks=chunks,
            reindex=reindex,
        )

"""Hierarchical container integration built on :mod:`xarray`.

Notes
-----
The original :class:`HierarchicalDataset` relied on GDAL subdatasets and
delegated all raster access to :class:`RasterDataset`. This refactored version
employs :class:`XarrayDataset` so that hierarchical formats benefit from the
same lazy Dask pipeline and GeoBox metadata planning as standalone GeoTIFF
inputs.

See Also
--------
faninsar.datasets.xarray_dataset
    Base implementation that performs metadata parsing and lazy graph
    construction.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from rasterio.enums import Resampling

from faninsar.datasets.xarray_dataset import XarrayDataset
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from rasterio.crs import CRS

    from faninsar.typing import ResamplingLike

logger = setup_logger(__name__)


class HierarchicalDataset(XarrayDataset):
    """Raster access for hierarchical container formats (NetCDF/HDF5/Zarr).

    Parameters
    ----------
    paths : Iterable[str | Path]
        Container file paths to open.
    crs : CRS | None, optional
        Desired output CRS override. Defaults to the CRS discovered from source
        rasters.
    res : float | tuple[float, float] | None, optional
        Output resolution override. Scalars are interpreted as square pixels.
    nodata : float | int | None, optional
        NoData override forwarded to :class:`XarrayDataset`.
    resampling : Resampling, optional
        Resampling strategy for lazy reprojection. Only
        :class:`~rasterio.enums.Resampling.nearest` is currently supported.
    verbose : bool, optional
        Whether to log warnings when containers cannot be read.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`xarray.open_dataset`.

    Raises
    ------
    ValueError
        If no paths are provided or containers cannot be read.
    NotImplementedError
        If unsupported feature toggles are requested.

    See Also
    --------
    XarrayDataset
        Parent class providing implementation details.

    """

    # Subclass attributes to be overridden
    group: str = ""
    """Default group path used for data access."""

    var: str | None = None
    """variable name within the group"""

    x_dim: str | None = None
    """name of horizontal dimension within the variable"""

    y_dim: str | None = None
    """name of vertical dimension within the variable"""

    def __init__(
        self,
        paths: Iterable[str | Path] | None = None,
        *,
        root_dir: str | Path = "data",
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        nodata: float | None = None,
        resampling: ResamplingLike = Resampling.nearest,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the hierarchical dataset."""
        # Discover container paths if not provided
        if paths is None:
            paths = self._discover_containers(root_dir)

        paths = [Path(p) for p in paths]
        if not paths:
            msg = "No container files found or provided."
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(
            "Initialising %s with %d containers", self.__class__.__name__, len(paths)
        )

        # Initialize parent class with class attributes
        super().__init__(
            paths=paths,
            out_crs=crs,
            out_res=res,
            out_nodata=nodata,
            group=self.group,
            variable=self.var,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            resampling=resampling,
            verbose=verbose,
            **kwargs,
        )

    def _discover_containers(
        self,
        root_dir: str | Path,
    ) -> list[Path]:
        """Discover container files in the specified directory.

        Parameters
        ----------
        root_dir : str | Path
            Root directory for discovery.

        Returns
        -------
        list[Path]
            Discovered container paths.

        """
        root = Path(root_dir)
        return sorted(root.rglob("*"))

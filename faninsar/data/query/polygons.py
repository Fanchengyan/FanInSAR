"""Polygons class for storing desired and undesired regions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd
from pyproj.crs.crs import CRS
from rasterio.errors import CRSError

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from matplotlib.axes import Axes

    from faninsar.core.types import CrsLike

    from .bbox import BoundingBox

logger = setup_logger(__name__)


class Polygons:
    """Polygons object is used to store the desired and undesired regions.

    desired regions will be retrieved from the dataset, while undesired regions
    will be removed from the dataset.

    .. tip::
        When a mixed-types Polygons, where both "desired" and "undesired"
        polygons, are provided:

        * the "undesired" polygons will only be useful when there are overlapping
          regions with the "desired" polygons. Otherwise, the "desired" polygons
          are enough.
        * you can use :meth:`to_desired` to get the desired polygons from a
          mixed-types Polygons object.

    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame | gpd.GeoSeries,
        types: (
            Literal["desired", "undesired"] | Sequence[Literal["desired", "undesired"]]
        ) = "desired",
        crs: CrsLike = None,
        all_touched: bool = True,
        pad: bool = False,
    ) -> None:
        """Initialize a Polygons object.

        Parameters
        ----------
        gdf : GeoDataFrame | GeoSeries
            The GeoDataFrame or GeoSeries containing the "desired" or "undesired"
            regions/polygons.
        types : 'desired' | 'undesired' | Sequence['desired', 'undesired']
            The types of polygons. If 'desired', the polygons are the desired
            regions. If 'undesired', the polygons are the regions to be removed.
            Default is 'desired'.
        crs : Any, optional
            The CRS of the polygons. Can be any object that can be passed to
            :meth:`pyproj.crs.CRS.from_user_input` .
            If None, the CRS of the input geometry will be used. Default is None.
        all_touched : bool, optional
            Whether to include all pixels touched by the polygon. Default is True.
        pad : bool, optional
            If True, the features will be padded in each direction by
            one half of a pixel prior to cropping raster. Defaults to False.

        """
        self._gdf: gpd.GeoDataFrame = self._format_geometry(
            gdf, types, crs
        ).sort_values(
            by="types",
            ascending=True,
        )
        self.pad = pad
        self.all_touched = all_touched

    def __str__(self) -> str:
        """Return a string representation of the Polygons object."""
        crs = self.crs.to_string() if self.crs else None
        return f"Polygons(count={len(self)}, crs='{crs}')"

    def __repr__(self) -> str:
        """Return a string representation of the Polygons object."""
        prefix = "Polygons:\n"
        middle = self._gdf.__repr__()
        crs = self.crs.to_string() if self.crs else None
        suffix = f"\n[count={len(self)}, crs='{crs}']"

        return f"{prefix}{middle}{suffix}"

    def __len__(self) -> int:
        """Return the number of polygons."""
        return len(self._gdf)

    def __add__(self, other: Polygons) -> Polygons:
        """Add two Polygons objects."""
        if not isinstance(other, Polygons):
            msg = f"other must be an instance of Polygon. Got {type(other)}"
            raise TypeError(msg)

        if self.crs != other.crs:
            if self.crs is None or other.crs is None:
                warnings.warn(
                    "CRS is found lacking, adding polygons without CRS.",
                    stacklevel=2,
                )
            else:
                other = other.to_crs(self.crs)

        gdf = pd.concat([self.geodataframe, other.geodataframe], ignore_index=True)
        return Polygons(gdf, types=gdf.types)

    def _format_geometry(
        self,
        gdf: gpd.GeoDataFrame | gpd.GeoSeries | Polygons,
        types: (
            Literal["desired", "undesired"] | Sequence[Literal["desired", "undesired"]]
        ),
        crs: CrsLike,
    ) -> gpd.GeoDataFrame:
        """Format the geometry column of the GeoDataFrame."""
        if isinstance(gdf, gpd.GeoDataFrame):
            df = gpd.GeoDataFrame(gdf.geometry)
            df["types"] = types
        elif isinstance(gdf, gpd.GeoSeries):
            df = gpd.GeoDataFrame(gdf)
            df["types"] = types
        elif isinstance(gdf, Polygons):
            df = gdf.geodataframe
        else:
            msg = f"gdf must be an instance of GeoDataFrame, GeoSeries. Got {type(gdf)}"
            logger.error(msg, stacklevel=2)
            raise TypeError(msg)
        return self._ensure_gdf_crs(df, crs)

    @staticmethod
    def _ensure_gdf_crs(
        gdf: gpd.GeoDataFrame,
        crs: CrsLike,
    ) -> gpd.GeoDataFrame:
        """Ensure the CRS of the GeoDataFrame."""
        if crs is None and gdf.crs is None:
            warnings.warn(
                "CRS is not found both in input geometries and parameters."
                " Set to None.",
                stacklevel=2,
            )
        else:
            if crs is None:
                return gdf
            if not isinstance(crs, CRS):
                crs = CRS.from_user_input(crs)
            if gdf.crs is None:
                gdf = gdf.set_crs(crs, inplace=False)
            elif gdf.crs != crs:
                gdf = gdf.to_crs(crs)
        return gdf

    @property
    def all_touched(self) -> bool:
        """Whether to include all pixels touched by the polygon."""
        return self._all_touched

    @all_touched.setter
    def all_touched(self, value: bool) -> None:
        if not isinstance(value, bool):
            msg = f"all_touched must be a bool. Got {type(value)}"
            raise TypeError(msg)
        self._all_touched = value

    @property
    def pad(self) -> bool:
        """Whether to pad the features in each direction by one half of a pixel.

        This is used prior to cropping raster. Defaults to False.
        """
        return self._pad

    @pad.setter
    def pad(self, value: bool) -> None:
        if not isinstance(value, bool):
            msg = f"pad must be a bool. Got {type(value)}"
            raise TypeError(msg)
        self._pad = value

    @property
    def geometry(self) -> gpd.GeoSeries:
        """The geometry column of the polygons."""
        return self._gdf.geometry

    @property
    def types(self) -> pd.Series:
        """The types of polygons."""
        return self._gdf["types"]

    @property
    def geodataframe(self) -> gpd.GeoDataFrame:
        """GeoDataFrame format of polygons."""
        return self._gdf

    @property
    def desired(self) -> Polygons:
        """Desired part of polygons."""
        return Polygons(self._gdf[self.types == "desired"], types="desired")

    @property
    def undesired(self) -> Polygons:
        """Undesired part of polygons."""
        return Polygons(self._gdf[self.types == "undesired"], types="undesired")

    @property
    def is_mixed(self) -> bool:
        """Whether the polygons contain both desired and undesired polygons."""
        return len(self.desired) > 0 and len(self.undesired) > 0

    def to_desired(self) -> Polygons:
        """Return a desired polygons, with regions of undesired polygons being removed.

        .. Warning::
            This method should only be used when the Polygons object contains both
            "desired" and "undesired" polygons. If the Polygons object only contains
            "undesired" polygons, the returned Polygons object will be empty.
        """
        _df = gpd.overlay(
            self.desired.geodataframe, self.undesired.geodataframe, how="difference"
        )
        return Polygons(_df, types="desired")

    def to_bbox(self) -> list[BoundingBox]:
        """Return a list of BoundingBox representing bounding boxes of polygons.

        .. Warning::
            This method will only return the bounding boxes of the desired polygons.
            If the Polygons object only contains "undesired" polygons, the returned
            list will be empty.
        """
        return self.to_desired().geodataframe

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame of the polygons.

        This method is an alias of :attr:`geodataframe` for API consistency with
        :class:`~faninsar.data.query.Points` and :class:`BoundingBox`.
        """
        return self.geodataframe

    @property
    def crs(self) -> CRS:
        """The CRS of the polygons."""
        return self._gdf.crs

    def to_crs(self, crs: CrsLike) -> Polygons:
        """Return a new Polygons object with new CRS.

        Parameters
        ----------
        crs : Any
            The new CRS. Can be any object that can be passed to
            :meth:`pyproj.crs.CRS.from_user_input`.

        Returns
        -------
        Polygons
            The new Polygons object.

        """
        if not isinstance(crs, CRS):
            crs = CRS.from_user_input(crs)
        if self.crs == crs:
            return self
        gdf = self._gdf.to_crs(crs)
        return Polygons(gdf, types=self.types)

    def set_crs(
        self,
        crs: CrsLike,
        allow_override: bool = False,
    ) -> None:
        """Set the CRS of polygons.

        .. warning::
            This method will only set the crs attribute without converting the
            geometries to a new coordinate reference system. If you want to convert
            the geometries to a new coordinate, please use :meth:`to_crs`

        Parameters
        ----------
        crs : Any
            The new CRS. Can be any object that can be passed to
            :meth:`pyproj.crs.CRS.from_user_input`.
        allow_override : bool, optional
            Whether to allow overriding the existing CRS. If False, a CRSError
            will be raised if the CRS has already been set. Default is False.

        Raises
        ------
        CRSError
            If the CRS has already been set and allow_override is False.

        """
        if not isinstance(crs, CRS):
            crs = CRS.from_user_input(crs)
        if self.crs != crs:
            if self.crs is None or allow_override:
                self._gdf.set_crs(crs, allow_override=True)
            else:
                msg = (
                    "The CRS has already been set. Set allow_override=True to override."
                )
                raise CRSError(msg)

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        types: (
            Literal["desired", "undesired"] | Sequence[Literal["desired", "undesired"]]
        ) = "desired",
        crs: CrsLike = None,
        **kwargs,
    ) -> Polygons:
        """Initialize a Polygon object from a shapefile.

        Parameters
        ----------
        filename : PathLike
            The path to the shapefile. file type can be any type that can be
            passed to :func:`geopandas.read_file`.
        types : 'desired' | 'undesired' | Sequence['desired', 'undesired'], optional
            The types of polygons. If 'desired', the polygons are the desired
            polygons. If 'undesired', the polygons are the polygons to be removed.
            Default is 'desired'.
        crs : Any, optional
            The CRS of the polygons. Can be any object that can be passed to
            :meth:`pyproj.crs.CRS.from_user_input`.
            If None, the CRS of the input geometries will be used. Default is None.
        **kwargs : dict
            Other parameters passed to :func:`geopandas.read_file`.

        Returns
        -------
        Polygons
            The Polygons object.

        """
        kwargs.update({"ignore_geometry": False})
        gdf = gpd.read_file(filename, **kwargs)
        return cls(gdf, types=types, crs=crs)

    def copy(self) -> Polygons:
        """Return a copy of the Polygons object."""
        return Polygons(self._gdf.copy(), types=self.types)

    def plot(self, **kwargs) -> Axes:
        """Plot the polygons on a map.

        Parameters
        ----------
        **kwargs : dict
            Other parameters passed to :meth:`geopandas.GeoDataFrame.plot`.

        Returns
        -------
        Axes
            The matplotlib axes.

        """
        kwargs.update({"column": "types", "kind": "geo"})
        kwargs.setdefault("legend", True)
        cmap = "RdYlGn" if not self.is_mixed and len(self.undesired) > 0 else "RdYlGn_r"
        kwargs.update(
            {
                "legend_kwds": {
                    "loc": "center left",
                    "bbox_to_anchor": (1.01, 0.5),
                },
                "cmap": cmap,
            },
        )

        return self.geodataframe.plot(**kwargs)

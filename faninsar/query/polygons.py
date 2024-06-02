from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

import geopandas as gpd
import pandas as pd
from matplotlib.axes import Axes
from pyproj.crs import CRS
from rasterio.errors import CRSError

from .bbox import BoundingBox
from .points import Points


class Polygons:
    """Polygons object is used to store the regions that need to be retrieved
    ("desired") or removed ("undesired") from a dataset.

    Tip:
        When a mixed-types Polygons, where both "desired" and "undesired"
        polygons, are provided:

        * the "undesired" polygons will only be useful when there are overlapping regions with the "desired" polygons. Otherwise, the "desired" polygons are enough.

        * you can use :meth:`to_desired` to get the desired polygons from a mixed-types Polygons object.
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame | gpd.GeoSeries,
        types: (
            Literal["desired", "undesired"] | Sequence[Literal["desired", "undesired"]]
        ) = "desired",
        crs: Any = None,
        all_touched=True,
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
        self._gdf = self._format_geometry(gdf, types, crs).sort_values(
            by="types", ascending=True
        )
        self.pad = pad
        self.all_touched = all_touched

    def __str__(self) -> str:
        return f"Polygons(count={len(self)}, crs='{self.crs}')"

    def __repr__(self) -> str:
        prefix = "Polygons:\n"
        middle = self._gdf.__repr__()
        suffix = f"\n[count={len(self)}, crs='{self.crs}']"

        return f"{prefix}{middle}{suffix}"

    def __len__(self) -> int:
        return len(self._gdf)

    def __add__(self, other: "Polygons") -> "Polygons":
        if not isinstance(other, Polygons):
            raise TypeError(f"other must be an instance of Polygon. Got {type(other)}")

        if self.crs != other.crs:
            if self.crs is None or other.crs is None:
                warnings.warn("CRS is found lacking, adding polygons without CRS.")
            else:
                other = other.to_crs(self.crs)

        gdf = pd.concat([self.frame, other.frame], ignore_index=True)
        return Polygons(gdf, types=gdf.types)

    def _format_geometry(
        self,
        gdf: gpd.GeoDataFrame | gpd.GeoSeries | "Polygons",
        types: (
            Literal["desired", "undesired"] | Sequence[Literal["desired", "undesired"]]
        ),
        crs: Any,
    ) -> gpd.GeoDataFrame:
        """Format the geometry column of the GeoDataFrame."""
        if isinstance(gdf, gpd.GeoDataFrame):
            df = gpd.GeoDataFrame(gdf.geometry)
            df["types"] = types
        elif isinstance(gdf, gpd.GeoSeries):
            df = gpd.GeoDataFrame(gdf)
            df["types"] = types
        elif isinstance(gdf, Polygons):
            df = gdf.frame
        else:
            raise TypeError(
                f"gdf must be an instance of GeoDataFrame, GeoSeries. Got {type(gdf)}"
            )
        df = self._ensure_gdf_crs(df, crs)
        return df

    def _ensure_gdf_crs(self, gdf: gpd.GeoDataFrame, crs: Any) -> CRS:
        """Ensure the CRS of the GeoDataFrame."""
        if crs is None and gdf.crs is None:
            warnings.warn(
                "CRS is not found both in input geometries and parameters. Set to None."
            )
        else:
            if crs is None:
                return gdf
            if not isinstance(crs, CRS):
                crs = CRS.from_user_input(crs)
            if gdf.crs is None:
                gdf = gdf.set_crs(crs)
            elif gdf.crs != crs:
                gdf = gdf.to_crs(crs)
        return gdf

    @property
    def all_touched(self) -> bool:
        """whether to include all pixels touched by the polygon."""
        return self._all_touched

    @all_touched.setter
    def all_touched(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"all_touched must be a bool. Got {type(value)}")
        self._all_touched = value

    @property
    def pad(self) -> bool:
        """whether to pad the features in each direction by one half of a pixel prior to cropping raster."""
        return self._pad

    @pad.setter
    def pad(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"pad must be a bool. Got {type(value)}")
        self._pad = value

    @property
    def geometry(self) -> gpd.GeoSeries:
        """the geometry column of the polygons."""
        return self._gdf.geometry

    @property
    def types(self) -> pd.Series:
        """the types of polygons."""
        return self._gdf["types"]

    @property
    def frame(self) -> gpd.GeoDataFrame:
        """GeoDataFrame format of polygons."""
        return self._gdf

    @property
    def desired(self) -> "Polygons":
        """desired part of polygons."""
        return Polygons(self._gdf[self.types == "desired"], types="desired")

    @property
    def undesired(self) -> "Polygons":
        """undesired part of polygons."""
        return Polygons(self._gdf[self.types == "undesired"], types="undesired")

    @property
    def is_mixed(self) -> bool:
        """whether the polygons contain both desired and undesired polygons."""
        return len(self.desired) > 0 and len(self.undesired) > 0

    def to_desired(self) -> "Polygons":
        """Return a desired polygons, with the regions of undesired polygons being removed.

        .. Warning::
            This method should only be used when the Polygons object contains both
            "desired" and "undesired" polygons. If the Polygons object only contains
            "undesired" polygons, the returned Polygons object will be empty.
        """
        df = gpd.overlay(self.desired.frame, self.undesired.frame, how="difference")
        return Polygons(df, types="desired")

    def to_bbox(self) -> list[BoundingBox]:
        """Return a list of BoundingBox objects representing the bounding boxes
        of the polygons.

        .. Warning::
            This method will only return the bounding boxes of the desired polygons.
            If the Polygons object only contains "undesired" polygons, the returned
            list will be empty.
        """
        df = self.to_desired().frame

    def to_GeoDataFrame(self) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame of the polygons. This method is an alias of 
        :attr:`frame` for API consistency with :class:`Points` and :class:`BoundingBox`.
        """
        return self.frame

    @property
    def crs(self) -> CRS:
        """the CRS of the polygons."""
        return self._gdf.crs

    def to_crs(self, crs: Any) -> "Polygons":
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
        crs: Any,
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
                raise CRSError(
                    "The CRS has already been set. Set allow_override=True to override."
                )

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        types: (
            Literal["desired", "undesired"] | Sequence[Literal["desired", "undesired"]]
        ) = "desired",
        crs: Any = None,
        **kwargs,
    ) -> "Polygons":
        """initialize a Polygon object from a shapefile.

        Parameters
        ----------
        filename : str | Path
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
            }
        )

        return self.frame.plot(**kwargs)

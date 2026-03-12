"""A module for handling points query."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj.crs import CRS
from rasterio.warp import transform as warp_transform

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from os import PathLike

    import numpy.typing as npt

    from faninsar.typing import CrsLike


class Points:
    """A class to represent a collection of points.

    Examples
    --------
    >>> pts = Points([[1, 2], [2, 3], [3, 4]])
    >>> pt = pts[1]

    set difference of two Points:

    >>> pts - pt
    Points:
        x    y
    0  1.0  2.0
    1  3.0  4.0
    [count=2, crs='None']

    set union of two Points:

    >>> pts + Points([1, 5])
    Points:
        x    y
    0  1.0  2.0
    1  2.0  3.0
    2  3.0  4.0
    3  1.0  5.0
    [count=4, crs='None']

    in operator:

    >>> pts[1] in pts
    True

    >>> Points([1, 5]) in pts
    False

    convert to numpy array using ``np.array``:

    >>> np.array(pts, dtype=np.int16)
    array([[1, 2],
            [2, 3],
            [3, 4]], dtype=int16)

    extract x and y coordinates:

    >>> pts.x
    array([1., 2., 3.], dtype=float32)
    >>> pts.y
    array([2., 3., 4.], dtype=float32)

    extract values:

    >>> pts.values
    array([[1., 2.],
            [2., 3.],
            [3., 4.]], dtype=float32)

    convert to GeoDataFrame:

    >>> pts.to_geodataframe()
        x	y	geometry
    0	1.0	2.0	POINT (1.00000 2.00000)
    1	2.0	3.0	POINT (2.00000 3.00000)
    2	3.0	4.0	POINT (3.00000 4.00000)

    """

    _values: np.ndarray
    _crs: CRS | str | None

    __slots__ = ["_crs", "_values"]

    def __init__(
        self,
        points: Sequence[float | Sequence[float]] | npt.ArrayLike,
        crs: CrsLike = None,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        """Initialize a Points object.

        Parameters
        ----------
        points : Sequence[float | Sequence[float]]
            The points to be sampled. The shape of the points can be (2) or (n, 2)
            where n is the number of points. The first column is the x coordinate
            and the second column is the y coordinate. if the shape is (2), the
            points will be reshaped to (1, 2).
        crs: Any, optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :meth:`pyproj.crs.CRS.from_user_input`.
        dtype : npt.DTypeLike, optional
            The data type of the points. Default is np.float32.

        Raises
        ------
        ValueError
            If the shape of the points is not (n, 2).

        """
        self._values = np.asarray(points, dtype=dtype)
        self._crs = crs
        if self._values.ndim == 1:
            self._values = self._values.reshape(1, -1)
        if self._values.ndim != 2 or self._values.shape[1] != 2:
            msg = f"points must be a 2D array with 2 columns. Got {self._values}"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of points."""
        return self._values.shape[0]

    def __iter__(self) -> Iterator:
        """Return an iterator of the points."""
        yield from self._values

    def __getitem__(self, key: int) -> Points:
        """Get the point by index."""
        return Points(self._values[key, :], crs=self.crs)

    def __contains__(self, item: Points | Sequence[float]) -> bool:
        """Check if the item is in the points."""
        item_val: np.ndarray
        if isinstance(item, Points):
            item_val = item.values
        elif isinstance(item, (SequenceABC, np.ndarray)):
            item_val = np.array(item, dtype=np.float64)
        else:
            msg = f"item must be an Points or Sequence. Got {type(item)}"
            logger.error(msg)
            raise TypeError(msg, stacklevel=2)
        if item_val.ndim > 2 or item_val.shape[1] != 2:
            msg = f"item must be a 2D array with shape (n, 2). Got {item_val.shape()}"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        return np.any(np.all(self._values == item_val, axis=1))

    def __str__(self) -> str:
        """Return the string representation of the Points."""
        return f"Points(count={len(self)}, crs='{self.crs}')"

    def __repr__(self) -> str:
        """Return the string representation of the Points."""
        prefix = "Points:\n"
        middle = self.to_dataframe().to_string(max_rows=10)
        suffix = f"\n[count={len(self)}, crs='{self.crs}']"

        return f"{prefix}{middle}{suffix}"

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Return the values of the points as a numpy array."""
        if dtype is not None:
            return self._values.astype(dtype)
        return self._values

    def __add__(self, other: Points) -> Points:
        """Return the union of two Points."""
        if not isinstance(other, Points):
            msg = f"other must be an instance of Points. Got {type(other)}"
            logger.error(msg, stacklevel=2)
            raise TypeError(msg)

        other, crs_new = self._ensure_points_crs(other)
        return Points(np.vstack([self.values, other.values]), crs=crs_new)

    def __sub__(self, other: Points) -> Points | None:
        """Return the set difference of two Points."""
        if not isinstance(other, Points):
            msg = f"other must be an instance of Points. Got {type(other)}"
            logger.error(msg, stacklevel=2)
            raise TypeError(msg)

        other, crs_new = self._ensure_points_crs(other)

        mask = ~np.all(self._values == other._values, axis=1)
        values = self._values[mask]
        if len(values) == 0:
            return None
        return Points(values, crs=crs_new)

    def _ensure_points_crs(self, other: Points) -> tuple[Points, CRS]:
        """Ensure the coordinate reference system of the points are the same."""
        if self.crs != other.crs:
            if self.crs is None or other.crs is None:
                crs_new = self.crs or other.crs
                warnings.warn(
                    "Cannot find the coordinate reference system of the points. "
                    "The crs of two points will assume to be the same. ",
                    stacklevel=2,
                )
            else:
                other = other.to_crs(self.crs)
                crs_new = self.crs
        else:
            crs_new = self.crs
        return other, crs_new

    @staticmethod
    def _find_field(
        gdf: gpd.GeoDataFrame | pd.DataFrame, field_names: list[str]
    ) -> str | None:
        """Find the field name in the GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to be searched.
        field_names : list[str]
            The field names to be searched.

        Returns
        -------
        str | None
            The field name found. If not found, return None.

        """
        for name in gdf.columns:
            if name.lower() in field_names:
                return name
        return None

    @classmethod
    def _ensure_fields(
        cls,
        gdf: gpd.GeoDataFrame | pd.DataFrame,
        x_field: str | None,
        y_field: str | None,
    ) -> tuple[str, str]:
        """Parse the field from the GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame | pd.DataFrame
            The GeoDataFrame to be parsed.
        x_field : str
            The field name of the x coordinate.
        y_field : str
            The field name of the y coordinate.

        Returns
        -------
        tuple[str, str]
            The field names of the x and y coordinates.

        Raises
        ------
        ValueError
            If the field does not specify and cannot be found automatically.

        """
        if x_field is None:
            x_field = cls._find_field(
                gdf,
                ["x", "xs", "lon", "longitude", "long", "longs", "longitudes"],
            )
            if x_field is None:
                msg = (
                    "Cannot find the field name of the x coordinate. "
                    "Please provide the field name manually."
                )
                logger.error(msg, stacklevel=2)
                raise ValueError(msg)
        if y_field is None:
            y_field = cls._find_field(
                gdf,
                ["y", "ys", "lat", "latitude", "lats", "latitudes"],
            )
            if y_field is None:
                msg = (
                    "Cannot find the field name of the y coordinate. "
                    "Please provide the field name manually."
                )
                logger.error(msg, stacklevel=2)
                raise ValueError(msg)
        return x_field, y_field

    @property
    def x(self) -> np.ndarray:
        """Return the x coordinates of the points."""
        return self._values[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return the y coordinates of the points."""
        return self._values[:, 1]

    @property
    def values(self) -> np.ndarray:
        """Return the values of the points with shape (n, 2).

        n is the number of points.
        """
        return self._values

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the points."""
        return self._values.dtype

    @property
    def crs(self) -> CRS | None:
        """Return the coordinate reference system of the points."""
        return self._crs

    def set_crs(self, crs: CrsLike) -> None:
        """Set the coordinate reference system of the points.

        .. warning::
            This method will only set the crs attribute without converting the
            points to a new coordinate reference system. If you want to convert
            the points values to a new coordinate, please use :meth:`to_crs`
        """
        self._crs = CRS.from_user_input(crs)

    def to_crs(self, crs: CrsLike) -> Points:
        """Convert the points values to a new coordinate reference system.

        Parameters
        ----------
        crs : Any
            The new coordinate reference system. Can be any object that can be
            passed to :meth:`pyproj.crs.CRS.from_user_input`.

        Returns
        -------
        Points
            The points in the new coordinate reference system.

        """
        if self.crs is None:
            msg = (
                "The current coordinate reference system is None. "
                "Please set the crs using set_crs() first."
            )
            raise ValueError(msg)
        if isinstance(crs, str):
            crs = CRS.from_user_input(crs)

        if self.crs == crs:
            return self
        values = np.array(list(zip(*warp_transform(self.crs, crs, self.x, self.y))))
        return Points(values, crs=crs)

    @classmethod
    def from_dataframe(
        cls,
        gdf: gpd.GeoDataFrame | pd.DataFrame,
        x_field: str | None = None,
        y_field: str | None = None,
        crs: CrsLike = None,
    ) -> Points:
        """Initialize a Points object from a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame | pd.DataFrame
            The GeoDataFrame to be parsed.
        x_field, y_field : str, optional, default: None
            The field name of the x/y coordinates if geometry does not exist.
            If None, will try to find the field name automatically from
            following fields (case insensitive):

            - **x**: x, xs, lon, longitude, long, longs, longitudes
            - **y**: y, ys, lat, latitude, lats, latitudes
        crs : Any, optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :meth:`pyproj.crs.CRS.from_user_input`. If
            gdf has a valid crs, it will be ignored.

        Returns
        -------
        Points
            The Points object.

        """
        if isinstance(gdf, gpd.GeoDataFrame) and hasattr(gdf, "geometry"):
            points = list(zip(gdf.geometry.x, gdf.geometry.y))
            crs = gdf.crs if gdf.crs is not None else crs
        else:
            x_field, y_field = cls._ensure_fields(gdf, x_field, y_field)
            points = gdf[[x_field, y_field]].values.tolist()

        return cls(points, crs=crs)

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        **kwargs,
    ) -> Points:
        """Initialize a Points object from a file.

        Parameters
        ----------
        filename : PathLike
            The path to the shapefile. file type can be any type that can be
            passed to :func:`geopandas.read_file`.
        **kwargs : dict
            Other parameters passed to :func:`geopandas.read_file`.

        Returns
        -------
        Points
            The Points object.

        """
        gdf = gpd.read_file(filename, **kwargs)
        geometry = gdf.geometry.explode().values

        points = list(zip(geometry.x, geometry.y))
        return cls(points, crs=gdf.crs)

    @classmethod
    def from_csv(
        cls,
        filename: PathLike,
        x_field: str | None = None,
        y_field: str | None = None,
        crs: CrsLike = None,
        **kwargs,
    ) -> Points:
        """Initialize a Points object from a csv/txt file.

        Parameters
        ----------
        filename : PathLike
            The path to the csv/txt file.
        x_field, y_field : str, optional, default: "auto"
            The field name of the x/y coordinates. If "auto", will try to
            find the field name automatically from following fields (case insensitive):

            * ``x`` : x, xs, lon, longitude
            * ``y`` : y, ys, lat, latitude

        crs : Any, optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :meth:`pyproj.crs.CRS.from_user_input`.
        **kwargs : dict
            Other parameters passed to :func:`pandas.read_csv`.

        Returns
        -------
        Points
            The Points object.

        """
        df = pd.read_csv(filename, **kwargs)
        return cls.from_dataframe(df, x_field, y_field, crs=crs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the Points to a DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with columns ``x`` and ``y``.

        """
        return pd.DataFrame(self._values, columns=["x", "y"])

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert the Points to a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame.

        """
        df = self.to_dataframe()
        geometry = (gpd.points_from_xy(df["x"], df["y"]),)
        return gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

    def to_file(self, filename: PathLike, **kwargs) -> None:
        """Save the Points to a file.

        Parameters
        ----------
        filename : PathLike
            The path to the file.
        **kwargs : dict
            Other parameters passed to :meth:`geopandas.GeoDataFrame.to_file`.

        """
        self.to_geodataframe().to_file(filename, **kwargs)

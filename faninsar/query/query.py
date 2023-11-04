from collections.abc import Iterator
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Optional, Sequence, Union, overload

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds


class BoundingBox:
    """Data class for indexing interferogram data using a spatial bounding box.

    .. Note::

        The :class:`BoundingBox` class a modified version of the ``BoundingBox``
        class from the torchgeo package. The modifications are:

        * date bounds are removed
        * the bounds is changed to (left, bottom, right, top) which is the same as
            rasterio :class:`rasterio.coords.BoundingBox`
        * added :meth:`to_rasterio_bounds` method to convert the bounding box to a
            rasterio bounds tuple
    """

    def __init__(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        crs: Optional[Union[CRS, str]] = None,
    ) -> None:
        """Initialize a BoundingBox.

        Parameters
        ----------
        left : float
            The western boundary.
        right : float
            The eastern boundary.
        bottom : float
            The southern boundary.
        top : float
            The northern boundary.
        crs : Optional[Union[CRS, str]], optional, default: None
            The coordinate reference system of the bounding box. Can be any object
            that can be passed to :func:`rasterio.crs.CRS.from_user_input`.
        """
        if left > right:
            raise ValueError(
                f"Bounding box is invalid: 'left={left}' > 'right={right}'"
            )
        if bottom > top:
            raise ValueError(
                f"Bounding box is invalid: 'bottom={bottom}' > 'top={top}'"
            )

        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.crs = crs

    def __str__(self) -> str:
        return f"BoundingBox(left={self.left}, bottom={self.bottom}, right={self.right}, top={self.top}, crs={self.crs})"

    def __repr__(self) -> str:
        return self.__str__()

    # https://github.com/PyCQA/pydocstyle/issues/525
    @overload
    def __getitem__(self, key: int) -> float:  # noqa: D105
        pass

    @overload
    def __getitem__(self, key: slice) -> list[float]:  # noqa: D105
        pass

    def __getitem__(self, key: Union[int, slice]) -> Union[float, list[float]]:
        """Index the (left, bottom, right,  top) tuple.

        Args:
            key: integer or slice object

        Returns:
            the value(s) at that index

        Raises:
            IndexError: if key is out of bounds
        """
        return [self.left, self.bottom, self.right, self.top][key]

    def __iter__(self) -> Iterator[float]:
        """Container iterator.

        Returns:
            iterator object that iterates over all objects in the container
        """
        yield from [self.left, self.bottom, self.right, self.top]

    def __contains__(self, other: "BoundingBox") -> bool:
        """Whether or not other is within the bounds of this bounding box.

        Args:
            other: another bounding box

        Returns:
            True if other is within this bounding box, else False
        """
        return (
            (self.left <= other.left <= self.right)
            and (self.left <= other.right <= self.right)
            and (self.bottom <= other.bottom <= self.top)
            and (self.bottom <= other.top <= self.top)
        )

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        """The union operator.

        Args:
            other: another bounding box

        Returns:
            the minimum bounding box that contains both self and other
        """
        return BoundingBox(
            min(self.left, other.left),
            max(self.right, other.right),
            min(self.bottom, other.bottom),
            max(self.top, other.top),
        )

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
        """The intersection operator.

        Args:
            other: another bounding box

        Returns:
            the intersection of self and other

        Raises:
            ValueError: if self and other do not intersect
        """
        try:
            return BoundingBox(
                max(self.left, other.left),
                min(self.right, other.right),
                max(self.bottom, other.bottom),
                min(self.top, other.top),
            )
        except ValueError:
            raise ValueError(f"Bounding boxes {self} and {other} do not overlap")

    @property
    def area(self) -> float:
        """Area of bounding box.

        Area is defined as spatial area.

        Returns:
            area
        """
        return (self.right - self.left) * (self.top - self.bottom)

    def to_crs(self, crs: Union[CRS, str]) -> "BoundingBox":
        """Convert the bounding box to a new coordinate reference system.

        Parameters
        ----------
        crs : Union[CRS, str]
            The new coordinate reference system. Can be any object that can be
            passed to :func:`rasterio.crs.CRS.from_user_input`.
        """
        if self.crs is None:
            raise ValueError(
                "The current coordinate reference system is None. "
                "Please set the crs using set_crs() first."
            )
        crs = CRS.from_user_input(crs)
        if self.crs == crs:
            return self
        else:
            left, bottom, right, top = transform_bounds(
                self.crs, crs, self.left, self.bottom, self.right, self.top
            )
            return BoundingBox(left, bottom, right, top, crs=crs)

    def set_crs(self, crs: Union[CRS, str]) -> None:
        """Set the coordinate reference system of the bounding box.

        .. warning::
            This method will only set the crs attribute without converting the
            bounding box to a new coordinate reference system. If you want to convert
            the bounding box values to a new coordinate, please use :meth:`to_crs`
        """
        self.crs = CRS.from_user_input(crs)

    def to_dict(self) -> dict[str, float]:
        """Convert the bounding box to a dictionary.

        Returns:
            dictionary with keys 'left', 'bottom', 'right', 'top'
        """
        return {
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "top": self.top,
        }

    def intersects(self, other: "BoundingBox") -> bool:
        """Whether or not two bounding boxes intersect.

        Args:
            other: another bounding box

        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.left <= other.right
            and self.right >= other.left
            and self.bottom <= other.top
            and self.top >= other.bottom
        )

    def split(
        self, proportion: float, horizontal: bool = True
    ) -> tuple["BoundingBox", "BoundingBox"]:
        """Split BoundingBox in two.

        Args:
            proportion: split proportion in range (0,1)
            horizontal: whether the split is horizontal or vertical

        Returns:
            A tuple with the resulting BoundingBoxes

        .. versionadded:: 0.5
        """
        if not (0.0 < proportion < 1.0):
            raise ValueError("Input proportion must be between 0 and 1.")

        if horizontal:
            w = self.right - self.left
            splitx = self.left + w * proportion
            bbox1 = BoundingBox(self.left, splitx, self.bottom, self.top)
            bbox2 = BoundingBox(splitx, self.right, self.bottom, self.top)
        else:
            h = self.top - self.bottom
            splity = self.bottom + h * proportion
            bbox1 = BoundingBox(self.left, self.right, self.bottom, splity)
            bbox2 = BoundingBox(self.left, self.right, splity, self.top)

        return bbox1, bbox2

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

    >>> pts + Points([1,5])
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

    >>> pts.to_GeoDataFrame()
        x	y	geometry
    0	1.0	2.0	POINT (1.00000 2.00000)
    1	2.0	3.0	POINT (2.00000 3.00000)
    2	3.0	4.0	POINT (3.00000 4.00000)
    """

    _values: np.ndarray
    _crs: Optional[Union[CRS, str]]

    __slots__ = ["_values", "_crs"]

    def __init__(
        self,
        points: Sequence[Union[float, Sequence[float]]],
        crs: Optional[Union[CRS, str]] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize a Points object.

        Parameters
        ----------
        points : Sequence[Union[float, Sequence[float]]]
            The points to be sampled. The shape of the points can be (2) or (n, 2)
            where n is the number of points. The first column is the x coordinate
            and the second column is the y coordinate. if the shape is (2), the
            points will be reshaped to (1, 2).
        crs: Optional[Union[CRS, str]], optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :func:`rasterio.crs.CRS.from_user_input`.
        dtype : np.dtype, optional
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
            raise ValueError(
                f"points must be a 2D array with 2 columns. Got {self._values}"
            )

    def __len__(self) -> int:
        return self._values.shape[0]

    def __iter__(self) -> Iterator:
        yield from self._values

    def __getitem__(self, key: int) -> 'Points':
        return Points(self._values[key, :], crs=self.crs)

    def __contains__(self, item: Union['Points', Sequence[float]]) -> bool:
        if isinstance(item, Points):
            item = item.values
        elif isinstance(item, SequenceABC):
            item = np.array(item, dtype=np.float64)
        else:
            raise TypeError(f"item must be an Points or Sequence. Got {type(item)}")
        if item.ndim > 2 or item.shape[1] != 2:
            raise ValueError(
                f"item must be a 2D array with shape (n, 2). Got {item.shape()}"
            )

        return np.any(np.all(self._values == item, axis=1))

    def __str__(self) -> str:
        return f"Points(count={len(self)}, crs='{self.crs}')"

    def __repr__(self) -> str:
        prefix = "Points:\n"
        middle = self.to_DataFrame().to_string(max_rows=10)
        suffix = f"\n\n[count={len(self)}, crs='{self.crs}']"
        
        return f"{prefix}{middle}{suffix}"

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype is not None:
            return self._values.astype(dtype)
        return self._values

    def __add__(self, other: "Points") -> "Points":
        if not isinstance(other, Points):
            raise TypeError(f"other must be an instance of Points. Got {type(other)}")
        return Points(np.vstack([self.values, other.values]))

    def __sub__(self, other: "Points") -> Optional["Points"]:
        if not isinstance(other, Points):
            raise TypeError(f"other must be an instance of Points. Got {type(other)}")

        mask = ~np.all(self._values == other._values, axis=1)
        values = self._values[mask]
        if len(values) == 0:
            return None
        return Points(values)

    @staticmethod
    def _find_field(gdf: gpd.GeoDataFrame, field_names: list[str]) -> Optional[str]:
        """Find the field name in the GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to be searched.
        field_names : list[str]
            The field names to be searched.

        Returns
        -------
        Optional[str]
            The field name found. If not found, return None.
        """
        for name in gdf.columns:
            if name.lower() in field_names:
                return name

    @classmethod
    def _ensure_fields(
        cls, gdf: gpd.GeoDataFrame, x_field: str, y_field: str
    ) -> Optional[np.ndarray]:
        """Parse the field from the GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to be parsed.
        field_name : str
            The field name to be parsed.

        Returns
        -------
        Optional[np.ndarray]
            The values of the field. If the field does not exist, return None.
        """
        if x_field == "auto":
            x_field = cls._find_field(
                gdf, ["x", "xs", "lon", "longitude", "long", "longs", "longitudes"]
            )
            if x_field is None:
                raise ValueError(
                    "Cannot find the field name of the x coordinate. "
                    "Please provide the field name manually."
                )
        if y_field == "auto":
            y_field = cls._find_field(
                gdf, ["y", "ys", "lat", "latitude", "lats", "latitudes"]
            )
            if y_field is None:
                raise ValueError(
                    "Cannot find the field name of the y coordinate. "
                    "Please provide the field name manually."
                )
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
        """Return the values of the points with shape (n, 2) where n is the
        number of points."""
        return self._values

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the points."""
        return self._values.dtype

    @property
    def crs(self) -> Optional[CRS]:
        """Return the coordinate reference system of the points."""
        return self._crs

    def set_crs(self, crs: Any) -> None:
        """Set the coordinate reference system of the points.

        .. warning::
            This method will only set the crs attribute without converting the
            points to a new coordinate reference system. If you want to convert
            the points values to a new coordinate, please use :meth:`to_crs`
        """
        self._crs = CRS.from_user_input(crs)

    def to_crs(self, crs: Union[CRS, str]) -> "Points":
        """Convert the points values to a new coordinate reference system.

        Parameters
        ----------
        crs : Union[CRS, str]
            The new coordinate reference system. Can be any object that can be
            passed to :func:`rasterio.crs.CRS.from_user_input`.

        Returns
        -------
        Points
            The points in the new coordinate reference system.
        """
        if self.crs is None:
            raise ValueError(
                "The current coordinate reference system is None. "
                "Please set the crs using set_crs() first."
            )
        if isinstance(crs, str):
            crs = CRS.from_user_input(crs)

        if self.crs == crs:
            return self
        else:
            values = np.array(list(zip(*warp_transform(self.crs, crs, self.x, self.y))))
            return Points(values, crs=crs)

    @classmethod
    def from_GeoDataFrame(
        cls,
        gdf: gpd.GeoDataFrame,
        x_field: str = "auto",
        y_field: str = "auto",
    ) -> "Points":
        """initialize a Points object from a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The GeoDataFrame to be parsed.
        x_field/y_field : str, optional, default: "auto"
            The field name of the x/y coordinates if ``geometry`` not exists.
            If ``auto``, will try to find the field name automatically from
            following fields (case insensitive):

            * ``x``: x, xs, lon, longitude, long, longs, longitudes
            * ``y``: y, ys, lat, latitude, lats, latitudes

        Returns
        -------
        Points
            The Points object.
        """
        if "geometry" not in gdf.columns:
            x_field, y_field = cls._ensure_fields(gdf, "auto", "auto")
            return cls(gdf[[x_field, y_field]].values, crs=gdf.crs)
        else:
            points = list(zip(gdf.geometry.values.x, gdf.geometry.values.y))
            return cls(points, crs=gdf.crs)

    @classmethod
    def from_shapefile(
        cls,
        filename: Union[str, Path],
        x_field: str = "auto",
        y_field: str = "auto",
        **kwargs,
    ) -> "Points":
        """initialize a Points object from a shapefile.

        Parameters
        ----------
        filename : Union[str, Path]
            The path to the shapefile. file type can be any type that can be
            passed to :func:`geopandas.read_file`.
        x_field/y_field : str, optional, default: "auto"
            The field name of the x/y coordinates. If ``auto``, will try to
            find the field name automatically from following fields (case insensitive):
            * ``x``: x, xs, lon, longitude
            * ``y``: y, ys, lat, latitude
        **kwargs : dict
            Other parameters passed to :func:`geopandas.read_file`.

        Returns
        -------
        Points
            The Points object.
        """
        gdf = gpd.read_file(filename, **kwargs)

        return cls.from_GeoDataFrame(gdf, x_field, y_field)

    @classmethod
    def from_csv(
        cls,
        filename: Union[str, Path],
        x_field: str = "auto",
        y_field: str = "auto",
        crs: Optional[Union[CRS, str]] = None,
        **kwargs,
    ) -> "Points":
        """initialize a Points object from a csv/txt file.

        Parameters
        ----------
        filename : Union[str, Path]
            The path to the csv/txt file.
        x_field/y_field : str, optional, default: "auto"
            The field name of the x/y coordinates. If ``auto``, will try to
            find the field name automatically from following fields (case insensitive):
            * ``x``: x, xs, lon, longitude
            * ``y``: y, ys, lat, latitude
        crs : Optional[Union[CRS, str]], optional, default: None
            The coordinate reference system of the points. Can be any object
            that can be passed to :func:`rasterio.crs.CRS.from_user_input`.
        **kwargs : dict
            Other parameters passed to :func:`pandas.read_csv`.

        Returns
        -------
        Points
            The Points object.
        """
        df = pd.read_csv(filename, **kwargs)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs
        )
        x_field, y_field = cls._ensure_fields(df, x_field, y_field)

        return cls.from_GeoDataFrame(gdf, x_field, y_field)

    def to_DataFrame(self) -> pd.DataFrame:
        """Convert the Points to a DataFrame.

        Return
        ------
        pd.DataFrame
            The DataFrame with columns ``x`` and ``y``.
        """
        df = pd.DataFrame(self._values, columns=["x", "y"])
        return df

    def to_GeoDataFrame(self) -> gpd.GeoDataFrame:
        """Convert the Points to a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            The GeoDataFrame.
        """
        df = self.to_DataFrame()
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=self.crs
        )

    def to_shapefile(self, filename: Union[str, Path], **kwargs) -> None:
        """Save the Points to a shapefile.

        Parameters
        ----------
        filename : Union[str, Path]
            The path to the shapefile.
        **kwargs : dict
            Other parameters passed to :func:`geopandas.GeoDataFrame.to_file`.
        """
        gdf = self.to_GeoDataFrame()
        gdf.to_file(filename, **kwargs)


class GeoQuery:
    """A class to represent a collection of bbox(es) or/and points queries that
    will be used to sample data from a GeoDataset.

    .. Note::
        This class is designed to sample data using multiple bounding boxes and
        points at the same time to improve the efficiency and reduce the IO
        overhead. You are recommended to use this class to sample data from a
        GeoDataset instead of using BoundingBox(es) and Points separately.
    """

    _bbox: Union[BoundingBox, list[BoundingBox]]
    _points: Optional[Points]

    __slots__ = ["_bbox", "_points"]

    def __init__(
        self,
        bbox: Optional[Union[BoundingBox, list[BoundingBox]]] = None,
        points: Optional[Points] = None,
    ) -> None:
        """Initialize a sampler.

        Parameters
        ----------
        bbox : Optional[Union[BoundingBox, list[BoundingBox]]], optional, default: None
            The :class:`BoundingBox` or a list of :class:`BoundingBox`. for querying
            the samples.If None, the samples will be queried from the points.
        points : Optional[Points], optional, default: None
            The :class:`Points` for querying the samples. If None, the samples
            will be queried from the bbox.

        Raises
        ------
        ValueError:
            If both bbox and points are None.
        TypeError:
            If bbox is not a BoundingBox or a list of BoundingBox.
        """
        if bbox is None and points is None:
            raise ValueError("bbox and points cannot be both None.")
        if bbox is not None:
            if isinstance(bbox, BoundingBox):
                bbox = [bbox]

            if not isinstance(bbox, list):
                try:
                    bbox = list(bbox)
                except TypeError:
                    raise TypeError(
                        f"bbox must be a BoundingBox or a list of BoundingBox. Got {type(bbox)}"
                    )

        self._bbox = bbox
        self._points = points

    def __str__(self) -> str:
        bbox = f"[{len(self.bbox)} BoundingBox]" if self.bbox is not None else None
        points = self.points if self.points is not None else None
        return f"GeoQuery(bbox={bbox}, points={points})"

    def __repr__(self) -> str:
        bbox = f"[{len(self.bbox)} BoundingBox]" if self.bbox is not None else None
        points = self.points if self.points is not None else None
        return f"GeoQuery(\n    bbox={bbox}\n    points={points}\n)"

    @property
    def bbox(self) -> Optional[list[BoundingBox]]:
        """Return the bounding boxes of the samples."""
        return self._bbox

    @property
    def points(self) -> Optional[Points]:
        """Return the points of the samples."""
        return self._points

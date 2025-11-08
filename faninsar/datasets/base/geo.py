"""GeoDataset base class extracted from faninsar.datasets.base."""

from __future__ import annotations

from typing import Any, Literal, overload

from ._base_common import (
    ABC,
    CRS,
    BoundingBox,
    Index,
    Points,
    Polygons,
    Profile,
    Property,
    calculate_default_transform,
    logger,
    np,
    ops,
    pyproj,
    shapely,
    warnings,
)


class GeoDataset(ABC):
    """Abstract base class for all :mod:`faninsar` datasets.

    This class is used to represent a geospatial dataset and provides methods to
    index the dataset and retrieve information about the dataset, such as CRS,
    resolution, data type, no data value, and a bounds.
    """

    # following attributes should be set by the subclass
    _crs: CRS | None = None
    _res: tuple[float, float] = (0.0, 0.0)
    _dtype: np.dtype | None = None
    _count: int = 0
    _roi: BoundingBox | None = None
    _nodata: Any = None
    _valid: np.ndarray

    def __init__(self) -> None:
        """Initialize a new GeoDataset instance."""
        self.index = Index(interleaved=True, properties=Property(dimension=2))

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"""\
{self.__class__.__name__} Dataset
    bbox: {self.bounds}
    file count: {len(self)}"""

    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return self.__repr__()

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns
        -------
            length of the dataset

        """
        return len(self.index)

    def __getstate__(
        self,
    ) -> tuple[dict[str, Any], list[tuple[Any, Any, Any]]]:
        """Define how instances are pickled.

        Returns
        -------
            the state necessary to unpickle the instance

        """
        objects = self.index.intersection(self.index.bounds, objects=True)
        tuples = [(item.id, item.bounds, item.object) for item in objects]
        return self.__dict__, tuples

    def __setstate__(
        self,
        state: tuple[
            dict[Any, Any],
            list[tuple[int, tuple[float, float, float, float, float, float], str]],
        ],
    ) -> None:
        """Define how to unpickle an instance.

        Args:
        ----
            state: the state of the instance when it was pickled

        """
        attrs, tuples = state
        self.__dict__.update(attrs)
        for item in tuples:
            self.index.insert(*item)

    @overload
    def _ensure_query_crs(self, query: BoundingBox) -> BoundingBox: ...

    @overload
    def _ensure_query_crs(self, query: Points) -> Points: ...

    @overload
    def _ensure_query_crs(self, query: Polygons) -> Polygons: ...

    def _ensure_query_crs(
        self,
        query: Points | BoundingBox | Polygons,
    ) -> Points | BoundingBox | Polygons:
        """Ensure that the query has the same CRS as the dataset."""
        if query.crs is None:
            warnings.warn(
                f"No CRS is specified for the {query}, assuming they are in the"
                f" same CRS as the dataset ({self.crs}).",
                stacklevel=2,
            )
        elif query.crs != self.crs:
            query = query.to_crs(self.crs)
        return query

    @property
    def crs(self) -> CRS | None:
        """Coordinate reference system (:term:`CRS`) of the dataset.

        Returns
        -------
            The coordinate reference system (:term:`CRS`).

        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS | str) -> None:
        """Change the coordinate reference system :term:`(CRS)` of a GeoDataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.

        Parameters
        ----------
        new_crs: CRS or str
            New coordinate reference system :term:`(CRS)`. It can be a CRS object
            or a string, which will be parsed to a CRS object. The string can be
            in any format supported by :meth:`pyproj.crs.CRS.from_user_input`.

        """
        if not isinstance(new_crs, CRS):
            new_crs = CRS.from_user_input(new_crs)
        if new_crs == self.crs:
            return

        if self.crs is not None and len(self) > 0:
            # update the resolution
            profile = self.get_profile("bounds")
            tf, *_ = calculate_default_transform(
                self.crs,
                new_crs,
                profile["width"],
                profile["height"],
                self.bounds[0],
                self.bounds[1],
                self.bounds[2],
                self.bounds[3],
            )
            new_res = (abs(float(tf.a)), abs(float(tf.e)))
            if new_res[0] != self.res[0] or new_res[1] != self.res[1]:
                msg = (
                    "the resolution of the dataset has been changed "
                    f"from {self.res} to {new_res}."
                )
                logger.warning(msg)
                self.res = new_res

            # reproject the index
            new_index = Index(interleaved=True, properties=Property(dimension=2))
            project = pyproj.Transformer.from_crs(
                pyproj.CRS(str(self.crs)),
                pyproj.CRS(str(new_crs)),
                always_xy=True,
            ).transform
            for hit in self.index.intersection(self.index.bounds, objects=True):
                old_xmin, old_xmax, old_ymin, old_ymax = hit.bounds
                old_box = shapely.geometry.box(old_xmin, old_ymin, old_xmax, old_ymax)
                new_box = ops.transform(project, old_box)
                new_bounds = tuple(new_box.bounds)
                new_index.insert(hit.id, new_bounds, hit.object)

            self.index = new_index

        self._crs = new_crs

    @property
    def same_crs(self) -> bool:
        """Whether all files in the dataset have the same CRS with the desired CRS."""
        return self._same_crs

    @property
    def res(self) -> tuple[float, float]:
        """Return the resolution of the dataset.

        Returns
        -------
        res: tuple of floats
            resolution of the dataset in x and y directions.

        """
        return self._res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Set the resolution of the dataset.

        Parameters
        ----------
        new_res : float or tuple of floats (x_res, y_res)
            resolution of the dataset . If a float is given, the same resolution
            will be used in both x and y directions.

        """
        if isinstance(new_res, (int, float, np.integer, np.floating)):
            new_res = (float(new_res), float(new_res))
        if len(new_res) != 2:
            msg = f"Resolution must be a float or a tuple of length 2, got {new_res}"
            raise ValueError(msg)
        if not all(isinstance(i, float) for i in new_res):
            try:
                new_res = (float(new_res[0]), float(new_res[1]))
            except TypeError as e:
                msg = "Resolution must be a float or a tuple of floats"
                raise TypeError(msg) from e
        self._res = new_res

    @property
    def roi(self) -> BoundingBox:
        """Return the region of interest of the dataset.

        Returns
        -------
        roi: BoundingBox object
            region of interest of the dataset. If None, the bounds of
            entire dataset will be used.

        """
        if self._roi:
            return self._roi
        return self.bounds

    @roi.setter
    def roi(self, new_roi: BoundingBox) -> None:
        """Set the region of interest of the dataset.

        Parameters
        ----------
        new_roi : BoundingBox object, optional
            region of interest of the dataset in the CRS of the dataset. If the
            crs of the new_roi is different from the crs of the dataset, the new_roi
            will be reprojected to the crs of the dataset. If None, the crs of the
            dataset will be used.

        """
        new_roi = self._check_roi(new_roi)

        self._roi = new_roi

    def _check_roi(self, roi: BoundingBox | None) -> BoundingBox:
        """Check the roi and return a valid roi.

        Parameters
        ----------
        roi : BoundingBox object, optional
            region of interest of the dataset in the CRS of the dataset. If the
            crs of the new_roi is different from the crs of the dataset, the new_roi
            will be reprojected to the crs of the dataset. If None, the crs of the
            dataset will be used.

        Returns
        -------
        roi: BoundingBox object
            region of interest of the dataset. If None, the bounds of
            entire dataset will be used.

        """
        if roi is None:
            return self.roi
        if not isinstance(roi, BoundingBox):
            msg = f"roi must be a BoundingBox object, got {type(roi)} instead."
            raise TypeError(msg)
        if roi.crs != self.crs:
            if roi.crs is None:
                roi = BoundingBox(*roi, crs=self.crs)
            else:
                roi = roi.to_crs(self.crs)
        return roi

    @property
    def dtype(self) -> np.dtype | None:
        """Data type of the dataset.

        Returns
        -------
        dtype: numpy.dtype object or None
            data type of the dataset

        """
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: np.dtype) -> None:
        """Set the data type of the dataset.

        Parameters
        ----------
        new_dtype : numpy.dtype
            data type of the dataset

        """
        self._dtype = new_dtype

    @property
    def nodata(self) -> float | None:
        """No data value of the dataset.

        Returns
        -------
        nodata: float or int
            no data value of the dataset

        """
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata: float) -> None:
        """Set the no data value of the dataset.

        Parameters
        ----------
        new_nodata : float or int
            no data value of the dataset

        """
        self._nodata = new_nodata

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the dataset.

        Returns
        -------
        shape: tuple of ints
            shape of the dataset in (height, width) format

        """
        profile = self.get_profile("bounds")
        return profile["height"], profile["width"]

    @property
    def valid(self) -> np.ndarray:
        """Return a boolean array indicating which files are valid.

        Returns
        -------
        valid: numpy.ndarray
            boolean array indicating which files are valid. True means the file
            is valid and can be read by rasterio, False means the file is invalid.

        """
        return self._valid

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the overall dataset.

        It is the union of all the files in the dataset.

        Returns
        -------
        bounds: BoundingBox object
            (minx, right, bottom, top) of the dataset

        """
        return BoundingBox(*self.index.bounds, crs=self.crs)

    def _ensure_bbox(
        self,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> BoundingBox:
        """Return the bounds of the dataset for the given bounding box type.

        Parameters
        ----------
        bbox : BoundingBox | Literal["roi", "bounds"], optional
            the bounding box used to calculate the bounds of the dataset.
            Default is 'roi'.

        Returns
        -------
        bounds: BoundingBox
            bounds of the dataset for the given bounding box type.

        Raises
        ------
        TypeError: if bbox is not one of {'bounds', 'roi'} or a BoundingBox

        """
        if bbox == "bounds":
            return self.bounds
        if bbox == "roi":
            return self.roi
        if isinstance(bbox, BoundingBox):
            return self._check_roi(bbox)
        msg = f"bbox must be one of ['bounds', 'roi'] or a BoundingBox, but got {bbox}"
        raise TypeError(msg)

    def get_profile(
        self, bbox: BoundingBox | Literal["roi", "bounds"] = "roi"
    ) -> Profile | None:
        """Get profile information of the dataset for the given bounding box type.

        The profile information includes the width, height, transform,
        count, data type, no data value, and CRS of the dataset.

        Parameters
        ----------
        bbox : BoundingBox | Literal["roi", "bounds"], optional
            the bounding box used to calculate the ``width``, ``height``
            and ``transform`` of the dataset for the profile. Default is
            'roi'.

        Returns
        -------
        profile: Profile object or None
            profile of the dataset for the given bounding box type.

        """
        msg = "get_profile method must be implemented in subclass"
        raise NotImplementedError(msg)

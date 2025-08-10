"""Result classes for the queries."""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    import numpy as np
    from rasterio.transform import Affine

    from .query import GeoQuery


class PointsResult(xr.DataArray):
    """A class to manage the result of :class:`~faninsar.query.Points` query.

    Inherits from xarray DataArray to represent point query results with dimensions
    for files, points, and optionally bands.
    """

    __slots__ = ()

    def __init__(
        self,
        data: np.ndarray | dict | None = None,
        coords: dict | None = None,
        dims: tuple | None = None,
        name: str | None = None,
        attrs: dict | None = None,
        indexes: dict | None = None,
        fastpath: bool = False,
        **kwargs,
    ) -> None:
        """Initialize PointsResult with point query data or xarray parameters.

        Parameters
        ----------
        data : np.ndarray or dict or None, optional
            Point query result data. If dict, should contain 'data' and 'dims' keys.
        coords : dict or None, optional
            Point query result coordinates.
        dims : tuple or None, optional
            Point query result dimensions.
        name : str or None, optional
            Point query result name.
        attrs : dict or None, optional
            Point query result attributes.
        indexes : dict or None, optional
            Point query result indexes.
        fastpath : bool, optional
            Point query result fastpath.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the xarray DataArray constructor.

        """
        # Handle legacy dict format - check if first argument is a dict
        if data is not None and isinstance(data, dict):
            # Convert dict to DataArray using simplified conversion
            data_array = self._convert_dict_to_dataarray(data)
            super().__init__(
                data=data_array.data,
                coords=data_array.coords,
                dims=data_array.dims,
                name=data_array.name,
                attrs=data_array.attrs,
            )
        else:
            # Handle xarray-style initialization
            super().__init__(
                data=data,
                coords=coords,
                dims=dims,
                name=name,
                attrs=attrs,
                indexes=indexes,
                fastpath=fastpath,
                **kwargs,
            )

    def _convert_dict_to_dataarray(self, result: dict) -> xr.DataArray:
        """Convert legacy dict format to xarray DataArray (simplified version)."""
        import numpy as np

        data = result["data"]
        dims_info = result.get("dims", "")

        # Parse dims string to extract dimension names and sizes
        dims = []
        coords = {}

        if dims_info and isinstance(data, np.ndarray):
            if isinstance(dims_info, list):
                # Handle list format like [("files", 3), ("points", 2)]
                if len(dims_info) == data.ndim:
                    for dim_name, dim_size in dims_info:
                        dims.append(dim_name)
                        coords[dim_name] = range(int(dim_size))
                else:
                    # Fallback if list length doesn't match
                    dims_info = None
            elif isinstance(dims_info, str):
                import re

                # Try to parse dims string like "(files:3, points:2)" or "points:3"
                dim_pattern = r"(\w+):(\d+)"
                matches = re.findall(dim_pattern, dims_info)

                if matches and len(matches) == data.ndim:
                    for dim_name, dim_size in matches:
                        dims.append(dim_name)
                        coords[dim_name] = range(int(dim_size))
                else:
                    # Fallback if parsing fails
                    dims_info = None

        if not dims and isinstance(data, np.ndarray):
            # Fallback to default naming based on data shape
            if data.ndim == 1:
                dims = ["points"]
                coords = {"points": range(data.shape[0])}
            elif data.ndim == 2:
                dims = ["files", "points"]
                coords = {"files": range(data.shape[0]), "points": range(data.shape[1])}
            elif data.ndim == 3:
                dims = ["files", "bands", "points"]
                coords = {
                    "files": range(data.shape[0]),
                    "bands": range(data.shape[1]),
                    "points": range(data.shape[2]),
                }
            else:
                dims = [f"dim_{i}" for i in range(data.ndim)]
                coords = {f"dim_{i}": range(data.shape[i]) for i in range(data.ndim)}

        if isinstance(data, np.ndarray):
            return xr.DataArray(data, dims=dims, coords=coords, name="point_values")
        # Fallback for non-array data
        return xr.DataArray(data, name="point_values")

    def __repr__(self) -> str:
        """Return the string representation as base xarray DataArray."""
        # Get the parent repr and replace the class name
        parent_repr = super().__repr__()
        # Replace any occurrence of the custom class name with DataArray
        return parent_repr.replace("PointsResult", "DataArray")

    def __str__(self) -> str:
        """Return the string representation as base xarray DataArray."""
        # Get the parent str and replace the class name
        parent_str = super().__str__()
        # Replace any occurrence of the custom class name with DataArray
        return parent_str.replace("PointsResult", "DataArray")

    def _repr_html_(self) -> str:
        """Return the HTML representation as base xarray DataArray for Jupyter."""
        # Get the parent HTML repr and replace the class name
        parent_html = super()._repr_html_()
        # Replace any occurrence of the custom class name with DataArray
        return parent_html.replace("PointsResult", "DataArray")


class BBoxesResult(xr.Dataset):
    """A class to manage the result of :class:`~faninsar.query.BoundingBox` query.

    Inherits from xarray Dataset to represent bounding box query results with data
    and transform information.

    Parameters
    ----------
    data_vars : dict or None, optional
        Bounding box query result data variables. If dict, should contain 'data' and
        'dims' keys.
    coords : dict or None, optional
        Bounding box query result coordinates.
    attrs : dict or None, optional
        Bounding box query result attributes.

    """

    __slots__ = ()

    def __init__(self, data_vars=None, coords=None, attrs=None, **kwargs) -> None:  # noqa: ANN001
        """Initialize BBoxesResult with bounding box query data or xarray parameters."""
        # Handle legacy dict result format - check if first argument is a dict
        if data_vars is not None and isinstance(data_vars, dict):
            # Convert dict to Dataset using simplified conversion
            dataset = self._convert_dict_to_dataset(data_vars)
            super().__init__(
                data_vars=dataset.data_vars,
                coords=dataset.coords,
                attrs=dataset.attrs,
            )
        else:
            # Handle xarray-style initialization
            super().__init__(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)

    def _convert_dict_to_dataset(self, result: dict) -> xr.Dataset:
        """Convert legacy dict format to xarray Dataset (simplified version)."""
        import numpy as np

        data = result["data"]
        dims_info = result.get("dims", "")
        transforms = result.get("transforms", [])

        # Parse dims info to extract dimension names and sizes
        dims = []
        coords = {}

        if dims_info and isinstance(data, np.ndarray):
            if isinstance(dims_info, list):
                # Handle list format like [("files", 3), ("points", 2)]
                if len(dims_info) == data.ndim:
                    for dim_name, dim_size in dims_info:
                        # Map common dimension names
                        mapped_dim_name = dim_name
                        if dim_name in {"height", "width"}:
                            mapped_dim_name = "y" if dim_name == "height" else "x"
                        dims.append(mapped_dim_name)
                        coords[mapped_dim_name] = range(int(dim_size))
                else:
                    # Fallback if list length doesn't match
                    dims_info = None
            elif isinstance(dims_info, str):
                import re

                # Try to parse dims string like "bbox:3, files:2, height:10, width:10"
                dim_pattern = r"(\w+):(\d+)"
                matches = re.findall(dim_pattern, dims_info)

                if matches and len(matches) == data.ndim:
                    for dim_name, dim_size in matches:
                        # Map common dimension names
                        mapped_dim_name = dim_name
                        if dim_name in {"height", "width"}:
                            mapped_dim_name = "y" if dim_name == "height" else "x"
                        dims.append(mapped_dim_name)
                        coords[mapped_dim_name] = range(int(dim_size))
                else:
                    # Fallback if parsing fails
                    dims_info = None

        if not dims and isinstance(data, np.ndarray):
            # Fallback to default naming based on data shape
            if data.ndim == 3:
                dims = ["files", "y", "x"]
                coords = {
                    "files": range(data.shape[0]),
                    "y": range(data.shape[1]),
                    "x": range(data.shape[2]),
                }
            elif data.ndim == 4:
                dims = ["files", "bands", "y", "x"]
                coords = {
                    "files": range(data.shape[0]),
                    "bands": range(data.shape[1]),
                    "y": range(data.shape[2]),
                    "x": range(data.shape[3]),
                }
            else:
                dims = [f"dim_{i}" for i in range(data.ndim)]
                coords = {f"dim_{i}": range(data.shape[i]) for i in range(data.ndim)}

        if isinstance(data, np.ndarray):
            data_vars = {"values": (dims, data)}
            attrs = {}

            # Add transforms to attributes
            if transforms:
                attrs["transforms"] = [
                    t.to_gdal() if hasattr(t, "to_gdal") else t for t in transforms
                ]

            return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        # Fallback for non-array data
        return xr.Dataset({"values": data})

    @property
    def transforms(self) -> list[Affine] | None:
        """Access to transform information."""
        if "transforms" in self.attrs:
            from rasterio.transform import Affine

            return list(starmap(Affine.from_gdal, self.attrs["transforms"]))
        if "transform" in self.attrs:
            from rasterio.transform import Affine

            return [Affine.from_gdal(*self.attrs["transform"])]
        return None

    def __repr__(self) -> str:
        """Return the string representation as base xarray Dataset."""
        # Get the parent repr and replace the class name
        parent_repr = super().__repr__()
        # Replace any occurrence of the custom class name with Dataset
        return parent_repr.replace("BBoxesResult", "Dataset")

    def __str__(self) -> str:
        """Return the string representation as base xarray Dataset."""
        # Get the parent str and replace the class name
        parent_str = super().__str__()
        # Replace any occurrence of the custom class name with Dataset
        return parent_str.replace("BBoxesResult", "Dataset")

    def _repr_html_(self) -> str:
        """Return the HTML representation as base xarray Dataset for Jupyter."""
        # Get the parent HTML repr and replace the class name
        parent_html = super()._repr_html_()
        # Replace any occurrence of the custom class name with Dataset
        return parent_html.replace("BBoxesResult", "Dataset")


class PolygonsResult(xr.Dataset):
    """A class to manage the result of :class:`~faninsar.query.Polygons` query.

    Inherits from xarray Dataset to represent polygon query results with data,
    transform, and mask information.

    Parameters
    ----------
    data_vars : dict or None, optional
        Polygon query result data variables. If dict, should contain 'data', 'dims',
        'transforms', and 'masks' keys.
    coords : dict or None, optional
        Polygon query result coordinates.
    attrs : dict or None, optional
        Polygon query result attributes.

    """

    __slots__ = ()

    def __init__(self, data_vars=None, coords=None, attrs=None, **kwargs) -> None:  # noqa: ANN001
        """Initialize PolygonsResult with polygon query data or xarray parameters."""
        # Handle legacy dict result format - check if first argument is a dict
        if data_vars is not None and isinstance(data_vars, dict):
            # Convert dict to Dataset using simplified conversion
            dataset = self._convert_dict_to_dataset(data_vars)
            super().__init__(
                data_vars=dataset.data_vars,
                coords=dataset.coords,
                attrs=dataset.attrs,
            )
        else:
            # Handle xarray-style initialization
            super().__init__(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)

    def _convert_dict_to_dataset(self, result: dict) -> xr.Dataset:
        """Convert legacy dict format to xarray Dataset (simplified version)."""
        import numpy as np

        data = result["data"]
        transforms = result.get("transforms", [])
        masks = result.get("masks", [])

        # Handle list of arrays (one per polygon)
        if isinstance(data, list) and len(data) > 0:
            first_array = data[0]
            if isinstance(first_array, (np.ndarray, np.ma.MaskedArray)):
                # Check if all arrays have the same shape
                all_same_shape = all(
                    isinstance(arr, (np.ndarray, np.ma.MaskedArray))
                    and arr.shape == first_array.shape
                    for arr in data
                )

                if all_same_shape:
                    # All arrays have same shape, can stack normally
                    if first_array.ndim == 3:
                        # Each polygon has shape (files, height, width)
                        n_polygons = len(data)
                        stacked_data = np.stack(
                            data, axis=0
                        )  # (polygons, files, height, width)
                        dims = ["polygon", "files", "y", "x"]
                        coords = {
                            "polygon": range(n_polygons),
                            "files": range(first_array.shape[0]),
                            "y": range(first_array.shape[1]),
                            "x": range(first_array.shape[2]),
                        }
                    elif first_array.ndim == 4:
                        # Each polygon has shape (files, bands, height, width)
                        n_polygons = len(data)
                        stacked_data = np.stack(
                            data, axis=0
                        )  # (polygons, files, bands, height, width)
                        dims = ["polygon", "files", "bands", "y", "x"]
                        coords = {
                            "polygon": range(n_polygons),
                            "files": range(first_array.shape[0]),
                            "bands": range(first_array.shape[1]),
                            "y": range(first_array.shape[2]),
                            "x": range(first_array.shape[3]),
                        }
                    else:
                        # Fallback
                        n_polygons = len(data)
                        stacked_data = np.stack(data, axis=0)
                        dims = ["polygon"] + [
                            f"dim_{i}" for i in range(first_array.ndim)
                        ]
                        coords = {"polygon": range(n_polygons)}
                        for i in range(first_array.ndim):
                            coords[f"dim_{i}"] = range(first_array.shape[i])
                else:
                    # Arrays have different shapes, store as object array
                    n_polygons = len(data)
                    # Create object array to hold variable-shaped arrays
                    stacked_data = np.empty(n_polygons, dtype=object)
                    for i, arr in enumerate(data):
                        stacked_data[i] = arr
                    dims = ["polygon"]
                    coords = {"polygon": range(n_polygons)}

                data_vars = {"values": (dims, stacked_data)}
                attrs = {}

                # Add transforms to attributes
                if transforms:
                    attrs["transforms"] = [
                        t.to_gdal() if hasattr(t, "to_gdal") else t for t in transforms
                    ]

                # Add masks if available
                if masks and isinstance(masks[0], np.ndarray):
                    stacked_masks = np.stack(masks, axis=0)
                    # Skip file dimension for masks
                    mask_dims = ["polygon"] + dims[2:]
                    data_vars["masks"] = (mask_dims, stacked_masks)

                return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Fallback for non-list data or when list processing fails
        if isinstance(data, list):
            # For list data that couldn't be processed above, store as object array
            n_polygons = len(data)
            stacked_data = np.empty(n_polygons, dtype=object)
            for i, arr in enumerate(data):
                stacked_data[i] = arr
            dims = ["polygon"]
            coords = {"polygon": range(n_polygons)}
            data_vars = {"values": (dims, stacked_data)}
            return xr.Dataset(data_vars=data_vars, coords=coords)
        return xr.Dataset({"values": data})

    @property
    def transforms(self) -> list[Affine] | None:
        """Access to transform information."""
        if "transforms" in self.attrs:
            from itertools import starmap

            from rasterio.transform import Affine

            return list(starmap(Affine.from_gdal, self.attrs["transforms"]))
        return None

    @property
    def masks(self) -> list[np.ndarray] | None:
        """Access to mask information."""
        if "masks" in self.data_vars:
            masks_data = self["masks"]
            return [
                masks_data.isel(polygon=i).values
                for i in range(masks_data.sizes["polygon"])
            ]
        return None

    def __repr__(self) -> str:
        """Return the string representation as base xarray Dataset."""
        # Get the parent repr and replace the class name
        parent_repr = super().__repr__()
        # Replace any occurrence of the custom class name with Dataset
        return parent_repr.replace("PolygonsResult", "Dataset")

    def __str__(self) -> str:
        """Return the string representation as base xarray Dataset."""
        # Get the parent str and replace the class name
        parent_str = super().__str__()
        # Replace any occurrence of the custom class name with Dataset
        return parent_str.replace("PolygonsResult", "Dataset")

    def _repr_html_(self) -> str:
        """Return the HTML representation as base xarray Dataset for Jupyter."""
        # Get the parent HTML repr and replace the class name
        parent_html = super()._repr_html_()
        # Replace any occurrence of the custom class name with Dataset
        return parent_html.replace("PolygonsResult", "Dataset")


class QueryResult(xr.DataTree):
    """A combined result of Queries inheriting from xarray DataTree.

    Combines :class:`PointsResult`, :class:`BBoxesResult`, and
    :class:`PolygonsResult` queries into a hierarchical xarray DataTree structure.
    This class is the default return type of the :ref:`query` results for the datasets.
    """

    __slots__ = ("_boxes", "_points", "_polygons", "_query")

    def __init__(
        self,
        points: PointsResult | dict | None = None,
        boxes: BBoxesResult | dict | None = None,
        polygons: PolygonsResult | dict | None = None,
        query: GeoQuery | None = None,
    ) -> None:
        """Initialize the QueryResult instance.

        Parameters
        ----------
        points : PointsResult | dict, optional
            Result of the :class:`~faninsar.query.Points` query.
        boxes : BBoxesResult | dict, optional
            Result of the :class:`~faninsar.query.BoundingBox` query.
        polygons : PolygonsResult | dict, optional
            Result of the :class:`~faninsar.query.Polygons` query.
        query : GeoQuery, optional
            The :class:`~faninsar.query.GeoQuery` instance used to generate results.

        """
        if isinstance(points, dict):
            points = PointsResult(data=points)
        if isinstance(boxes, dict):
            boxes = BBoxesResult(data_vars=boxes)
        if isinstance(polygons, dict):
            polygons = PolygonsResult(data_vars=polygons)

        self._points = points
        self._boxes = boxes
        self._polygons = polygons
        self._query = query

        # Create DataTree structure and initialize parent class
        datatree = self._create_datatree()
        if datatree is not None:
            super().__init__(
                dataset=datatree.dataset,
                children=datatree.children,
                name=datatree.name,
            )
        else:
            # Fallback if DataTree creation fails
            super().__init__(name="query_result")

    def _create_datatree(self) -> xr.DataTree | None:
        """Create a DataTree structure from the query results.

        Returns
        -------
        xr.DataTree | None
            DataTree with query results organized hierarchically.

        """
        try:
            from xarray import DataTree
        except ImportError:
            # DataTree not available, return None
            return None

        # Create root dataset with query metadata
        root_attrs = {}
        if self._query is not None:
            root_attrs["query_type"] = str(type(self._query).__name__)

        root_ds = xr.Dataset(attrs=root_attrs)

        # Create child nodes for each result type
        children = {}
        if self._points is not None:
            # Convert DataArray to Dataset for DataTree compatibility
            points_data = self._points
            if isinstance(points_data, xr.DataArray):
                points_data = points_data.to_dataset(name="values")
            children["points"] = DataTree(dataset=points_data, name="points")

        if self._boxes is not None:
            # BBoxesResult and PolygonsResult are already Dataset objects
            children["boxes"] = DataTree(dataset=self._boxes, name="boxes")

        if self._polygons is not None:
            children["polygons"] = DataTree(dataset=self._polygons, name="polygons")

        return DataTree(dataset=root_ds, name="query_result", children=children)

    def __repr__(self) -> str:
        """Return the string representation as base xarray DataTree."""
        # Get the parent repr and replace the class name
        parent_repr = super().__repr__()
        # Replace any occurrence of the custom class name with DataTree
        return parent_repr.replace("QueryResult", "DataTree")

    def __str__(self) -> str:
        """Return the string representation as base xarray DataTree."""
        # Get the parent str and replace the class name
        parent_str = super().__str__()
        # Replace any occurrence of the custom class name with DataTree
        return parent_str.replace("QueryResult", "DataTree")

    def _repr_html_(self) -> str:
        """Return the HTML representation as base xarray DataTree for Jupyter."""
        # Get the parent HTML repr and replace the class name
        parent_html = super()._repr_html_()
        # Replace any occurrence of the custom class name with DataTree
        return parent_html.replace("QueryResult", "DataTree")

    @property
    def points(self) -> PointsResult | None:
        """Result of the :class:`~faninsar.query.Points` query."""
        return self._points

    @property
    def boxes(self) -> BBoxesResult | None:
        """Result of the :class:`~faninsar.query.BoundingBox` query."""
        return self._boxes

    @property
    def polygons(self) -> PolygonsResult | None:
        """Result of the :class:`~faninsar.query.Polygons` query."""
        return self._polygons

    @property
    def query(self) -> GeoQuery | None:
        """The :class:`~faninsar.query.GeoQuery` instance used to generate results."""
        return self._query

"""xarray accessors to reconstruct query objects from dataset/datatree attrs.

- Dataset.fis.query returns Points | BoundingBox | Polygons
- DataTree.fis.query returns GeoQuery

Attributes used: attrs['query_json']
"""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import xarray as xr
from shapely import wkt as shapely_wkt

from faninsar.query import BoundingBox, GeoQuery, Points, Polygons


def _loads(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return None
    return obj


def _to_points(d: dict | None) -> Points | None:
    if not d:
        return None
    if d.get("type") != "Points":
        return None
    return Points(d.get("coords", []), crs=d.get("crs"))


def _to_bbox(d: dict | None) -> BoundingBox | None:
    if not d:
        return None
    if d.get("type") != "BoundingBox":
        return None
    return BoundingBox(
        d.get("left"), d.get("bottom"), d.get("right"), d.get("top"), crs=d.get("crs")
    )


def _to_polygons(d: dict | None) -> Polygons | None:
    if not d:
        return None
    t = d.get("type")
    if t not in {"Polygons", "Polygon"}:
        return None
    if shapely_wkt is None or gpd is None:
        return None
    wkts = d.get("wkt")
    if wkts is None:
        return None
    if isinstance(wkts, str):
        wkts = [wkts]
    geoms = [shapely_wkt.loads(w) for w in wkts]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=d.get("crs"))
    return Polygons(gdf, types="desired")


def _dataset_to_query(ds: xr.Dataset) -> Points | BoundingBox | Polygons | None:
    meta = _loads(ds.attrs.get("query_json"))
    if not isinstance(meta, dict):
        return None
    return _to_points(meta) or _to_bbox(meta) or _to_polygons(meta)


def _datatree_to_geoquery(dt: Any) -> GeoQuery | None:
    # dt is a DataTree; use its root dataset attrs
    root_ds = getattr(dt, "dataset", None)
    if root_ds is None:
        return None
    meta = _loads(root_ds.attrs.get("query_json"))
    if not isinstance(meta, dict):
        return None

    pts = (
        _to_points(meta.get("points")) if isinstance(meta.get("points"), dict) else None
    )

    bboxes = None
    if isinstance(meta.get("bboxes"), list):
        bb_objs = []
        for bd in meta["bboxes"]:
            bb = _to_bbox(bd)
            if bb is not None:
                bb_objs.append(bb)
        if len(bb_objs) == 1:
            bboxes = bb_objs  # keep list (GeoQuery accepts list or single)
        elif len(bb_objs) > 0:
            bboxes = bb_objs

    polys = (
        _to_polygons(meta.get("polygons"))
        if isinstance(meta.get("polygons"), dict)
        else None
    )

    if pts is None and bboxes is None and polys is None:
        return None
    return GeoQuery(points=pts, boxes=bboxes, polygons=polys)


@xr.register_dataset_accessor("fis")
class FanInSARDatasetAccessor:
    """Accessor for Dataset objects from faninsar."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialize the accessor."""
        self._obj = xarray_obj

    @property
    def query(self) -> Points | BoundingBox | Polygons | None:
        """The query used to generate the Dataset."""
        return _dataset_to_query(self._obj)


# Try to register a DataTree accessor; fall back to monkeypatching if not available
try:  # xarray>=2024.06 may expose DataTree
    from xarray import DataTree  # type: ignore[attr-defined]

    _HAS_DT = True
except Exception:
    DataTree = None
    _HAS_DT = False

if _HAS_DT and DataTree is not None:  # pragma: no cover - registration side-effect
    try:
        from xarray import register_datatree_accessor  # type: ignore[attr-defined]

        @register_datatree_accessor("fis")
        class FanInSARDataTreeAccessor:
            """Accessor for DataTree objects from faninsar."""

            def __init__(self, tree: DataTree) -> None:  # type: ignore[name-defined]
                """Initialize the accessor."""
                self._obj = tree

            @property
            def query(self) -> GeoQuery | None:
                """The query used to generate the DataTree."""
                return _datatree_to_geoquery(self._obj)

    except Exception:
        # Fallback: add a .fis property returning an object with .query
        class _DTAccessor:
            def __init__(self, tree: Any) -> None:
                self._obj = tree

            @property
            def query(self) -> GeoQuery | None:
                return _datatree_to_geoquery(self._obj)

        def _get_fis(self: Any) -> _DTAccessor:  # type: ignore[override]
            return _DTAccessor(self)

        DataTree.fis = property(_get_fis)

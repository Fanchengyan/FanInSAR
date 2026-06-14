"""STAC catalog generation from FanInSAR frame products.

This module converts the structured JSON metadata produced by
:class:`FrameGeometry` and :class:`FrameInterferogramCollection` into
`pystac <https://pystac.readthedocs.io/>`_ Catalog / Collection / Item objects.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    import pystac

    from .geometry import FrameGeometry
    from .interferogram import FrameInterferogramCollection

logger = setup_logger(__name__)


def _crs_to_epsg(crs_str: str) -> int | None:
    """Try to extract an EPSG code from a CRS string."""
    try:
        from pyproj import CRS

        return CRS.from_user_input(crs_str).to_epsg()
    except Exception:
        return None


def _bounds_to_wgs84(
    bounds: list[float], src_crs: str
) -> tuple[list[float], list[float]]:
    """Reproject ``[left, bottom, right, top]`` to EPSG:4326.

    Returns
    -------
    bbox : list[float]
        ``[west, south, east, north]`` in WGS 84.
    geom_coords : list[float]
        Same four values (the bbox is already axis-aligned).

    """
    from pyproj import CRS, Transformer

    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input("EPSG:4326")

    if src == dst:
        return bounds, bounds

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    west, south = transformer.transform(bounds[0], bounds[1])
    east, north = transformer.transform(bounds[2], bounds[3])

    wgs84 = [west, south, east, north]
    return wgs84, wgs84


def _parse_date(date_str: str) -> datetime:
    """Parse a ``YYYYMMDD`` string into a timezone-aware datetime."""
    return datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC)


def _build_geometry_item(geom: FrameGeometry) -> pystac.Item:
    """Build a STAC Item for the geometry assets."""
    from pystac import Asset, Item, MediaType
    from pystac.extensions.projection import ProjectionExtension

    meta = geom.metadata
    if meta is None:
        msg = "geometry.json not found; cannot build STAC item."
        raise FileNotFoundError(msg)

    bounds: list[float] = meta["bounds"]
    bbox, _ = _bounds_to_wgs84(bounds, meta["crs"])
    epsg = _crs_to_epsg(meta["crs"])

    from shapely.geometry import box as shapely_box

    geom_shape = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])

    item = Item(
        id="geometry",
        geometry=geom_shape.__geo_interface__,
        bbox=bbox,
        datetime=datetime.now(UTC),
        properties={
            "frame:type": "FrameGeometry",
            "frame:angle_unit": meta.get("angle_unit", "degree"),
            "frame:azimuth_convention": meta.get("azimuth_convention", ""),
            "frame:heading_convention": meta.get("heading_convention", ""),
        },
    )

    # Projection extension
    if epsg is not None:
        proj = ProjectionExtension.ext(item, add_if_missing=True)
        proj.epsg = epsg
        proj.transform = list(meta["transform"])[:6]
        proj.shape = [meta["height"], meta["width"]]

    # Assets
    for asset_name, asset_info in meta.get("assets", {}).items():
        href = asset_info["href"]
        item.add_asset(
            key=asset_name,
            asset=Asset(
                href=href,
                media_type=MediaType.GEOTIFF,
                title=asset_name.replace("_", " ").title(),
                roles=["data"],
                extra_fields={
                    "raster:bands": [
                        {
                            "nodata": asset_info.get("nodata"),
                            "data_type": asset_info.get("dtype", "float32"),
                        }
                    ]
                },
            ),
        )

    return item


def _build_ifg_item(
    ifgs: FrameInterferogramCollection,
    pair_name: str,
) -> pystac.Item:
    """Build a STAC Item for a single interferogram pair."""
    from pystac import Asset, Item, MediaType
    from pystac.extensions.projection import ProjectionExtension

    item_meta = ifgs.item(pair_name)
    grid = item_meta["grid"]
    bounds: list[float] = grid["bounds"]
    bbox, _ = _bounds_to_wgs84(bounds, grid["crs"])
    epsg = _crs_to_epsg(grid["crs"])

    ref_date = item_meta["reference_date"]
    sec_date = item_meta["secondary_date"]
    sec_dt = _parse_date(sec_date)

    from shapely.geometry import box as shapely_box

    geom_shape = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])

    properties: dict[str, Any] = {
        "frame:type": "FrameInterferogramItem",
        "frame:pair_name": pair_name,
        "frame:reference_date": ref_date,
        "frame:secondary_date": sec_date,
    }

    # Copy optional fields from item.json
    for key in (
        "temporal_baseline_days",
        "baseline",
        "source_processor",
        "source_product_id",
        "pass_direction",
        "orbit_number",
        "heading",
        "looks",
    ):
        val = item_meta.get(key)
        if val is not None:
            properties[f"frame:{key}"] = val

    item = Item(
        id=pair_name,
        geometry=geom_shape.__geo_interface__,
        bbox=bbox,
        datetime=sec_dt,
        properties=properties,
    )

    # start_datetime / end_datetime for STAC datetime range
    item.common_metadata.start_datetime = _parse_date(ref_date)
    item.common_metadata.end_datetime = sec_dt

    # Projection extension
    if epsg is not None:
        proj = ProjectionExtension.ext(item, add_if_missing=True)
        proj.epsg = epsg
        proj.transform = list(grid["transform"])[:6]
        proj.shape = [grid["height"], grid["width"]]

    # Assets
    for asset_name, asset_info in item_meta.get("assets", {}).items():
        href = asset_info["href"]
        item.add_asset(
            key=asset_name,
            asset=Asset(
                href=href,
                media_type=MediaType.GEOTIFF,
                title=asset_name.replace("_", " ").title(),
                roles=["data"],
            ),
        )

    return item


def _build_geometry_collection(
    geometry: FrameGeometry,
    temporal_extent: tuple[datetime | None, datetime | None] | None,
    ifgs: FrameInterferogramCollection | None,
) -> pystac.Collection:
    """Build a STAC Collection for geometry assets."""
    import pystac

    geom_meta = geometry.metadata
    if geom_meta is None:
        msg = "geometry.json not found; cannot build collection."
        raise FileNotFoundError(msg)
    bbox_wgs84, _ = _bounds_to_wgs84(geom_meta["bounds"], geom_meta["crs"])
    t_start, t_end = _resolve_temporal_extent(ifgs, temporal_extent)

    collection = pystac.Collection(
        id="geometry",
        description=(
            "Frame-level geometry rasters (incidence, DEM, water mask, etc.)."
        ),
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent(bboxes=[bbox_wgs84]),
            temporal=pystac.TemporalExtent(intervals=[[t_start, t_end]]),
        ),
    )
    collection.add_item(_build_geometry_item(geometry))
    return collection


def _build_ifg_collection(
    ifgs: FrameInterferogramCollection,
    temporal_extent: tuple[datetime | None, datetime | None] | None,
    geometry: FrameGeometry | None,
) -> pystac.Collection:
    """Build a STAC Collection for interferogram pairs."""
    import pystac

    index_meta = ifgs.index_metadata
    pair_names = index_meta["pairs"] if index_meta else []

    spatial = (
        _spatial_from_geom(geometry)
        if geometry is not None and geometry.metadata is not None
        else _spatial_from_ifgs(ifgs, pair_names)
    )
    t_start, t_end = _resolve_temporal_extent(ifgs, temporal_extent)

    collection = pystac.Collection(
        id="interferograms",
        description=f"Interferogram pairs ({len(pair_names)} pairs).",
        extent=pystac.Extent(
            spatial=spatial,
            temporal=pystac.TemporalExtent(intervals=[[t_start, t_end]]),
        ),
    )

    for i, pname in enumerate(pair_names):
        try:
            collection.add_item(_build_ifg_item(ifgs, pname))
        except Exception as e:
            logger.warning("Skipping pair %s: %s", pname, e)
            continue
        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d pairs...", i + 1, len(pair_names))

    return collection


def build_stac_catalog(
    geometry: FrameGeometry | None = None,
    ifgs: FrameInterferogramCollection | None = None,
    *,
    catalog_id: str = "insar-frame",
    description: str = "",
    temporal_extent: tuple[datetime | None, datetime | None] | None = None,
    output_dir: str | Path | None = None,
    catalog_type: Any = None,
) -> pystac.Catalog:
    """Build a STAC Catalog from frame products.

    Parameters
    ----------
    geometry : FrameGeometry, optional
        Geometry assets to include as a child collection.
    ifgs : FrameInterferogramCollection, optional
        Interferograms to include as a child collection.
    catalog_id : str
        STAC Catalog id.
    description : str
        Catalog description.
    temporal_extent : (start, end), optional
        Temporal extent for collections. If *None*, auto-detected.
    output_dir : str or Path, optional
        If provided, save the catalog here.
    catalog_type : pystac.CatalogType, optional
        STAC catalog type. Defaults to ``SELF_CONTAINED``.

    Returns
    -------
    pystac.Catalog

    """
    import pystac

    if geometry is None and ifgs is None:
        msg = "At least one of 'geometry' or 'ifgs' must be provided."
        raise ValueError(msg)

    if catalog_type is None:
        catalog_type = pystac.CatalogType.SELF_CONTAINED

    catalog = pystac.Catalog(id=catalog_id, description=description)

    if geometry is not None and geometry.metadata is not None:
        catalog.add_child(_build_geometry_collection(geometry, temporal_extent, ifgs))

    if ifgs is not None:
        catalog.add_child(_build_ifg_collection(ifgs, temporal_extent, geometry))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog.normalize_and_save(str(output_dir), catalog_type=catalog_type)
        logger.info("STAC catalog saved to %s", output_dir)

    return catalog


def _resolve_temporal_extent(
    ifgs: FrameInterferogramCollection | None,
    user_extent: tuple[datetime | None, datetime | None] | None,
) -> tuple[datetime | None, datetime | None]:
    """Return (start, end) temporal extent."""
    if user_extent is not None:
        return user_extent

    if ifgs is not None:
        index_meta = ifgs.index_metadata
        if index_meta and index_meta.get("pairs"):
            dates = []
            for pname in index_meta["pairs"]:
                parts = pname.split("_")
                for part in parts[:2]:
                    with contextlib.suppress(ValueError):
                        dates.append(_parse_date(part))
            if dates:
                return min(dates), max(dates)

    return None, None


def _spatial_from_geom(geometry: FrameGeometry) -> pystac.SpatialExtent:
    """Build SpatialExtent from geometry metadata."""
    import pystac

    meta = geometry.metadata
    if meta is None:
        return pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]])
    bbox_wgs84, _ = _bounds_to_wgs84(meta["bounds"], meta["crs"])
    return pystac.SpatialExtent(bboxes=[bbox_wgs84])


def _spatial_from_ifgs(
    ifgs: FrameInterferogramCollection, pair_names: list[str]
) -> pystac.SpatialExtent:
    """Build SpatialExtent from the first pair's grid."""
    import pystac

    if not pair_names:
        return pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]])
    try:
        item_meta = ifgs.item(pair_names[0])
        grid = item_meta["grid"]
        bbox_wgs84, _ = _bounds_to_wgs84(grid["bounds"], grid["crs"])
        return pystac.SpatialExtent(bboxes=[bbox_wgs84])
    except Exception:
        return pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]])


def _stac_to_frame_meta(
    catalog_path: str | Path,
    frame_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a faninsar STAC catalog back to on-disk frame directories.

    This is the shared helper used by :meth:`Frame.from_stac` (local) and
    the future :meth:`Frame.open_remote` (M5). It walks the catalog,
    identifies the ``geometry`` and ``interferograms`` collections, and
    resolves their asset hrefs to absolute directory paths on disk.

    Parameters
    ----------
    catalog_path : str or Path
        Path to the ``catalog.json`` written by ``Frame.to_stac()``.
    frame_root : str or Path, optional
        Explicit frame root directory. When provided, this overrides href
        resolution — the geometry/ and interferograms/ sub-directories are
        looked up directly under *frame_root*. This is the recommended way
        to call from_stac when the STAC catalog was written to a separate
        output directory (so the on-disk frame and the STAC index live in
        different trees). When *None*, hrefs are resolved from the catalog.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``frame_root``: :class:`pathlib.Path` — common parent of geometry/
          and interferograms/ (or whichever exists).
        - ``geometry_dir``: :class:`pathlib.Path` or ``None``.
        - ``interferograms_dir``: :class:`pathlib.Path` or ``None``.

    Raises
    ------
    FileNotFoundError
        If the catalog file does not exist.
    ValueError
        If the catalog has no recognisable geometry or interferograms
        collection.

    """
    import pystac

    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        msg = f"STAC catalog not found: {catalog_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # When frame_root is provided, resolve sub-dirs directly — no href parsing.
    # This is the robust path when the STAC catalog was written to a separate
    # output directory.
    if frame_root is not None:
        frame_root = Path(frame_root)
        geometry_dir = frame_root / "geometry"
        interferograms_dir = frame_root / "interferograms"
        result = {
            "frame_root": frame_root,
            "geometry_dir": geometry_dir if geometry_dir.is_dir() else None,
            "interferograms_dir": (
                interferograms_dir if interferograms_dir.is_dir() else None
            ),
        }
        logger.debug(
            "STAC→frame meta (explicit root): %s",
            dict(result),
        )
        return result

    catalog = pystac.Catalog.from_file(str(catalog_path))

    geometry_dir: Path | None = None
    interferograms_dir: Path | None = None

    for child in catalog.get_children():
        if child.id == "geometry":
            geometry_dir = _resolve_collection_dir(child)
        elif child.id == "interferograms":
            resolved = _resolve_collection_dir(child)
            # _resolve_collection_dir returns the dir of the first item's first
            # asset — for interferograms that is the pair dir, so step up one.
            if resolved is not None and resolved.name not in ("interferograms",):
                resolved = resolved.parent
            interferograms_dir = resolved

    # Single-item catalogs (no child collections) — fall back to scanning items.
    if geometry_dir is None and interferograms_dir is None:
        items = list(catalog.get_all_items())
        for item in items:
            props = item.properties or {}
            item_type = props.get("frame:type", "")
            d = _resolve_item_dir(item)
            if item_type == "FrameGeometry" and geometry_dir is None:
                geometry_dir = d
            elif item_type == "FrameInterferogramItem" and interferograms_dir is None:
                # The interferograms dir is the parent of the pair dir.
                interferograms_dir = d.parent if d is not None else None

    if geometry_dir is None and interferograms_dir is None:
        msg = (
            f"Catalog at {catalog_path} does not look like a faninsar frame "
            "catalog (no 'geometry' or 'interferograms' collection, and no "
            "items with frame:type markers)."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Derive frame root as the common parent.
    candidates = [d for d in (geometry_dir, interferograms_dir) if d is not None]
    if len(candidates) == 1:
        resolved_root = candidates[0].parent
    else:
        # common parent of two sibling dirs
        resolved_root = candidates[0].parent
        for d in candidates[1:]:
            if d.parent != resolved_root:
                # Find common ancestor
                resolved_root = _common_ancestor(resolved_root, d)

    logger.debug(
        "STAC→frame meta: root=%s, geometry=%s, interferograms=%s",
        resolved_root,
        geometry_dir,
        interferograms_dir,
    )
    return {
        "frame_root": resolved_root,
        "geometry_dir": geometry_dir,
        "interferograms_dir": interferograms_dir,
    }


def _resolve_collection_dir(collection: pystac.Collection) -> Path | None:
    """Resolve the on-disk directory a STAC collection's items live in."""
    items = list(collection.get_items())
    if not items:
        return None
    return _resolve_item_dir(items[0])


def _resolve_item_dir(item: pystac.Item) -> Path:
    """Resolve the on-disk directory an item's first asset lives in.

    Handles both absolute and relative hrefs (resolved against the item's
    self href).

    """
    assets = item.get_assets()
    if not assets:
        # Fall back to the item's own href's parent.
        self_href = item.get_self_href()
        if self_href is None:
            msg = f"STAC item {item.id!r} has no assets and no self href."
            logger.error(msg)
            raise ValueError(msg)
        return Path(self_href).parent

    first_asset = next(iter(assets.values()))
    href = first_asset.href

    if "://" in href:
        # Remote URL — not supported by from_stac (use open_remote in M5).
        msg = (
            f"STAC item {item.id!r} has a remote asset href {href!r}; "
            "Frame.from_stac() only supports local catalogs. "
            "Use Frame.open_remote() for remote catalogs (M5)."
        )
        logger.error(msg)
        raise ValueError(msg)

    asset_path = Path(href)
    if not asset_path.is_absolute():
        self_href = item.get_self_href()
        if self_href is not None:
            asset_path = Path(self_href).parent / href
        asset_path = asset_path.resolve()

    return asset_path.parent


def _common_ancestor(a: Path, b: Path) -> Path:
    """Return the deepest common parent directory of two paths."""
    a_parents = list(reversed(a.parents))
    b_parents = set(b.parents)
    for parent in a_parents:
        if parent in b_parents:
            return parent
    return Path("/")

"""Frame-level geometry assets for one InSAR frame."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from faninsar.data.datasets.geogrid import GeoGrid
from faninsar.logging import setup_logger

from .exceptions import (
    FrameGeometryError,
    GridMismatchError,
    MissingGeometryAssetError,
)
from .metadata import (
    CATEGORICAL_ASSETS,
    GEOMETRY_ASSETS,
    GeometryAssetName,
    build_geometry_metadata,
    load_json,
    save_json,
)
from .raster_io import (
    read_geogrid,
    reproject_to_geogrid,
    validate_alignment,
    write_cog,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from typing import Self

    import pystac

    from faninsar.datasets.frame.interferogram import FrameInterferogramCollection
    from faninsar.data.datasets.xarray_dataset import XarrayDataset
    from faninsar.data.query import BoundingBox

logger = setup_logger(__name__)


_ASSET_FILENAMES: dict[str, str] = {name: f"{name}.cog.tif" for name in GEOMETRY_ASSETS}


class FrameGeometry:
    """Standardized frame-level geometry assets for one InSAR frame.

    Parameters
    ----------
    root : str or Path
        Path to the geometry directory (e.g. ``frame/geometry``).

    Examples
    --------
    Create from raw rasters:

    >>> geom = FrameGeometry.from_rasters(
    ...     out_dir="frame",
    ...     incidence="/path/to/inc_map_ell.tif",
    ...     angle_unit="radian",
    ... )

    Open an asset:

    >>> inc = geom.open("incidence", chunks={"y": 512, "x": 512})

    Clip with a bounding box:

    >>> from faninsar.data.query import BoundingBox
    >>> bbox = BoundingBox(10.0, 45.0, 11.0, 46.0, crs="EPSG:4326")
    >>> clipped = geom.clip_bbox("incidence", bbox)

    """

    def __init__(self, root: str | Path) -> None:
        """Initialise FrameGeometry from an existing geometry directory."""
        self._root = Path(root)
        if not self._root.is_dir():
            msg = f"Geometry directory not found: {self._root}"
            raise FileNotFoundError(msg)

        self._metadata_path = self._root / "geometry.json"
        self._metadata: dict[str, Any] | None = None
        if self._metadata_path.exists():
            self._metadata = load_json(self._metadata_path)

    @property
    def root(self) -> Path:
        """Root directory of the geometry assets."""
        return self._root

    @property
    def crs(self) -> Any:
        """CRS of the geometry assets."""
        meta = self._require_metadata()
        return meta["crs"]

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Parsed geometry.json content, or None if the file is absent."""
        return self._metadata

    def _require_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            msg = (
                f"geometry.json not found at {self._metadata_path}. "
                "Create the geometry first with FrameGeometry.from_rasters()."
            )
            raise FrameGeometryError(msg)
        return self._metadata

    @classmethod
    def from_rasters(
        cls,
        out_dir: str | Path,
        incidence: str | Path,
        azimuth: str | Path | None = None,
        heading: str | Path | None = None,
        dem: str | Path | None = None,
        water_mask: str | Path | None = None,
        reference: str | Path | None = None,
        angle_unit: Literal["degree", "radian"] = "degree",
        azimuth_convention: Literal[
            "look_azimuth_from_north_clockwise",
            "look_azimuth_from_east_counterclockwise",
            "processor_native",
        ] = "look_azimuth_from_north_clockwise",
        heading_convention: Literal[
            "satellite_heading_from_north_clockwise",
            "processor_native",
        ] = "satellite_heading_from_north_clockwise",
        overwrite: bool = False,
    ) -> Self:
        """Standardize geocoded geometry rasters into a frame layout.

        Parameters
        ----------
        out_dir : str or Path
            Output directory. A ``geometry`` subdirectory is created unless
            *out_dir* already points to a directory named ``geometry``.
        incidence : str or Path
            Path to the incidence angle raster.
        azimuth : str or Path, optional
            Path to the look azimuth raster.
        heading : str or Path, optional
            Path to the satellite heading raster.
        dem : str or Path, optional
            Path to the DEM raster.
        water_mask : str or Path, optional
            Path to the water mask raster.
        reference : str or Path, optional
            Reference raster for the output grid. If *None*, ``incidence`` is
            used as the reference.
        angle_unit : ``"degree"`` or ``"radian"``
            Unit of the angle rasters.
        azimuth_convention : str
            Convention for the azimuth raster.
        heading_convention : str
            Convention for the heading raster.
        overwrite : bool
            If *True*, overwrite existing outputs.

        Returns
        -------
        FrameGeometry
            A new :class:`FrameGeometry` pointing to the output directory.

        """
        import rasterio.enums

        out_dir = Path(out_dir)
        geometry_dir = out_dir / "geometry" if out_dir.name != "geometry" else out_dir
        geometry_dir.mkdir(parents=True, exist_ok=True)

        ref_path = Path(reference) if reference is not None else Path(incidence)
        ref_grid = read_geogrid(ref_path)

        raster_inputs: dict[str, Path | None] = {
            "incidence": Path(incidence),
            "azimuth": Path(azimuth) if azimuth else None,
            "heading": Path(heading) if heading else None,
            "dem": Path(dem) if dem else None,
            "water_mask": Path(water_mask) if water_mask else None,
        }

        # Check for existing outputs before writing
        if not overwrite:
            for name, src_path in raster_inputs.items():
                if src_path is not None:
                    out_path = geometry_dir / _ASSET_FILENAMES[name]
                    if out_path.exists():
                        msg = f"Output file already exists: {out_path}"
                        raise FileExistsError(msg)

        assets_written: dict[str, dict[str, Any]] = {}
        source_assets_info: dict[str, dict[str, Any]] = {}

        for name, src_path in raster_inputs.items():
            if src_path is None:
                continue

            out_path = geometry_dir / _ASSET_FILENAMES[name]

            if out_path.exists() and not overwrite:
                logger.info("Asset '%s' already exists, skipping.", name)
                with __import__("rasterio").open(out_path) as ds:
                    assets_written[name] = {
                        "href": _ASSET_FILENAMES[name],
                        "dtype": str(ds.dtypes[0]),
                        "nodata": ds.nodata,
                        "shape": [ds.height, ds.width],
                    }
                continue

            is_categorical = name in CATEGORICAL_ASSETS

            if is_categorical:
                resampling = rasterio.enums.Resampling.nearest
                out_dtype = "uint8"
                out_nodata = 255
            else:
                resampling = rasterio.enums.Resampling.bilinear
                out_dtype = "float32"
                out_nodata = -9999.0

            arr = reproject_to_geogrid(
                src_path,
                ref_grid,
                resampling=resampling,
                dst_dtype=out_dtype,
                dst_nodata=out_nodata,
            )

            if angle_unit == "radian" and name in {"incidence", "azimuth", "heading"}:
                arr = np.degrees(arr)

            write_cog(
                arr,
                out_path,
                ref_grid,
                nodata=out_nodata,
                dtype=out_dtype,
                overwrite=overwrite,
            )

            assets_written[name] = {
                "href": _ASSET_FILENAMES[name],
                "dtype": out_dtype,
                "nodata": out_nodata,
                "shape": list(arr.shape[-2:]),
            }
            source_assets_info[name] = {
                "source_path": str(src_path),
                "source_dtype": str(out_dtype),
            }

        for name in assets_written:
            asset_path = geometry_dir / _ASSET_FILENAMES[name]
            if not validate_alignment(asset_path, ref_grid):
                msg = f"Asset '{name}' is not aligned with the reference grid."
                raise GridMismatchError(msg)

        bounds = ref_grid.bounds
        res = ref_grid.resolution
        resolution = (abs(res.x), abs(res.y))

        processing: dict[str, Any] = {"resampling": "bilinear"}
        if angle_unit == "radian":
            processing["angle_conversion"] = "radian_to_degree"

        # Canonical value ranges for geometry assets (post-conversion units).
        # Angles are normalized to degrees here; DEM is unbounded so it is left
        # out. Recording these prevents unit confusion downstream.
        geom_value_ranges: dict[str, tuple[float, float]] = {}
        if "incidence" in assets_written:
            geom_value_ranges["incidence"] = (0.0, 90.0)
        if "azimuth" in assets_written:
            geom_value_ranges["azimuth"] = (0.0, 360.0)
        if "heading" in assets_written:
            geom_value_ranges["heading"] = (0.0, 360.0)

        meta = build_geometry_metadata(
            crs=ref_grid.crs,
            width=ref_grid.width,
            height=ref_grid.height,
            transform=ref_grid.transform,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            resolution=resolution,
            angle_unit="degree" if angle_unit == "radian" else angle_unit,
            azimuth_convention=azimuth_convention,
            heading_convention=heading_convention,
            assets=assets_written,
            source_assets=source_assets_info,
            processing=processing,
            value_ranges=geom_value_ranges or None,
        )
        save_json(meta, geometry_dir / "geometry.json")
        logger.info("FrameGeometry created at %s", geometry_dir)

        return cls(geometry_dir)

    @classmethod
    def from_hyp3(
        cls,
        out_dir: str | Path,
        product_dir: str | Path,
        *,
        include_azimuth: bool = False,
        include_heading: bool = False,
        reference: str | Path | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Standardize geometry rasters from a HyP3 GAMMA product.

        Auto-discovers ``*_inc_map_ell.tif``, ``*_dem.tif``,
        ``*_water_mask.tif``, and optionally ``*_lv_phi.tif`` /
        ``*_lv_theta.tif`` from *product_dir*.

        Parameters
        ----------
        out_dir : str or Path
            Output directory for the frame layout.
        product_dir : str or Path
            Path to a single HyP3 product folder containing the geometry
            rasters (e.g. ``ReferenceIFG``).
        include_azimuth : bool
            If *True*, look for ``*_lv_phi.tif`` and ``*_lv_theta.tif`` to
            derive the look azimuth. Not yet implemented — raises
            ``NotImplementedError``.
        include_heading : bool
            If *True*, parse the ``*.txt`` metadata for satellite heading.
            Not yet implemented — raises ``NotImplementedError``.
        reference : str or Path, optional
            Reference raster for the output grid. If *None*, the incidence
            raster is used.
        overwrite : bool
            If *True*, overwrite existing outputs.

        Returns
        -------
        FrameGeometry

        Raises
        ------
        FileNotFoundError
            If required rasters (incidence, DEM, water mask) are not found
            in *product_dir*.

        """
        product_dir = Path(product_dir)
        if not product_dir.is_dir():
            msg = f"HyP3 product directory not found: {product_dir}"
            raise FileNotFoundError(msg)

        if include_azimuth:
            msg = "include_azimuth is not yet implemented."
            raise NotImplementedError(msg)
        if include_heading:
            msg = "include_heading is not yet implemented."
            raise NotImplementedError(msg)

        # Discover required rasters
        inc_candidates = list(product_dir.glob("*_inc_map_ell.tif"))
        dem_candidates = list(product_dir.glob("*_dem.tif"))
        water_candidates = list(product_dir.glob("*_water_mask.tif"))

        if not inc_candidates:
            msg = f"No *_inc_map_ell.tif found in {product_dir}"
            raise FileNotFoundError(msg)
        if not dem_candidates:
            msg = f"No *_dem.tif found in {product_dir}"
            raise FileNotFoundError(msg)
        if not water_candidates:
            msg = f"No *_water_mask.tif found in {product_dir}"
            raise FileNotFoundError(msg)

        incidence = inc_candidates[0]
        dem = dem_candidates[0]
        water_mask = water_candidates[0]

        logger.info(
            "HyP3 geometry discovery: incidence=%s, dem=%s, water_mask=%s",
            incidence.name,
            dem.name,
            water_mask.name,
        )

        return cls.from_rasters(
            out_dir=out_dir,
            incidence=incidence,
            dem=dem,
            water_mask=water_mask,
            reference=reference,
            angle_unit="radian",
            overwrite=overwrite,
        )

    def path(self, name: GeometryAssetName) -> Path:
        """Return the file path for a geometry asset."""
        if name not in _ASSET_FILENAMES:
            msg = f"Unknown geometry asset: {name}"
            raise ValueError(msg)
        return self._root / _ASSET_FILENAMES[name]

    def exists(self, name: GeometryAssetName) -> bool:
        """Check if a geometry asset file exists."""
        return self.path(name).exists()

    def open(
        self,
        name: GeometryAssetName,
        *,
        masked: bool = True,
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
    ) -> xr.DataArray:
        """Open a geometry asset as a lazy xarray DataArray.

        Parameters
        ----------
        name : GeometryAssetName
            Name of the asset to open.
        masked : bool
            If *True*, apply the nodata mask.
        chunks : dict, int, ``"auto"``, or None
            Chunk sizes for dask arrays. *None* loads eagerly.

        Returns
        -------
        xarray.DataArray
            The loaded raster data.

        """
        import rioxarray

        asset_path = self.path(name)
        if not asset_path.exists():
            raise MissingGeometryAssetError(name)

        da = rioxarray.open_rasterio(asset_path, masked=masked, chunks=chunks)
        if not isinstance(da, xr.DataArray):
            msg = f"Expected DataArray, got {type(da).__name__}"
            raise TypeError(msg)
        if da.ndim == 3:
            da = da.squeeze("band", drop=True)
        return da

    def validate_alignment(self, tol: float = 1e-6) -> bool:
        """Validate that all existing assets share the same grid.

        Returns
        -------
        bool
            *True* if all assets are mutually aligned.

        """
        meta = self._require_metadata()
        ref_grid = GeoGrid.from_bounds(
            meta["bounds"],
            crs=meta["crs"],
            shape=(meta["height"], meta["width"]),
            tight=True,
        )
        for name in GEOMETRY_ASSETS:
            p = self.path(name)
            if p.exists() and not validate_alignment(p, ref_grid, tol=tol):
                return False
        return True

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the geometry."""
        meta = self._require_metadata()
        present = {name: self.exists(name) for name in GEOMETRY_ASSETS}
        return {
            "root": str(self._root),
            "crs": meta["crs"],
            "width": meta["width"],
            "height": meta["height"],
            "resolution": meta["resolution"],
            "bounds": meta["bounds"],
            "assets_present": present,
        }

    def clip_bbox(
        self,
        name: GeometryAssetName,
        bbox: BoundingBox,
        *,
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
    ) -> xr.DataArray:
        """Clip a geometry asset to a bounding box.

        Parameters
        ----------
        name : GeometryAssetName
            Name of the asset.
        bbox : BoundingBox
            Spatial extent to clip. If the CRS differs from the raster, the
            bbox is reprojected to the raster CRS.
        chunks : dict, int, ``"auto"``, or None
            Chunk sizes for dask arrays.

        Returns
        -------
        xarray.DataArray
            Clipped raster data.

        """
        meta = self._require_metadata()
        raster_crs = meta["crs"]

        if bbox.crs is not None and str(bbox.crs) != str(raster_crs):
            bbox = bbox.to_crs(raster_crs)
        elif bbox.crs is None:
            logger.warning(
                "BoundingBox has no CRS; assuming raster CRS: %s", raster_crs
            )

        da = self.open(name, masked=True, chunks=chunks)

        x_dim = "x"
        y_dim = "y"
        return da.sel(
            {
                x_dim: slice(bbox.left, bbox.right),
                y_dim: slice(bbox.top, bbox.bottom),
            }
        )

    def compute_los_unit_vectors(
        self,
        *,
        incidence_name: GeometryAssetName = "incidence",
        azimuth_name: GeometryAssetName = "azimuth",
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
    ) -> xr.Dataset:
        """Compute line-of-sight (LOS) unit vector components.

        Parameters
        ----------
        incidence_name : GeometryAssetName
            Name of the incidence angle asset (degrees from vertical).
        azimuth_name : GeometryAssetName
            Name of the azimuth asset (look azimuth clockwise from north, in degrees).
        chunks : dict, int, ``"auto"``, or None
            Chunk sizes for dask arrays.

        Returns
        -------
        xarray.Dataset
            Dataset with variables ``los_east``, ``los_north``, ``los_up``.

        Raises
        ------
        MissingGeometryAssetError
            If either incidence or azimuth asset is missing.

        """
        if not self.exists(incidence_name):
            raise MissingGeometryAssetError(incidence_name)
        if not self.exists(azimuth_name):
            raise MissingGeometryAssetError(azimuth_name)

        inc = self.open(incidence_name, masked=True, chunks=chunks)
        azi = self.open(azimuth_name, masked=True, chunks=chunks)

        inc_rad = np.deg2rad(inc)
        azi_rad = np.deg2rad(azi)

        los_east = -np.sin(inc_rad) * np.sin(azi_rad)
        los_north = -np.sin(inc_rad) * np.cos(azi_rad)
        los_up = np.cos(inc_rad)

        return xr.Dataset(
            {
                "los_east": los_east,
                "los_north": los_north,
                "los_up": los_up,
            }
        )

    def to_xarray_dataset(
        self,
        names: Sequence[GeometryAssetName] | None = None,
        *,
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
    ) -> xr.Dataset:
        """Open multiple geometry assets as an xarray Dataset.

        Parameters
        ----------
        names : sequence of GeometryAssetName, optional
            Assets to include. If *None*, all existing assets are included.
        chunks : dict, int, ``"auto"``, or None
            Chunk sizes for dask arrays.

        Returns
        -------
        xarray.Dataset
            Dataset with one variable per asset.

        """
        if names is None:
            names = [n for n in GEOMETRY_ASSETS if self.exists(n)]

        data_vars: dict[str, xr.DataArray] = {}
        for name in names:
            if self.exists(name):
                data_vars[name] = self.open(name, masked=True, chunks=chunks)

        return xr.Dataset(data_vars)

    def to_faninsar_dataset(self, name: GeometryAssetName) -> XarrayDataset:
        """Wrap a single geometry asset in a FanInSAR :class:`XarrayDataset`.

        This bridges the geometry asset to FanInSAR's existing
        :class:`~faninsar.datasets.XarrayDataset` query system, enabling
        :class:`~faninsar.data.query.BoundingBox`-based ``boxes_query`` workflows.

        Parameters
        ----------
        name : GeometryAssetName
            Name of the geometry asset to wrap.

        Returns
        -------
        XarrayDataset
            A FanInSAR ``XarrayDataset`` backed by the named asset COG.

        Raises
        ------
        MissingGeometryAssetError
            If the named asset does not exist on disk.

        """
        from faninsar.data.datasets.xarray_dataset import XarrayDataset

        asset_path = self.path(name)
        if not asset_path.exists():
            raise MissingGeometryAssetError(name)
        return XarrayDataset(paths=[asset_path])

    def open_incidence(self, **kwargs: Any) -> xr.DataArray:
        """Open the incidence angle asset."""
        return self.open("incidence", **kwargs)

    def open_azimuth(self, **kwargs: Any) -> xr.DataArray:
        """Open the azimuth asset."""
        return self.open("azimuth", **kwargs)

    def open_heading(self, **kwargs: Any) -> xr.DataArray:
        """Open the heading asset."""
        return self.open("heading", **kwargs)

    def open_dem(self, **kwargs: Any) -> xr.DataArray:
        """Open the DEM asset."""
        return self.open("dem", **kwargs)

    def open_water_mask(self, **kwargs: Any) -> xr.DataArray:
        """Open the water mask asset."""
        return self.open("water_mask", **kwargs)

    def to_stac(
        self,
        ifgs: FrameInterferogramCollection | None = None,
        *,
        catalog_id: str = "insar-frame",
        description: str = "",
        temporal_extent: tuple[datetime | None, datetime | None] | None = None,
        output_dir: str | Path | None = None,
        catalog_type: pystac.CatalogType | None = None,
    ) -> pystac.Catalog:
        """Generate a STAC Catalog from this geometry (and optionally interferograms).

        Parameters
        ----------
        ifgs : FrameInterferogramCollection, optional
            If provided, interferogram items are added as a child collection.
        catalog_id : str
            STAC Catalog id.
        description : str
            Catalog description.
        temporal_extent : tuple of (start, end), optional
            Temporal extent. If *None*, auto-detected from pair dates or
            defaults to ``(2014-01-01, None)``.
        output_dir : str or Path, optional
            If provided, save the catalog to this directory.
        catalog_type : pystac.CatalogType, optional
            STAC catalog type (default: ``SELF_CONTAINED``).

        Returns
        -------
        pystac.Catalog
            The generated STAC Catalog.

        """
        from .stac import build_stac_catalog

        return build_stac_catalog(
            geometry=self,
            ifgs=ifgs,
            catalog_id=catalog_id,
            description=description,
            temporal_extent=temporal_extent,
            output_dir=output_dir,
            catalog_type=catalog_type,
        )

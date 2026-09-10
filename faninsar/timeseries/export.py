"""Export adapters for :class:`~faninsar.timeseries.results.TimeSeries`.

The module owns one small public seam: a process-local registry maps stable
names to zero-argument exporter classes.  The built-in adapters deliberately
write a complete temporary product and publish its STAC item last.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.timeseries.results import TimeSeries

logger = setup_logger(__name__)


class TimeSeriesExporter(Protocol):
    """Callable adapter used at the TimeSeries export seam."""

    def __call__(
        self,
        result: TimeSeries,
        path: Path,
        *,
        overwrite: bool,
    ) -> Path:
        """Write one complete result product and return its root."""


class ExporterRegistry:
    """Process-local registry for explicitly selected exporter classes."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._exporters: dict[str, type[TimeSeriesExporter]] = {}

    def register(self, name: str, exporter: type[TimeSeriesExporter]) -> None:
        """Register one zero-argument exporter class under ``name``."""
        if not isinstance(name, str) or not name or name != name.strip():
            message = "exporter name must be a non-empty string"
            logger.error(message)
            raise ValueError(message)
        if not isinstance(exporter, type) or not callable(exporter):
            message = "registered exporter must be a callable class"
            logger.error(message)
            raise TypeError(message)
        if name in self._exporters:
            message = f"exporter name already registered: {name}"
            logger.error(message)
            raise ValueError(message)
        self._exporters[name] = exporter

    def resolve(self, name: str) -> type[TimeSeriesExporter]:
        """Resolve one registered exporter name."""
        try:
            return self._exporters[name]
        except KeyError as error:
            message = f"no TimeSeries exporter is registered as {name!r}"
            logger.exception(message)
            raise KeyError(message) from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered exporter names in stable order."""
        return tuple(sorted(self._exporters))


registry = ExporterRegistry()


def _json_safe(value: Any) -> Any:
    """Convert metadata values into JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _metadata(result: TimeSeries) -> dict[str, Any]:
    """Return one structured metadata mapping for all built-in adapters."""
    metadata = _json_safe(dict(result.metadata))
    if result.spatial is not None:
        metadata["spatial"] = _json_safe(result.spatial.to_dict())
    metadata["pair_ids"] = list(result.pair_ids)
    metadata["dates"] = list(result.dates)
    return metadata


def _netcdf_attributes(result: TimeSeries) -> dict[str, Any]:
    """Return NetCDF-compatible attributes without losing structured metadata."""
    metadata = _metadata(result)
    return {
        key: value if isinstance(value, (str, int, float, bytes)) else json.dumps(value)
        for key, value in metadata.items()
    }


def _stac_properties(result: TimeSeries) -> dict[str, Any]:
    """Build STAC properties without deriving a geographic footprint."""
    properties: dict[str, Any] = {
        "start_datetime": f"{result.dates[0][:4]}-{result.dates[0][4:6]}-"
        f"{result.dates[0][6:]}T00:00:00Z",
        "end_datetime": f"{result.dates[-1][:4]}-{result.dates[-1][4:6]}-"
        f"{result.dates[-1][6:]}T00:00:00Z",
        "faninsar:pair_ids": list(result.pair_ids),
        "faninsar:dates": list(result.dates),
    }
    metadata = _json_safe(dict(result.metadata))
    if metadata:
        properties["faninsar:metadata"] = metadata
    if result.revision_id is not None:
        properties["faninsar:revision_id"] = result.revision_id
    if result.spatial is not None:
        spatial = _json_safe(result.spatial.to_dict())
        properties["faninsar:domain"] = spatial["domain"]
        properties["faninsar:shape"] = spatial["shape"]
        properties["faninsar:grid_identity"] = spatial["grid_identity"]
        for key in ("crs", "transform", "radar_coordinates"):
            if key in spatial:
                properties[f"faninsar:{key}"] = spatial[key]
    return properties


def _write_stac(
    result: TimeSeries,
    root: Path,
    assets: dict[str, tuple[str, str]],
) -> Path:
    """Write the final STAC Item after all assets are complete."""
    import pystac

    item = pystac.Item(
        id="timeseries",
        geometry=None,
        bbox=None,
        datetime=None,
        properties=_stac_properties(result),
    )
    for name, (href, media_type) in assets.items():
        item.add_asset(
            name,
            pystac.Asset(href=href, media_type=media_type, roles=["data"]),
        )
    item_path = root / "item.json"
    temporary_path = root / f".{item_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_text(
            json.dumps(item.to_dict(include_self_link=False), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(item_path)
    except Exception:
        logger.exception("could not publish STAC item: %s", item_path)
        temporary_path.unlink(missing_ok=True)
        raise
    return item_path


def _arrays(result: TimeSeries) -> dict[str, np.ndarray]:
    """Return scientific arrays using stable asset names."""
    arrays = {
        "increments": np.asarray(result.phase_increments_rad),
        "cumulative": np.asarray(result.phase_cumulative_rad),
        "residual_pairs": np.asarray(result.residual_phase_pairs_rad),
    }
    if result.displacement_increments_m is not None:
        arrays["displacement_increments_m"] = np.asarray(
            result.displacement_increments_m
        )
    if result.displacement_cumulative_m is not None:
        arrays["displacement_cumulative_m"] = np.asarray(
            result.displacement_cumulative_m
        )
    return arrays


def _write_stac_zarr(result: TimeSeries, root: Path, *, overwrite: bool) -> Path:
    """Write canonical Zarr and a STAC Item."""
    del overwrite
    from faninsar.timeseries.io import write_timeseries_zarr

    write_timeseries_zarr(result, root / "timeseries.zarr")
    _write_stac(
        result,
        root,
        {"timeseries": ("timeseries.zarr", "application/vnd+zarr")},
    )
    return root


def _geo_grid(result: TimeSeries) -> tuple[Any, Any]:
    """Validate geo metadata and return rasterio CRS and transform."""
    spatial = result.spatial
    if spatial is None or spatial.domain != "geo":
        message = "stac-cog export requires geo spatial metadata"
        logger.error(message)
        raise ValueError(message)
    if spatial.crs is None or spatial.transform is None:
        message = "stac-cog export requires complete geo CRS and transform"
        logger.error(message)
        raise ValueError(message)
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    return CRS.from_user_input(spatial.crs), Affine(*spatial.transform)


def _write_multiband_cog(
    array: np.ndarray,
    path: Path,
    *,
    crs: Any,
    transform: Any,
    labels: list[str],
) -> None:
    """Write one scientific variable as a multi-band Cloud Optimized GeoTIFF."""
    import rasterio

    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        message = "COG variables must be 2-D or 3-D arrays"
        raise ValueError(message)
    if len(labels) != values.shape[0]:
        message = "COG band labels must match the time or pair axis"
        raise ValueError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(
            path,
            "w",
            driver="COG",
            dtype="float32",
            width=values.shape[2],
            height=values.shape[1],
            count=values.shape[0],
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="DEFLATE",
            blocksize=256,
        ) as dataset:
            dataset.write(values)
            for index, label in enumerate(labels, start=1):
                dataset.set_band_description(index, label)
                dataset.update_tags(index, faninsar_identity=label)
    except Exception as error:
        logger.exception("could not write Cloud Optimized GeoTIFF: %s", path)
        message = "stac-cog export could not create a COG"
        raise RuntimeError(message) from error


def _write_stac_cog(result: TimeSeries, root: Path, *, overwrite: bool) -> Path:
    """Write geo-only multi-band COG assets and a STAC Item."""
    del overwrite
    crs, transform = _geo_grid(result)
    arrays = _arrays(result)
    assets: dict[str, tuple[str, str]] = {}
    labels_by_name = {
        "increments": list(result.dates[1:]),
        "cumulative": list(result.dates),
        "residual_pairs": list(result.pair_ids),
        "displacement_increments_m": list(result.dates[1:]),
        "displacement_cumulative_m": list(result.dates),
    }
    for name, array in arrays.items():
        filename = f"{name}.tif"
        _write_multiband_cog(
            array,
            root / filename,
            crs=crs,
            transform=transform,
            labels=labels_by_name[name],
        )
        assets[name] = (
            filename,
            "image/tiff; application=geotiff; profile=cloud-optimized",
        )
    _write_stac(result, root, assets)
    return root


def _write_netcdf(result: TimeSeries, root: Path, *, overwrite: bool) -> Path:
    """Write one self-describing NetCDF asset and a STAC Item."""
    del overwrite
    import xarray as xr

    arrays = _arrays(result)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "increments": (("date_interval", "y", "x"), arrays["increments"]),
        "cumulative": (("date", "y", "x"), arrays["cumulative"]),
        "residual_pairs": (("pair", "y", "x"), arrays["residual_pairs"]),
    }
    if "displacement_increments_m" in arrays:
        data_vars["displacement_increments_m"] = (
            ("date_interval", "y", "x"),
            arrays["displacement_increments_m"],
        )
    if "displacement_cumulative_m" in arrays:
        data_vars["displacement_cumulative_m"] = (
            ("date", "y", "x"),
            arrays["displacement_cumulative_m"],
        )
    dataset = xr.Dataset(
        data_vars,
        coords={
            "date": list(result.dates),
            "date_interval": list(result.dates[1:]),
            "pair": list(result.pair_ids),
        },
        attrs=_netcdf_attributes(result),
    )
    dataset.to_netcdf(root / "timeseries.nc")
    _write_stac(result, root, {"timeseries": ("timeseries.nc", "application/netcdf")})
    return root


class StacZarrExporter:
    """Built-in canonical STAC+Zarr exporter."""

    def __call__(self, result: TimeSeries, path: Path, *, overwrite: bool) -> Path:
        """Write canonical Zarr and STAC assets."""
        return _write_stac_zarr(result, path, overwrite=overwrite)


class StacCogExporter:
    """Built-in geo-only STAC+COG exporter."""

    def __call__(self, result: TimeSeries, path: Path, *, overwrite: bool) -> Path:
        """Write geo-referenced multi-band COG assets and STAC."""
        return _write_stac_cog(result, path, overwrite=overwrite)


class NetcdfExporter:
    """Built-in NetCDF exporter."""

    def __call__(self, result: TimeSeries, path: Path, *, overwrite: bool) -> Path:
        """Write NetCDF and STAC assets."""
        return _write_netcdf(result, path, overwrite=overwrite)


def write_timeseries_product(result: TimeSeries, path: str | Path) -> Path:
    """Persist one canonical STAC+Zarr time-series product.

    A repeated call to the same root creates a new immutable Zarr generation
    and updates the one STAC Item in place; no generation or publication
    choice is exposed to callers. The STAC Item is atomically replaced only
    after the Zarr generation is complete.

    Parameters
    ----------
    result : TimeSeries
        Scientific result to persist.
    path : str or pathlib.Path
        Canonical product root containing ``timeseries.zarr`` and
        ``item.json``.

    Returns
    -------
    pathlib.Path
        The canonical product root.

    Raises
    ------
    ValueError
        If ``path`` is an existing file or symbolic link.

    """
    root = Path(path)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        message = f"canonical time-series product root is not a directory: {root}"
        logger.error(message)
        raise ValueError(message)
    root.mkdir(parents=True, exist_ok=True)
    StacZarrExporter()(result, root, overwrite=True)
    return root


registry.register("stac-zarr", StacZarrExporter)
registry.register("stac-cog", StacCogExporter)
registry.register("netcdf", NetcdfExporter)


def _resolve_exporter(exporter: object | None) -> TimeSeriesExporter:
    """Resolve a default, registered name, class, or direct callable."""
    selected: object = "stac-cog" if exporter is None else exporter
    if isinstance(selected, str):
        selected = registry.resolve(selected)()
    elif isinstance(selected, type):
        try:
            selected = selected()
        except TypeError as error:
            message = "exporter classes must have a zero-argument constructor"
            logger.exception(message)
            raise TypeError(message) from error
    if not callable(selected):
        message = "exporter must be a registered name, class, or callable"
        logger.error(message)
        raise TypeError(message)
    return selected  # type: ignore[return-value]


def export_timeseries(
    result: TimeSeries,
    path: str | Path,
    *,
    overwrite: bool = False,
    exporter: object | None = None,
) -> object:
    """Export a TimeSeries through one selected adapter.

    Parameters
    ----------
    result : TimeSeries
        In-memory scientific result.
    path : str or pathlib.Path
        Complete destination product root.
    overwrite : bool, optional
        Replace an existing destination for built-in adapters.
    exporter : object, optional
        Registered exporter name, zero-argument exporter class, or callable.

    Returns
    -------
    object
        Exporter-defined return value.

    """
    root = Path(path)
    if root.exists() and not overwrite:
        raise FileExistsError(root)
    selected = _resolve_exporter(exporter)
    if not isinstance(selected, (StacZarrExporter, StacCogExporter, NetcdfExporter)):
        return selected(result, root, overwrite=overwrite)

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.export-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        selected(result, staging, overwrite=True)
        if root.exists():
            if root.is_dir():
                shutil.rmtree(root)
            else:
                root.unlink()
        staging.replace(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


__all__ = [
    "ExporterRegistry",
    "NetcdfExporter",
    "StacCogExporter",
    "StacZarrExporter",
    "TimeSeriesExporter",
    "export_timeseries",
    "registry",
    "write_timeseries_product",
]

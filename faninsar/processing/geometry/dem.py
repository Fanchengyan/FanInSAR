"""DEM sampling protocol and constant-height / raster DEM adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from rasterio.io import DatasetReader

logger = setup_logger(__name__)


def _natural_spline_six(
    values: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    """Evaluate the ISCE2 local six-sample natural spline."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.shape[-1] != 6:
        message = "six-sample spline requires a final axis of length 6"
        logger.error(message)
        raise ValueError(message)
    second = np.zeros_like(samples)
    recurrence = np.zeros(6, dtype=np.float64)
    for index in range(1, 5):
        denominator = recurrence[index - 1] / 2.0 + 2.0
        recurrence[index] = -0.5 / denominator
        second[..., index] = (
            3.0
            * (
                samples[..., index + 1]
                - 2.0 * samples[..., index]
                + samples[..., index - 1]
            )
            - second[..., index - 1] / 2.0
        ) / denominator
    for index in range(4, 0, -1):
        second[..., index] = (
            recurrence[index] * second[..., index + 1] + second[..., index]
        )
    local_fraction = np.asarray(fraction, dtype=np.float64)
    return samples[..., 1] + local_fraction * (
        samples[..., 2]
        - samples[..., 1]
        - second[..., 1] / 3.0
        - second[..., 2] / 6.0
        + local_fraction
        * (
            second[..., 1] / 2.0
            + local_fraction * (second[..., 2] - second[..., 1]) / 6.0
        )
    )


@runtime_checkable
class DEMSampler(Protocol):
    """Sample ellipsoidal height at geodetic coordinates."""

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return height in metres for each latitude/longitude sample."""


@dataclass(frozen=True, slots=True)
class ConstantHeightDEM:
    """DEM that returns a constant ellipsoidal height everywhere."""

    height_m: float = 0.0

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return a constant height array matching the coordinate shape."""
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lat_b, _ = np.broadcast_arrays(lat, lon)
        return np.full(lat_b.shape, self.height_m, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GeoidAdjustedDEM:
    """Convert orthometric DEM samples to ellipsoidal heights."""

    orthometric_dem: DEMSampler
    geoid: DEMSampler

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return orthometric height plus geoid undulation."""
        orthometric = self.orthometric_dem.sample(latitude_deg, longitude_deg)
        undulation = self.geoid.sample(latitude_deg, longitude_deg)
        return np.asarray(orthometric) + np.asarray(undulation)


# Canonical name used by new pipeline code. The implementation remains a
# sampler adapter until all geometry backends consume materialized RasterDEM.
DatumAdjustedDEM = GeoidAdjustedDEM


def admit_dem_device_identity(request: str) -> str:
    """Return a stored DEM device identity (`cpu` / `cuda` / `cuda:N`).

    ``cpu`` does not import Torch. Any other request, including ``auto``,
    goes through :func:`faninsar.processing.runtime.device.parse_device` after a lazy
    import.
    """
    stripped = str(request).strip()
    lowered = stripped.lower()
    if lowered == "cpu":
        return "cpu"
    from faninsar.processing.runtime.device import parse_device

    admitted = parse_device(None if lowered in {"", "auto", "gpu"} else stripped)
    return str(admitted)


def clone_raster_dem(
    dem: RasterDEM,
    *,
    path: Path | None = None,
    device: str | None = None,
) -> RasterDEM:
    """Copy path/nodata/interpolation/device/chunk fields onto a new sampler."""
    return RasterDEM(
        path if path is not None else dem.path,
        nodata=dem.nodata,
        interpolation=dem.interpolation,
        device=dem.device if device is None else device,
        sample_chunk_points=dem.sample_chunk_points,
    )


def pin_dem_sampler_device(dem: DEMSampler, identity: str) -> DEMSampler:
    """Bind a DEM sampler tree to an already-admitted device identity."""
    if isinstance(dem, GeoidAdjustedDEM):
        inner = pin_dem_sampler_device(dem.orthometric_dem, identity)
        if inner is dem.orthometric_dem:
            return dem
        return GeoidAdjustedDEM(inner, dem.geoid)
    if isinstance(dem, RasterDEM):
        if dem.device == identity:
            return dem
        return clone_raster_dem(dem, device=identity)
    return dem


@dataclass(slots=True)
class NetCDFGeoid:
    """Sample geoid undulation from a regular NetCDF longitude/latitude grid."""

    path: Path
    _interpolator: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Validate that the geoid grid exists."""
        if not self.path.exists():
            message = f"geoid grid does not exist: {self.path}"
            logger.error(message)
            raise FileNotFoundError(message)

    def _open(self) -> Callable[[np.ndarray], np.ndarray]:
        if self._interpolator is None:
            import xarray as xr
            from scipy.interpolate import RegularGridInterpolator

            grid = xr.open_dataarray(self.path)
            latitude_name = "lat" if "lat" in grid.coords else "y"
            longitude_name = "lon" if "lon" in grid.coords else "x"
            latitude = np.asarray(grid[latitude_name], dtype=np.float64)
            longitude = np.asarray(grid[longitude_name], dtype=np.float64)
            values = np.asarray(grid, dtype=np.float64)
            if latitude[0] > latitude[-1]:
                latitude = latitude[::-1]
                values = values[::-1]
            self._interpolator = RegularGridInterpolator(
                (latitude, longitude),
                values,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        return self._interpolator

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Bilinear-sample geoid undulation in metres."""
        latitude, longitude = np.broadcast_arrays(
            np.asarray(latitude_deg, dtype=np.float64),
            np.asarray(longitude_deg, dtype=np.float64),
        )
        interpolator = self._open()
        points = np.column_stack([latitude.ravel(), longitude.ravel()])
        values = interpolator(points)
        return np.asarray(values, dtype=np.float64).reshape(latitude.shape)


@dataclass(slots=True)
class RasterDEM:
    """Sample heights from a single-band georeferenced DEM raster."""

    path: Path
    nodata: float | None = None
    interpolation: Literal["bilinear", "bicubic", "biquintic"] = "biquintic"
    device: str = "auto"
    sample_chunk_points: int = 500_000
    _dataset: DatasetReader | None = None
    _height_array: np.ndarray | None = None
    _spline_coefficients: np.ndarray | None = None
    _height_tensor: object | None = None
    _height_tensor_identity: str | None = None

    def __post_init__(self) -> None:
        """Validate that the DEM path exists."""
        self.path = Path(self.path)
        if not self.path.exists():
            message = f"DEM path does not exist: {self.path}"
            logger.error(message)
            raise FileNotFoundError(message)
        if self.interpolation not in {"bilinear", "bicubic", "biquintic"}:
            message = f"unsupported DEM interpolation: {self.interpolation}"
            logger.error(message)
            raise ValueError(message)
        if int(self.sample_chunk_points) <= 0:
            message = "sample_chunk_points must be a positive integer"
            logger.error(message)
            raise ValueError(message)
        self.device = str(self.device)

    def __getstate__(self) -> dict[str, object]:
        """Return a picklable state without the open rasterio handle."""
        return {
            "path": self.path,
            "nodata": self.nodata,
            "interpolation": self.interpolation,
            "device": self.device,
            "sample_chunk_points": self.sample_chunk_points,
            "_height_array": self._height_array,
            "_spline_coefficients": self._spline_coefficients,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore the sampler state; the dataset reopens lazily."""
        self.path = Path(state["path"])
        self.nodata = state["nodata"]
        self.interpolation = state["interpolation"]
        stored = state.get("device")
        self.device = "cpu" if stored is None else str(stored)
        chunk = state.get("sample_chunk_points", 500_000)
        self.sample_chunk_points = int(chunk)
        self._height_array = state["_height_array"]
        self._spline_coefficients = state["_spline_coefficients"]
        self._height_tensor = None
        self._height_tensor_identity = None
        self._dataset = None

    def _resolved_identity(self) -> str:
        """Admit ``device`` once and store ``cpu`` / ``cuda`` / ``cuda:N``."""
        current = str(self.device).strip()
        if current.lower() == "cpu":
            return "cpu"
        admitted = admit_dem_device_identity(current)
        self.device = admitted
        return admitted

    def _open(self) -> DatasetReader:
        if self._dataset is None:
            import rasterio

            dataset = rasterio.open(self.path)
            self._dataset = dataset
            if self.nodata is None:
                self.nodata = dataset.nodata
        dataset = self._dataset
        if dataset is None:
            message = f"failed to open DEM dataset: {self.path}"
            logger.error(message)
            raise RuntimeError(message)
        return dataset

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Bilinear-sample DEM heights at lon/lat coordinates.

        DEM height is a real-valued smooth field, so bilinear interpolation
        is the appropriate kernel. The previous implementation used
        :meth:`rasterio.DatasetReader.sample`, which is nearest-neighbour —
        a doc/behaviour mismatch that also needlessly quantised the height
        field. This implementation converts the requested lon/lat to source
        pixel coordinates via the raster transform and performs an explicit
        bilinear interpolation, returning NaN for out-of-bounds samples.

        Parameters
        ----------
        latitude_deg, longitude_deg : numpy.ndarray
            Geodetic sample coordinates in degrees.

        Returns
        -------
        numpy.ndarray
            Sampled heights in metres. Out-of-bounds samples are NaN.

        """
        identity = self._resolved_identity()
        if identity != "cpu" and not identity.startswith("cuda"):
            message = f"DEM sampling has no identity on {identity}"
            logger.error(message)
            raise ValueError(message)
        if identity != "cpu" and self.interpolation != "biquintic":
            message = (
                "DEM interpolation "
                f"{self.interpolation!r} has no identity on {identity}"
            )
            logger.error(message)
            raise ValueError(message)
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        dataset = self._open()
        nodata = self.nodata
        if nodata is None:
            nodata = dataset.nodata

        if self._height_array is None:
            height_array = dataset.read(1).astype(np.float32, copy=False)
            if nodata is not None:
                height_array = np.where(
                    np.isclose(height_array, nodata),
                    np.nan,
                    height_array,
                ).astype(np.float32, copy=False)
            self._height_array = height_array
        h_arr = self._height_array

        transform = dataset.transform
        inv = ~transform
        # lon/lat -> CRS pixel coordinates. We assume the DEM is already in a
        # CRS matching the requested lon/lat (geographic). If the dataset is
        # projected, callers should reproject their coordinates first.
        xs = lon_b.ravel()
        ys = lat_b.ravel()
        # transform.invert expects (col, row) order via (x, y) -> (col, row).
        # Use the pixel-area convention (integer index = the containing
        # pixel's lower edge), matching ISCE2's DEM interpolation.  A half
        # pixel centre shift here produces ~5 m height differences in steep
        # terrain, which propagates into the flatten phase.
        cols, rows = inv * (xs, ys)
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)

        if self.interpolation == "biquintic":
            if identity != "cpu":
                return self._sample_biquintic_cuda(
                    rows, cols, h_arr, lat_b.shape, identity
                )
            row_floor = np.floor(rows).astype(np.int64)
            col_floor = np.floor(cols).astype(np.int64)
            height, width = h_arr.shape
            in_bounds = (
                (row_floor >= 1)
                & (row_floor <= height - 5)
                & (col_floor >= 1)
                & (col_floor <= width - 5)
            )
            heights = np.full(rows.shape, np.nan, dtype=np.float64)
            if in_bounds.any():
                selected = np.flatnonzero(in_bounds)
                row_base = row_floor[selected]
                col_base = col_floor[selected]
                neighbours = np.arange(-1, 5, dtype=np.int64)
                row_indices = row_base[:, None] + neighbours[None, :]
                col_indices = col_base[:, None] + neighbours[None, :]
                windows = h_arr[
                    row_indices[:, :, None],
                    col_indices[:, None, :],
                ]
                along_columns = _natural_spline_six(
                    windows,
                    (cols[selected] - col_base)[:, None],
                )
                heights[selected] = _natural_spline_six(
                    along_columns,
                    rows[selected] - row_base,
                )
            return heights.reshape(lat_b.shape)

        if self.interpolation == "bicubic":
            from scipy.ndimage import map_coordinates, spline_filter

            if self._spline_coefficients is None:
                coefficients = np.empty_like(h_arr, dtype=np.float32)
                spline_filter(h_arr, order=3, output=coefficients)
                self._spline_coefficients = coefficients
            coordinates = np.vstack([rows, cols])
            heights = map_coordinates(
                self._spline_coefficients,
                coordinates,
                order=3,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            return np.asarray(heights, dtype=np.float64).reshape(lat_b.shape)

        r0 = np.floor(rows).astype(np.int64)
        c0 = np.floor(cols).astype(np.int64)
        dr = rows - r0
        dc = cols - c0
        h, w = h_arr.shape
        in_bounds = (r0 >= 0) & (r0 <= h - 2) & (c0 >= 0) & (c0 <= w - 2)

        heights = np.full(xs.shape, np.nan, dtype=np.float64)
        if in_bounds.any():
            idx = np.where(in_bounds)[0]
            r0i = r0[idx]
            c0i = c0[idx]
            dri = dr[idx]
            dci = dc[idx]
            h00 = h_arr[r0i, c0i]
            h01 = h_arr[r0i, c0i + 1]
            h10 = h_arr[r0i + 1, c0i]
            h11 = h_arr[r0i + 1, c0i + 1]
            top = h00 * (1.0 - dci) + h01 * dci
            bot = h10 * (1.0 - dci) + h11 * dci
            vals = top * (1.0 - dri) + bot * dri
            heights[idx] = vals
        return heights.reshape(lat_b.shape)

    def _sample_biquintic_cuda(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height_array: np.ndarray,
        out_shape: tuple[int, ...],
        identity: str,
    ) -> np.ndarray:
        """Evaluate the six-sample spline on an admitted CUDA device."""
        import torch

        from faninsar.processing.geometry.torch_kernels import (
            _natural_spline_six as torch_spline,
        )

        torch_device = torch.device(identity)
        if self._height_tensor is None or self._height_tensor_identity != identity:
            self._height_tensor = torch.as_tensor(height_array, device=torch_device)
            self._height_tensor_identity = identity
        dem_tensor = self._height_tensor
        n_points = int(rows.size)
        heights_t = torch.full(
            (n_points,),
            float("nan"),
            dtype=torch.float64,
            device=torch_device,
        )
        chunk = int(self.sample_chunk_points)
        neighbours = torch.arange(-1, 5, dtype=torch.int64, device=torch_device)
        raster_h, raster_w = height_array.shape
        for start in range(0, n_points, chunk):
            stop = min(start + chunk, n_points)
            row_t = torch.as_tensor(
                rows[start:stop], dtype=torch.float64, device=torch_device
            )
            col_t = torch.as_tensor(
                cols[start:stop], dtype=torch.float64, device=torch_device
            )
            row_floor = torch.floor(row_t).to(torch.int64)
            col_floor = torch.floor(col_t).to(torch.int64)
            in_bounds = (
                (row_floor >= 1)
                & (row_floor <= raster_h - 5)
                & (col_floor >= 1)
                & (col_floor <= raster_w - 5)
            )
            if not bool(in_bounds.any()):
                continue
            selected = torch.nonzero(in_bounds, as_tuple=False).squeeze(-1)
            row_base = row_floor[selected]
            col_base = col_floor[selected]
            row_indices = row_base[:, None] + neighbours[None, :]
            col_indices = col_base[:, None] + neighbours[None, :]
            windows = dem_tensor[row_indices[:, :, None], col_indices[:, None, :]].to(
                torch.float64
            )
            along_columns = torch_spline(
                windows, (col_t[selected] - col_base.to(torch.float64))[:, None]
            )
            sampled = torch_spline(
                along_columns, row_t[selected] - row_base.to(torch.float64)
            )
            heights_t[start + selected] = sampled
        return heights_t.detach().cpu().numpy().reshape(out_shape)

    def close(self) -> None:
        """Close the underlying raster dataset if open."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
        self._height_array = None
        self._spline_coefficients = None
        self._height_tensor = None
        self._height_tensor_identity = None

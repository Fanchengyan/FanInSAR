"""Topographic phase computation and removal for interferograms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry import RadarGeometryModel, rdr2geo_with_dem
from faninsar.processing.geometry.dem import RasterDEM
from faninsar.processing.geometry.ellipsoid import llh_to_ecef

if TYPE_CHECKING:
    from faninsar.processing.geometry.dem import DEMSampler

logger = setup_logger(__name__)


def compute_topographic_phase(
    model_ref: RadarGeometryModel,
    model_sec: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: DEMSampler,
    *,
    wavelength_m: float | None = None,
) -> np.ndarray:
    r"""Compute topographic phase for a coregistered pair from geometry and DEM.

    Uses the perpendicular baseline formulation derived from path-length
    geometry:

    .. math::
        \phi_{\text{topo}} = -\frac{4\pi}{\lambda}
        \frac{B_{\perp}}{R \sin\theta} h

    where :math:`B_{\perp}` is the signed perpendicular baseline,
    :math:`R` is the slant range, :math:`\theta` is the incidence angle
    measured from the local horizontal, :math:`\lambda` is the wavelength,
    and :math:`h` is the DEM height.

    Parameters
    ----------
    model_ref : RadarGeometryModel
        Reference acquisition geometry model.
    model_sec : RadarGeometryModel
        Secondary acquisition geometry model.
    azimuth_index, range_index : numpy.ndarray
        Radar sample coordinates on the coregistered grid.
    dem : DEMSampler
        DEM height sampler.
    wavelength_m : float, optional
        Radar wavelength in metres. If None, uses ``model_ref.wavelength_m``.

    Returns
    -------
    numpy.ndarray
        Topographic phase array in radians. Non-converged pixels are NaN.

    """
    if wavelength_m is None:
        wavelength_m = model_ref.wavelength_m
    if wavelength_m <= 0:
        message = "wavelength must be positive"
        logger.error(message)
        raise ValueError(message)

    # Map radar indices to geodetic coordinates with DEM heights (reference geometry)
    geo = rdr2geo_with_dem(
        model_ref,
        azimuth_index,
        range_index,
        dem=dem,
    )

    phase = np.full(geo.latitude_deg.shape, np.nan, dtype=np.float64)
    if not np.any(geo.converged):
        return phase

    # Flatten converged indices for vectorized orbit evaluation
    conv_mask = geo.converged
    az_b, _ = np.broadcast_arrays(
        np.asarray(azimuth_index, dtype=np.float64),
        np.asarray(range_index, dtype=np.float64),
    )
    az_conv = az_b[conv_mask]

    # Compute azimuth times in seconds from orbit epoch
    sensing_offset_s = (model_ref.sensing_start - model_ref.orbit.epoch).total_seconds()
    times_s = sensing_offset_s + az_conv * model_ref.azimuth_time_interval_s

    # Evaluate orbits for converged pixels
    sat_ref, _vel_ref = model_ref.orbit.evaluate_array(times_s)
    sat_sec, _vel_sec = model_sec.orbit.evaluate_array(times_s)

    # Target ECEF positions for converged pixels
    lat_conv = geo.latitude_deg[conv_mask]
    lon_conv = geo.longitude_deg[conv_mask]
    h_conv = geo.height_m[conv_mask]
    tx, ty, tz = llh_to_ecef(lat_conv, lon_conv, h_conv)
    target = np.stack([tx, ty, tz], axis=-1)

    # Path-length geometric phase (flat-Earth + topography when DEM height ≠ 0):
    #   φ = -4π/λ * (R_sec - R_ref)
    # Secondary orbit is evaluated at the same absolute times as a first-order
    # approximation; residual Doppler timing is sub-sample for S1 IW baselines.
    look_ref = target - sat_ref
    look_sec = target - sat_sec
    range_ref = np.linalg.norm(look_ref, axis=-1)
    range_sec = np.linalg.norm(look_sec, axis=-1)
    valid = (range_ref > 1e-6) & (range_sec > 1e-6)

    phase_conv = np.full(lat_conv.shape, np.nan, dtype=np.float64)
    phase_conv[valid] = (
        -(4.0 * np.pi / wavelength_m) * (range_sec[valid] - range_ref[valid])
    )

    phase[conv_mask] = phase_conv
    return phase


# Path-length geometric phase (orbital + topographic).
compute_geometric_phase = compute_topographic_phase


def remove_topographic_phase(
    complex_ifg: np.ndarray,
    topo_phase: np.ndarray,
) -> np.ndarray:
    """Remove topographic phase from a complex interferogram.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram array.
    topo_phase : numpy.ndarray
        Topographic phase in radians, broadcastable to ``complex_ifg``.

    Returns
    -------
    numpy.ndarray
        Flattened complex interferogram.

    """
    ifg = np.asarray(complex_ifg)
    topo = np.asarray(topo_phase)
    if ifg.shape != topo.shape:
        try:
            np.broadcast_to(topo, ifg.shape)
        except ValueError as exc:
            message = (
                f"topo_phase shape {topo.shape} cannot broadcast to "
                f"ifg shape {ifg.shape}"
            )
            logger.exception(message)
            raise ValueError(message) from exc
    return ifg * np.exp(-1j * topo)


def copernicus_glo30_dem(
    latitude_deg: float,
    longitude_deg: float,
    *,
    base_path: str | Path = (
        "/Volumes/DATA2/TEST_sentinel-1/auxiliary/aws-sinergise/GLO-30-COG"
    ),
) -> RasterDEM:
    """Return a :class:`RasterDEM` for the Copernicus GLO-30 tile covering a coordinate.

    Expects the standard COG tile naming convention:
    ``Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_DEM.tif``.

    Parameters
    ----------
    latitude_deg, longitude_deg : float
        Geodetic coordinate in degrees.
    base_path : str or Path, optional
        Directory containing Copernicus GLO-30 COG tiles.

    Returns
    -------
    RasterDEM
        DEM sampler for the tile.

    Raises
    ------
    FileNotFoundError
        If the expected tile file does not exist.

    """
    base = Path(base_path)
    lat_tile = int(np.floor(latitude_deg))
    lon_tile = int(np.floor(longitude_deg))
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    lat_abs = abs(lat_tile)
    lon_abs = abs(lon_tile)
    filename = (
        f"Copernicus_DSM_COG_10_{ns}{lat_abs:02d}_00_"
        f"{ew}{lon_abs:03d}_00_DEM.tif"
    )
    tile_dir = f"{ns}{lat_abs:02d}_{ew}{lon_abs:03d}"
    candidates = [
        base / filename,
        base / tile_dir / filename,
        *sorted(base.glob(f"*/{tile_dir}/{filename}")),
        *sorted(base.glob(f"**/{filename}")),
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        message = f"Copernicus GLO-30 tile not found for {filename} under {base}"
        logger.error(message)
        raise FileNotFoundError(message)
    return RasterDEM(path=path)

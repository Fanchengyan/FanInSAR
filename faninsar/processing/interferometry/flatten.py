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
    secondary_azimuth_index: np.ndarray | None = None,
    wavelength_m: float | None = None,
) -> np.ndarray:
    r"""Compute topographic phase for a coregistered pair from geometry and DEM.

    Uses the two-way path-length geometric phase consistent with
    interferogram formation ``primary * conj(secondary)``:

    .. math::
        \phi_{\text{geo}} = +\frac{4\pi}{\lambda}
        \left(R_{\text{sec}} - R_{\text{ref}}\right)

    where :math:`R` is the one-way slant range from the sensor to the DEM
    target and :math:`\lambda` is the wavelength.  This matches the radar
    phase convention :math:`\varphi = -4\pi R / \lambda` so that
    :math:`\varphi_{\text{ref}} - \varphi_{\text{sec}} =
    +4\pi (R_{\text{sec}} - R_{\text{ref}}) / \lambda`.

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
    secondary_azimuth_index : numpy.ndarray, optional
        Secondary zero-Doppler azimuth coordinates for the same ground targets.
        Direct-remap workflows should pass the fractional source coordinates
        already computed during coregistration. If omitted, the reference
        azimuth coordinates are used for both orbit evaluations.
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

    conv_mask = (
        geo.converged
        & np.isfinite(geo.latitude_deg)
        & np.isfinite(geo.longitude_deg)
        & np.isfinite(geo.height_m)
    )
    az_b, _range_b = np.broadcast_arrays(
        np.asarray(azimuth_index, dtype=np.float64),
        np.asarray(range_index, dtype=np.float64),
    )
    if secondary_azimuth_index is None:
        secondary_az_b = az_b
    else:
        secondary_az_b = np.broadcast_to(
            np.asarray(secondary_azimuth_index, dtype=np.float64),
            az_b.shape,
        )
        conv_mask &= np.isfinite(secondary_az_b)
    if not np.any(conv_mask):
        return phase

    reference_azimuth = az_b[conv_mask]
    secondary_azimuth = secondary_az_b[conv_mask]

    reference_times_s = model_ref.azimuth_time_seconds(reference_azimuth)
    secondary_times_s = model_sec.azimuth_time_seconds(secondary_azimuth)

    sat_ref, _vel_ref = model_ref.orbit.evaluate_array(reference_times_s)
    sat_sec, _vel_sec = model_sec.orbit.evaluate_array(secondary_times_s)

    # Target ECEF positions for converged pixels
    lat_conv = geo.latitude_deg[conv_mask]
    lon_conv = geo.longitude_deg[conv_mask]
    h_conv = geo.height_m[conv_mask]
    tx, ty, tz = llh_to_ecef(lat_conv, lon_conv, h_conv)
    target = np.stack([tx, ty, tz], axis=-1)

    # Each orbit is evaluated at its own zero-Doppler time for the shared
    # ground target. Reusing reference time for the secondary creates a
    # baseline-dependent path error that aliases into topographic residual.
    look_ref = target - sat_ref
    look_sec = target - sat_sec
    range_ref = np.linalg.norm(look_ref, axis=-1)
    range_sec = np.linalg.norm(look_sec, axis=-1)
    valid = (range_ref > 1e-6) & (range_sec > 1e-6)

    phase_conv = np.full(lat_conv.shape, np.nan, dtype=np.float64)
    phase_conv[valid] = +(4.0 * np.pi / wavelength_m) * (
        range_sec[valid] - range_ref[valid]
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


def estimate_residual_azimuth_ramp(
    complex_ifg: np.ndarray,
    reference_phase: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.2,
    search: tuple[float, float] = (-0.5, 0.5),
    n_grid: int = 201,
) -> float:
    r"""Estimate residual linear azimuth phase ramp (rad per azimuth sample).

    Residual Doppler / TOPS differential carrier often leaves a nearly linear
    phase screen ``c \\cdot i_{az}`` on top of the geometric interferogram.
    The coefficient ``c`` is chosen to maximise the circular correlation of
    the interferogram against a reference phase model (typically the
    path-length geometric phase from orbits + DEM):

    .. math::

        \\hat{c} = \\arg\\max_c
        \\left| \\sum_p w_p \\,
        e^{i\\bigl(\\phi_{\\mathrm{ifg}}(p)
        - \\phi_{\\mathrm{ref}}(p) - c\\, i_{az}\\bigr)} \\right|

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.
    reference_phase : numpy.ndarray
        Reference phase in radians (e.g. geometric/topo phase), same shape.
    coherence : numpy.ndarray, optional
        Weights; pixels with coherence below ``coh_thr`` are ignored when
        provided.
    coh_thr : float, optional
        Coherence threshold. Default 0.2.
    search : tuple of float, optional
        Inclusive range of ``c`` to search (rad / azimuth sample).
    n_grid : int, optional
        Number of grid points in the search. Default 201.

    Returns
    -------
    float
        Estimated ramp coefficient ``c`` in rad per azimuth sample.

    """
    ifg = np.asarray(complex_ifg)
    ref = np.asarray(reference_phase, dtype=np.float64)
    if ifg.shape != ref.shape:
        message = f"reference_phase shape {ref.shape} must match ifg shape {ifg.shape}"
        logger.error(message)
        raise ValueError(message)
    ph = np.angle(ifg)
    mask = np.isfinite(ph) & np.isfinite(ref) & (np.abs(ifg) > 0)
    if coherence is not None:
        coh = np.asarray(coherence)
        mask = mask & np.isfinite(coh) & (coh >= coh_thr)
        weights = np.where(mask, coh, 0.0)
    else:
        weights = mask.astype(np.float64)
    if not np.any(mask):
        return 0.0

    height = ph.shape[0]
    az = np.arange(height, dtype=np.float64)[:, None]
    residual = ph - ref
    best_c = 0.0
    best_score = -1.0
    for c in np.linspace(search[0], search[1], int(n_grid)):
        score = float(np.abs(np.nansum(weights * np.exp(1j * (residual - c * az)))))
        if score > best_score:
            best_score = score
            best_c = float(c)
    return best_c


def remove_azimuth_phase_ramp(
    complex_ifg: np.ndarray,
    ramp_rad_per_az: float,
) -> np.ndarray:
    """Multiply interferogram by ``exp(-i * c * i_az)`` to remove a linear ramp.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.
    ramp_rad_per_az : float
        Phase ramp coefficient in rad per azimuth sample.

    Returns
    -------
    numpy.ndarray
        Ramp-corrected complex interferogram.

    """
    ifg = np.asarray(complex_ifg)
    if abs(ramp_rad_per_az) < 1e-15:
        return ifg.astype(np.complex64, copy=False)
    az = np.arange(ifg.shape[0], dtype=np.float64)[:, None]
    return (ifg * np.exp(-1j * ramp_rad_per_az * az)).astype(ifg.dtype, copy=False)


def copernicus_glo30_dem(
    latitude_deg: float,
    longitude_deg: float,
    *,
    base_path: str | Path,
) -> RasterDEM:
    """Return a :class:`RasterDEM` for the Copernicus GLO-30 tile covering a coordinate.

    Expects the standard COG tile naming convention:
    ``Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_DEM.tif``.

    Parameters
    ----------
    latitude_deg, longitude_deg : float
        Geodetic coordinate in degrees.
    base_path : str or Path
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
        f"Copernicus_DSM_COG_10_{ns}{lat_abs:02d}_00_{ew}{lon_abs:03d}_00_DEM.tif"
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

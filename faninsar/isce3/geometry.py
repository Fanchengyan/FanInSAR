"""
SAR coordinate transformations using ISCE3 geometry.

Provides functions for converting between geographic (lat/lon/height)  
and radar (azimuth/range) coordinates using ISCE3's geometry module.
"""

from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import logging

from .metadata import SARMetadata

try:
    import isce3
    from isce3.core import Orbit, Ellipsoid, LUT2d
    from isce3.product import RadarGridParameters
    from isce3.geometry import geo2rdr, rdr2geo
except ImportError:
    raise ImportError("ISCE3 is required")

logger = logging.getLogger(__name__)


def geo_to_radar(
    metadata: SARMetadata,
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
    height: Union[float, np.ndarray],
    *,
    threshold: float = 1.0e-8,
    max_iter: int = 50,
    delta_range: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert geographic coordinates to radar coordinates.
    
    Wraps ISCE3's geo2rdr function to map latitude, longitude, and height
    to azimuth time and slant range.
    
    Parameters
    ----------
    metadata : SARMetadata
        SAR metadata containing orbit and radar grid
    lat : float | np.ndarray
        Latitude in degrees (WGS84)
    lon : float | np.ndarray  
        Longitude in degrees (WGS84)
    height : float | np.ndarray
        Height above ellipsoid in meters
    threshold : float, optional
        Convergence threshold in meters, by default 1.0e-8
    max_iter : int, optional
        Maximum iterations, by default 50
    delta_range : float, optional
        Step size for numerical gradient in meters, by default 10.0
        
    Returns
    -------
    azimuth_time : np.ndarray
        Azimuth time in seconds since radar grid start
    slant_range : np.ndarray
        Slant range in meters
        
    Raises
    ------
    ValueError
        If metadata is not properly initialized
        
    Examples
    --------
    >>> from faninsar.isce3.utils import S1Metadata
    >>> meta = S1Metadata('burst_id', '/path/to/safe')
    >>> az_time, slant_range = geo_to_radar(meta, 35.0, -120.0, 100.0)
    """
    if metadata.orbit is None or metadata.radar_grid is None:
        raise ValueError("Metadata must have orbit and radar_grid initialized")
    
    # Convert to arrays
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    height = np.atleast_1d(np.asarray(height, dtype=np.float64))
    
    # Broadcast to same shape
    lat, lon, height = np.broadcast_arrays(lat, lon, height)
    
    ellipsoid = Ellipsoid()
    doppler = metadata.doppler if metadata.doppler is not None else LUT2d()
    
    # Pre-allocate output arrays
    azimuth_times = np.zeros_like(lat)
    slant_ranges = np.zeros_like(lat)
    
    # Call geo2rdr for each point
    for idx in np.ndindex(lat.shape):
        try:
            az_time, sr = geo2rdr(
                lon[idx],
                lat[idx],
                height[idx],
                ellipsoid,
                metadata.orbit,
                doppler,
                metadata.radar_grid.wavelength,
                metadata.radar_grid.lookside,
                threshold=threshold,
                maxiter=max_iter,
                delta_range=delta_range
            )
            
            azimuth_times[idx] = az_time
            slant_ranges[idx] = sr
            
        except Exception as e:
            logger.warning(f"geo2rdr failed at ({lat[idx]}, {lon[idx]}, {height[idx]}): {e}")
            azimuth_times[idx] = np.nan
            slant_ranges[idx] = np.nan
    
    return azimuth_times, slant_ranges


def radar_to_geo(
    metadata: SARMetadata,
    azimuth_time: Union[float, np.ndarray],
    slant_range: Union[float, np.ndarray],
    *,
    side: str = 'right',
    threshold: float = 1.0e-8,
    max_iter: int = 50,
    extraiter: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert radar coordinates to geographic coordinates.
    
    Wraps ISCE3's rdr2geo function to map azimuth time and slant range  
    to latitude, longitude, and height.
    
    Parameters
    ----------
    metadata : SARMetadata
        SAR metadata containing orbit and radar grid
    azimuth_time : float | np.ndarray
        Azimuth time in seconds since radar grid start
    slant_range : float | np.ndarray
        Slant range in meters
    side : str, optional
        Look side ('right' or 'left'), by default 'right'
    threshold : float, optional
        Convergence threshold in meters, by default 1.0e-8
    max_iter : int, optional
        Maximum iterations, by default 50
    extraiter : int, optional
        Extra iterations for refinement, by default 10
        
    Returns
    -------
    lat : np.ndarray
        Latitude in degrees (WGS84)
    lon : np.ndarray
        Longitude in degrees (WGS84)
    height : np.ndarray
        Height above ellipsoid in meters
        
    Examples
    --------
    >>> lat, lon, height = radar_to_geo(meta, 1.5, 850000.0)
    """
    if metadata.orbit is None or metadata.radar_grid is None:
        raise ValueError("Metadata must have orbit and radar_grid initialized")
    
    # Convert to arrays
    azimuth_time = np.atleast_1d(np.asarray(azimuth_time, dtype=np.float64))
    slant_range = np.atleast_1d(np.asarray(slant_range, dtype=np.float64))
    
    # Broadcast to same shape
    azimuth_time, slant_range = np.broadcast_arrays(azimuth_time, slant_range)
    
    ellipsoid = Ellipsoid()
    doppler = metadata.doppler if metadata.doppler is not None else LUT2d()
    
    # Pre-allocate output arrays
    lats = np.zeros_like(azimuth_time)
    lons = np.zeros_like(azimuth_time)
    heights = np.zeros_like(azimuth_time)
    
    # Determine look side
    lookside = isce3.core.LookSide.Right if side.lower() == 'right' else isce3.core.LookSide.Left
    
    # Call rdr2geo for each point
    for idx in np.ndindex(azimuth_time.shape):
        try:
            llh = rdr2geo(
                azimuth_time[idx],
                slant_range[idx],
                metadata.orbit,
                lookside,
                doppler,
                metadata.radar_grid.wavelength,
                ellipsoid,
                threshold=threshold,
                maxiter=max_iter,
                extraiter=extraiter
            )
            
            lats[idx] = llh[0]
            lons[idx] = llh[1]
            heights[idx] = llh[2]
            
        except Exception as e:
            logger.warning(f"rdr2geo failed at (az={azimuth_time[idx]}, range={slant_range[idx]}): {e}")
            lats[idx] = np.nan
            lons[idx] = np.nan
            heights[idx] = np.nan
    
    return lats, lons, heights


def compute_baseline(
    metadata_ref: SARMetadata,
    metadata_sec: SARMetadata,
    *,
    num_samples: int = 100
) -> Tuple[float, float]:
    """
    Compute perpendicular and parallel baselines between two acquisitions.
    
    Parameters
    ----------
    metadata_ref : SARMetadata
        Reference acquisition metadata
    metadata_sec : SARMetadata
        Secondary acquisition metadata  
    num_samples : int, optional
        Number of samples for baseline estimation, by default 100
        
    Returns
    -------
    baseline_perp : float
        Perpendicular baseline in meters (mean)
    baseline_para : float
        Parallel baseline in meters (mean)
        
    Examples
    --------
    >>> b_perp, b_para = compute_baseline(meta_ref, meta_sec)
    >>> print(f"Perpendicular baseline: {b_perp:.2f} m")
    """
    if metadata_ref.orbit is None or metadata_sec.orbit is None:
        raise ValueError("Both metadata objects must have orbits initialized")
    
    # Use ISCE3's Baseline class
    baseline = isce3.core.Baseline(metadata_ref.orbit, metadata_sec.orbit)
    
    # Sample along the orbit
    perp_baselines = []
    para_baselines = []
    
    sensing_start = metadata_ref.sensing_start
    duration = metadata_ref.radar_grid.length / metadata_ref.prf
    
    for i in range(num_samples):
        t = sensing_start + (i / num_samples) * duration
        
        try:
            b_perp, b_para = baseline.get_baseline(t, metadata_ref.radar_grid.starting_range)
            perp_baselines.append(b_perp)
            para_baselines.append(b_para)
        except:
            continue
    
    if not perp_baselines:
        logger.warning("Failed to compute any baseline samples")
        return 0.0, 0.0
    
    return float(np.mean(perp_baselines)), float(np.mean(para_baselines))

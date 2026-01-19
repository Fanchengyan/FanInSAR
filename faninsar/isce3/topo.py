"""
Topographic phase calculation and removal.

Provides functions for computing the topographic phase component
in radar interferometry, including flat-earth and height-induced phases.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import xarray as xr
import logging

from .metadata import SARMetadata

try:
    import isce3
    from isce3.core import Ellipsoid
except ImportError:
    raise ImportError("ISCE3 is required")

logger = logging.getLogger(__name__)


def compute_flat_earth_phase(
    metadata_ref: SARMetadata,
    metadata_sec: SARMetadata,
    *,
    use_isce3: bool = True
) -> xr.DataArray:
    """
    Compute flat-earth phase for interferometry.
    
    The flat-earth phase is the interferometric phase that would be observed
    over a perfectly flat surface at the reference ellipsoid.
    
    Parameters
    ----------
    metadata_ref : SARMetadata
        Reference acquisition metadata
    metadata_sec : SARMetadata
        Secondary acquisition metadata
    use_isce3 : bool, optional
        Use ISCE3's built-in flat-earth calculation, by default True
        
    Returns
    -------
    xr.DataArray
        Flat-earth phase in radians
        
    Notes
    -----
    The flat-earth phase is primarily a function of the baseline and
    viewing geometry. Removing it is essential for meaningful interferometric
    height measurements.
    
    Examples
    --------
    >>> phase_flat = compute_flat_earth_phase(meta_ref, meta_sec)
    >>> phase_corrected = phase_ifg - phase_flat
    """
    if metadata_ref.radar_grid is None or metadata_sec.radar_grid is None:
        raise ValueError("Both metadata objects must have radar_grid initialized")
    
    # Get radar grid dimensions
    length = metadata_ref.radar_grid.length
    width = metadata_ref.radar_grid.width
    
    # Compute flat-earth phase
    # This is a simplified implementation - full version would use ISCE3's functions
    logger.warning("Using simplified flat-earth phase - ISCE3 integration pending")
    
    # Create range and azimuth coordinates
    ranges = np.arange(width) * metadata_ref.radar_grid.range_pixel_spacing + \
             metadata_ref.radar_grid.starting_range
    
    azimuth = np.arange(length)
    
    # Placeholder: flat-earth phase depends on baseline geometry
    # Real implementation would compute this properly from orbit geometry
    phase = np.zeros((length, width), dtype=np.float32)
    
    result = xr.DataArray(
        phase,
        dims=['azimuth', 'range'],
        coords={
            'azimuth': azimuth,
            'range': ranges
        },
        attrs={
            'description': 'Flat-earth phase',
            'units': 'radians'
        }
    )
    
    return result


def compute_topo_phase(
    dem: xr.DataArray,
    metadata_ref: SARMetadata,
    metadata_sec: SARMetadata,
    *,
    ellipsoid: Optional[Ellipsoid] = None
) -> xr.DataArray:
    """
    Compute topographic phase from DEM.
    
    The topographic phase is the interferometric phase induced by terrain height
    variations, given the baseline and viewing geometry.
    
    Parameters
    ----------
    dem : xr.DataArray
        Digital elevation model in radar coordinates
    metadata_ref : SARMetadata
        Reference acquisition metadata
    metadata_sec : SARMetadata
        Secondary acquisition metadata
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid, by default WGS84
        
    Returns
    -------
    xr.DataArray
        Topographic phase in radians
        
    Notes
    -----
    The topographic phase is computed as:
    φ_topo = (4π/λ) * (B_perp / (r * sin(θ))) * h
    
    where:
    - λ is wavelength
    - B_perp is perpendicular baseline
    - r is slant range
    - θ is incidence angle
    - h is topographic height
    
    Examples
    --------
    >>> phase_topo = compute_topo_phase(dem_radar, meta_ref, meta_sec)
    """
    if ellipsoid is None:
        ellipsoid = Ellipsoid()
    
    # Get baseline
    from .geometry import compute_baseline
    b_perp, b_para = compute_baseline(metadata_ref, metadata_sec)
    
    logger.info(f"Computing topo phase with B_perp = {b_perp:.2f} m")
    
    # Get wavelength
    wavelength = metadata_ref.wavelength
    
    # Get slant range
    radar_grid = metadata_ref.radar_grid
    width = radar_grid.width
    length = radar_grid.length
    
    ranges = np.arange(width) * radar_grid.range_pixel_spacing + radar_grid.starting_range
    
    # Compute incidence angle (simplified - should vary with azimuth)
    # θ ≈ acos(h_satellite / r)
    # For now, use approximation
    earth_radius = ellipsoid.a  # Semi-major axis
    satellite_height = 700000.0  # ~700 km for Sentinel-1
    
    incidence_angles = np.arccos(earth_radius / (earth_radius + satellite_height))
    sin_theta = np.sin(incidence_angles)
    
    # Compute phase sensitivity (radians per meter of height)
    # κ = (4π/λ) * (B_perp / (r * sin(θ)))
    kappa = (4 * np.pi / wavelength) * (b_perp / (ranges * sin_theta))
    
    # Broadcast DEM heights and compute phase
    height_grid = dem.values
    kappa_grid = np.broadcast_to(kappa[np.newaxis, :], height_grid.shape)
    
    topo_phase = kappa_grid * height_grid
    
    result = xr.DataArray(
        topo_phase,
        dims=dem.dims,
        coords=dem.coords,
        attrs={
            'description': 'Topographic phase',
            'units': 'radians',
            'perpendicular_baseline': b_perp,
            'wavelength': wavelength
        }
    )
    
    return result


def remove_topo_phase(
    slc: xr.DataArray,
    topo_phase: xr.DataArray
) -> xr.DataArray:
    """
    Remove topographic phase from SLC data.
    
    Parameters
    ----------
    slc : xr.DataArray
        Complex SLC data
    topo_phase : xr.DataArray
        Topographic phase in radians
        
    Returns
    -------
    xr.DataArray
        SLC with topographic phase removed
        
    Examples
    --------
    >>> slc_flat = remove_topo_phase(slc, phase_topo)
    """
    # Multiply by conjugate of phase (equivalent to subtracting phase)
    return slc * np.exp(-1j * topo_phase)


def compute_flat_earth_topo_phase(
    dem:Optional[xr.DataArray],
    metadata_ref: SARMetadata,
    metadata_sec: SARMetadata
) -> xr.DataArray:
    """
    Compute combined flat-earth and topographic phase.
    
    This is a convenience function that computes both components together,
    which is the total geometric phase to remove for differential interferometry.
    
    Parameters
    ----------
    dem : xr.DataArray | None
        DEM in radar coordinates, or None to compute only flat-earth
    metadata_ref : SARMetadata
        Reference metadata
    metadata_sec : SARMetadata
        Secondary metadata
        
    Returns
    -------
    xr.DataArray
        Combined phase in radians
        
    Examples
    --------
    >>> phase_geom = compute_flat_earth_topo_phase(dem, meta_ref, meta_sec)
    >>> slc_corrected = slc * np.exp(-1j * phase_geom)
    """
    # Compute flat-earth
    phase_flat = compute_flat_earth_phase(metadata_ref, metadata_sec)
    
    if dem is None:
        return phase_flat
    
    # Compute topo
    phase_topo = compute_topo_phase(dem, metadata_ref, metadata_sec)
    
    # Combine
    return phase_flat + phase_topo

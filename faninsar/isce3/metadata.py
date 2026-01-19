"""
Generic SAR metadata handling using ISCE3 native objects.

This module provides base classes for SAR metadata management using
ISCE3's RadarGridParameters and Orbit classes for robust radar geometry handling.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from typing_extensions import Self
import numpy as np
import xarray as xr

try:
    import isce3
    from isce3.core import Orbit, LUT2d, Ellipsoid
    from isce3.product import RadarGridParameters
except ImportError as e:
    raise ImportError(
        "ISCE3 is required for faninsar.isce3. "
        "Please install it via conda: conda install -c conda-forge isce3"
    ) from e


class SARMetadata:
    """
    Generic SAR metadata container using ISCE3 native objects.
    
    This base class encapsulates radar grid parameters, orbit information,
    and Doppler centroid for any SAR satellite. Satellite-specific implementations
    should inherit from this class.
    
    Parameters
    ----------
    burst_id : str
        Unique identifier for this burst/scene
    data_dir : str
        Path to the SAR data directory
    satellite : str, optional
        Satellite identifier (e.g., 'S1', 'ALOS2', 'RS2'), by default 'S1'
        
    Attributes
    ----------
    burst_id : str
        Unique burst identifier
    satellite : str
        Satellite name
    orbit : isce3.core.Orbit | None
        Satellite orbit state vectors
    radar_grid : isce3.product.RadarGridParameters | None
        Radar grid parameters (geometry, timing, etc.)
    doppler : isce3.core.LUT2d | None
        Doppler centroid as 2D lookup table
    satellite_params : dict
        Satellite-specific parameters (e.g., TOPS FM rates for S1)
        
    Examples
    --------
    >>> # Use satellite-specific subclass in practice
    >>> from faninsar.isce3.utils.S1_safe_parser import S1Metadata
    >>> meta = S1Metadata(burst_id='s1a-iw1-slc-vv-20230129...', safe_dir='/path/to/safe')
    >>> print(meta.wavelength)
    """
    
    def __init__(
        self, 
        burst_id: str, 
        data_dir: str, 
        satellite: str = 'S1'
    ):
        self.burst_id = burst_id
        self.data_dir = data_dir
        self.satellite = satellite
        
        # ISCE3 native objects (to be populated by subclasses)
        self.orbit: Optional[Orbit] = None
        self.radar_grid: Optional[RadarGridParameters] = None
        self.doppler: Optional[LUT2d] = None
        
        # Satellite-specific parameters
        self.satellite_params: Dict[str, Any] = {}
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  burst_id={self.burst_id!r},\n"
            f"  satellite={self.satellite!r},\n"
            f"  orbit={'loaded' if self.orbit else 'None'},\n"
            f"  radar_grid={'loaded' if self.radar_grid else 'None'}\n"
            f")"
        )
    
    @property
    def wavelength(self) -> float:
        """Get radar wavelength in meters."""
        if self.radar_grid is None:
            raise ValueError("Radar grid not initialized")
        return self.radar_grid.wavelength
    
    @property
    def prf(self) -> float:
        """Get Pulse Repetition Frequency (PRF) in Hz."""
        if self.radar_grid is None:
            raise ValueError("Radar grid not initialized")
        return self.radar_grid.prf
    
    @property
    def range_pixel_spacing(self) -> float:
        """Get range pixel spacing in meters."""
        if self.radar_grid is None:
            raise ValueError("Radar grid not initialized")
        return self.radar_grid.range_pixel_spacing
    
    @property
    def sensing_start(self) -> Any:
        """Get sensing start time."""
        if self.radar_grid is None:
            raise ValueError("Radar grid not initialized")
        return self.radar_grid.sensing_start
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export metadata to dictionary format.
        
        Returns
        -------
        dict
            Dictionary containing all metadata fields
        """
        return {
            'burst_id': self.burst_id,
            'satellite': self.satellite,
            'wavelength': self.wavelength if self.radar_grid else None,
            'prf': self.prf if self.radar_grid else None,
            'range_pixel_spacing': self.range_pixel_spacing if self.radar_grid else None,
            **self.satellite_params
        }
    
    def to_xarray_attrs(self) -> Dict[str, Any]:
        """
        Convert metadata to xarray-compatible attributes.
        
        Returns
        -------
        dict
            Dictionary suitable for xarray.Dataset.attrs
        """
        attrs = self.to_dict()
        # Remove non-serializable objects
        return {k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool, type(None)))}

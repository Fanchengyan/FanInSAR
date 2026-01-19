"""
faninsar.isce3 - InSAR processing toolkit using ISCE3

A modern Python package for processing SAR interferometry data using ISCE3 native objects.
Supports Sentinel-1 and designed to be extensible to other SAR satellites.
"""

__version__ = "0.1.0"

# Core classes
from .S1_base import S1_base
from .metadata import SARMetadata
from .transform import SARProcessor, transform

# Main processing function
from .transform import transform

# Geometry functions
from .geometry import geo_to_radar, radar_to_geo, compute_baseline

# Geocoding functions
from .geocode import compute_geo_grid, geocode_slc_array, compute_transform_matrix

# Alignment functions
from .align import estimate_offsets, resample_slc, coregister_slc_pair

# Topographic phase functions
from .topo import (
    compute_flat_earth_phase,
    compute_topo_phase,
    remove_topo_phase,
    compute_flat_earth_topo_phase
)

# GPU utilities
from .gpu import is_gpu_available, GPUContext, GPU_ENABLED

__all__ = [
    # Version
    "__version__",
    
    # Core classes
    "S1_base",
    "SARMetadata",
    "SARProcessor",
    
    # Main function
    "transform",
    
    # Geometry
    "geo_to_radar",
    "radar_to_geo",
    "compute_baseline",
    
    # Geocoding
    "compute_geo_grid",
    "geocode_slc_array",
    "compute_transform_matrix",
    
    # Alignment
    "estimate_offsets",
    "resample_slc",
    "coregister_slc_pair",
    
    # Topographic phase
    "compute_flat_earth_phase",
    "compute_topo_phase",
    "remove_topo_phase",
    "compute_flat_earth_topo_phase",
    
    # GPU utilities
    "is_gpu_available",
    "GPUContext",
    "GPU_ENABLED",
]

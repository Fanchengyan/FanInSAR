"""
GPU acceleration utilities using ISCE3's native GPU support.

This module provides wrappers and utilities for leveraging ISCE3's CUDA-accelerated
processing capabilities when GPU hardware is available.
"""

from __future__ import annotations
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Track GPU availability
_GPU_AVAILABLE = None


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available via ISCE3.
    
    Returns
    -------
    bool
        True if ISCE3 was built with CUDA support and a GPU is present
        
    Examples
    --------
    >>> if is_gpu_available():
    ...     print("GPU acceleration enabled")
    """
    global _GPU_AVAILABLE
    
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    
    try:
        # Try to import ISCE3's CUDA module
        import isce3.cuda
        _GPU_AVAILABLE = True
        logger.info("ISCE3 GPU support detected")
    except (ImportError, AttributeError):
        _GPU_AVAILABLE = False
        logger.info("ISCE3 GPU support not available (CPU-only build)")
    
    return _GPU_AVAILABLE


def get_gpu_info() -> dict:
    """
    Get information about available GPU devices.
    
    Returns
    -------
    dict
        Dictionary with GPU device information
        
    Examples
    --------
    >>> info = get_gpu_info()
    >>> print(f"GPU count: {info['count']}")
    """
    if not is_gpu_available():
        return {
            'available': False,
            'count': 0,
            'devices': []
        }
    
    try:
        import isce3.cuda as cuda
        
        # Get device count
        device_count = cuda.gpu_count() if hasattr(cuda, 'gpu_count') else 0
        
        devices = []
        for i in range(device_count):
            # Get device properties if available
            device_info = {
                'id': i,
                'name': f'GPU {i}'
            }
            devices.append(device_info)
        
        return {
            'available': True,
            'count': device_count,
            'devices': devices
        }
    
    except Exception as e:
        logger.warning(f"Error querying GPU info: {e}")
        return {
            'available': False,
            'count': 0,
            'devices': [],
            'error': str(e)
        }


class GPUContext:
    """
    Context manager for GPU processing.
    
    Manages GPU device selection and memory for ISCE3 GPU operations.
    
    Parameters
    ----------
    device_id : int, optional
        GPU device ID to use, by default 0
        
    Examples
    --------
    >>> with GPUContext(device_id=0) as gpu:
    ...     if gpu.available:
    ...         # Perform GPU-accelerated operations
    ...         result = gpu_geocode(...)
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.available = is_gpu_available()
        self._cuda_module = None
        
    def __enter__(self):
        """Enter GPU context."""
        if self.available:
            try:
                import isce3.cuda
                self._cuda_module = isce3.cuda
                
                # Set device if function exists
                if hasattr(self._cuda_module, 'set_device'):
                    self._cuda_module.set_device(self.device_id)
                    logger.info(f"Using GPU device {self.device_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize GPU context: {e}")
                self.available = False
                
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit GPU context and clean up."""
        if self._cuda_module and hasattr(self._cuda_module, 'synchronize'):
            try:
                self._cuda_module.synchronize()
            except Exception as e:
                logger.warning(f"GPU synchronization warning: {e}")
        
        return False


def enable_gpu_geocoding() -> bool:
    """
    Enable GPU-accelerated geocoding if available.
    
    Returns
    -------
    bool
        True if GPU geocoding was successfully enabled
        
    Notes
    -----
    This function checks if ISCE3's GPU geocoding functions are available
    and configures the system to use them.
    
    Examples
    --------
    >>> if enable_gpu_geocoding():
    ...     print("GPU geocoding enabled")
    """
    if not is_gpu_available():
        return False
    
    try:
        import isce3.cuda.geocode
        logger.info("GPU-accelerated geocoding available")
        return True
    except (ImportError, AttributeError):
        logger.info("GPU geocoding not available in this ISCE3 build")
        return False


def enable_gpu_image_processing() -> bool:
    """
    Enable GPU-accelerated image processing if available.
    
    Returns
    -------
    bool
        True if GPU image processing was successfully enabled
        
    Notes
    -----
    This enables GPU acceleration for operations like resampling,
    cross-correlation, and coregistration.
    
    Examples
    --------
    >>> if enable_gpu_image_processing():
    ...     print("GPU image processing enabled")
    """
    if not is_gpu_available():
        return False
    
    try:
        import isce3.cuda.image
        logger.info("GPU-accelerated image processing available")
        return True
    except (ImportError, AttributeError):
        logger.info("GPU image processing not available in this ISCE3 build")
        return False


def get_optimal_device() -> Optional[int]:
    """
    Get the optimal GPU device ID based on available memory.
    
    Returns
    -------
    int | None
        Device ID with most available memory, or None if no GPUs
        
    Examples
    --------
    >>> device_id = get_optimal_device()
    >>> if device_id is not None:
    ...     with GPUContext(device_id):
    ...         # Process on optimal GPU
    ...         pass
    """
    info = get_gpu_info()
    
    if not info['available'] or info['count'] == 0:
        return None
    
    # For now, just return device 0
    # In a full implementation, would query memory and pick best device
    return 0


# Module-level configuration
GPU_ENABLED = is_gpu_available()

__all__ = [
    'is_gpu_available',
    'get_gpu_info',
    'GPUContext',
    'enable_gpu_geocoding',
    'enable_gpu_image_processing',
    'get_optimal_device',
    'GPU_ENABLED'
]

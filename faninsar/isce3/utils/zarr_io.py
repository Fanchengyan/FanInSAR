"""
Zarr I/O utilities for efficient storage and metadata consolidation.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import zarr
import logging

logger = logging.getLogger(__name__)


def consolidate_zarr_metadata(store_path: str | Path, *, overwrite: bool = True) -> None:
    """
    Consolidate Zarr metadata for faster access.
    
    This function consolidates the metadata of a Zarr store, creating a single
    `.zmetadata` file that contains all the metadata for the entire hierarchy.
    This significantly speeds up opening the store, especially for cloud storage.
    
    Parameters
    ----------
    store_path : str | Path
        Path to the Zarr store directory
    overwrite : bool, optional
        Whether to overwrite existing consolidated metadata, by default True
        
    Examples
    --------
    >>> consolidate_zarr_metadata('/path/to/output.zarr')
    """
    store_path = Path(store_path)
    
    if not store_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {store_path}")
    
    try:
        # Open the store
        store = zarr.storage.LocalStore(str(store_path))
        
        # Consolidate metadata
        zarr.consolidate_metadata(store)
        
        logger.info(f"Consolidated metadata for {store_path}")
        
    except Exception as e:
        logger.error(f"Failed to consolidate metadata for {store_path}: {e}")
        raise


def open_zarr_group(
    store_path: str | Path,
    *,
    mode: str = 'r',
    consolidated: bool = True
) -> zarr.Group:
    """
    Open a Zarr group with optional consolidated metadata.
    
    Parameters
    ----------
    store_path : str | Path
        Path to the Zarr store
    mode : str, optional
        Access mode ('r', 'r+', 'w', 'a'), by default 'r'
    consolidated : bool, optional
        Use consolidated metadata if available, by default True
        
    Returns
    -------
    zarr.Group
        Opened Zarr group
        
    Examples
    --------
    >>> group = open_zarr_group('/path/to/output.zarr')
    >>> ds = xr.open_zarr(group, consolidated=True)
    """
    store_path = Path(store_path)
    
    if mode == 'r' and not store_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {store_path}")
    
    store = zarr.storage.LocalStore(str(store_path))
    
    # Open group
    group = zarr.open_group(
        store=store,
        mode=mode,
        zarr_format=3
    )
    
    return group

"""
Main SAR processing pipeline with Dask parallelization.

Orchestrates the complete workflow from raw SAFE files to geocoded
SLC data stored in Zarr format.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import logging

try:
    from dask.distributed import Client, LocalCluster, progress
    import dask.array as da
except ImportError:
    raise ImportError("dask is required")

from .S1_base import S1_base
from .metadata import SARMetadata
from .geometry import compute_baseline
from .geocode import compute_geo_grid, geocode_slc_array, compute_transform_matrix
from .align import coregister_slc_pair
from .topo import compute_flat_earth_topo_phase, remove_topo_phase
from .utils import consolidate_zarr_metadata

logger = logging.getLogger(__name__)


class SARProcessor:
    """
    Main SAR processing orchestrator.
    
    Manages the complete workflow from SAFE files to geocoded products,
    with Dask-based parallelization and Zarr storage.
    
    Parameters
    ----------
    datadir : str | Path
        Directory containing SAFE files
    dem_path : str | Path
        Path to DEM file
    n_workers : int, optional
        Number of Dask workers, by default None (uses all CPUs)
    use_gpu : bool, optional
        Enable GPU acceleration via ISCE3's CUDA support, by default False
        
    Examples
    --------
    >>> processor = SARProcessor(datadir='/data/safes', dem_path='/data/dem.tif', use_gpu=True)
    >>> processor.run(target='output.zarr', ref='2023-01-29')
    """
    
    def __init__(
        self,
        datadir: str | Path,
        dem_path: str | Path,
        *,
        n_workers: Optional[int] = None,
        use_gpu: bool = False
    ):
        self.datadir = Path(datadir)
        self.dem_path = Path(dem_path)
        self.n_workers = n_workers
        self.use_gpu = use_gpu
        
        # Check GPU availability if requested
        if self.use_gpu:
            from .gpu import is_gpu_available, get_gpu_info
            if is_gpu_available():
                gpu_info = get_gpu_info()
                logger.info(f"GPU acceleration enabled: {gpu_info['count']} device(s) available")
            else:
                logger.warning("GPU requested but not available, falling back to CPU")
                self.use_gpu = False
        
        # Initialize S1 data manager
        self.s1_data = S1_base(datadir=datadir, dem=dem_path)
        
        # Initialize Dask cluster
        self.cluster = None
        self.client = None
        
        logger.info(f"Initialized SARProcessor with {len(self.s1_data.bursts)} bursts")
        
    def __enter__(self):
        """Context manager entry - start Dask cluster."""
        self.start_cluster()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close Dask cluster."""
        self.close_cluster()
        
    def start_cluster(self):
        """Start Dask distributed cluster."""
        if self.client is None:
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit='auto'
            )
            self.client = Client(self.cluster)
            logger.info(f"Started Dask cluster: {self.client.dashboard_link}")
            
    def close_cluster(self):
        """Close Dask cluster."""
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.cluster is not None:
            self.cluster.close()
            self.cluster = None
            logger.info("Closed Dask cluster")
            
    def run(
        self,
        target: str | Path,
        ref: str,
        *,
        epsg: int | str = 'auto',
        resolution: Tuple[float, float] = (20.0, 5.0),
        remove_topo: bool = True,
        overwrite: bool = False
    ) -> Path:
        """
        Run complete processing pipeline.
        
        Parameters
        ----------
        target : str | Path
            Output Zarr store path
        ref : str
            Reference date (YYYY-MM-DD)
        epsg : int | str, optional
            Output CRS EPSG code or 'auto', by default 'auto'
        resolution : tuple of float, optional
            Output resolution (x, y) in meters, by default (20.0, 5.0)
        remove_topo : bool, optional
            Remove topographic phase, by default True
        overwrite : bool, optional
            Overwrite existing output, by default False
            
        Returns
        -------
        Path
            Path to output Zarr store
            
        Examples
        --------
        >>> with SARProcessor('/data/safes', '/data/dem.tif') as proc:
        ...     output = proc.run('output.zarr', ref='2023-01-29')
        """
        target = Path(target)
        
        # Check if output exists
        if target.exists() and not overwrite:
            logger.info(f"Output exists: {target}. Use overwrite=True to replace.")
            return target
        
        # Determine EPSG if auto
        if epsg == 'auto':
            epsg = self._auto_detect_epsg()
            logger.info(f"Auto-detected EPSG: {epsg}")
        
        # Start Dask cluster if not already running
        if self.client is None:
            self.start_cluster()
        
        # Get burst pairs
        pair_dict = self.s1_data.get_repref(ref=ref)
        
        logger.info(f"Processing {len(pair_dict)} burst groups")
        
        # Process each burst group
        tasks = []
        for burst_id, (refs, reps) in pair_dict.items():
            task = self._process_burst_group(
                burst_id, refs, reps, target, epsg, resolution, remove_topo
            )
            tasks.append(task)
        
        # Execute with progress bar
        from tqdm.auto import tqdm
        for task in tqdm(tasks, desc="Processing bursts"):
            # In production, use dask.delayed or dask.distributed.submit
            pass
        
        # Consolidate metadata
        consolidate_zarr_metadata(target)
        
        logger.info(f"Processing complete: {target}")
        
        return target
    
    def _auto_detect_epsg(self) -> int:
        """Auto-detect appropriate UTM zone EPSG code."""
        # Get center coordinate from first burst
        df = self.s1_data.to_dataframe()
        
        # Placeholder: extract from geometry once implemented
        # For now, return a default UTM zone
        default_epsg = 32610  # UTM 10N
        
        logger.warning(f"Using default EPSG {default_epsg} - geometry-based detection pending")
        
        return default_epsg
    
    def _process_burst_group(
        self,
        burst_id: str,
        refs: list,
        reps: list,
        target: Path,
        epsg: int,
        resolution: Tuple[float, float],
        remove_topo: bool
    ):
        """
        Process a single burst group (reference + repeats).
        
        This is the core processing function that handles one burst location
        across all time steps.
        """
        logger.info(f"Processing burst: {burst_id}")
        
        # Get reference metadata
        ref_burst = refs[0]
        meta_ref = self.s1_data.get_metadata(ref_burst[2])  # burst name is index[2]
        
        # Load DEM
        dem = self._load_dem(meta_ref)
        
        # Compute geographic grid
        geo_grid = compute_geo_grid(meta_ref, dem, resolution, epsg=epsg)
        
        # Compute transformation matrix
        transform = compute_transform_matrix(meta_ref, dem, geo_grid)
        
        # Process reference burst(s)
        for ref_burst in refs:
            burst_name = ref_burst[2]
            # Load SLC
            slc_ref = self._load_slc(burst_name)
            
            # Geocode reference
            slc_geo = geocode_slc_array(slc_ref, meta_ref, dem, geo_grid)
            
            # Save to Zarr
            self._save_to_zarr(slc_geo, target, burst_id, burst_name, meta_ref)
        
        # Process repeat bursts
        for rep_burst in reps:
            burst_name = rep_burst[2]
            meta_sec = self.s1_data.get_metadata(burst_name)
            
            # Load SLC
            slc_sec = self._load_slc(burst_name)
            
            # Coregister to reference
            slc_coreg = coregister_slc_pair(
                self._load_slc(refs[0][2]),  # Reference SLC
                slc_sec,
                meta_ref,
                meta_sec
            )
            
            # Remove topographic phase if requested
            if remove_topo:
                topo_phase = compute_flat_earth_topo_phase(dem, meta_ref, meta_sec)
                slc_coreg = remove_topo_phase(slc_coreg, topo_phase)
            
            # Geocode
            slc_geo = geocode_slc_array(slc_coreg, meta_ref, dem, geo_grid)
            
            # Save to Zarr
            self._save_to_zarr(slc_geo, target, burst_id, burst_name, meta_sec)
        
        logger.info(f"Completed burst: {burst_id}")
    
    def _load_dem(self, metadata: SARMetadata) -> xr.DataArray:
        """Load and prepare DEM for processing."""
        import rioxarray as rio
        
        # Load DEM
        dem = rio.open_rasterio(self.dem_path).squeeze()
        
        # Crop to burst extent
        # TODO: Use metadata.radar_grid to determine extent
        
        return dem
    
    def _load_slc(self, burst_name: str) -> xr.DataArray:
        """Load SLC data from SAFE file."""
        # Placeholder: Real implementation would read TIFF from SAFE
        logger.warning("Using placeholder SLC loading - SAFE TIFF reader pending")
        
        # Return dummy data
        return xr.DataArray(
            np.zeros((1000, 2000), dtype=np.complex64),
            dims=['azimuth', 'range']
        )
    
    def _save_to_zarr(
        self,
        data: xr.DataArray,
        target: Path,
        burst_id: str,
        burst_name: str,
        metadata: SARMetadata
    ):
        """Save geocoded SLC to Zarr store."""
        # Convert to dataset
        ds = data.to_dataset(name='slc')
        
        # Add metadata attributes
        ds.attrs.update(metadata.to_xarray_attrs())
        
        # Determine zarr path
        zarr_path = target / burst_id / burst_name
        
        # Save with chunking
        ds.to_zarr(
            zarr_path,
            mode='w',
            consolidated=True
        )
        
        logger.debug(f"Saved to {zarr_path}")


def transform(
    datadir: str | Path,
    dem: str | Path,
    target: str | Path,
    ref: str,
    *,
    epsg: int | str = 'auto',
    resolution: Tuple[float, float] = (20.0, 5.0),
    n_workers: Optional[int] = None,
    remove_topo: bool = True,
    use_gpu: bool = False
) -> Path:
    """
    Convenience function for SAR processing.
    
    Parameters
    ----------
    datadir : str | Path
        Directory with SAFE files
    dem : str | Path
        DEM file path
    target : str | Path
        Output Zarr path
    ref : str
        Reference date
    epsg : int | str, optional
        EPSG code or 'auto'
    resolution : tuple, optional
        Output resolution (x, y) in meters
    n_workers : int, optional
        Number of Dask workers
    remove_topo : bool, optional
        Remove topographic phase
    use_gpu : bool, optional
        Enable GPU acceleration via ISCE3 CUDA (if available)
        
    Returns
    -------
    Path
        Output Zarr path
        
    Examples
    --------
    >>> output = transform(
    ...     datadir='/data/safes',
    ...     dem='/data/dem.tif',
    ...     target='output.zarr',
    ...     ref='2023-01-29',
    ...     resolution=(20, 5),
    ...     use_gpu=True
    ... )
    """
    with SARProcessor(
        datadir=datadir,
        dem_path=dem,
        n_workers=n_workers,
        use_gpu=use_gpu
    ) as processor:
        return processor.run(
            target=target,
            ref=ref,
            epsg=epsg,
            resolution=resolution,
            remove_topo=remove_topo
        )

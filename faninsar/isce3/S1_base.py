"""
Sentinel-1 data management and burst cataloging.

This module provides the main interface for working with Sentinel-1 SAFE datasets,
managing burst catalogs, and accessing metadata.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from typing_extensions import Self
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import logging

from .metadata import SARMetadata
from .utils import S1Metadata

logger = logging.getLogger(__name__)


class S1_base:
    """
    Sentinel-1 data manager with burst cataloging.

    This class manages a collection of Sentinel-1 SAFE files, builds a catalog
    of bursts with their metadata, and provides methods for data access and filtering.
    
    Parameters
    ----------
    datadir : str | Path
        Directory containing Sentinel-1 SAFE files
    dem : str | Path, optional
        Path to DEM file (GeoTIFF or NetCDF)
        
    Attributes
    ----------
    datadir : Path
        Path to data directory
    dem_path : Path | None
        Path to DEM file
    bursts : pd.DataFrame
        Catalog of all bursts with metadata
        
    Examples
    --------
    >>> s1 = S1_base(datadir='/data/sentinel1', dem='/data/dem.tif')
    >>> df = s1.to_dataframe()
    >>> print(f"Total bursts: {len(df)}")
    
    >>> # Filter by date range
    >>> df_filtered = s1.to_dataframe(ref='2023-01-29')
    """
    
    def __init__(self, datadir: str | Path, dem: Optional[str | Path] = None):
        self.datadir = Path(datadir)
        self.dem_path = Path(dem) if dem is not None else None
        
        if not self.datadir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.datadir}")
        
        # Build burst catalog
        self.bursts = self._build_catalog()
        logger.info(f"Loaded {len(self.bursts)} bursts from {self.datadir}")
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self.bursts)} bursts, datadir={self.datadir})"
    
    def _build_catalog(self) -> pd.DataFrame:
        """
        Scan SAFE directories and build burst catalog.
        
        Returns
        -------
        pd.DataFrame
            Catalog with columns: fullBurstId, burst, startTime, orbit, polarization, geometry
        """
        safe_dirs = sorted(self.datadir.glob('S1*.SAFE'))
        
        if not safe_dirs:
            raise ValueError(f"No Sentinel-1 SAFE directories found in {self.datadir}")
        
        records = []
        
        for safe_dir in safe_dirs:
            try:
                # Find annotation files
                annotation_dir = safe_dir / 'annotation'
                if not annotation_dir.exists():
                    logger.warning(f"No annotation directory in {safe_dir.name}")
                    continue
                
                for xml_file in annotation_dir.glob('s1*.xml'):
                    # Parse burst ID from filename
                    # Example: s1a-iw1-slc-vh-20230129t033343-20230129t033408-046872-05a1e0-004.xml
                    stem = xml_file.stem
                    parts = stem.split('-')
                    
                    if len(parts) < 8:
                        continue
                    
                    subswath = parts[1].upper()  # IW1, IW2, IW3
                    polarization = parts[3].upper()  # VV, VH, HH, HV
                    start_time_str = parts[4]  # 20230129t033343
                    
                    # Create burst ID
                    burst_id = stem
                    full_burst_id = f"{safe_dir.stem}_{subswath}_{polarization}"
                    
                    # Parse start time
                    start_time = pd.to_datetime(start_time_str, format='%Y%m%dt%H%M%S')
                    
                    # Extract orbit number from SAFE name
                    # S1A_IW_SLC__1SDV_20230129T033343_20230129T033408_046872_05A1E0_3C4E.SAFE
                    safe_parts = safe_dir.stem.split('_')
                    orbit_num = safe_parts[6] if len(safe_parts) > 6 else 'unknown'
                    
                    # TODO: Extract geometry from annotation XML
                    # For now, use placeholder
                    geometry = None
                    
                    records.append({
                        'fullBurstId': full_burst_id,
                        'burst': burst_id,
                        'startTime': start_time,
                        'orbit': orbit_num,
                        'polarization': polarization,
                        'subswath': subswath,
                        'safe_dir': str(safe_dir),
                        'geometry': geometry
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process {safe_dir.name}: {e}")
                continue
        
        if not records:
            raise ValueError(f"No valid bursts found in {self.datadir}")
        
        df = pd.DataFrame(records)
        
        # Set multi-index: (fullBurstId, polarization, burst)
        df = df.set_index(['fullBurstId', 'polarization', 'burst'])
        df = df.sort_index()
        
        return df
    
    def to_dataframe(
        self, 
        crs: int = 4326, 
        ref: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get burst catalog as DataFrame, optionally filtered by reference date.
        
        Parameters
        ----------
        crs : int, optional
            CRS code for geometry (not yet implemented), by default 4326
        ref : str, optional
            Reference date (YYYY-MM-DD) to filter bursts, by default None
            
        Returns
        -------
        pd.DataFrame
            Burst catalog DataFrame
            
        Examples
        --------
        >>> df = s1.to_dataframe()
        >>> df_ref = s1.to_dataframe(ref='2023-01-29')
        """
        if ref is None:
            return self.bursts.copy()
        
        # Get fullBurstIDs from reference date
        ref_mask = self.bursts['startTime'].dt.date.astype(str) == ref
        ref_burst_ids = self.bursts[ref_mask].index.get_level_values(0).unique()
        
        if len(ref_burst_ids) == 0:
            raise ValueError(f"Reference date '{ref}' not found in data")
        
        # Filter all bursts with these IDs
        df = self.bursts[self.bursts.index.get_level_values(0).isin(ref_burst_ids)]
        
        return df.copy()
    
    def get_metadata(self, burst_id: str) -> S1Metadata:
        """
        Load ISCE3 metadata for a specific burst.
        
        Parameters
        ----------
        burst_id : str
            Burst identifier
            
        Returns
        -------
        S1Metadata
            Metadata object with ISCE3 Orbit and RadarGridParameters
            
        Examples
        --------
        >>> meta = s1.get_metadata('s1a-iw1-slc-vh-20230129t033343-...')
        >>> print(meta.wavelength)
        """
        record = self.get_record(burst_id)
        safe_dir = record['safe_dir'].iloc[0]
        
        return S1Metadata(burst_id=burst_id, safe_dir=safe_dir)
    
    def get_record(self, burst_id: str) -> pd.DataFrame:
        """
        Get DataFrame record(s) for a specific burst.
        
        Parameters
        ----------
        burst_id : str
            Burst identifier (can be burst ID or fullBurstId)
            
        Returns
        -------
        pd.DataFrame
            Record(s) matching the burst ID
            
        Raises
        ------
        ValueError
            If burst ID not found
        """
        # Try matching by burst (index level 2)
        df = self.bursts[self.bursts.index.get_level_values(2) == burst_id]
        
        # If not found, try fullBurstId (index level 0)
        if len(df) == 0:
            df = self.bursts[self.bursts.index.get_level_values(0) == burst_id]
        
        if len(df) == 0:
            raise ValueError(f"Burst '{burst_id}' not found in catalog")
        
        return df
    
    def get_repref(
        self, 
        ref: str, 
        records: Optional[pd.DataFrame] = None
    ) -> Dict[str, Tuple[List, List]]:
        """
        Group bursts into reference and repeat pairs for interferometry.
        
        Parameters
        ----------
        ref : str
            Reference date (YYYY-MM-DD)
        records : pd.DataFrame, optional
            Subset of bursts to process, by default None (use all)
            
        Returns
        -------
        dict
            Dictionary mapping fullBurstId to (reference_bursts, repeat_bursts)
            
        Examples
        --------
        >>> pairs = s1.get_repref(ref='2023-01-29')
        >>> for burst_id, (refs, reps) in pairs.items():
        ...     print(f"{burst_id}: {len(refs)} ref, {len(reps)} repeat")
        """
        if records is None:
            records = self.to_dataframe(ref=ref)
        
        # Separate into reference and repeat
        ref_mask = records['startTime'].dt.date.astype(str) == ref
        recs_ref = records[ref_mask]
        recs_rep = records[~ref_mask]
        
        # Group by fullBurstId
        refs_dict: Dict[str, List] = {}
        for idx in recs_ref.index:
            full_burst_id = idx[0]
            refs_dict.setdefault(full_burst_id, []).append(idx)
        
        reps_dict: Dict[str, List] = {}
        for idx in recs_rep.index:
            full_burst_id = idx[0]
            reps_dict.setdefault(full_burst_id, []).append(idx)
        
        # Log unpaired bursts
        for key in refs_dict:
            if key not in reps_dict:
                logger.info(f"{key} has no repeat bursts, skipping")
        
        for key in reps_dict:
            if key not in refs_dict:
                logger.info(f"{key} has no reference burst, skipping")
        
        # Return only pairs with both
        return {
            key: (refs_dict[key], reps_dict[key])
            for key in refs_dict
            if key in reps_dict
        }
    
    @property
    def df(self) -> pd.DataFrame:
        """Alias for bursts catalog (for compatibility)."""
        return self.bursts

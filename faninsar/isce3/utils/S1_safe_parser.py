"""
Sentinel-1 SAFE format metadata parser.

Parses Sentinel-1 annotation XML files to extract orbit, radar grid, and
TOPS-specific parameters into ISCE3 native objects.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
from lxml import etree
import logging

from ..metadata import SARMetadata

try:
    import isce3
    from isce3.core import Orbit, LUT2d, StateVector, DateTime
    from isce3.product import RadarGridParameters
except ImportError:
    raise ImportError("ISCE3 is required")

logger = logging.getLogger(__name__)


class S1Metadata(SARMetadata):
    """
    Sentinel-1 specific metadata parser.
    
    Parses Sentinel-1 SAFE annotation XML files to extract:
    - Orbit state vectors → isce3.core.Orbit
    - Radar grid parameters → isce3.product.RadarGridParameters  
    - Doppler centroid → isce3.core.LUT2d
    - TOPS-specific parameters (FM rates, burst timing, etc.)
    
    Parameters
    ----------
    burst_id : str
        Sentinel-1 burst identifier (e.g., 'S1A_IW1_SLC__1SDV_20230129...')
    safe_dir : str
        Path to SAFE directory
        
    Examples
    --------
    >>> meta = S1Metadata('S1A_IW1_...', '/path/to/S1A.SAFE')
    >>> print(meta.wavelength)  # 0.055465763
    >>> print(meta.satellite_params['azimuth_fm_rate'])
    """
    
    def __init__(self, burst_id: str, safe_dir: str):
        super().__init__(burst_id, safe_dir, satellite='S1')
        
        self.safe_path = Path(safe_dir)
        self.annotation_path = self._find_annotation()
        
        # Parse XML and populate ISCE3 objects
        self.orbit = self._parse_orbit()
        self.radar_grid = self._parse_radar_grid()
        self.doppler = self._parse_doppler()
        self.satellite_params = self._parse_tops_params()
        
    def _find_annotation(self) -> Path:
        """Find the annotation XML file for this burst."""
        annotation_dir = self.safe_path / 'annotation'
        if not annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
        
        # Match burst_id to annotation file
        xml_files = list(annotation_dir.glob(f'{self.burst_id}.xml'))
        if not xml_files:
            # Try without full extension
            base_id = self.burst_id.split('.')[0] if '.' in self.burst_id else self.burst_id
            xml_files = list(annotation_dir.glob(f'{base_id}.xml'))
        
        if not xml_files:
            raise FileNotFoundError(
                f"Annotation XML not found for {self.burst_id} in {annotation_dir}"
            )
        
        return xml_files[0]
    
    def _parse_orbit(self) -> Orbit:
        """
        Parse orbit state vectors from annotation XML.
        
        Returns
        -------
        isce3.core.Orbit
            Orbit containing state vectors and interpolation parameters
        """
        tree = etree.parse(str(self.annotation_path))
        root = tree.getroot()
        
        # Find orbit state vectors
        orbit_list = root.find('.//orbitList')
        if orbit_list is None:
            raise ValueError("orbitList not found in annotation")
        
        state_vectors = []
        
        for orbit_elem in orbit_list.findall('orbit'):
            # Parse time
            time_str = orbit_elem.find('time').text
            # Convert ISO8601 to DateTime
            dt = DateTime(time_str)
            
            # Parse position (m)
            position = np.array([
                float(orbit_elem.find('position/x').text),
                float(orbit_elem.find('position/y').text),
                float(orbit_elem.find('position/z').text)
            ])
            
            # Parse velocity (m/s)
            velocity = np.array([
                float(orbit_elem.find('velocity/x').text),
                float(orbit_elem.find('velocity/y').text),
                float(orbit_elem.find('velocity/z').text)
            ])
            
            sv = StateVector()
            sv.datetime = dt
            sv.position = position
            sv.velocity = velocity
            
            state_vectors.append(sv)
        
        # Create Orbit object
        orbit = Orbit(state_vectors)
        return orbit
    
    def _parse_radar_grid(self) -> RadarGridParameters:
        """
        Parse radar grid parameters from annotation XML.
        
        Returns
        -------
        isce3.product.RadarGridParameters
            Radar grid geometry and timing parameters
        """
        tree = etree.parse(str(self.annotation_path))
        root = tree.getroot()
        
        # Extract parameters
        image_info = root.find('.//imageAnnotation/imageInformation')
        
        sensing_start_str = image_info.find('productFirstLineUtcTime').text
        sensing_start = DateTime(sensing_start_str)
        
        wavelength = float(root.find('.//radarFrequency').text) / isce3.core

.speed_of_light
        
        prf = float(image_info.find('azimuthTimeInterval').text)**-1  # Convert interval to frequency
        
        starting_range = float(image_info.find('slantRangeTime').text) * isce3.core.speed_of_light / 2
        
        range_pixel_spacing = float(image_info.find('rangePixelSpacing').text)
        
        lookside = isce3.core.LookSide.Right  # Sentinel-1 is always right-looking
        
        length = int(image_info.find('numberOfLines').text)
        width = int(image_info.find('numberOfSamples').text)
        
        # Create RadarGridParameters
        radar_grid = RadarGridParameters(
            sensing_start,
            wavelength,
            prf,
            starting_range,
            range_pixel_spacing,
            lookside,
            length,
            width,
            'zero'  # ref_epoch
        )
        
        return radar_grid
    
    def _parse_doppler(self) -> LUT2d:
        """
        Parse Doppler centroid from annotation XML.
        
        Returns
        -------
        isce3.core.LUT2d
            2D Doppler centroid lookup table
        """
        tree = etree.parse(str(self.annotation_path))
        root = tree.getroot()
        
        # Find Doppler centroid estimates
        dc_estimates = root.find('.//dopplerCentroid/dcEstimateList')
        
        if dc_estimates is None:
            # Return zero Doppler if not available
            logger.warning(f"Doppler centroid not found for {self.burst_id}, using zero Doppler")
            return LUT2d()
        
        # Parse Doppler polynomials
        times = []
        coeffs_list = []
        
        for dc_elem in dc_estimates.findall('dcEstimate'):
            az_time = float(dc_elem.find('azimuthTime').text)
            
            # Get polynomial coefficients
            coeffs_str = dc_elem.find('dataDcPolynomial').text
            coeffs = [float(c) for c in coeffs_str.split()]
            
            times.append(az_time)
            coeffs_list.append(coeffs)
        
        # Create LUT2d (simplified - in practice need to handle 2D properly)
        # For now, use a zero Doppler approximation
        return LUT2d()
    
    def _parse_tops_params(self) -> Dict[str, Any]:
        """
        Parse TOPS-specific parameters (azimuth FM rate, burst timing, etc.).
        
        Returns
        -------
        dict
            Dictionary of TOPS parameters
        """
        tree = etree.parse(str(self.annotation_path))
        root = tree.getroot()
        
        params = {}
        
        # Azimuth FM rate
        azimuth_fm_rate_list = root.find('.//azimuthFmRateList')
        if azimuth_fm_rate_list is not None:
            fm_rates = []
            for fm_elem in azimuth_fm_rate_list.findall('azimuthFmRate'):
                az_time = float(fm_elem.find('azimuthTime').text)
                coeffs_str = fm_elem.find('azimuthFmRatePolynomial').text
                coeffs = [float(c) for c in coeffs_str.split()]
                fm_rates.append({'time': az_time, 'coeffs': coeffs})
            params['azimuth_fm_rate'] = fm_rates
        
        # Burst information (if available)
        swath_timing = root.find('.//swathTiming')
        if swath_timing is not None:
            burst_list = swath_timing.find('burstList')
            if burst_list is not None:
                bursts = []
                for burst_elem in burst_list.findall('burst'):
                    burst_info = {
                        'azimuth_time': burst_elem.find('azimuthTime').text,
                        'azimuth_anx_time': float(burst_elem.find('azimuthAnxTime').text),
                        'sensing_time': burst_elem.find('sensingTime').text,
                        'byte_offset': int(burst_elem.find('byteOffset').text),
                        'first_valid_sample': [int(s) for s in burst_elem.find('firstValidSample').text.split()],
                        'last_valid_sample': [int(s) for s in burst_elem.find('lastValidSample').text.split()],
                    }
                    bursts.append(burst_info)
                params['bursts'] = bursts
        
        return params

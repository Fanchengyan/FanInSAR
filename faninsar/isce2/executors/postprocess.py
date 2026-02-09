"""Postprocessing executors for ISCE2 (geocoding and unwrapping).

This module provides functionality for geocoding and phase unwrapping.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Planet.Planet import Planet

from faninsar.isce2.isce_utils import load_product
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command

logger = setup_logger(__name__)


def _exec_geocode(cmd: Command) -> int:
    """Execute geocoding using GDAL.

    Parameters
    ----------
    cmd : Command
        Geocode command with parameters:
        - lat_file: Path to latitude file
        - lon_file: Path to longitude file
        - input_file: Path to input file to geocode
        - output_file: Path to output geocoded file
        - bbox: Bounding box [south, north, west, east]
        - lat_step: Output latitude step (degrees)
        - lon_step: Output longitude step (degrees)
        - method: Resampling method

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Extract parameters
        lat_file = Path(cmd.params["lat_file"])
        lon_file = Path(cmd.params["lon_file"])
        input_file = Path(cmd.params["input_file"])
        output_file = Path(cmd.params["output_file"])
        bbox = cmd.params.get("bbox")
        lat_step = float(cmd.params.get("lat_step", 0.001))
        lon_step = float(cmd.params.get("lon_step", 0.001))
        method = str(cmd.params.get("method", "near"))

        logger.info("Geocoding: %s", input_file)
        logger.info("Output: %s", output_file)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create VRT with geolocation metadata
        vrt_file = input_file.with_suffix(".geo.vrt")

        # Parse existing VRT or create one
        if not (input_file.parent / (input_file.name + ".vrt")).exists():
            # Create VRT using gdal_translate
            cmd_vrt = [
                "gdal_translate",
                "-of",
                "VRT",
                str(input_file),
                str(input_file) + ".vrt",
            ]
            subprocess.run(cmd_vrt, check=True, capture_output=True)

        # Load VRT and add geolocation metadata
        tree = ET.parse(str(input_file) + ".vrt")
        root = tree.getroot()

        # Add geolocation metadata
        meta = ET.SubElement(root, "metadata")
        meta.attrib["domain"] = "GEOLOCATION"
        meta.tail = "\n"
        meta.text = "\n    "

        rdict = {
            "Y_DATASET": str(lat_file),
            "X_DATASET": str(lon_file),
            "X_BAND": "1",
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "LINE_STEP": "1",
            "PIXEL_STEP": "1",
        }

        for key, val in rdict.items():
            data = ET.SubElement(meta, "mdi")
            data.text = val
            data.attrib["key"] = key
            data.tail = "\n    "

        data.tail = "\n"
        tree.write(str(vrt_file))

        # Build gdalwarp command
        gdalwarp_cmd = [
            "gdalwarp",
            "-of",
            "ENVI",
            "-geoloc",
            "-r",
            method,
            "-tr",
            str(lon_step),
            str(lat_step),
        ]

        # Add bounding box if provided
        if bbox and len(bbox) == 4:
            # bbox format: [south, north, west, east]
            gdalwarp_cmd.extend(
                ["-te", str(bbox[2]), str(bbox[0]), str(bbox[3]), str(bbox[1])]
            )

        # Add input and output
        gdalwarp_cmd.extend([str(vrt_file), str(output_file)])

        # Execute gdalwarp
        logger.info("Running gdalwarp: %s", " ".join(gdalwarp_cmd))
        result = subprocess.run(
            gdalwarp_cmd, check=True, capture_output=True, text=True
        )

        if result.returncode == 0:
            logger.info("Geocoding completed successfully")
            return 0
        logger.error("Gdalwarp failed: %s", result.stderr)
        return 1

    except Exception:
        logger.exception("Geocoding failed")
        return 1


def _exec_unwrap(cmd: Command) -> int:
    """Execute phase unwrapping using SNAPHU.

    Parameters
    ----------
    cmd : Command
        Unwrap command with parameters:
        - interferogram: Path to interferogram
        - coherence: Path to coherence file
        - unwrapped: Path to output unwrapped phase
        - reference: Path to reference directory
        - azimuth_looks: Number of azimuth looks
        - range_looks: Number of range looks
        - defo_max: Maximum deformation (cycles)
        - method: Unwrapping method ('snaphu' or 'icu')
        - nomcf: Whether to run full SNAPHU (not MCF)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Extract parameters
        int_file = Path(cmd.params["interferogram"])
        coh_file = Path(cmd.params["coherence"])
        unw_file = Path(cmd.params["unwrapped"])
        reference_dir = Path(cmd.params["reference"])
        azimuth_looks = int(cmd.params.get("azimuth_looks", 1))
        range_looks = int(cmd.params.get("range_looks", 1))
        defo_max = float(cmd.params.get("defo_max", 2.0))
        method = str(cmd.params.get("method", "snaphu"))
        nomcf = bool(cmd.params.get("nomcf", False))

        logger.info("Unwrapping: %s", int_file)
        logger.info("Method: %s", method)
        logger.info("Azimuth looks: %d, Range looks: %d", azimuth_looks, range_looks)

        # Ensure output directory exists
        unw_file.parent.mkdir(parents=True, exist_ok=True)

        if method.lower() != "snaphu":
            logger.warning("Only SNAPHU method is currently implemented")
            logger.info("Using SNAPHU for unwrapping")

        # Load reference product to extract metadata
        # Find first swath XML
        swath_xmls = list(reference_dir.glob("IW*.xml"))
        if not swath_xmls:
            logger.error("No swath XML files found in reference directory")
            return 1

        reference = load_product(swath_xmls[0])
        burst = reference.bursts[0]

        # Extract wavelength and geometry info
        wavelength = burst.radarWavelength

        # Calculate earth radius and altitude
        planet = Planet(pname="Earth")
        orbit = burst.orbit
        tmid = burst.sensingStart

        peg = orbit.interpolateOrbit(tmid, method="hermite")
        refElp = planet.ellipsoid
        llh = refElp.xyz_to_llh(peg.getPosition())
        hdg = orbit.getENUHeading(tmid)
        refElp.setSCH(llh[0], llh[1], hdg)

        earth_radius = refElp.pegRadCur
        altitude = llh[2]

        # Calculate correlation looks
        azfact = 0.8
        rngfact = 0.8
        corr_looks = range_looks * azimuth_looks / (azfact * rngfact)

        logger.info("Wavelength: %f m", wavelength)
        logger.info("Earth radius: %f m", earth_radius)
        logger.info("Altitude: %f m", altitude)
        logger.info("Correlation looks: %f", corr_looks)

        # Load interferogram and coherence images
        intImage = isceobj.createIntImage()
        intImage.load(str(int_file) + ".xml")
        intImage.setAccessMode("READ")

        cohImage = isceobj.createImage()
        cohImage.load(str(coh_file) + ".xml")
        cohImage.dataType = "FLOAT"
        cohImage.setAccessMode("READ")

        # Create SNAPHU object
        snp = Snaphu()
        snp.setInitOnly(False)
        snp.setInput(str(int_file))
        snp.setOutput(str(unw_file))
        snp.setWidth(intImage.getWidth())
        snp.setCostMode("DEFO")
        snp.setEarthRadius(earth_radius)
        snp.setWavelength(wavelength)
        snp.setAltitude(altitude)
        snp.setCorrfile(str(coh_file))
        snp.setInitMethod("MCF")
        snp.setCorrLooks(corr_looks)
        snp.setMaxComponents(100)
        snp.setDefoMaxCycles(defo_max)

        if nomcf:
            snp.setInitMethod("MST")

        # Prepare and run SNAPHU
        snp.prepare()
        snp.unwrap()

        # Finalize
        logger.info("Creating output image")
        unwImage = isceobj.createImage()
        unwImage.setFilename(str(unw_file))
        unwImage.setWidth(intImage.getWidth())
        unwImage.dataType = "FLOAT"
        unwImage.bands = 2
        unwImage.scheme = "BIP"
        unwImage.setAccessMode("read")
        unwImage.renderHdr()

        logger.info("Unwrapping completed successfully")
        return 0

    except Exception:
        logger.exception("Unwrapping failed")
        return 1

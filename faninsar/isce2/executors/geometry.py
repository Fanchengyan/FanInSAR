"""Geometry processing executors for ISCE2.

This module provides functionality for topo and geo2rdr processing.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import isceobj
from isceobj.Planet.Planet import Planet
from zerodop.geo2rdr import createGeo2rdr
from zerodop.topozero import createTopozero

from faninsar.isce2.isce_utils import get_swath_list, load_product
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command

logger = setup_logger(__name__)


def _exec_topo(cmd: Command) -> int:
    """Execute topo processing using zerodop.topozero.

    This function implements topo processing by calling createTopozero
    for each burst in each swath, following the approach from
    contrib/stack/topsStack/topo.py.

    Parameters
    ----------
    cmd : Command
        Topo command with parameters:
        - reference: Path to reference directory
        - dem: Path to DEM file
        - geom_dir: Path to output geometry directory
        - num_process: Number of parallel processes

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Use Path objects internally
        reference_dir = Path(cmd.params["reference"])
        dem_file = Path(cmd.params["dem"])
        geom_dir = Path(cmd.params["geom_dir"])

        # Get swath list
        swath_list = get_swath_list(reference_dir)

        if not swath_list:
            logger.error("No swaths found in reference directory")
            return 1

        logger.info("Processing swaths: %s", swath_list)

        # Load DEM
        demImage = isceobj.createDemImage()
        demImage.load(str(dem_file.with_suffix(".xml")))

        planet = Planet(pname="Earth")

        # Process each swath
        for swath in swath_list:
            logger.info("Processing swath IW%d", swath)

            # Load reference product
            reference = load_product(reference_dir / f"IW{swath}.xml")

            # Create output directory
            outdir = geom_dir / f"IW{swath}"
            outdir.mkdir(parents=True, exist_ok=True)

            # Process each burst
            for i, burst in enumerate(reference.bursts):
                logger.info("Processing burst %d", i)

                # Create output file paths
                latFilename = str(outdir / f"lat_{i + 1:02d}.rdr")
                lonFilename = str(outdir / f"lon_{i + 1:02d}.rdr")
                hgtFilename = str(outdir / f"hgt_{i + 1:02d}.rdr")

                # Create topo object
                topo = createTopozero()
                topo.slantRangePixelSpacing = burst.rangePixelSize
                topo.prf = 1.0 / burst.azimuthTimeInterval
                topo.radarWavelength = burst.radarWavelength
                topo.orbit = burst.orbit
                topo.width = burst.numberOfSamples
                topo.length = burst.numberOfLines
                topo.wireInputPort(name="dem", object=demImage)
                topo.wireInputPort(name="planet", object=planet)
                topo.numberRangeLooks = 1
                topo.numberAzimuthLooks = 1
                topo.lookSide = -1
                topo.sensingStart = burst.sensingStart
                topo.rangeFirstSample = burst.startingRange

                # Set output filenames
                topo.latFilename = latFilename
                topo.lonFilename = lonFilename
                topo.heightFilename = hgtFilename

                # Run topo
                topo.topo()

        logger.info("Topo processing completed successfully")
        return 0
    except Exception:
        logger.exception("Topo execution failed")
        return 1


def _exec_geo2rdr(cmd: Command) -> int:
    """Execute geo2rdr processing using zerodop.geo2rdr.

    This function implements geo2rdr processing to compute offset fields
    between geographic coordinates and radar coordinates.

    Parameters
    ----------
    cmd : Command
        Geo2rdr command with parameters:
        - secondary: Path to secondary directory
        - reference: Path to reference directory
        - geom_reference: Path to geometry reference directory
        - coreg_dir: Path to coregistered output directory
        - overlap: Whether this is overlap processing
        - misreg_az: Path to azimuth misregistration file
        - misreg_rng: Path to range misregistration file
        - use_gpu: Whether to use GPU acceleration

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Use Path objects internally
        secondary_dir = Path(cmd.params["secondary"])
        reference_dir = Path(cmd.params["reference"])
        geom_reference_dir = Path(cmd.params["geom_reference"])
        coreg_dir = Path(cmd.params["coreg_dir"])
        overlap = bool(cmd.params["overlap"])
        misreg_az_path = (
            Path(cmd.params["misreg_az"]) if cmd.params.get("misreg_az") else None
        )
        misreg_rng_path = (
            Path(cmd.params["misreg_rng"]) if cmd.params.get("misreg_rng") else None
        )
        use_gpu = bool(cmd.params["use_gpu"])

        # Check GPU availability
        run_gpu = False
        if use_gpu:
            try:
                from zerodop.GPUgeo2rdr.GPUgeo2rdr import PyGeo2rdr  # noqa: F401

                run_gpu = True
                logger.info("GPU mode enabled")
            except ImportError:
                logger.warning("GPU mode requested but GPU code not available")

        # Get swath list
        reference_swath_list = get_swath_list(reference_dir)
        secondary_swath_list = get_swath_list(secondary_dir)
        swath_list = sorted(set(reference_swath_list + secondary_swath_list))

        if not swath_list:
            logger.error("No swaths found")
            return 1

        logger.info("Processing swaths: %s", swath_list)

        # Read misregistration values
        misreg_az_val = 0.0
        if misreg_az_path and misreg_az_path.exists():
            with misreg_az_path.open() as f:
                misreg_az_val = float(f.readline())

        misreg_rg_val = 0.0
        if misreg_rng_path and misreg_rng_path.exists():
            with misreg_rng_path.open() as f:
                misreg_rg_val = float(f.readline())

        logger.info("Azimuth misregistration: %f", misreg_az_val)
        logger.info("Range misregistration: %f", misreg_rg_val)

        # Process each swath
        for swath in swath_list:
            # Load products (ISCE2 requires string paths)
            secondary = load_product(secondary_dir / f"IW{swath}.xml")
            reference = load_product(reference_dir / f"IW{swath}.xml")

            # Create output directory
            if overlap:
                outdir = coreg_dir / "overlap" / f"IW{swath}"
                geom_dir = geom_reference_dir / "overlap" / f"IW{swath}"
            else:
                outdir = coreg_dir / f"IW{swath}"
                geom_dir = geom_reference_dir / f"IW{swath}"

            outdir.mkdir(parents=True, exist_ok=True)

            # Get common burst limits
            burstoffset, minBurst, maxBurst = reference.getCommonBurstLimits(secondary)

            if overlap:
                maxBurst = maxBurst - 1

            secondary_burst_start = minBurst + burstoffset

            # Process each burst
            for m_burst in range(minBurst, maxBurst):
                s_burst = secondary_burst_start + (m_burst - minBurst)
                burst_top = secondary.bursts[s_burst]

                logger.info(
                    "Processing burst %d (reference) with burst %d (secondary)",
                    m_burst,
                    s_burst,
                )

                if overlap:
                    burst_bot = secondary.bursts[s_burst + 1]

                    # Process top burst (convert Path to str for ISCE2)
                    rdict_top = {
                        "lat": str(
                            geom_dir / f"lat_{m_burst + 1:02d}_{m_burst + 2:02d}.rdr"
                        ),
                        "lon": str(
                            geom_dir / f"lon_{m_burst + 1:02d}_{m_burst + 2:02d}.rdr"
                        ),
                        "hgt": str(
                            geom_dir / f"hgt_{m_burst + 1:02d}_{m_burst + 2:02d}.rdr"
                        ),
                        "rangeOffName": str(
                            outdir
                            / f"range_top_{m_burst + 1:02d}_{m_burst + 2:02d}.off"
                        ),
                        "azOffName": str(
                            outdir
                            / f"azimuth_top_{m_burst + 1:02d}_{m_burst + 2:02d}.off"
                        ),
                    }
                    _run_geo2rdr(
                        burst_top, rdict_top, misreg_az_val, misreg_rg_val, run_gpu
                    )

                    # Process bottom burst (convert Path to str for ISCE2)
                    rdict_bot = {
                        "lat": str(
                            geom_dir / f"lat_{m_burst + 1:02d}_{m_burst + 2:02d}.rdr"
                        ),
                        "lon": str(
                            geom_dir / f"lon_{m_burst + 1:02d}_{m_burst + 2:02d}.rdr"
                        ),
                        "hgt": str(
                            geom_dir / f"hgt_{m_burst + 1:02d}_{m_burst + 2:02d}.rdr"
                        ),
                        "rangeOffName": str(
                            outdir
                            / f"range_bot_{m_burst + 1:02d}_{m_burst + 2:02d}.off"
                        ),
                        "azOffName": str(
                            outdir
                            / f"azimuth_bot_{m_burst + 1:02d}_{m_burst + 2:02d}.off"
                        ),
                    }
                    _run_geo2rdr(
                        burst_bot, rdict_bot, misreg_az_val, misreg_rg_val, run_gpu
                    )
                else:
                    # Process single burst (convert Path to str for ISCE2)
                    rdict = {
                        "lat": str(geom_dir / f"lat_{m_burst + 1:02d}.rdr"),
                        "lon": str(geom_dir / f"lon_{m_burst + 1:02d}.rdr"),
                        "hgt": str(geom_dir / f"hgt_{m_burst + 1:02d}.rdr"),
                        "rangeOffName": str(outdir / f"range_{m_burst + 1:02d}.off"),
                        "azOffName": str(outdir / f"azimuth_{m_burst + 1:02d}.off"),
                    }
                    _run_geo2rdr(
                        burst_top, rdict, misreg_az_val, misreg_rg_val, run_gpu
                    )

        return 0
    except Exception:
        logger.exception("Geo2rdr execution failed")
        return 1


def _run_geo2rdr(
    info,
    rdict: dict,
    misreg_az: float = 0.0,
    misreg_rg: float = 0.0,
    use_gpu: bool = False,
) -> None:
    """Run geo2rdr for a single burst.

    Parameters
    ----------
    info : object
        Burst information object.
    rdict : dict
        Dictionary with paths to lat, lon, hgt, and output files.
    misreg_az : float, optional
        Azimuth misregistration in pixels. Default is 0.0.
    misreg_rg : float, optional
        Range misregistration in meters. Default is 0.0.
    use_gpu : bool, optional
        Whether to use GPU implementation. Default is False.

    """
    # Load geometry images
    latImage = isceobj.createImage()
    latImage.load(rdict["lat"] + ".xml")
    latImage.setAccessMode("READ")

    lonImage = isceobj.createImage()
    lonImage.load(rdict["lon"] + ".xml")
    lonImage.setAccessMode("READ")

    demImage = isceobj.createImage()
    demImage.load(rdict["hgt"] + ".xml")
    demImage.setAccessMode("READ")

    # Convert misregistration values
    misreg_az_time = misreg_az * info.azimuthTimeInterval
    delta = datetime.timedelta(seconds=misreg_az_time)

    logger.info("Additional time offset applied: %f secs", misreg_az_time)
    logger.info("Additional range offset applied: %f m", misreg_rg)

    # Create and configure geo2rdr
    planet = Planet(pname="Earth")
    grdr = createGeo2rdr()
    grdr.configure()

    grdr.slantRangePixelSpacing = info.rangePixelSize
    grdr.prf = 1.0 / info.azimuthTimeInterval
    grdr.radarWavelength = info.radarWavelength
    grdr.orbit = info.orbit
    grdr.width = info.numberOfSamples
    grdr.length = info.numberOfLines
    grdr.demLength = demImage.getLength()
    grdr.demWidth = demImage.getWidth()
    grdr.wireInputPort(name="planet", object=planet)
    grdr.numberRangeLooks = 1
    grdr.numberAzimuthLooks = 1
    grdr.lookSide = -1
    grdr.setSensingStart(info.sensingStart - delta)
    grdr.rangeFirstSample = info.startingRange - misreg_rg
    grdr.dopplerCentroidCoeffs = [0.0]  # Zero doppler

    grdr.rangeOffsetImageName = rdict["rangeOffName"]
    grdr.azimuthOffsetImageName = rdict["azOffName"]
    grdr.demImage = demImage
    grdr.latImage = latImage
    grdr.lonImage = lonImage

    grdr.geo2rdr()

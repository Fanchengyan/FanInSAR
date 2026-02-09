"""Coregistration and resampling executors for ISCE2.

This module provides functionality for image coregistration and resampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import isceobj
import stdproc
from isceobj.Util.Poly2D import Poly2D

from faninsar.isce2.isce_utils import get_swath_list, load_product
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command

logger = setup_logger(__name__)


def _exec_resamp(cmd: Command) -> int:
    """Execute resamp_withCarrier processing.

    This function resamples secondary SLC to reference geometry using offset
    fields from geo2rdr and optionally flattens the interferogram by removing
    topographic phase.

    Parameters
    ----------
    cmd : Command
        Resamp command with parameters:
        - reference: Path to reference directory
        - secondary: Path to secondary directory
        - coreg_dir: Path to coregistered output directory
        - misreg_az: Path to azimuth misregistration file (optional)
        - misreg_rng: Path to range misregistration file (optional)
        - flatten: Whether to flatten interferogram (default True)
        - overlap: Whether this is overlap burst (default False)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Extract parameters
        reference_dir = Path(cmd.params["reference"])
        secondary_dir = Path(cmd.params["secondary"])
        coreg_dir = Path(cmd.params["coreg_dir"])
        misreg_az_path = cmd.params.get("misreg_az")
        misreg_rng_path = cmd.params.get("misreg_rng")
        flatten = bool(cmd.params.get("flatten", True))
        overlap = bool(cmd.params.get("overlap", False))

        # Read misregistration values
        misreg_az_val = 0.0
        if misreg_az_path and Path(misreg_az_path).exists():
            with Path(misreg_az_path).open() as f:
                misreg_az_val = float(f.readline())

        misreg_rg_val = 0.0
        if misreg_rng_path and Path(misreg_rng_path).exists():
            with Path(misreg_rng_path).open() as f:
                misreg_rg_val = float(f.readline())

        logger.info("Azimuth misregistration: %f", misreg_az_val)
        logger.info("Range misregistration: %f", misreg_rg_val)
        logger.info("Flatten: %s", flatten)

        # Get swath list
        swath_list = get_swath_list(reference_dir)

        # Process each swath
        for swath in swath_list:
            logger.info("Resampling swath IW%d", swath)

            # Load products
            reference = load_product(reference_dir / f"IW{swath}.xml")
            secondary = load_product(secondary_dir / f"IW{swath}.xml")

            # Determine directory structure
            if overlap:
                geo_dir = coreg_dir / "overlap" / f"IW{swath}"
                outdir = coreg_dir / "overlap" / f"IW{swath}"
            else:
                geo_dir = coreg_dir / f"IW{swath}"
                outdir = coreg_dir / f"IW{swath}"

            outdir.mkdir(parents=True, exist_ok=True)

            # Get common burst limits
            burstoffset, minBurst, maxBurst = reference.getCommonBurstLimits(secondary)

            if overlap:
                maxBurst = maxBurst - 1

            secondary_burst_start = minBurst + burstoffset

            # Process each burst
            for m_burst in range(minBurst, maxBurst):
                s_burst = secondary_burst_start + (m_burst - minBurst)

                logger.info(
                    "Processing burst %d (reference) with burst %d (secondary)",
                    m_burst,
                    s_burst,
                )

                ref_burst = reference.bursts[m_burst]
                sec_burst = secondary.bursts[s_burst]

                # Build offset polynomial and doppler
                azpoly = Poly2D()
                azpoly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.0]])

                rgpoly = Poly2D()
                rgpoly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.0]])

                carrPoly = Poly2D()
                carrPoly.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.0]])

                doppPoly = Poly2D()
                doppPoly.initPoly(
                    rangeOrder=0,
                    azimuthOrder=0,
                    coeffs=[[0.0]],  # Zero doppler for TOPS
                )

                # Define offset file paths
                if overlap:
                    azoff_file = (
                        geo_dir / f"azimuth_{m_burst + 1:02d}_{m_burst + 2:02d}.off"
                    )
                    rgoff_file = (
                        geo_dir / f"range_{m_burst + 1:02d}_{m_burst + 2:02d}.off"
                    )
                    outname = outdir / f"burst_{m_burst + 1:02d}_{m_burst + 2:02d}.slc"
                else:
                    azoff_file = geo_dir / f"azimuth_{m_burst + 1:02d}.off"
                    rgoff_file = geo_dir / f"range_{m_burst + 1:02d}.off"
                    outname = outdir / f"burst_{m_burst + 1:02d}.slc"

                # Resample secondary to reference geometry
                _resamp_secondary_burst(
                    ref_burst,
                    sec_burst,
                    str(azoff_file),
                    str(rgoff_file),
                    azpoly,
                    rgpoly,
                    carrPoly,
                    doppPoly,
                    str(outname),
                    flatten,
                )

        logger.info("Resamp completed successfully")
        return 0
    except Exception:
        logger.exception("Resamp execution failed")
        return 1


def _resamp_secondary_burst(
    ref_burst,
    sec_burst,
    azoff_file: str,
    rgoff_file: str,
    azpoly,
    rgpoly,
    carrpoly,
    dopppoly,
    outname: str,
    flatten: bool,
):
    """Resample a single burst.

    Parameters
    ----------
    ref_burst : object
        Reference burst object.
    sec_burst : object
        Secondary burst object.
    azoff_file : str
        Path to azimuth offset file.
    rgoff_file : str
        Path to range offset file.
    azpoly : Poly2D
        Azimuth offset polynomial.
    rgpoly : Poly2D
        Range offset polynomial.
    carrpoly : Poly2D
        Carrier polynomial.
    dopppoly : Poly2D
        Doppler polynomial.
    outname : str
        Output file path.
    flatten : bool
        Whether to flatten the interferogram.

    """
    # Load offset images
    rngImg = isceobj.createImage()
    rngImg.load(rgoff_file + ".xml")
    rngImg.setAccessMode("READ")

    aziImg = isceobj.createImage()
    aziImg.load(azoff_file + ".xml")
    aziImg.setAccessMode("READ")

    # Load input SLC
    inimg = isceobj.createSlcImage()
    inimg.load(sec_burst.image.filename + ".xml")
    inimg.setAccessMode("READ")

    # Create resampler
    rObj = stdproc.createResamp_slc()
    rObj.slantRangePixelSpacing = sec_burst.rangePixelSize
    rObj.radarWavelength = sec_burst.radarWavelength
    rObj.azimuthCarrierPoly = carrpoly
    rObj.dopplerPoly = dopppoly

    rObj.azimuthOffsetsPoly = azpoly
    rObj.rangeOffsetsPoly = rgpoly
    rObj.imageIn = inimg

    width = ref_burst.numberOfSamples
    length = ref_burst.numberOfLines
    imgOut = isceobj.createSlcImage()
    imgOut.setWidth(width)
    imgOut.filename = outname
    imgOut.setAccessMode("write")

    rObj.outputWidth = width
    rObj.outputLines = length
    rObj.residualRangeImage = rngImg
    rObj.residualAzimuthImage = aziImg
    rObj.flatten = flatten

    # Execute resampling
    rObj.resamp_slc(imageOut=imgOut)

    # Finalize
    imgOut.renderHdr()

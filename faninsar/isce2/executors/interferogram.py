"""Interferogram generation and filtering executors for ISCE2.

This module provides functionality for interferogram generation and coherence
estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import isceobj
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu
from stdproc.stdproc import crossmul

from faninsar.isce2.isce_utils import get_swath_list, load_product
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.isce2.command_manager import Command

logger = setup_logger(__name__)


def _exec_generate_igram(cmd: Command) -> int:
    """Execute interferogram generation from resampled SLCs.

    Parameters
    ----------
    cmd : Command
        GenerateIgram command with parameters:
        - reference: Path to reference directory
        - secondary: Path to secondary directory (resampled)
        - coreg_dir: Path to coregistered directory
        - overlap: Whether this is overlap burst

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Extract parameters
        reference_dir = Path(cmd.params["reference"])
        coreg_dir = Path(cmd.params["coreg_dir"])
        overlap = bool(cmd.params.get("overlap", False))

        # Get swath list
        swath_list = get_swath_list(reference_dir)

        # Process each swath
        for swath in swath_list:
            logger.info("Generating interferogram for swath IW%d", swath)

            # Load products
            reference = load_product(reference_dir / f"IW{swath}.xml")

            # Determine directory structure
            if overlap:
                indir = coreg_dir / "overlap" / f"IW{swath}"
                outdir = coreg_dir / "overlap" / f"IW{swath}"
            else:
                indir = coreg_dir / f"IW{swath}"
                outdir = coreg_dir / f"IW{swath}"

            outdir.mkdir(parents=True, exist_ok=True)

            # Process each burst
            for m_burst, ref_burst in enumerate(reference.bursts):
                logger.info("Generating interferogram for burst %d", m_burst)

                # Define file paths
                if overlap:
                    ref_slc = (
                        reference_dir
                        / f"IW{swath}"
                        / f"burst_{m_burst + 1:02d}_{m_burst + 2:02d}.slc"
                    )
                    sec_slc = indir / f"burst_{m_burst + 1:02d}_{m_burst + 2:02d}.slc"
                    int_file = outdir / f"burst_{m_burst + 1:02d}_{m_burst + 2:02d}.int"
                    amp_file = outdir / f"burst_{m_burst + 1:02d}_{m_burst + 2:02d}.amp"
                else:
                    ref_slc = (
                        reference_dir / f"IW{swath}" / f"burst_{m_burst + 1:02d}.slc"
                    )
                    sec_slc = indir / f"burst_{m_burst + 1:02d}.slc"
                    int_file = outdir / f"burst_{m_burst + 1:02d}.int"
                    amp_file = outdir / f"burst_{m_burst + 1:02d}.amp"

                # Load SLC images
                refImage = isceobj.createSlcImage()
                refImage.load(str(ref_slc) + ".xml")
                refImage.setAccessMode("READ")

                secImage = isceobj.createSlcImage()
                secImage.load(str(sec_slc) + ".xml")
                secImage.setAccessMode("READ")

                # Create output images
                intImage = isceobj.createIntImage()
                intImage.setFilename(str(int_file))
                intImage.setWidth(ref_burst.numberOfSamples)
                intImage.setAccessMode("write")

                ampImage = isceobj.createAmpImage()
                ampImage.setFilename(str(amp_file))
                ampImage.setWidth(ref_burst.numberOfSamples)
                ampImage.setAccessMode("write")

                # Generate interferogram using crossmul
                crossmul(refImage, secImage, intImage, ampImage)

                # Render XML files
                intImage.renderHdr()
                ampImage.renderHdr()

        logger.info("Interferogram generation completed successfully")
        return 0
    except Exception:
        logger.exception("Interferogram generation failed")
        return 1


def _exec_filter_coherence(cmd: Command) -> int:
    """Execute filtering and coherence estimation.

    Parameters
    ----------
    cmd : Command
        FilterAndCoherence command with parameters:
        - interferogram: Path to input interferogram
        - coherence: Path to output coherence file
        - filtered_int: Path to output filtered interferogram
        - filter_strength: Filter strength (0.0-1.0)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    try:
        # Extract parameters
        int_file = Path(cmd.params["interferogram"])
        coh_file = Path(cmd.params["coherence"])
        filt_file = Path(cmd.params["filtered_int"])
        filter_strength = float(cmd.params.get("filter_strength", 0.5))

        logger.info("Filtering interferogram: %s", int_file)
        logger.info("Filter strength: %f", filter_strength)

        # Ensure output directories exist
        filt_file.parent.mkdir(parents=True, exist_ok=True)
        coh_file.parent.mkdir(parents=True, exist_ok=True)

        # Load input interferogram
        intImage = isceobj.createIntImage()
        intImage.load(str(int_file) + ".xml")
        intImage.setAccessMode("READ")
        intImage.createImage()

        # Create filtered interferogram
        filtImage = isceobj.createIntImage()
        filtImage.setFilename(str(filt_file))
        filtImage.setWidth(intImage.getWidth())
        filtImage.setAccessMode("write")
        filtImage.createImage()

        # Apply Goldstein-Werner filter
        objFilter = Filter()
        objFilter.wireInputPort(name="interferogram", object=intImage)
        objFilter.wireOutputPort(name="filtered interferogram", object=filtImage)
        objFilter.goldsteinWerner(alpha=filter_strength)

        intImage.finalizeImage()
        filtImage.finalizeImage()

        # Render filtered interferogram XML
        filtImage.renderHdr()

        # Estimate coherence
        logger.info("Estimating coherence")

        # Reload filtered interferogram for coherence estimation
        filtImage = isceobj.createIntImage()
        filtImage.load(str(filt_file) + ".xml")
        filtImage.setAccessMode("READ")
        filtImage.createImage()

        # Create coherence image
        cohImage = isceobj.createImage()
        cohImage.dataType = "FLOAT"
        cohImage.setFilename(str(coh_file))
        cohImage.setWidth(filtImage.getWidth())
        cohImage.setAccessMode("write")
        cohImage.createImage()

        # Estimate coherence
        icu = Icu(name="insarapp_filter_icu")
        icu.configure()
        icu.wireInputPort(name="interferogram", object=filtImage)
        icu.wireOutputPort(name="correlation", object=cohImage)
        icu.icu(filtImage=filtImage)

        filtImage.finalizeImage()
        cohImage.finalizeImage()

        # Render coherence XML
        cohImage.renderHdr()

        logger.info("Filter and coherence completed successfully")
        return 0
    except Exception:
        logger.exception("Filter and coherence failed")
        return 1

"""Utility functions for ISCE2 processing.

This module provides helper functions that interface with ISCE2 topsStack utilities,
adapted from contrib/stack/topsStack/s1a_isce_utils.py.
"""

from __future__ import annotations

import os
from pathlib import Path

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


def get_swath_list(reference_dir: str | Path) -> list[int]:
    """Get list of available swaths from reference directory.

    Parameters
    ----------
    reference_dir : str | Path
        Directory containing reference IW?.xml files.

    Returns
    -------
    list[int]
        List of swath numbers (e.g., [1, 2, 3] for IW1, IW2, IW3).

    Examples
    --------
    >>> swaths = get_swath_list("/path/to/reference")
    >>> print(swaths)
    [1, 2, 3]

    """
    import glob

    reference_dir = Path(reference_dir)
    swath_files = sorted(glob.glob(str(reference_dir / "IW*.xml")))

    if not swath_files:
        logger.warning("No IW*.xml files found in %s", reference_dir)
        return []

    swaths = []
    for swath_file in swath_files:
        # Extract swath number from IW1.xml, IW2.xml, etc.
        basename = os.path.basename(swath_file)
        if basename.startswith("IW") and basename.endswith(".xml"):
            swath_num = basename[2]  # Get '1' from 'IW1.xml'
            if swath_num.isdigit():
                swaths.append(int(swath_num))

    return sorted(swaths)


def load_product(xml_file: str | Path):
    """Load ISCE2 product from XML file.

    Parameters
    ----------
    xml_file : str | Path
        Path to the XML product file.

    Returns
    -------
    object
        ISCE2 product object.

    Examples
    --------
    >>> product = load_product("/path/to/reference/IW1.xml")

    """
    from iscesys.Component.ProductManager import ProductManager as PM

    pm = PM()
    pm.configure()

    obj = pm.loadProduct(str(xml_file))

    return obj

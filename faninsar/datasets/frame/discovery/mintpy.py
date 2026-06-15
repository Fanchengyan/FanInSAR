"""MintPy product discovery (M4 stub)."""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


class MintPyDiscoverer:
    """Discover MintPy products.

    MintPy produces a single ``inputs/`` directory with an HDF5 mintpy
    stack (e.g. ``timeseriesSet.h5`` or ``velocity.h5``). Geometry is
    typically inside the same HDF5 file; for the frame layout we treat
    the MintPy output directory itself as the "product".
    """

    name = "mintpy"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the first directory holding geometry rasters."""
        root_dir = Path(root_dir)
        # MintPy outputs geometry as a GeoTIFF under inputs/ or the root.
        for pattern in ("geometry*.geo.tif", "inputs/geometry*.tif", "*.geo.tif"):
            m = next(root_dir.glob(pattern), None)
            if m is not None:
                d = m.parent
                logger.debug("Discovered MintPy geometry product: %s", d)
                return d
        msg = f"No MintPy geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of valid pair product directories."""
        root_dir = Path(root_dir)
        # MintPy aggregates pairs in a single HDF5; there are no pair dirs.
        # We return the root if it contains an ifgramStack.h5.
        if next(root_dir.glob("*ifgramStack*.h5"), None) is not None:
            return [root_dir]
        return []


register(MintPyDiscoverer())

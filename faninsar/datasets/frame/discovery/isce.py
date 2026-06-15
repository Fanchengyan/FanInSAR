"""ISCE2/topsStack product discovery (M4 stub).

Recognises an ISCE2 product directory by the presence of an ``IW*.xml``
or ``geometry`` sub-directory. Real implementation will land when ISCE2
sample data is available for testing.
"""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


class ISCEDiscoverer:
    """Discover ISCE2 topsStack geometry products and interferogram pairs."""

    name = "isce"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the first directory holding geometry rasters."""
        root_dir = Path(root_dir)
        # ISCE2 stores geometry under a "geom" or "geometry" subdirectory,
        # with files like lat.rdr, lon.rdr, los.rdr.
        for cand in (root_dir / "geometry", root_dir / "geom"):
            if cand.is_dir() and next(cand.glob("*.rdr"), None) is not None:
                logger.debug("Discovered ISCE geometry product: %s", cand)
                return cand
        msg = f"No ISCE geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of valid pair product directories."""
        root_dir = Path(root_dir)
        pairs: list[Path] = []
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            # ISCE2 pair dirs contain "fine_interferogram"
            # or "filt_topophase.unw".
            if (
                next(d.rglob("filt_topophase.unw"), None) is not None
                or next(d.rglob("fine_interferogram.xml"), None) is not None
            ):
                pairs.append(d)
        return pairs


register(ISCEDiscoverer())

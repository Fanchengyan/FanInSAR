"""StaMPS product discovery (M4 stub)."""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


class StaMPSDiscoverer:
    """Discover StaMPS PS-InSAR products.

    StaMPS stores results in a MATLAB ``.mat`` file and GeoTIFF exports
    under a ``PATCH_*`` directory tree.
    """

    name = "stamps"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the first directory holding geometry rasters."""
        root_dir = Path(root_dir)
        # StaMPS uses look-up rasters (e.g. *.geo / *.lut) produced by
        # the preceding SNAP/ISCE step.
        for pattern in ("*.geo.tif", "*.geo", "geom/*.tif"):
            m = next(root_dir.glob(pattern), None)
            if m is not None:
                logger.debug("Discovered StaMPS geometry product: %s", m.parent)
                return m.parent
        msg = f"No StaMPS geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of valid pair product directories."""
        root_dir = Path(root_dir)
        # StaMPS is a PS method; there are no pair directories. Return the
        # root if it looks like a StaMPS output (contains PATCH_1 or similar).
        if next(root_dir.glob("PATCH_*"), None) is not None:
            return [root_dir]
        return []


register(StaMPSDiscoverer())

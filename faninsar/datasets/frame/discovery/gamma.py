"""GAMMA product discovery (M4 stub)."""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


class GAMMADiscoverer:
    """Discover GAMMA geometry products and interferogram pairs."""

    name = "gamma"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the first directory holding geometry rasters."""
        root_dir = Path(root_dir)
        # GAMMA geometry: files like *.slc.par, *.mli.geo, dem_seg
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if next(d.glob("dem_seg*"), None) is not None:
                logger.debug("Discovered GAMMA geometry product: %s", d)
                return d
        msg = f"No GAMMA geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of valid pair product directories."""
        root_dir = Path(root_dir)
        pairs: list[Path] = []
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            # GAMMA pairs contain unwrapped phase (*.unw)
            if next(d.glob("*.unw"), None) is not None:
                pairs.append(d)
        return pairs


register(GAMMADiscoverer())

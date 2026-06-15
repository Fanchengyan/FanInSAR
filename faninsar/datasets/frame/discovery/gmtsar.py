"""GMTSAR product discovery (M4 stub)."""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


class GMTSARDISCOVERER:
    """Discover GMTSAR geometry products and interferogram pairs."""

    name = "gmtsar"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the first directory holding geometry rasters."""
        root_dir = Path(root_dir)
        # GMTSAR stores geometry as *.grd files under a "merge" or "geom" dir
        for cand in (root_dir / "merge", root_dir / "geom", root_dir):
            if cand.is_dir() and next(cand.glob("*_geom.grd"), None) is not None:
                logger.debug("Discovered GMTSAR geometry product: %s", cand)
                return cand
        msg = f"No GMTSAR geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of valid pair product directories."""
        root_dir = Path(root_dir)
        pairs: list[Path] = []
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            # GMTSAR pairs contain unwrap.grd
            if next(d.glob("unwrap.grd"), None) is not None:
                pairs.append(d)
        return pairs


register(GMTSARDISCOVERER())

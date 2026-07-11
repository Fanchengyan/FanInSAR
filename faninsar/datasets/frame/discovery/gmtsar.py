"""GMTSAR product discovery (M4).

MintPy GMTSAR convention (https://mintpy.readthedocs.io/en/latest/dir_structure/):

```
$DATA_DIR/<project>
├── dem.grd
├── INDIA_*.raw
├── S1_*.raw
├── merged/
│   ├── dem.grd
│   ├── geom_reference/
│   │   ├── lat.rdr
│   │   ├── lon.rdr
│   │   └── hgt.rdr
│   └── interferograms/
│       └── <dates>/
│           ├── unwrap_ll.grd
│           ├── corr_ll.grd
│           └── phasefilt_ll.grd
└── ...
```
"""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


def _has(d: Path, pattern: str) -> bool:
    return next(d.glob(pattern), None) is not None


class GMTSARDiscoverer:
    """Discover GMTSAR geometry products and interferogram pairs."""

    name = "gmtsar"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the directory holding GMTSAR geometry rasters."""
        root_dir = Path(root_dir)

        # Canonical MintPy GMTSAR layout: merged/geom_reference/
        cand = root_dir / "merged" / "geom_reference"
        if cand.is_dir() and (_has(cand, "lat.rdr") or _has(cand, "lat.grd")):
            logger.debug("Discovered GMTSAR geometry (merged/geom_reference): %s", cand)
            return cand

        # merged/dem.grd
        cand = root_dir / "merged"
        if cand.is_dir() and _has(cand, "dem.grd"):
            logger.debug("Discovered GMTSAR geometry (merged/dem.grd): %s", cand)
            return cand

        # Top-level dem.grd
        if _has(root_dir, "dem.grd"):
            logger.debug("Discovered GMTSAR geometry (top-level dem.grd): %s", root_dir)
            return root_dir

        # Fallback: *_geom.grd
        for cand in (root_dir / "merge", root_dir / "geom", root_dir):
            if cand.is_dir() and _has(cand, "*_geom.grd"):
                logger.debug("Discovered GMTSAR geometry (fallback): %s", cand)
                return cand

        msg = f"No GMTSAR geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of valid GMTSAR interferogram pair directories."""
        root_dir = Path(root_dir)
        pairs: list[Path] = []

        # Canonical: merged/interferograms/<dates>/unwrap_ll.grd
        ifgs = root_dir / "merged" / "interferograms"
        if ifgs.is_dir():
            pairs.extend(
                d
                for d in sorted(ifgs.iterdir())
                if d.is_dir() and _has(d, "unwrap_ll.grd")
            )
            if pairs:
                return pairs

        # Fallback: any subdir with unwrap.grd / unwrap_ll.grd
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if _has(d, "unwrap.grd") or _has(d, "unwrap_ll.grd"):
                pairs.append(d)

        return pairs


register(GMTSARDiscoverer())

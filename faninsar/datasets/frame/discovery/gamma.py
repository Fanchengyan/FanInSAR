"""GAMMA product discovery (M4).

MintPy GAMMA convention (https://mintpy.readthedocs.io/en/latest/dir_structure/):

```
$DATA_DIR/<project>
├── geometry/
│   ├── 20050619_20070809.sim_*.rdc.dem
│   ├── *.geo.lat
│   ├── *.geo.lon
│   ├── *.geo.hgt
│   └── *.los
├── interferograms/
│   └── <dates>/
│       ├── diff_*.lat_*.unw
│       └── ...
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


class GAMMADiscoverer:
    """Discover GAMMA geometry products and interferogram pairs."""

    name = "gamma"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the directory holding GAMMA geometry rasters."""
        root_dir = Path(root_dir)

        # Canonical MintPy GAMMA layout: <root>/geometry/*.rdc.dem
        # File names like: 20050619_20070809.sim_01.rdc.dem
        geom = root_dir / "geometry"
        if geom.is_dir() and (
            _has(geom, "*.rdc.dem")
            or _has(geom, "*.geo.lat")
            or _has(geom, "*.geo.hgt")
        ):
            logger.debug("Discovered GAMMA geometry: %s", geom)
            return geom

        # Fallback: flat layout with dem_seg / sim_*.dem in any subdir
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if _has(d, "dem_seg*") or _has(d, "sim_*.dem"):
                logger.debug("Discovered GAMMA geometry (flat): %s", d)
                return d

        msg = f"No GAMMA geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of valid GAMMA interferogram pair directories."""
        root_dir = Path(root_dir)
        pairs: list[Path] = []

        # Canonical: interferograms/<dates>/ with diff_*.unw
        ifgs = root_dir / "interferograms"
        if ifgs.is_dir():
            pairs.extend(
                d
                for d in sorted(ifgs.iterdir())
                if d.is_dir() and _has(d, "diff_*.unw")
            )
            if pairs:
                return pairs

        # Fallback: any subdir with *.unw
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if _has(d, "*.unw"):
                pairs.append(d)

        return pairs


register(GAMMADiscoverer())

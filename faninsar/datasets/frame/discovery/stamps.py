"""StaMPS product discovery (M4).

StaMPS PS-InSAR convention (https://mtin-g.github.io/StaMPS/):

```
$WORKDIR/<project>
├── PATCH_1/
│   └── ...                         # per-patch MATLAB .mat + rsc files
├── PATCH_2/
├── ...
├── ps_plot_v-d.h5                  # velocity export
├── ps_plot_ts.h5                   # time series export
├── pm-c.shp / pm-c.h5              # coherent PS mask
└── ...
```

StaMPS is a persistent-scatterer method: there are no interferogram pair
directories. Pair/date information lives inside the ``.mat`` workspaces
(``PATCH_*/``) and HDF5 exports (``ps_plot_*.h5``).
"""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


def _has(d: Path, pattern: str) -> bool:
    return next(d.glob(pattern), None) is not None


class StaMPSDiscoverer:
    """Discover StaMPS PS-InSAR products."""

    name = "stamps"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the directory holding StaMPS geometry / look-up rasters.

        StaMPS itself does not re-generate geometry; it inherits
        look-up rasters (``*.geo`` / ``*.lut`` / ``*.rdr``) from the
        preceding SNAP / ISCE / GAMMA step. We look for these either at the
        top level or under a ``geom/`` / ``geometry/`` subdir.
        """
        root_dir = Path(root_dir)

        # Snap/StaMPS exports: *.geo.tif, *.geo
        for cand in (root_dir / "geometry", root_dir / "geom", root_dir):
            if not cand.is_dir():
                continue
            for pattern in ("*.geo.tif", "*.geo", "*.lut", "lat.rdr"):
                if _has(cand, pattern):
                    logger.debug("Discovered StaMPS geometry (%s): %s", pattern, cand)
                    return cand

        msg = f"No StaMPS geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of StaMPS patch/product entries.

        StaMPS organises results in ``PATCH_*`` subdirectories. There are no
        interferogram pair directories; we return the root if it contains
        ``PATCH_*`` or ``ps_plot_*.h5`` exports.
        """
        root_dir = Path(root_dir)

        # PATCH_* subdirectories (StaMPS workspaces)
        patches = sorted(root_dir.glob("PATCH_*"))
        if patches:
            logger.debug("Discovered %d StaMPS patches: %s", len(patches), root_dir)
            return [root_dir]

        # HDF5 exports (ps_plot_v-d.h5, ps_plot_ts.h5)
        if _has(root_dir, "ps_plot_*.h5"):
            logger.debug("Discovered StaMPS ps_plot exports: %s", root_dir)
            return [root_dir]

        return []


register(StaMPSDiscoverer())

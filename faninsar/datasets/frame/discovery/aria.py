"""ARIA (NASA JPL) product discovery (M4).

ARIA produces standard Sentinel-1 GUNW (Geocoded UNWrapped) products
distributed via AWS Open Data and the ARIA-tools ``ariaExtract`` /
``ariaTSsetup`` workflows. A typical ARIA-tools time-series working
directory looks like:

```
$WORKDIR/<project>
├── stack/
│   ├── coh/
│   │   └── *_coh.vrt
│   ├── connComp/
│   │   └── *_connComp.vrt
│   ├── unwrappedPhase/
│   │   └── *_unwrappedPhase.vrt
│   └── ...
├── products/
│   └── S1-GUNW-D-R-*.nc        # ARIA GUNW NetCDF products
├── DEM/
│   └── dem.vrt
├── incidence/
│   └── *_incidence.vrt
├── azimuth/
│   └── *_azimuth.vrt
├── mask/
│   └── watermask.vrt
└── ...
```

The frame-layout promotion step treats ``stack/`` as the interferogram
collection and the ``DEM/`` / ``incidence/`` / ``azimuth/`` / ``mask/``
subdirectories as geometry sources.
"""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


def _has(d: Path, pattern: str) -> bool:
    return next(d.glob(pattern), None) is not None


class ARIADiscoverer:
    """Discover ARIA-tools time-series working directory products."""

    name = "aria"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the directory holding ARIA geometry rasters.

        ARIA-tools writes geometry as per-layer VRT stacks under
        ``incidence/``, ``azimuth/``, ``DEM/`` and ``mask/``.
        """
        root_dir = Path(root_dir)

        # incidence/ or azimuth/ (look-angle layers)
        for name in ("incidence", "azimuth"):
            cand = root_dir / name
            if cand.is_dir() and (
                _has(cand, "*_incidence.vrt") or _has(cand, "*_azimuth.vrt")
            ):
                logger.debug("Discovered ARIA geometry (%s): %s", name, cand)
                return cand

        # DEM/
        cand = root_dir / "DEM"
        if cand.is_dir() and _has(cand, "dem*.vrt"):
            logger.debug("Discovered ARIA geometry (DEM): %s", cand)
            return cand

        # Top-level GUNW NetCDF products
        if _has(root_dir, "S1-GUNW-*.nc"):
            logger.debug("Discovered ARIA GUNW products: %s", root_dir)
            return root_dir

        msg = f"No ARIA geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of ARIA interferogram layers.

        ARIA-tools writes a VRT stack per layer under ``stack/``:
        ``stack/unwrappedPhase/``, ``stack/coh/``, ``stack/connComp/``.
        We return the layer directories (one entry per layer).
        """
        root_dir = Path(root_dir)
        pairs: list[Path] = []

        stack = root_dir / "stack"
        if stack.is_dir():
            for layer in ("unwrappedPhase", "coh", "connComp"):
                cand = stack / layer
                if cand.is_dir() and _has(cand, "*_" + layer + ".vrt"):
                    pairs.append(cand)
            if pairs:
                return pairs

        # Fallback: top-level GUNW products (one .nc per pair)
        gunws = sorted(root_dir.glob("S1-GUNW-*.nc"))
        if gunws:
            return [root_dir]

        return pairs


register(ARIADiscoverer())

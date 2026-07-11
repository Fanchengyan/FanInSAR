"""MintPy product discovery (M4).

MintPy convention (https://mintpy.readthedocs.io/en/latest/dir_structure/):

```
$DATA_DIR/<project>
├── inputs/
│   ├── ifgramStack.h5
│   └── geometryGeo.h5
├── velocity.h5
├── timeseries.h5
├── timeseriesRg.h5
└── ...
```

MintPy is the downstream analysis layer: it ingests an ISCE/GAMMA/GMTSAR
product and writes HDF5 stacks under ``inputs/``. For the frame layout we
treat the MintPy project directory as the "product"; pair information lives
inside ``inputs/ifgramStack.h5`` (not as separate directories).
"""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


def _has(d: Path, pattern: str) -> bool:
    return next(d.glob(pattern), None) is not None


class MintPyDiscoverer:
    """Discover MintPy analysis products (HDF5 stacks)."""

    name = "mintpy"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the directory/file holding MintPy geometry.

        MintPy geometry is stored as ``inputs/geometryGeo.h5`` (or
        ``geometryRadar.h5``). Some exports also write
        ``geometryGeo.tif`` / ``*.geo.tif`` GeoTIFFs.
        """
        root_dir = Path(root_dir)

        # HDF5 geometry (canonical)
        inputs = root_dir / "inputs"
        if inputs.is_dir() and (
            _has(inputs, "geometryGeo.h5")
            or _has(inputs, "geometryRadar.h5")
            or _has(inputs, "geometry*.h5")
        ):
            logger.debug("Discovered MintPy geometry (inputs/geometry*.h5): %s", inputs)
            return inputs

        # GeoTIFF exports (optional)
        for pattern in ("geometryGeo.tif", "*.geo.tif", "*.geo.h5"):
            m = next(root_dir.glob(pattern), None)
            if m is not None:
                d = m.parent
                logger.debug("Discovered MintPy geometry (%s): %s", pattern, d)
                return d

        msg = f"No MintPy geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of MintPy pair/product entries.

        MintPy aggregates all interferograms in a single
        ``inputs/ifgramStack.h5``; there are no per-pair directories. We
        return the parent directory of the HDF5 stack so downstream code can
        locate it.
        """
        root_dir = Path(root_dir)

        # Canonical: inputs/ifgramStack.h5
        inputs = root_dir / "inputs"
        if inputs.is_dir() and _has(inputs, "ifgramStack.h5"):
            logger.debug("Discovered MintPy ifgramStack.h5: %s", inputs)
            return [inputs]

        # Fallback: top-level ifgramStack.h5
        if _has(root_dir, "ifgramStack.h5") or _has(root_dir, "*ifgramStack*.h5"):
            logger.debug("Discovered MintPy ifgramStack.h5 (top-level): %s", root_dir)
            return [root_dir]

        return []


register(MintPyDiscoverer())

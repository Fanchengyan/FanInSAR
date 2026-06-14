"""HyP3 GAMMA product discovery."""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


class HyP3Discoverer:
    """Discover HyP3 GAMMA geometry products and interferogram pairs.

    A HyP3 product directory is recognised by the presence of a file
    matching ``*_inc_map_ell.tif``. Directories whose name starts with ``.``
    are skipped.

    """

    name = "hyp3"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Find the first HyP3 product directory containing geometry rasters.

        Parameters
        ----------
        root_dir : str or Path
            Directory that contains one or more HyP3 product folders.

        Returns
        -------
        Path
            Path to the first product directory that holds geometry rasters.

        Raises
        ------
        FileNotFoundError
            If no matching product directory is found.

        """
        root_dir = Path(root_dir)
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if next(d.glob("*_inc_map_ell.tif"), None) is not None:
                logger.debug("Discovered HyP3 geometry product: %s", d)
                return d

        msg = f"No HyP3 product with *_inc_map_ell.tif found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return all HyP3 product directories under *root_dir*.

        A directory qualifies as a pair product if it contains a file
        matching ``*_unw_phase.tif``.

        Parameters
        ----------
        root_dir : str or Path
            Directory that contains one or more HyP3 product folders.

        Returns
        -------
        list of Path
            Sorted list of matching product directories.

        """
        root_dir = Path(root_dir)
        pairs: list[Path] = []
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if next(d.glob("*_unw_phase.tif"), None) is not None:
                pairs.append(d)
        return pairs


# Self-register on import.
register(HyP3Discoverer())


# Module-level convenience function preserved for backwards compatibility with
# the old ``discovery.py`` single-file module.
def discover_hyp3_geometry_product(root_dir: str | Path) -> Path:
    """Find the first HyP3 product directory containing geometry rasters.

    Backwards-compat shim that delegates to the registered
    :class:`HyP3Discoverer`.

    """
    return HyP3Discoverer().discover_geometry_product(root_dir)

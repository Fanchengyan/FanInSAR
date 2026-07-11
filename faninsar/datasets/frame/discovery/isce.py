"""ISCE2 product discovery (M4).

Recognises three ISCE2 stack flavours per the MintPy directory convention
(https://mintpy.readthedocs.io/en/latest/dir_structure/):

- **topsStack** (Sentinel-1 TOPS):
  ``merged/geom_reference/{lat,lon,hgt,los,shadowMask}.rdr`` +
  ``merged/interferograms/<dates>/filt_*.unw``
- **stripmapStack**:
  ``geom_reference/{lat,lon,hgt,los}.rdr`` +
  ``Igrams/<dates>/filt_*_snaphu.unw``
- **alosStack** (ALOS PALSAR):
  ``dates_res*/<ref_date>/insar/*.{hgt,lat,lon,los,wbd}`` +
  ``pairs/*-*/insar/filt_*.unw``
"""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

from . import register

logger = setup_logger(__name__)


def _has(d: Path, pattern: str, *, rglob: bool = False) -> bool:
    """Return True if *pattern* matches at least one file under *d*."""
    it = d.rglob(pattern) if rglob else d.glob(pattern)
    return next(it, None) is not None


class ISCEDiscoverer:
    """Discover ISCE2 (topsStack / stripmapStack / alosStack) products."""

    name = "isce"

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the directory holding ISCE2 geometry rasters."""
        root_dir = Path(root_dir)

        # topsStack: merged/geom_reference/lat.rdr
        cand = root_dir / "merged" / "geom_reference"
        if cand.is_dir() and _has(cand, "lat.rdr"):
            logger.debug("Discovered ISCE topsStack geometry: %s", cand)
            return cand

        # stripmapStack: geom_reference/lat.rdr
        cand = root_dir / "geom_reference"
        if cand.is_dir() and _has(cand, "lat.rdr"):
            logger.debug("Discovered ISCE stripmapStack geometry: %s", cand)
            return cand

        # alosStack: dates_res*/<ref>/insar/*.los
        for dr in sorted(root_dir.glob("dates_res*")):
            if not dr.is_dir():
                continue
            for date_dir in sorted(dr.iterdir()):
                insar = date_dir / "insar"
                if insar.is_dir() and _has(insar, "*.los"):
                    logger.debug("Discovered ISCE alosStack geometry: %s", insar)
                    return insar

        msg = f"No ISCE geometry product found in {root_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return the list of ISCE2 interferogram pair directories."""
        root_dir = Path(root_dir)
        pairs: list[Path] = []

        # topsStack: merged/interferograms/<dates>/filt_*.unw
        ifgs_root = root_dir / "merged" / "interferograms"
        if ifgs_root.is_dir():
            pairs.extend(
                d
                for d in sorted(ifgs_root.iterdir())
                if d.is_dir() and _has(d, "filt_*.unw")
            )
            if pairs:
                return pairs

        # stripmapStack: Igrams/<dates>/filt_*_snaphu.unw
        igrams = root_dir / "Igrams"
        if igrams.is_dir():
            pairs.extend(
                d
                for d in sorted(igrams.iterdir())
                if d.is_dir() and _has(d, "filt_*_snaphu.unw")
            )
            if pairs:
                return pairs

        # alosStack: pairs/*-*/insar/filt_*.unw
        pairs_root = root_dir / "pairs"
        if pairs_root.is_dir():
            for d in sorted(pairs_root.iterdir()):
                insar = d / "insar"
                if insar.is_dir() and _has(insar, "filt_*.unw"):
                    pairs.append(d)
            if pairs:
                return pairs

        return pairs


register(ISCEDiscoverer())

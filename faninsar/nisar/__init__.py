"""NISAR RSLC/GSLC adapter stubs validating dual-coordinate contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from faninsar.logging import setup_logger
from faninsar.processing.errors import ProcessingContractError

logger = setup_logger(__name__)

NisarProductKind = Literal["RSLC", "GSLC"]


class NisarProductError(ProcessingContractError):
    """Raised when a NISAR product cannot be opened or validated."""


@dataclass(frozen=True, slots=True)
class NisarProductHandle:
    """Minimal typed handle for a NISAR RSLC/GSLC product."""

    path: Path
    kind: NisarProductKind
    frequency: str
    polarization: str
    coordinate_branch: Literal["radar", "geo"]


def open_nisar_product(
    path: str | Path,
    *,
    kind: NisarProductKind,
    frequency: str = "A",
    polarization: str = "HH",
) -> NisarProductHandle:
    """Open a NISAR product path and select the dual-coordinate branch.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a NISAR HDF5 product (validated for existence only in MVP).
    kind : {"RSLC", "GSLC"}
        Product kind. RSLC uses the radar branch; GSLC uses the geo branch.
    frequency : str, optional
        Frequency group identity (for example ``A``).
    polarization : str, optional
        Polarization channel.

    Returns
    -------
    NisarProductHandle
        Typed handle that does not leak Sentinel-1 TOPS metadata.

    Raises
    ------
    NisarProductError
        If the path is missing or the kind is unsupported.

    Notes
    -----
    Full HDF5 group parsing is intentionally deferred. This adapter freezes the
    mission-neutral branch selection contract used by dual-coordinate pipelines.

    """
    product_path = Path(path)
    if not product_path.exists():
        message = f"NISAR product does not exist: {product_path}"
        logger.error(message)
        raise NisarProductError(message)
    if kind not in ("RSLC", "GSLC"):
        message = f"unsupported NISAR product kind: {kind}"
        logger.error(message)
        raise NisarProductError(message)
    if not frequency or not polarization:
        message = "NISAR frequency and polarization must be non-empty"
        logger.error(message)
        raise NisarProductError(message)

    branch: Literal["radar", "geo"] = "radar" if kind == "RSLC" else "geo"
    logger.info(
        "Opened NISAR %s handle on %s branch (freq=%s pol=%s)",
        kind,
        branch,
        frequency,
        polarization,
    )
    return NisarProductHandle(
        path=product_path,
        kind=kind,
        frequency=frequency,
        polarization=polarization,
        coordinate_branch=branch,
    )

"""Persisted NESD misregistration product of a Stack (PROPOSAL-0059).

One product at ``<work_dir>/misreg`` records the measured pair arcs, the
inverted per-date constants, and whether NESD was skipped because the burst
selection had no overlap windows. Everything is metadata, so the product
has no payload assets; secondary scene products bind its ``write_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from faninsar.io.storage.product import commit_product, open_product
from faninsar.processing.coregistration.misreg_network import DateMisreg, MisregArc

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.io.storage.product import Product

__all__ = [
    "MISREG_KIND",
    "MisregRecord",
    "read_misreg",
    "write_misreg",
]

MISREG_KIND = "misreg"


@dataclass(frozen=True, slots=True)
class MisregRecord:
    """Arcs, constants, and NESD outcome restored from the misreg product.

    Attributes
    ----------
    write_id : str
        Product write id that coregistered secondaries bind.
    arcs : list[MisregArc]
        Measured pair arcs; omitted components stay ``None``.
    date_misreg : DateMisreg or None
        Inverted per-date constants, ``None`` before inversion.
    skipped_no_windows : bool
        Whether NESD was skipped because no overlap windows existed.

    """

    write_id: str
    arcs: list[MisregArc]
    date_misreg: DateMisreg | None
    skipped_no_windows: bool


def _optional_shift(value: object) -> float | None:
    """Return ``None`` for an omitted component, else a float."""
    return None if value is None else float(value)  # type: ignore[arg-type]


def _arc_record(arc: MisregArc) -> dict[str, object]:
    return {
        "primary": arc.primary,
        "secondary": arc.secondary,
        "azimuth_shift_px": _optional_shift(arc.azimuth_shift_px),
        "range_shift_px": _optional_shift(arc.range_shift_px),
        "azimuth_sigma_px": float(arc.azimuth_sigma_px),
        "range_sigma_px": float(arc.range_sigma_px),
        "method": arc.method,
        "n_valid": int(arc.n_valid),
    }


def _arc_from_record(row: dict[str, object]) -> MisregArc:
    return MisregArc(
        primary=str(row["primary"]),
        secondary=str(row["secondary"]),
        azimuth_shift_px=_optional_shift(row.get("azimuth_shift_px")),
        range_shift_px=_optional_shift(row.get("range_shift_px")),
        azimuth_sigma_px=float(row.get("azimuth_sigma_px", 1.0)),  # type: ignore[arg-type]
        range_sigma_px=float(row.get("range_sigma_px", 1.0)),  # type: ignore[arg-type]
        method=str(row.get("method", "unknown")),
        n_valid=int(row.get("n_valid", 0)),  # type: ignore[call-overload]
    )


def write_misreg(
    root: str | Path,
    *,
    arcs: list[MisregArc],
    date_misreg: DateMisreg | None,
    skipped_no_windows: bool,
) -> Product:
    """Commit the misreg product, replacing any previous one.

    Re-committing semantically identical state keeps the existing product
    and its ``write_id``: ``invert_misreg`` re-persists restored arcs and
    constants in every process, and a fresh random write id there would
    mark every secondary scene stale and cascade into stale
    interferograms, breaking cross-process resume.

    Parameters
    ----------
    root : path-like
        Product directory, ``<work_dir>/misreg``.
    arcs : list of MisregArc
        Measured pair arcs.
    date_misreg : DateMisreg or None
        Inverted per-date constants, if inverted.
    skipped_no_windows : bool
        NESD skip outcome.

    Returns
    -------
    Product
        The committed product.

    """
    constants = None
    if date_misreg is not None:
        constants = {
            "reference": date_misreg.reference,
            "azimuth_px": {k: float(v) for k, v in date_misreg.azimuth_px.items()},
            "range_px": {k: float(v) for k, v in date_misreg.range_px.items()},
            "metadata": dict(date_misreg.metadata),
        }
    properties = {
        "faninsar:arcs": [_arc_record(arc) for arc in arcs],
        "faninsar:date_misreg": constants,
        "faninsar:nesd_skipped_no_windows": bool(skipped_no_windows),
    }
    existing = _identical_existing_product(root, properties)
    if existing is not None:
        return existing
    return commit_product(
        root,
        kind=MISREG_KIND,
        properties=properties,
        write_payloads=lambda _staging: {},
        overwrite=True,
    )


def _identical_existing_product(
    root: str | Path, properties: dict[str, object]
) -> Product | None:
    """Return the committed misreg product when its state is identical.

    An unreadable or foreign product is reported as absent so the caller
    rewrites (repairs) it instead of reusing it.
    """
    from pathlib import Path

    from faninsar.processing.errors import InvalidProcessingStateError

    if not (Path(root) / "item.json").is_file():
        return None
    try:
        product = open_product(root, kind=MISREG_KIND)
    except InvalidProcessingStateError:
        return None
    if all(product.properties.get(key) == value for key, value in properties.items()):
        return product
    return None


def read_misreg(root: str | Path) -> MisregRecord | None:
    """Return the persisted misreg record, or ``None`` when absent.

    Raises
    ------
    InvalidProcessingStateError
        If the directory holds an unreadable or foreign product.

    """
    from pathlib import Path

    if not (Path(root) / "item.json").is_file():
        return None
    product = open_product(root, kind=MISREG_KIND)
    properties = product.properties
    constants = properties.get("faninsar:date_misreg")
    date_misreg = None
    if isinstance(constants, dict):
        date_misreg = DateMisreg(
            reference=str(constants["reference"]),
            azimuth_px={k: float(v) for k, v in constants["azimuth_px"].items()},
            range_px={k: float(v) for k, v in constants["range_px"].items()},
            metadata=dict(constants.get("metadata", {})),
        )
    return MisregRecord(
        write_id=product.write_id,
        arcs=[_arc_from_record(row) for row in properties.get("faninsar:arcs", [])],
        date_misreg=date_misreg,
        skipped_no_windows=bool(properties.get("faninsar:nesd_skipped_no_windows")),
    )

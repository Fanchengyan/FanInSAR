"""Physical product type lattice for workflow composition checks."""

from __future__ import annotations

from enum import StrEnum


class PhysicalType(StrEnum):
    """Seven-member lattice of InSAR product physical types.

    Used by ``Workflow.check()`` and ``assert_token`` to validate stage
    composition. Values are stable string identifiers for YAML/config.
    """

    SLC_RAW = "slc_raw"
    SLC_DERAMPED = "slc_deramped"
    SLC_COREG = "slc_coreg"
    IFG_COMPLEX = "ifg_complex"
    IFG_FLATTENED = "ifg_flattened"
    PHASE_UNWRAPPED = "phase_unwrapped"
    DISPLACEMENT = "displacement"


PHYSICAL_TYPE_MEMBERS: frozenset[PhysicalType] = frozenset(PhysicalType)

# Sequential stage transitions for DEFAULT_PAIR_STAGES lattice check.
SEQ_TRANSITIONS: dict[PhysicalType, PhysicalType] = {
    PhysicalType.SLC_RAW: PhysicalType.SLC_DERAMPED,
    PhysicalType.SLC_DERAMPED: PhysicalType.SLC_COREG,
    PhysicalType.SLC_COREG: PhysicalType.IFG_COMPLEX,
    PhysicalType.IFG_COMPLEX: PhysicalType.IFG_FLATTENED,
    PhysicalType.IFG_FLATTENED: PhysicalType.PHASE_UNWRAPPED,
    PhysicalType.PHASE_UNWRAPPED: PhysicalType.DISPLACEMENT,
}


def is_complete_lattice(members: set[PhysicalType] | frozenset[PhysicalType]) -> bool:
    """Return True if *members* is exactly the seven-type lattice."""
    return frozenset(members) == PHYSICAL_TYPE_MEMBERS


__all__ = [
    "PHYSICAL_TYPE_MEMBERS",
    "SEQ_TRANSITIONS",
    "PhysicalType",
    "is_complete_lattice",
]

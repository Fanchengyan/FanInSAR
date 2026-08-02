"""PhysicalType lattice exhaustiveness tests."""

from __future__ import annotations

from faninsar.core.physical import (
    PHYSICAL_TYPE_MEMBERS,
    PhysicalType,
    is_complete_lattice,
)


def test_seven_members() -> None:
    assert len(PhysicalType) == 7
    assert len(PHYSICAL_TYPE_MEMBERS) == 7


def test_member_names() -> None:
    expected = {
        "slc_raw",
        "slc_deramped",
        "slc_coreg",
        "ifg_complex",
        "ifg_flattened",
        "phase_unwrapped",
        "displacement",
    }
    assert {m.value for m in PhysicalType} == expected


def test_is_complete_lattice() -> None:
    assert is_complete_lattice(set(PhysicalType))
    assert not is_complete_lattice({PhysicalType.SLC_RAW})

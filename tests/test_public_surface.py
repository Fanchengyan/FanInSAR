"""Public surface gates: curated ``__all__`` ≤ 20 names."""

from __future__ import annotations

import faninsar as fis

REQUIRED_NAMES = {
    "Acquisition",
    "Acquisitions",
    "Baselines",
    "Frequency",
    "Network",
    "Pair",
    "Pairs",
    "Wavelength",
    "data",
    "network",
    "remote",
    "stack",
}


def test_all_is_exact_scientific_surface() -> None:
    assert set(fis.__all__) == REQUIRED_NAMES


def test_required_names_present() -> None:
    missing = REQUIRED_NAMES - set(fis.__all__)
    assert not missing, f"missing public names: {sorted(missing)}"


def test_removed_execution_names_absent() -> None:
    """Stack and Network are the only public interferometry roots."""
    for name in ("Frame", "Pipeline", "Workflow", "run", "run_pair"):
        assert name not in fis.__all__
        assert not hasattr(fis, name)


def test_ports_not_in_root_all() -> None:
    assert "ports" not in fis.__all__
    assert "ComputeBackend" not in fis.__all__
    assert "SensorAdapter" not in fis.__all__
    assert "IOBackend" not in fis.__all__


def test_pairs_importable() -> None:
    assert hasattr(fis, "Pairs")
    assert fis.Pairs is not None


def test_concrete_stacks_are_mission_entry_points() -> None:
    """Supported concrete Stacks are imported from the mission facade."""
    from faninsar.missions import NISARStack, S1Stack
    from faninsar.missions.nisar import NISARStack as NISARPackageStack
    from faninsar.missions.s1 import S1Stack as S1PackageStack

    assert S1Stack is S1PackageStack
    assert NISARStack is NISARPackageStack
    assert not hasattr(fis.stack, "S1Stack")
    assert not hasattr(fis.stack, "NISARStack")


def test_mission_neutral_stack_has_no_source_constructor() -> None:
    """Raw-source constructors remain on concrete mission adapters only."""
    from faninsar.stack import Stack

    assert not hasattr(Stack, "from_safes")

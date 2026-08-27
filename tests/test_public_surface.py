"""Public surface gates: curated ``__all__`` ≤ 20 names."""

from __future__ import annotations

import faninsar as fis


REQUIRED_NAMES = {
    "Pair",
    "Pairs",
    "Loops",
    "Acquisition",
    "Baselines",
    "Network",
    "Stack",
    "S1Stack",
    "NISARStack",
    "invert",
    "NSBAS",
    "SBAS",
    "Stage",
    "open_stac",
    "open_zarr",
    "register",
    "list_missions",
}


def test_all_length_at_most_20() -> None:
    assert len(fis.__all__) <= 20


def test_required_names_present() -> None:
    missing = REQUIRED_NAMES - set(fis.__all__)
    assert not missing, f"missing public names: {sorted(missing)}"


def test_removed_execution_names_absent() -> None:
    """Stack and Network are the only public interferometry roots."""
    for name in ("Frame", "Pipeline", "Workflow", "run"):
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

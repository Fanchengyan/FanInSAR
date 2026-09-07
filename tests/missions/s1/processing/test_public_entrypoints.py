"""Public production-stage entry-point regression tests."""

from __future__ import annotations

from faninsar.missions.s1 import processing as stages


def test_s1_processing_exports_have_no_pair_execution_entry_points() -> None:
    """The package surface must not expose removed pair/run wrappers."""
    removed = {
        "run_pair",
        "run_pair_pipeline",
        "run_pair_workflow",
        "PairWorkflowState",
        "run_stack_pipeline",
    }
    assert removed.isdisjoint(getattr(stages, "__all__", ()))
    for name in removed:
        assert not hasattr(stages, name)


def test_legacy_processing_modules_are_removed() -> None:
    """The old pair and stack execution modules are no longer importable."""
    import importlib.util

    assert importlib.util.find_spec("faninsar.processing.pipeline") is None

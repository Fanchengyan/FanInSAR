"""Public pipeline entry-point regression tests."""

from __future__ import annotations

from faninsar.processing import pipeline


def test_pipeline_exports_have_no_pair_execution_entry_points() -> None:
    """The package surface must not expose removed pair/run wrappers."""
    removed = {
        "run_pair",
        "run_pair_pipeline",
        "run_pair_workflow",
        "PairWorkflowState",
        "run_stack_pipeline",
    }
    assert removed.isdisjoint(pipeline.__all__)
    for name in removed:
        assert not hasattr(pipeline, name)


def test_legacy_pipeline_modules_are_removed() -> None:
    """The old pair and stack execution modules are no longer importable."""
    import importlib.util

    for name in (
        "faninsar.processing.pipeline.pair_pipeline",
        "faninsar.processing.pipeline.stack_pipeline",
    ):
        assert importlib.util.find_spec(name) is None

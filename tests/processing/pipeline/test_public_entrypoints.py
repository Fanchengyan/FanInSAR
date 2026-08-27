"""Public pipeline entry-point regression tests."""

from __future__ import annotations

import ast
from pathlib import Path

from faninsar.processing import pipeline
from faninsar.processing.pipeline import pair_pipeline, stack_pipeline, workflow


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


def test_legacy_module_wrappers_are_private() -> None:
    """Legacy module implementations remain internal-only compatibility code."""
    assert not hasattr(pair_pipeline, "run_pair_pipeline")
    assert not hasattr(workflow, "run_pair_workflow")
    assert not hasattr(stack_pipeline, "run_stack_pipeline")
    assert hasattr(pair_pipeline, "_run_pair_pipeline")
    assert hasattr(workflow, "_run_pair_workflow")
    assert hasattr(stack_pipeline, "_run_stack_pipeline")


def test_legacy_execution_names_are_not_defined_as_public_functions() -> None:
    """Static checks prevent accidental reintroduction of module wrappers."""
    package_root = Path(__file__).parents[3]
    modules = {
        "pair_pipeline.py": "run_pair_pipeline",
        "workflow.py": "run_pair_workflow",
        "stack_pipeline.py": "run_stack_pipeline",
    }
    for filename, function_name in modules.items():
        module_path = package_root / "faninsar" / "processing" / "pipeline" / filename
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        public_defs = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        }
        assert function_name not in public_defs

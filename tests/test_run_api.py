"""fis.run front-door smoke tests."""

from __future__ import annotations

import subprocess
import sys

import pytest

import faninsar
from faninsar.processing.errors import PairConfigurationMigrationError


def test_run_requires_stack_paths() -> None:
    """The config facade requires an explicit Stack source collection."""
    from faninsar.run import run

    with pytest.raises(ValueError, match="paths"):
        run({})


def test_root_has_no_third_execution_entry_point() -> None:
    """The root package exposes Stack/Network, not a module-level runner."""
    result = subprocess.run(
        [sys.executable, "-c", "import faninsar; assert not hasattr(faninsar, 'run')"],
        check=False,
    )
    assert result.returncode == 0


def test_run_rejects_pair_configuration_before_backend_resolution() -> None:
    """Legacy reference/secondary fields fail with a typed migration error."""
    from faninsar.run import run

    with pytest.raises(PairConfigurationMigrationError, match="paths"):
        run({"reference": "ref.SAFE", "secondary": "sec.SAFE"})


def test_processing_pipeline_hides_removed_pair_entry_points() -> None:
    """Removed execution callables are absent from the public pipeline."""
    from faninsar.processing import pipeline

    for name in (
        "run_pair",
        "run_pair_pipeline",
        "run_pair_workflow",
        "PairWorkflowState",
        "run_stack_pipeline",
    ):
        assert not hasattr(pipeline, name)
        assert name not in pipeline.__all__

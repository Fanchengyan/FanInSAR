"""Stack-only execution boundary tests for the frame command."""

from __future__ import annotations

import pytest

from faninsar.cli.frame import run_frame_cli
from faninsar.processing.errors import PairConfigurationMigrationError


def test_frame_rejects_pair_arguments_before_pipeline_imports() -> None:
    """The removed pair-shaped CLI invocation fails closed with guidance."""
    with pytest.raises(PairConfigurationMigrationError, match="--paths"):
        run_frame_cli(
            reference="reference.SAFE",
            secondary="secondary.SAFE",
            output="out",
        )

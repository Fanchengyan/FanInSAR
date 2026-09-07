"""CLI tests for the retained non-Frame command boundary."""

from __future__ import annotations

import pytest

from faninsar.cli.main import main


def test_retired_frame_command_fails_closed() -> None:
    """The deleted Frame workflow is not exposed as a CLI compatibility path."""
    with pytest.raises(SystemExit) as excinfo:
        main(["frame"])
    assert excinfo.value.code == 2

"""CLI tests for the retained non-Frame command boundary."""

from __future__ import annotations

import pytest

from faninsar.cli.main import main


def test_retired_frame_command_fails_closed() -> None:
    """The deleted Frame workflow is not exposed as a CLI compatibility path."""
    with pytest.raises(SystemExit) as excinfo:
        main(["frame"])
    assert excinfo.value.code == 2


def test_stack_verify_help_describes_structural_checks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``faninsar stack verify`` help does not claim byte checksum verification."""
    with pytest.raises(SystemExit) as excinfo:
        main(["stack", "verify", "--help"])
    assert excinfo.value.code == 0
    text = capsys.readouterr().out.lower()
    assert "checksum" not in text
    assert "structur" in text or "metadata" in text

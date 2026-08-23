"""CLI pre-gate tests for ``--dem-source`` (PROPOSAL-0030).

The ``frame`` command pre-validates ``--dem-source`` through the selection
grammar before any pipeline work; the ``auto`` alias is resolved by
DEMManager (not the grammar) and must pass through untouched.
"""

from __future__ import annotations

import sys
import types

import pytest

from faninsar.cli.main import main


class _FrameStub:
    """Stand-in for the frame subcommand recording its call arguments."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_frame_cli(self, **kwargs: object) -> int:
        self.calls.append(kwargs)
        return 0


@pytest.fixture
def frame_stub(monkeypatch: pytest.MonkeyPatch) -> _FrameStub:
    """Route the frame subcommand through the stub (no pipeline imports)."""
    stub = _FrameStub()
    module = types.ModuleType("faninsar.cli.frame")
    module.run_frame_cli = stub.run_frame_cli
    monkeypatch.setitem(sys.modules, "faninsar.cli.frame", module)
    return stub


_ARGS = ["--reference", "ref.SAFE", "--secondary", "sec.SAFE", "--output", "out"]


def test_dem_source_auto_passes_pre_gate(frame_stub: _FrameStub) -> None:
    """BLOCKER-0030-B3: ``--dem-source auto`` is accepted by the CLI."""
    code = main(["frame", *_ARGS, "--dem-source", "auto"])
    assert code == 0
    assert frame_stub.calls
    assert frame_stub.calls[0]["dem_source"] == "auto"


def test_dem_source_plain_product_passes_pre_gate(frame_stub: _FrameStub) -> None:
    """A plain product selection still validates and forwards."""
    code = main(["frame", *_ARGS, "--dem-source", "glo30"])
    assert code == 0
    assert frame_stub.calls[0]["dem_source"] == "glo30"


def test_dem_source_unknown_rejected_before_pipeline(
    frame_stub: _FrameStub,
    capsys: pytest.CaptureFixture,
) -> None:
    """Unknown selections exit 2 before any pipeline work."""
    with pytest.raises(SystemExit) as excinfo:
        main(["frame", *_ARGS, "--dem-source", "glo30:not-a-provider"])
    assert excinfo.value.code == 2
    assert "unknown DEM provider" in capsys.readouterr().err
    assert frame_stub.calls == []


def test_dem_source_omitted_keeps_none(frame_stub: _FrameStub) -> None:
    """Omitted --dem-source forwards None (env/default behavior)."""
    assert main(["frame", *_ARGS]) == 0
    assert frame_stub.calls[0]["dem_source"] is None
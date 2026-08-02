"""CompileManager unit tests."""

from __future__ import annotations

from faninsar.compute.compile import COMPILE_TARGETS, CompileManager


def test_unknown_target_raises() -> None:
    mgr = CompileManager()
    try:
        mgr.get("not_a_kernel", device="cpu")
        raised = False
    except KeyError:
        raised = True
    assert raised


def test_recompile_count_increments() -> None:
    mgr = CompileManager()
    mgr.get("deramp", device="cpu")
    assert mgr.recompile_counts.get("deramp", 0) >= 1
    # cache hit should not require new key missing
    mgr.get("deramp", device="cpu")
    assert "deramp" in COMPILE_TARGETS

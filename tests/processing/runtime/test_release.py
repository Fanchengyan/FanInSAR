"""Tests for release qualification gates."""

from __future__ import annotations

from faninsar.processing.runtime.release import run_release_gates


def test_release_gates_pass_in_development_tree() -> None:
    """Core release gates pass without external SAR runtimes."""
    result = run_release_gates()
    assert result.passed is True
    assert result.checks["no_external_sar_runtime"] is True
    assert result.checks["core_imports"] is True
    assert "complex_multiply" in result.details["kernels"]

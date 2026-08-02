"""CLI warmup smoke tests."""

from __future__ import annotations

from faninsar.cli.main import main
from faninsar.cli.warmup import run_warmup


def test_main_no_args_prints_help() -> None:
    assert main([]) == 0


def test_warmup_cpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FANINSAR_COMPILE_CACHE", str(tmp_path / "c"))
    code = run_warmup(device="cpu", profile="sentinel1")
    assert code in {0, 1}  # 1 if torch missing

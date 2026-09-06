"""Compile cache resolution tests."""

from __future__ import annotations

import os
from pathlib import Path

from faninsar.processing.runtime.compute.cache import apply_compile_cache_env, resolve_compile_cache_dir


def test_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FANINSAR_COMPILE_CACHE", str(tmp_path / "cache"))
    resolved = resolve_compile_cache_dir()
    assert resolved == (tmp_path / "cache").resolve()
    applied = apply_compile_cache_env()
    assert applied == resolved
    assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == resolved
    assert resolved.is_dir()


def test_default_not_tmp_only(monkeypatch) -> None:
    monkeypatch.delenv("FANINSAR_COMPILE_CACHE", raising=False)
    resolved = resolve_compile_cache_dir()
    assert "faninsar" in str(resolved)

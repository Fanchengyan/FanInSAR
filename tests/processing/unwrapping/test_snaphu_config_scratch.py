"""SNAPHU config resolution, passthrough, and WorkDir scratch lifecycle."""

from __future__ import annotations

import math
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from faninsar.processing.unwrapping.api import unwrap
from faninsar.processing.unwrapping.snaphu_backend import (
    Snaphu,
    SnaphuConfig,
    _AUTO_MAX_PROCESSES,
    _AUTO_TILE_OVERLAP,
    _AUTO_TILE_SIZE,
    _resolve_snaphu_config,
    snaphu_available,
    snaphu_unwrap,
)


def _phase_pair(shape: tuple[int, int] = (8, 8)) -> tuple[torch.Tensor, torch.Tensor]:
    phase = torch.zeros(shape, dtype=torch.float32)
    coherence = torch.ones(shape, dtype=torch.float32)
    return phase, coherence


def _install_fake_snaphu(monkeypatch: pytest.MonkeyPatch, handler):
    """Replace the installed snaphu.unwrap after FanInSAR imports the module."""
    import snaphu

    monkeypatch.setattr(snaphu, "unwrap", handler)
    return handler


def test_snaphu_config_legacy_positional_nproc_is_fourth() -> None:
    """Original positional order (cost, nlooks, ntiles, nproc) is unchanged."""
    config = SnaphuConfig("defo", 10.0, (2, 2), -1)
    assert config.cost == "defo"
    assert config.nlooks == 10.0
    assert config.ntiles == (2, 2)
    assert config.nproc == -1
    assert config.tile_overlap is None
    assert config.single_tile_reoptimize is True


def test_snaphu_config_new_fields_are_keyword_only() -> None:
    """tile_overlap and single_tile_reoptimize cannot be positional."""
    with pytest.raises(TypeError):
        SnaphuConfig("defo", 1.0, (1, 1), 1, 64)  # type: ignore[misc]
    config = SnaphuConfig(
        ntiles=(2, 2),
        tile_overlap=64,
        single_tile_reoptimize=False,
    )
    assert config.tile_overlap == 64
    assert config.single_tile_reoptimize is False


def test_resolve_auto_tiles_overlap_and_process_cap() -> None:
    """None fields use the private 2048/64/4 rule; explicit values win."""
    auto = _resolve_snaphu_config(SnaphuConfig(), shape=(4096, 1000))
    assert auto.ntiles == (2, 1)
    assert auto.tile_overlap == (_AUTO_TILE_OVERLAP, 0)
    assert auto.nproc == min(2, _AUTO_MAX_PROCESSES)
    assert auto.nlooks == 1.0
    assert auto.single_tile_reoptimize is True

    tall = _resolve_snaphu_config(SnaphuConfig(), shape=(100, 5000))
    assert tall.ntiles == (1, math.ceil(5000 / _AUTO_TILE_SIZE))
    assert tall.tile_overlap == (0, _AUTO_TILE_OVERLAP)

    explicit = _resolve_snaphu_config(
        SnaphuConfig(
            nlooks=12.0,
            ntiles=(4, 4),
            nproc=3,
            tile_overlap=0,
            single_tile_reoptimize=False,
        ),
        shape=(4096, 4096),
    )
    assert explicit.nlooks == 12.0
    assert explicit.ntiles == (4, 4)
    assert explicit.nproc == 3
    assert explicit.tile_overlap == (0, 0)
    assert explicit.single_tile_reoptimize is False


def test_resolve_nproc_none_is_bounded_and_nonpositive_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic nproc is capped; explicit nproc<=0 keeps upstream meaning."""
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    tiled = _resolve_snaphu_config(
        SnaphuConfig(ntiles=(3, 3)),
        shape=(4096, 4096),
        process_limit=8,
    )
    assert tiled.nproc == min(9, 32, _AUTO_MAX_PROCESSES, 8)

    preserved = _resolve_snaphu_config(
        SnaphuConfig(ntiles=(2, 2), nproc=0),
        shape=(512, 512),
    )
    assert preserved.nproc == 0
    negative = _resolve_snaphu_config(
        SnaphuConfig(ntiles=(2, 2), nproc=-1),
        shape=(512, 512),
    )
    assert negative.nproc == -1


def test_resolve_rejects_invalid_tile_geometry() -> None:
    """Explicit ntiles must fit the image; overlap must be a nonnegative pair."""
    with pytest.raises(ValueError, match="ntiles"):
        _resolve_snaphu_config(SnaphuConfig(ntiles=(32, 1)), shape=(8, 8))
    with pytest.raises(ValueError, match="tile_overlap"):
        _resolve_snaphu_config(
            SnaphuConfig(ntiles=(2, 2), tile_overlap=-1),
            shape=(32, 32),
        )
    kept = _resolve_snaphu_config(
        SnaphuConfig(ntiles=(2, 2), tile_overlap=2),
        shape=(32, 32),
    )
    assert kept.tile_overlap == (2, 2)


def test_explicit_overlap_on_untiled_axis_is_preserved() -> None:
    """Upstream accepts overlap 64 with ntiles (1, 3) on a 100 x 5000 grid."""
    resolved = _resolve_snaphu_config(
        SnaphuConfig(ntiles=(1, 3), tile_overlap=64),
        shape=(100, 5000),
    )
    assert resolved.ntiles == (1, 3)
    assert resolved.tile_overlap == (64, 64)


def test_invalid_config_raises_before_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Standalone invalid SNAPHU config is a ValueError, not a failed result."""
    called = {"n": 0}

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        called["n"] += 1
        raise AssertionError("backend should not run")

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    phase, coherence = _phase_pair()
    with pytest.raises(ValueError, match="nlooks"):
        Snaphu(SnaphuConfig(nlooks=0.5), work_dir=tmp_path).unwrap(
            phase, coherence=coherence
        )
    assert called["n"] == 0


def test_resolve_nlooks_from_looks_and_divisor() -> None:
    """Missing nlooks uses looks/divisor; explicit nlooks bypasses the divisor."""
    resolved = _resolve_snaphu_config(
        SnaphuConfig(),
        shape=(32, 32),
        looks=(5, 2),
        nlooks_divisor=0.64,
    )
    assert resolved.nlooks == pytest.approx(5 * 2 / 0.64)
    explicit = _resolve_snaphu_config(
        SnaphuConfig(nlooks=12.0),
        shape=(32, 32),
        looks=(5, 2),
        nlooks_divisor=0.64,
    )
    assert explicit.nlooks == 12.0
    with pytest.raises(ValueError, match="nlooks"):
        _resolve_snaphu_config(
            SnaphuConfig(),
            shape=(8, 8),
            looks=(1, 1),
            nlooks_divisor=4.0,
        )


def test_scratch_is_workdir_local_unique_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each wrapper call owns a unique WorkDir scratch that is deleted afterwards."""
    seen: list[Path] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        scratch = Path(kwargs["scratchdir"])
        seen.append(scratch)
        assert scratch.is_dir()
        assert kwargs.get("delete_scratch") is False
        assert scratch.parent == tmp_path / "tmp" / "snaphu"
        assert scratch.name.startswith("snaphu-20240101_20240113-")
        (scratch / "native.bin").write_bytes(b"x")
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    strategy = Snaphu(SnaphuConfig(nlooks=1.0, ntiles=(1, 1)), work_dir=tmp_path)
    strategy._pair_id = "20240101_20240113"
    phase, coherence = _phase_pair()
    result = strategy.unwrap(phase, coherence=coherence)
    assert result.converged
    assert len(seen) == 1
    assert not seen[0].exists()
    assert (tmp_path / "tmp" / "snaphu").is_dir()


def test_failed_wrapper_still_deletes_owned_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ordinary backend exceptions still unwind the invocation directory."""
    seen: list[Path] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        scratch = Path(kwargs["scratchdir"])
        seen.append(scratch)
        (scratch / "partial").write_text("x", encoding="utf-8")
        raise RuntimeError("native failed")

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    sentinel = tmp_path / "tmp" / "snaphu"
    sentinel.mkdir(parents=True)
    keep = sentinel / "keep-me.txt"
    keep.write_text("sentinel", encoding="utf-8")
    phase, coherence = _phase_pair()
    result = Snaphu(work_dir=tmp_path).unwrap(phase, coherence=coherence)
    assert not result.converged
    assert result.failure_reason == "backend_failure"
    assert seen and not seen[0].exists()
    assert keep.read_text(encoding="utf-8") == "sentinel"


def test_concurrent_same_pair_gets_distinct_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retries and concurrent same-Pair calls never share a run directory."""
    barrier = threading.Barrier(2)
    seen: list[Path] = []
    lock = threading.Lock()

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        scratch = Path(kwargs["scratchdir"])
        with lock:
            seen.append(scratch)
        barrier.wait(timeout=5)
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    phase, coherence = _phase_pair()

    def run() -> None:
        strategy = Snaphu(SnaphuConfig(ntiles=(1, 1)), work_dir=tmp_path)
        strategy._pair_id = "20240101_20240113"
        strategy.unwrap(phase, coherence=coherence)

    workers = [threading.Thread(target=run) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert all(path.name.startswith("snaphu-20240101_20240113-") for path in seen)
    assert all(not path.exists() for path in seen)


def test_standalone_default_scratch_is_cwd_not_os_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitted work_dir uses Path.cwd()/tmp/snaphu, never the OS temp root."""
    monkeypatch.chdir(tmp_path)
    seen: list[Path] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        scratch = Path(kwargs["scratchdir"])
        seen.append(scratch.resolve())
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    phase, coherence = _phase_pair()
    Snaphu().unwrap(phase, coherence=coherence)
    assert seen
    scratch = seen[0]
    assert tmp_path.resolve() / "tmp" / "snaphu" == scratch.parent
    os_temp = Path(tempfile.gettempdir()).resolve()
    assert scratch.parent != os_temp
    assert scratch.parent.name == "snaphu"


def test_low_level_functions_forward_work_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Array helpers and the dispatcher accept work_dir."""
    seen: list[Path] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        seen.append(Path(kwargs["scratchdir"]))
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    ifg = np.ones((6, 6), dtype=np.complex64)
    coh = np.ones((6, 6), dtype=np.float32)
    snaphu_unwrap(ifg, coh, config=SnaphuConfig(nlooks=2.0), work_dir=tmp_path)
    unwrap(
        ifg,
        coh,
        method="snaphu",
        snaphu_config=SnaphuConfig(nlooks=2.0),
        work_dir=tmp_path,
    )
    assert len(seen) == 2
    assert all(path.parent == tmp_path / "tmp" / "snaphu" for path in seen)


@pytest.mark.skipif(not snaphu_available(), reason="snaphu-py is required")
def test_real_wrapper_small_array_uses_space_free_workdir(tmp_path: Path) -> None:
    """A bounded native call succeeds on a workspace path without spaces."""
    if " " in str(tmp_path.resolve()):
        pytest.skip("successful native wrapper tests cannot use paths with spaces")
    rng = np.random.default_rng(0)
    phase = rng.normal(0.0, 0.2, size=(24, 24)).astype(np.float32)
    ifg = np.exp(1j * phase).astype(np.complex64)
    coh = np.full(phase.shape, 0.9, dtype=np.float32)
    result = snaphu_unwrap(
        ifg,
        coh,
        config=SnaphuConfig(nlooks=1.0, ntiles=(1, 1), nproc=1),
        work_dir=tmp_path,
    )
    assert result.converged
    assert result.phase.shape == phase.shape
    assert bool(torch.all(torch.isfinite(result.phase[result.valid_mask])))
    leftover = list((tmp_path / "tmp" / "snaphu").glob("snaphu-*"))
    assert leftover == []

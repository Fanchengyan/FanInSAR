"""Stack nlooks_divisor inheritance, unwrap identity, and ionosphere reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
from tests.support.local_stack import LocalStack

from faninsar import Pairs
from faninsar.io.storage.ifg_store import write_ifg_artifact
from faninsar.io.storage.product import open_product
from faninsar.missions.s1 import S1Stack
from faninsar.processing.atmosphere import IonosphereEstimationConfig
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.unwrapping.common import SpatialUnwrapper, SpatialUnwrapResult
from faninsar.processing.unwrapping.snaphu_backend import Snaphu, SnaphuConfig
from faninsar.stack import Stack, StackConfig
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.identity import save_stack_identity
from tests.support.scene_stores import write_test_scene


class _GenericStack(LocalStack):
    """Local fixture Stack that uses the mission-neutral divisor default."""

    _default_nlooks_divisor: ClassVar[float] = 1.0


class _IdentityUnwrapper(SpatialUnwrapper):
    """Deterministic spatial strategy that does not invoke SNAPHU."""

    def unwrap(
        self,
        wrapped_phase: torch.Tensor,
        *,
        coherence: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> SpatialUnwrapResult:
        del coherence
        support = torch.isfinite(wrapped_phase)
        if valid_mask is not None:
            support &= valid_mask
        labels = torch.where(
            support,
            torch.zeros_like(wrapped_phase, dtype=torch.int64),
            torch.full_like(wrapped_phase, -1, dtype=torch.int64),
        )
        return SpatialUnwrapResult(
            phase=torch.where(support, wrapped_phase, torch.nan),
            valid_mask=support,
            component_labels=labels,
            reference_values=torch.zeros(1, device=wrapped_phase.device),
            converged=True,
            iterations=0,
            pcg_iterations=0,
            residual_norm=0.0,
            failure_reason=None,
        )


class _CustomSnaphu(Snaphu):
    """Subclass used to prove Stack does not replace custom strategies."""

    def unwrap(self, wrapped_phase, *, coherence=None, valid_mask=None):
        return _IdentityUnwrapper().unwrap(
            wrapped_phase, coherence=coherence, valid_mask=valid_mask
        )


def _safe_paths(tmp_path: Path, dates: tuple[str, ...]) -> list[Path]:
    paths = []
    for date in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date}T000000.SAFE"
        path.mkdir(parents=True)
        paths.append(path)
    return paths


def _stack_with_ifg(
    tmp_path: Path,
    *,
    looks: tuple[int, int] = (1, 1),
    nlooks_divisor: float | None = None,
    cls: type[LocalStack] = LocalStack,
) -> LocalStack:
    dates = ("20240101", "20240113")
    kwargs: dict[str, object] = {
        "work_dir": tmp_path / "out",
        "pairs": Pairs.from_names(["20240101_20240113"]),
        "multilook": looks,
    }
    if nlooks_divisor is not None:
        kwargs["nlooks_divisor"] = nlooks_divisor
    stack = cls.from_paths(_safe_paths(tmp_path, dates), **kwargs)
    stack.prepare_data()
    phase = np.full((2, 3), 0.2, dtype=np.float32)
    look_dir = f"ml_{looks[0]}x{looks[1]}"
    write_ifg_artifact(
        stack.config.work_dir / "ifg" / look_dir / "20240101_20240113",
        pair=dates,
        looks=looks,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
        complex_ifg=np.exp(1j * phase).astype(np.complex64),
        coherence=np.ones_like(phase),
    )
    stack.ifg_dirs = [stack.config.work_dir / "ifg" / look_dir / "20240101_20240113"]
    return stack


def _install_fake_snaphu(monkeypatch: pytest.MonkeyPatch, handler):
    import snaphu

    monkeypatch.setattr(snaphu, "unwrap", handler)
    return handler


def test_nlooks_divisor_defaults_and_explicit(tmp_path: Path) -> None:
    """Stack resolves 1.0, S1 resolves 0.64, and explicit 1.0 is preserved."""
    generic = _GenericStack.from_paths(
        _safe_paths(tmp_path / "g", ("20240101", "20240113")),
        work_dir=tmp_path / "g" / "out",
    )
    s1 = LocalStack.from_paths(
        _safe_paths(tmp_path / "s", ("20240101", "20240113")),
        work_dir=tmp_path / "s" / "out",
    )
    explicit = LocalStack.from_paths(
        _safe_paths(tmp_path / "e", ("20240101", "20240113")),
        work_dir=tmp_path / "e" / "out",
        nlooks_divisor=1.0,
    )
    assert generic.config.nlooks_divisor == 1.0
    assert s1.config.nlooks_divisor == pytest.approx(0.64)
    assert explicit.config.nlooks_divisor == 1.0


def test_nlooks_divisor_does_not_mutate_caller_config(tmp_path: Path) -> None:
    """Installing a class default copies the config instead of mutating it."""
    cfg = StackConfig(work_dir=tmp_path / "work")
    assert cfg.nlooks_divisor is None
    stack = LocalStack(
        catalog=SceneCatalog.from_paths(
            _safe_paths(tmp_path, ("20240101", "20240113"))
        ),
        config=cfg,
        pairs=Pairs.from_names(["20240101_20240113"]),
    )
    assert cfg.nlooks_divisor is None
    assert stack.config.nlooks_divisor == pytest.approx(0.64)
    assert stack.config is not cfg


def test_nlooks_divisor_validation(tmp_path: Path) -> None:
    """The divisor must be a finite number strictly greater than zero."""
    with pytest.raises(ValueError, match="nlooks_divisor"):
        StackConfig(work_dir=tmp_path, nlooks_divisor=0.0)
    with pytest.raises(ValueError, match="nlooks_divisor"):
        StackConfig(work_dir=tmp_path, nlooks_divisor=-0.5)
    with pytest.raises(ValueError, match="nlooks_divisor"):
        StackConfig(work_dir=tmp_path, nlooks_divisor=float("nan"))


def test_nlooks_divisor_reopen_and_tracks(tmp_path: Path) -> None:
    """Numeric values persist; missing legacy fields take the class default."""
    stack = LocalStack.from_paths(
        _safe_paths(tmp_path, ("20240101", "20240113")),
        work_dir=tmp_path / "work",
        nlooks_divisor=0.8,
        pairs=Pairs.from_names(["20240101_20240113"]),
    )
    save_stack_identity(stack)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened.config.nlooks_divisor == pytest.approx(0.8)

    document = json.loads(
        (stack.config.work_dir / "stack.json").read_text(encoding="utf-8")
    )
    del document["science"]["nlooks_divisor"]
    (stack.config.work_dir / "stack.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    legacy = Stack.open(stack.config.work_dir)
    assert legacy.config.nlooks_divisor == pytest.approx(0.64)


def test_track_documents_persist_numeric_divisor(tmp_path: Path) -> None:
    """Child track identities inherit the resolved numeric divisor."""
    from faninsar.stack.catalog import SceneCatalog
    from faninsar.stack.identity import save_stack_identity as save_identity

    dates = ("20240101", "20240113", "20240125")
    paths = _safe_paths(tmp_path / "inputs", dates)
    stack = S1Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(work_dir=tmp_path / "work", multilook=(1, 1)),
        pairs=Pairs.from_names(
            ["20240101_20240113", "20240113_20240125", "20240101_20240125"]
        ),
        misreg_pairs=Pairs.from_names(["20240101_20240113"]),
        reference="20240101",
    )
    records = {
        "20240101": {
            "track_key": "88A",
            "direction": "ASCENDING",
            "relative_orbit": 88,
            "mission_id": "S1A",
            "polarizations": ("VV",),
            "absolute_burst_ids": (3001,),
        },
        "20240113": {
            "track_key": "88A",
            "direction": "ASCENDING",
            "relative_orbit": 88,
            "mission_id": "S1A",
            "polarizations": ("VV",),
            "absolute_burst_ids": (3001,),
        },
        "20240125": {
            "track_key": "12D",
            "direction": "DESCENDING",
            "relative_orbit": 12,
            "mission_id": "S1A",
            "polarizations": ("VV",),
            "absolute_burst_ids": (3001,),
        },
    }
    stack.config.extra["track_metadata"] = {"dates": records}
    save_identity(stack)
    sub = json.loads(
        (stack.config.work_dir / "tracks" / "88A" / "stack.json").read_text(
            encoding="utf-8"
        )
    )
    assert sub["science"]["nlooks_divisor"] == pytest.approx(0.64)
    child = Stack.open(stack.config.work_dir / "tracks" / "88A")
    assert child.config.nlooks_divisor == pytest.approx(0.64)


def test_coreg_identity_excludes_divisor(tmp_path: Path) -> None:
    """Changing the unwrap-only coefficient does not alter coreg science identity."""
    stack = _stack_with_ifg(tmp_path)
    before = stack._coreg_science_config()
    stack.config.nlooks_divisor = 1.0
    after = stack._coreg_science_config()
    assert "nlooks_divisor" not in before
    assert before == after


def test_stack_unwrap_uses_product_looks_unless_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Actual IFG looks/divisor supply nlooks; an explicit value bypasses them."""
    captured: list[float] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        captured.append(float(nlooks))
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    stack = _stack_with_ifg(tmp_path, looks=(5, 2))
    stack.unwrap()
    assert captured[-1] == pytest.approx(5 * 2 / 0.64)

    stack.config.nlooks_divisor = 1.0
    with pytest.raises(InvalidProcessingStateError, match="overwrite=True"):
        stack.unwrap()
    stack.unwrap(overwrite=True)
    assert captured[-1] == pytest.approx(10.0)

    stack.unwrap(
        Snaphu(SnaphuConfig(nlooks=12.0, ntiles=(1, 1))),
        overwrite=True,
    )
    assert captured[-1] == 12.0
    stack.config.nlooks_divisor = 0.5
    stack.unwrap(Snaphu(SnaphuConfig(nlooks=12.0, ntiles=(1, 1))))
    assert captured[-1] == 12.0


def test_explicit_nlooks_reuse_ignores_divisor_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Divisor is not method identity when an explicit nlooks bypasses it."""
    calls = {"n": 0}

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        calls["n"] += 1
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    stack = _stack_with_ifg(tmp_path)
    stack.unwrap(Snaphu(SnaphuConfig(nlooks=9.0, ntiles=(1, 1))))
    stack.config.nlooks_divisor = 1.0
    stack.unwrap(Snaphu(SnaphuConfig(nlooks=9.0, ntiles=(1, 1))))
    assert calls["n"] == 1


def test_default_unwrap_is_fresh_snaphu_and_custom_subclass_is_kept(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """None builds a fresh Snaphu; subclasses are not replaced."""
    import inspect

    assert inspect.signature(Stack.unwrap).parameters["unwrapper"].default is None
    ids: list[int] = []

    def fake_unwrap(self, wrapped_phase, *, coherence=None, valid_mask=None):
        ids.append(id(self))
        return _IdentityUnwrapper().unwrap(
            wrapped_phase, coherence=coherence, valid_mask=valid_mask
        )

    monkeypatch.setattr(Snaphu, "unwrap", fake_unwrap)
    stack = _stack_with_ifg(tmp_path)
    stack.unwrap()
    stack.unwrap(overwrite=True)
    assert len(ids) == 2
    assert ids[0] != ids[1]

    custom = _CustomSnaphu()
    stack.unwrap(custom, overwrite=True)
    item = open_product(
        stack.config.work_dir / "unwrap" / "ml_1x1" / "20240101_20240113"
    )
    assert item.properties["faninsar:method"]["name"] == "_CustomSnaphu"


def test_persist_divisor_mutation_before_unwrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A mutable config change before unwrap is persisted for reopen reuse."""
    captured: list[float] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        captured.append(float(nlooks))
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    stack = _stack_with_ifg(tmp_path, looks=(5, 2))
    before_coreg = stack._coreg_science_config()
    stack.config.nlooks_divisor = 1.0
    stack.unwrap()
    assert captured == [pytest.approx(10.0)]
    assert stack._coreg_science_config() == before_coreg
    document = json.loads(
        (stack.config.work_dir / "stack.json").read_text(encoding="utf-8")
    )
    assert document["science"]["nlooks_divisor"] == pytest.approx(1.0)
    reopened = Stack.open(stack.config.work_dir)
    assert reopened.config.nlooks_divisor == pytest.approx(1.0)
    reopened.unwrap()
    assert captured == [pytest.approx(10.0)]


def test_legacy_request_json_without_divisor_still_prepares(tmp_path: Path) -> None:
    """Old request.json missing nlooks_divisor still prepares; live change is unwrap-only."""
    dates = ("20240101", "20240113")
    stack = LocalStack.from_paths(
        _safe_paths(tmp_path, dates),
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    )
    report = stack.prepare_data()
    assert report.complete
    assert stack.data_ready
    request_path = stack.config.work_dir / "inputs" / "request.json"
    document = json.loads(request_path.read_text(encoding="utf-8"))
    assert document["science"]["nlooks_divisor"] == pytest.approx(0.64)
    document["science"].pop("nlooks_divisor")
    request_path.write_text(json.dumps(document), encoding="utf-8")
    stack.config.nlooks_divisor = 1.0
    again = stack.prepare_data()
    assert again.complete
    assert stack.data_ready
    reopened = Stack.open(stack.config.work_dir)
    reopened.prepare_data()
    assert reopened.data_ready


def test_request_only_reopen_preserves_nondefault_divisor(tmp_path: Path) -> None:
    """Request-only reopen keeps a non-default divisor recorded in request.json."""
    stack = LocalStack.from_paths(
        _safe_paths(tmp_path, ("20240101", "20240113")),
        work_dir=tmp_path / "out",
        nlooks_divisor=0.8,
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    )
    stack.prepare_data()
    request = json.loads(
        (stack.config.work_dir / "inputs" / "request.json").read_text(encoding="utf-8")
    )
    assert request["science"]["nlooks_divisor"] == pytest.approx(0.8)
    (stack.config.work_dir / "stack.json").unlink()
    reopened = Stack.open(stack.config.work_dir)
    assert reopened.config.nlooks_divisor == pytest.approx(0.8)


def test_live_reprepare_overlays_current_divisor(tmp_path: Path) -> None:
    """A live divisor mutation is overlaid onto the immutable request snapshot."""
    stack = LocalStack.from_paths(
        _safe_paths(tmp_path, ("20240101", "20240113")),
        work_dir=tmp_path / "out",
        nlooks_divisor=0.8,
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    )
    stack.prepare_data()
    snapshot = stack._request_config.nlooks_divisor
    stack.config.nlooks_divisor = 1.0
    stack.preparation_report = None
    stack.prepare_data()
    assert stack._request_config.nlooks_divisor == pytest.approx(snapshot)
    assert stack.config.nlooks_divisor == pytest.approx(1.0)
    request = json.loads(
        (stack.config.work_dir / "inputs" / "request.json").read_text(encoding="utf-8")
    )
    assert request["science"]["nlooks_divisor"] == pytest.approx(1.0)
    identity = json.loads(
        (stack.config.work_dir / "stack.json").read_text(encoding="utf-8")
    )
    assert identity["science"]["nlooks_divisor"] == pytest.approx(1.0)


def test_unwrap_records_nlooks_diagnostics_outside_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """nlooks source, looks, and divisor are unwrap diagnostics, not reuse identity."""

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    stack = _stack_with_ifg(tmp_path, looks=(5, 2))
    stack.unwrap()
    item = open_product(
        stack.config.work_dir / "unwrap" / "ml_5x2" / "20240101_20240113"
    )
    recorded = item.properties["faninsar:nlooks"]
    assert recorded["source"] == "multilook_divisor"
    assert recorded["nlooks"] == pytest.approx(10.0 / 0.64)
    assert recorded["nlooks_divisor"] == pytest.approx(0.64)
    assert recorded["looks"] == [5, 2]
    method = item.properties["faninsar:method"]
    assert "source" not in method["parameters"].get("config", {})

    stack.unwrap(Snaphu(SnaphuConfig(nlooks=12.0, ntiles=(1, 1))), overwrite=True)
    explicit = open_product(
        stack.config.work_dir / "unwrap" / "ml_5x2" / "20240101_20240113"
    )
    recorded = explicit.properties["faninsar:nlooks"]
    assert recorded["source"] == "explicit"
    assert recorded["nlooks"] == pytest.approx(12.0)
    assert recorded["nlooks_divisor"] == pytest.approx(0.64)
    assert recorded["looks"] == [5, 2]


def test_stack_scratch_uses_pair_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stack injects WorkDir and the normalized Pair prefix privately."""
    seen: list[Path] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        seen.append(Path(kwargs["scratchdir"]))
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    stack = _stack_with_ifg(tmp_path)
    stack.unwrap()
    assert seen
    scratch = seen[0]
    assert scratch.parent == stack.config.work_dir / "tmp" / "snaphu"
    assert scratch.name.startswith("snaphu-20240101_20240113-")
    assert not scratch.exists()


def _ion_config() -> IonosphereEstimationConfig:
    return IonosphereEstimationConfig(
        f0=1257.5e6,
        freq_low=1257.5e6 - 28.0e6 / 3,
        freq_high=1257.5e6 + 28.0e6 / 3,
    )


def _slc_with_subband_energy(shape: tuple[int, int] = (4, 16)) -> np.ndarray:
    """Return an SLC whose range spectrum has energy in both split halves."""
    spectrum = np.zeros(shape, dtype=np.complex128)
    spectrum[:, 1] = 1.0
    spectrum[:, -2] = 0.6
    return (np.fft.ifft(spectrum, axis=-1) * 100.0).astype(np.complex64)


def _stack_with_scenes(tmp_path: Path) -> LocalStack:
    dates = ("20240101", "20240113")
    stack = LocalStack.from_paths(
        _safe_paths(tmp_path, dates),
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    )
    stack.prepare_data()
    payload = _slc_with_subband_energy()
    for date_id in dates:
        write_test_scene(
            stack.config.work_dir,
            date_id=date_id,
            reference_id=stack.reference,
            payload=payload,
            reference_payload=payload,
            grid_shape=(4, 16),
            wavelength_m=0.236,
        )
    return stack


def test_ionosphere_records_effective_snaphu_and_requires_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ion reuse compares effective SNAPHU settings, including legacy-missing."""
    scratches: list[Path] = []

    def fake_unwrap(igram, corr, nlooks, **kwargs):
        scratches.append(Path(kwargs["scratchdir"]))
        return np.zeros(igram.shape, dtype=np.float32), np.zeros(
            igram.shape, dtype=np.int32
        )

    _install_fake_snaphu(monkeypatch, fake_unwrap)
    stack = _stack_with_scenes(tmp_path)
    stores = stack.estimate_ionosphere(
        config=_ion_config(),
        range_sampling_rate_hz=32.0e6,
        ion_multilook=(1, 1),
        filter_sigma=(0.0, 0.0),
        min_cluster_pixels=1,
        unwrap_method="snaphu",
        device="cpu",
    )
    assert stores
    params = stores[0].method_parameters
    assert "snaphu" in params
    assert params["snaphu"]["nlooks"] == pytest.approx(1 / 0.64)
    assert len(scratches) == 2
    assert scratches[0] != scratches[1]
    assert all(
        path.parent == stack.config.work_dir / "tmp" / "snaphu" for path in scratches
    )
    assert all(path.name.startswith("snaphu-20240101_20240113-") for path in scratches)

    stack.config.nlooks_divisor = 1.0
    with pytest.raises(InvalidProcessingStateError, match="overwrite=True"):
        stack.estimate_ionosphere(
            config=_ion_config(),
            range_sampling_rate_hz=32.0e6,
            ion_multilook=(1, 1),
            unwrap_method="snaphu",
            device="cpu",
        )

    recorded = dict(stores[0].method_parameters)
    recorded.pop("snaphu")
    # Legacy products without the snaphu block also require overwrite.
    item_path = stores[0].root / "item.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    item["properties"]["faninsar:method"]["parameters"] = recorded
    item_path.write_text(json.dumps(item), encoding="utf-8")
    with pytest.raises(InvalidProcessingStateError, match="overwrite=True"):
        stack.estimate_ionosphere(
            config=_ion_config(),
            range_sampling_rate_hz=32.0e6,
            ion_multilook=(1, 1),
            unwrap_method="snaphu",
            device="cpu",
        )
    stack.estimate_ionosphere(
        config=_ion_config(),
        range_sampling_rate_hz=32.0e6,
        ion_multilook=(1, 1),
        filter_sigma=(0.0, 0.0),
        min_cluster_pixels=1,
        unwrap_method="snaphu",
        device="cpu",
        overwrite=True,
    )
    identity = json.loads(
        (stack.config.work_dir / "stack.json").read_text(encoding="utf-8")
    )
    assert identity["science"]["nlooks_divisor"] == pytest.approx(1.0)

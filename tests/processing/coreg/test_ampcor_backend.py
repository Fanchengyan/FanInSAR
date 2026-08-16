"""Focused tests for the prepared Ampcor backend boundary."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coreg.ampcor_backend import (
    AmpcorBackendRegistry,
    AmpcorCandidateError,
    AmpcorEnergyCandidate,
    eager_ampcor_candidate,
    native_workspace_bytes,
    prepare_ampcor_compile,
    torch_integral_energy,
)

_NATIVE_ABI = "faninsar.ampcor_prefix_energy.v1"


def test_torch_integral_energy_matches_reference() -> None:
    """The pure Tensor operation matches an unfold/square reference."""
    torch = pytest.importorskip("torch")
    values = torch.arange(24, dtype=torch.float64).reshape(1, 4, 6)
    result = torch_integral_energy(values, 2, 3)
    expected = values.square().unfold(-2, 2, 1).unfold(-2, 3, 1).sum(dim=(-1, -2))
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


def test_registry_requires_prepared_exact_candidates() -> None:
    """Registry lookup is exact and never creates a candidate."""
    registry = AmpcorBackendRegistry()
    candidate = eager_ampcor_candidate(
        device="cpu", window_shape=(2, 3), workspace_bytes=128
    )
    registry.register(candidate)
    assert (
        registry.get(
            "cpu",
            "eager",
            (2, 3),
            runtime_profile="torch-eager",
            abi_version=_NATIVE_ABI,
            allow_partial_batch=True,
        )
        is candidate
    )
    assert registry.get("cpu", "native", (2, 3)) is None
    cuda_candidate = eager_ampcor_candidate(device="cuda:0", window_shape=(2, 3))
    registry.register(cuda_candidate)
    assert (
        registry.get(
            "cuda",
            "eager",
            (2, 3),
            runtime_profile="torch-eager",
            abi_version=_NATIVE_ABI,
            allow_partial_batch=True,
        )
        is cuda_candidate
    )


def test_native_workspace_packet_is_shape_derived() -> None:
    """Native workspace admission is derived from input and output shapes."""
    assert native_workspace_bytes((2, 5, 6), (2, 3)) == 8 * (
        2 * 5 * 6 + 2 * 6 * 7 + 2 * 4 * 4
    )


def test_candidate_executes_only_its_prepared_callable() -> None:
    """Candidate execution does not expose a compiler or loader hook."""
    torch = pytest.importorskip("torch")
    calls: list[str] = []

    candidate = AmpcorEnergyCandidate(
        backend="native",
        device="cpu",
        window_shape=(2, 2),
        executor=lambda value: calls.append("native")
        or torch_integral_energy(value, 2, 2),
        input_shape=(1, 4, 4),
        abi_version=_NATIVE_ABI,
    )
    values = torch.tensor(
        [
            [
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, 1.0],
            ]
        ],
        dtype=torch.float64,
    )
    result = candidate.execute(values)
    assert tuple(result.shape) == (1, 3, 3)
    assert calls == ["native"]


def test_candidate_rejects_non_centered_input() -> None:
    """The native ABI boundary requires a centered float64 tensor."""
    torch = pytest.importorskip("torch")
    candidate = eager_ampcor_candidate(device="cpu", window_shape=(2, 2))
    with pytest.raises(AmpcorCandidateError, match="centered"):
        candidate.execute(torch.ones((1, 3, 3), dtype=torch.float64))


def test_candidate_allows_partial_batch_when_capacity_is_declared() -> None:
    """Native/eager candidates may consume a final batch below capacity."""
    torch = pytest.importorskip("torch")
    candidate = AmpcorEnergyCandidate(
        backend="native",
        device="cpu",
        window_shape=(2, 2),
        executor=lambda value: torch_integral_energy(value, 2, 2),
        input_shape=(4, 4, 4),
        allow_partial_batch=True,
    )
    result = candidate.execute(torch.zeros((2, 4, 4), dtype=torch.float64))
    assert tuple(result.shape) == (2, 3, 3)


def test_candidate_rejects_partial_batch_for_fixed_shape_compile() -> None:
    """A fixed-shape compile candidate rejects a partial batch before execute."""
    torch = pytest.importorskip("torch")
    calls: list[str] = []
    candidate = AmpcorEnergyCandidate(
        backend="compile",
        device="cpu",
        window_shape=(2, 2),
        executor=lambda value: calls.append("compile")
        or torch_integral_energy(value, 2, 2),
        input_shape=(4, 4, 4),
    )
    with pytest.raises(AmpcorCandidateError, match="shape"):
        candidate.execute(torch.zeros((2, 4, 4), dtype=torch.float64))
    assert calls == []


def test_candidate_rejects_spatial_shape_mismatch_before_execute() -> None:
    """Prepared candidates reject an input with the wrong spatial shape."""
    torch = pytest.importorskip("torch")
    calls: list[str] = []
    candidate = AmpcorEnergyCandidate(
        backend="native",
        device="cpu",
        window_shape=(2, 2),
        executor=lambda value: calls.append("native")
        or torch_integral_energy(value, 2, 2),
        input_shape=(4, 4, 4),
        allow_partial_batch=True,
    )
    with pytest.raises(AmpcorCandidateError, match="shape"):
        candidate.execute(torch.zeros((2, 5, 4), dtype=torch.float64))
    assert calls == []


def test_compile_preparation_compiles_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile preparation invokes Torch compile before candidate execution."""
    torch = pytest.importorskip("torch")
    compile_calls: list[object] = []

    def recording_compile(*args: object, **kwargs: object) -> object:
        compile_calls.append((args, kwargs))
        return args[0]

    monkeypatch.setattr(torch, "compile", recording_compile)
    candidate = prepare_ampcor_compile(
        device="cpu", window_shape=(2, 2), input_shape=(1, 4, 4)
    )
    assert candidate.backend == "compile"
    assert len(compile_calls) == 1
    result = candidate.execute(torch.zeros((1, 4, 4), dtype=torch.float64))
    assert tuple(result.shape) == (1, 3, 3)
    assert len(compile_calls) == 1


def test_compile_preparation_rejects_parity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compiled candidate is not published when eager parity fails."""
    torch = pytest.importorskip("torch")

    def bad_compile(function: object, **_kwargs: object) -> object:
        return lambda sample: function(sample) + 1.0

    monkeypatch.setattr(torch, "compile", bad_compile)
    with pytest.raises(RuntimeError, match="eager parity"):
        prepare_ampcor_compile(device="cpu", window_shape=(2, 2), input_shape=(1, 4, 4))


def _patch_fake_native_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
    *,
    source_abi: str,
    add_one: bool = False,
) -> None:
    """Install a deterministic fake native preparation boundary for tests."""
    import torch

    from faninsar.processing.coreg import ampcor_backend
    from faninsar.processing.geometry.native_v2.builder import (
        BuildPlan,
        NativeBackend,
        NativeOperation,
        PreparationStatus,
        PreparedNativeCandidate,
    )

    source = tmp_path / "ampcor.cpp"
    source.write_text("fake source", encoding="utf-8")
    plan = BuildPlan(
        NativeOperation.AMPCOR_PREFIX_ENERGY,
        NativeBackend.CPU,
        "fake_ampcor",
        "fake_ampcor",
        (source,),
        (),
        (),
        runtime_name="libomp",
    )

    def fake_plan(_builder: object, _request: object) -> object:
        return plan

    def fake_prepare(
        _builder: object, _request: object, *, build: object
    ) -> PreparedNativeCandidate:
        artifact = build(plan)
        return PreparedNativeCandidate(
            plan=plan,
            status=PreparationStatus.PREPARED,
            artifact=artifact,
        )

    monkeypatch.setattr(ampcor_backend.NativeBuilder, "plan", fake_plan)
    monkeypatch.setattr(ampcor_backend.NativeBuilder, "prepare", fake_prepare)

    def native_energy(value: object, window_az: int, window_rg: int) -> object:
        result = ampcor_backend.torch_integral_energy(value, window_az, window_rg)
        return result + 1.0 if add_one else result

    module = type(
        "FakeNativeModule",
        (),
        {
            "__file__": str(source),
            "native_source_abi": staticmethod(lambda: source_abi),
            "ampcor_prefix_energy_cpu": staticmethod(native_energy),
        },
    )()
    from torch.utils import cpp_extension

    monkeypatch.setattr(cpp_extension, "load", lambda **_kwargs: module)


def test_native_preparation_rejects_source_abi_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Native preparation fails closed when the module ABI is not exact."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg.ampcor_backend import prepare_ampcor_native

    _patch_fake_native_loader(monkeypatch, tmp_path, source_abi="wrong.abi")
    with pytest.raises(RuntimeError, match="source ABI mismatch"):
        prepare_ampcor_native(
            device="cpu",
            window_shape=(2, 2),
            source_root=tmp_path,
            build_dir=tmp_path / "build",
            input_shape=(1, 4, 4),
        )


def test_native_preparation_rejects_eager_parity_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Native preparation does not publish a numerically divergent module."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg.ampcor_backend import prepare_ampcor_native

    _patch_fake_native_loader(
        monkeypatch,
        tmp_path,
        source_abi="faninsar.ampcor_prefix_energy.v1",
        add_one=True,
    )
    with pytest.raises(RuntimeError, match="eager parity"):
        prepare_ampcor_native(
            device="cpu",
            window_shape=(2, 2),
            source_root=tmp_path,
            build_dir=tmp_path / "build",
            input_shape=(1, 4, 4),
        )


def test_explicit_native_without_candidate_fails_closed() -> None:
    """Explicit native selection never silently becomes eager."""
    from faninsar.processing.coreg import estimate_patch_amplitude_shift
    from faninsar.processing.errors import InvalidProcessingStateError

    samples = np.ones((32, 32), dtype=np.complex64)
    with pytest.raises(InvalidProcessingStateError, match="prepared same-device"):
        estimate_patch_amplitude_shift(
            samples,
            samples,
            window_az=8,
            window_rg=8,
            search_az=2,
            search_rg=2,
            n_az=1,
            n_rg=1,
            margin_az=8,
            margin_rg=8,
            executor="torch",
            device="cpu",
            backend="native",
        )


def test_product_passes_centered_secondary_to_energy_candidate() -> None:
    """The product NCC path gives native-compatible centered secondary input."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets

    observed: list[float] = []

    def energy(secondary: object) -> object:
        observed.append(float(secondary.mean().item()))
        return torch_integral_energy(secondary, 2, 2)

    candidate = AmpcorEnergyCandidate(
        backend="native",
        device="cpu",
        window_shape=(2, 2),
        executor=energy,
    )
    values = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4) / 16.0
    offsets._torch_patch_ncc_batch(
        values[:, :2, :2],
        values + 3.0,
        search_az=1,
        search_rg=1,
        subpixel=False,
        energy_candidate=candidate,
    )
    assert observed == [0.0]

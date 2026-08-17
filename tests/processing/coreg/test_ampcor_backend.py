"""Focused tests for the prepared Ampcor backend boundary."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coreg.ampcor_backend import (
    AmpcorBackendRegistry,
    AmpcorCandidateError,
    AmpcorEnergyCandidate,
    AmpcorNccCandidate,
    AmpcorNccCandidateError,
    AmpcorNccExecutionError,
    AmpcorNccRuntimeProfile,
    AmpcorNccWorkspaceMeasurement,
    ampcor_ncc_postprocess_reference,
    eager_ampcor_candidate,
    native_workspace_bytes,
    prepare_ampcor_compile,
    torch_integral_energy,
)

_NATIVE_ABI = "faninsar.ampcor_prefix_energy.v1"
_NCC_ABI = "faninsar.ampcor_ncc_postprocess.v1"
_NCC_OPERATION = "ampcor_ncc_postprocess.v1"
_NCC_SOURCE = "c9ab63fc3e1904ecd3ccee1121e349d05de908ac05fd59c4bb6b837641f987d5"


def _ncc_profile(
    *,
    source_digest: str = _NCC_SOURCE,
    device_name: str = "NVIDIA A100 80GB PCIe",
    torch_version: str = "2.8.0+cu128",
    cuda_runtime: str = "12.8",
) -> AmpcorNccRuntimeProfile:
    """Build a deterministic NCC qualification profile for registry tests."""
    return AmpcorNccRuntimeProfile(
        device_name=device_name,
        compute_capability=(8, 0),
        torch_version=torch_version,
        cuda_runtime=cuda_runtime,
        memory_bytes=80 * 1024**3,
        source_digest=source_digest,
        abi_version=_NCC_ABI,
    )


def _ncc_measurement(
    *,
    search_shape: tuple[int, int] = (17, 17),
    batch_size: int = 16,
    runtime_profile: str,
    candidate_generation: str | None = None,
) -> AmpcorNccWorkspaceMeasurement:
    """Build a measured NCC packet for registry-only unit tests."""
    if candidate_generation is None:
        candidate_generation = runtime_profile
    return AmpcorNccWorkspaceMeasurement(
        operation=_NCC_OPERATION,
        search_shape=search_shape,
        batch_size=batch_size,
        runtime_profile=runtime_profile,
        source_digest=_NCC_SOURCE,
        abi_version=_NCC_ABI,
        candidate_generation=candidate_generation,
        peak_allocated_bytes=batch_size * 26,
        output_bytes=batch_size * 26,
        global_scratch_bytes=0,
        shared_memory_bytes=6144,
        launch_blocks=batch_size,
        threads_per_block=256,
    )


def test_ncc_reference_uses_first_flat_peak_and_excludes_three_by_three() -> None:
    """The reference preserves Torch argmax tie order and sidelobe masking."""
    torch = pytest.importorskip("torch")
    corr = torch.zeros((1, 3, 5), dtype=torch.float64)
    energy = torch.ones_like(corr)
    corr[0, 1, 1] = 2.0
    corr[0, 1, 3] = 2.0
    d_rg, d_az, snr, boundary, valid = ampcor_ncc_postprocess_reference(
        corr,
        energy,
        search_az=1,
        search_rg=2,
        subpixel=False,
        snr_threshold=1.0,
        max_abs_residual=3.0,
    )
    assert d_rg.item() == -1.0
    assert d_az.item() == 0.0
    assert boundary.item() is False
    assert valid.item() is True
    # The second tied peak is in the excluded 3x3 neighborhood of the first.
    assert snr.item() == pytest.approx(6.0)


def test_ncc_reference_nan_sidelobe_and_inclusive_cull() -> None:
    """All-NaN sidelobes are invalid and cull thresholds are inclusive."""
    torch = pytest.importorskip("torch")
    corr = torch.ones((1, 3, 3), dtype=torch.float64)
    corr[0, 1, 1] = 2.0
    energy = torch.ones_like(corr)
    d_rg, d_az, snr, boundary, valid = ampcor_ncc_postprocess_reference(
        corr,
        energy,
        search_az=1,
        search_rg=1,
        subpixel=True,
        snr_threshold=2.0,
        max_abs_residual=0.0,
    )
    assert d_rg.item() == 0.0
    assert d_az.item() == 0.0
    assert torch.isnan(snr).item()
    assert boundary.item() is False
    assert valid.item() is False


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_ncc_reference_nonfinite_snr_is_invalid(value: float) -> None:
    """Positive, negative, and NaN SNR values never pass the valid mask."""
    torch = pytest.importorskip("torch")
    corr = torch.full((1, 3, 3), value, dtype=torch.float64)
    energy = torch.ones_like(corr)
    _, _, snr, _, valid = ampcor_ncc_postprocess_reference(
        corr,
        energy,
        search_az=1,
        search_rg=1,
        subpixel=False,
        snr_threshold=0.0,
        max_abs_residual=1.0,
    )
    assert not bool(valid.item())
    assert not bool(torch.isfinite(snr).item())


def test_ncc_reference_boundary_and_subpixel_guard() -> None:
    """Edge peaks are marked and tiny quadratic denominators do not divide."""
    torch = pytest.importorskip("torch")
    corr = torch.ones((1, 3, 3), dtype=torch.float64)
    corr[0, 0, 0] = 3.0
    energy = torch.ones_like(corr)
    _, _, snr, boundary, valid = ampcor_ncc_postprocess_reference(
        corr,
        energy,
        search_az=1,
        search_rg=1,
        subpixel=True,
        snr_threshold=0.0,
        max_abs_residual=2.0,
    )
    assert boundary.item() is True
    assert torch.isfinite(snr).item()
    assert valid.item() is True


def test_ncc_reference_rejects_invalid_abi_shape_and_dtype() -> None:
    """The prototype reference enforces its float64 contiguous ABI."""
    torch = pytest.importorskip("torch")
    corr = torch.ones((1, 3, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        ampcor_ncc_postprocess_reference(
            corr,
            corr,
            search_az=1,
            search_rg=1,
            subpixel=False,
            snr_threshold=0.0,
            max_abs_residual=1.0,
        )


def test_ncc_registry_requires_exact_abi_and_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NCC registry never widens a qualified ABI or shape lookup."""
    from faninsar.processing.coreg import ampcor_backend

    profile = _ncc_profile()
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_runtime_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        ampcor_backend, "_current_ncc_source_digest", lambda: _NCC_SOURCE
    )
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=16,
        runtime_profile=profile.canonical(),
        source_digest=_NCC_SOURCE,
        abi_version=_NCC_ABI,
        profile=profile,
        candidate_generation=profile.canonical(),
        workspace_measurement=_ncc_measurement(runtime_profile=profile.canonical()),
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert (
        registry.get_ncc(
            "cuda",
            (17, 17),
            batch_size=16,
            runtime_profile=profile.canonical(),
            source_digest=_NCC_SOURCE,
        )
        is candidate
    )
    assert (
        registry.get_ncc(
            "cuda",
            (9, 9),
            batch_size=16,
            runtime_profile=profile.canonical(),
            source_digest=_NCC_SOURCE,
        )
        is None
    )
    assert (
        registry.get_ncc(
            "cuda",
            (17, 17),
            batch_size=16,
            runtime_profile=profile.canonical(),
            source_digest=_NCC_SOURCE,
            abi_version="wrong.abi",
        )
        is None
    )


def test_ncc_candidate_rank_zero_fails_with_candidate_error() -> None:
    """Rank validation precedes every shape index and uses the public error."""
    torch = pytest.importorskip("torch")
    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(3, 3),
        executor=lambda *_args: (),
        batch_size=1,
    )
    scalar = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(AmpcorNccCandidateError, match="rank-3"):
        candidate.execute(
            scalar,
            scalar,
            subpixel=True,
            snr_threshold=1.0,
            max_abs_residual=1.0,
        )


def test_ncc_candidate_accepts_only_prepared_batch_capacity() -> None:
    """A prepared batch accepts a final partial batch but never a larger one."""
    torch = pytest.importorskip("torch")
    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(3, 3),
        executor=lambda correlation, energy, subpixel, threshold, residual: (
            ampcor_ncc_postprocess_reference(
                correlation,
                energy,
                search_az=1,
                search_rg=1,
                subpixel=subpixel,
                snr_threshold=threshold,
                max_abs_residual=residual,
            )
        ),
        batch_size=4,
    )
    correlation = torch.ones((2, 3, 3), dtype=torch.float64)
    energy = torch.ones_like(correlation)
    result = candidate.execute(
        correlation,
        energy,
        subpixel=False,
        snr_threshold=0.0,
        max_abs_residual=1.0,
    )
    assert all(value.shape == (2,) for value in result)
    with pytest.raises(AmpcorNccCandidateError, match="batch_size"):
        candidate.execute(
            torch.ones((5, 3, 3), dtype=torch.float64),
            torch.ones((5, 3, 3), dtype=torch.float64),
            subpixel=False,
            snr_threshold=0.0,
            max_abs_residual=1.0,
        )


@pytest.mark.parametrize(
    ("search_shape", "batch_size"), [((17, 17), 16), ((33, 33), 32)]
)
def test_ncc_registry_dispatches_qualified_shape(
    search_shape: tuple[int, int],
    batch_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both measured A100 surfaces are eligible for registry dispatch."""
    from faninsar.processing.coreg import ampcor_backend

    profile = _ncc_profile()
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_runtime_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        ampcor_backend, "_current_ncc_source_digest", lambda: _NCC_SOURCE
    )
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=search_shape,
        executor=lambda *_args: (),
        batch_size=batch_size,
        runtime_profile=profile.canonical(),
        source_digest=_NCC_SOURCE,
        profile=profile,
        candidate_generation=profile.canonical(),
        workspace_measurement=_ncc_measurement(
            search_shape=search_shape,
            batch_size=batch_size,
            runtime_profile=profile.canonical(),
            candidate_generation=profile.canonical(),
        ),
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert (
        registry.get_ncc(
            "cuda", search_shape, batch_size=batch_size, source_digest=_NCC_SOURCE
        )
        is candidate
    )
    other_batch = 32 if batch_size == 16 else 16
    assert (
        registry.get_ncc(
            "cuda", search_shape, batch_size=other_batch, source_digest=_NCC_SOURCE
        )
        is None
    )


def test_ncc_registry_persistent_quarantine_survives_next_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native execution quarantine disables later public-call lookup."""
    from faninsar.processing.coreg import ampcor_backend

    profile = _ncc_profile()
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_runtime_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        ampcor_backend, "_current_ncc_source_digest", lambda: _NCC_SOURCE
    )
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=16,
        runtime_profile=profile.canonical(),
        source_digest=_NCC_SOURCE,
        profile=profile,
        candidate_generation=profile.canonical(),
        workspace_measurement=_ncc_measurement(runtime_profile=profile.canonical()),
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert registry.get_ncc("cuda", (17, 17), batch_size=16) is candidate
    candidate.quarantine()
    assert registry.get_ncc("cuda", (17, 17), batch_size=16) is None


def test_ncc_registry_quarantines_candidate_after_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current source mutation invalidates a previously prepared candidate."""
    from faninsar.processing.coreg import ampcor_backend

    profile = _ncc_profile()
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_runtime_profile",
        lambda *_args, **_kwargs: profile,
    )
    current_digest = [_NCC_SOURCE]
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_source_digest",
        lambda: current_digest[0],
    )
    generation = profile.canonical()
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=16,
        runtime_profile=generation,
        source_digest=_NCC_SOURCE,
        profile=profile,
        candidate_generation=generation,
        workspace_measurement=_ncc_measurement(runtime_profile=generation),
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert registry.get_ncc("cuda", (17, 17), batch_size=16) is candidate
    current_digest[0] = "f" * 64
    assert registry.get_ncc("cuda", (17, 17), batch_size=16) is None
    assert candidate.quarantined


def test_ncc_registry_does_not_dispatch_unknown_shape() -> None:
    """Correctness qualification alone does not enable an unbenchmarked shape."""
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(9, 9),
        executor=lambda *_args: (),
        batch_size=16,
        performance_eligible=False,
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert registry.get_ncc("cuda", (9, 9), batch_size=16) is None


def test_ncc_registry_ignores_manual_performance_flag() -> None:
    """A caller cannot enable an unprofiled candidate by setting a flag."""
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=8,
        source_digest=_NCC_SOURCE,
        performance_eligible=True,
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert (
        registry.get_ncc("cuda", (17, 17), batch_size=8, source_digest=_NCC_SOURCE)
        is None
    )
    assert registry.get_ncc("cuda", (17, 17), source_digest=_NCC_SOURCE) is None


def test_ncc_registry_rejects_unmeasured_workspace() -> None:
    """A qualified native candidate without a smoke measurement is unusable."""
    profile = _ncc_profile()
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=16,
        runtime_profile=profile.canonical(),
        source_digest=_NCC_SOURCE,
        profile=profile,
        candidate_generation=profile.canonical(),
        workspace_measurement=_ncc_measurement(runtime_profile=profile.canonical()),
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert registry.get_ncc("cuda", (17, 17), batch_size=16) is None


def test_ncc_workspace_identity_and_zero_packet_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace packets must bind to candidate identity and have bytes."""
    from dataclasses import replace

    from faninsar.processing.coreg import ampcor_backend

    profile = _ncc_profile()
    generation = profile.canonical()
    measurement = _ncc_measurement(runtime_profile=generation)
    with pytest.raises(ValueError, match="workspace measurement"):
        AmpcorNccCandidate(
            device="cuda:0",
            search_shape=(17, 17),
            executor=lambda *_args: (),
            batch_size=16,
            runtime_profile=generation,
            source_digest=_NCC_SOURCE,
            profile=profile,
            candidate_generation="different-generation",
            workspace_measurement=measurement,
        )
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_runtime_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        ampcor_backend, "_current_ncc_source_digest", lambda: _NCC_SOURCE
    )
    zero = replace(
        measurement, peak_allocated_bytes=0, output_bytes=0, shared_memory_bytes=0
    )
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=16,
        runtime_profile=generation,
        source_digest=_NCC_SOURCE,
        profile=profile,
        candidate_generation=generation,
        workspace_measurement=zero,
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert registry.get_ncc("cuda", (17, 17), batch_size=16) is None


@pytest.mark.parametrize(
    ("device_name", "torch_version", "cuda_runtime"),
    [
        ("NVIDIA H100 80GB", "2.8.0+cu128", "12.8"),
        ("NVIDIA A100-SXM4-80GB", "2.7.1+cu118", "11.8"),
    ],
)
def test_ncc_registry_rejects_unqualified_runtime(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    torch_version: str,
    cuda_runtime: str,
) -> None:
    """A same-shape candidate is rejected outside the qualified runtime."""
    from faninsar.processing.coreg import ampcor_backend

    candidate_profile = _ncc_profile()
    actual_profile = _ncc_profile(
        device_name=device_name,
        torch_version=torch_version,
        cuda_runtime=cuda_runtime,
    )
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_runtime_profile",
        lambda *_args, **_kwargs: actual_profile,
    )
    monkeypatch.setattr(
        ampcor_backend, "_current_ncc_source_digest", lambda: _NCC_SOURCE
    )
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=16,
        runtime_profile=candidate_profile.canonical(),
        source_digest=_NCC_SOURCE,
        profile=candidate_profile,
        performance_eligible=True,
        candidate_generation=candidate_profile.canonical(),
        workspace_measurement=_ncc_measurement(
            runtime_profile=candidate_profile.canonical(),
        ),
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert (
        registry.get_ncc("cuda", (17, 17), batch_size=16, source_digest=_NCC_SOURCE)
        is None
    )


def test_ncc_registry_rejects_source_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source digest mismatch cannot select the native candidate."""
    from faninsar.processing.coreg import ampcor_backend

    profile = _ncc_profile(source_digest=_NCC_SOURCE)
    monkeypatch.setattr(
        ampcor_backend,
        "_current_ncc_runtime_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        ampcor_backend, "_current_ncc_source_digest", lambda: _NCC_SOURCE
    )
    candidate = AmpcorNccCandidate(
        device="cuda:0",
        search_shape=(17, 17),
        executor=lambda *_args: (),
        batch_size=16,
        runtime_profile=profile.canonical(),
        source_digest=_NCC_SOURCE,
        profile=profile,
        candidate_generation=profile.canonical(),
        workspace_measurement=_ncc_measurement(runtime_profile=profile.canonical()),
    )
    registry = AmpcorBackendRegistry()
    registry.register_ncc(candidate)
    assert (
        registry.get_ncc("cuda", (17, 17), batch_size=16, source_digest="b" * 64)
        is None
    )


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


def test_torch_ncc_batch_uses_prepared_native_candidate() -> None:
    """A full qualified-shape batch dispatches the prepared NCC callable."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets

    calls: list[int] = []

    def execute(
        correlation: object,
        energy: object,
        subpixel: bool,
        snr_threshold: float,
        max_abs_residual: float,
    ) -> tuple[object, ...]:
        calls.append(int(correlation.shape[0]))
        return ampcor_ncc_postprocess_reference(
            correlation,
            energy,
            search_az=1,
            search_rg=1,
            subpixel=subpixel,
            snr_threshold=snr_threshold,
            max_abs_residual=max_abs_residual,
        )

    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(3, 3),
        executor=execute,
        batch_size=2,
    )
    correlation = torch.ones((2, 3, 3), dtype=torch.float64)
    result = offsets._torch_patch_ncc_batch(
        correlation,
        correlation,
        search_az=1,
        search_rg=1,
        subpixel=False,
        ncc_candidate=candidate,
        ncc_quarantine=set(),
    )
    assert all(value.shape == (2,) for value in result)
    assert calls == [2]


def test_torch_ncc_batch_falls_back_for_partial_final_batch() -> None:
    """A fixed prepared NCC capacity never receives a partial final batch."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets

    calls: list[str] = []

    def execute(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        calls.append("native")
        raise AssertionError

    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(3, 3),
        executor=execute,
        batch_size=2,
    )
    values = torch.ones((1, 3, 3), dtype=torch.float64)
    result = offsets._torch_patch_ncc_batch(
        values,
        values,
        search_az=1,
        search_rg=1,
        subpixel=False,
        ncc_candidate=candidate,
        ncc_quarantine=set(),
    )
    assert all(value.shape == (1,) for value in result)
    assert calls == []


def test_torch_ncc_batch_pre_dispatch_failure_falls_back_without_quarantine() -> None:
    """An input eligibility error selects Torch and leaves candidate healthy."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets

    calls: list[str] = []

    def execute(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        calls.append("native")
        raise AssertionError

    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(5, 5),
        executor=execute,
        batch_size=1,
    )
    values = torch.ones((1, 3, 3), dtype=torch.float64)
    result = offsets._torch_patch_ncc_batch(
        values,
        values,
        search_az=1,
        search_rg=1,
        subpixel=False,
        ncc_candidate=candidate,
        ncc_quarantine=set(),
    )
    assert all(value.shape == (1,) for value in result)
    assert calls == []
    assert not candidate.quarantined


def test_torch_ncc_batch_propagates_and_quarantines_native_exception() -> None:
    """A native NCC error propagates and later calls fail closed."""
    torch = pytest.importorskip("torch")
    from faninsar.processing.coreg import offsets

    calls: list[str] = []

    def execute(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        calls.append("native")
        raise RuntimeError

    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(3, 3),
        executor=execute,
        batch_size=1,
    )
    values = torch.ones((1, 3, 3), dtype=torch.float64)
    quarantine: set[int] = set()
    with pytest.raises(AmpcorNccExecutionError):
        offsets._torch_patch_ncc_batch(
            values,
            values,
            search_az=1,
            search_rg=1,
            subpixel=False,
            ncc_candidate=candidate,
            ncc_quarantine=quarantine,
        )
    assert candidate.quarantined
    result = offsets._torch_patch_ncc_batch(
        values,
        values,
        search_az=1,
        search_rg=1,
        subpixel=False,
        ncc_candidate=candidate,
        ncc_quarantine=quarantine,
    )
    assert all(value.shape == (1,) for value in result)
    assert calls == ["native"]
    assert id(candidate) in quarantine


def test_ncc_output_invariant_failure_is_execution_error() -> None:
    """An inconsistent native valid mask is persistent execution failure."""
    torch = pytest.importorskip("torch")

    def invalid_output(
        correlation: object,
        energy: object,
        subpixel: bool,
        snr_threshold: float,
        max_abs_residual: float,
    ) -> tuple[object, ...]:
        result = ampcor_ncc_postprocess_reference(
            correlation,
            energy,
            search_az=1,
            search_rg=1,
            subpixel=subpixel,
            snr_threshold=snr_threshold,
            max_abs_residual=max_abs_residual,
        )
        return (*result[:4], torch.logical_not(result[4]))

    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(3, 3),
        executor=invalid_output,
        batch_size=1,
    )
    values = torch.ones((1, 3, 3), dtype=torch.float64)
    with pytest.raises(AmpcorNccExecutionError, match="valid output"):
        candidate.execute(
            values,
            values,
            subpixel=False,
            snr_threshold=0.0,
            max_abs_residual=1.0,
        )
    assert candidate.quarantined


def test_public_ampcor_keeps_ncc_native_disabled_on_cpu() -> None:
    """The public path never dispatches a CUDA NCC candidate on CPU."""
    pytest.importorskip("torch")
    from faninsar.processing.coreg import estimate_patch_amplitude_shift

    calls: list[str] = []

    def execute(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        calls.append("native")
        raise AssertionError

    candidate = AmpcorNccCandidate(
        device="cpu",
        search_shape=(3, 3),
        executor=execute,
        batch_size=1,
    )
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    result = estimate_patch_amplitude_shift(
        values,
        values,
        window_az=2,
        window_rg=2,
        search_az=1,
        search_rg=1,
        n_az=1,
        n_rg=1,
        margin_az=2,
        margin_rg=2,
        executor="torch",
        device="cpu",
        ampcor_ncc_candidate=candidate,
        batch_size=1,
    )
    assert result.n_attempted == 1
    assert calls == []

"""Tests for the operation-aware native-v2 preparation slice."""

from __future__ import annotations

from pathlib import Path

import pytest

from faninsar.processing.geometry.native_v2 import (
    GeometryOperation,
    NativeBackend,
    NativeBuilder,
    NativeBuildRequest,
    OpenMPTelemetry,
    PreparationStatus,
    openmp_provider_for_platform,
    qualify_openmp_telemetry,
    select_native_sources,
)


@pytest.mark.parametrize(
    ("operation", "backend", "symbol"),
    [
        (GeometryOperation.GEO2RDR, NativeBackend.CPU, "faninsar_geo2rdr_v2_cpu"),
        (GeometryOperation.RDR2GEO, NativeBackend.CUDA, "faninsar_rdr2geo_v2_cuda"),
    ],
)
def test_operation_and_source_selection_is_exact(
    operation: GeometryOperation,
    backend: NativeBackend,
    symbol: str,
) -> None:
    """Each operation/device pair selects its own ABI symbol and source."""
    plan = NativeBuilder().plan(
        NativeBuildRequest(operation, backend, source_root="src")
    )

    assert plan.geometry_symbol == symbol
    assert plan.sources == select_native_sources(operation, backend, "src")
    expected_suffix = ".cu" if backend is NativeBackend.CUDA else ".cpp"
    assert plan.sources[-1].suffix == expected_suffix


def test_default_source_root_is_package_local_after_cwd_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default CPU and CUDA plans resolve sources independently of cwd."""
    monkeypatch.chdir(tmp_path)
    expected_root = (
        Path(__file__).parents[3] / "faninsar" / "processing" / "geometry" / "native_v2"
    )

    for backend in (NativeBackend.CPU, NativeBackend.CUDA):
        plan = NativeBuilder().plan(
            NativeBuildRequest(
                GeometryOperation.GEO2RDR,
                backend,
                platform="windows",
            )
        )

        assert plan.sources
        assert all(source.is_absolute() for source in plan.sources)
        assert all(source.is_relative_to(expected_root) for source in plan.sources)
        assert all(source.exists() for source in plan.sources)


def test_native_plans_include_both_operation_sources_and_shared_binding() -> None:
    """CPU and CUDA plans carry both operation units behind one binding module."""
    source_root = (
        Path(__file__).parents[3] / "faninsar" / "processing" / "geometry" / "native_v2"
    )
    for backend in (NativeBackend.CPU, NativeBackend.CUDA):
        plan = NativeBuilder().plan(
            NativeBuildRequest(
                GeometryOperation.GEO2RDR,
                backend,
                source_root=source_root,
                platform="linux",
                compiler="g++",
            )
        )
        assert plan.sources[0] == source_root / "bindings.cpp"
        assert any("geo2rdr" in source.name for source in plan.sources)
        assert any("rdr2geo" in source.name for source in plan.sources)
        assert all(source.exists() for source in plan.sources)
    cuda_plan = NativeBuilder().plan(
        NativeBuildRequest(
            GeometryOperation.RDR2GEO,
            NativeBackend.CUDA,
            source_root=source_root,
            platform="linux",
        )
    )
    assert "-DFANINSAR_NATIVE_V2_CUDA=1" in cuda_plan.compile_flags
    assert "-DFANINSAR_NATIVE_V2_BLOCKS_PER_SM=4" in cuda_plan.compile_flags


def test_cuda_launch_profile_is_a_build_request_option() -> None:
    """CUDA occupancy candidates are selected at preparation time."""
    plan = NativeBuilder().plan(
        NativeBuildRequest(
            GeometryOperation.RDR2GEO,
            NativeBackend.CUDA,
            cuda_blocks_per_sm=10,
        )
    )

    assert "-DFANINSAR_NATIVE_V2_BLOCKS_PER_SM=10" in plan.compile_flags


def test_cuda_launch_profile_rejects_non_positive_values() -> None:
    """A launch profile cannot create an empty worker grid."""
    with pytest.raises(ValueError, match="cuda_blocks_per_sm must be positive"):
        NativeBuildRequest(
            GeometryOperation.RDR2GEO,
            NativeBackend.CUDA,
            cuda_blocks_per_sm=0,
        )


def test_linux_cpu_plan_separates_openmp_flags_and_fails_closed() -> None:
    """CPU plans retain OpenMP metadata but remain unavailable until qualified."""
    plan = NativeBuilder().plan(
        NativeBuildRequest(
            GeometryOperation.GEO2RDR,
            NativeBackend.CPU,
            platform="linux",
            compiler="g++",
        )
    )

    assert not plan.supported
    assert "not scientifically qualified" in plan.unsupported_reason
    assert plan.compile_flags == ("-fopenmp", "-DFANINSAR_OPENMP_RUNTIME_LIBGOMP")
    assert plan.link_flags == ("-fopenmp",)
    assert plan.runtime_name == "libgomp"


def test_macos_without_libomp_is_explicitly_unsupported(tmp_path: Path) -> None:
    """MacOS does not silently degrade to a serial CPU artifact."""
    result = openmp_provider_for_platform("macos", libomp_root=tmp_path / "missing")
    candidate = NativeBuilder().prepare(
        NativeBuildRequest(
            GeometryOperation.RDR2GEO,
            NativeBackend.CPU,
            platform="macos",
            libomp_root=tmp_path / "missing",
        )
    )

    assert not result.supported
    assert result.reason
    assert candidate.status is PreparationStatus.UNSUPPORTED
    assert candidate.reason == result.reason


def test_macos_libomp_provider_has_absolute_runtime_flags(tmp_path: Path) -> None:
    """An explicit LLVM libomp root produces deterministic macOS flags."""
    (tmp_path / "include").mkdir()
    (tmp_path / "lib").mkdir()
    result = openmp_provider_for_platform("darwin", libomp_root=tmp_path)

    assert result.supported
    assert result.flags.include_dirs == (tmp_path / "include",)
    assert result.flags.library_dirs == (tmp_path / "lib",)
    assert f"-I{tmp_path / 'include'}" in result.flags.compile_flags
    assert f"-L{tmp_path / 'lib'}" in result.flags.link_flags


def test_macos_libomp_provider_discovers_pixi_environment_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Pixi environment prefix is a valid LLVM libomp installation root."""
    (tmp_path / "include").mkdir()
    (tmp_path / "lib").mkdir()
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

    result = openmp_provider_for_platform("macos")

    assert result.supported
    assert result.flags.runtime_path == tmp_path / "lib" / "libomp.dylib"
    assert f"-Wl,-rpath,{tmp_path / 'lib'}" in result.flags.link_flags


def test_unqualified_native_candidate_cannot_compile_or_dispatch() -> None:
    """Unqualified native sources fail closed before any build callback."""
    calls: list[str] = []

    def build(_plan: object) -> Path:
        calls.append("build")
        return Path("extension.so")

    candidate = NativeBuilder().prepare(
        NativeBuildRequest(
            GeometryOperation.GEO2RDR,
            NativeBackend.CPU,
            platform="linux",
        ),
        build=build,
        entry_point=lambda value: value + 1,
    )

    assert candidate.status is PreparationStatus.UNSUPPORTED
    assert calls == []


def test_qualification_binds_exact_extension_and_geometry_symbol() -> None:
    """Telemetry from another operation or artifact cannot qualify a candidate."""
    telemetry = OpenMPTelemetry(
        extension_name="faninsar_geo2rdr_v2_cpu",
        geometry_symbol="faninsar_geo2rdr_v2_cpu",
        openmp_defined=True,
        runtime_name="libgomp",
        observed_thread_ids=(1, 2),
        fixture_point_count=4,
        processed_point_count=4,
        visit_counts=(1, 1, 1, 1),
        observed_affinity=(0, 1),
    )
    result = qualify_openmp_telemetry(
        telemetry,
        expected_extension="faninsar_rdr2geo_v2_cpu",
        expected_geometry_symbol="faninsar_rdr2geo_v2_cpu",
        expected_runtime="libgomp",
        requested_threads=2,
        expected_affinity=(0, 1),
    )

    assert not result.qualified
    assert result.reason == "extension identity mismatch"


def test_qualification_requires_exact_coverage_and_thread_profile() -> None:
    """A serial or partial loop is never reported as qualified OpenMP."""
    telemetry = OpenMPTelemetry(
        extension_name="faninsar_geo2rdr_v2_cpu",
        geometry_symbol="faninsar_geo2rdr_v2_cpu",
        openmp_defined=True,
        runtime_name="libgomp",
        observed_thread_ids=(1,),
        fixture_point_count=3,
        processed_point_count=2,
        visit_counts=(1, 1, 0),
    )
    result = qualify_openmp_telemetry(
        telemetry,
        expected_extension="faninsar_geo2rdr_v2_cpu",
        expected_geometry_symbol="faninsar_geo2rdr_v2_cpu",
        expected_runtime="libgomp",
        requested_threads=2,
    )

    assert not result.qualified
    assert "exactly the fixture point count" in result.reason

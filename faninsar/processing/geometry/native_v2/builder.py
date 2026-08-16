"""Operation-aware native geometry build planning and preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .openmp import openmp_provider_for_platform

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class NativePreparationError(RuntimeError):
    """Raised when a candidate cannot be dispatched."""

    def __init__(self) -> None:
        """Create the fixed missing-entry-point error."""
        super().__init__("prepared native candidate has no entry point")


class NativeOperation(StrEnum):
    """Exact operation identities supported by the native v2 builder."""

    GEO2RDR = "geo2rdr"
    RDR2GEO = "rdr2geo"
    AMPCOR_PREFIX_ENERGY = "ampcor_prefix_energy.v1"


# Keep the P25 public name valid for geometry callers while the builder grows
# a neutral operation namespace for non-geometry native primitives.
GeometryOperation = NativeOperation


class NativeBackend(StrEnum):
    """Native execution devices."""

    CPU = "cpu"
    CUDA = "cuda"


class PreparationStatus(StrEnum):
    """Candidate preparation outcome."""

    PREPARED = "prepared"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NativeBuildRequest:
    """Inputs needed to make one operation/device build plan."""

    operation: NativeOperation
    backend: NativeBackend
    source_root: Path = Path()
    platform: str | None = None
    compiler: str | None = None
    libomp_root: Path | None = None
    cuda_blocks_per_sm: int = 4

    def __post_init__(self) -> None:
        """Normalize string enum inputs and source paths."""
        object.__setattr__(self, "operation", NativeOperation(self.operation))
        object.__setattr__(self, "backend", NativeBackend(self.backend))
        object.__setattr__(self, "source_root", Path(self.source_root))
        if self.libomp_root is not None:
            object.__setattr__(self, "libomp_root", Path(self.libomp_root))
        if self.cuda_blocks_per_sm <= 0:
            message = "cuda_blocks_per_sm must be positive"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Deterministic native build plan with compile/link separation."""

    operation: NativeOperation
    backend: NativeBackend
    extension_name: str
    geometry_symbol: str
    sources: tuple[Path, ...]
    compile_flags: tuple[str, ...]
    link_flags: tuple[str, ...]
    include_dirs: tuple[Path, ...] = ()
    library_dirs: tuple[Path, ...] = ()
    runtime_name: str | None = None
    supported: bool = True
    unsupported_reason: str = ""

    @property
    def all_flags(self) -> tuple[str, ...]:
        """Return flags for diagnostics without conflating build phases."""
        return self.compile_flags + self.link_flags


class BuildExecutor(Protocol):
    """Explicit preparation-time build callback."""

    def __call__(self, plan: BuildPlan) -> Path:
        """Build ``plan`` and return the resulting extension path."""


@dataclass(frozen=True, slots=True)
class PreparedNativeCandidate:
    """Prepared candidate consumed by dispatch without compiler access."""

    plan: BuildPlan
    status: PreparationStatus
    artifact: Path | None = None
    reason: str = ""
    _entry_point: Callable[..., object] | None = field(default=None, repr=False)
    native_context_inputs: Mapping[str, object] | None = None

    def dispatch(self, *args: object, **kwargs: object) -> object:
        """Invoke the already prepared extension entry point.

        Raises
        ------
        RuntimeError
            If preparation was unsupported, failed, or supplied no entry point.

        """
        if self.status is not PreparationStatus.PREPARED:
            raise RuntimeError(self.reason or "native candidate is not prepared")
        if self._entry_point is None:
            raise NativePreparationError
        return self._entry_point(*args, **kwargs)


def select_native_sources(
    operation: NativeOperation | str,
    backend: NativeBackend | str,
    source_root: str | Path = ".",
) -> tuple[Path, ...]:
    """Select operation/device-specific source names deterministically.

    Parameters
    ----------
    operation : GeometryOperation or str
        Exact ``geo2rdr`` or ``rdr2geo`` operation.
    backend : NativeBackend or str
        ``cpu`` or ``cuda`` native implementation.
    source_root : path-like, optional
        Root containing the v2 native source tree.

    Returns
    -------
        tuple[pathlib.Path, ...]
        Shared bindings, common ABI source, and the operation-specific source.
        CUDA uses ``.cu`` files while CPU uses ``.cpp`` files.

    """
    operation = NativeOperation(operation)
    backend = NativeBackend(backend)
    root = Path(source_root)
    if operation is NativeOperation.AMPCOR_PREFIX_ENERGY:
        return (
            root / "ampcor_bindings.cpp",
            root / "ampcor_prefix_energy.cpp",
        )
    suffix = ".cu" if backend is NativeBackend.CUDA else ".cpp"
    # Keep the binding translation unit first: torch's extension loader needs
    # exactly one ``PYBIND11_MODULE`` unit in every build artifact.  CPU uses
    # the shared ABI implementation and both operation translation units;
    # CUDA has one common header and two operation CUDA translation units.
    if backend is NativeBackend.CUDA:
        return (
            root / "bindings.cpp",
            root / "cuda" / "geo2rdr_cuda.cu",
            root / "cuda" / "rdr2geo_tcn_cuda.cu",
        )
    return (
        root / "bindings.cpp",
        root / "native_v2_abi.cpp",
        root / f"{GeometryOperation.GEO2RDR.value}{suffix}",
        root / f"{GeometryOperation.RDR2GEO.value}{suffix}",
    )


class NativeBuilder:
    """Plan and explicitly prepare operation-aware native extensions.

    ``dispatch`` only looks up a candidate prepared through :meth:`prepare`;
    it has no compiler callback and cannot start compilation on a cache miss.
    """

    def plan(self, request: NativeBuildRequest) -> BuildPlan:
        """Create a deterministic build plan without compiling anything."""
        operation_name = request.operation.value.replace(".", "_")
        extension = f"faninsar_{operation_name}_v2_{request.backend.value}"
        if request.operation is NativeOperation.AMPCOR_PREFIX_ENERGY:
            extension = f"faninsar_{operation_name}_{request.backend.value}"
        symbol = extension
        sources = select_native_sources(
            request.operation,
            request.backend,
            request.source_root,
        )
        if request.operation is NativeOperation.AMPCOR_PREFIX_ENERGY:
            if request.backend is NativeBackend.CUDA:
                return BuildPlan(
                    request.operation,
                    request.backend,
                    extension,
                    symbol,
                    sources,
                    (),
                    (),
                    supported=False,
                    unsupported_reason=(
                        "Ampcor prefix energy has no native CUDA implementation"
                    ),
                )
            provider = openmp_provider_for_platform(
                request.platform,
                compiler=request.compiler,
                libomp_root=request.libomp_root,
            )
            flags = provider.flags
            compile_flags = flags.compile_flags
            if flags.runtime_name == "libomp":
                compile_flags += ("-DFANINSAR_OPENMP_RUNTIME_LIBOMP",)
            elif flags.runtime_name == "libgomp":
                compile_flags += ("-DFANINSAR_OPENMP_RUNTIME_LIBGOMP",)
            return BuildPlan(
                request.operation,
                request.backend,
                extension,
                symbol,
                sources,
                compile_flags,
                flags.link_flags,
                flags.include_dirs,
                flags.library_dirs,
                flags.runtime_name or None,
                supported=flags.supported,
                unsupported_reason=flags.reason,
            )
        if request.backend is NativeBackend.CPU:
            provider = openmp_provider_for_platform(
                request.platform,
                compiler=request.compiler,
                libomp_root=request.libomp_root,
            )
            flags = provider.flags
            compile_flags = flags.compile_flags
            if flags.runtime_name == "libomp":
                compile_flags += ("-DFANINSAR_OPENMP_RUNTIME_LIBOMP",)
            elif flags.runtime_name == "libgomp":
                compile_flags += ("-DFANINSAR_OPENMP_RUNTIME_LIBGOMP",)
            return BuildPlan(
                request.operation,
                request.backend,
                extension,
                symbol,
                sources,
                compile_flags,
                flags.link_flags,
                flags.include_dirs,
                flags.library_dirs,
                flags.runtime_name or None,
                False,
                (
                    "native-v2 CPU kernels are experimental and not scientifically "
                    "qualified"
                    if flags.supported
                    else flags.reason
                ),
            )
        return BuildPlan(
            request.operation,
            request.backend,
            extension,
            symbol,
            sources,
            (
                "-O3",
                "-DFANINSAR_NATIVE_V2_CUDA=1",
                f"-DFANINSAR_NATIVE_V2_BLOCKS_PER_SM={request.cuda_blocks_per_sm}",
            ),
            (),
            (request.source_root / "cuda",),
            runtime_name=None,
            supported=False,
            unsupported_reason=(
                "native-v2 CUDA kernels are experimental and not scientifically "
                "qualified"
            ),
        )

    def prepare(
        self,
        request: NativeBuildRequest,
        *,
        build: BuildExecutor | None = None,
        entry_point: Callable[..., object] | None = None,
    ) -> PreparedNativeCandidate:
        """Prepare one candidate, optionally invoking an explicit build callback.

        The callback is intentionally accepted only here.  Dispatch has no
        callback parameter and therefore cannot compile on demand.
        """
        plan = self.plan(request)
        if not plan.supported:
            return PreparedNativeCandidate(
                plan,
                PreparationStatus.UNSUPPORTED,
                reason=plan.unsupported_reason,
            )
        artifact = build(plan) if build is not None else None
        if build is not None and artifact is None:
            return PreparedNativeCandidate(
                plan,
                PreparationStatus.FAILED,
                reason="explicit native build callback returned no artifact",
            )
        return PreparedNativeCandidate(
            plan,
            PreparationStatus.PREPARED,
            Path(artifact) if artifact is not None else None,
            _entry_point=entry_point,
        )

    def dispatch(
        self,
        candidate: PreparedNativeCandidate,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Dispatch a prepared candidate without compiling or fallback."""
        return candidate.dispatch(*args, **kwargs)

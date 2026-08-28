# ruff: noqa: EM101, EM102, TRY003, PLR0911

"""Ampcor-style production helper for P20/P25/P26 geometry (PROPOSAL-0031).

Callers pass ``device`` only. UUID, native executor, and candidate keys stay
inside this module. Failed native prepare stays on same-device Torch.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.backend_dispatch import (
    DispatchError,
    resolve_geometry_device,
)
from faninsar.processing.geometry.dem import ConstantHeightDEM
from faninsar.processing.geometry.native_v2.builder import (
    NativeBackend,
    NativeBuilder,
    NativeBuildRequest,
    NativeOperation,
    PreparationStatus,
)
from faninsar.processing.geometry.public import execute_geometry, prepare_geometry
from faninsar.processing.geometry.transforms import TransformResult
from faninsar.processing.geometry.v2 import (
    GeometryValidationError,
    Operation,
    SolverSettings,
    TransformResultV2,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.geometry.public import PreparedGeometry
    from faninsar.processing.geometry.transforms import RadarGeometryModel
    from faninsar.typing import DeviceLike

logger = setup_logger(__name__)

_NATIVE_SOURCE_ROOT = Path(__file__).resolve().parent / "native_v2"
_CUDA_MODULES: dict[str, object] = {}
_PREPARED_CACHE: dict[tuple[object, ...], object] = {}
_PREPARED_CACHE_LIMIT = 12
_GPU_DEM_TABLE: dict[tuple[str, str], dict[str, object]] = {}
_RDR2GEO_DEM_FIELDS = ("dem_values", "dem_metadata", "dem_height_bounds")


def _require_device(device: DeviceLike) -> object:
    """Resolve a production device and reject MPS after parse_device."""
    resolved = resolve_geometry_device(device)
    if resolved.type == "mps":
        message = (
            "production geometry supports only cpu and cuda after Newton "
            "deletion; got mps"
        )
        logger.error(message)
        raise DispatchError(message)
    if resolved.type not in {"cpu", "cuda"}:
        message = f"production geometry supports only cpu and cuda, got {device!r}"
        logger.error(message)
        raise DispatchError(message)
    return resolved


def _device_cache_identity(
    resolved: object,
    physical_uuid: str | None,
    mig_uuid: str | None,
) -> str:
    """Stable device identity for prepared cache and GPU DEM residency."""
    kind = getattr(resolved, "type", None)
    if kind == "cuda" and physical_uuid:
        if mig_uuid:
            return f"cuda-uuid:{physical_uuid}:mig:{mig_uuid}"
        return f"cuda-uuid:{physical_uuid}"
    if kind:
        index = getattr(resolved, "index", None)
        if index is not None:
            return f"{kind}:{index}"
        return str(kind)
    return str(resolved)


def _native_dem_context_bound(context: object) -> bool:
    """Return True when Native rdr2geo context carries DEM raster tensors."""
    if not isinstance(context, dict):
        return False
    return all(context.get(name) is not None for name in _RDR2GEO_DEM_FIELDS)


def _resident_gpu_dem(
    dem_digest: str,
    device_identity: str,
    torch_device: object,
    dem: DEMSampler | None,
) -> dict[str, object]:
    """Return device DEM tensors, uploading once per digest and device.

    Distinct prepared shapes share this table (PROPOSAL-0034). A digest
    miss re-uploads; a hit aliases the resident tensors.
    """
    import torch

    key = (dem_digest, device_identity)
    cached = _GPU_DEM_TABLE.get(key)
    if cached is not None:
        return cached
    dem_values, dem_metadata, dem_bounds = _dem_native_arrays(dem)
    resident = {
        "dem_values": torch.as_tensor(
            dem_values, dtype=torch.float64, device=torch_device
        ).contiguous(),
        "dem_metadata": torch.as_tensor(
            dem_metadata, dtype=torch.float64, device=torch_device
        ),
        "dem_height_bounds": torch.as_tensor(
            dem_bounds, dtype=torch.float64, device=torch_device
        ),
    }
    _GPU_DEM_TABLE[key] = resident
    return resident


def _prepared_cache_key(
    op: Operation,
    model: RadarGeometryModel,
    *,
    shape: tuple[int, ...],
    resolved: object,
    dem: DEMSampler | None,
    solver: SolverSettings,
    physical_uuid: str | None,
    mig_uuid: str | None,
) -> tuple[object, ...]:
    """Operational prepared identity (PROPOSAL-0034).

    Keyed by operation, model_digest, orbit digest, DEM digest, shape,
    solver.for_operation, and device UUID. Excludes toolchain/runtime and
    ``id(model)``.
    """
    from faninsar.processing.geometry.torch_backends_v2 import (
        _dem_digest,
        _model_digests,
    )

    model_digest, orbit_digest = _model_digests(model)
    solver_items = tuple(sorted(solver.for_operation(op).items()))
    return (
        op.value,
        model_digest,
        orbit_digest,
        _dem_digest(dem),
        shape,
        solver_items,
        _device_cache_identity(resolved, physical_uuid, mig_uuid),
    )


def _cuda_uuids(resolved: object) -> tuple[str, str | None]:
    """Read the CUDA physical UUID (and MIG UUID when Torch exposes it)."""
    import torch

    props = torch.cuda.get_device_properties(resolved)
    physical = str(props.uuid)
    mig = getattr(props, "mig_uuid", None)
    if mig is None:
        mig = getattr(props, "uuid_mig", None)
    return physical, str(mig) if mig is not None else None


def _native_build_dir(operation: NativeOperation | None = None) -> Path:
    """Return the operation-isolated P25 cache directory.

    Parameters
    ----------
    operation : NativeOperation, optional
        Geometry operation whose Ninja graph and objects must be isolated.
        A shared fallback is retained for compatibility with diagnostics that
        run before an operation is selected.

    """
    from faninsar.compute.cache import resolve_compile_cache_dir

    root = resolve_compile_cache_dir().parent / "native_v2_geometry"
    path = root / (operation.value if operation is not None else "shared")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pixi_nvcc() -> Path | None:
    """Return the interpreter-local nvcc, never ``/usr/bin/nvcc``."""
    nvcc = Path(sys.executable).resolve().parent / "nvcc"
    if not nvcc.is_file():
        return None
    if nvcc.resolve() == Path("/usr/bin/nvcc").resolve():
        return None
    return nvcc


def _prepend_packaged_cuda_toolchain(
    operation: NativeOperation | None = None,
) -> Path | None:
    """Bind Torch cpp_extension to the pixi CUDA 12 toolkit.

    Login PATH on the A100 host finds system CUDA 10 ``/usr/bin/nvcc`` first.
    Torch records ``CUDA_HOME`` at ``cpp_extension`` import time, so this
    must rewrite both the environment and the already-imported module.
    """
    nvcc = _pixi_nvcc()
    if nvcc is None:
        logger.error(
            "pixi nvcc is required for native CUDA geometry; "
            "refusing system /usr/bin/nvcc"
        )
        return None
    pixi_bin = nvcc.parent
    cuda_home = pixi_bin.parent
    os.environ["PATH"] = str(pixi_bin) + os.pathsep + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["CUDA_NVCC_EXECUTABLE"] = str(nvcc)
    os.environ["CUDACXX"] = str(nvcc)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    try:
        from torch.utils import cpp_extension

        cpp_extension.CUDA_HOME = str(cuda_home)
    except ImportError:
        pass
    _scrub_stale_system_nvcc_ninja(nvcc, operation)
    return nvcc


def _scrub_stale_system_nvcc_ninja(
    nvcc: Path,
    operation: NativeOperation | None = None,
) -> None:
    """Drop stale Ninja graphs and abandoned JIT locks for one operation.

    ``torch.utils.cpp_extension`` coordinates JIT builds with an existence
    based ``FileBaton`` lock.  A process terminated during compilation can
    leave that zero-byte lock behind; subsequent callers then wait forever
    even though no compiler owns the file.  The build graph is also tied to
    absolute source and Torch include paths, so a graph from another checkout
    must not be reused silently.

    Only the operation-specific cache directory is touched.  A lock that is
    still open by any process is preserved and its graph is left untouched.
    Stale graphs are moved atomically aside instead of deleting files in
    place, so a compiler that races with the probe retains an isolated build
    directory.
    """
    build_dir = _native_build_dir(operation)
    # All FanInSAR preparation callers take this advisory guard.  It does not
    # replace FileBaton (the external builder still owns that lock), but it
    # serializes stale-lock cleanup among our callers before the atomic rename.
    guard_fd: int | None = None
    fcntl_module: object | None = None
    try:
        import fcntl as fcntl_module

        guard_path = build_dir.parent / f"{build_dir.name}.guard"
        guard_fd = os.open(guard_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl_module.flock(guard_fd, fcntl_module.LOCK_EX)
    except (ImportError, OSError):
        logger.exception("cannot acquire native cache guard for %s", build_dir)
        if guard_fd is not None:
            os.close(guard_fd)
        return

    claimed_fd: int | None = None
    try:
        lock = build_dir / "lock"
        lock_active = _native_lock_is_open(lock)
        ninja = build_dir / "build.ninja"
        if lock_active or not ninja.is_file():
            if lock.exists() and not lock_active:
                logger.warning("removing abandoned native JIT lock %s", lock)
                lock.unlink(missing_ok=True)
            return
        text = ninja.read_text(errors="replace")
        expected_nvcc = f"nvcc = {nvcc}"
        expected_source_root = str(_NATIVE_SOURCE_ROOT)
        stale_toolchain = "nvcc = /usr/bin/nvcc" in text or expected_nvcc not in text
        stale_source = expected_source_root not in text
        # The include path is emitted by the currently running Torch
        # installation.  Checking it prevents a graph from another
        # environment being considered up to date merely because mtimes match.
        try:
            import torch

            torch_root = str(Path(torch.__file__).resolve().parent)
        except (ImportError, OSError):
            torch_root = ""
        stale_torch = bool(torch_root) and torch_root not in text
        if not (stale_toolchain or stale_source or stale_torch):
            if lock.exists() and not lock_active:
                logger.warning("removing abandoned native JIT lock %s", lock)
                lock.unlink(missing_ok=True)
            return

        # If no FileBaton marker exists, claim it before renaming.  A builder
        # that wins the race creates the marker first; in that case return and
        # leave its live directory completely untouched.
        if not lock.exists():
            try:
                claimed_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                logger.warning(
                    "native build appeared while isolating %s; preserving it",
                    build_dir,
                )
                return
        stale_dir = build_dir.with_name(
            f"{build_dir.name}.stale-{os.getpid()}-{time.time_ns()}"
        )
        try:
            build_dir.rename(stale_dir)
        except OSError:
            logger.exception(
                "cannot isolate stale native ninja graph %s",
                build_dir,
            )
            # Do not strand the lock we claimed if the atomic rename fails;
            # otherwise the next caller would wait on our now-closed marker.
            # Compare inodes before unlinking so a concurrent replacement is
            # never mistaken for the marker created by this process.
            if claimed_fd is not None:
                try:
                    lock_stat = lock.stat()
                    claimed_stat = os.fstat(claimed_fd)
                    if (
                        lock_stat.st_dev == claimed_stat.st_dev
                        and lock_stat.st_ino == claimed_stat.st_ino
                    ):
                        lock.unlink(missing_ok=True)
                except OSError:
                    logger.exception("cannot release claimed native lock %s", lock)
            return
        logger.warning(
            "isolated stale native ninja graph %s -> %s "
            "(toolchain=%s source=%s torch=%s); rebuilding with %s",
            build_dir,
            stale_dir,
            stale_toolchain,
            stale_source,
            stale_torch,
            nvcc,
        )
    finally:
        if claimed_fd is not None:
            os.close(claimed_fd)
        if guard_fd is not None and fcntl_module is not None:
            fcntl_module.flock(guard_fd, fcntl_module.LOCK_UN)
            os.close(guard_fd)


def _native_lock_is_open(lock: Path) -> bool:
    """Return whether a live process currently has a native JIT lock open.

    PyTorch's ``FileBaton`` lock is an existence marker rather than an OS
    advisory lock.  On Linux, inspecting ``/proc/*/fd`` lets us distinguish a
    live compiler from a stale marker left by a terminated process.  If the
    process table cannot be inspected, fail closed and retain the lock.
    """
    if not lock.exists():
        return False
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return True
    try:
        target = lock.resolve()
        for process in proc_root.iterdir():
            if not process.name.isdigit():
                continue
            fd_root = process / "fd"
            try:
                descriptors = tuple(fd_root.iterdir())
            except OSError:
                continue
            for descriptor in descriptors:
                try:
                    if descriptor.resolve() == target:
                        return True
                except OSError:
                    continue
    except OSError:
        return True
    return False


def _try_load_cuda_module(operation: NativeOperation) -> object | None:
    """Prepare a CUDA geo module; return None on any failure."""
    cached = _CUDA_MODULES.get(operation.value)
    if cached is not None:
        return cached
    nvcc = _prepend_packaged_cuda_toolchain(operation)
    if nvcc is None:
        return None
    import torch
    from torch.utils import cpp_extension

    if not torch.cuda.is_available():
        return None
    request = NativeBuildRequest(
        operation,
        NativeBackend.CUDA,
        source_root=_NATIVE_SOURCE_ROOT,
    )
    builder = NativeBuilder()
    plan = builder.plan(request)
    if not plan.supported:
        logger.warning(
            "CUDA %s native plan unsupported: %s",
            operation.value,
            plan.unsupported_reason,
        )
        return None
    module_holder: dict[str, object] = {}

    def build(plan_to_build: object) -> Path:
        """Explicit P25 cpp_extension callback; never compile on import."""
        plan_value = plan_to_build
        cpp_extension.CUDA_HOME = str(nvcc.parent.parent)
        module = cpp_extension.load(
            name=plan_value.extension_name,
            sources=[str(source) for source in plan_value.sources],
            extra_cflags=list(plan_value.compile_flags),
            extra_cuda_cflags=list(plan_value.compile_flags),
            extra_ldflags=list(plan_value.link_flags),
            extra_include_paths=[str(path) for path in plan_value.include_dirs],
            build_directory=str(_native_build_dir(operation)),
            with_cuda=True,
            verbose=False,
        )
        ninja = _native_build_dir(operation) / "build.ninja"
        if ninja.is_file() and "nvcc = /usr/bin/nvcc" in ninja.read_text(
            errors="replace"
        ):
            raise RuntimeError(
                "native ninja still invokes /usr/bin/nvcc; "
                f"expected pixi nvcc {nvcc}"
            )
        module_holder["module"] = module
        return Path(getattr(module, "__file__", _native_build_dir(operation)))

    try:
        prepared = builder.prepare(request, build=build)
    except Exception as error:
        logger.warning("CUDA %s native prepare failed: %s", operation.value, error)
        return None
    if prepared.status is not PreparationStatus.PREPARED:
        logger.warning(
            "CUDA %s native prepare status=%s reason=%s",
            operation.value,
            prepared.status,
            prepared.reason,
        )
        return None
    module = module_holder.get("module")
    if module is not None:
        _CUDA_MODULES[operation.value] = module
    return module


def to_transform_result(result: TransformResultV2) -> TransformResult:
    """Project the fourteen-field v2 result onto the production value type."""
    return TransformResult(
        latitude_deg=result.latitude_deg,
        longitude_deg=result.longitude_deg,
        height_m=result.height_m,
        range_index=result.range_index,
        azimuth_index=result.azimuth_index,
        converged=result.converged,
        residual_range_m=result.residual_range_m,
        residual_doppler_hz=result.residual_doppler_hz,
    )


def _prepare_geometry_identities(
    op: Operation,
    model: RadarGeometryModel,
    *,
    shape: tuple[int, ...],
    resolved: object,
    dem: DEMSampler | None,
    solver: SolverSettings,
    native_ok: bool,
    loaded: object,
    native_key: object,
    native_context: object,
    physical_uuid: str | None,
    mig_uuid: str | None,
    use_compile: bool,
) -> PreparedGeometry:
    """Register Native (when bound) and optional Compile, then Eager."""
    try:
        return prepare_geometry(
            op,
            model,
            shape=shape,
            device=resolved,
            dem=dem,
            settings=solver,
            native_executor=loaded if native_ok else None,
            native_key=native_key if native_ok else None,
            native_context_inputs=native_context if native_ok else None,
            native_correctness_qualified=native_ok,
            native_performance_eligible=native_ok,
            compile=use_compile,
            compile_performance_eligible=use_compile,
            physical_uuid=physical_uuid,
            mig_uuid=mig_uuid,
        )
    except (DispatchError, GeometryValidationError) as error:
        logger.warning(
            "native geometry registration failed (%s); continuing on Torch",
            error,
        )
        return prepare_geometry(
            op,
            model,
            shape=shape,
            device=resolved,
            dem=dem,
            settings=solver,
            native_correctness_qualified=False,
            native_performance_eligible=False,
            compile=use_compile,
            compile_performance_eligible=use_compile,
            physical_uuid=physical_uuid,
            mig_uuid=mig_uuid,
        )


def prepare_production_geometry(
    operation: Operation | str,
    model: RadarGeometryModel,
    *,
    device: DeviceLike,
    shape: Sequence[int],
    dem: DEMSampler | None = None,
    settings: SolverSettings | None = None,
) -> PreparedGeometry:
    """Prepare production geometry without exposing UUID or native executor.

    Parameters
    ----------
    operation : Operation or str
        ``geo2rdr`` or ``rdr2geo``.
    model : RadarGeometryModel
        Production geometry model.
    device : DeviceLike
        Required device request. ``auto`` resolves to cpu or cuda first.
        ``mps`` fails closed after resolution.
    shape : sequence of int
        Exact input shape for this prepare (tile shape, not full grid).
    dem : DEMSampler, optional
        Height sampler for ``rdr2geo``.
    settings : SolverSettings, optional
        Iteration and tolerance settings.

    Returns
    -------
    PreparedGeometry
        Ready for :func:`execute_geometry`.  CUDA prepares Native when the
        operation ABI is bound (``geo2rdr`` orbit context, ``rdr2geo`` orbit
        plus DEM raster/affine; PROPOSAL-0020 / PROPOSAL-0025 / PROPOSAL-0026).
        ``torch.compile`` is prepared only when Native is not bound. CPU stays
        Eager so CI does not pay compile. Native or compile failure stays on
        same-device Eager.

    Raises
    ------
    DispatchError
        If ``device`` resolves to MPS or another non-cpu/cuda backend.
    TypeError
        If ``device`` is omitted (callers must pass it).

    """
    if device is None:
        message = "prepare_production_geometry requires device"
        logger.error(message)
        raise TypeError(message)
    resolved = _require_device(device)
    physical_uuid = None
    mig_uuid = None
    if resolved.type == "cuda":
        physical_uuid, mig_uuid = _cuda_uuids(resolved)
    solver = settings or SolverSettings()
    op = Operation(operation)
    normalized_shape = tuple(int(value) for value in shape)
    dem_arg = dem
    if op is Operation.RDR2GEO and dem_arg is None:
        dem_arg = ConstantHeightDEM(0.0)
    cache_key = _prepared_cache_key(
        op,
        model,
        shape=normalized_shape,
        resolved=resolved,
        dem=dem_arg,
        solver=solver,
        physical_uuid=physical_uuid,
        mig_uuid=mig_uuid,
    )
    cached = _PREPARED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    native_op = (
        NativeOperation.GEO2RDR if op is Operation.GEO2RDR else NativeOperation.RDR2GEO
    )
    loaded = None
    native_key = None
    native_context = None
    native_ok = False
    if resolved.type == "cuda":
        loaded = _try_load_cuda_module(native_op)
        if loaded is None:
            logger.warning(
                "CUDA %s native module not loaded; continuing on same-device Torch",
                native_op.value,
            )
        elif op is Operation.GEO2RDR:
            try:
                native_key, native_context = _geo2rdr_native_manifest(
                    model=model,
                    shape=normalized_shape,
                    solver=solver,
                    resolved=resolved,
                    physical_uuid=physical_uuid,
                    mig_uuid=mig_uuid,
                    module=loaded,
                )
                native_ok = True
                logger.info(
                    "CUDA geo2rdr native module loaded and registered for auto dispatch"
                )
            except Exception as error:
                logger.warning(
                    "CUDA geo2rdr native registration failed (%s); "
                    "continuing on same-device Torch",
                    error,
                )
                loaded = None
        else:
            try:
                native_key, native_context = _rdr2geo_native_manifest(
                    model=model,
                    shape=normalized_shape,
                    solver=solver,
                    resolved=resolved,
                    physical_uuid=physical_uuid,
                    mig_uuid=mig_uuid,
                    module=loaded,
                    dem=dem_arg,
                )
                if not _native_dem_context_bound(native_context):
                    logger.warning(
                        "CUDA rdr2geo Native DEM-bound identity refused without "
                        "DEM context; continuing on same-device Torch "
                        "(PROPOSAL-0025 / PROPOSAL-0026)"
                    )
                    loaded = None
                    native_key = None
                    native_context = None
                else:
                    native_ok = True
                    logger.info(
                        "CUDA rdr2geo native module loaded and registered "
                        "with DEM context"
                    )
            except Exception as error:
                logger.warning(
                    "CUDA rdr2geo native registration failed (%s); "
                    "continuing on same-device Torch",
                    error,
                )
                loaded = None
    use_compile = resolved.type == "cuda" and not native_ok
    try:
        prepared = _prepare_geometry_identities(
            op,
            model,
            shape=normalized_shape,
            resolved=resolved,
            dem=dem_arg,
            solver=solver,
            native_ok=native_ok,
            loaded=loaded,
            native_key=native_key,
            native_context=native_context,
            physical_uuid=physical_uuid,
            mig_uuid=mig_uuid,
            use_compile=use_compile,
        )
    except Exception as error:
        if not use_compile:
            raise
        logger.warning(
            "CUDA compile prepare failed (%s); continuing on same-device Eager",
            error,
        )
        prepared = _prepare_geometry_identities(
            op,
            model,
            shape=normalized_shape,
            resolved=resolved,
            dem=dem_arg,
            solver=solver,
            native_ok=native_ok,
            loaded=loaded,
            native_key=native_key,
            native_context=native_context,
            physical_uuid=physical_uuid,
            mig_uuid=mig_uuid,
            use_compile=False,
        )
    if len(_PREPARED_CACHE) >= _PREPARED_CACHE_LIMIT:
        oldest = next(iter(_PREPARED_CACHE))
        _PREPARED_CACHE.pop(oldest, None)
    _PREPARED_CACHE[cache_key] = prepared
    return prepared


def _geo2rdr_native_manifest(
    *,
    model: RadarGeometryModel,
    shape: tuple[int, ...],
    solver: SolverSettings,
    resolved: object,
    physical_uuid: str | None,
    mig_uuid: str | None,
    module: object,
) -> tuple[object, dict[str, object]]:
    """Build the explicit CandidateKey + orbit context for CUDA geo2rdr."""
    from faninsar.processing.geometry.backend_dispatch import CandidateKey
    from faninsar.processing.geometry.torch_backends_v2 import prepare_torch_geometry
    from faninsar.processing.geometry.v2 import DeviceKey, ExecutionProfile

    if not physical_uuid:
        message = "CUDA native geo2rdr requires a physical UUID"
        raise DispatchError(message)
    torch_prepared = prepare_torch_geometry(
        Operation.GEO2RDR,
        model,
        shape=shape,
        device=resolved,
        max_iter=solver.max_iter,
        extra_iter=solver.extra_iter,
        range_tol_m=solver.range_tolerance_m,
        doppler_tol_hz=solver.doppler_tolerance_hz,
        compile_kernel=False,
    )
    device_key = DeviceKey.cuda(physical_uuid, mig_uuid)
    artifact = str(getattr(module, "__file__", "") or "cuda-geo2rdr")
    digest = hashlib.sha256(artifact.encode()).hexdigest()
    key_solver = torch_prepared.settings.operation_settings(Operation.GEO2RDR).solver
    key = CandidateKey(
        operation=Operation.GEO2RDR,
        backend="native",
        device=device_key,
        dtype=torch_prepared.dtype,
        shape=shape,
        solver=key_solver,
        orbit_digest=torch_prepared.identity.orbit_digest,
        dem_digest=torch_prepared.identity.dem_digest,
        model_digest=torch_prepared.identity.model_digest,
        source_digest=digest,
        toolchain_digest=digest,
        runtime_digest=digest,
        artifact_digest=digest,
        abi_digest=hashlib.sha256(b"faninsar.geometry.native_v2.14-field.v1").hexdigest(),
        support_contract_digest=torch_prepared.identity.settings_digest,
        profile=ExecutionProfile(device_key),
    )
    import torch

    torch_device = torch.device(str(resolved))
    orbit = model.orbit
    times = np.asarray(orbit.times_s, dtype=np.float64)
    positions = np.stack(
        [spline(orbit.times_s) for spline in orbit.trajectory_splines],
        axis=-1,
    ).astype(np.float64, copy=False)
    velocities = np.stack(
        [spline(orbit.times_s, 1) for spline in orbit.trajectory_splines],
        axis=-1,
    ).astype(np.float64, copy=False)
    parameters = np.array(
        [
            (model.sensing_start - model.orbit.epoch).total_seconds(),
            model.azimuth_time_interval_s,
            model.starting_slant_range_m,
            model.range_spacing_m,
            model.wavelength_m,
        ],
        dtype=np.float64,
    )
    context = {
        "orbit_times": torch.as_tensor(times, dtype=torch.float64, device=torch_device),
        "orbit_positions": torch.as_tensor(
            positions, dtype=torch.float64, device=torch_device
        ),
        "orbit_velocities": torch.as_tensor(
            velocities, dtype=torch.float64, device=torch_device
        ),
        "model_parameters": torch.as_tensor(
            parameters, dtype=torch.float64, device=torch_device
        ),
        "look_right": model.look_direction == "right",
    }
    return key, context


def _dem_native_arrays(
    dem: DEMSampler | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize DEM raster, affine metadata, and height bounds.

    Metadata is canonical ``[lat_start, lon_start, lat_spacing, lon_spacing]``.
    CUDA dispatch reorders that tuple to the native ABI. Constant-height DEMs
    become a coarse global raster so the six-point stencil stays in-bounds.
    """
    from faninsar.processing.geometry.dem import GeoidAdjustedDEM, RasterDEM

    sampler = ConstantHeightDEM(0.0) if dem is None else dem
    if isinstance(sampler, GeoidAdjustedDEM):
        ortho = sampler.orthometric_dem
        geoid = sampler.geoid
        if not isinstance(ortho, RasterDEM) or not isinstance(geoid, RasterDEM):
            message = "native CUDA rdr2geo needs RasterDEM members on GeoidAdjustedDEM"
            raise DispatchError(message)
        values, metadata, bounds = _raster_dem_native_arrays(ortho)
        geoid_values, geoid_metadata, _geoid_bounds = _raster_dem_native_arrays(geoid)
        if not np.allclose(metadata, geoid_metadata):
            message = "native CUDA rdr2geo GeoidAdjustedDEM grids must share affine"
            raise DispatchError(message)
        if geoid_values.shape != values.shape:
            message = "native CUDA rdr2geo GeoidAdjustedDEM grids must share shape"
            raise DispatchError(message)
        values = values + geoid_values
        finite = np.isfinite(values)
        if not np.any(finite):
            raise DispatchError("native CUDA rdr2geo DEM has no finite samples")
        bounds = np.array(
            [float(np.min(values[finite])), float(np.max(values[finite]))],
            dtype=np.float64,
        )
        return values, metadata, bounds
    if isinstance(sampler, RasterDEM):
        return _raster_dem_native_arrays(sampler)
    if isinstance(sampler, ConstantHeightDEM):
        rows, cols = 80, 160
        height = float(sampler.height_m)
        values = np.full((rows, cols), height, dtype=np.float64)
        metadata = np.array(
            [90.0, -180.0, -180.0 / float(rows - 1), 360.0 / float(cols - 1)],
            dtype=np.float64,
        )
        bounds = np.array([height, height], dtype=np.float64)
        return values, metadata, bounds
    message = f"unsupported DEM type for native CUDA rdr2geo: {type(sampler).__name__}"
    raise DispatchError(message)


def _raster_dem_native_arrays(
    dem: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upload one RasterDEM into the native context arrays."""
    dataset = dem._open()
    samples = dem._height_array
    if samples is None:
        samples = dataset.read(1).astype(np.float32, copy=False)
        nodata = dem.nodata
        if nodata is None:
            nodata = dataset.nodata
        if nodata is not None:
            samples = np.where(np.isclose(samples, nodata), np.nan, samples)
        dem._height_array = samples
    transform = dataset.transform
    if abs(float(transform.b)) > 1.0e-12 or abs(float(transform.d)) > 1.0e-12:
        raise DispatchError(
            "native CUDA rdr2geo requires an unrotated geographic DEM affine"
        )
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 6:
        raise DispatchError("native CUDA rdr2geo DEM must be at least 6x6")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise DispatchError("native CUDA rdr2geo DEM has no finite samples")
    metadata = np.array(
        [
            float(transform.f),
            float(transform.c),
            float(transform.e),
            float(transform.a),
        ],
        dtype=np.float64,
    )
    bounds = np.array(
        [float(np.min(values[finite])), float(np.max(values[finite]))],
        dtype=np.float64,
    )
    return values, metadata, bounds


def _rdr2geo_native_manifest(
    *,
    model: RadarGeometryModel,
    shape: tuple[int, ...],
    solver: SolverSettings,
    resolved: object,
    physical_uuid: str | None,
    mig_uuid: str | None,
    module: object,
    dem: DEMSampler | None,
) -> tuple[object, dict[str, object]]:
    """Build the CandidateKey + orbit/DEM context for CUDA rdr2geo."""
    from faninsar.processing.geometry.backend_dispatch import CandidateKey
    from faninsar.processing.geometry.torch_backends_v2 import prepare_torch_geometry
    from faninsar.processing.geometry.v2 import DeviceKey, ExecutionProfile

    if not physical_uuid:
        message = "CUDA native rdr2geo requires a physical UUID"
        raise DispatchError(message)
    dem_arg = ConstantHeightDEM(0.0) if dem is None else dem
    torch_prepared = prepare_torch_geometry(
        Operation.RDR2GEO,
        model,
        shape=shape,
        dem=dem_arg,
        device=resolved,
        max_iter=solver.max_iter,
        extra_iter=solver.extra_iter,
        range_tol_m=solver.slant_range_tolerance_m,
        doppler_tol_hz=solver.doppler_tolerance_hz,
        compile_kernel=False,
    )
    device_key = DeviceKey.cuda(physical_uuid, mig_uuid)
    artifact = str(getattr(module, "__file__", "") or "cuda-rdr2geo")
    digest = hashlib.sha256(artifact.encode()).hexdigest()
    key_solver = torch_prepared.settings.operation_settings(Operation.RDR2GEO).solver
    key = CandidateKey(
        operation=Operation.RDR2GEO,
        backend="native",
        device=device_key,
        dtype=torch_prepared.dtype,
        shape=shape,
        solver=key_solver,
        orbit_digest=torch_prepared.identity.orbit_digest,
        dem_digest=torch_prepared.identity.dem_digest,
        model_digest=torch_prepared.identity.model_digest,
        source_digest=digest,
        toolchain_digest=digest,
        runtime_digest=digest,
        artifact_digest=digest,
        abi_digest=hashlib.sha256(b"faninsar.geometry.native_v2.14-field.v1").hexdigest(),
        support_contract_digest=torch_prepared.identity.settings_digest,
        profile=ExecutionProfile(device_key),
    )
    import torch

    torch_device = torch.device(str(resolved))
    orbit = model.orbit
    times = np.asarray(orbit.times_s, dtype=np.float64)
    positions = np.stack(
        [spline(orbit.times_s) for spline in orbit.trajectory_splines],
        axis=-1,
    ).astype(np.float64, copy=False)
    velocities = np.stack(
        [spline(orbit.times_s, 1) for spline in orbit.trajectory_splines],
        axis=-1,
    ).astype(np.float64, copy=False)
    parameters = np.array(
        [
            (model.sensing_start - model.orbit.epoch).total_seconds(),
            model.azimuth_time_interval_s,
            model.starting_slant_range_m,
            model.range_spacing_m,
            model.wavelength_m,
        ],
        dtype=np.float64,
    )
    resident_dem = _resident_gpu_dem(
        torch_prepared.identity.dem_digest,
        _device_cache_identity(resolved, physical_uuid, mig_uuid),
        torch_device,
        dem_arg,
    )
    context = {
        "orbit_times": torch.as_tensor(times, dtype=torch.float64, device=torch_device),
        "orbit_positions": torch.as_tensor(
            positions, dtype=torch.float64, device=torch_device
        ),
        "orbit_velocities": torch.as_tensor(
            velocities, dtype=torch.float64, device=torch_device
        ),
        "model_parameters": torch.as_tensor(
            parameters, dtype=torch.float64, device=torch_device
        ),
        "look_right": model.look_direction == "right",
        "dem_values": resident_dem["dem_values"],
        "dem_metadata": resident_dem["dem_metadata"],
        "dem_height_bounds": resident_dem["dem_height_bounds"],
    }
    return key, context


def _reshape_transform_result(
    result: TransformResult,
    shape: tuple[int, ...],
) -> TransformResult:
    """Restore the public ND layout after a 1-D native or Torch solve."""
    if result.azimuth_index.shape == shape:
        return result
    return TransformResult(
        latitude_deg=np.asarray(result.latitude_deg).reshape(shape),
        longitude_deg=np.asarray(result.longitude_deg).reshape(shape),
        height_m=np.asarray(result.height_m).reshape(shape),
        range_index=np.asarray(result.range_index).reshape(shape),
        azimuth_index=np.asarray(result.azimuth_index).reshape(shape),
        converged=np.asarray(result.converged).reshape(shape),
        residual_range_m=np.asarray(result.residual_range_m).reshape(shape),
        residual_doppler_hz=np.asarray(result.residual_doppler_hz).reshape(shape),
    )


def run_geo2rdr(
    model: RadarGeometryModel,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray,
    *,
    device: DeviceLike,
    max_iter: int = 20,
    range_tol_m: float = 0.01,
    doppler_tol_hz: float = 0.1,
) -> TransformResult:
    """Execute production geo2rdr through the mandatory helper.

    CUDA native geo2rdr accepts 1-D or 2-D tiles (ISCE3 gpuGeo2rdr style:
    flatten to one thread per pixel, restore the raster layout on output).
    """
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    lat, lon, height = np.broadcast_arrays(lat, lon, height)
    original_shape = lat.shape
    # The native ABI is lane-oriented and accepts only contiguous 1-D
    # coordinate arrays.  Keep the public helper N-D friendly by flattening
    # before dispatch and restoring the caller's raster shape on return.
    flat_shape = (lat.size,)
    prepared = prepare_production_geometry(
        Operation.GEO2RDR,
        model,
        device=device,
        shape=flat_shape,
        settings=SolverSettings(
            max_iter=max_iter,
            range_tolerance_m=range_tol_m,
            doppler_tolerance_hz=doppler_tol_hz,
        ),
    )
    result = to_transform_result(
        execute_geometry(
            prepared,
            lat.reshape(-1),
            lon.reshape(-1),
            height.reshape(-1),
        )
    )
    return _reshape_transform_result(result, original_shape)


def run_rdr2geo(
    model: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: DEMSampler | None = None,
    *,
    device: DeviceLike,
    max_iter: int = 20,
    range_tol_m: float = 0.01,
    doppler_tol_hz: float = 0.1,
) -> TransformResult:
    """Execute production rdr2geo through the mandatory helper."""
    az = np.asarray(azimuth_index, dtype=np.float64)
    rg = np.asarray(range_index, dtype=np.float64)
    az, rg = np.broadcast_arrays(az, rg)
    prepared = prepare_production_geometry(
        Operation.RDR2GEO,
        model,
        device=device,
        shape=az.shape,
        dem=dem,
        settings=SolverSettings(
            max_iter=max_iter,
            slant_range_tolerance_m=range_tol_m,
            doppler_tolerance_hz=doppler_tol_hz,
        ),
    )
    return to_transform_result(execute_geometry(prepared, az, rg))


def run_rdr2geo_chunked(
    model: RadarGeometryModel,
    azimuth_index: np.ndarray,
    range_index: np.ndarray,
    dem: DEMSampler | None,
    *,
    device: DeviceLike,
    chunk_size: tuple[int, int] = (256, 256),
    max_iter: int = 20,
    range_tol_m: float = 0.01,
    doppler_tol_hz: float = 0.1,
) -> TransformResult:
    """Tile rdr2geo so peak memory stays at current geo_chunk_size scale.

    Full tiles reuse one :class:`PreparedGeometry` built for ``chunk_size``.
    A ragged remainder is prepared at its own shape.
    """
    az = np.asarray(azimuth_index, dtype=np.float64)
    rg = np.asarray(range_index, dtype=np.float64)
    az_b, rg_b = np.broadcast_arrays(az, rg)
    shape = az_b.shape
    lat = np.full(shape, np.nan, dtype=np.float64)
    lon = np.full(shape, np.nan, dtype=np.float64)
    height = np.full(shape, np.nan, dtype=np.float64)
    converged = np.zeros(shape, dtype=bool)
    residual_range = np.full(shape, np.nan, dtype=np.float64)
    residual_doppler = np.full(shape, np.nan, dtype=np.float64)
    az_chunk, rg_chunk = chunk_size
    prepared_full = None
    full_tile = (min(az_chunk, shape[0]), min(rg_chunk, shape[1]))
    for row in range(0, shape[0], az_chunk):
        for col in range(0, shape[1], rg_chunk):
            slc = (
                slice(row, min(row + az_chunk, shape[0])),
                slice(col, min(col + rg_chunk, shape[1])),
            )
            tile_az = az_b[slc]
            tile_shape = tile_az.shape
            if tile_shape == full_tile:
                if prepared_full is None:
                    prepared_full = prepare_production_geometry(
                        Operation.RDR2GEO,
                        model,
                        device=device,
                        shape=full_tile,
                        dem=dem,
                        settings=SolverSettings(
                            max_iter=max_iter,
                            slant_range_tolerance_m=range_tol_m,
                            doppler_tolerance_hz=doppler_tol_hz,
                        ),
                    )
                prepared = prepared_full
            else:
                prepared = prepare_production_geometry(
                    Operation.RDR2GEO,
                    model,
                    device=device,
                    shape=tile_shape,
                    dem=dem,
                    settings=SolverSettings(
                        max_iter=max_iter,
                        slant_range_tolerance_m=range_tol_m,
                        doppler_tolerance_hz=doppler_tol_hz,
                    ),
                )
            chunk_result = to_transform_result(
                execute_geometry(prepared, tile_az, rg_b[slc])
            )
            lat[slc] = chunk_result.latitude_deg
            lon[slc] = chunk_result.longitude_deg
            height[slc] = chunk_result.height_m
            converged[slc] = chunk_result.converged
            residual_range[slc] = chunk_result.residual_range_m
            residual_doppler[slc] = chunk_result.residual_doppler_hz
    return TransformResult(
        latitude_deg=lat,
        longitude_deg=lon,
        height_m=height,
        range_index=rg_b.copy(),
        azimuth_index=az_b.copy(),
        converged=converged,
        residual_range_m=residual_range,
        residual_doppler_hz=residual_doppler,
    )


def interpolate_control_field(
    az_ctrl: np.ndarray,
    rg_ctrl: np.ndarray,
    values_ctrl: np.ndarray,
    shape: tuple[int, int],
    *,
    device: DeviceLike,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Bilinear interpolate a control grid onto ``shape`` on the given device."""
    import torch

    resolved = _require_device(device)
    torch_device = torch.device(str(resolved))
    az_t = torch.as_tensor(az_ctrl, dtype=torch.float64, device=torch_device)
    rg_t = torch.as_tensor(rg_ctrl, dtype=torch.float64, device=torch_device)
    val_t = torch.as_tensor(values_ctrl, dtype=torch.float64, device=torch_device)
    height, width = shape
    az_q = torch.arange(height, dtype=torch.float64, device=torch_device)
    rg_q = torch.arange(width, dtype=torch.float64, device=torch_device)
    az_idx = torch.searchsorted(az_t, az_q, right=True).clamp(1, az_t.numel() - 1)
    rg_idx = torch.searchsorted(rg_t, rg_q, right=True).clamp(1, rg_t.numel() - 1)
    az0 = az_idx - 1
    rg0 = rg_idx - 1
    az_span = (az_t[az_idx] - az_t[az0]).clamp(min=1e-12)
    rg_span = (rg_t[rg_idx] - rg_t[rg0]).clamp(min=1e-12)
    waz = ((az_q - az_t[az0]) / az_span).clamp(0.0, 1.0)[:, None]
    wrg = ((rg_q - rg_t[rg0]) / rg_span).clamp(0.0, 1.0)[None, :]
    v00 = val_t[az0[:, None], rg0[None, :]]
    v01 = val_t[az0[:, None], rg_idx[None, :]]
    v10 = val_t[az_idx[:, None], rg0[None, :]]
    v11 = val_t[az_idx[:, None], rg_idx[None, :]]
    out = (
        v00 * (1.0 - waz) * (1.0 - wrg)
        + v01 * (1.0 - waz) * wrg
        + v10 * waz * (1.0 - wrg)
        + v11 * waz * wrg
    )
    finite = torch.isfinite(out)
    if not bool(torch.all(finite)):
        out = torch.where(finite, out, torch.as_tensor(fill_value, device=torch_device))
    return out.detach().cpu().numpy()

"""Ampcor-style production helper for P20/P25/P26 geometry (PROPOSAL-0031).

Callers pass ``device`` only. UUID, native executor, and candidate keys stay
inside this module. Failed native prepare stays on same-device Torch.
"""

from __future__ import annotations

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
from faninsar.processing.geometry.v2 import Operation, SolverSettings, TransformResultV2

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.geometry.public import PreparedGeometry
    from faninsar.processing.geometry.transforms import RadarGeometryModel
    from faninsar.typing import DeviceLike

logger = setup_logger(__name__)

_NATIVE_SOURCE_ROOT = Path(__file__).resolve().parent / "native_v2"


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


def _cuda_uuids(resolved: object) -> tuple[str, str | None]:
    """Read the CUDA physical UUID (and MIG UUID when Torch exposes it)."""
    import torch

    props = torch.cuda.get_device_properties(resolved)
    physical = str(props.uuid)
    mig = getattr(props, "mig_uuid", None)
    if mig is None:
        mig = getattr(props, "uuid_mig", None)
    return physical, str(mig) if mig is not None else None


def _native_build_dir() -> Path:
    """Process-scoped P25-style cache root for geometry extensions."""
    from faninsar.compute.cache import resolve_compile_cache_dir

    path = resolve_compile_cache_dir().parent / "native_v2_geometry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _try_load_cuda_module(operation: NativeOperation) -> object | None:
    """Prepare a CUDA geo module; return None on any failure."""
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
        module = cpp_extension.load(
            name=plan_value.extension_name,
            sources=[str(source) for source in plan_value.sources],
            extra_cflags=list(plan_value.compile_flags),
            extra_cuda_cflags=list(plan_value.compile_flags),
            extra_ldflags=list(plan_value.link_flags),
            extra_include_paths=[str(path) for path in plan_value.include_dirs],
            build_directory=str(_native_build_dir()),
            with_cuda=True,
            verbose=False,
        )
        module_holder["module"] = module
        return Path(getattr(module, "__file__", _native_build_dir()))

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
    return module_holder.get("module")


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
        Ready for :func:`execute_geometry`.

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
    native_op = (
        NativeOperation.GEO2RDR if op is Operation.GEO2RDR else NativeOperation.RDR2GEO
    )
    if resolved.type == "cuda":
        loaded = _try_load_cuda_module(native_op)
        if loaded is None:
            logger.warning(
                "CUDA %s native module not loaded; continuing on same-device Torch",
                native_op.value,
            )
        else:
            logger.info(
                "CUDA %s native module loaded; auto stays on Torch until a "
                "same-process native-vs-Torch compare supplies CandidateKey "
                "and qualification flags",
                native_op.value,
            )
    dem_arg = dem
    if op is Operation.RDR2GEO and dem_arg is None:
        dem_arg = ConstantHeightDEM(0.0)
    return prepare_geometry(
        op,
        model,
        shape=tuple(int(value) for value in shape),
        device=resolved,
        dem=dem_arg,
        settings=solver,
        native_correctness_qualified=False,
        native_performance_eligible=False,
        physical_uuid=physical_uuid,
        mig_uuid=mig_uuid,
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
    """Execute production geo2rdr through the mandatory helper."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    lat, lon, height = np.broadcast_arrays(lat, lon, height)
    prepared = prepare_production_geometry(
        Operation.GEO2RDR,
        model,
        device=device,
        shape=lat.shape,
        settings=SolverSettings(
            max_iter=max_iter,
            range_tolerance_m=range_tol_m,
            doppler_tolerance_hz=doppler_tol_hz,
        ),
    )
    return to_transform_result(execute_geometry(prepared, lat, lon, height))


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

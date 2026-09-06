"""Dask scheduling layer for Torch GPU kernels.

Dask and Torch stay fully decoupled: NumPy arrays are the only contract
crossing the scheduling boundary.  The block kernels follow the
dask-torch-numpy pattern (numpy in, numpy out, torch internal) and every
GPU-scheduled task is annotated with ``resources={"gpu": 1}`` so the
scheduler only routes it to workers that advertise a GPU.

Distributed execution is explicit: callers must inject a trusted Dask client.
An ambient active client is never discovered automatically. Without an injected
client, the same functions run eager Torch on the local resolved device.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.processing.coregistration.tops.deramp import TOPSCarrierModel

logger = setup_logger(__name__)

__all__ = [
    "CUDA_PERFORMANCE_QUALIFIED",
    "has_gpu_workers",
    "run_carrier_multiply",
    "run_carrier_multiply_at_points",
    "run_esd_azimuth_shift",
    "run_goldstein_filter",
    "run_multilook_interferogram",
    "run_multilook_real",
    "run_reclaim_checkpoint",
    "run_remove_topographic_phase",
    "should_accelerate",
    "should_use_cuda",
    "use_gpu_resources",
    "validate_cuda_worker_binding",
]

# CryoGPU HK qualification: these stages met the 1.5x CUDA speed gate.
# After CUDA admission the set may only log or choose among same-device
# identities. It must not hop deramp/multilook/flatten back to host NumPy.
CUDA_PERFORMANCE_QUALIFIED: frozenset[str] = frozenset(
    {"goldstein_filter", "esd_azimuth_shift"}
)
_PUBLISHED_BACKENDS: frozenset[str] = frozenset({"auto", "cpu", "cuda", "gpu"})


def has_gpu_workers(client: Any | None) -> bool:
    """Return True when an explicitly injected client advertises GPU workers."""
    if client is None:
        return False
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        logger.exception(
            "Unable to inspect Dask worker resources; treating the cluster "
            "as GPU-unavailable"
        )
        return False
    if not isinstance(workers, dict):
        logger.error("Dask scheduler returned malformed worker metadata")
        return False
    for worker in workers.values():
        if not isinstance(worker, dict):
            continue
        resources = worker.get("resources", {})
        if not isinstance(resources, dict):
            continue
        value = resources.get("gpu", 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            logger.warning("Ignoring malformed Dask GPU resource metadata")
            continue
        if math.isfinite(value) and value >= 1:
            return True
    return False


def validate_cuda_worker_binding(client: Any | None) -> bool:
    """Verify that each advertised GPU worker exposes one CUDA ordinal.

    Dask's logical ``gpu`` resource only controls scheduling. This preflight
    checks each worker process environment so concurrent tasks do not all
    select CUDA device zero on a multi-GPU host.

    Parameters
    ----------
    client : object or None
        Explicitly injected Dask client.

    Returns
    -------
    bool
        ``True`` when every advertised GPU worker exposes exactly one visible
        CUDA device through ``CUDA_VISIBLE_DEVICES``.

    """
    if client is None or not hasattr(client, "run"):
        logger.error("GPU worker binding preflight requires an explicit Dask client")
        return False
    try:
        workers = client.scheduler_info().get("workers", {})
        gpu_workers = []
        for name, worker in workers.items():
            if not isinstance(worker, dict):
                continue
            resources = worker.get("resources")
            if not isinstance(resources, dict):
                continue
            value = resources.get("gpu", 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value != 1
            ):
                logger.warning(
                    "Ignoring malformed Dask GPU resource metadata during "
                    "CUDA binding validation"
                )
                continue
            gpu_workers.append(name)
        if not gpu_workers:
            return False
        visible = client.run(_cuda_worker_identity)
    except Exception:
        logger.exception("Unable to verify Dask CUDA worker binding")
        return False
    physical_ids: set[str] = set()
    for worker_name in gpu_workers:
        record = visible.get(worker_name)
        if isinstance(record, dict):
            value = record.get("visible")
            identities = record.get("identities", [])
            physical_id = identities[0] if len(identities) == 1 else None
        else:
            value = record
            physical_id = value
        visible_ordinals = (
            []
            if value is None
            else [token for token in value.split(",") if token.strip()]
        )
        if len(visible_ordinals) != 1 or physical_id is None:
            logger.error(
                "GPU worker %s does not expose exactly one "
                "CUDA_VISIBLE_DEVICES ordinal",
                worker_name,
            )
            return False
        if physical_id in physical_ids:
            logger.error("GPU workers share physical CUDA device %s", physical_id)
            return False
        physical_ids.add(physical_id)
    return True


def _reclaim_checkpoint_on_worker(
    device: str,
    policy: str,
    kind: str,
) -> bool:
    """Evaluate reclaim policy on the process that owns the CUDA pool."""
    from faninsar.processing.runtime.device import reclaim_checkpoint

    return reclaim_checkpoint(device, policy, kind=kind)  # type: ignore[arg-type]


def run_reclaim_checkpoint(
    client: Any,
    device: str,
    policy: str,
    *,
    kind: str,
) -> object:
    """Invoke :func:`reclaim_checkpoint` on Dask workers, never on tiles.

    Parameters
    ----------
    client : object
        Injected Dask client that owns GPU workers.
    device : str
        Device request resolved on each worker.
    policy : str
        ``gpu_memory_reclaim`` policy.
    kind : str
        Checkpoint kind (``persist``, ``stage``, ``oom``, ``explicit``).

    Returns
    -------
    object
        ``client.run`` result mapping workers to whether they reclaimed.

    """
    return client.run(_reclaim_checkpoint_on_worker, device, policy, kind)


def _device_backend(device: str) -> str:
    """Return the backend token without stripping a ``cuda:N`` ordinal."""
    return device.strip().lower().split(":", 1)[0]


def _reject_unpublished_device(device: str) -> None:
    """Fail closed for backends that are not published production devices."""
    backend = _device_backend(device)
    if backend in _PUBLISHED_BACKENDS:
        return
    message = (
        f"device {device!r} is not a published production backend; "
        "refusing host NumPy fallback"
    )
    logger.error(message)
    raise RuntimeError(message)


def _cuda_is_admitted(device: str, client: Any | None) -> bool:
    """Return whether CUDA is the requested or auto-resolved published device."""
    backend = _device_backend(device)
    if backend == "cpu":
        return False
    if has_gpu_workers(client) or backend == "cuda":
        return True
    if backend in {"auto", "gpu"}:
        from faninsar.processing.runtime.device import parse_device

        return parse_device(device).type == "cuda"
    return False


def _allow_remote_kernel(
    device: str,
    client: Any | None,
    kernel: str,
) -> bool:
    """Return whether a public runner may submit this kernel remotely."""
    return (
        client is not None
        and use_gpu_resources(device, client)
        and kernel in CUDA_PERFORMANCE_QUALIFIED
    )


def _local_device_after_remote_rejection(
    device: str,
    client: Any | None,
    kernel: str,
) -> str:
    """Keep local execution on the admitted device; do not hop to host NumPy."""
    if (
        client is not None
        and has_gpu_workers(client)
        and kernel not in CUDA_PERFORMANCE_QUALIFIED
    ):
        logger.info(
            "%s is not CUDA-performance-qualified; running the Torch kernel "
            "on the admitted device",
            kernel,
        )
    return device


def _cuda_worker_identity() -> dict[str, object]:
    """Return worker-local CUDA visibility and physical device identities."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        import torch

        identities = [
            str(torch.cuda.get_device_properties(index).uuid)
            for index in range(torch.cuda.device_count())
        ]
    except Exception:
        identities = []
    return {"visible": visible, "identities": identities}


def use_gpu_resources(device: str, client: Any | None = None) -> bool:
    """Return True when GPU resource annotation is appropriate for *device*."""
    if device.lower() in {"cpu", "mps"}:
        return False
    return has_gpu_workers(client)


def should_accelerate(
    device: str,
    client: Any | None = None,
    kernel: str | None = None,
) -> bool:
    """Return whether a stage should use its unified Torch kernel.

    Published CPU and CUDA devices (including ``auto`` that resolves to
    either) keep the NumPy-in/NumPy-out Torch path. ``CUDA_PERFORMANCE_QUALIFIED``
    may only log or choose among same-device identities after CUDA admission.
    Explicit unpublished backends such as MPS, TPU, or NPU fail closed.
    """
    _reject_unpublished_device(device)
    if (
        kernel is not None
        and kernel not in CUDA_PERFORMANCE_QUALIFIED
        and _cuda_is_admitted(device, client)
    ):
        logger.info(
            "%s is not CUDA-performance-qualified; keeping the admitted "
            "CUDA Torch path",
            kernel,
        )
    return True


def should_use_cuda(device: str, client: Any | None = None) -> bool:
    """Return whether a CUDA-capable local or distributed path is available."""
    _reject_unpublished_device(device)
    return _cuda_is_admitted(device, client)


def _annotate(use_gpu: bool) -> Any:
    """Return a GPU-resource annotation context when scheduling GPUs."""
    import dask

    if use_gpu:
        return dask.annotate(resources={"gpu": 1})
    return nullcontext()


def _scheduled_device(device: str, client: Any | None) -> str:
    """Return the fail-closed device forwarded to scheduled worker tasks."""
    if use_gpu_resources(device, client) and not validate_cuda_worker_binding(client):
        message = "GPU worker-to-CUDA binding preflight failed"
        logger.error(message)
        raise RuntimeError(message)
    if use_gpu_resources(device, client):
        lowered = device.strip().lower()
        if lowered.startswith("cuda:"):
            return lowered
        return "cuda"
    return device


def _validate_chunk_rows(chunk_rows: int) -> None:
    """Reject invalid chunk sizes before constructing a Dask graph."""
    if chunk_rows < 1:
        message = "chunk_rows must be >= 1"
        logger.error(message)
        raise ValueError(message)


def _to_row_chunked(
    array: np.ndarray,
    chunk_rows: int,
) -> Any:
    """Wrap a NumPy array as a Dask array chunked along axis 0."""
    import dask.array as da

    _validate_chunk_rows(chunk_rows)
    return da.from_array(np.asarray(array), chunks=(chunk_rows, -1))


def _map_torch_blocks(
    func: Any,
    *arrays: np.ndarray,
    chunk_rows: int,
    dtype: np.dtype | type,
    device: str,
    name: str,
) -> Any:
    """Map a Torch kernel over row chunks with optional GPU annotation."""
    import dask.array as da

    dask_arrays = [_to_row_chunked(array, chunk_rows) for array in arrays]
    use_gpu = use_gpu_resources(device)
    with _annotate(use_gpu):
        return da.map_blocks(func, *dask_arrays, dtype=dtype, name=name)


def _schedule_carrier_multiply(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    *,
    sign: float,
    row0: int = 0,
    col0: int = 0,
    native_height: int | None = None,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> Any:
    """Schedule a TOPS carrier multiply as a lazy Dask graph.

    Each block evaluates the analytical carrier at its global row/column
    location (via ``block_info``) so tiled kernels match the full-array
    reference exactly.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex burst window.
    model : TOPSCarrierModel
        Carrier model matching the window geometry.
    sign : float
        ``-1`` for deramp, ``+1`` for reramp.
    row0, col0 : int, optional
        Offset of the window inside the native burst.
    native_height : int, optional
        Native burst height for the carrier centre row.
    device : str, optional
        Torch device string forwarded to the block kernels.
    chunk_rows : int, optional
        Azimuth rows per scheduled task.
    client : object or None, optional
        Connected Dask client used to inspect worker GPU resources.

    Returns
    -------
    dask.array.Array
        Lazy complex array of the same shape as ``samples``.

    """
    import dask.array as da

    from faninsar.processing.runtime.torch_kernels import (
        carrier_multiply_torch,
        carrier_phase_at_points_torch,
    )

    _validate_chunk_rows(chunk_rows)
    worker_device = _scheduled_device(device, client)
    n_lines, _ = samples.shape
    centre_row = float(n_lines // 2 if native_height is None else native_height // 2)

    def block(
        s_block: np.ndarray,
        block_info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        assert block_info is not None
        row_slice, col_slice = block_info[None]["array-location"]
        r0, r1 = row_slice
        c0, c1 = col_slice
        rows = np.arange(r0 + row0, r1 + row0, dtype=np.float64)[:, None]
        cols = np.arange(c0 + col0, c1 + col0, dtype=np.float64)[None, :]
        phase = carrier_phase_at_points_torch(
            model,
            rows,
            cols,
            centre_row=centre_row,
            dtype=np.float64,
            device=worker_device,
        )
        return carrier_multiply_torch(s_block, phase, sign=sign, device=worker_device)

    dask_array = _to_row_chunked(samples, chunk_rows)
    with _annotate(use_gpu_resources(device, client)):
        return da.map_blocks(
            block,
            dask_array,
            dtype=samples.dtype,
            name="carrier-multiply",
        )


def _schedule_remove_topographic_phase(
    complex_ifg: np.ndarray,
    topo_phase: np.ndarray,
    *,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> Any:
    """Schedule topographic flattening as a lazy Dask graph."""
    import dask.array as da

    from faninsar.processing.runtime.torch_kernels import remove_topographic_phase_torch

    _validate_chunk_rows(chunk_rows)
    worker_device = _scheduled_device(device, client)

    def block(
        ifg_block: np.ndarray,
        topo_block: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            remove_topographic_phase_torch(ifg_block, topo_block, device=worker_device),
            dtype=complex_ifg.dtype,
        )

    dask_arrays = [
        _to_row_chunked(array, chunk_rows) for array in (complex_ifg, topo_phase)
    ]
    with _annotate(use_gpu_resources(device, client)):
        return da.map_blocks(
            block,
            *dask_arrays,
            dtype=complex_ifg.dtype,
            name="remove-topographic-phase",
        )


def _schedule_multilook_interferogram(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    multilook: tuple[int, int],
    dead_pixel_amp_threshold: float = 0.0,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Schedule multilooked interferogram formation as lazy Dask graphs.

    Returns
    -------
    tuple of dask.array.Array
        Lazy ``(complex_ifg, coherence, wrapped_phase, amplitude)`` graphs.

    """
    import dask
    import dask.array as da

    from faninsar.processing.runtime.torch_kernels import multilook_interferogram_torch

    _validate_chunk_rows(chunk_rows)
    worker_device = _scheduled_device(device, client)
    az_looks, rg_looks = multilook
    height, width = primary.shape
    h = (height // az_looks) * az_looks
    w = (width // rg_looks) * rg_looks
    if h == 0 or w == 0:
        complex_dtype = np.result_type(primary.dtype, secondary.dtype)
        float_dtype = (
            np.float64 if complex_dtype == np.dtype(np.complex128) else np.float32
        )
        shape = (h // az_looks, w // rg_looks)
        return tuple(
            da.from_array(np.empty(shape, dtype=dtype), chunks=-1)
            for dtype in (
                complex_dtype,
                float_dtype,
                float_dtype,
                float_dtype,
            )
        )
    chunk_az = max(az_looks, (chunk_rows // az_looks) * az_looks)
    out_w = w // rg_looks
    use_gpu = use_gpu_resources(device, client)

    def block(
        primary_chunk: np.ndarray,
        secondary_chunk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        product = multilook_interferogram_torch(
            primary_chunk,
            secondary_chunk,
            multilook=multilook,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            device=worker_device,
        )
        return (
            product.complex_ifg,
            product.coherence,
            product.wrapped_phase,
            product.amplitude,
        )

    fields: list[list[Any]] = [[], [], [], []]
    with _annotate(use_gpu):
        for row_start in range(0, h, chunk_az):
            row_stop = min(row_start + chunk_az, h)
            out_h = (row_stop - row_start) // az_looks
            task = dask.delayed(block)(
                primary[row_start:row_stop, :w],
                secondary[row_start:row_stop, :w],
            )
            shapes = (
                (out_h, out_w),
                (out_h, out_w),
                (out_h, out_w),
                (out_h, out_w),
            )
            complex_dtype = np.result_type(primary.dtype, secondary.dtype)
            float_dtype = (
                np.float64 if complex_dtype == np.dtype(np.complex128) else np.float32
            )
            dtypes = (
                complex_dtype,
                float_dtype,
                float_dtype,
                float_dtype,
            )
            for field_index, (shape, dtype) in enumerate(
                zip(shapes, dtypes, strict=True)
            ):
                fields[field_index].append(
                    da.from_delayed(task[field_index], shape=shape, dtype=dtype)
                )
    return tuple(da.concatenate(field, axis=0) for field in fields)


def _schedule_multilook_real(
    array: np.ndarray,
    az_looks: int,
    rg_looks: int,
    *,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> Any:
    """Schedule real-field multilook as a lazy Dask graph."""
    import dask
    import dask.array as da

    from faninsar.processing.runtime.torch_kernels import multilook_real_torch

    _validate_chunk_rows(chunk_rows)
    worker_device = _scheduled_device(device, client)
    height, width = array.shape[:2]
    h = (height // az_looks) * az_looks
    w = (width // rg_looks) * rg_looks
    if h == 0 or w == 0:
        return da.from_array(
            np.empty((h // az_looks, w // rg_looks), dtype=array.dtype),
            chunks=-1,
        )
    chunk_az = max(az_looks, (chunk_rows // az_looks) * az_looks)
    out_w = w // rg_looks
    use_gpu = use_gpu_resources(device, client)
    parts: list[Any] = []
    with _annotate(use_gpu):
        for row_start in range(0, h, chunk_az):
            row_stop = min(row_start + chunk_az, h)
            out_h = (row_stop - row_start) // az_looks
            task = dask.delayed(multilook_real_torch)(
                array[row_start:row_stop, :w],
                az_looks,
                rg_looks,
                device=worker_device,
            )
            parts.append(da.from_delayed(task, shape=(out_h, out_w), dtype=array.dtype))
    return da.concatenate(parts, axis=0)


def _schedule_goldstein_filter(
    complex_ifg: np.ndarray,
    *,
    alpha: float,
    window: int,
    device: str = "auto",
    chunk_rows: int = 128,
    client: Any | None = None,
) -> Any:
    """Schedule Goldstein filtering as a lazy Dask graph.

    Each scheduled task receives a ``window - 1`` row halo and computes only
    its interior output rows, so chunked filtering reproduces the full-array
    reference exactly.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram.
    alpha : float
        Filter exponent in ``[0, 1]``.
    window : int
        Square FFT patch size.
    device : str, optional
        Torch device string forwarded to the block kernels.
    chunk_rows : int, optional
        Output rows per scheduled task.
    client : object or None, optional
        Connected Dask client used to inspect worker GPU resources.

    Returns
    -------
    dask.array.Array
        Lazy complex array of the same shape as ``complex_ifg``.

    """
    import dask
    import dask.array as da

    from faninsar.processing.runtime.torch_kernels import goldstein_filter_torch

    _validate_chunk_rows(chunk_rows)
    worker_device = _scheduled_device(device, client)
    height, width = complex_ifg.shape
    if height < window or width < window:
        return da.from_array(np.asarray(complex_ifg).copy(), chunks=-1)
    halo = window - 1
    use_gpu = use_gpu_resources(device, client)
    parts: list[Any] = []
    with _annotate(use_gpu):
        for row_start in range(0, height, chunk_rows):
            row_stop = min(row_start + chunk_rows, height)
            slice0 = max(0, row_start - halo)
            slice1 = min(height, row_stop + halo)
            task = dask.delayed(goldstein_filter_torch)(
                complex_ifg[slice0:slice1],
                alpha=alpha,
                window=window,
                device=worker_device,
                output_row0=row_start,
                output_rows=row_stop - row_start,
                input_row0=slice0,
            )
            parts.append(
                da.from_delayed(
                    task,
                    shape=(row_stop - row_start, width),
                    dtype=complex_ifg.dtype,
                )
            )
    return da.concatenate(parts, axis=0)


def run_carrier_multiply(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    *,
    sign: float,
    row0: int = 0,
    col0: int = 0,
    native_height: int | None = None,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> np.ndarray:
    """Carrier-multiply via Dask scheduling, or eager Torch on the device."""
    from faninsar.processing.runtime.torch_kernels import tops_carrier_multiply_torch

    _reject_unpublished_device(device)
    resolved_client = client
    if _allow_remote_kernel(device, resolved_client, "carrier_multiply"):
        graph = _schedule_carrier_multiply(
            samples,
            model,
            sign=sign,
            row0=row0,
            col0=col0,
            native_height=native_height,
            device=device,
            chunk_rows=chunk_rows,
            client=resolved_client,
        )
        future = resolved_client.compute(graph)
        return np.asarray(resolved_client.gather(future))
    return tops_carrier_multiply_torch(
        samples,
        model,
        sign=sign,
        row0=row0,
        col0=col0,
        native_height=native_height,
        device=_local_device_after_remote_rejection(
            device, resolved_client, "carrier_multiply"
        ),
    )


def _carrier_multiply_at_points(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    *,
    centre_row: float,
    sign: float,
    device: str,
) -> np.ndarray:
    """Evaluate and apply a TOPS carrier at arbitrary source coordinates."""
    from faninsar.processing.runtime.torch_kernels import (
        carrier_multiply_torch,
        carrier_phase_at_points_torch,
    )

    phase = carrier_phase_at_points_torch(
        model,
        azimuth,
        range_index,
        centre_row=centre_row,
        dtype=np.float64,
        device=device,
    )
    return carrier_multiply_torch(samples, phase, sign=sign, device=device)


def run_carrier_multiply_at_points(
    samples: np.ndarray,
    model: TOPSCarrierModel,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    *,
    centre_row: float,
    sign: float,
    device: str = "auto",
    client: Any | None = None,
) -> np.ndarray:
    """Apply a carrier locally or as one task on an explicit GPU client."""
    _reject_unpublished_device(device)
    if _allow_remote_kernel(device, client, "carrier_multiply"):
        if not validate_cuda_worker_binding(client):
            message = "GPU worker-to-CUDA binding preflight failed"
            logger.error(message)
            raise RuntimeError(message)
        future = client.submit(
            _carrier_multiply_at_points,
            samples,
            model,
            azimuth,
            range_index,
            centre_row=centre_row,
            sign=sign,
            device="cuda",
            resources={"gpu": 1},
            pure=False,
        )
        return np.asarray(client.gather(future))
    return _carrier_multiply_at_points(
        samples,
        model,
        azimuth,
        range_index,
        centre_row=centre_row,
        sign=sign,
        device=_local_device_after_remote_rejection(device, client, "carrier_multiply"),
    )


def run_multilook_interferogram(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    multilook: tuple[int, int],
    dead_pixel_amp_threshold: float = 0.0,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> Any:
    """Form a multilooked interferogram via Dask, or eager Torch."""
    from faninsar.processing.interferometry.pair import InterferogramProduct
    from faninsar.processing.runtime.torch_kernels import multilook_interferogram_torch

    _reject_unpublished_device(device)
    resolved_client = client
    if _allow_remote_kernel(device, resolved_client, "multilook_interferogram"):
        graphs = _schedule_multilook_interferogram(
            primary,
            secondary,
            multilook=multilook,
            dead_pixel_amp_threshold=dead_pixel_amp_threshold,
            device=device,
            chunk_rows=chunk_rows,
            client=resolved_client,
        )
        futures = resolved_client.compute(list(graphs))
        computed = resolved_client.gather(futures)
        return InterferogramProduct(
            complex_ifg=computed[0],
            coherence=computed[1],
            wrapped_phase=computed[2],
            amplitude=computed[3],
        )
    return multilook_interferogram_torch(
        primary,
        secondary,
        multilook=multilook,
        dead_pixel_amp_threshold=dead_pixel_amp_threshold,
        device=_local_device_after_remote_rejection(
            device, resolved_client, "multilook_interferogram"
        ),
    )


def run_goldstein_filter(
    complex_ifg: np.ndarray,
    *,
    alpha: float,
    window: int,
    device: str = "auto",
    chunk_rows: int = 128,
    client: Any | None = None,
) -> np.ndarray:
    """Apply the Goldstein filter via Dask, or eager Torch."""
    from faninsar.processing.runtime.torch_kernels import goldstein_filter_torch

    _reject_unpublished_device(device)
    resolved_client = client
    if _allow_remote_kernel(device, resolved_client, "goldstein_filter"):
        graph = _schedule_goldstein_filter(
            complex_ifg,
            alpha=alpha,
            window=window,
            device=device,
            chunk_rows=chunk_rows,
            client=resolved_client,
        )
        future = resolved_client.compute(graph)
        return np.asarray(resolved_client.gather(future))
    return goldstein_filter_torch(
        complex_ifg,
        alpha=alpha,
        window=window,
        device=device,
    )


def run_remove_topographic_phase(
    complex_ifg: np.ndarray,
    topo_phase: np.ndarray,
    *,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> np.ndarray:
    """Flatten via Dask scheduling, or eager Torch."""
    _reject_unpublished_device(device)
    resolved_client = client
    if _allow_remote_kernel(device, resolved_client, "remove_topographic_phase"):
        graph = _schedule_remove_topographic_phase(
            complex_ifg,
            topo_phase,
            device=device,
            chunk_rows=chunk_rows,
            client=resolved_client,
        )
        future = resolved_client.compute(graph)
        return np.asarray(resolved_client.gather(future))
    from faninsar.processing.runtime.torch_kernels import remove_topographic_phase_torch

    return remove_topographic_phase_torch(
        complex_ifg,
        topo_phase,
        device=_local_device_after_remote_rejection(
            device, resolved_client, "remove_topographic_phase"
        ),
    )


def run_multilook_real(
    array: np.ndarray,
    az_looks: int,
    rg_looks: int,
    *,
    device: str = "auto",
    chunk_rows: int = 256,
    client: Any | None = None,
) -> np.ndarray:
    """Multilook a real field via Dask scheduling, or eager Torch."""
    _reject_unpublished_device(device)
    resolved_client = client
    if _allow_remote_kernel(device, resolved_client, "multilook_real"):
        graph = _schedule_multilook_real(
            array,
            az_looks,
            rg_looks,
            device=device,
            chunk_rows=chunk_rows,
            client=resolved_client,
        )
        future = resolved_client.compute(graph)
        return np.asarray(resolved_client.gather(future))
    from faninsar.processing.runtime.torch_kernels import multilook_real_torch

    return multilook_real_torch(
        array,
        az_looks,
        rg_looks,
        device=_local_device_after_remote_rejection(
            device, resolved_client, "multilook_real"
        ),
    )


def run_esd_azimuth_shift(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    device: str = "auto",
    client: Any | None = None,
    range_chunk_size: int = 512,
) -> Any:
    """Estimate ESD locally or as one chunked task on a trusted client."""
    from faninsar.processing.runtime.torch_kernels import esd_azimuth_shift_torch

    _reject_unpublished_device(device)
    if client is not None and use_gpu_resources(device, client):
        if not validate_cuda_worker_binding(client):
            message = "GPU worker-to-CUDA binding preflight failed"
            logger.error(message)
            raise RuntimeError(message)
        future = client.submit(
            esd_azimuth_shift_torch,
            reference,
            secondary,
            device="cuda",
            range_chunk_size=range_chunk_size,
            resources={"gpu": 1},
            pure=False,
        )
        return client.gather(future)
    return esd_azimuth_shift_torch(
        reference,
        secondary,
        device=device,
        range_chunk_size=range_chunk_size,
    )

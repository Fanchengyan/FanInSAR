"""Validated Python boundary for native-v2 fourteen-field results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.v2 import Operation, TransformResultV2

if TYPE_CHECKING:
    from collections.abc import Sequence

NATIVE_RESULT_FIELDS: tuple[str, ...] = (
    "latitude_deg",
    "longitude_deg",
    "height_m",
    "range_index",
    "azimuth_index",
    "converged",
    "iterations",
    "decision_residual",
    "final_residual",
    "tolerance",
    "max_iter_exhausted",
    "boundary_rechecked",
    "residual_range_m",
    "residual_doppler_hz",
)
NATIVE_ECEF_RESULT_FIELD_COUNT = len(NATIVE_RESULT_FIELDS) + 3
_NATIVE_FLOAT_FIELDS = {
    "latitude_deg",
    "longitude_deg",
    "height_m",
    "range_index",
    "azimuth_index",
    "decision_residual",
    "final_residual",
    "tolerance",
    "residual_range_m",
    "residual_doppler_hz",
}
_NATIVE_BOOL_FIELDS = {"converged", "max_iter_exhausted", "boundary_rechecked"}
logger = setup_logger(__name__)


def ecef_from_native_outputs(
    outputs: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and transfer only ECEF fields from the typed native ABI.

    Parameters
    ----------
    outputs : sequence of object
        The seventeen-field CUDA ECEF result: the canonical fourteen fields
        followed by ``x``, ``y``, and ``z`` ECEF tensors.

    Returns
    -------
    tuple of numpy.ndarray
        Host ECEF arrays.  The first fourteen fields are validated by shape,
        dtype, and field count but are intentionally not transferred.

    Raises
    ------
    ValueError
        If the typed ECEF result does not satisfy the native ABI contract.

    """
    if len(outputs) != NATIVE_ECEF_RESULT_FIELD_COUNT:
        message = (
            "native geometry ECEF ABI must return exactly seventeen fields; "
            f"received {len(outputs)}"
        )
        logger.error(message)
        raise ValueError(message)
    shape = getattr(outputs[0], "shape", None)
    if shape is None:
        message = "native geometry ECEF outputs must expose shapes"
        logger.error(message)
        raise ValueError(message)
    for name, value in zip(NATIVE_RESULT_FIELDS, outputs[:14], strict=True):
        if getattr(value, "shape", None) != shape:
            message = f"native field {name} has an incompatible shape"
            logger.error(message)
            raise ValueError(message)
        dtype = str(getattr(value, "dtype", None))
        expected = (
            {"torch.float64", "float64"}
            if name in _NATIVE_FLOAT_FIELDS
            else {"torch.bool", "bool"}
            if name in _NATIVE_BOOL_FIELDS
            else {"torch.int32", "int32"}
        )
        if dtype not in expected:
            message = f"native field {name} has invalid dtype {dtype}"
            logger.error(message)
            raise ValueError(message)
    ecef: list[np.ndarray] = []
    for name, value in zip(
        ("ecef_x_m", "ecef_y_m", "ecef_z_m"),
        outputs[14:],
        strict=True,
    ):
        if getattr(value, "shape", None) != shape:
            message = f"native field {name} has an incompatible shape"
            logger.error(message)
            raise ValueError(message)
        dtype = getattr(value, "dtype", None)
        if str(dtype) not in {"torch.float64", "float64"}:
            message = f"native field {name} must have float64 dtype"
            logger.error(message)
            raise ValueError(message)
        detach = getattr(value, "detach", None)
        candidate = detach() if callable(detach) else value
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        numpy = getattr(candidate, "numpy", None)
        ecef.append(np.asarray(numpy() if callable(numpy) else candidate))
    return ecef[0], ecef[1], ecef[2]


def result_from_native_outputs(
    outputs: Sequence[object], *, operation: Operation | str
) -> TransformResultV2:
    """Convert one native output sequence through the central v2 validator.

    Parameters
    ----------
    outputs : sequence of object
        The ordered fourteen tensors returned by ``geo2rdr_cpu`` or
        ``rdr2geo_cpu``.  CPU tensors must expose ``detach().cpu().numpy()``.
    operation : Operation or str
        Operation identity used for operation-specific tolerance validation.

    Returns
    -------
    TransformResultV2
        Validated NumPy arrays with the exact public result contract. CUDA
        tensor fields retain the independent host allocation created by their
        ``.cpu()`` transfer; CPU tensor and NumPy fields are copied.

    Raises
    ------
    ValueError
        If the native ABI returns anything other than fourteen fields.

    """
    if len(outputs) != len(NATIVE_RESULT_FIELDS):
        message = (
            "native geometry ABI must return exactly fourteen fields; "
            f"received {len(outputs)}"
        )
        raise ValueError(message)
    fields: dict[str, np.ndarray] = {}
    for name, value in zip(NATIVE_RESULT_FIELDS, outputs, strict=True):
        candidate = value
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            tensor = detach()
            candidate = tensor.cpu().numpy()
            if getattr(getattr(tensor, "device", None), "type", None) == "cuda":
                # Each CUDA ``.cpu()`` call creates a fresh host allocation;
                # retaining its NumPy view avoids a second 14-field copy.
                fields[name] = np.asarray(candidate)
                continue
        fields[name] = np.array(candidate, copy=True)
    # Do not coerce ABI dtypes here.  The central result contract must reject
    # an extension that silently changes its field types; native CPU/CUDA
    # kernels are responsible for publishing float64/int32/bool fields.
    invalid_mask = ~np.isfinite(np.asarray(fields["latitude_deg"], dtype=np.float64))
    return TransformResultV2.from_arrays(
        fields,
        operation=operation,
        invalid_mask=invalid_mask,
    )

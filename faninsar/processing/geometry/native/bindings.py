"""Validated Python boundary for native fourteen-field results.

The native extension ABI stays at the shared fourteen solver fields.  The
unified public ``TransformResult`` contract is seventeen fields; this module
computes ``incidence_deg``, ``look_azimuth_deg``, and the rdr2geo
``geom_flags`` layover/shadow bitmask at the Python boundary from the
prepared geometry model and the solved ground position so the compiled
artifact remains byte-identical.  Extending the ABI to emit the angles and
flags natively is a recorded follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import reject
from faninsar.processing.geometry.transforms import (
    Operation,
    TransformResult,
    geom_flags,
    look_angles_deg,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.processing.geometry.transforms import RadarGeometryModel

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
        reject(message, ValueError)
    shape = getattr(outputs[0], "shape", None)
    if shape is None:
        message = "native geometry ECEF outputs must expose shapes"
        reject(message, ValueError)
    for name, value in zip(NATIVE_RESULT_FIELDS, outputs[:14], strict=True):
        if getattr(value, "shape", None) != shape:
            message = f"native field {name} has an incompatible shape"
            reject(message, ValueError)
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
            reject(message, ValueError)
    ecef: list[np.ndarray] = []
    for name, value in zip(
        ("ecef_x_m", "ecef_y_m", "ecef_z_m"),
        outputs[14:],
        strict=True,
    ):
        if getattr(value, "shape", None) != shape:
            message = f"native field {name} has an incompatible shape"
            reject(message, ValueError)
        dtype = getattr(value, "dtype", None)
        if str(dtype) not in {"torch.float64", "float64"}:
            message = f"native field {name} must have float64 dtype"
            reject(message, ValueError)
        detach = getattr(value, "detach", None)
        candidate = detach() if callable(detach) else value
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        numpy = getattr(candidate, "numpy", None)
        ecef.append(np.asarray(numpy() if callable(numpy) else candidate))
    return ecef[0], ecef[1], ecef[2]


def result_from_native_outputs(
    outputs: Sequence[object],
    *,
    operation: Operation | str,
    model: RadarGeometryModel | None = None,
) -> TransformResult:
    """Convert one native output sequence through the central validator.

    Parameters
    ----------
    outputs : sequence of object
        The ordered fourteen tensors returned by ``geo2rdr_cpu`` or
        ``rdr2geo_cpu``.  CPU tensors must expose ``detach().cpu().numpy()``.
    operation : Operation or str
        Operation identity used for operation-specific tolerance validation.
    model : RadarGeometryModel, optional
        Prepared geometry model.  When given, ``incidence_deg`` and
        ``look_azimuth_deg`` are computed from the orbit and the solved
        ground position; without it both fields publish NaN.  rdr2geo also
        publishes the ``geom_flags`` layover/shadow bitmask; without a model
        or for geo2rdr the field is all zeros.

    Returns
    -------
    TransformResult
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
    op = Operation(operation)
    if model is not None:
        incidence_deg, look_azimuth_deg = look_angles_deg(
            model,
            fields["azimuth_index"],
            fields["latitude_deg"],
            fields["longitude_deg"],
            fields["height_m"],
            fields["converged"],
        )
        fields["incidence_deg"] = incidence_deg
        fields["look_azimuth_deg"] = look_azimuth_deg
        if op is Operation.RDR2GEO:
            fields["geom_flags"] = geom_flags(
                model,
                fields["azimuth_index"],
                fields["range_index"],
                fields["latitude_deg"],
                fields["longitude_deg"],
                fields["height_m"],
                fields["converged"],
            )
        else:
            fields["geom_flags"] = np.zeros(
                fields["latitude_deg"].shape, dtype=np.uint8
            )
    else:
        shape = fields["latitude_deg"].shape
        fields["incidence_deg"] = np.full(shape, np.nan, dtype=np.float64)
        fields["look_azimuth_deg"] = np.full(shape, np.nan, dtype=np.float64)
        fields["geom_flags"] = np.zeros(shape, dtype=np.uint8)
    return TransformResult.from_arrays(
        fields,
        operation=operation,
        invalid_mask=invalid_mask,
    )

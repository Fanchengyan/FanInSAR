"""Validated Python boundary for native-v2 fourteen-field results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
        Validated, owned NumPy arrays with the exact public result contract.

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
            candidate = detach().cpu().numpy()
        fields[name] = np.array(candidate, copy=True)
    fields["iterations"] = np.asarray(fields["iterations"], dtype=np.int32)
    for name in (
        "converged",
        "max_iter_exhausted",
        "boundary_rechecked",
    ):
        fields[name] = np.asarray(fields[name], dtype=bool)
    for name in NATIVE_RESULT_FIELDS:
        if name not in {
            "iterations",
            "converged",
            "max_iter_exhausted",
            "boundary_rechecked",
        }:
            fields[name] = np.asarray(fields[name], dtype=np.float64)
    return TransformResultV2.from_arrays(fields, operation=operation)

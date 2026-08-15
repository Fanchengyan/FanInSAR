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

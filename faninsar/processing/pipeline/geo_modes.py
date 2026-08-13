"""Direct geographic-grid SLC coregistration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.memory import release_memmap_pages
from faninsar.processing.pipeline.geo_resample import (
    compose_secondary_coordinates,
    resample_complex_at_coordinates,
)

if TYPE_CHECKING:
    from faninsar.processing.coreg.offsets import OffsetFieldResult
    from faninsar.processing.memory import MemoryWatchdog
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT
    from faninsar.processing.tops.deramp import TOPSCarrierModel

__all__ = ["coregister_geocoded_slcs", "coregister_geocoded_slcs_chunked"]

logger = setup_logger(__name__)


def _apply_reramp(
    deramped: np.ndarray,
    carrier: TOPSCarrierModel,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    *,
    native_height: int,
    device: str,
    dask_client: Any | None,
) -> np.ndarray:
    from faninsar.backends.dask_gpu import should_accelerate

    if not should_accelerate(device, dask_client, kernel="carrier_multiply"):
        logger.warning(
            "Geographic reramp is not CUDA-qualified; using the NumPy CPU "
            "reference path"
        )
        from faninsar.processing.tops.deramp import carrier_phase_at_points

        phase = carrier_phase_at_points(
            carrier,
            azimuth,
            range_index,
            centre_row=float(native_height // 2),
            dtype=np.float32,
        )
        output = np.asarray(deramped, dtype=np.complex64) * np.exp(
            1j * np.asarray(phase, dtype=np.float64)
        )
        return output.astype(np.complex64, copy=False)

    from faninsar.backends.dask_gpu import run_carrier_multiply_at_points

    return run_carrier_multiply_at_points(
        np.asarray(deramped, dtype=np.complex64),
        carrier,
        azimuth,
        range_index,
        centre_row=float(native_height // 2),
        sign=1.0,
        device=device,
        client=dask_client,
    )


def coregister_geocoded_slcs(
    reference_deramped: np.ndarray,
    secondary_deramped: np.ndarray,
    *,
    reference_carrier: TOPSCarrierModel,
    secondary_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    executor: Literal["torch"] = "torch",
    device: str = "auto",
    dask_client: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coregister two deramped SLCs directly on a geographic grid.

    Parameters
    ----------
    reference_deramped, secondary_deramped : numpy.ndarray
        Native-resolution deramped SLCs.
    reference_carrier, secondary_carrier : TOPSCarrierModel
        Acquisition-specific TOPS carrier models.
    reference_lut : Geo2RdrLUT
        Reference radar coordinates at geographic pixel centres.
    offsets : OffsetFieldResult
        Dense reference-to-secondary radar offsets.
    executor : {"torch"}, optional
        Unified Torch Lanczos implementation.
    device : {"auto", "cpu", "cuda"}, optional
        Torch device for the accelerated implementation.
    dask_client : object or None, optional
        Explicitly trusted Dask client for remote CUDA reramp tasks.

    Returns
    -------
    reference_geo, secondary_geo, valid : tuple[numpy.ndarray, ...]
        Reramped aligned SLCs and their shared validity mask.

    Notes
    -----
    Each SLC is interpolated exactly once in the deramped domain. Reramping is
    evaluated analytically at the same fractional source coordinates used by
    that interpolation.

    """
    secondary_azimuth, secondary_range, secondary_valid = compose_secondary_coordinates(
        reference_lut, offsets
    )
    reference_geo, reference_valid = resample_complex_at_coordinates(
        reference_deramped,
        reference_lut.az_full,
        reference_lut.rg_full,
        valid=reference_lut.valid,
        executor=executor,
        device=device,
    )
    secondary_geo, secondary_valid = resample_complex_at_coordinates(
        secondary_deramped,
        secondary_azimuth,
        secondary_range,
        valid=secondary_valid,
        executor=executor,
        device=device,
    )
    valid = (
        reference_valid
        & secondary_valid
        & np.isfinite(reference_geo.real)
        & np.isfinite(reference_geo.imag)
        & np.isfinite(secondary_geo.real)
        & np.isfinite(secondary_geo.imag)
    )
    reference_geo = _apply_reramp(
        reference_geo,
        reference_carrier,
        reference_lut.az_full,
        reference_lut.rg_full,
        native_height=reference_deramped.shape[0],
        device=device,
        dask_client=dask_client,
    )
    secondary_geo = _apply_reramp(
        secondary_geo,
        secondary_carrier,
        secondary_azimuth,
        secondary_range,
        native_height=secondary_deramped.shape[0],
        device=device,
        dask_client=dask_client,
    )
    invalid_value = np.complex64(np.nan + 1j * np.nan)
    reference_geo = np.where(valid, reference_geo, invalid_value).astype(np.complex64)
    secondary_geo = np.where(valid, secondary_geo, invalid_value).astype(np.complex64)
    return reference_geo, secondary_geo, valid


def coregister_geocoded_slcs_chunked(
    reference_deramped: np.ndarray,
    secondary_deramped: np.ndarray,
    *,
    reference_carrier: TOPSCarrierModel,
    secondary_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
    output_dir: str | Path,
    row_chunk: int = 256,
    executor: Literal["torch"] = "torch",
    device: str = "auto",
    dask_client: Any | None = None,
    watchdog: MemoryWatchdog | None = None,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    """Coregister geographic SLCs into disk-backed row tiles.

    Parameters
    ----------
    reference_deramped, secondary_deramped : numpy.ndarray
        Native-resolution deramped SLCs.
    reference_carrier, secondary_carrier : TOPSCarrierModel
        Acquisition-specific TOPS carrier models.
    reference_lut : Geo2RdrLUT
        Reference radar coordinates on the geographic grid.
    offsets : OffsetFieldResult
        Dense reference-to-secondary offsets.
    output_dir : str or pathlib.Path
        Directory for disk-backed intermediate arrays.
    row_chunk : int, optional
        Geographic rows processed per tile.
    executor : {"torch"}, optional
        Unified Torch Lanczos implementation.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch device for accelerated Lanczos.
    dask_client : object or None, optional
        Explicitly trusted Dask client for remote CUDA reramp tasks.
    watchdog : MemoryWatchdog, optional
        Memory guard sampled after every completed tile.

    Returns
    -------
    reference, secondary, valid : tuple[numpy.memmap, ...]
        Disk-backed complete geographic SLCs and shared validity mask.

    """
    from faninsar.processing.errors import reject_invalid_state
    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT

    if row_chunk < 1:
        reject_invalid_state("row_chunk must be >= 1")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shape = reference_lut.shape
    reference_output = np.memmap(
        directory / "reference_geo.complex64",
        mode="w+",
        dtype=np.complex64,
        shape=shape,
    )
    secondary_output = np.memmap(
        directory / "secondary_geo.complex64",
        mode="w+",
        dtype=np.complex64,
        shape=shape,
    )
    valid_output = np.memmap(
        directory / "geo_valid.bool",
        mode="w+",
        dtype=np.bool_,
        shape=shape,
    )
    for row_start in range(0, shape[0], row_chunk):
        row_stop = min(row_start + row_chunk, shape[0])
        rows = slice(row_start, row_stop)
        tile_lut = Geo2RdrLUT(
            az_full=reference_lut.az_full[rows],
            rg_full=reference_lut.rg_full[rows],
            valid=reference_lut.valid[rows],
            full_radar_shape=reference_lut.full_radar_shape,
            height_m=reference_lut.height_m,
            height_full=(
                None
                if reference_lut.height_full is None
                else reference_lut.height_full[rows]
            ),
        )
        reference_tile, secondary_tile, valid_tile = coregister_geocoded_slcs(
            reference_deramped,
            secondary_deramped,
            reference_carrier=reference_carrier,
            secondary_carrier=secondary_carrier,
            reference_lut=tile_lut,
            offsets=offsets,
            executor=executor,
            device=device,
            dask_client=dask_client,
        )
        reference_output[rows] = reference_tile
        secondary_output[rows] = secondary_tile
        valid_output[rows] = valid_tile
        del reference_tile, secondary_tile, valid_tile
        for array in (reference_output, secondary_output, valid_output):
            release_memmap_pages(array)
        for array in (
            reference_lut.az_full,
            reference_lut.rg_full,
            reference_lut.valid,
        ):
            if isinstance(array, np.memmap):
                release_memmap_pages(array)
        if watchdog is not None:
            watchdog.sample(f"geo_coregister:{row_start}:{row_stop}")
    reference_output.flush()
    secondary_output.flush()
    valid_output.flush()
    return reference_output, secondary_output, valid_output

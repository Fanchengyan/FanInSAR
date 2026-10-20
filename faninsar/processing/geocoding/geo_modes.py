"""Direct geographic-grid SLC coregistration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geocoding.coordinates import RadarCoordinates
from faninsar.processing.geocoding.geo_resample import sample
from faninsar.processing.runtime.memory import release_memmap_pages

if TYPE_CHECKING:
    from faninsar.processing.coregistration.offsets import OffsetFieldResult
    from faninsar.processing.coregistration.tops.deramp import TOPSCarrierModel
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
    from faninsar.processing.runtime.memory import MemoryWatchdog

__all__ = [
    "coregister_geocoded_slcs",
    "coregister_geocoded_slcs_chunked",
    "geocode_reference_slc",
    "geocode_reference_slc_chunked",
    "geocode_secondary_slc",
    "geocode_secondary_slc_chunked",
]

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
    from faninsar.processing.runtime.dask_gpu import should_accelerate

    if not should_accelerate(device, dask_client, kernel="carrier_multiply"):
        logger.warning(
            "Geographic reramp is not CUDA-qualified; using the NumPy CPU "
            "reference path"
        )
        from faninsar.processing.coregistration.tops.deramp import (
            carrier_phase_at_points,
        )

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

    from faninsar.processing.runtime.dask_gpu import run_carrier_multiply_at_points

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


def geocode_reference_slc(
    reference_deramped: np.ndarray,
    *,
    reference_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    device: str = "auto",
    dask_client: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the Reference SLC on the geographic grid.

    The Reference is masked only by its own LUT and sampling support, so the
    result is the same for every secondary of a Stack.

    Returns
    -------
    reference_geo, reference_valid : tuple[numpy.ndarray, numpy.ndarray]
        Reramped Reference samples (NaN outside support) and its support.

    """
    coordinates = RadarCoordinates(
        reference_lut.az_full,
        reference_lut.rg_full,
        reference_lut.valid,
    )
    reference_geo, reference_valid = sample(
        reference_deramped,
        coordinates,
        kind="complex",
        device=device,
    )
    reference_valid = (
        reference_valid
        & np.isfinite(reference_geo.real)
        & np.isfinite(reference_geo.imag)
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
    invalid_value = np.complex64(np.nan + 1j * np.nan)
    reference_geo = np.where(reference_valid, reference_geo, invalid_value).astype(
        np.complex64
    )
    return reference_geo, reference_valid


def geocode_secondary_slc(
    secondary_deramped: np.ndarray,
    *,
    secondary_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    reference_valid: np.ndarray,
    offsets: OffsetFieldResult,
    device: str = "auto",
    dask_client: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one secondary SLC on the geographic grid of the Reference.

    Parameters
    ----------
    secondary_deramped : numpy.ndarray
        Native-resolution deramped secondary SLC.
    secondary_carrier : TOPSCarrierModel
        Secondary TOPS carrier model.
    reference_lut : Geo2RdrLUT
        Reference radar coordinates on the geographic grid.
    reference_valid : numpy.ndarray
        Reference support on the same rows (from :func:`geocode_reference_slc`
        or the persisted Reference). The pair validity is its intersection with
        the secondary sampling support.
    offsets : OffsetFieldResult
        Dense Reference-to-secondary radar offsets.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch device.
    dask_client : object or None, optional
        Explicitly trusted Dask client for remote CUDA reramp tasks.

    Returns
    -------
    secondary_geo, valid : tuple[numpy.ndarray, numpy.ndarray]
        Reramped secondary samples (NaN outside the pair support) and the joint
        pair validity mask.

    """
    reference_coordinates = RadarCoordinates(
        reference_lut.az_full,
        reference_lut.rg_full,
        reference_lut.valid,
    )
    secondary_coordinates = reference_coordinates.minus_offsets(
        offsets,
        device=device,
    )
    secondary_geo, secondary_valid = sample(
        secondary_deramped,
        secondary_coordinates,
        kind="complex",
        device=device,
    )
    valid = (
        np.asarray(reference_valid, dtype=bool)
        & secondary_valid
        & np.isfinite(secondary_geo.real)
        & np.isfinite(secondary_geo.imag)
    )
    secondary_geo = _apply_reramp(
        secondary_geo,
        secondary_carrier,
        secondary_coordinates.az,
        secondary_coordinates.rg,
        native_height=secondary_deramped.shape[0],
        device=device,
        dask_client=dask_client,
    )
    invalid_value = np.complex64(np.nan + 1j * np.nan)
    secondary_geo = np.where(valid, secondary_geo, invalid_value).astype(np.complex64)
    return secondary_geo, valid


def coregister_geocoded_slcs(
    reference_deramped: np.ndarray,
    secondary_deramped: np.ndarray,
    *,
    reference_carrier: TOPSCarrierModel,
    secondary_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    offsets: OffsetFieldResult,
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
    device : {"auto", "cpu", "cuda"}, optional
        Torch device for the secondary offset composition and the
        accelerated Lanczos implementation.
    dask_client : object or None, optional
        Explicitly trusted Dask client for remote CUDA reramp tasks.

    Returns
    -------
    reference_geo, secondary_geo, valid : tuple[numpy.ndarray, ...]
        Reramped aligned SLCs and the joint pair validity mask.

    Notes
    -----
    Each SLC is interpolated exactly once in the deramped domain. Reramping is
    evaluated analytically at the same fractional source coordinates used by
    that interpolation. The reference is masked only by its own LUT and
    sampling support, so it is identical for every secondary of a Stack; the
    secondary and ``valid`` carry the joint pair support.

    """
    reference_geo, reference_valid = geocode_reference_slc(
        reference_deramped,
        reference_carrier=reference_carrier,
        reference_lut=reference_lut,
        device=device,
        dask_client=dask_client,
    )
    secondary_geo, valid = geocode_secondary_slc(
        secondary_deramped,
        secondary_carrier=secondary_carrier,
        reference_lut=reference_lut,
        reference_valid=reference_valid,
        offsets=offsets,
        device=device,
        dask_client=dask_client,
    )
    return reference_geo, secondary_geo, valid


def _row_lut(lut: Geo2RdrLUT, rows: slice) -> Geo2RdrLUT:
    """Return the LUT restricted to one geographic row tile."""
    from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT

    return Geo2RdrLUT(
        az_full=lut.az_full[rows],
        rg_full=lut.rg_full[rows],
        valid=lut.valid[rows],
        full_radar_shape=lut.full_radar_shape,
        height_m=lut.height_m,
        height_full=None if lut.height_full is None else lut.height_full[rows],
    )


def _release_lut_pages(lut: Geo2RdrLUT) -> None:
    for array in (lut.az_full, lut.rg_full, lut.valid):
        if isinstance(array, np.memmap):
            release_memmap_pages(array)


def geocode_reference_slc_chunked(
    reference_deramped: np.ndarray,
    *,
    reference_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    output_dir: str | Path,
    row_chunk: int = 256,
    device: str = "auto",
    dask_client: Any | None = None,
) -> tuple[np.memmap, np.memmap]:
    """Sample the Reference SLC on the geographic grid into row tiles.

    Parameters
    ----------
    reference_deramped : numpy.ndarray
        Native-resolution deramped Reference SLC.
    reference_carrier : TOPSCarrierModel
        Reference TOPS carrier model.
    reference_lut : Geo2RdrLUT
        Reference radar coordinates on the geographic grid.
    output_dir : str or pathlib.Path
        Directory for the disk-backed outputs.
    row_chunk : int, optional
        Geographic rows per tile.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch device.
    dask_client : object or None, optional
        Explicitly trusted Dask client for remote CUDA reramp tasks.

    Returns
    -------
    reference, reference_valid : tuple[numpy.memmap, numpy.memmap]
        Reramped Reference samples (NaN outside support) and its support.

    """
    from faninsar.processing.errors import reject_invalid_state

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
    reference_valid = np.memmap(
        directory / "reference_valid.bool",
        mode="w+",
        dtype=np.bool_,
        shape=shape,
    )
    for row_start in range(0, shape[0], row_chunk):
        rows = slice(row_start, min(row_start + row_chunk, shape[0]))
        tile, tile_valid = geocode_reference_slc(
            reference_deramped,
            reference_carrier=reference_carrier,
            reference_lut=_row_lut(reference_lut, rows),
            device=device,
            dask_client=dask_client,
        )
        reference_output[rows] = tile
        reference_valid[rows] = tile_valid
        del tile, tile_valid
        release_memmap_pages(reference_output)
        _release_lut_pages(reference_lut)
    reference_output.flush()
    reference_valid.flush()
    return reference_output, reference_valid


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
        Dense reference-to-secondary radar offsets.
    output_dir : str or pathlib.Path
        Directory for disk-backed intermediate arrays.
    row_chunk : int, optional
        Geographic rows processed per tile; each tile's secondary
        coordinates are composed exactly once.
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
    reference_output, reference_valid = geocode_reference_slc_chunked(
        reference_deramped,
        reference_carrier=reference_carrier,
        reference_lut=reference_lut,
        output_dir=output_dir,
        row_chunk=row_chunk,
        device=device,
        dask_client=dask_client,
    )
    directory = Path(output_dir)
    secondary_output, valid_output = geocode_secondary_slc_chunked(
        secondary_deramped,
        secondary_carrier=secondary_carrier,
        reference_lut=reference_lut,
        reference_valid=reference_valid,
        offsets=offsets,
        output_dir=directory,
        row_chunk=row_chunk,
        device=device,
        dask_client=dask_client,
        watchdog=watchdog,
    )
    return reference_output, secondary_output, valid_output


def geocode_secondary_slc_chunked(
    secondary_deramped: np.ndarray,
    *,
    secondary_carrier: TOPSCarrierModel,
    reference_lut: Geo2RdrLUT,
    reference_valid: np.ndarray,
    offsets: OffsetFieldResult,
    output_dir: str | Path,
    row_chunk: int = 256,
    device: str = "auto",
    dask_client: Any | None = None,
    watchdog: MemoryWatchdog | None = None,
) -> tuple[np.memmap, np.memmap]:
    """Sample one secondary on the Reference geographic grid in row tiles.

    Returns
    -------
    secondary, valid : tuple[numpy.memmap, numpy.memmap]
        Disk-backed geographic secondary SLC and joint pair validity.

    """
    from faninsar.processing.errors import reject_invalid_state

    if row_chunk < 1:
        reject_invalid_state("row_chunk must be >= 1")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shape = reference_lut.shape
    if tuple(np.shape(reference_valid)) != tuple(shape):
        reject_invalid_state("reference support does not match the geo LUT shape")
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
        secondary_tile, valid_tile = geocode_secondary_slc(
            secondary_deramped,
            secondary_carrier=secondary_carrier,
            reference_lut=_row_lut(reference_lut, rows),
            reference_valid=np.asarray(reference_valid[rows], dtype=bool),
            offsets=offsets,
            device=device,
            dask_client=dask_client,
        )
        secondary_output[rows] = secondary_tile
        valid_output[rows] = valid_tile
        del secondary_tile, valid_tile
        for array in (secondary_output, valid_output):
            release_memmap_pages(array)
        _release_lut_pages(reference_lut)
        if watchdog is not None:
            watchdog.sample(f"geo_coregister:{row_start}:{row_stop}")
    secondary_output.flush()
    valid_output.flush()
    return secondary_output, valid_output

"""Byte-offset fast path for extracting TOPS bursts from Sentinel-1 SAFE.

ESA annotates every burst with a ``byteOffset`` field that locates the
burst's first line inside the (uncompressed) measurement GeoTIFF.  When the
source is a local SAFE ZIP the burst can be reconstructed with a single
``seek + read`` on the deflated member followed by an interleaved int16 ->
complex64 reinterpretation, avoiding the per-line rasterio-window overhead.

When the source is not a byte-seekable archive (e.g. an in-memory TIFF used
by the test-suite, or a directory on disk) the functions here transparently
delegate to :func:`faninsar.missions.sentinel1.io.read.read_full_burst`, so
both paths return identical :class:`BurstArray` objects.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger
from faninsar.missions.sentinel1.errors import reject_product
from faninsar.missions.sentinel1.io.read import (
    BurstArray,
    _burst_line_offset,
    _valid_column_bounds,
)

if TYPE_CHECKING:
    from faninsar.missions.sentinel1.types import S1Burst, S1Swath

logger = setup_logger(__name__)

# Sentinel-1 SLC measurement pixels are stored as interleaved int16
# (real, imag) pairs — the GDAL "complex_int16" sample format.
_BYTES_PER_PIXEL = 4


def _measurement_zip_member(swath: S1Swath) -> tuple[Path, str] | None:
    """Return ``(zip_path, member_name)`` if the measurement lives in a ZIP.

    Returns ``None`` when the measurement is not a ZIP-backed asset, signalling
    the caller should fall back to the rasterio read path.
    """
    mp = str(swath.measurement_path)
    # /vsizip//<zip_path>/<member>  produced by faninsar.missions.sentinel1.safe
    if mp.startswith("/vsizip/"):
        body = mp[len("/vsizip/") :].lstrip("/")
        sep = body.find(".zip/")
        if sep == -1:
            return None
        zip_path = Path(body[: sep + 4])
        member = body[sep + 5 :].lstrip("/")
        if zip_path.exists():
            return zip_path, member
    # raw .zip path (rare; measurement_path normally uses /vsizip/)
    if mp.endswith(".tiff") and ".zip" in mp:
        idx = mp.find(".zip")
        candidate = Path(mp[: idx + 4])
        if candidate.exists():
            member = mp[idx + 5 :].lstrip("/")
            return candidate, member
    return None


def _burst_pixel_offset(swath: S1Swath, burst: S1Burst) -> int:
    """Return the byte offset of the burst's first line in the pixel stream.

    The annotation ``byteOffset`` already encodes this.  For burst 0 the
    annotation value is small (points past the TIFF header) and is used
    verbatim; subsequent bursts add ``lines_per_burst * samples_per_burst *
    4``.  We recompute from the index rather than trusting the annotation
    value directly so the call is robust against quirky fixtures.
    """
    if burst.index == 0:
        return int(burst.byte_offset)
    first = swath.bursts[0]
    return int(first.byte_offset) + burst.index * (
        swath.lines_per_burst * swath.samples_per_burst * _BYTES_PER_PIXEL
    )


def _bytes_to_complex(raw: bytes, lines: int, samples: int) -> np.ndarray:
    """Reinterpret interleaved int16 bytes as a complex64 array."""
    expected = lines * samples * _BYTES_PER_PIXEL
    if len(raw) != expected:
        reject_product(
            f"burst byte slice is {len(raw)} bytes, expected {expected} "
            f"({lines}x{samples}x{_BYTES_PER_PIXEL})"
        )
    pairs = np.frombuffer(raw, dtype="<i2").reshape(lines, samples, 2)
    real = pairs[..., 0].astype(np.float32)
    imag = pairs[..., 1].astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def _extract_from_zip(
    zip_path: Path,
    member: str,
    swath: S1Swath,
    burst: S1Burst,
) -> BurstArray:
    """Read a burst straight from a SAFE ZIP member via byte offset."""
    offset = _burst_pixel_offset(swath, burst)
    length = burst.lines * swath.samples_per_burst * _BYTES_PER_PIXEL
    with zipfile.ZipFile(zip_path) as archive:
        try:
            with archive.open(member) as stream:
                stream.seek(offset)
                raw = stream.read(length)
        except KeyError as err:  # pragma: no cover - defensive
            reject_product(
                f"measurement member {member!r} missing from {zip_path}: {err}"
            )
            raise
    full = _bytes_to_complex(raw, burst.lines, swath.samples_per_burst)

    # Apply the same valid-sample envelope + masking used by read_full_burst
    # so both paths return bitwise-identical BurstArrays.
    col0, col1 = _valid_column_bounds(burst)
    first = np.asarray(burst.first_valid_sample, dtype=np.int32)
    last = np.asarray(burst.last_valid_sample, dtype=np.int32)
    cols = np.arange(col0, col1, dtype=np.int32)[None, :]
    valid_mask = (
        (first[:, None] >= 0) & (cols >= first[:, None]) & (cols <= last[:, None])
    )
    samples = full[:, col0:col1]
    samples = np.where(valid_mask, samples, 0)

    row0 = _burst_line_offset(swath, burst)
    logger.info(
        "Extracted burst %s via byteOffset from %s (offset=%d, %d MiB complex64)",
        burst.index,
        zip_path.name,
        offset,
        samples.nbytes // (1024 * 1024),
    )
    return BurstArray(
        samples=samples,
        row0=row0,
        col0=col0,
        burst_index=burst.index,
        valid_mask=valid_mask,
    )


def extract_burst(swath: S1Swath, *, burst_index: int = 0) -> BurstArray:
    """Extract one TOPS burst from a SAFE product.

    Uses the annotation ``byteOffset`` fast path when the measurement is a
    local ZIP member, otherwise delegates to :func:`read_full_burst`.  Both
    paths return a :class:`BurstArray` with identical samples and valid mask.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath.
    burst_index : int, optional
        Burst index within the swath.

    Returns
    -------
    BurstArray
        Complex64 samples of shape ``(lines_per_burst, n_range_valid)``.

    Raises
    ------
    Sentinel1ProductError
        If ``burst_index`` is out of range or the burst has no valid samples.

    Examples
    --------
    >>> from faninsar.missions.sentinel1 import open_safe_product, extract_burst
    >>> product = open_safe_product("S1A_..._VV.SAFE.zip")
    >>> iw1 = product.swath("IW1", "VV")
    >>> burst = extract_burst(iw1, burst_index=4)

    """
    if burst_index < 0 or burst_index >= len(swath.bursts):
        reject_product(f"burst_index {burst_index} out of range for {swath.swath}")
    burst = swath.bursts[burst_index]
    zipped = _measurement_zip_member(swath)
    if zipped is not None:
        zip_path, member = zipped
        return _extract_from_zip(zip_path, member, swath, burst)
    # Fallback: in-memory / directory TIFFs -> rasterio window path.
    from faninsar.missions.sentinel1.io.read import read_full_burst

    return read_full_burst(swath, burst_index=burst_index)


def extract_bursts(
    swath: S1Swath,
    *,
    burst_indices: list[int] | None = None,
) -> list[BurstArray]:
    """Extract one or more bursts from a sub-swath.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath.
    burst_indices : list of int, optional
        Bursts to extract. Defaults to all bursts.

    Returns
    -------
    list[BurstArray]
        Burst arrays in burst-index order.

    """
    if burst_indices is None:
        burst_indices = list(range(len(swath.bursts)))
    return [extract_burst(swath, burst_index=i) for i in burst_indices]


def estimate_burst_bytes(swath: S1Swath, burst_index: int = 0) -> int:
    """Return the uncompressed byte size of one burst's pixel block.

    Useful to estimate the I/O cost of extracting a burst, both for local
    archives and (when a COG-tiled source eventually exists) for remote
    byte-range planning.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath.
    burst_index : int, optional
        Burst index within the swath.

    Returns
    -------
    int
        Number of uncompressed bytes occupied by the burst pixel block
        (``lines_per_burst * samples_per_burst * 4``).

    """
    if burst_index < 0 or burst_index >= len(swath.bursts):
        reject_product(f"burst_index {burst_index} out of range for {swath.swath}")
    return swath.lines_per_burst * swath.samples_per_burst * _BYTES_PER_PIXEL

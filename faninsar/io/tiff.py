"""Bounded inspection, storage mapping and local decoding of stripped TIFFs."""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
    from typing import NoReturn

logger = setup_logger(__name__)
HEADER_PROBE_BYTES = 16
TIFF_PLANNER_VERSION = "strips-v1"


@dataclass(frozen=True, slots=True)
class ByteExtent:
    """Half-open encoded region given by byte offset and length."""

    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class PixelWindow:
    """Half-open physical image rectangle, independent of target placement."""

    row0: int
    row1: int
    col0: int
    col1: int


@dataclass(frozen=True, slots=True)
class TiffLayout:
    """Validated uncompressed stripped complex-int16 image layout.

    Parameters
    ----------
    width, height : int
        Physical image dimensions.
    strip_offsets, strip_byte_counts : tuple of int
        Encoded strip regions in the original object.
    rows_per_strip : int
        Maximum rows stored per strip.
    endian : {"<", ">"}
        Sample byte order.
    bytes_per_sample : int
        Storage stride, four bytes for complex signed int16.

    """

    width: int
    height: int
    strip_offsets: tuple[int, ...]
    strip_byte_counts: tuple[int, ...]
    rows_per_strip: int
    endian: str = "<"
    bytes_per_sample: int = 4

    @property
    def digest(self) -> str:
        """Digest of the immutable storage and decoding facts."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class TiffRegion:
    """Explicit extent-to-pixel mapping; labels are never used for placement."""

    extent: ByteExtent
    window: PixelWindow


@dataclass(frozen=True, slots=True)
class TiffReadPlan:
    """Immutable layout, physical windows and exact encoded dependencies."""

    layout: TiffLayout
    regions: tuple[TiffRegion, ...]
    version: str = TIFF_PLANNER_VERSION


@dataclass(frozen=True, slots=True)
class RasterBlock:
    """Bounded decoded samples and their original physical pixel rectangle."""

    window: PixelWindow
    samples: np.ndarray


@dataclass(frozen=True, slots=True)
class TiffMetadataLimits:
    """Bounds checked before remote metadata reads or table allocation."""

    max_requests: int = 32
    max_bytes: int = 8 * 1024 * 1024
    max_entries: int = 512
    max_elements: int = 1_000_000
    max_seconds: float = 60.0


_DEFAULT_LIMITS = TiffMetadataLimits()


def layout_from_tables(
    offsets: Sequence[int],
    counts: Sequence[int],
    rows_per_strip: int,
    *,
    samples: int,
    endian: str = "<",
    bytes_per_sample: int = 4,
) -> TiffLayout:
    """Validate a strip table and return its immutable image layout."""
    if rows_per_strip <= 0 or samples <= 0 or bytes_per_sample <= 0:
        _invalid("TIFF dimensions and storage stride must be positive")
    height = _validate_strip_tables(
        offsets,
        counts,
        rows_per_strip=rows_per_strip,
        row_stride=samples * bytes_per_sample,
    )
    return TiffLayout(
        samples,
        height,
        tuple(offsets),
        tuple(counts),
        rows_per_strip,
        endian,
        bytes_per_sample,
    )


async def inspect_tiff(
    read_at: Callable[[int, int], Awaitable[bytes]],
    *,
    size_bytes: int,
    limits: TiffMetadataLimits = _DEFAULT_LIMITS,
    samples: int | None = None,
) -> TiffLayout:
    """Inspect the first qualified strip image with bounded absolute reads.

    Parameters
    ----------
    read_at : callable
        Provider-configured bounded metadata reader, accepting offset/length.
    size_bytes : int
        Object size used to reject invalid offsets before making requests.
    limits : TiffMetadataLimits
        Request, allocation and elapsed-time limits.
    samples : int, optional
        Expected width for qualified legacy strip files missing ImageWidth.

    Returns
    -------
    TiffLayout
        Validated uncompressed complex signed-int16 storage layout.

    Raises
    ------
    ValueError
        If metadata exceeds a limit or the layout/encoding is unsupported.

    """
    requests = total = elements = 0
    started = time.monotonic()

    async def read(offset: int, length: int) -> bytes:
        nonlocal requests, total
        if (
            offset < 0
            or length <= 0
            or offset + length > size_bytes
            or requests >= limits.max_requests
            or total + length > limits.max_bytes
            or time.monotonic() - started >= limits.max_seconds
        ):
            _invalid("TIFF metadata read exceeds object or inspection limits")
        requests += 1
        total += length
        remaining = limits.max_seconds - (time.monotonic() - started)
        import asyncio

        try:
            payload = await asyncio.wait_for(read_at(offset, length), remaining)
        except TimeoutError:
            _invalid("TIFF metadata inspection timed out")
        if len(payload) != length:
            _invalid("TIFF metadata read returned a truncated region")
        return payload

    header = await read(0, min(size_bytes, HEADER_PROBE_BYTES))
    offset, endian, big = _parse_tiff_header(header)
    count_size, entry_size, inline_size = (8, 20, 8) if big else (2, 12, 4)
    count_body = await read(offset, count_size)
    count = int.from_bytes(count_body, "little" if endian == "<" else "big")
    if not 0 < count <= min(limits.max_entries, _MAX_IFD_ENTRIES):
        _invalid("TIFF directory entry count exceeds inspection limits")
    directory = count_body + await read(offset + count_size, count * entry_size)
    wanted = {256, 257, 258, 259, 273, 277, 278, 279, 284, 317, 339} | _TILE_TAGS
    # Fetch only required tag tables, validating element counts before reads.
    tags = {}
    for index in range(count):
        start = count_size + index * entry_size
        tag, typ, n = struct.unpack_from(
            f"{endian}HH{'Q' if big else 'I'}", directory, start
        )
        if tag not in wanted:
            continue
        unit = _TIFF_TYPE_SIZE.get(typ)
        elements += n
        if unit is None or n <= 0 or elements > limits.max_elements:
            _invalid("TIFF referenced table exceeds inspection limits")
        length = unit * n
        value_offset = start + (12 if big else 8)
        inline = directory[value_offset : value_offset + inline_size]
        if length <= inline_size:
            payload = inline[:length]
        else:
            pointer = int.from_bytes(inline, "little" if endian == "<" else "big")
            payload = await read(pointer, length)
        tags[tag] = (typ, payload)
    if any(tag in tags for tag in _TILE_TAGS):
        _invalid("tiled TIFF is not supported for range decoding")

    def first(tag: int, default: int | None = None) -> int | None:
        return _first_int_tag(tags, tag, endian=endian, default=default)

    if first(259, 1) != 1 or first(317, 1) != 1 or first(284, 1) != 1:
        _invalid("compressed, predicted or planar TIFF is unsupported")
    width = first(256, samples)
    if width is None or (samples is not None and width != samples):
        _invalid("TIFF image width disagrees with expected sample geometry")
    bits = _int_tag_values(tags, 258, endian=endian) or [32]
    formats = _int_tag_values(tags, 339, endian=endian) or [5]
    if not (
        (bits == [32] and formats == [5] and first(277, 1) == 1)
        or (
            bits in ([16], [16, 16])
            and all(v == 2 for v in formats)
            and first(277, 2) == 2
        )
    ):
        _invalid("TIFF sample encoding is not complex signed int16")
    layout = layout_from_tables(
        _int_tag_values(tags, 273, endian=endian),
        _int_tag_values(tags, 279, endian=endian),
        first(278, 0),
        samples=width,
        endian=endian,
    )
    if first(257, layout.height) != layout.height:
        _invalid("TIFF height disagrees with strip table rows")
    if any(
        o + n > size_bytes
        for o, n in zip(layout.strip_offsets, layout.strip_byte_counts, strict=True)
    ):
        _invalid("TIFF strip extent exceeds object size")
    return layout


def plan_tiff_windows(
    layout: TiffLayout,
    windows: Sequence[PixelWindow],
    *,
    max_extent_bytes: int | None = None,
    exact_columns: bool = False,
) -> TiffReadPlan:
    """Map physical windows to extents, preserving strip gaps and row boundaries.

    The caller owns scientific unions and whole-row expansion. Extent limits
    only split at row boundaries and never manufacture samples in strip gaps.
    """
    regions: list[TiffRegion] = []
    stride = layout.width * layout.bytes_per_sample
    for window in windows:
        if not (
            0 <= window.row0 < window.row1 <= layout.height
            and 0 <= window.col0 < window.col1 <= layout.width
        ):
            _invalid("TIFF pixel window falls outside the image")
        for row in range(window.row0, window.row1):
            offset = (
                layout.strip_offsets[row // layout.rows_per_strip]
                + row % layout.rows_per_strip * stride
                + window.col0 * layout.bytes_per_sample
            )
            length = (window.col1 - window.col0) * layout.bytes_per_sample
            current = TiffRegion(
                ByteExtent(offset, length),
                PixelWindow(row, row + 1, window.col0, window.col1),
            )
            if regions and not exact_columns:
                previous = regions[-1]
                w = previous.window
                if (
                    w.row1 == row
                    and w.col0 == window.col0
                    and w.col1 == window.col1
                    and previous.extent.offset + previous.extent.length == offset
                    and (
                        max_extent_bytes is None
                        or previous.extent.length + length <= max_extent_bytes
                    )
                ):
                    regions[-1] = TiffRegion(
                        ByteExtent(
                            previous.extent.offset, previous.extent.length + length
                        ),
                        PixelWindow(w.row0, row + 1, w.col0, w.col1),
                    )
                    continue
            regions.append(current)
    return TiffReadPlan(layout, tuple(regions))


def decode_tiff_blocks(
    plan: TiffReadPlan,
    read_at: Callable[[int, int], bytes],
    *,
    max_working_bytes: int,
    window: PixelWindow | None = None,
) -> Iterator[RasterBlock]:
    """Decode verified local extents without retaining a full encoded span.

    Parameters
    ----------
    plan : TiffReadPlan
        Explicit source mappings and sample encoding.
    read_at : callable
        Verified local absolute-offset reader; no network access is performed.
    max_working_bytes : int
        Budget for encoded reads, conversion, and current/previous yielded block.
        The writer's own fixed buffers must be reserved by the caller first.
    window : PixelWindow, optional
        Crop inside the physical plan, without changing encoded dependencies.

    Yields
    ------
    RasterBlock
        Complex64 blocks of at most 512 rows and columns.

    """
    if plan.layout.bytes_per_sample != 4 or max_working_bytes < 32:
        _invalid("TIFF decoder requires complex-int16 and at least 32 working bytes")
    max_pixels = max_working_bytes // 32
    for region in plan.regions:
        w = region.window
        crop = (
            w
            if window is None
            else PixelWindow(
                max(w.row0, window.row0),
                min(w.row1, window.row1),
                max(w.col0, window.col0),
                min(w.col1, window.col1),
            )
        )
        width = min(512, crop.col1 - crop.col0, max_pixels)
        if width <= 0 or crop.row1 <= crop.row0:
            continue
        height = min(512, max(1, max_pixels // width))
        for row in range(crop.row0, crop.row1, height):
            for col in range(crop.col0, crop.col1, width):
                nr, nc = min(height, crop.row1 - row), min(width, crop.col1 - col)
                output = np.empty((nr, nc), dtype=np.complex64)
                for local in range(nr):
                    offset = (
                        region.extent.offset
                        + ((row + local - w.row0) * (w.col1 - w.col0) + col - w.col0)
                        * 4
                    )
                    raw = read_at(offset, nc * 4)
                    if len(raw) != nc * 4:
                        _invalid("local TIFF extent read is truncated")
                    pairs = np.frombuffer(raw, dtype=f"{plan.layout.endian}i2").reshape(
                        nc, 2
                    )
                    output.real[local] = pairs[:, 0]
                    output.imag[local] = pairs[:, 1]
                yield RasterBlock(PixelWindow(row, row + nr, col, col + nc), output)


def decode_complex_int16(
    raw: bytes,
    *,
    rows: int,
    samples: int,
    col0: int = 0,
    n_cols: int | None = None,
    endian: str = "<",
) -> np.ndarray:
    """Decode a qualified complex-int16 buffer for compatibility callers."""
    if rows <= 0 or samples <= 0 or len(raw) != rows * samples * 4:
        _invalid("row span does not contain the expected whole-row bytes")
    n_cols = samples - col0 if n_cols is None else n_cols
    if not 0 <= col0 < col0 + n_cols <= samples:
        _invalid("complex sample crop is outside the row")
    pairs = np.frombuffer(raw, dtype=f"{endian}i2").reshape(rows, samples, 2)[
        :, col0 : col0 + n_cols
    ]
    output = np.empty((rows, n_cols), dtype=np.complex64)
    output.real = pairs[..., 0]
    output.imag = pairs[..., 1]
    return output


def _invalid(message: str) -> NoReturn:
    return reject(message, ValueError, logger=logger)


_TIFF_TYPE_SIZE = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    7: 1,  # UNDEFINED
    12: 8,  # DOUBLE
    16: 8,  # LONG8
    17: 8,  # SLONG8
    18: 8,  # IFD8
}
_TIFF_INT_FORMAT = {1: "B", 3: "H", 4: "I", 16: "Q"}
_TAG_COMPRESSION = 259
_TAG_STRIP_OFFSETS = 273
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTE_COUNTS = 279
_TAG_TILE_WIDTH = 322
_TAG_TILE_LENGTH = 323
_TAG_TILE_OFFSETS = 324
_TAG_TILE_BYTE_COUNTS = 325
_TILE_TAGS = frozenset(
    {
        _TAG_TILE_WIDTH,
        _TAG_TILE_LENGTH,
        _TAG_TILE_OFFSETS,
        _TAG_TILE_BYTE_COUNTS,
    }
)
_MAX_IFD_ENTRIES = 512


def _probe_body(bodies: Mapping[str, bytes], label: str) -> bytes:
    """Return one named probe body or fail closed."""
    payload = bodies.get(label)
    if payload is None:
        _invalid(f"IFD probe body {label!r} is missing")
    return payload


def _ifd_bytes_from_regions(
    regions: Sequence[tuple[int, bytes]],
    ifd_offset: int,
) -> bytes:
    """Return the IFD starting at the header pointer inside a covering probe."""
    for start, payload in regions:
        relative = ifd_offset - start
        if relative >= 0 and relative < len(payload):
            return payload[relative:]
    return _invalid(f"IFD at file offset {ifd_offset} is not covered by any probe body")


def _parse_tiff_header(header: bytes) -> tuple[int, str, bool]:
    """Return ``(ifd_offset, endian, bigtiff)`` from a TIFF header probe."""
    if len(header) < 8:
        _invalid("TIFF header probe is shorter than 8 bytes")
    endian_code = header[:2]
    if endian_code == b"II":
        endian = "<"
    elif endian_code == b"MM":
        endian = ">"
    else:
        _invalid(f"unrecognized TIFF byte order {endian_code!r}")
    magic = struct.unpack_from(f"{endian}H", header, 2)[0]
    if magic == 42:
        ifd_offset = int(struct.unpack_from(f"{endian}I", header, 4)[0])
        return ifd_offset, endian, False
    if magic == 43:
        if len(header) < HEADER_PROBE_BYTES:
            _invalid("BigTIFF header probe is shorter than 16 bytes")
        offset_size, reserved = struct.unpack_from(f"{endian}HH", header, 4)
        if offset_size != 8 or reserved != 0:
            _invalid("BigTIFF header has an unexpected offset size")
        ifd_offset = int(struct.unpack_from(f"{endian}Q", header, 8)[0])
        return ifd_offset, endian, True
    return _invalid(f"unrecognized TIFF magic number {magic}")


def _probe_regions(bodies: Mapping[str, bytes]) -> list[tuple[int, bytes]]:
    """Map probe labels to ``(file_offset, payload)`` regions."""
    regions: list[tuple[int, bytes]] = []
    for label, payload in bodies.items():
        if label == "tiff-header":
            start = 0
        elif label.startswith("ifd-"):
            start = int(label[4:])
        elif label.startswith("strip-tables-"):
            start = int(label[len("strip-tables-") :])
        else:
            continue
        regions.append((start, payload))
    return regions


def _bytes_at(
    regions: Sequence[tuple[int, bytes]], offset: int, nbytes: int
) -> bytes | None:
    """Return ``nbytes`` at a file offset, or None if no probe covers it."""
    for start, payload in regions:
        relative = offset - start
        if relative >= 0 and relative + nbytes <= len(payload):
            return payload[relative : relative + nbytes]
    return None


def _parse_ifd_tags(
    ifd_body: bytes,
    *,
    endian: str,
    bigtiff: bool,
    regions: Sequence[tuple[int, bytes]],
    ifd_file_offset: int,
) -> dict[int, tuple[int, bytes]]:
    """Parse classic or BigTIFF IFD entries from a probe body."""
    count_fmt = "Q" if bigtiff else "H"
    count_size = 8 if bigtiff else 2
    entry_size = 20 if bigtiff else 12
    offset_size = 8 if bigtiff else 4
    if len(ifd_body) < count_size:
        _invalid("IFD probe is shorter than the directory count")
    n_entries = int(struct.unpack_from(f"{endian}{count_fmt}", ifd_body, 0)[0])
    if n_entries <= 0 or n_entries > _MAX_IFD_ENTRIES:
        _invalid(f"IFD entry count {n_entries} is not usable")
    needed = count_size + n_entries * entry_size
    if len(ifd_body) < needed:
        _invalid(
            f"IFD probe is {len(ifd_body)} bytes, need {needed} for {n_entries} entries"
        )
    tags: dict[int, tuple[int, bytes]] = {}
    for index in range(n_entries):
        start = count_size + index * entry_size
        if bigtiff:
            tag, typ, count = struct.unpack_from(f"{endian}HHQ", ifd_body, start)
            value_offset = start + 12
        else:
            tag, typ, count = struct.unpack_from(f"{endian}HHI", ifd_body, start)
            value_offset = start + 8
        payload = _ifd_value_bytes(
            ifd_body,
            typ=int(typ),
            count=int(count),
            value_offset=value_offset,
            endian=endian,
            offset_size=offset_size,
            regions=regions,
            ifd_file_offset=ifd_file_offset,
        )
        tags[int(tag)] = (int(typ), payload)
    return tags


def _ifd_value_bytes(
    ifd_body: bytes,
    *,
    typ: int,
    count: int,
    value_offset: int,
    endian: str,
    offset_size: int,
    regions: Sequence[tuple[int, bytes]],
    ifd_file_offset: int,
) -> bytes:
    """Return the value bytes of one IFD entry, following file offsets."""
    unit = _TIFF_TYPE_SIZE.get(typ)
    if unit is None:
        _invalid(f"unsupported TIFF field type {typ}")
    nbytes = unit * count
    inline = ifd_body[value_offset : value_offset + offset_size]
    if nbytes <= offset_size:
        return inline[:nbytes]
    offset_fmt = "Q" if offset_size == 8 else "I"
    if len(inline) < offset_size:
        _invalid("IFD value offset is truncated")
    pointer = int(struct.unpack_from(f"{endian}{offset_fmt}", inline, 0)[0])
    payload = _bytes_at(regions, pointer, nbytes)
    if payload is not None:
        return payload
    relative = pointer - ifd_file_offset
    if relative >= 0 and relative + nbytes <= len(ifd_body):
        return ifd_body[relative : relative + nbytes]
    return _invalid(
        "IFD extra values at file offset "
        f"{pointer} ({nbytes} bytes) are not in any probe"
    )


def _int_tag_values(
    tags: Mapping[int, tuple[int, bytes]],
    tag: int,
    *,
    endian: str,
) -> list[int]:
    """Decode an integer TIFF tag into a list of ints."""
    entry = tags.get(tag)
    if entry is None:
        return []
    typ, payload = entry
    fmt = _TIFF_INT_FORMAT.get(typ)
    if fmt is None:
        _invalid(f"TIFF tag {tag} has non-integer type {typ}")
    unit = struct.calcsize(fmt)
    if len(payload) % unit != 0:
        _invalid(f"TIFF tag {tag} payload is not a whole number of integers")
    count = len(payload) // unit
    values = struct.unpack(f"{endian}{count}{fmt}", payload)
    return [int(value) for value in values]


def _first_int_tag(
    tags: Mapping[int, tuple[int, bytes]],
    tag: int,
    *,
    endian: str,
    default: int | None = None,
) -> int | None:
    """Return the first integer of a tag, or ``default`` when absent."""
    values = _int_tag_values(tags, tag, endian=endian)
    if not values:
        return default
    return values[0]


def _strip_tables_from_probes(
    bodies: Mapping[str, bytes],
    *,
    samples: int,
) -> tuple[list[int], list[int], int]:
    """Parse strip tables from IFD probe bodies for an uncompressed strip TIFF."""
    del samples
    header = _probe_body(bodies, "tiff-header")
    ifd_offset, endian, bigtiff = _parse_tiff_header(header)
    regions = _probe_regions(bodies)
    ifd_body = _ifd_bytes_from_regions(regions, ifd_offset)
    tags = _parse_ifd_tags(
        ifd_body,
        endian=endian,
        bigtiff=bigtiff,
        regions=regions,
        ifd_file_offset=ifd_offset,
    )
    if any(tag in tags for tag in _TILE_TAGS):
        _invalid(
            "tiled measurement TIFF cannot be planned for range fetch; "
            "row-aligned spans require stripped organization"
        )
    compression = _first_int_tag(tags, _TAG_COMPRESSION, endian=endian, default=1)
    if compression != 1:
        _invalid(
            "compressed measurement TIFF cannot be planned for range fetch; "
            "strip rows have no fixed byte stride"
        )
    rows_per_strip = _first_int_tag(tags, _TAG_ROWS_PER_STRIP, endian=endian)
    if rows_per_strip is None or rows_per_strip <= 0:
        _invalid("measurement TIFF IFD is missing a positive RowsPerStrip")
    offsets = _int_tag_values(tags, _TAG_STRIP_OFFSETS, endian=endian)
    counts = _int_tag_values(tags, _TAG_STRIP_BYTE_COUNTS, endian=endian)
    if not offsets or not counts:
        _invalid("measurement TIFF IFD is missing StripOffsets or StripByteCounts")
    if len(offsets) != len(counts):
        _invalid(
            "StripOffsets length "
            f"{len(offsets)} != StripByteCounts length {len(counts)}"
        )
    return offsets, counts, rows_per_strip


def _validate_strip_tables(
    strip_offsets: Sequence[int],
    strip_byte_counts: Sequence[int],
    *,
    rows_per_strip: int,
    row_stride: int,
) -> int:
    """Validate the strip tables against the row stride and count the rows.

    Parameters
    ----------
    strip_offsets, strip_byte_counts : sequence of int
        TIFF strip tables.
    rows_per_strip, row_stride : int
        Declared rows per strip and derived bytes per row.

    Returns
    -------
    int
        Total number of whole rows the tables describe.

    Raises
    ------
    ValueError
        If the tables are empty, mismatched in length, negatively offset,
        or inconsistent with whole rows of the declared stride.

    """
    strip_count = len(strip_offsets)
    if strip_count == 0:
        _invalid("strip tables are empty; cannot plan range spans")
    if len(strip_byte_counts) != strip_count:
        _invalid(
            f"strip table length mismatch: {strip_count} offsets vs "
            f"{len(strip_byte_counts)} byte counts"
        )
    total_rows = 0
    for index, (offset, count) in enumerate(
        zip(strip_offsets, strip_byte_counts, strict=True)
    ):
        if offset < 0:
            _invalid(f"strip {index} has a negative offset ({offset})")
        if count <= 0:
            _invalid(f"strip {index} has a non-positive byte count ({count})")
        if count % row_stride != 0:
            _invalid(
                f"strip {index} byte count {count} is not a multiple of the "
                f"row stride {row_stride}"
            )
        rows = count // row_stride
        if index < strip_count - 1 and rows != rows_per_strip:
            _invalid(
                f"strip {index} holds {rows} rows, expected {rows_per_strip} "
                "(only the last strip may be partial)"
            )
        if index == strip_count - 1 and rows > rows_per_strip:
            _invalid(
                f"final strip holds {rows} rows, more than RowsPerStrip "
                f"{rows_per_strip}"
            )
        total_rows += rows
    return total_rows

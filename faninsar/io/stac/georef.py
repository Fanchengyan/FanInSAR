"""Pure georeferenced Zarr view descriptors for committed STAC products (P0089).

The canonical scientific stores stay unchanged. This module describes a
derived, read-only Zarr v3 group that GDAL can open with the product CRS,
affine, and dtype: ``values`` reuses the original data chunks, ``x`` and ``y``
are small generated pixel-centre coordinate arrays, and the root group carries
inline consolidated metadata for Zarr-Python.

Building a descriptor is metadata-only. The caller supplies the already
validated source ``zarr.json`` mapping and is responsible for bounded reads,
HTTP Range handling, and serving the chunks named by the descriptor.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NoReturn

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from pyproj import CRS

    from faninsar.io.stac.record import StacRecord

__all__ = [
    "GeoreferencedZarrError",
    "GeoreferencedZarrView",
    "build_georeferenced_zarr_view",
]

logger = setup_logger(__name__)

#: Largest number of generated coordinate samples in one chunk.
_MAX_COORDINATE_SAMPLES: Final = 1024

#: Zarr v3 data types whose GDAL read-back dtype is the stored dtype.
_SUPPORTED_DATA_TYPES: Final[frozenset[str]] = frozenset(
    {"float16", "float32", "float64", "complex64", "complex128"}
)

#: Companion suffixes written beside a scene unit SLC asset.
_COMPANION_SUFFIXES: Final = (".valid", ".phase")


class GeoreferencedZarrError(ValueError):
    """Report why one source asset cannot become a georeferenced Zarr view.

    Parameters
    ----------
    reason : str
        Stable machine-readable rejection reason, such as
        ``"unsupported-domain"`` or ``"grid-shape-mismatch"``.
    detail : str
        Human-readable diagnostic naming the offending metadata value.

    Notes
    -----
    The exception is a :class:`ValueError` so the service can treat any
    rejected view as a normal unsupported asset and omit it from the Item.

    """

    def __init__(self, reason: str, detail: str) -> None:
        """Record the rejection reason and its diagnostic."""
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class _Unit:
    """One scene unit's placement inside the parent scene grid."""

    rows: int
    cols: int
    row_origin: int
    col_origin: int


@dataclass(frozen=True, slots=True)
class _SourceArray:
    """Validated metadata of one native Zarr v3 source array."""

    shape: tuple[int, int]
    data_type: str
    chunk_shape: tuple[int, int]
    chunk_key_encoding: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GeoreferencedZarrView:
    """Immutable descriptor of one derived georeferenced Zarr group.

    Parameters
    ----------
    source_asset_key : str
        Native STAC asset key whose chunks this view reuses.
    asset_key : str
        Derived asset key, always ``f"{source_asset_key}_georef"``.
    shape : tuple of int
        ``(rows, cols)`` of the source array.
    transform : tuple of float
        Six-value unrotated affine ``(a, b, c, d, e, f)`` in the product CRS.
    crs : str
        CRS text from the validated Item or per-asset projection metadata.
    data_type : str
        Native Zarr v3 data type; never converted.
    _values_metadata : mapping
        ``values/zarr.json`` response payload.
    _x_metadata : mapping
        ``x/zarr.json`` response payload.
    _y_metadata : mapping
        ``y/zarr.json`` response payload.
    _root_metadata : mapping
        Root ``zarr.json`` response payload with inline consolidated metadata.
    _chunk_key_encoding : mapping
        Native source chunk-key encoding.
    _source_chunk_shape : tuple of int
        Native source chunk shape used to validate source chunk indices.

    """

    source_asset_key: str
    asset_key: str
    shape: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]
    crs: str
    data_type: str
    _values_metadata: Mapping[str, Any] = field(repr=False, compare=False)
    _x_metadata: Mapping[str, Any] = field(repr=False, compare=False)
    _y_metadata: Mapping[str, Any] = field(repr=False, compare=False)
    _root_metadata: Mapping[str, Any] = field(repr=False, compare=False)
    _chunk_key_encoding: Mapping[str, Any] = field(repr=False, compare=False)
    _source_chunk_shape: tuple[int, int] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Freeze every internal metadata mapping and sequence."""
        object.__setattr__(
            self, "_values_metadata", _freeze_json(self._values_metadata)
        )
        object.__setattr__(self, "_x_metadata", _freeze_json(self._x_metadata))
        object.__setattr__(self, "_y_metadata", _freeze_json(self._y_metadata))
        object.__setattr__(self, "_root_metadata", _freeze_json(self._root_metadata))
        object.__setattr__(
            self, "_chunk_key_encoding", _freeze_json(self._chunk_key_encoding)
        )

    def metadata(self, relative: str) -> dict[str, Any] | None:
        """Return one generated Zarr metadata document, or ``None``.

        Parameters
        ----------
        relative : str
            Relative child path. Exactly ``"zarr.json"``,
            ``"values/zarr.json"``, ``"x/zarr.json"``, and
            ``"y/zarr.json"`` are generated.

        Returns
        -------
        dict or None
            A detached strict-JSON-compatible copy, or ``None`` when the
            path is not a known generated metadata document.

        """
        path = _clean_relative(relative)
        if path == "zarr.json":
            return _thaw_json(self._root_metadata)
        if path == "values/zarr.json":
            return _thaw_json(self._values_metadata)
        if path == "x/zarr.json":
            return _thaw_json(self._x_metadata)
        if path == "y/zarr.json":
            return _thaw_json(self._y_metadata)
        return None

    def coordinate_chunk(self, relative: str) -> bytes | None:
        """Return one generated little-endian float64 coordinate chunk.

        The coordinate arrays use pixel centres and at most 1024 samples per
        chunk. The final chunk is zero-padded to the declared chunk shape, as
        required by the Zarr codec contract.

        Parameters
        ----------
        relative : str
            Exactly ``"x/c/<index>"`` or ``"y/c/<index>"``.

        Returns
        -------
        bytes or None
            ``numpy.float64`` little-endian bytes, or ``None`` when the path
            is unknown, unsafe, or outside the coordinate grid.

        """
        path = _clean_relative(relative)
        if path is None:
            return None
        parts = path.split("/")
        if len(parts) != 3 or parts[1] != "c" or parts[0] not in {"x", "y"}:
            return None
        index = _canonical_index(parts[2])
        if index is None:
            return None
        axis = parts[0]
        length = self.shape[1] if axis == "x" else self.shape[0]
        chunk_shape = min(_MAX_COORDINATE_SAMPLES, length)
        chunk_count = (length + chunk_shape - 1) // chunk_shape
        if index >= chunk_count:
            return None
        start = index * chunk_shape
        valid = min(chunk_shape, length - start)
        values = np.zeros(chunk_shape, dtype=np.float64)
        positions = start + np.arange(valid, dtype=np.float64)
        if axis == "x":
            values[:valid] = self.transform[2] + self.transform[0] * (positions + 0.5)
        else:
            values[:valid] = self.transform[5] + self.transform[4] * (positions + 0.5)
        return np.asarray(values, dtype="<f8").tobytes()

    def source_chunk(self, relative: str) -> str | None:
        """Map one ``values`` chunk request to the native source chunk path.

        The returned path is relative to the native asset root and preserves
        the original chunk-key encoding. It is never used to open a file here;
        the service resolves it through its pinned safe asset reader so native
        endianness and compression bytes are served unmodified. A missing
        native chunk stays unmapped so the service can answer ``404`` and Zarr
        can apply the declared fill value.

        Parameters
        ----------
        relative : str
            Exactly ``"values/<native-chunk-key>"``.

        Returns
        -------
        str or None
            Native relative chunk key, or ``None`` when the path is unknown,
            unsafe, uses another encoding, or lies outside the chunk grid.

        """
        path = _clean_relative(relative)
        if path is None or not path.startswith("values/"):
            return None
        tail = path.removeprefix("values/")
        if not tail or tail == "zarr.json":
            return None
        coordinates = _native_chunk_coordinates(tail, self._chunk_key_encoding)
        if coordinates is None:
            return None
        row, column, native = coordinates
        chunk_rows, chunk_cols = self._source_chunk_shape
        if row >= (self.shape[0] + chunk_rows - 1) // chunk_rows:
            return None
        if column >= (self.shape[1] + chunk_cols - 1) // chunk_cols:
            return None
        return native


def build_georeferenced_zarr_view(
    record: StacRecord,
    asset_key: str,
    metadata: Mapping[str, Any],
) -> GeoreferencedZarrView:
    """Describe one georeferenced Zarr view for a committed asset.

    Parameters
    ----------
    record : StacRecord
        Normalized immutable record of the committed product.
    asset_key : str
        Native asset key to expose. The derived key is
        ``f"{asset_key}_georef"``; collision handling belongs to the service.
    metadata : mapping
        Already bounded, securely read native Zarr v3 array metadata. This
        function performs no filesystem or network access.

    Returns
    -------
    GeoreferencedZarrView
        Frozen descriptor carrying validated CRS, affine, shape, dtype, and
        generated metadata and coordinate chunks.

    Raises
    ------
    GeoreferencedZarrError
        If the asset is missing, the domain is not geographic, the array is
        not a supported rank-2 Zarr v3 array, the affine is rotated or
        degenerate, dimensions are too small, or the grid cannot be resolved
        without ambiguity.

    """
    assets = _record_assets(record)
    asset = assets.get(asset_key)
    if not isinstance(asset, Mapping):
        _reject("unknown-asset", f"record has no asset {asset_key!r}")
    source = _validate_source_metadata(metadata)
    crs_text, crs, transform = _resolve_grid(record, asset_key, asset, source.shape)
    values_metadata = _values_metadata(source.metadata, _crs_payload(crs))
    x_metadata = _coordinate_metadata("x", source.shape[1], crs)
    y_metadata = _coordinate_metadata("y", source.shape[0], crs)
    return GeoreferencedZarrView(
        source_asset_key=asset_key,
        asset_key=f"{asset_key}_georef",
        shape=source.shape,
        transform=transform,
        crs=crs_text,
        data_type=source.data_type,
        _values_metadata=values_metadata,
        _x_metadata=x_metadata,
        _y_metadata=y_metadata,
        _root_metadata=_root_metadata(values_metadata, x_metadata, y_metadata),
        _chunk_key_encoding=source.chunk_key_encoding,
        _source_chunk_shape=source.chunk_shape,
    )


def _record_assets(record: StacRecord) -> Mapping[str, Any]:
    """Return the record's asset mapping or reject an invalid record."""
    item = record.item
    assets = item.get("assets") if isinstance(item, Mapping) else None
    if not isinstance(assets, Mapping):
        _reject("invalid-record", "normalized Item has no assets mapping")
    return assets


def _record_properties(record: StacRecord) -> Mapping[str, Any]:
    """Return the record's Item properties or reject an invalid record."""
    item = record.item
    properties = item.get("properties") if isinstance(item, Mapping) else None
    if not isinstance(properties, Mapping):
        _reject("invalid-record", "normalized Item has no properties mapping")
    return properties


def _validate_source_metadata(metadata: Mapping[str, Any]) -> _SourceArray:
    """Validate the bounded native Zarr v3 array metadata."""
    if not isinstance(metadata, Mapping):
        _reject("invalid-metadata", "source array metadata must be a mapping")
    if metadata.get("zarr_format") != 3:
        _reject("unsupported-format", "only native Zarr v3 arrays are supported")
    if metadata.get("node_type") != "array":
        _reject("unsupported-node-type", "source metadata must describe an array")
    raw_shape = metadata.get("shape")
    if not isinstance(raw_shape, list | tuple) or len(raw_shape) != 2:
        _reject(
            "unsupported-rank",
            f"source array must be rank 2, got shape {raw_shape!r}",
        )
    shape = _positive_int_tuple(raw_shape, "shape", 2)
    if min(shape) < 2:
        _reject(
            "grid-too-small",
            f"shape {shape} has a length-1 dimension; GDAL needs at least two "
            "samples on each coordinate axis to derive a transform",
        )
    data_type = metadata.get("data_type")
    if not isinstance(data_type, str) or data_type not in _SUPPORTED_DATA_TYPES:
        _reject(
            "unsupported-data-type",
            f"data_type {data_type!r} is not a driver-proven float or complex type",
        )
    chunk_grid = metadata.get("chunk_grid")
    if not isinstance(chunk_grid, Mapping) or chunk_grid.get("name") != "regular":
        _reject("unsupported-chunk-grid", "only a regular chunk grid is supported")
    configuration = chunk_grid.get("configuration")
    if not isinstance(configuration, Mapping):
        _reject("unsupported-chunk-grid", "chunk grid has no configuration mapping")
    chunk_shape = _positive_int_tuple(
        configuration.get("chunk_shape"), "chunk_shape", 2
    )
    encoding = metadata.get("chunk_key_encoding")
    if not isinstance(encoding, Mapping):
        _reject("unsupported-chunk-key-encoding", "chunk_key_encoding is missing")
    _encoding_kind(encoding)
    codecs = metadata.get("codecs")
    if not isinstance(codecs, list | tuple) or not codecs:
        _reject("invalid-metadata", "codecs must be a non-empty sequence")
    if any(not isinstance(codec, Mapping) for codec in codecs):
        _reject("invalid-metadata", "each codec must be a mapping")
    if "fill_value" not in metadata:
        _reject("invalid-metadata", "fill_value is missing")
    attributes = metadata.get("attributes", {})
    if not isinstance(attributes, Mapping):
        _reject("invalid-metadata", "attributes must be a mapping")
    transformers = metadata.get("storage_transformers")
    if transformers is not None and not (
        isinstance(transformers, list | tuple) and not transformers
    ):
        _reject(
            "unsupported-storage-transformers",
            "storage transformers would change the native chunk bytes",
        )
    return _SourceArray(
        shape=shape,
        data_type=data_type,
        chunk_shape=chunk_shape,
        chunk_key_encoding=encoding,
        metadata=_plain_copy(metadata),
    )


def _resolve_grid(
    record: StacRecord,
    asset_key: str,
    asset: Mapping[str, Any],
    shape: tuple[int, int],
) -> tuple[str, CRS, tuple[float, float, float, float, float, float]]:
    """Resolve the asset-local CRS and affine under the bounded grid rules."""
    properties = _record_properties(record)
    kind = str(record.kind)
    _require_geo_domain(properties, kind)
    product_crs_text = properties.get("proj:code")
    if not isinstance(product_crs_text, str) or not product_crs_text:
        _reject("missing-projection", "Item has no proj:code")
    _, product_crs = _parse_crs(product_crs_text)
    product_transform_value = properties.get("proj:transform")
    if product_transform_value is None:
        _reject("missing-projection", "Item has no proj:transform")
    product_transform = _parse_transform(product_transform_value, "proj:transform")
    product_shape = _product_shape(properties)
    explicit = _explicit_projection(asset)
    if explicit is not None:
        explicit_text, explicit_crs, explicit_transform, explicit_shape = explicit
        _validate_explicit_projection(
            record,
            kind,
            asset_key,
            asset,
            shape,
            explicit_crs,
            explicit_transform,
            explicit_shape,
            product_crs,
            product_transform,
            product_shape,
        )
        return explicit_text, explicit_crs, explicit_transform
    if kind == "scene":
        unit = _scene_unit_for_asset(record, asset_key, asset)
        if unit is None:
            if shape != product_shape:
                _reject(
                    "grid-shape-mismatch",
                    f"asset shape {shape} differs from product shape {product_shape}",
                )
            return product_crs_text, product_crs, product_transform
        if shape != (unit.rows, unit.cols):
            _reject(
                "unit-shape-mismatch",
                f"asset shape {shape} does not match scene unit shape "
                f"{(unit.rows, unit.cols)}",
            )
        return (
            product_crs_text,
            product_crs,
            _shift_transform(product_transform, unit.row_origin, unit.col_origin),
        )
    if shape != product_shape:
        _reject(
            "grid-shape-mismatch",
            f"asset shape {shape} does not match product shape {product_shape}",
        )
    return product_crs_text, product_crs, product_transform


def _require_geo_domain(properties: Mapping[str, Any], kind: str) -> None:
    """Require a geographic product domain and reject radar or unknown ones."""
    raw: object = None
    if kind == "mask" and "faninsar:mask_domain" in properties:
        raw = properties.get("faninsar:mask_domain")
    elif "faninsar:domain" in properties:
        raw = properties.get("faninsar:domain")
    elif kind == "geocoded":
        return
    if raw == "geo":
        return
    if raw is None:
        _reject("unsupported-domain", "geo products require faninsar:domain='geo'")
    _reject("unsupported-domain", f"domain {raw!r} has no geographic grid")


def _product_shape(properties: Mapping[str, Any]) -> tuple[int, int]:
    """Return the product grid shape from projection or native properties."""
    value = properties.get("proj:shape")
    if value is None:
        value = properties.get("faninsar:shape")
    if value is None:
        _reject("missing-projection", "geo product has no proj:shape")
    return _positive_int_tuple(value, "proj:shape", 2)


def _explicit_projection(
    asset: Mapping[str, Any],
) -> (
    tuple[str, CRS, tuple[float, float, float, float, float, float], tuple[int, int]]
    | None
):
    """Return complete explicit asset projection metadata, else ``None``."""
    keys = ("proj:code", "proj:transform", "proj:shape")
    present = [key in asset for key in keys]
    if not any(present):
        return None
    if not all(present):
        return None
    crs_text, crs = _parse_crs(asset["proj:code"])
    transform = _parse_transform(asset["proj:transform"], "asset proj:transform")
    shape = _positive_int_tuple(asset["proj:shape"], "asset proj:shape", 2)
    return crs_text, crs, transform, shape


def _validate_explicit_projection(
    record: StacRecord,
    kind: str,
    asset_key: str,
    asset: Mapping[str, Any],
    metadata_shape: tuple[int, int],
    explicit_crs: CRS,
    explicit_transform: tuple[float, float, float, float, float, float],
    explicit_shape: tuple[int, int],
    product_crs: CRS,
    product_transform: tuple[float, float, float, float, float, float],
    product_shape: tuple[int, int],
) -> None:
    """Validate complete explicit projection metadata before preferring it."""
    if explicit_shape != metadata_shape:
        _reject(
            "asset-projection-mismatch",
            f"asset {asset_key!r} proj:shape {explicit_shape} does not match "
            f"metadata shape {metadata_shape}",
        )
    if not explicit_crs.equals(product_crs):
        _reject(
            "asset-projection-mismatch",
            f"asset {asset_key!r} CRS differs from the product CRS",
        )
    if kind != "scene":
        if explicit_shape != product_shape:
            _reject(
                "asset-projection-mismatch",
                f"asset {asset_key!r} shape {explicit_shape} does not match "
                f"product shape {product_shape}",
            )
        if not _transforms_equal(explicit_transform, product_transform):
            _reject(
                "asset-projection-mismatch",
                f"asset {asset_key!r} transform differs from the product transform",
            )
        return
    unit = _direct_unit(asset)
    if unit is None and asset_key.endswith(_COMPANION_SUFFIXES):
        unit = _parent_unit(record, asset_key, must=False)
    if unit is not None:
        expected = _shift_transform(product_transform, unit.row_origin, unit.col_origin)
        if not _transforms_equal(explicit_transform, expected):
            _reject(
                "asset-projection-mismatch",
                f"asset {asset_key!r} transform does not match its scene unit origin",
            )
        return
    if explicit_shape == product_shape and not _transforms_equal(
        explicit_transform, product_transform
    ):
        _reject(
            "asset-projection-mismatch",
            f"asset {asset_key!r} transform differs from the product transform",
        )


def _scene_unit_for_asset(
    record: StacRecord, asset_key: str, asset: Mapping[str, Any]
) -> _Unit | None:
    """Return the scene unit placement for an SLC or companion asset."""
    unit = _direct_unit(asset)
    if unit is not None:
        return unit
    if asset_key.endswith(_COMPANION_SUFFIXES):
        return _parent_unit(record, asset_key, must=True)
    return None


def _direct_unit(asset: Mapping[str, Any]) -> _Unit | None:
    """Return a directly declared scene unit, or ``None`` when absent."""
    keys = (
        "faninsar:rows",
        "faninsar:cols",
        "faninsar:row_origin",
        "faninsar:col_origin",
    )
    present = [key in asset for key in keys]
    if not any(present):
        return None
    if not all(present):
        _reject(
            "incomplete-unit-metadata",
            "scene unit assets require rows, cols, row_origin, and col_origin",
        )
    rows = _required_int(asset, "faninsar:rows", minimum=1)
    cols = _required_int(asset, "faninsar:cols", minimum=1)
    row_origin = _required_int(asset, "faninsar:row_origin", minimum=0)
    col_origin = _required_int(asset, "faninsar:col_origin", minimum=0)
    return _Unit(rows, cols, row_origin, col_origin)


def _parent_unit(record: StacRecord, asset_key: str, *, must: bool) -> _Unit | None:
    """Resolve the exact same-unit SLC parent of a companion asset."""
    for suffix in _COMPANION_SUFFIXES:
        if not asset_key.endswith(suffix):
            continue
        parent_key = asset_key[: -len(suffix)]
        parent = _record_assets(record).get(parent_key)
        if isinstance(parent, Mapping):
            unit = _direct_unit(parent)
            if unit is not None:
                return unit
        if must:
            _reject(
                "companion-unresolved",
                f"companion {asset_key!r} has no unambiguous SLC parent {parent_key!r}",
            )
        return None
    if must:
        _reject(
            "companion-unresolved",
            f"asset {asset_key!r} is not a scene unit companion",
        )
    return None


def _required_int(asset: Mapping[str, Any], key: str, *, minimum: int) -> int:
    """Return one integral asset field at or above ``minimum``."""
    value = asset.get(key)
    if isinstance(value, bool):
        _reject("incomplete-unit-metadata", f"{key} must be an integer")
    try:
        number = operator.index(value)
    except TypeError:
        _reject("incomplete-unit-metadata", f"{key} must be an integer")
    if number < minimum:
        _reject("incomplete-unit-metadata", f"{key} must be at least {minimum}")
    return int(number)


def _positive_int_tuple(value: object, field_name: str, length: int) -> tuple[int, ...]:
    """Return a positive integral tuple of exactly ``length`` values."""
    if not isinstance(value, list | tuple) or len(value) != length:
        _reject(
            "invalid-metadata",
            f"{field_name} must be a length-{length} integer sequence",
        )
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            _reject("invalid-metadata", f"{field_name} contains a boolean")
        try:
            number = operator.index(item)
        except TypeError:
            _reject("invalid-metadata", f"{field_name} contains a non-integer")
        if number <= 0:
            _reject("invalid-metadata", f"{field_name} must be positive")
        result.append(int(number))
    return tuple(result)


def _parse_transform(
    value: object, field_name: str
) -> tuple[float, float, float, float, float, float]:
    """Return one validated unrotated finite six-value affine."""
    if not isinstance(value, list | tuple) or len(value) not in {6, 9}:
        _reject(
            "invalid-transform",
            f"{field_name} must contain six or nine numbers",
        )
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            _reject("invalid-transform", f"{field_name} contains a non-number")
        number = float(item)
        if not math.isfinite(number):
            _reject("invalid-transform", f"{field_name} is not finite")
        numbers.append(number)
    if len(numbers) == 9 and not (
        abs(numbers[6]) <= 1e-12
        and abs(numbers[7]) <= 1e-12
        and abs(numbers[8] - 1.0) <= 1e-12
    ):
        _reject("invalid-transform", f"{field_name} bottom row must be [0, 0, 1]")
    a, b, c, d, e, f = numbers[:6]
    if b != 0.0 or d != 0.0:
        _reject("invalid-transform", f"{field_name} must not be rotated or sheared")
    if a == 0.0 or e == 0.0:
        _reject("invalid-transform", f"{field_name} steps must be nonzero")
    return (a, b, c, d, e, f)


def _parse_crs(value: object) -> tuple[str, CRS]:
    """Return validated CRS text and the parsed pyproj CRS."""
    if not isinstance(value, str) or not value:
        _reject("invalid-crs", "CRS must be a non-empty string")
    from pyproj import CRS

    try:
        crs = CRS.from_user_input(value)
    except Exception as error:
        _reject("invalid-crs", f"{value!r} is not recognized: {error}")
    if not (crs.is_geographic or crs.is_projected):
        _reject("invalid-crs", f"{value!r} is neither geographic nor projected")
    return value, crs


def _shift_transform(
    transform: tuple[float, float, float, float, float, float],
    row_origin: int,
    col_origin: int,
) -> tuple[float, float, float, float, float, float]:
    """Shift a parent affine by one scene unit origin exactly once."""
    a, b, c, d, e, f = transform
    return (
        a,
        b,
        c + a * col_origin + b * row_origin,
        d,
        e,
        f + d * col_origin + e * row_origin,
    )


def _transforms_equal(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> bool:
    """Return whether two affines agree within a bounded numeric tolerance."""
    return all(
        math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
        for left, right in zip(first, second, strict=True)
    )


def _crs_payload(crs: CRS) -> dict[str, Any]:
    """Return the GDAL ``_CRS`` attribute payload: an EPSG URL or WKT2."""
    try:
        authority = crs.to_authority()
    except Exception:
        authority = None
    if (
        authority is not None
        and authority[0].upper() == "EPSG"
        and str(authority[1]).isdigit()
    ):
        return {"url": f"http://www.opengis.net/def/crs/EPSG/0/{int(authority[1])}"}
    try:
        return {"wkt": crs.to_wkt(version="WKT2_2019")}
    except Exception:
        return {"wkt": crs.to_wkt()}


def _values_metadata(
    metadata: Mapping[str, Any], crs_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy native array metadata and add the view CRS and dimension names."""
    values = _plain_copy(metadata)
    attributes = values.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    attributes["_CRS"] = _plain_copy(crs_payload)
    values["attributes"] = attributes
    values["dimension_names"] = ["y", "x"]
    return values


def _coordinate_metadata(axis: str, length: int, crs: CRS) -> dict[str, Any]:
    """Return one generated pixel-centre coordinate array metadata document."""
    chunk_shape = min(_MAX_COORDINATE_SAMPLES, length)
    if crs.is_geographic:
        standard_name = "longitude" if axis == "x" else "latitude"
        units = "degrees_east" if axis == "x" else "degrees_north"
    else:
        standard_name = (
            "projection_x_coordinate" if axis == "x" else "projection_y_coordinate"
        )
        units = _projected_units(crs, 0 if axis == "x" else 1)
    return {
        "shape": [length],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [chunk_shape]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {
            "standard_name": standard_name,
            "axis": axis.upper(),
            "units": units,
        },
        "dimension_names": [axis],
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def _projected_units(crs: CRS, index: int) -> str:
    """Return the CRS axis unit in a compact CF-style spelling."""
    axis_info = crs.axis_info
    unit = axis_info[index].unit_name if len(axis_info) > index else ""
    if not unit:
        return "m"
    normalized = unit.strip().lower()
    return {
        "metre": "m",
        "meter": "m",
        "metres": "m",
        "meters": "m",
        "kilometre": "km",
        "kilometer": "km",
        "kilometres": "km",
        "kilometers": "km",
    }.get(normalized, unit)


def _root_metadata(
    values: Mapping[str, Any],
    x: Mapping[str, Any],
    y: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the synthetic root group metadata with inline consolidation."""
    return {
        "attributes": {},
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                "values": _plain_copy(values),
                "x": _plain_copy(x),
                "y": _plain_copy(y),
            },
        },
    }


def _native_chunk_coordinates(
    tail: str, encoding: Mapping[str, Any]
) -> tuple[int, int, str] | None:
    """Return native chunk row/column indices and path for one view tail."""
    name, separator = _encoding_kind(encoding)
    if name == "default" and separator == "/":
        parts = tail.split("/")
        if len(parts) != 3 or parts[0] != "c":
            return None
        row = _canonical_index(parts[1])
        column = _canonical_index(parts[2])
        if row is None or column is None:
            return None
        return row, column, f"c/{row}/{column}"
    parts = tail.split(".")
    if len(parts) != 2:
        return None
    row = _canonical_index(parts[0])
    column = _canonical_index(parts[1])
    if row is None or column is None:
        return None
    return row, column, f"{row}.{column}"


def _encoding_kind(encoding: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(name, separator)`` for a supported native chunk encoding."""
    name = encoding.get("name")
    configuration = encoding.get("configuration", {})
    if not isinstance(configuration, Mapping):
        _reject(
            "unsupported-chunk-key-encoding",
            "chunk_key_encoding configuration must be a mapping",
        )
    if name == "default":
        separator = configuration.get("separator", "/")
        if separator not in {"/", "."}:
            _reject(
                "unsupported-chunk-key-encoding",
                f"unsupported chunk separator {separator!r}",
            )
        return "default", separator
    if name == "v2":
        return "v2", "."
    return _reject(
        "unsupported-chunk-key-encoding",
        f"unsupported chunk key encoding {name!r}",
    )


def _clean_relative(relative: object) -> str | None:
    """Return a safe relative path, or ``None`` for traversal or empties."""
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "\\" in relative
        or "\x00" in relative
    ):
        return None
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return relative


def _canonical_index(value: str) -> int | None:
    """Return a canonical non-negative decimal index, or ``None``."""
    if not value.isdigit() or (len(value) > 1 and value.startswith("0")):
        return None
    return int(value)


def _plain_copy(value: Any) -> Any:
    """Return a detached mutable copy of a JSON-like value."""
    if isinstance(value, Mapping):
        return {key: _plain_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_copy(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    """Return a recursively read-only view of a JSON-like value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a mutable deep copy of a frozen JSON-like value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _reject(reason: str, detail: str) -> NoReturn:
    """Log one rejection with context, then raise the typed error."""
    message = f"{reason}: {detail}"
    logger.error("%s", message)
    raise GeoreferencedZarrError(reason, detail)

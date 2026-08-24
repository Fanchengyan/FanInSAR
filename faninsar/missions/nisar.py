"""Optional NISAR RSLC reader boundary (PROPOSAL-0035)."""

# The adapter preserves actionable domain messages at the mission boundary;
# these exception-style rules are intentionally handled locally.
# ruff: noqa: E501, EM101, EM102, TRY003, TRY004, TRY400

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from faninsar.logging import setup_logger
from faninsar.missions.base import Sensor, register
from faninsar.processing.contracts import (
    ArrayDescriptor,
    ArrayRepresentation,
    CalibrationState,
    CarrierState,
    CoregistrationState,
    OrbitMetadata,
    OrbitStateVector,
    SLCProduct,
)
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.readers import (
    DopplerCentroidPolynomial,
    SLCReadResult,
    ValidSampleMask,
    normalize_selection,
)

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class _AdmissionSnapshot:
    """Immutable identity captured before opening one RSLC source."""

    source_id: str
    source_digest: str
    source_size_bytes: int = 0
    policy: NisarAdmissionPolicy | None = None


@dataclass(frozen=True, slots=True)
class NisarAdmissionPolicy:
    """Fail-closed policy for opening a local NISAR RSLC source.

    The policy is intentionally independent of the optional NISAR reader.  It
    therefore runs before importing or constructing a native HDF5 reader and
    can be used by applications to make source provenance an explicit input.

    Parameters
    ----------
    trusted_roots : sequence of path-like, optional
        Existing local directories under which a source must reside.  Every
        existing component is checked for symbolic links.
    expected_sha256 : mapping, optional
        Source inventory.  Keys may be the source path (or its canonical
        spelling), and values are hexadecimal SHA-256 digests.  A value may
        also be a mapping containing ``sha256``.
    max_size_bytes : int, optional
        Upper bound on the admitted regular file size.
    reject_external_links : bool, default=True
        Inspect HDF5 links without dereferencing them and reject external
        links.  Non-HDF5 fake-reader fixtures are left to the native reader.
    policy_id : str, default="nisar-trusted-preopen-v1"
        Stable policy identity recorded in lineage and scene manifests.

    """

    trusted_roots: tuple[str | Path, ...] = ()
    expected_sha256: Mapping[str, object] = field(default_factory=dict)
    max_size_bytes: int | None = None
    reject_external_links: bool = True
    policy_id: str = "nisar-trusted-preopen-v1"

    def __post_init__(self) -> None:
        """Validate policy scalars before any source is opened."""
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("NISAR admission policy_id must be non-empty")
        if self.max_size_bytes is not None and (
            isinstance(self.max_size_bytes, bool) or self.max_size_bytes < 0
        ):
            raise ValueError("NISAR admission max_size_bytes must be non-negative")

    def as_metadata(self) -> dict[str, object]:
        """Return JSON-compatible policy metadata for provenance manifests."""
        inventory = {
            str(key): _inventory_digest(value)
            for key, value in self.expected_sha256.items()
        }
        inventory_payload = "\n".join(
            f"{key}={inventory[key]}" for key in sorted(inventory)
        ).encode()
        return {
            "policy_id": self.policy_id,
            "trusted_roots": [
                str(Path(root).expanduser()) for root in self.trusted_roots
            ],
            "expected_sha256": inventory,
            "inventory_digest": hashlib.sha256(inventory_payload).hexdigest(),
            "max_size_bytes": self.max_size_bytes,
            "external_links": "rejected" if self.reject_external_links else "checked",
            "regular_file": True,
            "no_follow": True,
        }


_ADMISSION_SNAPSHOTS: dict[int, _AdmissionSnapshot] = {}


def _admission_error(message: str) -> None:
    """Log and raise one NISAR pre-open admission failure."""
    logger.error("NISAR source admission rejected: %s", message)
    raise InvalidProcessingStateError(message)


def _inventory_digest(value: object) -> str:
    """Extract and validate one digest from an inventory value."""
    if isinstance(value, Mapping):
        value = value.get("sha256", value.get("digest", value.get("hash")))
    if not isinstance(value, str):
        _admission_error("expected SHA-256 inventory values must be strings")
    digest = value.strip().lower()
    digest = digest.removeprefix("sha256:")
    if len(digest) != 64:
        _admission_error("expected SHA-256 inventory digest must contain 64 hex digits")
    try:
        int(digest, 16)
    except ValueError:
        _admission_error("expected SHA-256 inventory digest is not hexadecimal")
    return digest


def _normalise_inventory(value: object) -> dict[str, object]:
    """Normalize common path-to-digest inventory spellings."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        nested = value.get("inventory")
        if nested is not None:
            return _normalise_inventory(nested)
        result: dict[str, object] = {}
        for key, item in value.items():
            if str(key).lower() in {"sha256", "digest", "hash"}:
                continue
            result[str(key)] = item
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for item in value:
            if not isinstance(item, Mapping):
                _admission_error("expected SHA-256 inventory entries must be objects")
            source = item.get("source_id", item.get("source", item.get("path")))
            if source is None:
                _admission_error("expected SHA-256 inventory entry has no source_id")
            result[str(source)] = item
        return result
    _admission_error("expected SHA-256 inventory must be a mapping or sequence")
    return {}


def _coerce_admission_policy(
    admission: NisarAdmissionPolicy | Mapping[str, object] | None,
    *,
    trusted_roots: object = None,
    expected_sha256: object = None,
    max_size_bytes: object = None,
    reject_external_links: object = None,
) -> NisarAdmissionPolicy | None:
    """Build one policy from explicit API or configuration metadata."""
    direct = any(
        value is not None
        for value in (
            trusted_roots,
            expected_sha256,
            max_size_bytes,
            reject_external_links,
        )
    )
    if admission is None and not direct:
        return None
    if admission is None:
        values: Mapping[str, object] = {}
    elif isinstance(admission, NisarAdmissionPolicy):
        if direct:
            raise TypeError(
                "NISAR admission policy cannot mix object and scalar options"
            )
        return admission
    elif isinstance(admission, Mapping):
        values = admission
    else:
        raise TypeError("NISAR admission must be a NisarAdmissionPolicy or mapping")
    roots = values.get("trusted_roots", values.get("trusted_root", trusted_roots))
    if roots is None:
        root_values: tuple[str | Path, ...] = ()
    elif isinstance(roots, (str, Path)):
        root_values = (roots,)
    else:
        try:
            root_values = tuple(roots)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError(
                "NISAR trusted_roots must be path-like or a sequence"
            ) from error
    inventory = values.get(
        "expected_sha256",
        values.get(
            "expected_sha256_inventory",
            values.get("sha256_inventory", expected_sha256),
        ),
    )
    maximum = values.get("max_size_bytes", values.get("max_size", max_size_bytes))
    if maximum is not None:
        if isinstance(maximum, bool):
            raise TypeError("NISAR admission max_size_bytes must be an integer")
        try:
            maximum = int(maximum)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "NISAR admission max_size_bytes must be an integer"
            ) from error
    external = values.get(
        "reject_external_links",
        values.get("check_external_links", reject_external_links),
    )
    return NisarAdmissionPolicy(
        trusted_roots=root_values,
        expected_sha256=_normalise_inventory(inventory),
        max_size_bytes=maximum,
        reject_external_links=True if external is None else bool(external),
        policy_id=str(values.get("policy_id", "nisar-trusted-preopen-v1")),
    )


def _reject_symlink_components(path: Path) -> None:
    """Reject symbolic links in an existing local source path."""
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                _admission_error(
                    f"NISAR source path contains a symbolic link: {current}"
                )
        except OSError as error:
            _admission_error(f"NISAR source path cannot be inspected: {error}")


def _local_source_path(uri: str | Path) -> Path:
    """Validate a source URI as a local path without credential parsing."""
    raw = os.fspath(uri)
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if not isinstance(raw, str) or not raw.strip():
        _admission_error("NISAR source URI must be a non-empty local path")
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or raw.startswith("//"):
        _admission_error(
            "NISAR source URI must be a local path; remote or credential URI rejected"
        )
    return Path(raw).expanduser()


def _secure_file_snapshot(
    path: Path,
    policy: NisarAdmissionPolicy,
) -> _AdmissionSnapshot:
    """Inspect, hash, and security-check one source before reader access."""
    _reject_symlink_components(path)
    source = path if path.is_absolute() else Path.cwd() / path
    source_id = str(source.resolve(strict=False))
    if policy.trusted_roots:
        allowed = []
        for root in policy.trusted_roots:
            root_path = Path(root).expanduser()
            _reject_symlink_components(root_path)
            if (
                not root_path.exists()
                or not root_path.is_dir()
                or root_path.is_symlink()
            ):
                _admission_error(
                    f"NISAR trusted root is not a regular directory: {root}"
                )
            allowed.append(root_path.resolve())
        try:
            resolved_source = source.resolve(strict=True)
        except FileNotFoundError:
            _admission_error(
                f"NISAR source does not exist under trusted roots: {source}"
            )
        if not any(resolved_source.is_relative_to(root) for root in allowed):
            _admission_error(f"NISAR source is outside trusted roots: {source}")
    flags = os.O_RDONLY
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags | no_follow)
    except FileNotFoundError as error:
        _admission_error(f"NISAR source does not exist: {source}")
        raise AssertionError from error
    except OSError as error:
        _admission_error(
            f"NISAR source cannot be opened without following links: {source}: {error}"
        )
        raise AssertionError from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _admission_error(f"NISAR source is not a regular file: {source}")
        size = int(info.st_size)
        if policy.max_size_bytes is not None and size > policy.max_size_bytes:
            _admission_error(
                f"NISAR source exceeds max_size_bytes ({size} > {policy.max_size_bytes})"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual_digest = digest.hexdigest()
    inventory = policy.expected_sha256
    expected = None
    for key in (str(path), str(source), source_id):
        if key in inventory:
            expected = _inventory_digest(inventory[key])
            break
    if inventory and expected is None:
        _admission_error(
            f"NISAR source is absent from expected SHA-256 inventory: {source_id}"
        )
    if expected is not None and actual_digest != expected:
        _admission_error(
            f"NISAR source SHA-256 mismatch for {source_id}: expected {expected}, got {actual_digest}"
        )
    _check_hdf5_links(source, policy)
    return _AdmissionSnapshot(
        source_id=source_id,
        source_digest=actual_digest,
        source_size_bytes=size,
        policy=policy,
    )


def _check_hdf5_links(path: Path, policy: NisarAdmissionPolicy) -> None:
    """Inspect HDF5 links with ``getlink=True`` so external targets are not followed."""
    if not policy.reject_external_links:
        return
    try:
        import h5py
    except ImportError:
        logger.warning("h5py unavailable; external-link inspection is not applicable")
        return
    try:
        if not h5py.is_hdf5(path):
            return
        with h5py.File(path, "r") as source:
            external_type = h5py.ExternalLink
            soft_type = h5py.SoftLink
            pending = [source]
            while pending:
                group = pending.pop()
                for name in group:
                    link = group.get(name, getlink=True)
                    if isinstance(link, (external_type, soft_type)):
                        _admission_error(
                            f"NISAR HDF5 source contains an unsafe {type(link).__name__}: {name}"
                        )
                    if isinstance(link, h5py.HardLink):
                        child = group.get(name, getlink=False)
                        if isinstance(child, h5py.Group):
                            pending.append(child)
    except OSError as error:
        # Tiny fake-reader fixtures are often empty/non-HDF5 files.  A real
        # HDF5 source is checked above; malformed files remain the native
        # reader's responsibility for compatibility with the optional reader.
        logger.warning("NISAR source HDF5 link check skipped for %s: %s", path, error)


def _source_snapshot(path: str | Path) -> _AdmissionSnapshot:
    """Return a reproducible path/content identity for one source.

    A missing path is represented by a deterministic sentinel so lightweight
    reader fakes can still exercise the adapter.  A real NISAR reader will
    reject such a path while opening it; if the file subsequently appears,
    the sentinel changes to its content digest and stale reuse is rejected.
    """
    resolved = Path(path).expanduser().resolve(strict=False)
    digest = hashlib.sha256()
    size = 0
    try:
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except (FileNotFoundError, NotADirectoryError):
        digest.update(f"missing:{resolved}".encode())
    except OSError as error:
        logger.error("NISAR RSLC source cannot be admitted: %s", resolved)
        raise InvalidProcessingStateError(
            f"NISAR RSLC source cannot be admitted: {resolved}: {error}"
        ) from error
    return _AdmissionSnapshot(str(resolved), digest.hexdigest(), size)


def _remember_admission(handle: Any, snapshot: _AdmissionSnapshot) -> None:
    """Remember an admission snapshot even for native handles without attrs."""
    _ADMISSION_SNAPSHOTS[id(handle)] = snapshot
    with suppress(AttributeError, TypeError):
        handle._faninsar_nisar_admission = snapshot
    # pybind11 handles can be non-extensible; the id-keyed table covers those
    # objects for the lifetime of the opened handle.


def _handle_admission(handle: Any) -> _AdmissionSnapshot | None:
    """Return a previously captured snapshot for a native reader handle."""
    snapshot = getattr(handle, "_faninsar_nisar_admission", None)
    if isinstance(snapshot, _AdmissionSnapshot):
        return snapshot
    return _ADMISSION_SNAPSHOTS.get(id(handle))


def _validate_admission(
    handle: Any, *, expected_source: str | Path | None = None
) -> _AdmissionSnapshot | None:
    """Reject a changed path or source byte stream before native access."""
    snapshot = _handle_admission(handle)
    if snapshot is None:
        if expected_source is None:
            return None
        snapshot = _source_snapshot(expected_source)
        _remember_admission(handle, snapshot)
    handle_source = _value(handle, "filename", "file_name", "source_path")
    if handle_source is not None:
        current_id = str(Path(handle_source).expanduser().resolve(strict=False))
        if current_id != snapshot.source_id:
            logger.error("NISAR RSLC source path changed: %s", snapshot.source_id)
            raise InvalidProcessingStateError(
                "NISAR RSLC source path changed after admission"
            )
    current = (
        _secure_file_snapshot(Path(snapshot.source_id), snapshot.policy)
        if snapshot.policy is not None
        else _source_snapshot(snapshot.source_id)
    )
    if current.source_digest != snapshot.source_digest:
        logger.error("NISAR RSLC source content changed: %s", snapshot.source_id)
        raise InvalidProcessingStateError(
            "NISAR RSLC source content changed after admission"
        )
    if current.source_size_bytes != snapshot.source_size_bytes:
        logger.error("NISAR RSLC source size changed: %s", snapshot.source_id)
        raise InvalidProcessingStateError(
            "NISAR RSLC source size changed after admission"
        )
    if expected_source is not None:
        expected_id = str(Path(expected_source).expanduser().resolve(strict=False))
        if expected_id != snapshot.source_id:
            logger.error("NISAR RSLC source lineage changed: %s", expected_source)
            raise InvalidProcessingStateError(
                "NISAR RSLC source lineage changed after admission"
            )
    return snapshot


def _value(value: Any, *names: str) -> Any:
    """Return the first attribute or mapping key present on ``value``."""
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        item = getattr(value, name, None)
        if item is not None:
            return item() if callable(item) else item
    return None


def _datetime(value: Any, reference: Any | None = None) -> datetime:
    """Convert NISAR relative or ISO time values to timezone-aware UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, Real) and reference is not None:
        return _datetime(reference) + timedelta(seconds=float(value))
    text = str(value).replace("Z", "+00:00")
    if text.endswith("000") and "." in text:
        text = text[:-3]
    try:
        result = datetime.fromisoformat(text)
    except ValueError as error:
        raise InvalidProcessingStateError(f"invalid NISAR time {value!r}") from error
    return result if result.tzinfo else result.replace(tzinfo=UTC)


def _select_channel(handle: Any, frequency: str, polarization: str) -> tuple[str, str]:
    """Select one available NISAR frequency and polarization."""
    frequencies = _value(handle, "frequencies")
    polarizations = _value(handle, "polarizations")
    if frequencies is None or polarizations is None:
        message = "NISAR RSLC must expose frequency and polarization metadata"
        logger.error(message)
        raise InvalidProcessingStateError(message)
    if isinstance(frequencies, str):
        frequencies = (frequencies,)
    available = {_normalize_channel(item, field="frequency") for item in frequencies}
    selected_frequency = _normalize_channel(frequency, field="frequency")
    if selected_frequency not in available:
        message = f"NISAR frequency {selected_frequency!r} is unavailable"
        logger.error(message)
        raise ValueError(message)
    values = (
        polarizations.get(selected_frequency)
        or polarizations.get(selected_frequency.lower())
        if isinstance(polarizations, Mapping)
        else polarizations
    )
    if values is None:
        message = f"NISAR frequency {selected_frequency!r} has no polarizations"
        logger.error(message)
        raise InvalidProcessingStateError(message)
    if isinstance(values, str):
        values = (values,)
    available_pols = {_normalize_channel(item, field="polarization") for item in values}
    selected_polarization = _normalize_channel(polarization, field="polarization")
    if selected_polarization not in available_pols:
        message = (
            f"NISAR polarization {selected_polarization!r} is unavailable for "
            f"frequency {selected_frequency!r}"
        )
        logger.error(message)
        raise ValueError(message)
    return selected_frequency, selected_polarization


def _radar_grid(handle: Any, frequency: str, shape: tuple[int, int]) -> RadarGrid:
    """Map an ISCE3 ``RadarGridParameters`` object to the public contract."""
    getter = getattr(handle, "getRadarGrid", None)
    grid = getter(frequency) if callable(getter) else None
    if grid is None:
        raise InvalidProcessingStateError("NISAR RSLC has no radar-grid metadata")
    fields = (
        _value(grid, "starting_range", "starting_slant_range_m"),
        _value(grid, "range_pixel_spacing", "range_spacing_m"),
        _value(grid, "az_time_interval", "azimuth_time_interval_s"),
        _value(grid, "wavelength", "wavelength_m"),
        _value(grid, "lookside", "look_direction"),
    )
    if any(item is None for item in fields):
        raise InvalidProcessingStateError("NISAR radar grid metadata is incomplete")
    look = str(fields[4]).lower()
    look_direction = "left" if "left" in look else "right" if "right" in look else None
    if look_direction is None:
        raise InvalidProcessingStateError(
            f"unsupported NISAR look direction {fields[4]!r}"
        )
    sensing_start = _datetime(_value(grid, "sensing_start"), _value(grid, "ref_epoch"))
    grid_shape = tuple(int(item) for item in _value(grid, "shape") or shape)
    if grid_shape != shape:
        raise InvalidProcessingStateError(
            f"NISAR radar-grid shape {grid_shape} does not match SLC shape {shape}"
        )
    return RadarGrid(
        shape=shape,
        starting_slant_range_m=float(fields[0]),
        range_spacing_m=float(fields[1]),
        sensing_start=sensing_start,
        azimuth_time_interval_s=float(fields[2]),
        wavelength_m=float(fields[3]),
        look_direction=look_direction,  # type: ignore[arg-type]
    )


def _orbit_metadata(handle: Any) -> OrbitMetadata:
    """Map the native ISCE3 orbit state vectors to FanInSAR metadata."""
    orbit = handle.getOrbit()
    size = int(_value(orbit, "size") or 0)
    positions = np.asarray(_value(orbit, "position"))
    velocities = np.asarray(_value(orbit, "velocity"))
    times = tuple(float(item) for item in _value(orbit, "time"))
    if size <= 0 or positions.shape != (size, 3) or velocities.shape != (size, 3):
        raise InvalidProcessingStateError("NISAR orbit state vectors are incomplete")
    reference = _value(orbit, "reference_epoch")
    vectors = tuple(
        OrbitStateVector(
            _datetime(time, reference),
            tuple(float(item) for item in position),
            tuple(float(item) for item in velocity),
        )
        for time, position, velocity in zip(times, positions, velocities, strict=True)
    )
    return OrbitMetadata("ITRF", "nisar-rslc", vectors)


def _valid_samples(handle: Any, frequency: str) -> ValidSampleMask:
    """Read per-line valid samples from the admitted HDF5 source."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("NISAR metadata mapping requires h5py") from error
    path = f"/science/LSAR/RSLC/swaths/frequency{frequency}/validSamplesSubSwath1"
    with h5py.File(str(handle.filename), "r") as source:
        if path not in source:
            raise InvalidProcessingStateError(
                f"missing NISAR valid-sample dataset {path}"
            )
        values = np.asarray(source[path])
    if values.ndim != 2 or values.shape[1] != 2:
        raise InvalidProcessingStateError(
            "NISAR valid-sample dataset must have shape (lines, 2)"
        )
    return ValidSampleMask(
        tuple(int(item) for item in values[:, 0]),
        tuple(int(item) for item in values[:, 1]),
    )


def _doppler(handle: Any, frequency: str) -> DopplerCentroidPolynomial:
    """Approximate the native Doppler LUT by a declared range polynomial."""
    lut = handle.getDopplerCentroid(frequency)
    data = np.asarray(_value(lut, "data"))
    x_axis = np.asarray(tuple(_value(lut, "x_axis")), dtype=float)
    if data.ndim != 2 or x_axis.size != data.shape[1] or not np.isfinite(data).any():
        raise InvalidProcessingStateError("NISAR Doppler LUT is empty or malformed")
    values = np.nanmean(data, axis=0)
    reference_range = float(np.nanmean(x_axis))
    centered = x_axis - reference_range
    degree = 1 if x_axis.size > 1 else 0
    fit = np.polyfit(centered, values, degree)
    coefficients = tuple(float(item) for item in fit[::-1])
    return DopplerCentroidPolynomial(
        coefficients, reference_range, float(_value(lut, "y_start") or 0.0)
    )


def _normalize_channel(value: str, *, field: str) -> str:
    """Normalize and validate one NISAR channel identifier.

    Parameters
    ----------
    value : str
        Frequency or polarization identifier supplied by the caller.
    field : str
        Name used in an actionable validation error.

    Returns
    -------
    str
        Uppercase, non-empty identifier.

    """
    if not isinstance(value, str) or not value.strip():
        message = f"NISAR {field} must be a non-empty string"
        logger.error(message)
        raise ValueError(message)
    return value.strip().upper()


def _reader_factory(readers: Any) -> Any:
    """Resolve a supported optional NISAR reader factory.

    The NISAR package has exposed more than one public spelling across
    environments. Resolving it here keeps the optional dependency at the
    mission boundary and gives callers one stable FanInSAR entry point.
    """
    for name in ("open_product", "RSLC", "RslcReader", "RSLCReader"):
        factory = getattr(readers, name, None)
        if callable(factory):
            return factory
    message = (
        "nisar.products.readers exposes no supported RSLC factory "
        "(expected open_product, RSLC, RslcReader, or RSLCReader)"
    )
    logger.error(message)
    raise ImportError(message)


@register(name="nisar")
class NisarSensor(Sensor):
    """Lazy NISAR RSLC window reader with explicit B/HH defaults.

    This MVP owns optional reader loading and bounded native complex reads. It
    deliberately does not fabricate the orbit, Doppler, valid-sample, or
    source-admission metadata required to build an :class:`SLCProduct`; that
    normalized metadata seam is a later PROPOSAL-0035 implementation slice.
    """

    name = "nisar"

    def admit_source(
        self,
        uri: str | Path,
        *,
        admission: NisarAdmissionPolicy | Mapping[str, object] | None = None,
        **policy_options: Any,
    ) -> _AdmissionSnapshot:
        """Run trusted pre-open admission and return immutable source identity.

        Parameters
        ----------
        uri : path-like
            Local RSLC path.  URI schemes, credentials, symlinks, and
            non-regular files are rejected whenever a policy is supplied.
        admission : NisarAdmissionPolicy or mapping, optional
            Explicit trusted-root, inventory, size, and HDF5-link policy.
        **policy_options : Any
            Convenience scalar policy options accepted by the policy object.

        Returns
        -------
        _AdmissionSnapshot
            Source id, SHA-256, size, and the effective policy.

        Raises
        ------
        InvalidProcessingStateError
            If the source fails the explicit admission policy.

        """
        policy = _coerce_admission_policy(admission, **policy_options)
        if policy is None:
            raise ValueError("NISAR admit_source requires explicit admission metadata")
        return _secure_file_snapshot(_local_source_path(uri), policy)

    def admission_metadata(self, handle: Any) -> dict[str, object] | None:
        """Return source admission metadata captured for one reader handle."""
        snapshot = _handle_admission(handle)
        if snapshot is None:
            return None
        return {
            "source_id": snapshot.source_id,
            "source_digest": snapshot.source_digest,
            "source_size_bytes": snapshot.source_size_bytes,
            "policy": (
                snapshot.policy.as_metadata() if snapshot.policy is not None else None
            ),
        }

    def open_product(
        self,
        uri: str,
        *,
        admission: NisarAdmissionPolicy | Mapping[str, object] | None = None,
        admission_policy: NisarAdmissionPolicy | Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open one RSLC lazily through ``nisar.products.readers``.

        Parameters
        ----------
        uri : str
            Local, already-admitted RSLC path supplied by the caller.
        admission, admission_policy : NisarAdmissionPolicy or mapping, optional
            Explicit trusted pre-open source policy.  If omitted, the legacy
            lightweight reader-fake semantics are retained.
        **kwargs : Any
            Optional reader-specific construction arguments.  Scalar admission
            options (``trusted_roots``, ``expected_sha256``, and
            ``max_size_bytes``) are also accepted explicitly.

        Returns
        -------
        Any
            Native NISAR RSLC reader handle.

        Raises
        ------
        ImportError
            If the optional NISAR reader package or an RSLC factory is absent.
        ValueError
            If the opened object does not expose the native-complex RSLC API.

        """
        if admission is not None and admission_policy is not None:
            raise TypeError("NISAR open_product accepts only one admission policy")
        configured_policy = admission if admission is not None else admission_policy
        policy_options = {
            name: kwargs.pop(name)
            for name in (
                "trusted_roots",
                "expected_sha256",
                "expected_sha256_inventory",
                "sha256_inventory",
                "max_size_bytes",
                "max_size",
                "reject_external_links",
                "check_external_links",
                "policy_id",
            )
            if name in kwargs
        }
        if policy_options:
            if configured_policy is None:
                configured_policy = policy_options
            elif isinstance(configured_policy, Mapping):
                configured_policy = {**configured_policy, **policy_options}
            else:
                raise TypeError(
                    "NISAR admission policy object cannot mix scalar policy options"
                )
        policy = _coerce_admission_policy(configured_policy)
        local_uri = _local_source_path(uri)
        # Capture the source identity before importing/opening the optional
        # reader. Native readers are allowed to cache metadata, so this is the
        # admission boundary rather than a post-open diagnostic.
        snapshot = (
            _secure_file_snapshot(local_uri, policy)
            if policy is not None
            else _source_snapshot(local_uri)
        )
        try:
            readers = import_module("nisar.products.readers")
        except ImportError as error:
            message = (
                "NISAR RSLC support requires optional nisar.products.readers; "
                "install the NISAR/ISCE3 environment before opening an RSLC"
            )
            logger.error(message)
            raise ImportError(message) from error
        factory = _reader_factory(readers)
        try:
            try:
                handle = factory(str(uri), **kwargs)
            except TypeError:
                # The ISCE3/NISAR RSLC class accepts ``hdf5file=`` rather than
                # a positional path; keep compatibility with simple fakes.
                handle = factory(hdf5file=str(uri), **kwargs)
        except Exception:
            logger.exception("Failed to open NISAR RSLC at %s", uri)
            raise
        product_type = str(
            _value(handle, "productType", "product_type", "_ProductType") or ""
        ).upper()
        if product_type and product_type.rsplit(".", 1)[-1] != "RSLC":
            raise ValueError(f"NISAR product {uri!r} is {product_type}, expected RSLC")
        if not callable(getattr(handle, "getSlcDatasetAsNativeComplex", None)):
            message = (
                "NISAR product does not expose getSlcDatasetAsNativeComplex; "
                "expected an RSLC reader handle"
            )
            logger.error(message)
            raise ValueError(message)
        _remember_admission(handle, snapshot)
        return handle

    def to_slc_product(
        self,
        handle: Any,
        *,
        frequency: str = "B",
        polarization: str = "HH",
        acquisition_id: str | None = None,
        source_path: str | Path | None = None,
        **kwargs: Any,
    ) -> SLCReadResult:
        """Normalize one admitted NISAR RSLC channel without reading its raster."""
        if "freq" in kwargs:
            alias = _normalize_channel(kwargs.pop("freq"), field="frequency")
            direct = _normalize_channel(frequency, field="frequency")
            if direct not in {"B", alias}:
                message = "NISAR frequency and freq aliases specify different channels"
                logger.error(message)
                raise ValueError(message)
            frequency = alias
        if "pol" in kwargs:
            alias = _normalize_channel(kwargs.pop("pol"), field="polarization")
            direct = _normalize_channel(polarization, field="polarization")
            if direct not in {"HH", alias}:
                message = (
                    "NISAR polarization and pol aliases specify different channels"
                )
                logger.error(message)
                raise ValueError(message)
            polarization = alias
        frequency, polarization = _select_channel(
            handle,
            frequency,
            polarization,
        )
        if kwargs:
            message = f"Unsupported NISAR product options: {tuple(kwargs)}"
            logger.error(message)
            raise TypeError(message)
        snapshot = _validate_admission(handle, expected_source=source_path)
        dataset = handle.getSlcDatasetAsNativeComplex(frequency, polarization)
        shape = tuple(int(item) for item in getattr(dataset, "shape", ()))
        if len(shape) != 2 or min(shape) <= 0:
            raise InvalidProcessingStateError(
                "NISAR SLC dataset must be a positive 2-D array"
            )
        grid = _radar_grid(handle, frequency, shape)
        resolved_source = str(source_path or _value(handle, "filename") or "")
        if not resolved_source:
            raise InvalidProcessingStateError("NISAR RSLC handle has no source path")
        acquisition = acquisition_id or Path(resolved_source).stem
        product = SLCProduct(
            acquisition_id=acquisition,
            grid=grid,
            samples=ArrayDescriptor(
                uri=f"hdf5://{resolved_source}#{handle.slcPath(frequency, polarization)}",
                shape=shape,
                dtype=str(getattr(dataset, "dtype", "complex64")),
                representation=ArrayRepresentation.COMPLEX,
            ),
            orbit=_orbit_metadata(handle),
            carrier=CarrierState.PRESENT,
            coregistration=CoregistrationState.NOT_REGISTERED,
            calibration=CalibrationState.RAW_DN,
        )
        return SLCReadResult(
            product=product,
            valid_samples=_valid_samples(handle, frequency),
            doppler_centroid=_doppler(handle, frequency),
            source_path=resolved_source,
            mission_native={
                "mission": "NISAR",
                "product": "RSLC",
                "frequency": frequency,
                "polarization": polarization,
                "source_id": snapshot.source_id if snapshot else resolved_source,
                "source_digest": snapshot.source_digest if snapshot else "",
                "source_size_bytes": snapshot.source_size_bytes if snapshot else 0,
                "admission_policy": (
                    snapshot.policy.as_metadata()
                    if snapshot is not None and snapshot.policy is not None
                    else None
                ),
                "lineage": snapshot.source_id if snapshot else resolved_source,
            },
        )

    def read_slc_window(
        self,
        handle: Any,
        window: tuple[slice, slice],
        *,
        frequency: str = "B",
        polarization: str = "HH",
        **kwargs: Any,
    ) -> np.ndarray:
        """Read a bounded complex native RSLC window without full materialization.

        Parameters
        ----------
        handle : Any
            Handle returned by :meth:`open_product` or a compatible reader.
        window : tuple[slice, slice]
            Native line/sample window; both steps must be one and bounds must
            be inside the selected dataset.
        frequency : str, default="B"
            NISAR frequency identifier. The MVP's explicit default is B.
        polarization : str, default="HH"
            NISAR polarization identifier. The MVP's explicit default is HH.
        **kwargs : Any
            Compatibility aliases ``freq`` and ``pol`` are accepted.

        Returns
        -------
        numpy.ndarray
            Two-dimensional complex sample window.

        Raises
        ------
        ValueError
            If channel values, dataset shape, selection, or dtype are invalid.

        """
        frequency = _normalize_channel(kwargs.pop("freq", frequency), field="frequency")
        polarization = _normalize_channel(
            kwargs.pop("pol", polarization), field="polarization"
        )
        if kwargs:
            message = f"Unsupported NISAR window-read options: {tuple(kwargs)}"
            logger.error(message)
            raise TypeError(message)
        _validate_admission(handle)
        getter = getattr(handle, "getSlcDatasetAsNativeComplex", None)
        if not callable(getter):
            message = "NISAR handle has no getSlcDatasetAsNativeComplex method"
            logger.error(message)
            raise ValueError(message)
        dataset = getter(frequency, polarization)
        shape = tuple(int(item) for item in getattr(dataset, "shape", ()))
        if len(shape) != 2 or min(shape) <= 0:
            message = "NISAR RSLC dataset must expose a positive two-dimensional shape"
            logger.error(message)
            raise ValueError(message)
        for axis, (size, item) in enumerate(zip(shape, window, strict=True)):
            if item.step not in (None, 1):
                raise ValueError(f"selection step must be 1 on axis {axis}")
            if item.start is not None and not 0 <= item.start < size:
                raise ValueError(f"selection out of bounds on axis {axis}: {item!r}")
            if item.stop is not None and not 0 < item.stop <= size:
                raise ValueError(f"selection out of bounds on axis {axis}: {item!r}")
        selection = normalize_selection(shape, window)
        samples = np.asarray(dataset[selection])
        if samples.shape != tuple(item.stop - item.start for item in selection):
            message = "NISAR reader returned a window with an unexpected shape"
            logger.error(message)
            raise ValueError(message)
        if not np.issubdtype(samples.dtype, np.complexfloating):
            message = "NISAR RSLC window must retain complex samples"
            logger.error(message)
            raise ValueError(message)
        return samples


def admit_nisar_source(
    uri: str | Path,
    *,
    admission: NisarAdmissionPolicy | Mapping[str, object] | None = None,
    **policy_options: Any,
) -> dict[str, object]:
    """Admit one NISAR source without importing the optional reader package."""
    snapshot = NisarSensor().admit_source(uri, admission=admission, **policy_options)
    return {
        "source_id": snapshot.source_id,
        "source_digest": snapshot.source_digest,
        "source_size_bytes": snapshot.source_size_bytes,
        "policy": (
            snapshot.policy.as_metadata() if snapshot.policy is not None else None
        ),
    }


__all__ = [
    "NisarAdmissionPolicy",
    "NisarSensor",
    "admit_nisar_source",
]

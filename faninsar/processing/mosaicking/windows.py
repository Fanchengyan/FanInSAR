"""Typed ROI-local windows for midpoint burst selection (PROPOSAL-0074)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from faninsar.processing.errors import reject_invalid_state

MergeWindow: TypeAlias = tuple[int, int, int, int]
MergeDomain: TypeAlias = Literal["radar", "geo"]
PhaseDomain: TypeAlias = Literal["complex", "unwrapped"]
BurstRepresentation: TypeAlias = Literal["burst", "merged"]
SeamAxis: TypeAlias = Literal["azimuth", "range"]

_SHA256_HEX = frozenset("0123456789abcdef")


def validate_merge_window(window: object, *, name: str = "window") -> MergeWindow:
    """Return a half-open ``(r0, r1, c0, c1)`` window or reject it."""
    if (
        not isinstance(window, tuple)
        or len(window) != 4
        or any(type(value) is not int for value in window)
    ):
        reject_invalid_state(f"{name} must be a tuple of four ints (r0, r1, c0, c1)")
    row0, row1, col0, col1 = window
    if row1 <= row0 or col1 <= col0:
        reject_invalid_state(f"{name} must be a non-empty half-open rectangle")
    return row0, row1, col0, col1


def merge_window_shape(window: MergeWindow) -> tuple[int, int]:
    """Return ``(rows, cols)`` for a validated half-open window."""
    row0, row1, col0, col1 = window
    return row1 - row0, col1 - col0


def intersect_windows(left: MergeWindow, right: MergeWindow) -> MergeWindow | None:
    """Return the overlapping half-open window, or ``None``."""
    row0 = max(left[0], right[0])
    row1 = min(left[1], right[1])
    col0 = max(left[2], right[2])
    col1 = min(left[3], right[3])
    if row0 >= row1 or col0 >= col1:
        return None
    return row0, row1, col0, col1


def window_slices(parent: MergeWindow, region: MergeWindow) -> tuple[slice, slice]:
    """Return local row/column slices of ``region`` inside ``parent``."""
    overlap = intersect_windows(parent, region)
    if overlap is None:
        reject_invalid_state("window region does not intersect parent window")
    return (
        slice(overlap[0] - parent[0], overlap[1] - parent[0]),
        slice(overlap[2] - parent[2], overlap[3] - parent[2]),
    )


def _require_array(
    value: np.ndarray,
    *,
    shape: tuple[int, int],
    name: str,
    dtype: np.dtype[object] | type[np.generic] | None = None,
    kind: str | None = None,
) -> np.ndarray:
    """Reject an array that does not match the window contract."""
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        reject_invalid_state(f"{name} must be a 2-D array")
    if value.shape != shape:
        reject_invalid_state(f"{name} shape {value.shape} must match window {shape}")
    if dtype is not None and value.dtype != dtype:
        reject_invalid_state(f"{name} must have dtype {np.dtype(dtype).name}")
    if kind is not None and value.dtype.kind != kind:
        reject_invalid_state(f"{name} must have dtype kind {kind!r}")
    return value


@dataclass(frozen=True, slots=True)
class MergeScope:
    """Track/channel/Reference/grid/polygon identity for one merge."""

    domain: MergeDomain
    track_id: str
    channel: str
    reference_id: str
    grid_identity: str
    polygon_identity: str

    def __post_init__(self) -> None:
        """Reject incomplete or malformed scope keys."""
        if self.domain not in ("radar", "geo"):
            reject_invalid_state("merge scope domain must be 'radar' or 'geo'")
        for field_name in (
            "track_id",
            "channel",
            "reference_id",
            "polygon_identity",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                reject_invalid_state(
                    f"merge scope {field_name} must be a non-empty string"
                )
        identity = self.grid_identity
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(character not in _SHA256_HEX for character in identity)
        ):
            reject_invalid_state(
                "merge scope grid_identity must be a lowercase SHA-256"
            )


@dataclass(frozen=True, slots=True)
class MergeSeam:
    """One precomputed midpoint cut between two canonical burst identities."""

    axis: SeamAxis
    boundary: int
    earlier_id: str
    later_id: str

    def __post_init__(self) -> None:
        """Reject malformed seam metadata."""
        if self.axis not in ("azimuth", "range"):
            reject_invalid_state("merge seam axis must be 'azimuth' or 'range'")
        if type(self.boundary) is not int:
            reject_invalid_state("merge seam boundary must be an int")
        if not self.earlier_id or not self.later_id or self.earlier_id == self.later_id:
            reject_invalid_state(
                "merge seam identities must be distinct non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class LayoutSourceWindow:
    """One burst's ROI-local support on the shared layout grid."""

    burst_id: str
    window: MergeWindow
    scientific_support: np.ndarray
    swath: str

    def __post_init__(self) -> None:
        """Validate source identity and window-local support."""
        if not isinstance(self.burst_id, str) or not self.burst_id:
            reject_invalid_state("layout source burst_id must be a non-empty string")
        if not isinstance(self.swath, str) or not self.swath:
            reject_invalid_state("layout source swath must be a non-empty string")
        window = validate_merge_window(self.window, name="layout source window")
        object.__setattr__(self, "window", window)
        support = _require_array(
            np.asarray(self.scientific_support, dtype=bool),
            shape=merge_window_shape(window),
            name="layout source scientific_support",
            dtype=bool,
        )
        object.__setattr__(self, "scientific_support", support)


@dataclass(frozen=True, slots=True)
class MergeLayoutWindow:
    """ROI-local preferred-owner map for one merge scope."""

    scope: MergeScope
    window: MergeWindow
    source_ids: tuple[str, ...]
    owner: np.ndarray
    scientific_support: np.ndarray
    seams: tuple[MergeSeam, ...] = ()
    source_windows: tuple[LayoutSourceWindow, ...] = ()

    def __post_init__(self) -> None:
        """Validate owner labels against the ROI window and source table."""
        if not isinstance(self.scope, MergeScope):
            reject_invalid_state("merge layout requires a MergeScope")
        window = validate_merge_window(self.window, name="layout window")
        object.__setattr__(self, "window", window)
        if not isinstance(self.source_ids, tuple) or not self.source_ids:
            reject_invalid_state("merge layout source_ids must be a non-empty tuple")
        if len(set(self.source_ids)) != len(self.source_ids):
            reject_invalid_state("merge layout source_ids must be unique")
        if any(not isinstance(item, str) or not item for item in self.source_ids):
            reject_invalid_state("merge layout source_ids must be non-empty strings")
        shape = merge_window_shape(window)
        owner = np.asarray(self.owner)
        if owner.dtype != np.uint16:
            owner = owner.astype(np.uint16, copy=False)
        owner = _require_array(owner, shape=shape, name="layout owner")
        maximum = int(owner.max()) if owner.size else 0
        if maximum > len(self.source_ids):
            reject_invalid_state("layout owner labels exceed source_ids")
        support = _require_array(
            np.asarray(self.scientific_support, dtype=bool),
            shape=shape,
            name="layout scientific_support",
            dtype=bool,
        )
        object.__setattr__(self, "owner", np.where(support, owner, np.uint16(0)))
        object.__setattr__(self, "scientific_support", support)
        object.__setattr__(self, "seams", tuple(self.seams))
        object.__setattr__(self, "source_windows", tuple(self.source_windows))


@dataclass(frozen=True, slots=True)
class BurstWindow:
    """Window-local burst samples admitted to :func:`merge_bursts`."""

    scope: MergeScope
    burst_id: str
    acquisitions: tuple[str, ...]
    write_ids: tuple[str, ...]
    phase_convention: str
    window: MergeWindow
    complex: np.ndarray
    valid_mask: np.ndarray
    coherence: np.ndarray | None = None
    primary_power: np.ndarray | None = None
    secondary_power: np.ndarray | None = None
    phase_domain: PhaseDomain = "complex"
    representation: BurstRepresentation = "burst"

    def __post_init__(self) -> None:
        """Validate identities, window-local arrays, and phase-domain defaults."""
        if not isinstance(self.scope, MergeScope):
            reject_invalid_state("burst window requires a MergeScope")
        if not isinstance(self.burst_id, str) or not self.burst_id:
            reject_invalid_state("burst_id must be a non-empty string")
        if not isinstance(self.acquisitions, tuple) or not self.acquisitions:
            reject_invalid_state("burst acquisitions must be a non-empty tuple")
        if not isinstance(self.write_ids, tuple) or not self.write_ids:
            reject_invalid_state("burst write_ids must be a non-empty tuple")
        if not isinstance(self.phase_convention, str) or not self.phase_convention:
            reject_invalid_state("phase_convention must be a non-empty string")
        if self.phase_domain != "complex":
            reject_invalid_state("burst window phase_domain must be 'complex'")
        if self.representation not in ("burst", "merged"):
            reject_invalid_state(
                "burst window representation must be 'burst' or 'merged'"
            )
        window = validate_merge_window(self.window, name="burst window")
        object.__setattr__(self, "window", window)
        shape = merge_window_shape(window)
        samples = np.asarray(self.complex)
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64, copy=False)
        samples = _require_array(samples, shape=shape, name="burst complex", kind="c")
        mask = _require_array(
            np.asarray(self.valid_mask, dtype=bool),
            shape=shape,
            name="burst valid_mask",
            dtype=bool,
        )
        object.__setattr__(self, "complex", samples)
        object.__setattr__(self, "valid_mask", mask)
        for field_name in ("coherence", "primary_power", "secondary_power"):
            value = getattr(self, field_name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.kind != "f":
                array = array.astype(np.float32, copy=False)
            object.__setattr__(
                self,
                field_name,
                _require_array(
                    array, shape=shape, name=f"burst {field_name}", kind="f"
                ),
            )


@dataclass(frozen=True, slots=True)
class MergedWindow:
    """Selected native samples on the layout ROI window."""

    scope: MergeScope
    window: MergeWindow
    complex: np.ndarray
    valid_mask: np.ndarray
    source: np.ndarray
    coherence: np.ndarray | None = None
    primary_power: np.ndarray | None = None
    secondary_power: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate merged arrays against the layout window."""
        window = validate_merge_window(self.window, name="merged window")
        object.__setattr__(self, "window", window)
        shape = merge_window_shape(window)
        samples = np.asarray(self.complex)
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64, copy=False)
        object.__setattr__(
            self,
            "complex",
            _require_array(samples, shape=shape, name="merged complex", kind="c"),
        )
        object.__setattr__(
            self,
            "valid_mask",
            _require_array(
                np.asarray(self.valid_mask, dtype=bool),
                shape=shape,
                name="merged valid_mask",
                dtype=bool,
            ),
        )
        source = np.asarray(self.source)
        if source.dtype != np.uint16:
            source = source.astype(np.uint16, copy=False)
        object.__setattr__(
            self, "source", _require_array(source, shape=shape, name="merged source")
        )

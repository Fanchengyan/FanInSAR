"""Mission-neutral registration values used by the Stack facade.

The values in this module deliberately contain only data.  Mission adapters own
selection and numerical execution; the shared Stack owns identity and report
publication.
"""

# ruff: noqa: EM101, TRY003

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

RegistrationStatus: TypeAlias = Literal[
    "skipped/no-window",
    "observed-but-qc-empty",
    "observed-partial",
    "observed-valid",
    "applied",
    "failed/unsupported-capability",
    "failed/fingerprint-mismatch",
]


def _json_value(value: object) -> object:
    """Convert a supported identity value into deterministic JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (Sequence, set, frozenset)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    return str(value)


def identity_digest(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 digest for a data-only identity payload.

    Parameters
    ----------
    payload : mapping
        Scene, workflow, grid, frame, unit, and provenance values.

    Returns
    -------
    str
        Lowercase hexadecimal digest of canonical JSON.

    """
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RegistrationCapability:
    """Registered mission/product/grid/strategy workflow key."""

    mission: str
    product: str
    grid: str
    strategy: str

    def as_dict(self) -> dict[str, str]:
        """Return the capability as JSON-compatible data."""
        return {
            "mission": self.mission,
            "product": self.product,
            "grid": self.grid,
            "strategy": self.strategy,
        }


@dataclass(frozen=True, slots=True)
class CampaignSelection:
    """Immutable scene selection shared by all phases of one campaign."""

    scenes: tuple[str, ...]
    reference: str
    grid: str
    frame: str = "radar"
    units: str = "pixel"
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the selection's minimal identity contract."""
        scenes = tuple(str(scene) for scene in self.scenes)
        if not scenes:
            raise ValueError("campaign selection requires at least one scene")
        if self.reference not in scenes:
            raise ValueError("campaign reference must be one of the selected scenes")
        object.__setattr__(self, "scenes", scenes)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def as_dict(self) -> dict[str, object]:
        """Return deterministic, serializable selection data."""
        return {
            "scenes": list(self.scenes),
            "reference": self.reference,
            "grid": self.grid,
            "frame": self.frame,
            "units": self.units,
            "provenance": _json_value(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class RegistrationCorrection:
    """Mission-neutral correction payload with explicit omitted axes."""

    kind: str
    units: str
    azimuth: Mapping[str, float] = field(default_factory=dict)
    range: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy mappings so correction values cannot change after publication."""
        object.__setattr__(self, "azimuth", dict(self.azimuth))
        object.__setattr__(self, "range", dict(self.range))
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True, slots=True)
class CoregistrationReport:
    """Versioned, mission-neutral report envelope for one Stack run."""

    capability: RegistrationCapability
    scenes: tuple[str, ...]
    reference: str
    grid: str
    frame: str
    units: str
    status: RegistrationStatus
    run_identity: str
    correction_kind: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = 1
    report_kind: str = "coregistration"

    def __post_init__(self) -> None:
        """Normalize tuple and mappings at the immutable report boundary."""
        object.__setattr__(self, "scenes", tuple(str(scene) for scene in self.scenes))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "details", dict(self.details))

    def as_dict(self) -> dict[str, object]:
        """Return the report envelope as data-only JSON-compatible values."""
        return {
            "schema_version": self.schema_version,
            "report_kind": self.report_kind,
            "capability": self.capability.as_dict(),
            "scenes": list(self.scenes),
            "reference": self.reference,
            "grid": self.grid,
            "frame": self.frame,
            "units": self.units,
            "status": self.status,
            "run_identity": self.run_identity,
            "correction_kind": self.correction_kind,
            "provenance": _json_value(self.provenance),
            "details": _json_value(self.details),
        }


class RegistrationWorkflow(Protocol):
    """Internal mission workflow protocol used by :class:`Stack`."""

    capability: RegistrationCapability

    def resolve_selection(self, context: object) -> CampaignSelection:
        """Resolve one immutable selection for all workflow phases."""

    def apply(
        self,
        selection: CampaignSelection,
        correction: RegistrationCorrection | None,
    ) -> object:
        """Apply a mission-owned correction to the selected scenes."""

    def report(self, result: object) -> CoregistrationReport:
        """Build the generic report envelope for an apply result."""


__all__ = [
    "CampaignSelection",
    "CoregistrationReport",
    "RegistrationCapability",
    "RegistrationCorrection",
    "RegistrationStatus",
    "RegistrationWorkflow",
    "identity_digest",
]

"""Deterministic JSON and STAC provenance contracts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from faninsar.processing.errors import reject_invalid_state

_GIT_REVISION_CACHE: dict[str, str | None] = {}
_GIT_REVISION_SENTINEL: object = object()


def _read_git_revision(repository: Path) -> str | None:
    """Read the checked-out revision of a git repository without subprocess.

    Only plain checkouts (``.git/HEAD`` pointing at a ref file, or a detached
    ``.git/HEAD``) are resolved; anything else returns ``None``.
    """
    git_dir = repository / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8", errors="replace").strip()
        marker = "gitdir: "
        if not text.startswith(marker):
            return None
        git_dir = repository / text[len(marker) :].strip()
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8", errors="replace").strip()
    if re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    if not value.startswith("ref: "):
        return None
    return _read_git_ref(git_dir, value[len("ref: ") :].strip())


def _read_git_ref(git_dir: Path, ref: str) -> str | None:
    """Resolve one git ref name to a commit id, loose file or packed."""
    if not re.fullmatch(r"[A-Za-z0-9/._-]+", ref):
        return None
    for candidate in (git_dir, _common_git_dir(git_dir)):
        if candidate is None:
            continue
        ref_file = candidate / ref
        if ref_file.is_file():
            resolved = ref_file.read_text(encoding="utf-8", errors="replace").strip()
            if resolved:
                return resolved
        packed = candidate / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.strip().endswith(f" {ref}"):
                    return line.split()[0]
    return None


def _common_git_dir(git_dir: Path) -> Path | None:
    """Return the common git directory of a possible worktree git dir."""
    commondir = git_dir / "commondir"
    if not commondir.is_file():
        return None
    value = commondir.read_text(encoding="utf-8", errors="replace").strip()
    if not value:
        return None
    common = Path(value)
    return common if common.is_absolute() else (git_dir / common).resolve()


def faninsar_git_revision() -> str | None:
    """Return the enclosing FanInSAR Git revision, when the checkout has one.

    The revision is resolved once per process. It is product provenance, not a
    resume or reuse key: a later checkout of the same library must not force
    scientific products to be recomputed.

    Returns
    -------
    str or None
        A 40-character commit id, or ``None`` when FanInSAR is not running
        from a readable Git checkout.

    """
    cached = _GIT_REVISION_CACHE.get("revision", _GIT_REVISION_SENTINEL)
    if cached is not _GIT_REVISION_SENTINEL:
        return cached  # type: ignore[return-value]
    repository = Path(__file__).resolve().parents[3]
    revision = None
    try:
        revision = _read_git_revision(repository)
    except OSError:
        revision = None
    _GIT_REVISION_CACHE["revision"] = revision
    return revision


@dataclass(frozen=True, slots=True)
class SoftwareIdentity:
    """Name and version of software responsible for a processing event."""

    name: str
    version: str

    def __post_init__(self) -> None:
        """Validate that software identity is complete."""
        if not self.name or not self.version:
            reject_invalid_state("software name and version are required")


@dataclass(frozen=True, slots=True)
class ProcessingEvent:
    """One deterministic processing operation and its scalar parameters."""

    event_id: str
    operation: str
    parameters: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Validate event identity and unique parameter keys."""
        if not self.event_id or not self.operation:
            reject_invalid_state("provenance event ID and operation are required")
        keys = tuple(key for key, _ in self.parameters)
        if len(keys) != len(set(keys)):
            reject_invalid_state("provenance parameter keys must be unique")
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))


@dataclass(frozen=True, slots=True)
class StacProvenanceProperties:
    """STAC-compatible scalar processing extension properties."""

    product_id: str
    software_name: str
    software_version: str
    input_ids: tuple[str, ...]
    operation_ids: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize properties with canonical key and value ordering."""
        payload = {
            "faninsar:input_ids": sorted(self.input_ids),
            "faninsar:operation_ids": sorted(self.operation_ids),
            "faninsar:product_id": self.product_id,
            "processing:software": self.software_name,
            "processing:software_version": self.software_version,
        }
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Canonical provenance graph for a single processing product."""

    product_id: str
    software: SoftwareIdentity
    inputs: tuple[str, ...]
    events: tuple[ProcessingEvent, ...]

    def __post_init__(self) -> None:
        """Validate product identity and unique event IDs."""
        if not self.product_id:
            reject_invalid_state("provenance product ID must not be empty")
        event_ids = tuple(event.event_id for event in self.events)
        if len(event_ids) != len(set(event_ids)):
            reject_invalid_state("provenance event IDs must be unique")
        if len(self.inputs) != len(set(self.inputs)):
            reject_invalid_state("provenance input IDs must be unique")
        object.__setattr__(self, "inputs", tuple(sorted(self.inputs)))
        object.__setattr__(
            self, "events", tuple(sorted(self.events, key=lambda event: event.event_id))
        )

    def to_json(self) -> str:
        """Serialize the record deterministically."""
        payload = {
            "events": [
                {
                    "event_id": event.event_id,
                    "operation": event.operation,
                    "parameters": dict(sorted(event.parameters)),
                }
                for event in sorted(self.events, key=lambda item: item.event_id)
            ],
            "inputs": sorted(self.inputs),
            "product_id": self.product_id,
            "software": {
                "name": self.software.name,
                "version": self.software.version,
            },
        }
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_json(cls, value: str) -> ProvenanceRecord:
        """Parse a record previously emitted by :meth:`to_json`."""
        try:
            payload = json.loads(value)
            events = tuple(
                ProcessingEvent(
                    event_id=event["event_id"],
                    operation=event["operation"],
                    parameters=tuple(sorted(event["parameters"].items())),
                )
                for event in payload["events"]
            )
            return cls(
                product_id=payload["product_id"],
                software=SoftwareIdentity(**payload["software"]),
                inputs=tuple(payload["inputs"]),
                events=events,
            )
        except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
            reject_invalid_state("value must be valid provenance JSON")

    def to_stac_properties(self) -> StacProvenanceProperties:
        """Return deterministic STAC processing extension properties."""
        return StacProvenanceProperties(
            product_id=self.product_id,
            software_name=self.software.name,
            software_version=self.software.version,
            input_ids=tuple(sorted(self.inputs)),
            operation_ids=tuple(sorted(event.event_id for event in self.events)),
        )


__all__ = [
    "ProcessingEvent",
    "ProvenanceRecord",
    "SoftwareIdentity",
    "StacProvenanceProperties",
    "faninsar_git_revision",
]

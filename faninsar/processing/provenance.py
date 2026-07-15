"""Deterministic JSON and STAC provenance contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .errors import reject_invalid_state


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
]

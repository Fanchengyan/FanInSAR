from __future__ import annotations

import json

import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.provenance import (
    ProcessingEvent,
    ProvenanceRecord,
    SoftwareIdentity,
)


def test_provenance_json_and_stac_are_deterministic() -> None:
    record = ProvenanceRecord(
        product_id="pair-20240101-20240113",
        software=SoftwareIdentity(name="faninsar", version="0.3.0"),
        inputs=("s1-20240113", "s1-20240101"),
        events=(
            ProcessingEvent(
                event_id="coreg-1",
                operation="coregistration",
                parameters=(("method", "tops"), ("oversampling", "32")),
            ),
        ),
    )

    first = record.to_json()
    second = record.to_json()
    parsed = json.loads(first)

    assert first == second
    assert parsed["inputs"] == ["s1-20240101", "s1-20240113"]
    assert ProvenanceRecord.from_json(first) == record
    assert (
        record.to_stac_properties().to_json() == record.to_stac_properties().to_json()
    )


def test_provenance_round_trip_preserves_metadata() -> None:
    record = ProvenanceRecord(
        product_id="geo-track",
        software=SoftwareIdentity(name="faninsar", version="0.3.0"),
        inputs=("radar-track",),
        events=(),
    )

    assert ProvenanceRecord.from_json(record.to_json()) == record


def test_provenance_rejects_malformed_json_with_typed_error() -> None:
    with pytest.raises(InvalidProcessingStateError, match="valid provenance JSON"):
        ProvenanceRecord.from_json("not-json")


def test_provenance_rejects_malformed_nested_parameters() -> None:
    malformed = json.dumps(
        {
            "events": [
                {
                    "event_id": "event-1",
                    "operation": "coregistration",
                    "parameters": [],
                }
            ],
            "inputs": ["slc-1"],
            "product_id": "pair-1",
            "software": {"name": "faninsar", "version": "0.3.0"},
        }
    )

    with pytest.raises(InvalidProcessingStateError, match="valid provenance JSON"):
        ProvenanceRecord.from_json(malformed)

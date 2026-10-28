from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from faninsar.io.storage.provenance import (
    _GIT_REVISION_CACHE,
    ProcessingEvent,
    ProvenanceRecord,
    SoftwareIdentity,
    _read_git_revision,
    faninsar_git_revision,
)
from faninsar.io.storage.validation import (
    ManifestValidationError,
    validate_pipeline_rebuild_manifest,
)
from faninsar.processing.errors import InvalidProcessingStateError


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


def test_read_git_revision_from_detached_head(tmp_path: Path) -> None:
    """A detached checkout reports the SHA in ``.git/HEAD``."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
    assert _read_git_revision(tmp_path) == "a" * 40


def test_read_git_revision_from_branch_ref(tmp_path: Path) -> None:
    """A branch checkout follows ``HEAD`` to the loose ref file."""
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text("b" * 40 + "\n", encoding="utf-8")
    assert _read_git_revision(tmp_path) == "b" * 40


def test_faninsar_git_revision_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process-wide lookup is resolved once and then reused."""
    from faninsar.io.storage import provenance as provenance_module

    calls = {"n": 0}

    def fake_read(_repository: Path) -> str:
        calls["n"] += 1
        return "c" * 40

    monkeypatch.setattr(provenance_module, "_read_git_revision", fake_read)
    _GIT_REVISION_CACHE.clear()
    try:
        first = faninsar_git_revision()
        second = faninsar_git_revision()
    finally:
        _GIT_REVISION_CACHE.clear()
    assert first == "c" * 40
    assert second == first
    assert calls["n"] == 1


def test_rebuild_manifest_rejects_mutated_checksum(tmp_path: Path) -> None:
    source = Path("tests/reference/pipeline_rebuild_manifest.yaml")
    payload = yaml.safe_load(source.read_text())
    payload["corpus"]["scenes"][0]["sha256"] = "0" * 64
    mutated = tmp_path / source.name
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_pipeline_rebuild_manifest(mutated)


def test_rebuild_manifest_rejects_invalid_retirement_archive_digest(
    tmp_path: Path,
) -> None:
    source = Path("tests/reference/pipeline_rebuild_manifest.yaml")
    payload = yaml.safe_load(source.read_text())
    payload["baseline"]["retired_history"]["archive_sha256"] = "0" * 63
    mutated = tmp_path / source.name
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ManifestValidationError, match="archive_sha256"):
        validate_pipeline_rebuild_manifest(mutated)

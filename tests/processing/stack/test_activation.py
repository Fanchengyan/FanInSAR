"""Tests for the owner-controlled Stack activation authority."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from faninsar.processing.contracts import ActivationToken
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.activation import LocalActivationAuthority

if TYPE_CHECKING:
    from pathlib import Path


def _token_template() -> ActivationToken:
    """Return a qualified token subject with an empty issuer binding."""
    return ActivationToken(
        intent_id="intent",
        parent_id="stack-generation",
        parent_manifest_digest="c" * 64,
        root_device=1,
        root_inode=2,
        namespace="scene_artifact_v1",
        mode="qualified",
        domain="radar",
        policy_identity="policy-v1",
        code_identity="code-v1",
        schema="scene_artifact_v1",
        provider_receipt_digest="a" * 64,
        threshold_configuration_hash="d" * 64,
        qualification_evidence_digest="e" * 64,
        p19_qualified_event_ids=("p19-qualified",),
        p18_stack_gate_event_id="p18-stack",
        fence_epoch=1,
        issuer_record_digest="0" * 64,
    )


def _qualified_authority(root: Path) -> LocalActivationAuthority:
    """Create an authority with the required P18/P19 activation events."""
    authority = LocalActivationAuthority.initialize(root)
    _ = authority.issue_gate_event(
        event_id="p18-provider",
        gate_id="P18-provider-qualified",
        producer_commit="p18-provider-commit",
        predecessor_event_ids=(),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="b" * 64,
        evidence_digest="1" * 64,
        activation_mode="qualified",
    )
    _ = authority.issue_gate_event(
        event_id="p19-correctness",
        gate_id="P19-stack-correctness-verified",
        producer_commit="p19-correctness-commit",
        predecessor_event_ids=("p18-provider",),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="c" * 64,
        evidence_digest="2" * 64,
        activation_mode="reference",
    )
    _ = authority.issue_gate_event(
        event_id="p18-stack",
        gate_id="P18-stack-qualified",
        producer_commit="p18-commit",
        predecessor_event_ids=("p19-correctness",),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="c" * 64,
        evidence_digest="e" * 64,
        activation_mode="qualified",
    )
    _ = authority.issue_gate_event(
        event_id="p19-qualified",
        gate_id="P19-stack-qualified-activation",
        producer_commit="p19-commit",
        predecessor_event_ids=("p18-stack",),
        provider_receipt_digest="a" * 64,
        parent_manifest_digest="c" * 64,
        evidence_digest="e" * 64,
        activation_mode="qualified",
    )
    return authority


def test_authority_issues_and_verifies_qualified_token(tmp_path: Path) -> None:
    """A token is usable only after signed gate events are durably present."""
    authority = _qualified_authority(tmp_path / "authority")

    token = authority.issue_token(_token_template())
    record = authority.verify_token(token)

    assert token.issuer_record_digest == record.digest()
    assert record.event_record_digests


def test_authority_rejects_caller_asserted_issuer_digest(tmp_path: Path) -> None:
    """A caller cannot replace authority issuance with an arbitrary digest."""
    authority = _qualified_authority(tmp_path / "authority")
    forged = replace(_token_template(), issuer_record_digest="f" * 64)

    with pytest.raises(InvalidProcessingStateError, match="unavailable"):
        authority.verify_token(forged)


def test_authority_rejects_tampered_gate_event(tmp_path: Path) -> None:
    """Changing a persisted event invalidates its coordinator signature."""
    authority = _qualified_authority(tmp_path / "authority")
    event_path = next((authority.root / "events").glob("*.json"))
    record = json.loads(event_path.read_text(encoding="utf-8"))
    record["producer_commit"] = "tampered"
    event_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(InvalidProcessingStateError, match="signature"):
        authority.issue_token(_token_template())


def test_authority_rejects_token_subject_mutation(tmp_path: Path) -> None:
    """A signed token cannot be replayed for another parent or code identity."""
    authority = _qualified_authority(tmp_path / "authority")
    token = authority.issue_token(_token_template())
    changed = replace(token, code_identity="other-code")

    with pytest.raises(InvalidProcessingStateError, match="subject"):
        authority.verify_token(changed)


def test_authority_rejects_gate_without_typed_predecessor(tmp_path: Path) -> None:
    """The event store enforces the accepted acyclic P18/P19 gate graph."""
    authority = LocalActivationAuthority.initialize(tmp_path / "authority")

    with pytest.raises(InvalidProcessingStateError, match="predecessor"):
        authority.issue_gate_event(
            event_id="p18-stack",
            gate_id="P18-stack-qualified",
            producer_commit="p18-commit",
            predecessor_event_ids=(),
            provider_receipt_digest="a" * 64,
            parent_manifest_digest="c" * 64,
            evidence_digest="e" * 64,
            activation_mode="qualified",
        )

"""Owner-controlled activation authority for qualified Stack execution.

The authority stores signed gate events and token issuer records under a
caller-owned local root.  Qualified Stack entry points verify these durable
records before accepting an :class:`~faninsar.processing.contracts.ActivationToken`.
The implementation uses HMAC-SHA256 from the Python standard library so the
private coordinator key never leaves the owner-only authority root.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from faninsar.processing.contracts.prepared_geometry import ActivationToken

GateId = Literal[
    "P18-provider-qualified",
    "P19-stack-correctness-verified",
    "P18-stack-qualified",
    "P19-stack-qualified-activation",
]

_AUTHORITY_SCHEMA = "stack_activation_authority.v1"
_ISSUER_SCHEMA = "stack_activation_issuer.v1"
_EVENT_SCHEMA = "stack_gate_event.v1"
_MAX_RECORD_BYTES = 64 * 1024
_ZERO_DIGEST = "0" * 64
_REQUIRED_PREDECESSOR: dict[GateId, GateId | None] = {
    "P18-provider-qualified": None,
    "P19-stack-correctness-verified": "P18-provider-qualified",
    "P18-stack-qualified": "P19-stack-correctness-verified",
    "P19-stack-qualified-activation": "P18-stack-qualified",
}

logger = setup_logger(__name__)


def _canonical_json(value: object) -> bytes:
    """Serialize an authority record with one deterministic encoding."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"activation record is not canonicalizable: {error}")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        reject_invalid_state(f"{name} must be a lowercase SHA-256 digest")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        reject_invalid_state(f"{name} must be a non-empty string")


def _record_name(identity: str) -> str:
    """Return a traversal-safe content name for an external identifier."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json"


@dataclass(frozen=True, slots=True)
class StackGateEvent:
    """Signed evidence for one transition in the P18/P19 gate graph."""

    event_id: str
    gate_id: GateId
    producer_commit: str
    predecessor_event_ids: tuple[str, ...]
    provider_receipt_digest: str
    parent_manifest_digest: str
    evidence_digest: str
    activation_mode: Literal["qualified", "reference"]
    issued_at_ns: int
    expires_at_ns: int
    key_id: str
    signature: str

    def __post_init__(self) -> None:
        """Validate bounded event fields before persistence or verification."""
        object.__setattr__(
            self, "predecessor_event_ids", tuple(self.predecessor_event_ids)
        )
        for name in ("event_id", "gate_id", "producer_commit", "key_id"):
            _require_text(str(getattr(self, name)), name)
        for name in (
            "provider_receipt_digest",
            "parent_manifest_digest",
            "evidence_digest",
            "signature",
        ):
            _require_digest(getattr(self, name), name)
        if self.activation_mode not in ("qualified", "reference"):
            reject_invalid_state("unsupported gate-event activation mode")
        if self.issued_at_ns < 0 or self.expires_at_ns <= self.issued_at_ns:
            reject_invalid_state("gate-event lifetime is invalid")
        if any(not event_id for event_id in self.predecessor_event_ids):
            reject_invalid_state("gate-event predecessors must be non-empty")

    def signing_frame(self) -> dict[str, object]:
        """Return the exact event fields covered by the coordinator MAC."""
        payload = asdict(self)
        payload.pop("signature")
        payload["schema"] = _EVENT_SCHEMA
        return payload

    def digest(self) -> str:
        """Return the immutable digest of the complete signed record."""
        return _digest({"schema": _EVENT_SCHEMA, **asdict(self)})


@dataclass(frozen=True, slots=True)
class ActivationIssuerRecord:
    """Signed record proving that an activation token was authority-issued."""

    token_subject_digest: str
    event_record_digests: tuple[str, ...]
    issued_at_ns: int
    expires_at_ns: int
    nonce: str
    key_id: str
    signature: str

    def __post_init__(self) -> None:
        """Validate issuer-record fields."""
        object.__setattr__(
            self, "event_record_digests", tuple(self.event_record_digests)
        )
        _require_digest(self.token_subject_digest, "token_subject_digest")
        for digest in self.event_record_digests:
            _require_digest(digest, "event_record_digest")
        for name in ("nonce", "key_id"):
            _require_text(getattr(self, name), name)
        _require_digest(self.signature, "signature")
        if self.issued_at_ns < 0 or self.expires_at_ns <= self.issued_at_ns:
            reject_invalid_state("issuer-record lifetime is invalid")

    def signing_frame(self) -> dict[str, object]:
        """Return the exact issuer fields covered by the coordinator MAC."""
        payload = asdict(self)
        payload.pop("signature")
        payload["schema"] = _ISSUER_SCHEMA
        return payload

    def digest(self) -> str:
        """Return the immutable digest named by ``ActivationToken``."""
        return _digest({"schema": _ISSUER_SCHEMA, **asdict(self)})


class LocalActivationAuthority:
    """Owner-only local issuer and verifier for qualified Stack tokens."""

    def __init__(self, root: Path, *, key: bytes, key_id: str) -> None:
        """Bind a validated authority root to its coordinator key."""
        self.root = root
        self._key = key
        self.key_id = key_id

    @classmethod
    def initialize(cls, root: str | Path) -> Self:
        """Create or open an owner-only authority root.

        Parameters
        ----------
        root : path-like
            Dedicated authority directory.  It is created with mode ``0700``;
            the coordinator key is created once with mode ``0600``.

        Returns
        -------
        LocalActivationAuthority
            Validated authority bound to the local owner and key.

        """
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
        for child in (path / "events", path / "issuer_records"):
            child.mkdir(exist_ok=True, mode=0o700)
            child.chmod(0o700)
        key_path = path / "coordinator.key"
        if not key_path.exists():
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                key = secrets.token_bytes(32)
                os.write(descriptor, key)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            cls._fsync_directory(path)
        return cls.open(path)

    @classmethod
    def open(cls, root: str | Path) -> Self:
        """Open and validate an existing authority root."""
        path = Path(root)
        cls._validate_owned_directory(path)
        key_path = path / "coordinator.key"
        cls._validate_owned_regular_file(key_path, required_mode=0o600)
        key = key_path.read_bytes()
        if len(key) != 32:
            reject_invalid_state("activation authority key length is invalid")
        key_id = hashlib.sha256(key).hexdigest()
        return cls(path, key=key, key_id=key_id)

    def issue_gate_event(
        self,
        *,
        event_id: str,
        gate_id: GateId,
        producer_commit: str,
        predecessor_event_ids: tuple[str, ...],
        provider_receipt_digest: str,
        parent_manifest_digest: str,
        evidence_digest: str,
        activation_mode: Literal["qualified", "reference"],
        lifetime_seconds: int = 86_400,
    ) -> StackGateEvent:
        """Sign and durably publish one gate event."""
        if lifetime_seconds <= 0:
            reject_invalid_state("gate-event lifetime must be positive")
        self._verify_predecessor_set(gate_id, predecessor_event_ids)
        now = time.time_ns()
        unsigned = StackGateEvent(
            event_id=event_id,
            gate_id=gate_id,
            producer_commit=producer_commit,
            predecessor_event_ids=predecessor_event_ids,
            provider_receipt_digest=provider_receipt_digest,
            parent_manifest_digest=parent_manifest_digest,
            evidence_digest=evidence_digest,
            activation_mode=activation_mode,
            issued_at_ns=now,
            expires_at_ns=now + lifetime_seconds * 1_000_000_000,
            key_id=self.key_id,
            signature=_ZERO_DIGEST,
        )
        event = replace(unsigned, signature=self._sign(unsigned.signing_frame()))
        self._write_record(self.root / "events" / _record_name(event_id), asdict(event))
        return event

    def verify_gate_lineage(
        self,
        event_id: str,
        *,
        expected_gate_id: GateId,
    ) -> StackGateEvent:
        """Verify one event and its complete typed predecessor chain."""
        event = self.verify_gate_event(
            event_id,
            expected_gate_id=expected_gate_id,
        )
        self._verify_predecessor_set(event.gate_id, event.predecessor_event_ids)
        return event

    def verify_gate_event(
        self,
        event_id: str,
        *,
        expected_gate_id: GateId | None = None,
    ) -> StackGateEvent:
        """Load and authenticate one durable gate event."""
        record = self._read_record(self.root / "events" / _record_name(event_id))
        try:
            event = StackGateEvent(**record)
        except TypeError as error:
            reject_invalid_state(f"gate-event schema is invalid: {error}")
        if event.event_id != event_id:
            reject_invalid_state("gate-event identity does not match its lookup key")
        if expected_gate_id is not None and event.gate_id != expected_gate_id:
            reject_invalid_state("gate-event type does not match activation lineage")
        if event.key_id != self.key_id:
            reject_invalid_state("gate-event signing key does not match authority")
        if event.expires_at_ns <= time.time_ns():
            reject_invalid_state("gate event has expired")
        self._verify_signature(event.signing_frame(), event.signature, "gate event")
        return event

    def issue_token(
        self,
        template: ActivationToken,
        *,
        lifetime_seconds: int = 86_400,
    ) -> ActivationToken:
        """Issue a qualified token after verifying its complete gate lineage.

        ``template.issuer_record_digest`` must be the all-zero placeholder.  A
        caller-created token with any other issuer digest is never adopted.
        """
        if template.mode != "qualified":
            reject_invalid_state(
                "production activation authority issues qualified tokens only"
            )
        if template.issuer_record_digest != _ZERO_DIGEST:
            reject_invalid_state("activation token template already claims an issuer")
        if lifetime_seconds <= 0:
            reject_invalid_state("activation token lifetime must be positive")
        events = self._verify_token_events(template)
        now = time.time_ns()
        unsigned = ActivationIssuerRecord(
            token_subject_digest=self._token_subject_digest(template),
            event_record_digests=tuple(event.digest() for event in events),
            issued_at_ns=now,
            expires_at_ns=now + lifetime_seconds * 1_000_000_000,
            nonce=secrets.token_hex(16),
            key_id=self.key_id,
            signature=_ZERO_DIGEST,
        )
        record = replace(unsigned, signature=self._sign(unsigned.signing_frame()))
        digest = record.digest()
        self._write_record(
            self.root / "issuer_records" / f"{digest}.json",
            asdict(record),
        )
        return replace(template, issuer_record_digest=digest)

    def verify_token(self, token: ActivationToken) -> ActivationIssuerRecord:
        """Authenticate a qualified token and every referenced gate event."""
        if token.mode != "qualified":
            reject_invalid_state("qualified Stack requires a qualified token")
        _require_digest(token.issuer_record_digest, "issuer_record_digest")
        raw = self._read_record(
            self.root / "issuer_records" / f"{token.issuer_record_digest}.json"
        )
        try:
            record = ActivationIssuerRecord(**raw)
        except TypeError as error:
            reject_invalid_state(f"issuer-record schema is invalid: {error}")
        if record.digest() != token.issuer_record_digest:
            reject_invalid_state("activation issuer record digest is invalid")
        if record.key_id != self.key_id:
            reject_invalid_state("activation issuer key does not match authority")
        if record.expires_at_ns <= time.time_ns():
            reject_invalid_state("activation issuer record has expired")
        if record.token_subject_digest != self._token_subject_digest(token):
            reject_invalid_state(
                "activation token subject does not match issuer record"
            )
        self._verify_signature(
            record.signing_frame(), record.signature, "issuer record"
        )
        events = self._verify_token_events(token)
        if record.event_record_digests != tuple(event.digest() for event in events):
            reject_invalid_state(
                "activation event collection does not match issuer record"
            )
        return record

    def _verify_token_events(
        self, token: ActivationToken
    ) -> tuple[StackGateEvent, ...]:
        if token.p18_stack_gate_event_id is None:
            reject_invalid_state("qualified token is missing the P18 Stack gate event")
        p18_event = self.verify_gate_lineage(
            token.p18_stack_gate_event_id,
            expected_gate_id="P18-stack-qualified",
        )
        if p18_event.activation_mode != "qualified":
            reject_invalid_state("P18 Stack gate is not qualified")
        events = [p18_event]
        for event_id in token.p19_qualified_event_ids:
            event = self.verify_gate_lineage(
                event_id,
                expected_gate_id="P19-stack-qualified-activation",
            )
            if token.p18_stack_gate_event_id not in event.predecessor_event_ids:
                reject_invalid_state(
                    "P19 activation event omits the P18 Stack predecessor"
                )
            events.append(event)
        for event in events:
            if event.provider_receipt_digest != token.provider_receipt_digest:
                reject_invalid_state("activation event receipt does not match token")
            if event.parent_manifest_digest != token.parent_manifest_digest:
                reject_invalid_state("activation event parent does not match token")
            if event.evidence_digest != token.qualification_evidence_digest:
                reject_invalid_state("activation event evidence does not match token")
        return tuple(events)

    def _verify_predecessor_set(
        self,
        gate_id: GateId,
        predecessor_event_ids: tuple[str, ...],
    ) -> None:
        required_gate = _REQUIRED_PREDECESSOR[gate_id]
        if required_gate is None:
            if predecessor_event_ids:
                reject_invalid_state(
                    "P18 provider qualification cannot claim predecessor events"
                )
            return
        if not predecessor_event_ids:
            reject_invalid_state(f"{gate_id} is missing its required predecessor")
        matched = False
        for predecessor_id in predecessor_event_ids:
            predecessor = self.verify_gate_event(predecessor_id)
            if predecessor.gate_id == required_gate:
                self._verify_predecessor_set(
                    predecessor.gate_id,
                    predecessor.predecessor_event_ids,
                )
                matched = True
        if not matched:
            reject_invalid_state(
                f"{gate_id} has no {required_gate} predecessor event"
            )

    @staticmethod
    def _token_subject_digest(token: ActivationToken) -> str:
        payload = asdict(token)
        payload.pop("issuer_record_digest")
        payload["schema_domain"] = "stack_activation_token_subject.v1"
        return _digest(payload)

    def _sign(self, frame: object) -> str:
        return hmac.new(self._key, _canonical_json(frame), hashlib.sha256).hexdigest()

    def _verify_signature(
        self, frame: object, signature: str, record_name: str
    ) -> None:
        expected = self._sign(frame)
        if not hmac.compare_digest(expected, signature):
            reject_invalid_state(f"activation {record_name} signature is invalid")

    @classmethod
    def _validate_owned_directory(cls, path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as error:
            reject_invalid_state(f"activation authority root is unavailable: {error}")
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            reject_invalid_state("activation authority root must be a real directory")
        if metadata.st_uid != os.getuid():
            reject_invalid_state("activation authority root owner is invalid")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            reject_invalid_state("activation authority root must have mode 0700")

    @classmethod
    def _validate_owned_regular_file(
        cls,
        path: Path,
        *,
        required_mode: int | None = None,
    ) -> None:
        try:
            metadata = path.lstat()
        except OSError as error:
            reject_invalid_state(f"activation record is unavailable: {error}")
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            reject_invalid_state("activation record must be a regular file")
        if metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
            reject_invalid_state("activation record owner or link count is invalid")
        mode = stat.S_IMODE(metadata.st_mode)
        if required_mode is not None and mode != required_mode:
            reject_invalid_state("activation record permissions are invalid")
        if metadata.st_size > _MAX_RECORD_BYTES:
            reject_invalid_state("activation record exceeds the size limit")

    @classmethod
    def _read_record(cls, path: Path) -> dict[str, object]:
        cls._validate_owned_regular_file(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            reject_invalid_state(f"activation record cannot be decoded: {error}")
        if not isinstance(value, dict):
            reject_invalid_state("activation record must contain a JSON object")
        return value

    @classmethod
    def _write_record(cls, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        payload = _canonical_json(value) + b"\n"
        if len(payload) > _MAX_RECORD_BYTES:
            reject_invalid_state("activation record exceeds the size limit")
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary.replace(path)
        cls._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["ActivationIssuerRecord", "LocalActivationAuthority", "StackGateEvent"]

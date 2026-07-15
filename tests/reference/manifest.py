"""Typed loader for the frozen scientific reference manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManifestFormatError(RuntimeError):
    """Raised when YAML does not satisfy the frozen manifest shape."""

    path: Path
    detail: str = "invalid manifest shape"

    def __str__(self) -> str:
        """Return the manifest-specific parse failure."""
        return f"invalid reference manifest: {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class DownloadableRegime:
    """One checksummed Sentinel-1 oracle regime."""

    identifier: str
    url: str
    filename: str
    sha256: str
    size_bytes: int
    terrain: str
    coherence: str
    baseline: str
    burst_boundary: bool
    deformation: str
    source: str
    access_policy: str
    processor: str
    processor_version: str
    config_digest: str
    artifact_kind: str


@dataclass(frozen=True, slots=True)
class LocalScene:
    """One exact local Sentinel-1 scene identity."""

    identifier: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class LocalPair:
    """One execution-tier scene pair."""

    identifier: str
    reference: str
    secondary: str


@dataclass(frozen=True, slots=True)
class LocalExecution:
    """Local execution tier without copied multi-gigabyte data."""

    root_hint: str
    scenes: tuple[LocalScene, ...]
    pairs: tuple[LocalPair, ...]


@dataclass(frozen=True, slots=True)
class CommittedFixture:
    """Tiny fixture stored directly in the repository."""

    identifier: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ReferenceManifest:
    """Frozen corpus manifest used by reference tests."""

    schema_version: int
    committed_fixture: CommittedFixture
    downloadable_regimes: tuple[DownloadableRegime, ...]
    local_execution: LocalExecution
    metric_gates: dict[str, float]


def load_manifest(path: Path) -> ReferenceManifest:
    """Parse a frozen corpus manifest into immutable values.

    Parameters
    ----------
    path
        YAML manifest path.

    Returns
    -------
    ReferenceManifest
        Parsed manifest.

    """
    try:
        payload = yaml.safe_load(path.read_text())
        fixture = CommittedFixture(**payload["committed_fixture"])
        regimes = tuple(
            DownloadableRegime(**regime) for regime in payload["downloadable_regimes"]
        )
        local_payload = payload["local_execution"]
        local_execution = LocalExecution(
            root_hint=local_payload["root_hint"],
            scenes=tuple(LocalScene(**scene) for scene in local_payload["scenes"]),
            pairs=tuple(LocalPair(**pair) for pair in local_payload["pairs"]),
        )
        manifest = ReferenceManifest(
            schema_version=payload["schema_version"],
            committed_fixture=fixture,
            downloadable_regimes=regimes,
            local_execution=local_execution,
            metric_gates=payload["metric_gates"],
        )
    except (KeyError, TypeError, yaml.YAMLError):
        logger.exception("Invalid reference manifest: %s", path)
        raise ManifestFormatError(path) from None

    regime_ids = [regime.identifier for regime in manifest.downloadable_regimes]
    scene_ids = [scene.identifier for scene in manifest.local_execution.scenes]
    pair_ids = [pair.identifier for pair in manifest.local_execution.pairs]
    validation_error: str | None = None
    if len(regime_ids) != len(set(regime_ids)):
        validation_error = "duplicate regime identifier"
    elif len(scene_ids) != len(set(scene_ids)):
        validation_error = "duplicate scene identifier"
    elif len(pair_ids) != len(set(pair_ids)):
        validation_error = "duplicate pair identifier"
    else:
        known_scenes = set(scene_ids)
        for pair in manifest.local_execution.pairs:
            if pair.reference not in known_scenes or pair.secondary not in known_scenes:
                validation_error = f"pair {pair.identifier} references unknown scene"
                break
    if validation_error is not None:
        logger.error("Invalid reference manifest: %s: %s", path, validation_error)
        raise ManifestFormatError(path, validation_error)
    return manifest

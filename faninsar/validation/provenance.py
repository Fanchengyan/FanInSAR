"""Validation boundary for the frozen radar/geo rebuild manifest."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import yaml

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_CLASSES = {
    "clean-room input",
    "primary specification",
    "behavior oracle",
    "quarantined",
}


@dataclass(slots=True)
class ManifestValidationError(RuntimeError):
    """Raised when a rebuild manifest does not satisfy its pinned contract."""

    path: Path
    detail: str

    def __str__(self) -> str:
        """Return a stable manifest-validation diagnostic."""
        return f"invalid rebuild manifest: {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ManifestValidationSummary:
    """Counts and identity returned after a manifest has been verified."""

    manifest_path: Path
    verified_file_count: int
    scene_count: int
    pair_count: int
    orbit_count: int
    dem_tile_count: int
    primary_processors: tuple[str, ...]
    out_of_scope_hashes_match: bool


def _fail(path: Path, detail: str) -> NoReturn:
    logger.error("rebuild manifest rejected: %s: %s", path, detail)
    raise ManifestValidationError(path, detail)


def _mapping(value: object, path: Path, detail: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        _fail(path, detail)
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail(path, f"{detail} keys must be strings")
        result[key] = item
    return result


def _items(value: object, path: Path, detail: str) -> list[object]:
    if not isinstance(value, list):
        _fail(path, detail)
    result: list[object] = []
    result.extend(item for item in value)
    return result


def _text(mapping: Mapping[str, object], key: str, path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        _fail(path, f"{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str, path: Path) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, f"{key} must be a non-negative integer")
    return value


def _digest(mapping: Mapping[str, object], key: str, path: Path) -> str:
    value = _text(mapping, key, path)
    if _SHA256.fullmatch(value) is None:
        _fail(path, f"{key} must be 64 lowercase hex characters")
    return value


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pin(value: object, root: Path, manifest: Path, detail: str) -> int:
    pin = _mapping(value, manifest, detail)
    raw_path = _text(pin, "path", manifest)
    expected = _text(pin, "sha256", manifest)
    if _SHA256.fullmatch(expected) is None:
        _fail(manifest, f"{detail} sha256 must be 64 lowercase hex characters")
    target = _resolve(root, raw_path)
    if not target.is_file():
        _fail(manifest, f"{detail} file is missing: {target}")
    expected_size = _integer(pin, "size_bytes", manifest)
    actual_size = target.stat().st_size
    if actual_size != expected_size:
        _fail(manifest, f"{detail} size mismatch: {target}")
    actual = _sha256(target)
    if actual != expected:
        _fail(manifest, f"{detail} sha256 mismatch: {target}")
    return 1


def _validate_corpus(
    root: Path, manifest: Path, corpus: Mapping[str, object]
) -> tuple[int, int, int, int, int]:
    scenes = _items(corpus.get("scenes"), manifest, "corpus.scenes")
    scene_ids: set[str] = set()
    verified = 0
    for index, value in enumerate(scenes):
        item = _mapping(value, manifest, f"scene[{index}]")
        scene_id = _text(item, "id", manifest)
        if scene_id in scene_ids:
            _fail(manifest, f"duplicate scene ID: {scene_id}")
        scene_ids.add(scene_id)
        verified += _verify_pin(item, root, manifest, f"scene[{scene_id}]")
    pairs = _items(corpus.get("pairs"), manifest, "corpus.pairs")
    pair_ids: set[str] = set()
    for index, value in enumerate(pairs):
        item = _mapping(value, manifest, f"pair[{index}]")
        pair_id = _text(item, "id", manifest)
        if pair_id in pair_ids:
            _fail(manifest, f"duplicate pair ID: {pair_id}")
        pair_ids.add(pair_id)
        if (
            _text(item, "reference", manifest) not in scene_ids
            or _text(item, "secondary", manifest) not in scene_ids
        ):
            _fail(manifest, f"pair references unknown scene: {pair_id}")
    orbits = _items(corpus.get("orbits"), manifest, "corpus.orbits")
    for index, value in enumerate(orbits):
        verified += _verify_pin(value, root, manifest, f"orbit[{index}]")
    dem = _mapping(corpus.get("dem"), manifest, "corpus.dem")
    for field in ("vertical_datum", "geometry_datum", "geoid_model"):
        _text(dem, field, manifest)
    tiles = _items(dem.get("tiles"), manifest, "corpus.dem.tiles")
    for index, value in enumerate(tiles):
        verified += _verify_pin(value, root, manifest, f"dem[{index}]")
    return verified, len(scenes), len(pairs), len(orbits), len(tiles)


def validate_pipeline_rebuild_manifest(
    path: Path,
    repository_root: Path | None = None,
) -> ManifestValidationSummary:
    """Load and verify every frozen input, oracle, environment, and baseline pin.

    Parameters
    ----------
    path
        YAML manifest to load.
    repository_root
        Optional checkout root used for repository-relative paths.

    Returns
    -------
    ManifestValidationSummary
        Counts of verified corpus and oracle files.

    Raises
    ------
    ManifestValidationError
        If YAML shape, source classification, checksum, baseline, or oracle
        identity validation fails.

    """
    from ._manifest_checks import (
        repository_root as find_repository_root,
    )
    from ._manifest_checks import (
        validate_oracles,
        verify_baseline,
    )

    root = repository_root or find_repository_root(path)
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        _fail(path, f"cannot load YAML: {error}")
    document = _mapping(payload, path, "manifest root")
    classes = {
        _text({"value": item}, "value", path)
        for item in _items(document.get("source_classes"), path, "source_classes")
    }
    if classes != _REQUIRED_CLASSES:
        _fail(path, "source_classes must contain the four required classifications")
    baseline = _mapping(document.get("baseline"), path, "baseline")
    verify_baseline(root, path, baseline)
    environment = _mapping(document.get("environment"), path, "environment")
    validation = _mapping(document.get("validation"), path, "validation")
    for field in _items(
        validation.get("required_environment_fields"),
        path,
        "required_environment_fields",
    ):
        name = _text({"value": field}, "value", path)
        if not isinstance(environment.get(name), dict):
            _fail(path, f"required environment field missing: {name}")
    specification_verified = 0
    for index, item in enumerate(
        _items(document.get("specifications"), path, "specifications")
    ):
        spec = _mapping(item, path, "specification")
        if _text(spec, "classification", path) != "primary specification":
            _fail(path, "specification classification is not primary specification")
        if "path" in spec:
            specification_verified += _verify_pin(
                spec, root, path, f"specification[{index}]"
            )
            continue
        note = _text(spec, "waymark_note", path)
        if not note.startswith("NOTE-"):
            _fail(path, "retired specification must reference a Waymark Note")
        _digest(spec, "source_sha256", path)
        _integer(spec, "source_size_bytes", path)
        _digest(spec, "retirement_inventory_sha256", path)
        specification_verified += 1
    corpus = _mapping(document.get("corpus"), path, "corpus")
    if _text(corpus, "source_classification", path) != "clean-room input":
        _fail(path, "corpus must be classified as clean-room input")
    verified, scenes, pairs, orbits, tiles = _validate_corpus(root, path, corpus)
    oracles = _mapping(document.get("oracles"), path, "oracles")
    if _text(oracles, "source_classification", path) != "behavior oracle":
        _fail(path, "oracles must be classified as behavior oracle")
    fixed_path = _text(validation, "fixed_isce2_primary_path", path)
    oracle_verified, processors = validate_oracles(root, path, oracles, fixed_path)
    required_processors = tuple(
        _text({"value": item}, "value", path)
        for item in _items(
            validation.get("required_primary_processors"),
            path,
            "required_primary_processors",
        )
    )
    if processors != required_processors:
        _fail(path, "primary processors do not match required_primary_processors")
    expected = {
        "required_scene_count": scenes,
        "required_pair_count": pairs,
        "required_orbit_count": orbits,
        "required_dem_tile_count": tiles,
    }
    for key, actual in expected.items():
        if _integer(validation, key, path) != actual:
            _fail(path, f"{key} does not match manifest entries")
    return ManifestValidationSummary(
        manifest_path=path,
        verified_file_count=specification_verified + verified + oracle_verified,
        scene_count=scenes,
        pair_count=pairs,
        orbit_count=orbits,
        dem_tile_count=tiles,
        primary_processors=processors,
        out_of_scope_hashes_match=True,
    )


__all__ = [
    "ManifestValidationError",
    "ManifestValidationSummary",
    "validate_pipeline_rebuild_manifest",
]

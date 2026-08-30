"""Declarative mask plans for Stack stages (PROPOSAL-0040).

This module is deliberately a small configuration boundary.  A plan records
mask recipes and references, but never opens a raster, reads a vector file, or
contacts a water provider.  Materialization belongs to the stage consumer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NoReturn, Self, TypeAlias

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from faninsar.processing.masking.mask import Mask

StageName = Literal["roi", "interferogram", "unwrap", "ionosphere"]
MaskKind = Literal["raster", "vector", "water"]
STAGES: tuple[StageName, ...] = ("roi", "interferogram", "unwrap", "ionosphere")
MASK_KINDS = frozenset({"raster", "vector", "water"})
_STAGE_SET = frozenset(STAGES)
_LEGACY_KEYS = frozenset(
    {
        "mask",
        "mask_source",
        "mask_on_failure",
        "mask_apply_ionosphere",
        "water_mask",
        "auto_water_mask",
        "failure_policy",
        "on_failure",
    }
)


def _fail(message: str, error_type: type[Exception] = ValueError) -> NoReturn:
    """Log and raise a configuration error."""
    logger.error(message)
    raise error_type(message)


@dataclass(frozen=True, slots=True)
class MaskDefinition:
    """One inert mask recipe in a :class:`MaskPlan` registry.

    Parameters
    ----------
    name : str
        Registry name.
    kind : {"raster", "vector", "water"}
        Recipe representation.
    path : pathlib.Path, optional
        Source path for raster and vector recipes.
    provider : str, optional
        Water provider selection.  Provider failures are intentionally not
        configurable and are always raised by the eventual consumer.

    """

    name: str
    kind: MaskKind
    path: Path | None = None
    provider: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize one recipe without materializing it."""
        if not isinstance(self.name, str) or not self.name.strip():
            _fail("mask definition names must be non-empty strings")
        if self.name != self.name.strip() or any(char.isspace() for char in self.name):
            _fail(f"invalid mask definition name {self.name!r}")
        if self.kind not in MASK_KINDS:
            _fail(f"unknown mask kind {self.kind!r}; expected raster, vector, or water")
        if self.kind in {"raster", "vector"}:
            if self.path is None:
                _fail(f"{self.kind} mask {self.name!r} requires 'path'")
            if self.provider is not None:
                _fail(f"{self.kind} mask {self.name!r} does not accept 'provider'")
            object.__setattr__(self, "path", Path(self.path))
        else:
            if self.path is not None:
                _fail("water masks do not accept 'path'")
            if self.provider is None:
                object.__setattr__(self, "provider", "auto")
            elif not isinstance(self.provider, str) or not self.provider.strip():
                _fail("water mask provider must be a non-empty string")
            elif self.provider != self.provider.strip():
                _fail("water mask provider must not contain surrounding whitespace")

    @classmethod
    def from_mapping(
        cls,
        name: str,
        value: Mapping[str, object],
        *,
        base_dir: Path | None = None,
    ) -> Self:
        """Build a recipe from a strict mapping.

        Parameters
        ----------
        name : str
            Registry name.
        value : mapping
            Recipe mapping containing ``kind`` and kind-specific fields.
        base_dir : pathlib.Path, optional
            Directory used to resolve relative source paths.

        """
        if not isinstance(value, Mapping):
            _fail(f"mask definition {name!r} must be a mapping")
        keys = set(value)
        if "kind" not in keys:
            _fail(f"mask definition {name!r} requires 'kind'")
        kind = value["kind"]
        if not isinstance(kind, str) or kind not in MASK_KINDS:
            _fail(f"unknown mask kind {kind!r}; expected raster, vector, or water")
        allowed = (
            {"kind", "path"}
            if kind in {"raster", "vector"}
            else {
                "kind",
                "provider",
            }
        )
        unknown = keys - allowed
        if unknown:
            _fail(
                f"unknown keys in mask definition {name!r}: "
                + ", ".join(sorted(map(str, unknown)))
            )
        path_value = value.get("path")
        path: Path | None = None
        if path_value is not None:
            if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
                _fail(f"mask definition {name!r} path must be a non-empty path")
            path = Path(path_value)
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            path = path.resolve()
        provider = value.get("provider")
        if provider is not None and not isinstance(provider, str):
            _fail(f"water mask {name!r} provider must be a string")
        return cls(name=name, kind=kind, path=path, provider=provider)

    @property
    def identity_payload(self) -> dict[str, str]:
        """Return canonical, JSON-compatible identity fields."""
        payload: dict[str, str] = {"kind": self.kind}
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.provider is not None:
            payload["provider"] = self.provider
        return payload


MaskValue: TypeAlias = "MaskDefinition | Mask | Mapping[str, object]"


def _is_mask(value: object) -> bool:
    """Return whether ``value`` is a concrete masking algebra object."""
    from faninsar.processing.masking.mask import Mask

    return isinstance(value, Mask)


def _identity_payload(value: MaskDefinition | Mask) -> dict[str, str]:
    """Return a deterministic identity payload for a registry value."""
    if isinstance(value, MaskDefinition):
        return value.identity_payload
    return {"kind": "materialized", "identity": value.identity}


def _unique_names(values: Iterable[object], *, stage: str) -> tuple[str, ...]:
    """Validate and canonicalize stage references."""
    if isinstance(values, (str, bytes)):
        _fail(f"stage {stage!r} references must be a sequence of mask names")
    try:
        names = tuple(values)
    except TypeError:
        _fail(f"stage {stage!r} references must be a sequence of mask names")
    if any(not isinstance(value, str) or not value.strip() for value in names):
        _fail(f"stage {stage!r} references must contain non-empty strings")
    if any(value != value.strip() for value in names):
        _fail(f"stage {stage!r} references must not contain surrounding whitespace")
    return tuple(sorted(set(names)))


class MaskPlan:
    """Immutable registry and explicit stage references.

    A plan has no implicit water mask.  Use :meth:`water` when that behavior
    is wanted, or define a water recipe and reference it explicitly.

    Parameters
    ----------
    masks : mapping, optional
        Registry values as concrete :class:`~faninsar.processing.masking.mask.Mask`
        instances, :class:`MaskDefinition` instances, or strict recipe mappings.
    stages : mapping, optional
        Explicit references for the four supported stage names.
    base_dir : pathlib.Path, optional
        Base directory for paths in mapping recipes.

    """

    __slots__ = ("_identity", "_masks", "_stages")

    def __init__(
        self,
        masks: Mapping[str, MaskValue] | None = None,
        stages: Mapping[str, Sequence[str]] | None = None,
        *,
        base_dir: str | Path | None = None,
    ) -> None:
        """Create and validate an immutable plan."""
        base = None if base_dir is None else Path(base_dir).resolve()
        if masks is None:
            masks = {}
        if not isinstance(masks, Mapping):
            _fail("mask registry must be a mapping")
        normalized: dict[str, MaskDefinition | Mask] = {}
        for name, value in masks.items():
            if not isinstance(name, str) or not name.strip():
                _fail("mask registry names must be non-empty strings")
            if name in normalized:
                _fail(f"duplicate mask definition {name!r}")
            if isinstance(value, MaskDefinition):
                definition = value
                if definition.name != name:
                    _fail(
                        f"mask definition name mismatch: registry key {name!r}, "
                        f"definition name {definition.name!r}"
                    )
            elif _is_mask(value):
                # Concrete masks are already normalized, immutable public
                # values.  Keep the instance so ``for_stage`` returns exactly
                # the object supplied by Python callers.
                definition = value
            else:
                definition = MaskDefinition.from_mapping(name, value, base_dir=base)
            normalized[name] = definition

        stage_values: dict[StageName, tuple[str, ...]] = dict.fromkeys(STAGES, ())
        if stages is not None:
            if not isinstance(stages, Mapping):
                _fail("stages must be a mapping")
            unknown_stages = set(stages) - _STAGE_SET
            if unknown_stages:
                _fail(
                    "unknown stage(s): " + ", ".join(sorted(map(str, unknown_stages)))
                )
            for stage, refs in stages.items():
                if isinstance(refs, Mapping):
                    if set(refs) != {"masks"}:
                        _fail(f"unknown keys in stage {stage!r}; expected 'masks'")
                    stage_refs_value = refs["masks"]
                else:
                    stage_refs_value = refs
                stage_values[stage] = _unique_names(stage_refs_value, stage=stage)
        missing = {
            ref
            for refs in stage_values.values()
            for ref in refs
            if ref not in normalized
        }
        if missing:
            _fail("stage references undefined mask(s): " + ", ".join(sorted(missing)))
        self._masks = MappingProxyType(normalized)
        self._stages = MappingProxyType(stage_values)
        self._identity = self._make_identity()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, base_dir: str | Path | None = None
    ) -> Self:
        """Normalize a Python mapping using the strict plan schema."""
        if not isinstance(value, Mapping):
            _fail("MaskPlan configuration must be a mapping")
        legacy = _LEGACY_KEYS & set(value)
        if legacy:
            _fail(
                "legacy mask configuration is not supported: "
                + ", ".join(sorted(legacy))
            )
        unknown = set(value) - {"masks", "stages"}
        if unknown:
            _fail("unknown MaskPlan keys: " + ", ".join(sorted(map(str, unknown))))
        masks = value.get("masks", {})
        if isinstance(masks, list):
            entries: dict[str, object] = {}
            for entry in masks:
                if (
                    not isinstance(entry, Mapping)
                    or set(entry)
                    - {
                        "name",
                        "kind",
                        "path",
                        "provider",
                    }
                    or "name" not in entry
                ):
                    _fail(
                        "mask list entries require name, kind, and kind-specific fields"
                    )
                name = entry["name"]
                if not isinstance(name, str):
                    _fail("mask definition name must be a string")
                if name in entries:
                    _fail(f"duplicate mask definition {name!r}")
                entries[name] = {key: entry[key] for key in entry if key != "name"}
            masks = entries
        if not isinstance(masks, Mapping):
            _fail("'masks' must be a mapping")
        stages = value.get("stages", {})
        return cls(masks, stages, base_dir=base_dir)

    from_python = from_mapping
    normalize = from_mapping

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Read and normalize a strict YAML plan.

        Relative raster/vector paths are resolved against the YAML file's
        parent directory.  Duplicate YAML mapping keys are rejected.
        """
        config_path = Path(path).resolve()
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - optional dependency
            _fail("PyYAML is required to load a MaskPlan YAML file", ImportError)
            raise AssertionError from error

        class UniqueLoader(yaml.SafeLoader):
            """Safe loader that rejects duplicate mapping keys."""

        def construct_mapping(loader: object, node: object, deep: bool = False) -> dict:
            result: dict[object, object] = {}
            for key_node, value_node in node.value:  # type: ignore[attr-defined]
                key = loader.construct_object(  # type: ignore[attr-defined]
                    key_node, deep=deep
                )
                if key in result:
                    _fail(f"duplicate YAML key {key!r}")
                result[key] = loader.construct_object(  # type: ignore[attr-defined]
                    value_node, deep=deep
                )
            return result

        UniqueLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
        )
        with config_path.open(encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=UniqueLoader)
        if not isinstance(data, Mapping):
            _fail("MaskPlan YAML root must be a mapping")
        if "masks" in data and not isinstance(data["masks"], Mapping):
            _fail("'masks' in MaskPlan YAML must be a mapping")
        return cls.from_mapping(data, base_dir=config_path.parent)

    @classmethod
    def water(
        cls,
        *,
        provider: str = "auto",
        stages: Sequence[StageName] = ("roi", "interferogram", "unwrap"),
    ) -> Self:
        """Return the explicit opt-in water preset.

        The ionosphere stage is intentionally not selected by this preset;
        callers must opt into that stage explicitly.
        """
        selected = tuple(stages)
        return cls(
            {"water": MaskDefinition("water", "water", provider=provider)},
            dict.fromkeys(selected, ("water",)),
        )

    @property
    def masks(self) -> Mapping[str, MaskDefinition | Mask]:
        """Return the immutable mask registry."""
        return self._masks

    @property
    def stages(self) -> Mapping[StageName, tuple[str, ...]]:
        """Return immutable explicit stage references."""
        return self._stages

    @property
    def identity(self) -> str:
        """Return the normalized SHA-256 identity of used stage masks."""
        return self._identity

    @property
    def cache_key(self) -> str:
        """Return the cache key for this normalized plan."""
        return self._identity

    def __repr__(self) -> str:
        """Return a concise representation of the normalized plan."""
        return f"MaskPlan(masks={dict(self._masks)!r}, stages={dict(self._stages)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare normalized plans by their identities."""
        if not isinstance(other, MaskPlan):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        """Hash a normalized plan identity."""
        return hash(self.identity)

    def _make_identity(self) -> str:
        used = sorted({ref for refs in self._stages.values() for ref in refs})
        payload = {
            "masks": {name: _identity_payload(self._masks[name]) for name in used},
            "stages": {stage: list(self._stages[stage]) for stage in STAGES},
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def references(self, stage: StageName) -> tuple[str, ...]:
        """Return canonical references for one stage."""
        self._validate_stage(stage)
        return self._stages[stage]

    stage_refs = references

    def for_stage(self, stage: StageName) -> tuple[MaskDefinition | Mask, ...]:
        """Return definitions or concrete masks referenced by one stage."""
        return tuple(self._masks[name] for name in self.references(stage))

    masks_for = for_stage

    def with_stage_masks(self, stage: StageName, refs: Sequence[str]) -> Self:
        """Return a copy with one stage's references replaced."""
        self._validate_stage(stage)
        stages = dict(self._stages)
        stages[stage] = refs
        return type(self)(self._masks, stages)

    set_stage_masks = with_stage_masks

    def with_mask(
        self,
        name: str,
        definition: MaskValue,
        *,
        stages: Sequence[StageName] = (),
    ) -> Self:
        """Return a copy with a registry entry and optional stage refs added."""
        if isinstance(definition, Mapping):
            definition = MaskDefinition.from_mapping(name, definition)
        if isinstance(definition, MaskDefinition) and definition.name != name:
            _fail(f"mask definition name mismatch for {name!r}")
        if not isinstance(definition, (MaskDefinition,)) and not _is_mask(definition):
            _fail(f"mask {name!r} must be a MaskDefinition, Mask, or recipe mapping")
        masks = dict(self._masks)
        masks[name] = definition
        stage_map = dict(self._stages)
        for stage in stages:
            self._validate_stage(stage)
            stage_map[stage] = tuple(sorted({*stage_map[stage], name}))
        return type(self)(masks, stage_map)

    def without_mask(self, name: str) -> Self:
        """Return a copy without a registry entry or its stage references."""
        if name not in self._masks:
            _fail(f"unknown mask definition {name!r}")
        masks = dict(self._masks)
        del masks[name]
        stages = {
            stage: tuple(ref for ref in refs if ref != name)
            for stage, refs in self._stages.items()
        }
        return type(self)(masks, stages)

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in _STAGE_SET:
            _fail(f"unknown stage {stage!r}; expected {', '.join(STAGES)}")


__all__ = [
    "MASK_KINDS",
    "STAGES",
    "MaskDefinition",
    "MaskKind",
    "MaskPlan",
    "StageName",
]

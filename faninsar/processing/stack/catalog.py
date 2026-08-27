"""Scene catalog for logical acquisitions (PROPOSAL-0037)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = setup_logger(__name__)


def scene_id_from_path(path: str | Path) -> str:
    """Extract a YYYYMMDD-style scene id from a source path when possible."""
    stem = Path(path).stem.replace(".SAFE", "")
    for part in stem.split("_"):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
    return stem


@dataclass(frozen=True, slots=True)
class SceneCatalog:
    """Immutable map from acquisition date to ordered source paths.

    A logical acquisition can contain multiple compatible source segments.
    Grouping remains a source concern; pair-network discovery remains a Stack
    adapter concern.
    """

    paths: Mapping[str, tuple[Path, ...]]

    def __post_init__(self) -> None:
        """Normalize direct mappings and reject duplicate source paths."""
        normalized: dict[str, tuple[Path, ...]] = {}
        seen: set[str] = set()
        for raw_date, raw_paths in self.paths.items():
            scene_id = _normalize_scene_id(raw_date)
            source_paths = (
                (Path(raw_paths),)
                if isinstance(raw_paths, (str, Path))
                else tuple(Path(path) for path in raw_paths)
            )
            if not source_paths:
                reject_invalid_state(f"catalog date {scene_id} has no source paths")
            ordered = tuple(sorted(source_paths, key=_canonical_path_key))
            for path in ordered:
                key = _duplicate_path_key(path)
                if key in seen:
                    reject_invalid_state(f"duplicate source path {path}")
                seen.add(key)
            normalized[scene_id] = ordered
        if not normalized:
            reject_invalid_state("catalog requires at least one scene path")
        object.__setattr__(self, "paths", normalized)

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path]) -> Self:
        """Build a catalog, grouping compatible source segments by date."""
        mapping: dict[str, list[Path]] = {}
        seen: set[str] = set()
        for raw in paths:
            path = Path(raw)
            scene_id = scene_id_from_path(path)
            key = _duplicate_path_key(path)
            if key in seen:
                reject_invalid_state(f"duplicate source path {path}")
            seen.add(key)
            mapping.setdefault(scene_id, []).append(path)
        logger.info("Scene catalog: %s scenes", len(mapping))
        return cls(
            paths={scene_id: tuple(values) for scene_id, values in mapping.items()}
        )

    @property
    def dates(self) -> tuple[str, ...]:
        """Return sorted logical acquisition date IDs."""
        return tuple(sorted(self.paths))

    def paths_for(self, date_like: object) -> tuple[Path, ...]:
        """Return all source paths for one logical acquisition."""
        scene_id = _normalize_scene_id(date_like)
        if scene_id not in self.paths:
            reject_invalid_state(f"unknown scene id {scene_id}")
        return self.paths[scene_id]

    def path_for(self, date_like: object) -> Path:
        """Return a singleton path, rejecting multi-segment acquisitions."""
        paths = self.paths_for(date_like)
        if len(paths) != 1:
            reject_invalid_state(
                "acquisition "
                f"{_normalize_scene_id(date_like)} has multiple source paths"
            )
        return paths[0]

    def __len__(self) -> int:
        """Return the number of logical acquisitions."""
        return len(self.paths)


def _canonical_path_key(path: Path) -> str:
    """Return a platform-independent ordering key for a source path."""
    return path.as_posix().replace("\\", "/")


def _duplicate_path_key(path: Path) -> str:
    """Return a normalized key used to detect repeated source paths."""
    return path.resolve(strict=False).as_posix()


def _normalize_scene_id(date_like: object) -> str:
    """Normalize a date-like value to a ``YYYYMMDD`` scene ID."""
    if isinstance(date_like, str):
        value = date_like.strip()
        if len(value) == 8 and value.isdigit():
            return value
        if value:
            try:
                return pd.Timestamp(value).strftime("%Y%m%d")
            except (TypeError, ValueError, OverflowError):
                return value
    if isinstance(date_like, (date, datetime, pd.Timestamp)):
        timestamp = pd.Timestamp(date_like)
    else:
        try:
            timestamp = pd.Timestamp(date_like)
        except (TypeError, ValueError, OverflowError):
            reject_invalid_state(f"invalid acquisition date {date_like!r}")
    if pd.isna(timestamp):
        reject_invalid_state(f"invalid acquisition date {date_like!r}")
    return timestamp.strftime("%Y%m%d")

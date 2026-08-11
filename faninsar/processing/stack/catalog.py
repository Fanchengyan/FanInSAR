"""Scene catalog: date id → product path (PROPOSAL-0017)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


def scene_id_from_path(path: str | Path) -> str:
    """Extract a YYYYMMDD-style scene id from a SAFE path when possible."""
    stem = Path(path).stem.replace(".SAFE", "")
    for part in stem.split("_"):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
    return stem


@dataclass(frozen=True, slots=True)
class SceneCatalog:
    """Immutable map from acquisition date id to on-disk product path."""

    paths: Mapping[str, Path]
    """Keys are ``YYYYMMDD`` scene ids."""

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path] | tuple[str | Path, ...],
    ) -> SceneCatalog:
        """Build a catalog from SAFE paths; reject duplicate date ids."""
        mapping: dict[str, Path] = {}
        for raw in paths:
            path = Path(raw)
            sid = scene_id_from_path(path)
            if sid in mapping:
                reject_invalid_state(f"duplicate scene id {sid} in catalog")
            mapping[sid] = path
        if not mapping:
            reject_invalid_state("catalog requires at least one scene path")
        logger.info("Scene catalog: %s scenes", len(mapping))
        return cls(paths=mapping)

    @property
    def dates(self) -> tuple[str, ...]:
        """Sorted date ids."""
        return tuple(sorted(self.paths))

    def path_for(self, date_id: str) -> Path:
        """Return path for a date id or raise."""
        if date_id not in self.paths:
            reject_invalid_state(f"unknown scene id {date_id}")
        return self.paths[date_id]

    def __len__(self) -> int:
        """Return the number of scenes."""
        return len(self.paths)

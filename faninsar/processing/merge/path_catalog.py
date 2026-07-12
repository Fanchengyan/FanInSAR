"""Burst inventory grouped by relative orbit + look direction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from faninsar.processing.merge.products import BurstGeoProduct

logger = setup_logger(__name__)

__all__ = ["PathCatalog"]


@dataclass
class PathCatalog:
    """Catalog of burst products grouped by ``path_id``.

    Attributes
    ----------
    paths : dict of str -> list of BurstGeoProduct
        Mapping from path identifier (e.g. ``"T100_A"``) to its bursts.

    """

    paths: dict[str, list[BurstGeoProduct]]

    @classmethod
    def from_products(
        cls,
        products: Iterable[BurstGeoProduct],
    ) -> PathCatalog:
        """Build a catalog from an iterable of burst products.

        Parameters
        ----------
        products : iterable of BurstGeoProduct
            Products to group.

        Returns
        -------
        PathCatalog

        """
        paths: dict[str, list[BurstGeoProduct]] = {}
        for p in products:
            paths.setdefault(p.path_id, []).append(p)
        return cls(paths=paths)

    def same_path(self, path_id: str) -> list[BurstGeoProduct]:
        """Return all products belonging to one path."""
        return list(self.paths.get(path_id, ()))

    def by_look_direction(self, look: str) -> list[BurstGeoProduct]:
        """Return all products with a given look direction."""
        return [
            p
            for plist in self.paths.values()
            for p in plist
            if p.look_direction == look
        ]

    def cross_path_pairs(self) -> list[tuple[str, str]]:
        """Return path-id pairs that may share a footprint (all combinations)."""
        ids = list(self.paths)
        return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]

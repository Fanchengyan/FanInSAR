"""Concrete local metadata mission for shared Stack contracts only."""
from __future__ import annotations

from pathlib import Path

from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.preparation import PreparedInputs
from faninsar.stack.search import MissionPreparedInputs
from faninsar.stack.session import Stack


from faninsar.missions.s1 import S1Stack
from faninsar.missions import base as mission_registry


class LocalStack(S1Stack):
    """Qualify explicit fixture paths without impersonating mission I/O."""

    _sensor_registry_key = None

    def _track_metadata(self):
        return self.config.extra.get("track_metadata")

    def _ensure_precise_orbits(self):
        self._ensure_prepared()

    def _prepare_inputs(self, context):
        catalog = SceneCatalog.from_paths(context.request["paths"])
        return PreparedInputs(sources=catalog.paths, mission=MissionPreparedInputs(selected_footprints=tuple(self.config.extra.get("selected_footprints", ()))))

    @classmethod
    def from_paths(cls, paths, **kwargs):
        stack = cls._from_safes(paths, **kwargs)
        stack.scene_provider = cls._default_scene_provider()
        catalog = SceneCatalog.from_paths(paths)
        # Fixture identity tests need resolved topology before publishing products.
        from faninsar.stack.preparation import PreparedStackData
        from faninsar.stack.session import _pairs_from_factory
        from faninsar import Pairs
        options = stack.request["topology"]
        stack._inputs_state = PreparedStackData(catalog, Pairs.from_names(list(options["pairs"])) if options["pairs"] is not None else _pairs_from_factory(catalog.dates), Pairs.from_names(list(options["misreg_pairs"])) if options["misreg_pairs"] is not None else _pairs_from_factory(catalog.dates), options["reference"] or catalog.dates[0])
        return stack


mission_registry._MISSION_STACKS["LocalStack"] = (__name__, "LocalStack")

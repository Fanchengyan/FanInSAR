"""Read-only scene burst observation collection (PROPOSAL-0084)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.io.storage.scene_store import CoregisteredSceneStore
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)


def _track_for_date(stack: Stack, date_id: str) -> str | None:
    """Return the track label of one date when catalog metadata has it."""
    infos = stack._catalog_track_infos()
    info = infos.get(date_id)
    if info is not None:
        track = getattr(info, "track_key", None) or getattr(info, "group_key", None)
        if track:
            return str(track)
    groups = stack.track_groups
    if len(groups) == 1:
        return groups[0]
    return None


def _official_burst_id(unit: object) -> int | None:
    """Return a persisted official absolute burst ID, never an ANX identity."""
    value = getattr(unit, "absolute_burst_id", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class SceneBurstObservation:
    """One committed scene-unit observation in a parent scene collection view.

    Parameters
    ----------
    date : str
        Canonical acquisition date id (``YYYYMMDD``).
    burst : str
        Product-local unit tag, the same spelling as ``scene.sel(bursts=)``.
    track : str or None
        Track label when catalog metadata provides one.
    absolute_burst_id : int or None
        Optional official ESA absolute burst identifier. Missing metadata is
        ``None``; the internal ``faninsar:burst_id`` is never substituted.

    """

    date: str
    burst: str
    track: str | None = None
    absolute_burst_id: int | None = None
    _collection: SceneBurstCollection | None = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot this observation's source footprint with the P0084 labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes.
        **kwargs
            Forwarded to the plotting renderer.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing this observation's polygon and labels.

        """
        from faninsar.plotting.burst_footprints import plot_burst_footprints

        if self._collection is None:
            from faninsar.logging import reject

            reject(
                "burst observation is not bound to a collection",
                ValueError,
                logger=logger,
            )
        frame = self._collection.footprints
        mask = (frame["date"] == self.date) & (frame["burst"] == self.burst)
        return plot_burst_footprints(frame.loc[mask], ax=ax, **kwargs)


class SceneBurstCollection(Sequence[SceneBurstObservation]):
    """Read-only sequence of per-acquisition scene burst observations.

    Constructing the collection enumerates committed scene units. It does not
    open source annotation, read SLC arrays, prepare inputs, or plot.
    """

    def __init__(
        self,
        stack: Stack,
        observations: tuple[SceneBurstObservation, ...],
        *,
        stores: dict[str, CoregisteredSceneStore] | None = None,
    ) -> None:
        """Bind observations to the parent Stack view.

        Parameters
        ----------
        stack : Stack
            Stack whose committed scenes back the view.
        observations : tuple of SceneBurstObservation
            Date-then-tag ordered handles for the parent selection.
        stores : dict, optional
            Date-keyed scene stores used later for footprint resolution.

        """
        self._stack = stack
        self._observations = observations
        self._stores = stores or {}
        self._footprints_cache: Any = None

    def __len__(self) -> int:
        """Return the number of selected burst observations."""
        return len(self._observations)

    def __iter__(self) -> Iterator[SceneBurstObservation]:
        """Iterate observations in date then unit-tag order."""
        return iter(self._observations)

    def __getitem__(self, index: int) -> SceneBurstObservation:  # type: ignore[override]
        """Return the observation at *index*."""
        return self._observations[index]

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot selected source footprints with the P0084 two-line labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes.
        **kwargs
            Forwarded to the plotting renderer.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing polygons and labels.

        """
        from faninsar.plotting.burst_footprints import plot_burst_footprints

        return plot_burst_footprints(self.footprints, ax=ax, **kwargs)

    def __repr__(self) -> str:
        """Return a one-line summary of this collection."""
        return f"SceneBurstCollection({len(self)} observations)"

    def _resolve_footprints(self) -> Any:
        """Resolve footprints through the mission Sensor hook."""
        from faninsar.logging import reject
        from faninsar.missions.base import get_mission

        key = getattr(self._stack, "_sensor_registry_key", None)
        if not key:
            reject(
                f"{type(self._stack).__name__} does not support scene burst inspection",
                NotImplementedError,
                logger=logger,
            )
        sensor = get_mission(str(key))
        resolver = getattr(sensor, "resolve_scene_burst_footprints", None)
        if resolver is None:
            name = getattr(sensor, "name", type(sensor).__name__)
            reject(
                f"{name} does not support scene burst inspection",
                NotImplementedError,
                logger=logger,
            )
        return resolver(self._stack, self._observations)

    @property
    def footprints(self) -> Any:
        """Return a caller-owned EPSG:4326 GeoDataFrame of source footprints.

        Access is lazy: constructing the collection does not open annotation.
        Each access returns an independent frame; caller edits do not affect
        later reads. Missing local geometry stays a null-geometry row.

        Returns
        -------
        geopandas.GeoDataFrame
            Columns ``date``, ``track``, ``burst``, ``absolute_burst_id``,
            ``geometry`` with CRS EPSG:4326.

        Raises
        ------
        NotImplementedError
            If the Stack's mission does not provide a footprint hook.

        """
        if self._footprints_cache is None:
            self._footprints_cache = self._resolve_footprints()
        return self._footprints_cache.copy(deep=True)


def collect_scene_bursts(
    stack: Stack,
    stores: dict[str, CoregisteredSceneStore],
    dates: Sequence[str],
) -> SceneBurstCollection:
    """Build a read-only observation collection from committed scene units.

    Parameters
    ----------
    stack : Stack
        Parent Stack session.
    stores : dict
        Date-keyed committed scene stores in the parent view.
    dates : sequence of str
        Selected date ids in view order.

    Returns
    -------
    SceneBurstCollection
        Handles for every committed unit of *dates*.

    """
    collection = SceneBurstCollection(stack, (), stores=dict(stores))
    observations: list[SceneBurstObservation] = []
    for date_id in dates:
        store = stores[date_id]
        track = _track_for_date(stack, date_id)
        observations.extend(
            SceneBurstObservation(
                date=str(date_id),
                burst=str(unit.tag),
                track=track,
                absolute_burst_id=_official_burst_id(unit),
                _collection=collection,
            )
            for unit in store.units
        )
    collection._observations = tuple(observations)
    return collection

"""Composable Sentinel-1 acquisition result set (PROPOSAL-0062).

:class:`S1Search` is an immutable wrapper over the ``list[CatalogItem]``
returned by provider searches.  It adds set algebra (``+`` union, ``-``
difference, ``&`` intersection) and chainable client-side filters (time,
date, satellite, orbit direction, arbitrary predicate) so a deliberately
loose first-pass query can be refined before any bytes are downloaded.  A
filtered ``S1Search`` feeds :meth:`S1Stack.from_search` directly.

``CatalogItem.acquisition`` already carries the fields each filter reads:
``start``/``end`` datetimes, ``platform``, ``orbit``, the frame ``geometry``
footprint, and ``properties`` (incl. ``sat:orbit_state``).  Provider adapters
normalize differently — CDSE emits ``platform="sentinel-1d"`` and keeps
``start``/``end`` on ``raw_metadata`` — so field readers fall back across
``acquisition`` and ``raw_metadata``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.data.query import BoundingBox, Points, Polygons
    from faninsar.remote.records import CatalogItem

logger = setup_logger(__name__)

_PLATFORMS = frozenset({"S1A", "S1B", "S1C", "S1D"})
_DIRECTIONS = frozenset({"ASCENDING", "DESCENDING"})
_DATE_FMTS = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d")


def _utc(value: Any) -> datetime | None:
    """Normalize a date-like value to a timezone-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_date(value: Any) -> datetime:
    """Parse an acquisition-date selector to a tz-aware UTC datetime.

    Accepts ``datetime``, or ``YYYYMMDD`` / ``YYYY-MM-DD`` / ``YYYY/MM/DD``
    strings.  Raises ``ValueError`` on an unparseable value.
    """
    if isinstance(value, datetime):
        return _utc(value) or value.replace(tzinfo=UTC)
    text = str(value).strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    message = f"unparseable acquisition date: {value!r}"
    raise ValueError(message)


def _item_id(item: CatalogItem) -> str:
    return str(getattr(item, "item_id", ""))


def _raw(item: CatalogItem) -> Mapping[str, Any]:
    return getattr(item, "raw_metadata", {}) or {}


def _platform(item: CatalogItem) -> str:
    """Return the short satellite id ``S1A``/``B``/``C``/``D``, or ``""``."""
    for source in (
        getattr(item.acquisition, "platform", None),
        _raw(item).get("platform"),
        _item_id(item)[:3],
    ):
        text = str(source or "").upper().replace("-", "")
        if text.startswith("SENTINEL"):
            text = "S1" + text[-1]
        if text in _PLATFORMS:
            return text
    return ""


def _start(item: CatalogItem) -> datetime | None:
    """Return the acquisition start time across adapter layouts."""
    start = getattr(item.acquisition, "start", None)
    if start is None:
        start = _raw(item).get("start")
    return _utc(start)


def _date_key(item: CatalogItem) -> str:
    start = _start(item)
    return start.strftime("%Y%m%d") if start else ""


def _direction(item: CatalogItem) -> str:
    """Return ``ASCENDING``/``DESCENDING`` across adapter layouts, or ``""``."""
    props = getattr(item.acquisition, "properties", {}) or {}
    raw = _raw(item)
    nested = raw.get("properties")
    candidates = (
        props.get("sat:orbit_state"),
        props.get("orbitDirection"),
        raw.get("sat:orbit_state"),
        raw.get("orbitDirection"),
        (nested.get("sat:orbit_state") or nested.get("orbitDirection"))
        if isinstance(nested, Mapping)
        else None,
    )
    for source in candidates:
        text = str(source or "").upper()
        if text in _DIRECTIONS:
            return text
    return ""


@dataclass(frozen=True)
class S1Search:
    """Immutable, composable set of Sentinel-1 catalog acquisitions.

    Each result is a provider :class:`~faninsar.remote.records.CatalogItem`
    carrying an :class:`~faninsar.remote.records.AcquisitionMetadata`.  Every
    operation returns a new ``S1Search``, so filters chain and set algebra
    composes.  Items dedup on ``item_id``.

    Attributes
    ----------
    items : tuple of CatalogItem
        Selected acquisitions in provider-return order.
    catalog : str or None
        Registered catalog name that produced the items (``sentinel1-cdse``,
        ``sentinel1-asf``); used by ``from_search`` to pick the provider path.
    spatial : object or None
        The spatial query that produced the items.
    query : mapping
        The first-pass query context (datetime_range, flight_direction,
        filters) for provenance.

    """

    items: tuple[CatalogItem, ...] = ()
    catalog: str | None = None
    spatial: object | None = None
    query: Mapping[str, Any] = field(default_factory=dict)

    # -- sequence protocol ------------------------------------------------
    def __iter__(self) -> Any:
        """Iterate the selected acquisitions in provider order."""
        return iter(self.items)

    def __len__(self) -> int:
        """Return the number of selected acquisitions."""
        return len(self.items)

    def __getitem__(self, index: int) -> CatalogItem:
        """Return the ``index``-th acquisition."""
        return self.items[index]

    def _wrap(self, items: Iterable[CatalogItem]) -> S1Search:
        return S1Search(
            tuple(items),
            catalog=self.catalog,
            spatial=self.spatial,
            query=dict(self.query),
        )

    # -- set algebra ------------------------------------------------------
    def __add__(self, other: S1Search) -> S1Search:
        """Union: items in either set, deduplicated by ``item_id``."""
        seen = {_item_id(i) for i in self.items}
        merged = list(self.items) + [i for i in other.items if _item_id(i) not in seen]
        return self._wrap(merged)

    def __sub__(self, other: S1Search) -> S1Search:
        """Difference: items in ``self`` not in ``other``."""
        drop = {_item_id(i) for i in other.items}
        return self._wrap([i for i in self.items if _item_id(i) not in drop])

    def __and__(self, other: S1Search) -> S1Search:
        """Intersection: items present in both sets."""
        keep = {_item_id(i) for i in other.items}
        return self._wrap([i for i in self.items if _item_id(i) in keep])

    # -- chainable filters ------------------------------------------------
    def filter(self, predicate: Callable[[CatalogItem], bool]) -> S1Search:
        """Keep items for which ``predicate`` returns true."""
        return self._wrap([i for i in self.items if predicate(i)])

    def between(self, lo: Any, hi: Any) -> S1Search:
        """Keep acquisitions whose start falls in ``[lo, hi]``."""
        lo_dt, hi_dt = _parse_date(lo), _parse_date(hi)
        return self.filter(
            lambda i: _start(i) is not None and lo_dt <= _start(i) <= hi_dt
        )

    def dates(self, *dates: Any) -> S1Search:
        """Keep acquisitions on the listed dates."""
        keep = {_parse_date(d).strftime("%Y%m%d") for d in dates}
        return self.filter(lambda i: _date_key(i) in keep)

    def exclude_dates(self, *dates: Any) -> S1Search:
        """Drop acquisitions on the listed dates."""
        drop = {_parse_date(d).strftime("%Y%m%d") for d in dates}
        return self.filter(lambda i: _date_key(i) not in drop)

    def platforms(self, *platforms: str) -> S1Search:
        """Keep the listed satellites (``"S1A"`` … ``"S1D"``)."""
        keep = {p.upper() for p in platforms}
        return self.filter(lambda i: _platform(i) in keep)

    def direction(self, direction: str) -> S1Search:
        """Keep one orbit direction (``"ASCENDING"``/``"DESCENDING"``)."""
        want = direction.upper()
        return self.filter(lambda i: _direction(i) == want)

    def platforms_present(self) -> dict[str, list[str]]:
        """Return ``{platform: sorted acquisition dates}`` for inspection."""
        out: dict[str, set[str]] = {}
        for item in self.items:
            out.setdefault(_platform(item), set()).add(_date_key(item))
        return {key: sorted(values) for key, values in sorted(out.items())}

    # -- visualization ----------------------------------------------------
    def plot(
        self,
        roi: BoundingBox | Polygons | None = None,
        ax: Any = None,
    ) -> Any:
        """Draw frame footprints and the ROI outline, colored by platform.

        Frame-level footprints come from ``acquisition.geometry`` (GeoJSON).
        Returns the matplotlib Axes.
        """
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import box, shape

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        geoms: list[Any] = []
        labels: list[str] = []
        for item in self.items:
            geometry = getattr(item.acquisition, "geometry", None)
            if not geometry:
                continue
            try:
                geoms.append(shape(geometry))
                labels.append(_platform(item) or "?")
            except Exception:  # skip malformed footprint
                continue
        if geoms:
            gdf = gpd.GeoDataFrame(
                {"platform": labels}, geometry=geoms, crs="EPSG:4326"
            )
            for platform, subset in gdf.groupby("platform"):
                subset.boundary.plot(ax=ax, label=platform, linewidth=1.4)
            ax.legend(title="platform")
        if roi is not None:
            try:
                if hasattr(roi, "left"):
                    roi_geometry = box(roi.left, roi.bottom, roi.right, roi.top)
                else:
                    roi_geometry = shape(getattr(roi, "geometry", roi))
                gpd.GeoSeries([roi_geometry], crs="EPSG:4326").boundary.plot(
                    ax=ax,
                    color="black",
                    linewidth=2,
                    linestyle="--",
                    label="roi",
                )
            except Exception:  # ROI overlay is advisory
                pass
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"S1Search ({len(self.items)} items)")
        return ax


def search_acquisitions(
    spatial: BoundingBox | Points | Polygons,
    *,
    catalog: str = "sentinel1-cdse",
    collections: Sequence[str] | None = None,
    datetime_range: tuple[datetime, datetime] | None = None,
    flight_direction: str | None = None,
    filters: Mapping[str, Any] | None = None,
    auth_profile: str | None = None,
    limit: int = 100,
    budget: Any | None = None,
) -> S1Search:
    """Search a registered catalog and return a composable ``S1Search``.

    This is the provider-neutral first pass: the spatial + coarse
    ``datetime_range`` bounds go to the server (cheap traversal), while exact
    dates / satellites / orbit direction are applied client-side on the
    returned set.  ``catalog`` selects the provider — ``sentinel1-cdse`` for
    CDSE range downloads, ``sentinel1-asf`` for ASF burst/scene products.
    """
    from faninsar import remote

    if catalog == "sentinel1-cdse":
        from faninsar.remote.providers.cdse import ensure_cdse_catalog

        ensure_cdse_catalog()
    if budget is None and limit > 100:
        budget = remote.RemoteResourceBudget(max_items=limit)
    items = remote.search(
        spatial,
        catalog=catalog,
        collections=collections,
        datetime_range=datetime_range,
        flight_direction=flight_direction,
        filters=filters,
        auth_profile=auth_profile,
        limit=limit,
        budget=budget,
    )
    return S1Search(
        tuple(items),
        catalog=catalog,
        spatial=spatial,
        query={
            "datetime_range": datetime_range,
            "flight_direction": flight_direction,
            "filters": dict(filters or {}),
            "collections": tuple(collections) if collections is not None else None,
        },
    )


__all__ = ["S1Search", "search_acquisitions"]

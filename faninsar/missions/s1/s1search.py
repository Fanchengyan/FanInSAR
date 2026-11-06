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
from typing import TYPE_CHECKING, Any, Literal

from faninsar.logging import reject, setup_logger
from faninsar.remote.errors import RemoteQueryError, _fail

if TYPE_CHECKING:
    from faninsar.data.query import BoundingBox, Points, Polygons
    from faninsar.missions.s1.search import S1PolarizationRequest
    from faninsar.remote.records import CatalogItem, RemoteResourceBudget

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
    props = getattr(item.acquisition, "properties", {}) or {}
    nested = _raw(item).get("properties")
    nested_map = nested if isinstance(nested, Mapping) else {}
    for source in (
        getattr(item.acquisition, "platform", None),
        props.get("mission"),
        props.get("platform"),
        _raw(item).get("platform"),
        nested_map.get("mission"),
        nested_map.get("platform"),
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


def _end(item: CatalogItem) -> datetime | None:
    """Return the acquisition stop time across adapter layouts."""
    end = getattr(item.acquisition, "end", None)
    if end is None:
        end = _raw(item).get("end")
    return _utc(end)


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
        Selected acquisitions.  :func:`search_acquisitions` returns them in
        canonical order (acquisition start, item ID, asset hrefs); filters
        and set algebra preserve the left operand's order.
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
        """Iterate the selected acquisitions in stored order."""
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


def _apply_named(
    merged: dict[str, Any],
    key: str,
    value: Any,
    *,
    name: str,
) -> None:
    """Set one ASF filter from a named parameter, overriding its alias."""
    if key in merged and merged[key] != value:
        logger.warning(
            "named %s=%r overrides advanced filter %s=%r",
            name,
            value,
            key,
            merged[key],
        )
    merged[key] = value


def _compose_query(
    *,
    missions: tuple[str, ...] | None,
    polarizations: tuple[str, ...] | Literal["all"],
    flight_direction: str | None,
    track: int | Sequence[int] | None,
    burst_id: int | Sequence[int] | None,
    filters: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Compose ASF burst filters with one precedence order.

    Precedence is profile defaults, then validated advanced aliases from
    ``filters``, then named values.  A named value (including the documented
    ``polarizations`` default) is authoritative; a named ``None`` for any
    other optional field leaves the advanced alias effective.  The orbit
    direction occupies exactly one slot: the dedicated ``flight_direction``
    argument of :func:`faninsar.remote.search`.

    Returns
    -------
    tuple of (dict, str or None)
        Provider filters and the single effective orbit direction.

    """
    from faninsar.missions.s1.search import mission_platform_filter, product_filters

    # Profile defaults: the ASF burst dataset/processing level only.
    merged = product_filters("burst", "all")
    merged.update(filters)

    alias_direction = merged.pop("flightDirection", None)
    direction = flight_direction
    if direction is None:
        direction = alias_direction
    elif alias_direction is not None and alias_direction != direction:
        logger.warning(
            "named flight_direction=%r overrides advanced filter flightDirection=%r",
            direction,
            alias_direction,
        )

    if missions is not None:
        _apply_named(
            merged, "platform", mission_platform_filter(missions), name="missions"
        )
    if track is not None:
        _apply_named(merged, "relativeOrbit", track, name="track")
    if burst_id is not None:
        _apply_named(merged, "absoluteBurstID", burst_id, name="burst_id")

    if polarizations != "all" and len(polarizations) == 1:
        # ASF burst queries accept one polarization per request.
        _apply_named(merged, "polarization", polarizations[0], name="polarizations")
    elif "polarization" in merged:
        # A multi-channel or 'all' request cannot coexist with a
        # single-channel server filter; channels are selected client-side.
        logger.warning(
            "polarizations=%r removes conflicting advanced filter polarization=%r",
            polarizations,
            merged.pop("polarization"),
        )
    return merged, direction


def search_acquisitions(
    roi: BoundingBox | Points | Polygons,
    *,
    missions: str | Sequence[str] | None = None,
    polarizations: S1PolarizationRequest | Sequence[str] | str = ("VV",),
    collections: Sequence[str] | None = None,
    datetime_range: tuple[Any, Any] | None = None,
    flight_direction: Literal["ASCENDING", "DESCENDING"] | None = None,
    track: int | Sequence[int] | None = None,
    burst_id: int | Sequence[int] | None = None,
    auth_profile: str | None = "earthdata-asf",
    limit: int | None = None,
    budget: RemoteResourceBudget | None = None,
    **filters: Any,
) -> S1Search:
    """Search Sentinel-1 ASF burst records and return a composable ``S1Search``.

    This is the single Sentinel-1 search implementation; ``S1Stack.search``
    is the same function.  The spatial bound, coarse ``datetime_range`` and
    the composed ASF filters go to the server.  Results are then finished
    locally in one canonical order (acquisition start, item ID, sorted asset
    hrefs), multi-channel polarization requests are filtered on catalog
    metadata, and the result is truncated to ``limit``.  The download
    provider is chosen later in :meth:`S1Stack.from_search`; this search
    only selects the bursts.

    Filter precedence is: Sentinel-1 burst profile defaults
    (``dataset="SLC-BURST"``, ``processingLevel="BURST"``), then advanced
    ``**filters`` aliases, then named parameters.  A named value, including
    the documented ``polarizations`` default, is authoritative; a named
    ``None`` leaves the matching advanced alias effective.

    Parameters
    ----------
    roi : BoundingBox, Points, or Polygons
        Search geometry using exact FanInSAR spatial semantics — catalog
        items are kept only when their footprint intersects it.
    missions : str or sequence of str, optional
        Sentinel-1 satellite letters to keep — ``"A"``, ``("A", "C", "D")``,
        ``"ACD"``, or the ``"S1A"``/``"Sentinel-1B"`` spellings (normalized
        to letters).  ``None`` (default) keeps every satellite ASF indexes.
        Translated to the ASF ``platform`` filter.
    polarizations : sequence of str or str, default ``("VV",)``
        Channels to keep — ``("VV",)``, ``"VV"``, ``"VV+VH"``, or ``"all"``.
        Every channel must be requested explicitly with ``"all"``.  A single
        channel becomes the provider-side ``polarization`` filter; a
        multi-channel or ``"all"`` request removes any ``polarization``
        alias and multi-channel results are filtered client-side.
    collections : sequence of str, optional
        ASF collections to restrict the query; ``None`` keeps the registered
        default.
    datetime_range : tuple of (datetime | str), optional
        Inclusive ``(start, end)`` acquisition window.  Each bound accepts a
        ``datetime`` or a ``"YYYY-MM-DD"`` / ``"YYYYMMDD"`` / ``"YYYY/MM/DD"``
        string.  ``None`` searches the full archive.
    flight_direction : {"ASCENDING", "DESCENDING"}, optional
        Orbit direction filter; ``None`` accepts both unless a
        ``flightDirection`` alias is given.
    track : int or sequence of int, optional
        Relative-orbit (track) number(s) to keep — ``117`` or ``[117, 88]``.
        Translated to the ASF ``relativeOrbit`` filter.
    burst_id : int or sequence of int, optional
        ``absoluteBurstID`` value(s) to keep — ``3001`` or ``[3001, 3002]``.
        Translated to the ASF ``absoluteBurstID`` filter.
    auth_profile : str, default ``"earthdata-asf"``
        ASF authentication profile.
    limit : int or None, default None
        Maximum number of matching items returned. None exhausts provider
        pagination. Multi-channel polarization filtering runs before the
        final local truncation to a positive limit.
    budget : RemoteResourceBudget, optional
        Advanced resource guards. An explicit ``max_items`` guards candidate
        traversal and fails if exceeded rather than truncating the query.
    **filters : Any
        Advanced ASF query aliases for fields with no dedicated parameter,
        validated by the ASF adapter.  Accepted keys:

        - ``beamMode``         : "IW"/"EW"/"WV"/"S1".."S6"
        - ``lookDirection``    : "LEFT"/"RIGHT"
        - ``absoluteOrbit``    : absolute orbit number(s)
        - ``relativeBurstID``  : relative burst id within the track
        - ``fullBurstID``      : full burst id string
        - ``groupID`` / ``granule_list`` / ``product_list`` / ``instrument``
        - ``processingLevel`` / ``dataset`` : override the burst defaults
        - ``platform`` / ``flightDirection`` / ``relativeOrbit`` /
          ``absoluteBurstID`` / ``polarization`` : aliases of the dedicated
          parameters above, which win when given

    Returns
    -------
    S1Search
        Composable search result in canonical acquisition order
        (``.items``, ``.platforms()``, ``.dates()``, ``.between()``, set
        algebra); feeds :meth:`S1Stack.from_search` directly.

    Raises
    ------
    ValueError
        If ``polarizations`` is ``None``, a mission/polarization/datetime
        value is not supported, or a multi-channel burst item lacks
        unambiguous polarization metadata.
    RemoteQueryError
        If ``limit``, ``datetime_range``, ``flight_direction``, or a filter
        fails remote query validation.
    RemoteAccessError
        If the ASF catalog or ``auth_profile`` is not registered.

    Examples
    --------
    >>> from faninsar.missions.s1 import S1Stack
    >>> search = S1Stack.search(  # doctest: +SKIP
    ...     roi,
    ...     datetime_range=("2026-01-01", "2026-03-01"),
    ...     polarizations=("VV",),
    ...     track=88,
    ... )
    >>> stack = S1Stack.from_search(search, roi=roi, work_dir="study")  # doctest: +SKIP

    """
    from faninsar import remote
    from faninsar.missions.s1.search import (
        canonical_items,
        ensure_asf_catalog,
        filter_burst_polarizations,
        normalize_missions,
        normalize_polarizations,
    )

    if polarizations is None:
        message = (
            "polarizations=None is not accepted; use the default ('VV',) or "
            "request every channel explicitly with polarizations='all'"
        )
        reject(message, ValueError, logger=logger)
    selected_polarizations = normalize_polarizations(polarizations)
    normalized_missions = normalize_missions(missions)

    catalog = "sentinel1-asf"
    ensure_asf_catalog(catalog, product_type="burst")
    merged, direction = _compose_query(
        missions=normalized_missions,
        polarizations=selected_polarizations,
        flight_direction=flight_direction,
        track=track,
        burst_id=burst_id,
        filters=filters,
    )

    # datetime_range accepts datetimes or date strings.
    parsed_range: tuple[datetime, datetime] | None = None
    if datetime_range is not None:
        lo, hi = datetime_range
        parsed_range = (_parse_date(lo), _parse_date(hi))

    if limit is not None and (type(limit) is not int or limit <= 0):
        _fail(RemoteQueryError, "invalid_limit")
    items = remote.search(
        roi,
        catalog=catalog,
        collections=collections,
        datetime_range=parsed_range,
        flight_direction=direction,
        filters=merged,
        auth_profile=auth_profile,
        limit=None
        if selected_polarizations != "all" and len(selected_polarizations) > 1
        else limit,
        budget=budget,
    )
    # Canonical local finishing precedes the final local truncation.
    ordered = canonical_items(items)
    if selected_polarizations != "all":
        ordered = filter_burst_polarizations(ordered, selected_polarizations)
    return S1Search(
        tuple(ordered[:limit]),
        catalog=catalog,
        spatial=roi,
        query={
            "missions": normalized_missions,
            "polarizations": selected_polarizations,
            "datetime_range": parsed_range,
            "flight_direction": direction,
            "track": track,
            "burst_id": burst_id,
            "filters": dict(filters),
            "collections": tuple(collections) if collections is not None else None,
        },
    )


__all__ = ["S1Search", "search_acquisitions"]

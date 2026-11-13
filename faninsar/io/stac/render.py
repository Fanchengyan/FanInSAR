"""Shared STAC rendering for HTTP responses and portable disk views.

One normalization feeds both presentation targets. ``LinkTargets`` constructs
absolute HTTP URLs or POSIX relative disk hrefs, and the renderers add only
links and asset presentation hrefs; they never change scientific properties.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote

if TYPE_CHECKING:
    from collections.abc import Callable

    from faninsar.io.stac.index import StacCollection, StacIndex
    from faninsar.io.stac.record import StacRecord

__all__ = [
    "LinkTargets",
    "render_collection",
    "render_family_catalog",
    "render_item",
    "render_root_catalog",
]

_JSON_TYPE = "application/json"
_ITEM_TYPE = "application/geo+json"
_STAC_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class LinkTargets:
    """Href factories for one presentation view."""

    root: str
    self: str
    item: Callable[[str], str]
    asset: Callable[[StacRecord, str], str]
    collection: Callable[[str], str]
    family: Callable[[str], str]

    @classmethod
    def for_disk(cls, *, work_dir: Path, view_path: Path) -> Self:
        """Build POSIX relative hrefs for one portable disk view.

        Parameters
        ----------
        work_dir : Path
            Stack workspace root.
        view_path : Path
            Final path of the document being rendered.

        """
        root_dir = Path(work_dir).expanduser().resolve()
        view = Path(view_path)
        if not view.is_absolute():
            view = root_dir / view
        view = view.resolve()
        parent = view.parent

        def item(public_id: str) -> str:
            return _relative_href(
                root_dir / "stac" / "items" / f"{public_id}.json", parent
            )

        def collection(collection_id: str) -> str:
            return _relative_href(
                root_dir / "stac" / "collections" / f"{collection_id}.json", parent
            )

        def family(name: str) -> str:
            return _relative_href(
                root_dir / "stac" / "catalogs" / f"{name}.json", parent
            )

        def asset(record: StacRecord, key: str) -> str:
            try:
                target = Path(record.asset_paths[key])
            except KeyError as error:
                message = f"asset {key!r} is not registered for {record.public_id}"
                raise KeyError(message) from error
            if not target.is_absolute():
                target = root_dir / target
            return _relative_href(target, parent)

        return cls(
            root=_relative_href(root_dir / "catalog.json", parent),
            self=_relative_href(view, parent),
            item=item,
            asset=asset,
            collection=collection,
            family=family,
        )

    @classmethod
    def for_http(cls, *, base_url: str, view_path: str) -> Self:
        """Build absolute HTTP hrefs for one service view.

        Parameters
        ----------
        base_url : str
            Service origin without a trailing slash.
        view_path : str
            URL path of the document being rendered, beginning with ``/``.

        """
        base = base_url.rstrip("/")
        path = view_path if view_path.startswith("/") else f"/{view_path}"

        def item(public_id: str) -> str:
            return f"{base}/items/{quote(public_id, safe='')}"

        def collection(collection_id: str) -> str:
            return f"{base}/collections/{quote(collection_id, safe='')}"

        def family(name: str) -> str:
            return f"{base}/catalogs/{quote(name, safe='')}.json"

        def asset(record: StacRecord, key: str) -> str:
            return (
                f"{base}/assets/{quote(record.public_id, safe='')}/"
                f"{quote(key, safe='')}"
            )

        return cls(
            root=f"{base}/",
            self=f"{base}{path}",
            item=item,
            asset=asset,
            collection=collection,
            family=family,
        )


def render_item(record: StacRecord, targets: LinkTargets) -> dict[str, Any]:
    """Render one normalized Item as a valid STAC Feature."""
    item = record.as_item()
    links = [
        _link("root", targets.root),
        _link("self", targets.item(record.public_id), media_type=_ITEM_TYPE),
        _link("parent", targets.family(record.family)),
    ]
    if record.collection_id is not None:
        item["collection"] = record.collection_id
        links.append(_link("collection", targets.collection(record.collection_id)))
    else:
        item.pop("collection", None)
    assets = item.get("assets")
    if isinstance(assets, dict):
        for key in assets:
            if key in record.asset_paths:
                assets[key]["href"] = targets.asset(record, key)
    item["links"] = links
    return item


def render_root_catalog(index: StacIndex, targets: LinkTargets) -> dict[str, Any]:
    """Render the root Catalog with family and real Collection children."""
    slug = _stack_slug(index.work_dir)
    links: list[dict[str, Any]] = []
    if index.records or index.collections:
        links.append(_link("root", targets.root))
        links.append(_link("self", targets.self))
        links.extend(
            _link("child", targets.family(family), title=family)
            for family in index.families()
        )
        links.extend(
            _link(
                "child",
                targets.collection(collection.collection_id),
                title=collection.collection_id,
            )
            for collection in index.collections
        )
    return {
        "type": "Catalog",
        "stac_version": _STAC_VERSION,
        "id": f"stack-{slug}",
        "title": f"FanInSAR Stack {slug}",
        "description": f"FanInSAR STAC catalog for workspace '{slug}'.",
        "links": links,
    }


def render_family_catalog(
    index: StacIndex,
    family: str,
    targets: LinkTargets,
) -> dict[str, Any]:
    """Render one family Catalog with Item and Collection children."""
    links = [
        _link("root", targets.root),
        _link("self", targets.self),
        _link("parent", targets.root),
    ]
    links.extend(
        _link(
            "item",
            targets.item(record.public_id),
            media_type=_ITEM_TYPE,
            title=record.public_id,
        )
        for record in index.records_for_family(family)
    )
    links.extend(
        _link(
            "child",
            targets.collection(collection.collection_id),
            title=collection.collection_id,
        )
        for collection in index.collections
        if collection.family == family
    )
    return {
        "type": "Catalog",
        "stac_version": _STAC_VERSION,
        "id": family,
        "title": family,
        "description": f"FanInSAR STAC family '{family}'.",
        "links": links,
    }


def render_collection(
    collection: StacCollection,
    index: StacIndex,  # noqa: ARG001 - shared renderer signature
    targets: LinkTargets,
) -> dict[str, Any]:
    """Render one real extent-backed Collection with its member Items."""
    links = [
        _link("root", targets.root),
        _link("self", targets.collection(collection.collection_id)),
        _link("parent", targets.family(collection.family)),
    ]
    links.extend(
        _link("item", targets.item(public_id), media_type=_ITEM_TYPE)
        for public_id in collection.item_ids
    )
    start, end = collection.interval
    return {
        "type": "Collection",
        "stac_version": _STAC_VERSION,
        "id": collection.collection_id,
        "title": collection.collection_id,
        "description": (
            f"Geographic items of the FanInSAR '{collection.family}' family."
        ),
        "license": "proprietary",
        "extent": {
            "spatial": {"bbox": [list(collection.bbox)]},
            "temporal": {
                "interval": [[_datetime_string(start), _datetime_string(end)]]
            },
        },
        "links": links,
    }


def _link(
    rel: str,
    href: str,
    *,
    media_type: str = _JSON_TYPE,
    title: str | None = None,
) -> dict[str, Any]:
    """Return one STAC link object."""
    link: dict[str, Any] = {"rel": rel, "href": href, "type": media_type}
    if title is not None:
        link["title"] = title
    return link


def _relative_href(target: Path, start: Path) -> str:
    """Return a POSIX relative href from ``start`` to ``target``."""
    target_parts = target.resolve().parts
    start_parts = start.resolve().parts
    common = 0
    for left, right in zip(target_parts, start_parts, strict=False):
        if left != right:
            break
        common += 1
    relative = Path(
        *([".."] * (len(start_parts) - common)),
        *target_parts[common:],
    )
    return relative.as_posix() or "."


def _datetime_string(value: datetime) -> str:
    """Return one UTC RFC 3339 timestamp with microsecond precision."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _stack_slug(work_dir: Path) -> str:
    """Return the deterministic lower-case workspace slug for catalog ids."""
    name = Path(work_dir).expanduser().resolve().name
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if len(slug) > 48:
        slug = slug[:48].strip("-")
    if not slug:
        slug = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return slug

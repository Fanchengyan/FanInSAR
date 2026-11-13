"""Private cross-module interface for the derived Stack STAC view (P0089).

This package is imported by the Stack service and CLI; it is not re-exported
from :mod:`faninsar.io`. Normalization owns canonical-product interpretation,
the index owns discovery and assembly, search owns Item Search, render owns
shared presentation, and catalog owns portable persistence.
"""

from __future__ import annotations

from faninsar.io.stac.catalog import (
    CatalogPersistenceReport,
    write_portable_catalog,
)
from faninsar.io.stac.index import (
    IndexExclusion,
    StacCollection,
    StacIndex,
    build_index,
)
from faninsar.io.stac.normalize import normalize_product
from faninsar.io.stac.record import (
    StacNormalizationError,
    StacRecord,
    TemporalCoverage,
    public_item_id,
)
from faninsar.io.stac.render import (
    LinkTargets,
    render_collection,
    render_family_catalog,
    render_item,
    render_root_catalog,
)
from faninsar.io.stac.search import (
    SearchPage,
    SearchRequest,
    StacSearchError,
    StaleSearchTokenError,
    search,
)

__all__ = [
    "CatalogPersistenceReport",
    "IndexExclusion",
    "LinkTargets",
    "SearchPage",
    "SearchRequest",
    "StacCollection",
    "StacIndex",
    "StacNormalizationError",
    "StacRecord",
    "StacSearchError",
    "StaleSearchTokenError",
    "TemporalCoverage",
    "build_index",
    "normalize_product",
    "public_item_id",
    "render_collection",
    "render_family_catalog",
    "render_item",
    "render_root_catalog",
    "search",
    "write_portable_catalog",
]

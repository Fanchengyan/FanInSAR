"""Pluggable processor discovery registry for frame products.

Each processor (HyP3, ISCE, GAMMA, GMTSAR, MintPy, StaMPS, ...) provides a
discoverer that knows how to scan a product tree and locate the directories
holding geometry rasters and interferogram pairs. Discoverers self-register
on import, so adding a new processor is a matter of dropping a new module
into this package — no edits to existing code.

Public API
----------

- :func:`register` — register a discoverer instance (idempotent by ``name``).
- :func:`get` — fetch a registered discoverer by name.
- :func:`available` — list registered discoverer names.
- :func:`discover_geometry_product` — convenience: dispatch by name.

The HyP3 discoverer is registered eagerly when this package is imported:

>>> from faninsar.datasets.frame import discovery
>>> discovery.available()
['hyp3']
>>> discovery.get("hyp3").discover_geometry_product(root_dir)
PosixPath('...')

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)


@runtime_checkable
class GeometryProductDiscoverer(Protocol):
    """Protocol: find a representative product directory holding geometry rasters."""

    name: str

    def discover_geometry_product(self, root_dir: str | Path) -> Path:
        """Return the first product directory containing geometry rasters."""
        ...


@runtime_checkable
class InterferogramPairsDiscoverer(Protocol):
    """Protocol: discover valid interferogram pair product directories."""

    name: str

    def discover_pairs(self, root_dir: str | Path) -> list[Path]:
        """Return a list of product directories, one per valid pair."""
        ...


_REGISTRY: dict[str, object] = {}


def register(discoverer: object) -> None:
    """Register a discoverer instance. Idempotent by ``name`` attribute.

    Re-registering the same name overwrites the previous entry.

    Parameters
    ----------
    discoverer : object
        A discoverer instance with a ``name`` attribute conforming to
        :class:`GeometryProductDiscoverer`.

    """
    name = getattr(discoverer, "name", None)
    if not isinstance(name, str) or not name:
        msg = (
            f"Cannot register discoverer without a string 'name' attribute: "
            f"{discoverer!r}"
        )
        logger.error(msg)
        raise TypeError(msg)
    if name in _REGISTRY:
        logger.debug("Overwriting existing discoverer registration: %s", name)
    _REGISTRY[name] = discoverer
    logger.debug("Registered discoverer: %s", name)


def get(name: str) -> object:
    """Return the registered discoverer for *name*.

    Raises
    ------
    KeyError
        If no discoverer with that name is registered.

    """
    if name not in _REGISTRY:
        msg = f"No discoverer registered for {name!r}. Available: {available()}"
        logger.error(msg)
        raise KeyError(msg)
    return _REGISTRY[name]


def available() -> list[str]:
    """Return a sorted list of registered discoverer names."""
    return sorted(_REGISTRY)


def discover_geometry_product(name: str, root_dir: str | Path) -> Path:
    """Dispatch geometry-product discovery to the named registered discoverer."""
    discoverer = get(name)
    # mypy/Protocol: we trust the registered object conforms.
    return discoverer.discover_geometry_product(root_dir)  # type: ignore[attr-defined]


# --- Eager registration of built-in discoverers ---
# Importing the submodule triggers its module-level register() call.
from . import hyp3 as _hyp3
from .hyp3 import discover_hyp3_geometry_product

__all__ = [
    "GeometryProductDiscoverer",
    "InterferogramPairsDiscoverer",
    "available",
    "discover_geometry_product",
    "discover_hyp3_geometry_product",
    "get",
    "register",
]

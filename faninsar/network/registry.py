"""Explicit, lazily discovered Network reader registry."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from .protocols import NetworkReader

logger = setup_logger(__name__)

ENTRY_POINT_GROUP = "faninsar.network_readers"


class ReaderRegistryError(RuntimeError):
    """Base error raised for invalid Network reader registrations."""


class DuplicateReaderError(ReaderRegistryError):
    """Raised when a reader name has more than one registration."""


class ReaderNotFoundError(ReaderRegistryError):
    """Raised when no reader is registered under a requested name."""


class InvalidReaderError(ReaderRegistryError):
    """Raised when a reader does not implement the one-method protocol."""


def _validate_name(name: object) -> str:
    """Validate and return one stable reader registration name."""
    if not isinstance(name, str) or not name or name != name.strip():
        logger.error("invalid Network reader name: %r", name)
        raise ValueError("reader name must be a non-empty string")  # noqa: EM101, TRY003
    if any(character.isspace() for character in name):
        logger.error("Network reader name contains whitespace: %r", name)
        raise ValueError("reader name must not contain whitespace")  # noqa: EM101, TRY003
    return name


def _validate_reader_class(reader: object) -> type[NetworkReader]:
    """Validate a zero-argument reader class without importing an instance."""
    if not isinstance(reader, type):
        logger.error("Network reader registration is not a class: %r", reader)
        raise InvalidReaderError("registered readers must be classes")  # noqa: EM101, TRY003
    if not callable(getattr(reader, "read", None)):
        logger.error("Network reader class has no callable read method: %r", reader)
        message = "reader class must define a callable read method"
        raise InvalidReaderError(message)
    return reader


def _entry_points() -> tuple[Any, ...]:
    """Return metadata for the versioned reader entry-point group."""
    try:
        discovered = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Python 3.9-style importlib metadata has no ``group`` keyword.
        discovered = metadata.entry_points()
        discovered = discovered.select(group=ENTRY_POINT_GROUP)
    if hasattr(discovered, "select"):
        discovered = discovered.select(group=ENTRY_POINT_GROUP)
    return tuple(discovered)


class ReaderRegistry:
    """Registry for explicitly selected Network reader classes.

    Parameters
    ----------
    entry_point_group : str, optional
        Entry-point group used for lazy installed-reader discovery.  The
        default is FanInSAR's stable public group.

    Notes
    -----
    Entry-point metadata is inspected before any plugin is imported.  A
    duplicate name, including a collision with an in-process registration,
    fails deterministically.  Passing a registry to :meth:`Network.open` is
    intentionally isolated; it never merges this registry with a default one.

    """

    def __init__(self, *, entry_point_group: str = ENTRY_POINT_GROUP) -> None:
        """Create an empty registry whose installed readers remain lazy."""
        self._entry_point_group = _validate_name(entry_point_group)
        self._readers: dict[str, type[NetworkReader]] = {}

    def register(self, name: str, reader: type[NetworkReader]) -> None:
        """Register one zero-argument reader class under *name*.

        Parameters
        ----------
        name : str
            Stable selector used by :func:`Network.open`.
        reader : type[NetworkReader]
            Class instantiated without arguments when selected.

        Raises
        ------
        DuplicateReaderError
            If *name* is already registered in this process.
        InvalidReaderError
            If *reader* is not a class with a callable ``read`` method.
        ValueError
            If *name* is not a stable non-empty selector.

        """
        normalized_name = _validate_name(name)
        reader_class = _validate_reader_class(reader)
        if normalized_name in self._readers:
            logger.error("Network reader name is already registered: %s", name)
            message = f"reader name already registered: {name}"
            raise DuplicateReaderError(message)
        self._readers[normalized_name] = reader_class

    @property
    def names(self) -> tuple[str, ...]:
        """Return names registered directly in this process."""
        return tuple(sorted(self._readers))

    def resolve(self, name: str) -> type[NetworkReader]:
        """Resolve *name*, importing only its selected installed entry point.

        All entry-point metadata is checked first.  This deliberately means a
        duplicate unrelated name also fails before any plugin import, making
        discovery deterministic rather than dependent on enumeration order.
        """
        normalized_name = _validate_name(name)
        candidates = self._metadata_candidates()
        if normalized_name in self._readers:
            return self._readers[normalized_name]
        entry_point = candidates.get(normalized_name)
        if entry_point is None:
            logger.error("unknown Network reader name: %s", normalized_name)
            message = f"no Network reader is registered as {normalized_name!r}"
            raise ReaderNotFoundError(message)
        try:
            loaded = entry_point.load()
        except Exception as error:
            logger.exception("could not import Network reader %s", normalized_name)
            message = f"could not load Network reader {normalized_name!r}"
            raise InvalidReaderError(message) from error
        reader_class = _validate_reader_class(loaded)
        self._readers[normalized_name] = reader_class
        return reader_class

    def _metadata_candidates(self) -> dict[str, Any]:
        """Validate entry-point names before importing a selected plugin."""
        discovered = _entry_points_for_group(self._entry_point_group)
        grouped: dict[str, list[Any]] = {}
        for entry_point in discovered:
            name = _validate_name(getattr(entry_point, "name", None))
            grouped.setdefault(name, []).append(entry_point)
        duplicate_names = {
            name for name, entries in grouped.items() if len(entries) > 1
        }
        duplicate_names.update(set(grouped).intersection(self._readers))
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            logger.error("duplicate Network reader registrations: %s", names)
            message = f"duplicate Network reader names: {names}"
            raise DuplicateReaderError(message)
        return {name: entries[0] for name, entries in grouped.items()}


def _entry_points_for_group(group: str) -> tuple[Any, ...]:
    """Enumerate one entry-point group without importing its plugins."""
    try:
        discovered = metadata.entry_points(group=group)
    except TypeError:
        discovered = metadata.entry_points()
        discovered = discovered.select(group=group)
    # Some test doubles and older metadata objects return all groups even when
    # passed ``group``; filter by metadata without calling ``load``.
    if hasattr(discovered, "select"):
        discovered = discovered.select(group=group)
    return tuple(discovered)


__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateReaderError",
    "InvalidReaderError",
    "ReaderNotFoundError",
    "ReaderRegistry",
    "ReaderRegistryError",
]

"""Path-based public access to a standardized InSAR network.

The on-disk representation of a network is the standardized frame product
already implemented by :mod:`faninsar.datasets.frame`.  ``Network`` is the
public name for that path-based product; it intentionally uses the concrete
Frame implementation rather than introducing a second compatibility proxy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from faninsar.logging import setup_logger

from .frame.frame import Frame

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from os import PathLike


class NetworkConstructionError(Exception):
    """Base error raised when a path cannot construct a :class:`Network`."""


class NetworkPathError(NetworkConstructionError, FileNotFoundError):
    """Raised when the requested network root is missing or not a directory."""


class LegacyNetworkLayoutError(NetworkConstructionError):
    """Raised when a path contains a pre-standardization network layout."""

    def __init__(self, root: Path, markers: tuple[Path, ...]) -> None:
        """Initialize an error describing the legacy paths that were found.

        Parameters
        ----------
        root : pathlib.Path
            Requested network root.
        markers : tuple[pathlib.Path, ...]
            Legacy marker paths found below *root*.

        """
        self.root = root
        self.markers = markers
        marker_text = ", ".join(str(path) for path in markers)
        super().__init__(
            f"Legacy InSAR network layout under {root}: {marker_text}. "
            "Convert the product to the standardized interferograms/ layout "
            "before constructing Network."
        )


# Short aliases make the typed failure categories discoverable without making
# callers depend on the implementation's longer class names.
NetworkLayoutError = LegacyNetworkLayoutError
LegacyLayoutError = LegacyNetworkLayoutError


def _legacy_markers(root: Path) -> tuple[Path, ...]:
    """Return legacy interferogram markers directly under *root*.

    ``Frame`` historically accepted ``ifg/`` and ``ifg_index.json``.  The
    latter can occur either at the product root or inside the interferogram
    collection, so both locations are checked.  This check is deliberately
    shallow: unrelated nested source products must not prevent a standard
    network from being mounted.
    """
    candidates = (
        root / "ifg",
        root / "ifg_index.json",
        root / "interferograms" / "ifg_index.json",
    )
    return tuple(path for path in candidates if path.exists())


class Network(Frame):
    """Concrete path-based view of one standardized InSAR network.

    ``Network`` exposes the geometry, interferogram, and time-series members
    of the existing Dataset-backed :class:`~faninsar.datasets.frame.Frame`.
    Its constructor admits only the standardized product layout and rejects
    legacy ``ifg/`` and ``ifg_index.json`` products before Frame performs any
    dataset discovery.

    Parameters
    ----------
    root : str or os.PathLike or pathlib.Path
        Existing directory containing the standardized ``geometry/`` and/or
        ``interferograms/`` product members.

    Examples
    --------
    >>> network = Network("standardized-frame")
    >>> network.interferograms

    """

    def __init__(self, root: str | PathLike[str]) -> None:
        """Mount a standardized network from *root*.

        Parameters
        ----------
        root : str or os.PathLike or pathlib.Path
            Existing standardized network directory.

        Raises
        ------
        NetworkPathError
            If *root* does not exist or is not a directory.
        LegacyNetworkLayoutError
            If *root* contains a legacy ``ifg/`` or ``ifg_index.json`` marker.
        NetworkConstructionError
            If *root* cannot be interpreted as a filesystem path.

        """
        try:
            resolved_root = Path(root)
        except TypeError as exc:
            msg = f"Network root must be path-like, got {root!r}"
            logger.exception(msg)
            raise NetworkConstructionError(msg) from exc

        if not resolved_root.exists() or not resolved_root.is_dir():
            msg = f"Network directory not found: {resolved_root}"
            logger.error(msg)
            raise NetworkPathError(msg)

        markers = _legacy_markers(resolved_root)
        if markers:
            logger.error(
                "Refusing legacy Network layout at %s; markers=%s",
                resolved_root,
                markers,
            )
            raise LegacyNetworkLayoutError(resolved_root, markers)

        # Frame is the concrete implementation.  Calling super() directly
        # keeps Network a real class while preserving all Dataset behavior.
        super().__init__(resolved_root)

    @classmethod
    def from_path(cls, root: str | PathLike[str]) -> Self:
        """Construct a network from a filesystem path.

        Parameters
        ----------
        root : str or os.PathLike or pathlib.Path
            Existing standardized network directory.

        Returns
        -------
        Network
            Mounted network backed by the Dataset Frame implementation.

        """
        return cls(root)

    def __repr__(self) -> str:
        """Return a concise Network summary."""
        parts = [f"Network(root={self.root!r})"]
        if self.geometry is not None:
            parts.append(f"  geometry: {self.geometry.root}")
        if self.interferograms is not None:
            summary: dict[str, Any] = self.interferograms.summary()
            parts.append(f"  interferograms: {summary['pair_count']} pairs")
        return "\n".join(parts)


__all__ = [
    "LegacyLayoutError",
    "LegacyNetworkLayoutError",
    "Network",
    "NetworkConstructionError",
    "NetworkLayoutError",
    "NetworkPathError",
]

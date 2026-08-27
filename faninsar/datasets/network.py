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


class NetworkAnalysisError(NetworkConstructionError):
    """Base error raised when a Network cannot schedule analysis."""


class IncompleteNetworkProductError(NetworkAnalysisError):
    """Raised when required interferogram products are absent or incomplete."""


# Short aliases make the typed failure categories discoverable without making
# callers depend on the implementation's longer class names.
NetworkLayoutError = LegacyNetworkLayoutError
LegacyLayoutError = LegacyNetworkLayoutError
IncompleteNetworkError = IncompleteNetworkProductError


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
        metadata = self.interferograms.index_metadata if self.interferograms else None
        if metadata is not None:
            product_type = metadata.get("type")
            if isinstance(product_type, str) and product_type.startswith("Frame"):
                marker = self.interferograms.root / "interferograms_index.json"
                message = (
                    "Frame metadata is not a canonical Network product; rebuild "
                    f"the interferogram index at {marker}"
                )
                logger.error(message)
                raise LegacyNetworkLayoutError(resolved_root, (marker,))
        self._product_index = None

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

    @property
    def product_index(self) -> Any:
        """Return the validated logical product index, when registered."""
        return self._product_index

    def register_products(self, products: Any) -> Any:
        """Register one homogeneous set of logical product records.

        Product records contain locators and lineage only; Dataset objects are
        still opened internally by the Network data layer.  Registration
        rejects duplicate keys or mixed analysis cohorts before solving.

        Parameters
        ----------
        products : iterable of NetworkProduct
            Logical product records to register.

        Returns
        -------
        NetworkProductIndex
            The validated index retained by this Network.

        """
        from faninsar.core.network import NetworkProductIndex

        try:
            index = NetworkProductIndex(tuple(products)).homogeneous()
        except (TypeError, ValueError) as exc:
            message = f"Network product registration rejected: {exc}"
            logger.exception(message)
            raise NetworkConstructionError(message) from exc
        self._product_index = index
        return index

    def analyze_time_series(
        self,
        *,
        solver: str = "sbas",
        model: Any | None = None,
        pairs: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run time-series analysis over the Network's committed products.

        Dataset discovery remains internal to :class:`Network`; callers pass
        solver options, not raster objects.  The method consumes the existing
        ``InterferogramStack`` seam and fails before solver construction when
        no complete unwrapped product set is available.

        Parameters
        ----------
        solver : {"sbas", "nsbas"}, default="sbas"
            Time-series solver family.
        model : object, optional
            Optional NSBAS temporal model.
        pairs : Pairs, optional
            Optional homogeneous subset of registered pairs.
        **kwargs : Any
            Solver keyword arguments.

        Returns
        -------
        Any
            Existing FanInSAR time-series result type.

        Raises
        ------
        ValueError
            If products are absent or *solver* is unknown.

        """
        if self.interferograms is None:
            message = "Network has no interferogram products to analyze"
            logger.error(message)
            raise IncompleteNetworkProductError(message)
        if self.interferograms.index_metadata is not None:
            product_type = self.interferograms.index_metadata.get("type")
            if product_type not in {None, "NetworkInterferogramIndex"}:
                message = "Network product index is not a canonical Network index"
                logger.error(message)
                raise IncompleteNetworkProductError(message)
        try:
            stack = self.to_ifg_stack(pairs=pairs)
        except Exception as exc:
            if isinstance(exc, NetworkAnalysisError):
                raise
            message = f"Network products cannot be opened for analysis: {exc}"
            logger.exception(message)
            raise IncompleteNetworkProductError(message) from exc
        normalized_solver = solver.lower()
        if normalized_solver == "sbas":
            from faninsar.timeseries.invert import SBAS

            return SBAS.solve(stack, **kwargs)
        if normalized_solver == "nsbas":
            from faninsar.timeseries.invert import invert

            return invert(stack, model=model, **kwargs)
        message = f"unknown Network time-series solver {solver!r}"
        logger.error(message)
        raise ValueError(message)

    def __repr__(self) -> str:
        """Return a concise Network summary."""
        parts = [f"Network(root={self.root!r})"]
        if self.geometry is not None:
            parts.append(f"  geometry: {self.geometry.root}")
        if self.interferograms is not None:
            summary: dict[str, Any] = self.interferograms.summary()
            parts.append(f"  interferograms: {summary['pair_count']} pairs")
        return "\n".join(parts)


class ISCE2Network(Network):
    """Network adapter for products authored by ISCE2.

    The adapter intentionally reuses the canonical path contract.  Format
    discovery is explicit at the class boundary and never probes unrelated
    processor layouts.
    """


class ISCE3Network(Network):
    """Network adapter for products authored by ISCE3."""


class GAMMANetwork(Network):
    """Network adapter for products authored by GAMMA."""


class GMTSARNetwork(Network):
    """Network adapter for products authored by GMTSAR."""


class SNAPNetwork(Network):
    """Network adapter for products authored by SNAP."""


__all__ = [
    "GAMMANetwork",
    "GMTSARNetwork",
    "ISCE2Network",
    "ISCE3Network",
    "IncompleteNetworkError",
    "IncompleteNetworkProductError",
    "LegacyLayoutError",
    "LegacyNetworkLayoutError",
    "Network",
    "NetworkAnalysisError",
    "NetworkConstructionError",
    "NetworkLayoutError",
    "NetworkPathError",
    "SNAPNetwork",
]

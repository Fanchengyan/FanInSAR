"""Path-based access to the acquisition Network of a Stack work directory.

``Network`` is a view over the committed ``ifg/``, ``unwrap/``, and
``timeseries/`` products of one Stack (PROPOSAL-0059); it copies nothing.
Existing interferogram and time-series Dataset components are reused
internally. External processor layouts are admitted through registered
readers rather than a parallel hierarchy of processor-named Network classes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from faninsar.logging import setup_logger
from faninsar.network.products import (
    NetworkProduct,
    NetworkProductIndex,
    _require_text,
)

if TYPE_CHECKING:
    from os import PathLike

    from faninsar.network.readers.geometry import NetworkGeometry
    from faninsar.network.readers.interferogram import InterferogramCollection
    from faninsar.stack.session import Stack
    from faninsar.timeseries.models import TimeSeriesModels

logger = setup_logger(__name__)


class NetworkConstructionError(RuntimeError):
    """Base error raised when a path cannot construct a :class:`Network`."""


class NetworkPathError(NetworkConstructionError, FileNotFoundError):
    """Raised when the requested Network root is missing or not a directory."""


class NetworkAnalysisError(NetworkConstructionError):
    """Base error raised when a Network cannot be analyzed."""


class IncompleteNetworkProductError(NetworkAnalysisError):
    """Raised when a Network has no complete interferogram product set."""


class NetworkContract:
    """Small analysis base that owns a snapshot of the product index.

    ``Network`` deliberately knows nothing about raster decoding.  A concrete
    Network or Stack supplies ``_analyze_network_products``; this base class
    validates and snapshots the product records before dispatching analysis.
    The snapshot id is derived from the products' write ids, so an index
    replacement cannot mix product sets during one call.
    """

    _network_generation_id: str | None = None
    _network_product_index: NetworkProductIndex | None = None

    @property
    def network_generation_id(self) -> str | None:
        """Return the id of the product snapshot visible to analysis."""
        return getattr(self, "_network_generation_id", None)

    @property
    def network_product_index(self) -> NetworkProductIndex | None:
        """Return the immutable Network product index, if refreshed."""
        return getattr(self, "_network_product_index", None)

    @property
    def analysis_ready(self) -> bool:
        """Whether a complete, homogeneous product snapshot is available."""
        return (
            self.network_generation_id is not None
            and self.network_product_index is not None
        )

    def refresh_generation(
        self,
        generation_id: str,
        products: (
            NetworkProductIndex | tuple[NetworkProduct, ...] | list[NetworkProduct]
        ),
    ) -> Self:
        """Atomically replace the product index with one validated snapshot.

        Parameters
        ----------
        generation_id : str
            Snapshot identity derived from the products' write ids.
        products : NetworkProductIndex or sequence of NetworkProduct
            Products of that snapshot.  Validation completes before either
            field on this object is changed.

        Raises
        ------
        ValueError
            If the identity is empty or products are mixed.

        """
        _require_text(generation_id, "generation_id")
        index = (
            products
            if isinstance(products, NetworkProductIndex)
            else NetworkProductIndex(tuple(products))
        )
        if not index.products:
            logger.error("Network snapshot contains no products")
            message = "Network snapshot requires at least one product"
            raise ValueError(message)
        index.homogeneous()
        object.__setattr__(self, "_network_product_index", index)
        object.__setattr__(self, "_network_generation_id", generation_id)
        return self

    def analyze_time_series(self, *args: Any, **kwargs: Any) -> Any:
        """Analyze the currently refreshed snapshot through the concrete seam."""
        index = self.network_product_index
        generation_id = self.network_generation_id
        if generation_id is None or index is None:
            logger.error("Network analysis requested before a product refresh")
            message = (
                "Network analysis is unavailable before a complete product refresh"
            )
            raise RuntimeError(message)
        callback = getattr(self, "_analyze_network_products", None)
        if not callable(callback):
            logger.error("Network has no concrete analysis implementation")
            message = "concrete Network must implement _analyze_network_products"
            raise NotImplementedError(message)
        # Keep a local immutable reference for the whole call.  A concurrent
        # refresh may advance the object for the next call, but cannot alter
        # this invocation's product cohort.
        return callback(index, *args, generation_id=generation_id, **kwargs)


class Network(NetworkContract):
    """View of the acquisition Network of one Stack work directory.

    Parameters
    ----------
    root : str or os.PathLike
        A Stack work directory (it holds ``stack.json``).

    Raises
    ------
    NetworkPathError
        If ``root`` is not a directory.
    IncompleteNetworkProductError
        If the planned Pair network has no complete IFG product set.

    """

    readers: ClassVar[Any]

    def __init__(self, root: str | PathLike[str]) -> None:
        """Open the Stack at ``root`` and bind its committed products."""
        try:
            resolved_root = Path(root)
        except TypeError as exc:
            message = f"Network root must be path-like, got {root!r}"
            logger.exception(message)
            raise NetworkConstructionError(message) from exc
        if not resolved_root.is_dir():
            message = f"Network directory not found: {resolved_root}"
            logger.error(message)
            raise NetworkPathError(message)
        from faninsar.stack.session import Stack

        stack = Stack.open(resolved_root)
        if not stack.analysis_ready:
            message = (
                f"Stack at {resolved_root} has no complete interferogram product set"
            )
            logger.error(message)
            raise IncompleteNetworkProductError(message)
        self._bind_stack(stack)

    @classmethod
    def _from_stack(cls, stack: Stack) -> Self:
        """Return a view over an open Stack without reopening it."""
        view = cls.__new__(cls)
        view._bind_stack(stack)
        return view

    def _bind_stack(self, stack: Stack) -> None:
        """Share one Stack's product index and readers."""
        from faninsar.network.readers.stack import StackInterferogramCollection

        self._root = stack.config.work_dir
        self._geometry = None
        self._interferograms = StackInterferogramCollection(stack)
        self._timeseries = stack.timeseries
        self._product_index = stack.network_product_index
        self._network_generation_id = stack.network_generation_id
        self._network_product_index = stack.network_product_index

    @classmethod
    def open(
        cls,
        path: str | PathLike[str],
        *,
        reader: object | None = None,
        registry: object | None = None,
    ) -> Any:
        """Open a Stack work directory or an explicitly selected reader.

        Parameters
        ----------
        path : str or os.PathLike
            Stack work directory, or a source understood by a custom reader.
        reader : str, type, or NetworkReader, optional
            ``None`` opens a Stack work directory.  A string is resolved
            through ``registry`` or :attr:`readers`; a reader class is
            constructed without arguments; and a reader instance is used
            directly.  External layouts are never auto-probed.
        registry : ReaderRegistry, optional
            Isolated registry used only with a string ``reader`` selector.

        Returns
        -------
        Network
            The object returned by the selected reader.

        Raises
        ------
        TypeError
            If ``registry`` is supplied without a string reader selector.
        NetworkConstructionError
            If a selected reader cannot be instantiated or does not expose
            the required ``read`` method.

        """
        if registry is not None and not isinstance(reader, str):
            message = "registry is valid only with a registered reader string"
            logger.error(message)
            raise TypeError(message)

        if reader is None:
            # Canonical mode intentionally does not consult the entry-point
            # registry and therefore cannot probe an external layout.
            return cls(path)

        selected: object
        if isinstance(reader, str):
            resolver = getattr(registry or cls.readers, "resolve", None)
            if not callable(resolver):
                message = "reader registry must expose a callable resolve method"
                logger.error(message)
                raise TypeError(message)
            selected_class = resolver(reader)
            selected = cls._instantiate_reader(selected_class, reader)
        elif isinstance(reader, type):
            selected = cls._instantiate_reader(reader, reader.__name__)
        else:
            selected = reader

        read = getattr(selected, "read", None)
        if not callable(read):
            message = "selected Network reader must define a callable read method"
            logger.error(message)
            raise NetworkConstructionError(message)
        return read(path)

    @staticmethod
    def _instantiate_reader(reader: object, name: str) -> object:
        """Construct one zero-argument reader class with typed diagnostics."""
        if not isinstance(reader, type):
            message = f"registered Network reader {name!r} must be a class"
            logger.error(message)
            raise NetworkConstructionError(message)
        try:
            return reader()
        except Exception as error:
            message = f"could not instantiate Network reader {name!r}"
            logger.exception(message)
            raise NetworkConstructionError(message) from error

    @property
    def root(self) -> Path:
        """Return the mounted Network root directory."""
        return self._root

    @property
    def geometry(self) -> NetworkGeometry | None:
        """Return the optional geometry Dataset facade."""
        return self._geometry

    @property
    def interferograms(self) -> InterferogramCollection:
        """Return the interferogram Dataset facade."""
        return self._interferograms

    @property
    def timeseries(self) -> Any:
        """Return the time series of this Network, if one exists."""
        return self._timeseries

    def to_ifg_stack(self, *, pairs: Any = None) -> Any:
        """Build the analysis stack by reading products through Dataset.

        Parameters
        ----------
        pairs : Pairs, optional
            Subset of Pair products; defaults to all products in the Network.

        Returns
        -------
        InterferogramStack
            Unwrapped phase and optional coherence accepted by time-series
            solvers.

        Raises
        ------
        ValueError
            If the Network has no readable unwrapped phase products.

        """
        from faninsar.processing.interferometry.contracts import InterferogramStack

        pair_obj = pairs if pairs is not None else self.interferograms.pairs()
        unwrapped: np.ndarray | None = None
        coherence: np.ndarray | None = None
        try:
            unwrapped_data = self.interferograms.open_stack("unw_phase", pairs=pair_obj)
            unwrapped = np.asarray(unwrapped_data.values, dtype=np.float64)
            if unwrapped.ndim == 3:
                unwrapped = unwrapped.reshape(unwrapped.shape[0], -1)
        except Exception as exc:
            logger.exception("Network unwrapped phase products cannot be read")
            message = "could not load unwrapped phase stack from Network Dataset"
            raise ValueError(message) from exc
        try:
            coherence_data = self.interferograms.open_stack("coherence", pairs=pair_obj)
            coherence = np.asarray(coherence_data.values, dtype=np.float64)
            if coherence.ndim == 3:
                coherence = coherence.reshape(coherence.shape[0], -1)
        except Exception:
            coherence = None
        return InterferogramStack.from_unwrapped(
            stack_id=str(self._root),
            pairs=pair_obj,
            unwrapped=unwrapped,
            coherence=coherence,
        )

    @classmethod
    def from_path(cls, root: str | PathLike[str]) -> Self:
        """Construct a Network from a canonical filesystem path."""
        return cls(root)

    @property
    def product_index(self) -> Any:
        """Return the validated logical product index, when registered."""
        return getattr(self, "_product_index", None) or self.network_product_index

    def register_products(self, products: Any) -> Any:
        """Register one homogeneous set of logical product records.

        Dataset discovery remains internal to Network; callers provide only
        immutable product metadata and locators.  Duplicate or mixed cohorts
        are rejected before a solver can be scheduled.
        """
        from faninsar.network.products import NetworkProductIndex

        try:
            index = NetworkProductIndex(tuple(products)).homogeneous()
        except (TypeError, ValueError) as exc:
            message = f"Network product registration rejected: {exc}"
            logger.exception(message)
            raise NetworkConstructionError(message) from exc
        # Keep the Dataset-facing registration and the shared snapshot
        # contract on one immutable index.  Stack uses the same protected
        # refresh operation directly; external Networks use this public path.
        snapshot_id = hashlib.sha256(
            "|".join(
                sorted(
                    product.content_digest or product.canonical
                    for product in index.products
                )
            ).encode("utf-8")
        ).hexdigest()
        NetworkContract.refresh_generation(self, snapshot_id, index)
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
        """Compatibility alias for :meth:`invert`.

        ``invert`` is the preferred interface.  The historical ``solver``
        selector remains here only so existing callers can migrate without a
        second Network construction.
        """
        if not isinstance(solver, str):
            message = f"Network time-series solver must be text, got {solver!r}"
            logger.error(message)
            raise TypeError(message)
        if solver.lower() not in {"sbas", "nsbas"}:
            message = f"unknown Network time-series solver {solver!r}"
            logger.error(message)
            raise ValueError(message)
        if solver.lower() == "nsbas" and model is None:
            # Preserve the historical compatibility path.  The new ``invert``
            # seam intentionally defines model omission as SBAS, so routing
            # this legacy spelling through it would silently change behavior.
            return self._invert_legacy_nsbas(pairs=pairs, **kwargs)
        return self.invert(model=model, pairs=pairs, **kwargs)

    def invert(
        self,
        *,
        model: type[TimeSeriesModels] | TimeSeriesModels | None = None,
        pairs: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invert unwrapped Network products into a time series.

        Parameters
        ----------
        model : type[TimeSeriesModels] or TimeSeriesModels, optional
            Omit the model for pure SBAS.  A model class is instantiated with
            the selected acquisition dates.  A configured model instance is
            accepted for models that need additional domain inputs and must
            describe exactly those dates; either form selects NSBAS.
        pairs : Pairs, optional
            Pair subset to read and invert.  The selected pairs are resolved
            before a model class is instantiated.
        **kwargs : Any
            Options forwarded to the selected inversion implementation.

        Returns
        -------
        TimeSeries
            Self-describing phase time series, committed as the Network's
            time-series product bound to the unwrap products it consumed.

        Raises
        ------
        TypeError
            If ``model`` is not a model class or configured model instance,
            or if a model class needs constructor arguments beyond ``dates``.
        ValueError
            If a configured model's dates do not match the selected pairs.

        """
        if self.interferograms is None:
            message = "Network has no interferogram products to invert"
            logger.error(message)
            raise IncompleteNetworkProductError(message)
        try:
            stack = self.to_ifg_stack(pairs=pairs)
        except Exception as exc:
            message = f"Network products cannot be opened for inversion: {exc}"
            logger.exception(message)
            raise IncompleteNetworkProductError(message) from exc

        normalized_model = self._normalize_inversion_model(model, stack.pairs.dates)
        if normalized_model is None:
            from faninsar.timeseries.results import invert_unwrapped_pairs

            values = self.interferograms.open_stack("unw_phase", pairs=stack.pairs)
            pair_ids = [str(value) for value in values.coords["pair"].values]
            pair_phases = {
                pair_id: np.asarray(values.isel(pair=index).values)
                for index, pair_id in enumerate(pair_ids)
            }
            result = invert_unwrapped_pairs(
                pair_phases,
                device=kwargs.pop("device", "cpu"),
            )
            return self._persist_result(result)

        return self._run_nsbas(stack, model=normalized_model, **kwargs)

    def _run_nsbas(
        self,
        stack: Any,
        *,
        model: TimeSeriesModels | None,
        **kwargs: Any,
    ) -> Any:
        """Run NSBAS and adapt its result to a ``TimeSeries``."""
        from faninsar.timeseries.invert import invert
        from faninsar.timeseries.solver import NSBASResult

        raw = invert(stack, model=model, **kwargs)
        if isinstance(raw, NSBASResult):
            increments = raw.increments
            residual_pairs = raw.pair_residuals
        elif isinstance(raw, tuple) and len(raw) == 4:
            increments, _parameters, residual_pairs, _residual_model = raw
        else:
            message = "NSBAS solver returned an invalid result"
            logger.error(message)
            raise NetworkAnalysisError(message)
        from faninsar.timeseries.results import TimeSeries

        increments_array = np.asarray(increments, dtype=np.float32)
        residual_array = np.asarray(residual_pairs, dtype=np.float32)
        n_pixels = int(np.asarray(stack.unwrapped_matrix()).shape[1])
        n_dates = len(stack.pairs.dates)
        if increments_array.shape != (n_dates - 1, n_pixels):
            message = "NSBAS solver returned increments with an invalid shape"
            logger.error(message)
            raise NetworkAnalysisError(message)
        if residual_array.shape != (len(stack.pairs), n_pixels):
            message = "NSBAS solver returned residuals with an invalid shape"
            logger.error(message)
            raise NetworkAnalysisError(message)
        cumulative_array = np.concatenate(
            [
                np.zeros((1, n_pixels), dtype=np.float32),
                np.cumsum(increments_array, axis=0),
            ],
            axis=0,
        )
        result = TimeSeries(
            pair_ids=tuple(str(name) for name in stack.pairs.names),
            dates=tuple(
                str(date.date()).replace("-", "") for date in stack.pairs.dates
            ),
            phase_increments_rad=increments_array,
            residual_phase_pairs_rad=residual_array,
            phase_cumulative_rad=cumulative_array,
            metadata={"method": "nsbas", "n_pairs": len(stack.pairs)},
        )
        return self._persist_result(result)

    @staticmethod
    def _normalize_inversion_model(
        model: type[TimeSeriesModels] | TimeSeriesModels | None,
        dates: Any,
    ) -> TimeSeriesModels | None:
        """Resolve and validate the optional model at the inversion seam."""
        from faninsar.timeseries.models import TimeSeriesModels

        if model is None:
            return None
        if isinstance(model, type):
            if not issubclass(model, TimeSeriesModels):
                message = (
                    "model must be a TimeSeriesModels subclass or configured instance"
                )
                logger.error(message, extra={"model_type": model.__name__})
                raise TypeError(message)
            try:
                return model(dates=dates)
            except TypeError as exc:
                message = (
                    f"{model.__name__} could not be constructed from dates; "
                    "pass a configured instance instead"
                )
                logger.exception(message)
                raise TypeError(message) from exc
        if not isinstance(model, TimeSeriesModels):
            message = "model must be a TimeSeriesModels subclass or configured instance"
            logger.error(message, extra={"model_type": type(model).__name__})
            raise TypeError(message)
        if not np.array_equal(np.asarray(model.dates), np.asarray(dates)):
            message = "configured model dates must exactly match selected pairs dates"
            logger.error(message)
            raise ValueError(message)
        return model

    def _spatial_metadata_for_result(self, result: Any) -> Any:
        """Build spatial metadata from the mounted interferogram facade."""
        from faninsar.timeseries.results import SpatialMetadata

        source = getattr(self.interferograms, "spatial_metadata", None)
        if source is None:
            return getattr(result, "spatial", None)
        try:
            raw = source() if callable(source) else source
        except Exception as error:
            message = "Network spatial metadata could not be read"
            logger.exception(message)
            raise NetworkAnalysisError(message) from error
        if raw is None:
            return getattr(result, "spatial", None)
        if isinstance(raw, SpatialMetadata):
            payload = raw.to_dict()
        elif isinstance(raw, Mapping):
            payload = dict(raw)
        else:
            payload = {
                name: getattr(raw, name)
                for name in (
                    "domain",
                    "shape",
                    "grid_identity",
                    "crs",
                    "transform",
                    "radar_coordinates",
                )
                if hasattr(raw, name)
            }
        try:
            shape = tuple(int(size) for size in payload["shape"])
            spatial = SpatialMetadata(
                domain=payload["domain"],
                shape=shape,
                grid_identity=payload["grid_identity"],
                crs=payload.get("crs"),
                transform=(
                    tuple(float(value) for value in payload["transform"])
                    if payload.get("transform") is not None
                    else None
                ),
                radar_coordinates=payload.get("radar_coordinates"),
            )
        except (KeyError, TypeError, ValueError) as error:
            message = "Network spatial metadata is invalid"
            logger.exception(message)
            raise NetworkAnalysisError(message) from error
        expected_pixels = shape[0] * shape[1]
        for name in (
            "phase_increments_rad",
            "phase_cumulative_rad",
            "residual_phase_pairs_rad",
            "displacement_increments_m",
            "displacement_cumulative_m",
        ):
            values = getattr(result, name, None)
            if values is None:
                continue
            array = np.asarray(values)
            matches = (array.ndim >= 3 and tuple(array.shape[-2:]) == shape) or (
                array.ndim == 2 and array.shape[-1] == expected_pixels
            )
            if not matches:
                message = (
                    f"Network spatial shape {shape} does not match "
                    f"TimeSeries {name} shape {array.shape}"
                )
                logger.error(message)
                raise ValueError(message)
        return spatial

    def _invert_legacy_nsbas(
        self,
        *,
        pairs: Any | None,
        **kwargs: Any,
    ) -> Any:
        """Run the historical ``analyze_time_series(solver='nsbas')`` path."""
        if self.interferograms is None:
            message = "Network has no interferogram products to invert"
            logger.error(message)
            raise IncompleteNetworkProductError(message)
        try:
            stack = self.to_ifg_stack(pairs=pairs)
        except Exception as exc:
            message = f"Network products cannot be opened for inversion: {exc}"
            logger.exception(message)
            raise IncompleteNetworkProductError(message) from exc

        return self._run_nsbas(stack, model=None, **kwargs)

    def _persist_result(self, result: Any) -> Any:
        """Commit a time-series result as this Network's time-series product.

        The product Item records the ``write_id`` of every unwrap product the
        result consumed, so it becomes stale when one is replaced.
        """
        from dataclasses import is_dataclass, replace

        if not is_dataclass(result) or not hasattr(result, "spatial"):
            return result
        from faninsar.timeseries.io import write_timeseries_product

        bound = replace(result, spatial=self._spatial_metadata_for_result(result))
        source = getattr(self.interferograms, "unwrap_write_ids", None)
        inputs: dict[str, str] = {}
        if callable(source):
            available = source()
            inputs = {pair_id: available[pair_id] for pair_id in bound.pair_ids}
        path = write_timeseries_product(
            bound,
            self.root / "timeseries",
            inputs=inputs,
            properties={"faninsar:phase_source": "unwrap"} if inputs else None,
            overwrite=True,
        )
        persisted = replace(bound, canonical_path=path)
        self._timeseries = persisted
        return persisted

    def __repr__(self) -> str:
        """Return a concise Network summary."""
        pair_count = len(self.interferograms.pairs()) if self.interferograms else 0
        return f"Network(root={self.root!r}, pairs={pair_count})"


# Keep one process-wide default registry on the canonical data-backed class.
# The import is intentionally after the class definition to avoid a cycle
# through ``faninsar.network``.
from faninsar.network.registry import ReaderRegistry  # noqa: E402

Network.readers = ReaderRegistry()


__all__ = [
    "IncompleteNetworkProductError",
    "Network",
    "NetworkAnalysisError",
    "NetworkConstructionError",
    "NetworkPathError",
]

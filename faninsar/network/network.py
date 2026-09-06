"""Canonical, path-based access to a FanInSAR acquisition Network.

``Network`` is the strict public seam: a product must carry a versioned
manifest, a complete immutable generation, and the canonical interferogram
index type. Existing geometry, interferogram, and time-series Dataset
components are reused internally; they do not define the Network lifecycle.
External processor names are declaration-only adapters in this MVP; they do
not probe or infer unrelated processor layouts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from faninsar.core.network import (
    AssetKind,
    PhaseConvention,
)
from faninsar.core.network import (
    Network as NetworkContract,
)
from faninsar.logging import setup_logger
from faninsar.network.readers.geometry import NetworkGeometry
from faninsar.network.readers.interferogram import InterferogramCollection
from faninsar.network.readers.timeseries import NetworkTimeSeries

if TYPE_CHECKING:
    from os import PathLike

logger = setup_logger(__name__)

NETWORK_SCHEMA_VERSION = "network_v1"
NETWORK_MANIFEST_NAME = "manifest.json"
NETWORK_CURRENT_NAME = "CURRENT"
NETWORK_GENERATIONS_NAME = ".network_generations"
NETWORK_INDEX_TYPE = "NetworkInterferogramIndex"
NETWORK_CURRENT_SCHEMA_VERSION = "network_current_v1"
_MAX_MANIFEST_BYTES = 1024 * 1024
_REQUIRED_PRODUCT_FIELDS = (
    "id",
    "primary_id",
    "secondary_id",
    "product_kind",
    "asset_location",
    "geometry_identity",
    "source_software",
    "phase_convention",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NetworkConstructionError(RuntimeError):
    """Base error raised when a path cannot construct a :class:`Network`."""


class NetworkPathError(NetworkConstructionError, FileNotFoundError):
    """Raised when the requested Network root is missing or not a directory."""


class NetworkManifestError(NetworkConstructionError):
    """Raised when a Network manifest is missing, malformed, or unsupported."""


class NetworkAnalysisError(NetworkConstructionError):
    """Base error raised when a Network cannot be analyzed."""


class NetworkGenerationError(NetworkConstructionError):
    """Raised when the selected Network generation is absent or incomplete."""


class NetworkCurrentError(NetworkGenerationError):
    """Raised when the current-generation pointer is missing or inconsistent."""


class UnknownNetworkIndexTypeError(NetworkManifestError):
    """Raised when a product declares an index type outside the MVP contract."""


class IncompleteNetworkProductError(NetworkAnalysisError):
    """Raised when a Network has no non-empty interferogram product index."""


class LegacyNetworkLayoutError(NetworkConstructionError):
    """Raised when a path contains a pre-standardization Network layout."""

    def __init__(self, root: Path, markers: tuple[Path, ...]) -> None:
        """Initialize an error describing legacy paths found below *root*."""
        self.root = root
        self.markers = markers
        marker_text = ", ".join(str(path) for path in markers)
        super().__init__(
            f"Legacy InSAR network layout under {root}: {marker_text}. "
            "Publish a canonical versioned Network generation first."
        )


class ExternalNetworkLayoutError(NetworkConstructionError):
    """Raised when a processor adapter lacks an explicit layout declaration."""


# Compatibility aliases for callers that used the first Network seam.
NetworkLayoutError = LegacyNetworkLayoutError
LegacyLayoutError = LegacyNetworkLayoutError
IncompleteNetworkError = IncompleteNetworkProductError


def _legacy_markers(root: Path) -> tuple[Path, ...]:
    """Return legacy interferogram markers directly under *root*."""
    candidates = (
        root / "ifg",
        root / "ifg_index.json",
        root / "interferograms" / "ifg_index.json",
    )
    return tuple(path for path in candidates if path.exists())


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one bounded, regular JSON manifest object."""
    if not path.is_file() or path.is_symlink():
        message = f"Network manifest is missing or unsafe: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    try:
        size = path.stat().st_size
    except OSError as exc:
        message = f"Network manifest cannot be read: {path}"
        logger.exception(message)
        raise NetworkManifestError(message) from exc
    if size > _MAX_MANIFEST_BYTES:
        message = f"Network manifest exceeds size limit: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        message = f"Network manifest cannot be read: {path}"
        logger.exception(message)
        raise NetworkManifestError(message) from exc
    if not isinstance(data, dict):
        message = f"Network manifest must be an object: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    return data


def _canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    """Compute the digest over a manifest without its self-referential field."""
    unsigned = dict(manifest)
    unsigned.pop("manifest_digest", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest_digest(manifest: dict[str, Any], *, path: Path) -> str:
    """Require a valid SHA-256 digest for one immutable manifest."""
    digest = manifest.get("manifest_digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        message = f"Network manifest_digest is invalid: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    if digest != _canonical_manifest_digest(manifest):
        message = f"Network manifest_digest mismatch: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    return digest


def _validate_asset_location(value: object, *, path: Path, product_id: str) -> None:
    """Reject absolute, traversing, or platform-ambiguous asset locations."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        message = f"Network product asset_location is invalid: {product_id!r}"
        logger.error(message)
        raise NetworkManifestError(message)
    if "\\" in value:
        message = f"Network product asset_location uses unsafe separators: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
    ):
        message = f"Network product asset_location must be a safe relative path: {path}"
        logger.error(message)
        raise NetworkManifestError(message)


def _validate_manifest(
    path: Path,
    *,
    expected_generation: str | None = None,
) -> dict[str, Any]:
    """Validate schema, complete status, generation identity, and index type."""
    manifest = _read_manifest(path)
    _validate_manifest_digest(manifest, path=path)
    if manifest.get("schema_version") != NETWORK_SCHEMA_VERSION:
        message = (
            f"unsupported Network manifest version at {path}: "
            f"{manifest.get('schema_version')!r}"
        )
        logger.error(message)
        raise NetworkManifestError(message)
    if manifest.get("status") != "complete":
        message = f"Network manifest is not complete: {path}"
        logger.error(message)
        raise NetworkGenerationError(message)
    generation_id = manifest.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id.strip():
        message = f"Network manifest has no generation_id: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    if expected_generation is not None and generation_id != expected_generation:
        message = f"Network generation identity mismatch at {path}"
        logger.error(message)
        raise NetworkGenerationError(message)
    index_type = manifest.get("index_type", manifest.get("type"))
    if index_type != NETWORK_INDEX_TYPE:
        message = f"unknown Network index type {index_type!r} at {path}"
        logger.error(message)
        raise UnknownNetworkIndexTypeError(message)
    products = manifest.get("products")
    if not isinstance(products, list) or not products:
        message = f"Network product index is empty or missing: {path}"
        logger.error(message)
        raise IncompleteNetworkProductError(message)
    phase_convention = manifest.get("phase_convention")
    if not isinstance(phase_convention, str) or not phase_convention.strip():
        message = f"Network manifest has no phase_convention: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    try:
        PhaseConvention(phase_convention)
    except (TypeError, ValueError) as error:
        message = f"unknown Network phase_convention {phase_convention!r}: {path}"
        logger.exception(message)
        raise NetworkManifestError(message) from error
    product_ids: set[str] = set()
    for position, product in enumerate(products):
        if not isinstance(product, dict):
            message = f"Network product record {position} is not an object: {path}"
            logger.error(message)
            raise NetworkManifestError(message)
        for field_name in _REQUIRED_PRODUCT_FIELDS:
            value = product.get(field_name)
            if not isinstance(value, str) or not value.strip():
                message = (
                    f"Network product record {position} has no {field_name}: {path}"
                )
                logger.error(message)
                raise NetworkManifestError(message)
        product_id = product["id"]
        content_digest = product.get("content_digest")
        if (
            not isinstance(content_digest, str)
            or _SHA256_RE.fullmatch(content_digest) is None
        ):
            message = f"Network product record {position} has no content_digest: {path}"
            logger.error(message)
            raise NetworkManifestError(message)
        lineage = product.get("lineage")
        if (
            not isinstance(lineage, list)
            or not lineage
            or any(
                not isinstance(item, str) or not item.strip() or item != item.strip()
                for item in lineage
            )
        ):
            message = f"Network product lineage is invalid: {product_id!r}"
            logger.error(message)
            raise NetworkManifestError(message)
        if product_id in product_ids:
            message = f"Network product IDs are not unique: {product_id!r}"
            logger.error(message)
            raise NetworkManifestError(message)
        product_ids.add(product_id)
        _validate_asset_location(
            product["asset_location"], path=path, product_id=product_id
        )
        if product["primary_id"] == product["secondary_id"]:
            message = f"Network product has identical Pair roles: {product_id!r}"
            logger.error(message)
            raise NetworkManifestError(message)
        try:
            AssetKind(product["product_kind"])
        except (TypeError, ValueError) as error:
            message = (
                f"unknown Network product_kind {product['product_kind']!r}: {path}"
            )
            logger.exception(message)
            raise NetworkManifestError(message) from error
        if product["phase_convention"] != phase_convention:
            message = (
                f"Network product phase_convention disagrees with manifest: {path}"
            )
            logger.error(message)
            raise NetworkManifestError(message)
    return manifest


def _product_set(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return a deterministic signature for a manifest's product records."""
    products = manifest["products"]
    return tuple(
        sorted(
            json.dumps(product, sort_keys=True, separators=(",", ":"))
            for product in products
        )
    )


def _validate_current(path: Path, *, expected_generation: str) -> dict[str, Any]:
    """Validate the versioned pointer to the selected Network generation."""
    current = _read_manifest(path)
    if current.get("schema_version") != NETWORK_CURRENT_SCHEMA_VERSION:
        message = f"unsupported Network CURRENT version at {path}"
        logger.error(message)
        raise NetworkCurrentError(message)
    if current.get("status") != "complete":
        message = f"Network CURRENT is not complete: {path}"
        logger.error(message)
        raise NetworkCurrentError(message)
    generation_id = current.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id.strip():
        message = f"Network CURRENT has no generation_id: {path}"
        logger.error(message)
        raise NetworkCurrentError(message)
    if generation_id != expected_generation:
        message = "Network CURRENT does not select the root manifest generation"
        logger.error(message)
        raise NetworkCurrentError(message)
    manifest_digest = current.get("manifest_digest")
    if (
        not isinstance(manifest_digest, str)
        or _SHA256_RE.fullmatch(manifest_digest) is None
    ):
        message = f"Network CURRENT manifest_digest is invalid: {path}"
        logger.error(message)
        raise NetworkCurrentError(message)
    return current


def _validate_network_layout(
    root: Path,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Validate the canonical root manifest and selected immutable generation."""
    markers = _legacy_markers(root)
    if markers:
        logger.error("Refusing legacy Network layout at %s; markers=%s", root, markers)
        raise LegacyNetworkLayoutError(root, markers)

    root_manifest = _validate_manifest(root / NETWORK_MANIFEST_NAME)
    generations = root / NETWORK_GENERATIONS_NAME
    if not generations.is_dir() or generations.is_symlink():
        message = f"Network generation root is missing or unsafe: {generations}"
        logger.error(message)
        raise NetworkGenerationError(message)
    current_path = root / NETWORK_CURRENT_NAME
    generation_id = str(root_manifest["generation_id"])
    if revision is None:
        current = _validate_current(current_path, expected_generation=generation_id)
    else:
        # Historical generations are immutable and may no longer be selected
        # by CURRENT.  Validate the requested generation directly while still
        # validating the canonical root manifest above.
        if (
            not isinstance(revision, str)
            or not revision.strip()
            or revision != revision.strip()
            or revision in {".", ".."}
            or "/" in revision
            or "\\" in revision
        ):
            message = f"Network revision is invalid: {revision!r}"
            logger.error(message)
            raise NetworkGenerationError(message)
        generation_id = revision
        current = None
    generation_root = generations / str(generation_id)
    if not generation_root.is_dir() or generation_root.is_symlink():
        message = (
            f"Network generation directory is missing or unsafe: {generation_root}"
        )
        logger.error(message)
        raise NetworkCurrentError(message)
    generation_manifest = _validate_manifest(
        generation_root / NETWORK_MANIFEST_NAME,
        expected_generation=generation_id,
    )
    if revision is None and _product_set(generation_manifest) != _product_set(
        root_manifest
    ):
        message = "Network generation manifest does not match root product set"
        logger.error(message)
        raise NetworkGenerationError(message)
    if (
        revision is None
        and generation_manifest["manifest_digest"] != root_manifest["manifest_digest"]
    ):
        message = "Network generation manifest_digest does not match root manifest"
        logger.error(message)
        raise NetworkGenerationError(message)
    if (
        current is not None
        and current["manifest_digest"] != root_manifest["manifest_digest"]
    ):
        message = "Network CURRENT manifest_digest does not select root manifest"
        logger.error(message)
        raise NetworkCurrentError(message)
    if generation_manifest["phase_convention"] != root_manifest["phase_convention"]:
        message = "Network generation manifest phase convention mismatch"
        logger.error(message)
        raise NetworkGenerationError(message)
    if not (
        (root / "interferograms").is_dir()
        or (generation_root / "interferograms").is_dir()
    ):
        message = "Network requires an interferograms/ product collection"
        logger.error(message)
        raise IncompleteNetworkProductError(message)
    return generation_manifest


class Network(NetworkContract):
    """Concrete path-based view of one canonical InSAR Network."""

    readers: ClassVar[Any]

    def __init__(
        self,
        root: str | PathLike[str],
        *,
        revision: str | None = None,
    ) -> None:
        """Mount and validate a canonical Network product."""
        try:
            resolved_root = Path(root)
        except TypeError as exc:
            message = f"Network root must be path-like, got {root!r}"
            logger.exception(message)
            raise NetworkConstructionError(message) from exc
        if not resolved_root.exists() or not resolved_root.is_dir():
            message = f"Network directory not found: {resolved_root}"
            logger.exception(message)
            raise NetworkPathError(message)
        self.manifest = _validate_network_layout(resolved_root, revision=revision)
        self.generation_root = (
            resolved_root
            / NETWORK_GENERATIONS_NAME
            / str(self.manifest["generation_id"])
        )
        # Network owns the path-level data facade directly. Sharing the
        # component Dataset readers avoids inheriting broader discovery and
        # fallback rules from an unrelated lifecycle facade.
        self._root = resolved_root
        geometry_root = resolved_root / "geometry"
        self._geometry = (
            NetworkGeometry(geometry_root) if geometry_root.is_dir() else None
        )
        generation_interferograms_root = self.generation_root / "interferograms"
        interferograms_root = (
            generation_interferograms_root
            if generation_interferograms_root.is_dir()
            else resolved_root / "interferograms"
        )
        index_path = interferograms_root / "interferograms_index.json"
        index_version = None
        if index_path.is_file():
            try:
                index_version = json.loads(index_path.read_text()).get("version")
            except (OSError, UnicodeError, ValueError, AttributeError):
                index_version = None
        if index_version == "stack_artifact_v1":
            from faninsar.stack.network import StackInterferogramCollection

            self._interferograms = StackInterferogramCollection(
                interferograms_root.parent
            )
        else:
            self._interferograms = InterferogramCollection(interferograms_root)
        timeseries_root = resolved_root / "timeseries"
        self._timeseries = (
            NetworkTimeSeries(timeseries_root) if timeseries_root.is_dir() else None
        )
        self._product_index: Any = None
        self._network_generation_id = str(self.manifest["generation_id"])
        index = self.interferograms.index_metadata if self.interferograms else None
        if index is None:
            message = "Network interferograms have no canonical index"
            logger.exception(message)
            raise IncompleteNetworkProductError(message)
        index_type = index.get("type", index.get("index_type"))
        if index_type != NETWORK_INDEX_TYPE:
            message = f"unknown Network index type {index_type!r}"
            logger.error(message)
            raise UnknownNetworkIndexTypeError(message)
        if len(self.interferograms.pairs().names) == 0:
            message = "Network interferogram index contains no products"
            logger.error(message)
            raise IncompleteNetworkProductError(message)

    @classmethod
    def open(
        cls,
        path: str | PathLike[str],
        *,
        reader: object | None = None,
        revision: str | None = None,
        registry: object | None = None,
    ) -> Any:
        """Open a canonical Network or an explicitly selected reader.

        Parameters
        ----------
        path : str or os.PathLike
            Canonical Network root, or a source understood by a custom reader.
        reader : str, type, or NetworkReader, optional
            ``None`` selects only the canonical FanInSAR layout.  A string is
            resolved through ``registry`` or :attr:`readers`; a reader class
            is constructed without arguments; and a reader instance is used
            directly.  External layouts are never auto-probed.
        revision : str, optional
            Immutable generation identifier.  Canonical reads accept the
            current generation only; custom readers receive this value and
            must reject it if their format cannot address snapshots.
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
        NetworkGenerationError
            If a canonical revision does not match the current generation.
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
            return cls(path, revision=revision)

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
        try:
            return read(path, revision=revision)
        except TypeError:
            # A reader that cannot honor the explicit revision must fail
            # honestly instead of falling back to a mutable current view.
            if revision is not None:
                message = "selected Network reader cannot honor revision"
                logger.exception(message)
                raise NetworkGenerationError(message) from None
            raise

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
        """Return the validated interferogram Dataset facade."""
        return self._interferograms

    @property
    def timeseries(self) -> NetworkTimeSeries | None:
        """Return the optional time-series Dataset facade."""
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
        from faninsar.processing.contracts.ifg import InterferogramStack

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
        from faninsar.core.network import NetworkProductIndex

        try:
            index = NetworkProductIndex(tuple(products)).homogeneous()
        except (TypeError, ValueError) as exc:
            message = f"Network product registration rejected: {exc}"
            logger.exception(message)
            raise NetworkConstructionError(message) from exc
        # Keep the Dataset-facing registration and the shared generation
        # contract on one immutable snapshot.  Stack uses the same protected
        # refresh operation directly; external Networks use this public path.
        NetworkContract.refresh_generation(
            self,
            str(self.manifest["generation_id"]),
            index,
        )
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
        """Analyze the mounted Network through its Dataset-backed stack.

        The raster Dataset is opened internally from the canonical path.  No
        caller-provided Dataset object is accepted at this boundary.
        """
        if self.interferograms is None:
            message = "Network has no interferogram products to analyze"
            logger.error(message)
            raise IncompleteNetworkProductError(message)
        try:
            stack = self.to_ifg_stack(pairs=pairs)
        except Exception as exc:
            message = f"Network products cannot be opened for analysis: {exc}"
            logger.exception(message)
            raise IncompleteNetworkProductError(message) from exc
        normalized_solver = solver.lower()
        if normalized_solver == "sbas":
            # Use the persisted processing result contract for the canonical
            # path.  The lower-level historical solver returns four arrays;
            # Network callers need one self-describing TimeSeriesResult.
            from faninsar.processing.timeseries.inversion import (
                invert_unwrapped_pairs,
            )

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
            return self._bind_analysis_revision(result)
        if normalized_solver == "nsbas":
            from faninsar.timeseries.invert import invert

            raw = invert(stack, model=model, **kwargs)
            if not isinstance(raw, tuple) or len(raw) != 4:
                message = "NSBAS solver returned an invalid result"
                logger.error(message)
                raise NetworkAnalysisError(message)
            from faninsar.processing.timeseries.inversion import TimeSeriesResult

            increments, _parameters, residual_pairs, _residual_model = raw
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
            result = TimeSeriesResult(
                pair_ids=tuple(str(name) for name in stack.pairs.names),
                dates=tuple(
                    str(date.date()).replace("-", "") for date in stack.pairs.dates
                ),
                increments=increments_array,
                residual_pairs=residual_array,
                cumulative=cumulative_array,
                metadata={"method": "nsbas", "n_pairs": len(stack.pairs)},
            )
            return self._bind_analysis_revision(result)
        message = f"unknown Network time-series solver {solver!r}"
        logger.exception(message)
        raise ValueError(message)

    def _bind_analysis_revision(self, result: Any) -> Any:
        """Attach this Network generation to a returned time-series result."""
        revision = self.network_generation_id or str(self.manifest["generation_id"])
        if hasattr(result, "revision_id"):
            from dataclasses import replace

            return replace(result, revision_id=revision)
        return result

    def __repr__(self) -> str:
        """Return a concise Network summary."""
        pair_count = len(self.interferograms.pairs()) if self.interferograms else 0
        return (
            f"Network(root={self.root!r}, "
            f"generation={self.manifest['generation_id']!r}, pairs={pair_count})"
        )


class _DeclaredProcessorNetwork(Network):
    """Network adapter whose processor identity is an explicit manifest marker."""

    processor_marker: str

    def __init__(self, root: str | PathLike[str]) -> None:
        """Mount only when ``source_software`` declares this processor."""
        path = Path(root)
        if not path.exists() or not path.is_dir():
            super().__init__(path)
        manifest = _read_manifest(path / NETWORK_MANIFEST_NAME)
        if manifest.get("source_software") != self.processor_marker:
            message = (
                f"{type(self).__name__} requires manifest source_software="
                f"{self.processor_marker!r}; no format discovery is performed"
            )
            logger.error(message)
            raise ExternalNetworkLayoutError(message)
        super().__init__(path)


class ISCE2Network(_DeclaredProcessorNetwork):
    """Canonical Network explicitly declared as authored by ISCE2."""

    processor_marker = "isce2"


class ISCE3Network(_DeclaredProcessorNetwork):
    """Canonical Network explicitly declared as authored by ISCE3."""

    processor_marker = "isce3"


class GAMMANetwork(_DeclaredProcessorNetwork):
    """Canonical Network explicitly declared as authored by GAMMA."""

    processor_marker = "gamma"


class GMTSARNetwork(_DeclaredProcessorNetwork):
    """Canonical Network explicitly declared as authored by GMTSAR."""

    processor_marker = "gmtsar"


class SNAPNetwork(_DeclaredProcessorNetwork):
    """Canonical Network explicitly declared as authored by SNAP."""

    processor_marker = "snap"


# Keep one process-wide default registry on the canonical data-backed class.
# The import is intentionally after class definitions so importing this legacy
# compatibility module cannot trigger a cycle through ``faninsar.network``.
from faninsar.network.registry import ReaderRegistry  # noqa: E402

Network.readers = ReaderRegistry()


__all__ = [
    "ExternalNetworkLayoutError",
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
    "NetworkCurrentError",
    "NetworkGenerationError",
    "NetworkLayoutError",
    "NetworkManifestError",
    "NetworkPathError",
    "SNAPNetwork",
    "UnknownNetworkIndexTypeError",
]

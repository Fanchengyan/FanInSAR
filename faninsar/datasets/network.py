"""Canonical, path-based access to a FanInSAR acquisition Network.

``Frame`` remains the reader for historical frame products.  ``Network`` is
the stricter public seam: a product must carry a versioned manifest, a
complete immutable generation, and the canonical interferogram index type.
External processor names are declaration-only adapters in this MVP; they do
not probe or infer unrelated processor layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from faninsar.logging import setup_logger

from .frame.frame import Frame

if TYPE_CHECKING:
    from os import PathLike

logger = setup_logger(__name__)

NETWORK_SCHEMA_VERSION = "network_v1"
NETWORK_MANIFEST_NAME = "manifest.json"
NETWORK_CURRENT_NAME = "CURRENT"
NETWORK_GENERATIONS_NAME = ".network_generations"
NETWORK_INDEX_TYPE = "NetworkInterferogramIndex"
_MAX_MANIFEST_BYTES = 1024 * 1024


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
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            message = f"Network manifest exceeds size limit: {path}"
            logger.error(message)
            raise NetworkManifestError(message)
        data = json.loads(path.read_text(encoding="utf-8"))
    except NetworkManifestError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        message = f"Network manifest cannot be read: {path}"
        logger.error("%s: %s", message, exc)
        raise NetworkManifestError(message) from exc
    if not isinstance(data, dict):
        message = f"Network manifest must be an object: {path}"
        logger.error(message)
        raise NetworkManifestError(message)
    return data


def _validate_manifest(
    path: Path,
    *,
    expected_generation: str | None = None,
) -> dict[str, Any]:
    """Validate schema, complete status, generation identity, and index type."""
    manifest = _read_manifest(path)
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
    return manifest


def _validate_network_layout(root: Path) -> dict[str, Any]:
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
    current = _read_manifest(current_path)
    generation_id = current.get("generation_id")
    if generation_id != root_manifest["generation_id"]:
        message = "Network CURRENT does not select the root manifest generation"
        logger.error(message)
        raise NetworkCurrentError(message)
    generation_root = generations / str(generation_id)
    if not generation_root.is_dir() or generation_root.is_symlink():
        message = (
            "Network generation directory is missing or unsafe: "
            f"{generation_root}"
        )
        logger.error(message)
        raise NetworkCurrentError(message)
    _validate_manifest(
        generation_root / NETWORK_MANIFEST_NAME,
        expected_generation=str(generation_id),
    )
    if not (root / "interferograms").is_dir():
        message = "Network requires an interferograms/ product collection"
        logger.error(message)
        raise IncompleteNetworkProductError(message)
    return root_manifest


class Network(Frame):
    """Concrete path-based view of one canonical InSAR Network."""

    def __init__(self, root: str | PathLike[str]) -> None:
        """Mount and validate a canonical Network product."""
        try:
            resolved_root = Path(root)
        except TypeError as exc:
            message = f"Network root must be path-like, got {root!r}"
            logger.error(message)
            raise NetworkConstructionError(message) from exc
        if not resolved_root.exists() or not resolved_root.is_dir():
            message = f"Network directory not found: {resolved_root}"
            logger.error(message)
            raise NetworkPathError(message)
        self.manifest = _validate_network_layout(resolved_root)
        self.generation_root = (
            resolved_root
            / NETWORK_GENERATIONS_NAME
            / str(self.manifest["generation_id"])
        )
        super().__init__(resolved_root)
        self._product_index: Any = None
        index = self.interferograms.index_metadata if self.interferograms else None
        if index is None:
            message = "Network interferograms have no canonical index"
            logger.error(message)
            raise IncompleteNetworkProductError(message)
        index_type = index.get("type", index.get("index_type"))
        if index_type != NETWORK_INDEX_TYPE:
            message = f"unknown Network index type {index_type!r}"
            logger.error(message)
            raise UnknownNetworkIndexTypeError(message)
        if not self.interferograms.pairs().names:
            message = "Network interferogram index contains no products"
            logger.error(message)
            raise IncompleteNetworkProductError(message)

    @classmethod
    def from_path(cls, root: str | PathLike[str]) -> Self:
        """Construct a Network from a canonical filesystem path."""
        return cls(root)

    @property
    def product_index(self) -> Any:
        """Return the validated logical product index, when registered."""
        return self._product_index

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
            logger.error(message)
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
            logger.error(message)
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


__all__ = [
    "ExternalNetworkLayoutError",
    "GAMMANetwork",
    "GMTSARNetwork",
    "ISCE2Network",
    "ISCE3Network",
    "IncompleteNetworkProductError",
    "IncompleteNetworkError",
    "LegacyLayoutError",
    "LegacyNetworkLayoutError",
    "Network",
    "NetworkConstructionError",
    "NetworkAnalysisError",
    "NetworkCurrentError",
    "NetworkGenerationError",
    "NetworkLayoutError",
    "NetworkManifestError",
    "NetworkPathError",
    "SNAPNetwork",
    "UnknownNetworkIndexTypeError",
]

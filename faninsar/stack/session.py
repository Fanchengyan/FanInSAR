"""Mission-neutral Stack session (PROPOSAL-0017).

Orchestrates Reference-relative coregistration and pair products. Co-registration
and interferogram formation are separate stages: coreg caches per-date SLCs;
``form_interferograms`` only reads those products.
"""

# The Stack session preserves established runtime exception messages while
# the public mask-plan seam performs strict validation at its boundary.
# ruff: noqa: EM101, TRY003

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, ParamSpec, Self, TypeVar

import numpy as np

from faninsar.core.network import (
    AcquisitionKey,
    AssetKind,
    AssetTransform,
    NetworkProduct,
    NetworkProductIndex,
    NetworkProductKey,
    PhaseConvention,
)
from faninsar.core.network import (
    Network as NetworkContract,
)
from faninsar.logging import setup_logger
from faninsar.network.network import Network
from faninsar.processing.coreg.misreg_network import (
    DateMisreg,
    MisregArc,
    invert_pair_misregistration,
)
from faninsar.processing.dem import RasterDEM, SourceDEM
from faninsar.processing.errors import (
    reject_invalid_state,
)
from faninsar.processing.interferometry.pair import validate_coherence_window
from faninsar.processing.interferometry.phase_filter import (
    FilterProvenance,
    GoldsteinWerner,
    PhaseFilter,
)
from faninsar.processing.unwrap.errors import UnwrapFailedError
from faninsar.processing.unwrap.irls import SpatialIRLS
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.config import (
    ActivationMode,
    CoregMode,
    EsdMethod,
    FlattenStage,
    StackConfig,
)
from faninsar.stack.mask_plan import MaskPlan, StageName
from faninsar.stack.provider import SourceHandle
from faninsar.stack.scene_store import (
    CoregisteredSceneStore,
    copy_reference_units,
    form_merged_scene_interferogram,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from faninsar._core.device import GpuMemoryReclaim
    from faninsar.core.acquisition import Acquisition
    from faninsar.core.pairs import Pairs
    from faninsar.data.query import BoundingBox, Polygons
    from faninsar.processing.contracts.prepared_geometry import (
        ActivationToken,
        StackActivationBinding,
    )
    from faninsar.processing.dem import DEM, GridSpec
    from faninsar.processing.merge.grid import GeoGridSpec
    from faninsar.processing.resources import ResourceBudget
    from faninsar.processing.stages import (
        BurstSelection,
        CoregistrationGrid,
        ProductionPairState,
    )
    from faninsar.processing.timeseries.inversion import TimeSeriesResult
    from faninsar.processing.unwrap.common import SpatialUnwrapper
    from faninsar.processing.unwrap.quality import StackQualityCriteria
    from faninsar.processing.unwrap.stack import SpatialExecutor, StackUnwrapResult
    from faninsar.stack.ifg_store import (
        InterferogramArtifactStore,
        UnwrappedArtifact,
    )
    from faninsar.stack.provider import StackSceneProvider
    from faninsar.stack.stack_generation import StackResultGeneration

logger = setup_logger(__name__)

_DEFAULT_PHASE_FILTER = GoldsteinWerner(alpha=0.5, patch_size=32)
_DEFAULT_UNWRAPPER = SpatialIRLS()


def _phase_filter_metadata(  # noqa: PLR0911
    phase_filter: PhaseFilter | None,
) -> tuple[str, dict[str, int | float | bool | str | None]]:
    """Return bounded inert provenance for a runtime phase filter.

    A custom filter is trusted for execution but its description is never
    required for persistence.  Missing, raising, or malformed descriptions
    therefore degrade to the stable ``custom``/empty metadata pair.
    """
    if phase_filter is None:
        return "none", {}
    try:
        description = phase_filter.describe()
    except Exception as error:  # pragma: no cover - defensive custom boundary
        logger.warning(
            "phase filter description failed; using custom metadata: %s", error
        )
        return "custom", {}
    if not isinstance(description, FilterProvenance):
        logger.warning("phase filter description is malformed; using custom metadata")
        return "custom", {}
    name = description.name
    parameters = description.parameters
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 64
        or not name.isascii()
        or any(not (character.isalnum() or character in "-_.") for character in name)
        or not isinstance(parameters, dict)
        or len(parameters) > 32
    ):
        logger.warning(
            "phase filter description is out of bounds; using custom metadata"
        )
        return "custom", {}
    for key, value in parameters.items():
        if not isinstance(key, str) or len(key) > 64 or not key.isascii():
            logger.warning(
                "phase filter parameter key is invalid; using custom metadata"
            )
            return "custom", {}
        if isinstance(value, float) and not np.isfinite(value):
            logger.warning(
                "phase filter parameter is non-finite; using custom metadata"
            )
            return "custom", {}
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            logger.warning(
                "phase filter parameter is not a JSON scalar; using custom metadata"
            )
            return "custom", {}
        if isinstance(value, str) and len(value) > 256:
            logger.warning(
                "phase filter parameter string is too long; using custom metadata"
            )
            return "custom", {}
    return name, dict(parameters)


def _raise_unwrap_failed(
    message: str,
    result: Any | None = None,
) -> NoReturn:
    """Raise the typed Stack boundary error after logging its cause."""
    logger.error(message)
    raise UnwrapFailedError(message, result)


# ---------------------------------------------------------------------------
# PROPOSAL-0040 canonical mask-plan integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EffectiveRoi:
    """Mask-aware effective-ROI resolution for one Stack run.

    Attributes
    ----------
    roi
        ROI feeding the existing burst-selection path: the original ROI when
        the mask is inactive or unresolvable, a :class:`~faninsar.data.query.\
Polygons` wrapping the subtracted geometry when reshaped, or ``None``
        when no ROI was configured.
    geometry
        EPSG:4326 shapely geometry of the effective ROI (canonical coreg
        identity input), or ``None`` when the ROI was not reshaped.
    lineage
        Run-manifest record: ``{}`` for unmasked runs, ``{"mask": "absent",
        "reason": ...}`` after a degraded resolution, and ``{"mask":
        "present", ...}`` with provenance after a resolved mask.

    """

    roi: object | None
    geometry: object | None
    lineage: dict[str, object]


def _roi_to_geometry(roi: object) -> Any:
    """Convert a Stack ROI into one EPSG:4326 shapely geometry.

    Mirrors ``faninsar.processing.stages._roi_geometry`` so the
    mask subtraction operates on exactly the geometry the burst-selection
    path consumes.
    """
    from shapely.geometry import box

    from faninsar.data.query import BoundingBox

    if isinstance(roi, BoundingBox):
        return box(roi.left, roi.bottom, roi.right, roi.top)
    series = roi.geometry  # Polygons
    union = series.union_all() if hasattr(series, "union_all") else series.unary_union
    crs = getattr(roi, "crs", None)
    if crs is not None and str(crs) != "EPSG:4326":
        import geopandas as gpd

        return gpd.GeoSeries([union], crs=crs).to_crs("EPSG:4326").iloc[0]
    return union


def _polygonal_parts(geometry: Any) -> list[Any]:
    """Return the non-empty polygonal parts of a difference result."""
    from shapely.geometry import MultiPolygon, Polygon

    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return [geometry]
    return [
        part
        for part in getattr(geometry, "geoms", ())
        if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
    ]


def _effective_roi_polygons(parts: list[Any]) -> object:
    """Wrap polygonal effective-ROI parts into a Polygons object."""
    import geopandas as gpd

    from faninsar.data.query import Polygons

    return Polygons(
        gpd.GeoDataFrame(geometry=parts, crs="EPSG:4326"),
        types="desired",
        crs="EPSG:4326",
    )


def _mask_vector_for_roi(plan: MaskPlan, roi: object) -> object | None:
    """Materialize the explicit ROI stage and return excluded geometry only."""
    refs = plan.references("roi")
    if not refs:
        return None
    from faninsar.processing.masking.mask import Mask, VectorMask

    bounds = _roi_to_geometry(roi)
    parts: list[VectorMask] = []
    for definition in plan.for_stage("roi"):
        if definition.kind == "raster":
            # Raster ROI masks require a target grid and therefore cannot be
            # used for geometric subtraction at this seam.
            raise ValueError("ROI stage raster masks require an explicit grid")
        if definition.kind == "vector":
            import geopandas as gpd
            import pyproj
            import shapely.ops

            frame = gpd.read_file(definition.path)
            vector = Mask.from_vector(frame).to_vector()
            transformer = pyproj.Transformer.from_crs(
                vector.crs, "EPSG:4326", always_xy=True
            )
            projected = [
                shapely.ops.transform(transformer.transform, geometry)
                for geometry in vector.geometry
            ]
            clipped = [geometry.intersection(bounds) for geometry in projected]
            parts.append(
                VectorMask(
                    clipped,
                    crs="EPSG:4326",
                    roles=vector.roles,
                    categories=vector.categories,
                    provenance=vector.provenance,
                )
            )
        else:
            recipe = Mask.from_water(
                bounds=tuple(float(value) for value in bounds.bounds),
                provider=definition.provider or "auto",
            )
            parts.append(recipe.to_vector(bounds))
    geometries = [
        geometry
        for vector in parts
        for geometry, role in zip(vector.geometry, vector.roles, strict=True)
        if role == "excluded"
    ]
    if not geometries:
        return None
    return Mask.from_vector(geometries, crs=parts[0].crs).to_vector()


def _resolve_effective_roi(
    roi: BoundingBox | Polygons | None,
    *,
    mask_plan: MaskPlan,
) -> _EffectiveRoi:
    """Subtract only explicitly excluded ROI-stage geometry from the ROI."""
    if roi is None or not mask_plan.references("roi"):
        return _EffectiveRoi(roi=roi, geometry=None, lineage={})
    import shapely

    roi_geometry = _roi_to_geometry(roi)
    excluded = _mask_vector_for_roi(mask_plan, roi)
    if excluded is None or not excluded.geometry:
        return _EffectiveRoi(roi=roi, geometry=None, lineage={})
    geometries = [
        geometry
        for geometry, role in zip(excluded.geometry, excluded.roles, strict=True)
        if role == "excluded"
    ]
    difference = roi_geometry.difference(shapely.union_all(geometries))
    parts = _polygonal_parts(difference)
    if not parts:
        reject_invalid_state("ROI is empty after applying excluded mask geometry")
    return _EffectiveRoi(
        roi=_effective_roi_polygons(parts),
        geometry=parts[0] if len(parts) == 1 else shapely.union_all(parts),
        lineage={"mask_plan": mask_plan.identity},
    )


# ---------------------------------------------------------------------------
# PROPOSAL-0039 product-level mask application (Stack integration, Slice D2)
# ---------------------------------------------------------------------------


def _grid_crs_epsg(grid: Any) -> int | None:
    """Return the EPSG code of a GeoGridSpec CRS.

    ``None`` when the CRS is unresolvable.
    """
    try:
        from pyproj import CRS

        return CRS.from_user_input(grid.crs).to_epsg()
    except Exception as error:  # pragma: no cover - defensive CRS boundary
        logger.debug("geo grid CRS %r is unresolvable: %s", grid.crs, error)
        return None


def _mask_removed_plane(
    mask: object,
    *,
    transform: Any,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a nearest-neighbour canonical plane on one geographic grid."""
    from faninsar.processing.masking.mask import GridSpec

    target = GridSpec("EPSG:4326", transform, shape=shape)
    result = mask.to_raster(target)
    return np.asarray(result.data, dtype=np.uint8)


def _apply_mask_to_valid_mask(
    valid_mask: np.ndarray | None,
    mask_plane: np.ndarray | None,
) -> np.ndarray | None:
    """Intersect a support mask with the inverted mask (``valid &= ~mask``).

    The mask is a *support* input (PROPOSAL-0038): only plane value ``1``
    (water/removed) removes support and the intersection never touches the
    complex/phase numerics — in particular the mask is never routed through a
    :class:`~faninsar.processing.interferometry.phase_filter.PhaseFilter`.
    ``mask_plane=None`` (mask absent) returns ``valid_mask`` unchanged.
    """
    if mask_plane is None:
        return valid_mask
    removed = np.asarray(mask_plane) != 0
    if valid_mask is None:
        return ~removed
    intersected = np.asarray(valid_mask, dtype=bool).copy()
    intersected &= ~removed
    return intersected


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _callable_identity(callback: object | None) -> dict[str, str] | None:
    """Return the stable import identity of a callback, when available.

    Parameters
    ----------
    callback : object, optional
        Provider callback or callable object.

    Returns
    -------
    dict[str, str] or None
        Module and qualified name, without serializing the callable repr.

    """
    if callback is None:
        return None
    return {
        "module": str(getattr(callback, "__module__", type(callback).__module__)),
        "qualname": str(getattr(callback, "__qualname__", type(callback).__qualname__)),
    }


def _provider_callback(provider: object | None) -> object | None:
    """Return a provider's production callback without inspecting its repr."""
    if provider is None:
        return None
    callback = getattr(provider, "produce_pair", None)
    if callback is None and callable(provider):
        callback = provider
    return callback


def _distribution_version(names: tuple[str, ...]) -> str | None:
    """Read the first installed distribution version from a name allowlist."""
    for name in names:
        try:
            return str(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            logger.debug("Unable to inspect distribution version: %s", name)
            continue
    return None


def _loaded_module_version(names: tuple[str, ...]) -> str | None:
    """Read a version from an already-loaded optional module, if exposed."""
    for name in names:
        module = sys.modules.get(name)
        value = getattr(module, "__version__", None) if module is not None else None
        if value is not None:
            return str(value)
    return None


def _faninsar_git_revision() -> str | None:
    """Return the enclosing FanInSAR Git revision, when the checkout exposes it."""
    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _runtime_image_identity() -> dict[str, object]:
    """Return bounded platform metadata identifying the runtime image."""
    os_release: dict[str, str] = {}
    try:
        raw_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        raw_release = {}
    for name in ("ID", "VERSION_ID", "BUILD_ID", "IMAGE_ID", "IMAGE_VERSION"):
        value = raw_release.get(name)
        if value:
            os_release[name.lower()] = str(value)
    return {
        "platform": platform.platform(aliased=True, terse=True),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "os_release": os_release,
    }


def _cuda_runtime_identity() -> dict[str, object]:
    """Return serializable CUDA availability and physical-device identity."""
    identity: dict[str, object] = {
        "available": False,
        "driver": None,
        "runtime": None,
        "devices": [],
    }
    try:
        import torch
    except Exception:
        return identity
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return identity
    try:
        available = bool(cuda.is_available())
    except Exception:
        return identity
    identity["available"] = available
    version = getattr(torch, "version", None)
    cuda_runtime = getattr(version, "cuda", None)
    identity["runtime"] = None if cuda_runtime is None else str(cuda_runtime)
    driver = getattr(cuda, "driver_version", None)
    if driver is None:
        driver = getattr(cuda, "get_driver_version", None)
    try:
        if callable(driver):
            driver = driver()
    except Exception:
        driver = None
    identity["driver"] = None if driver is None else str(driver)
    if not available:
        return identity
    try:
        count = int(cuda.device_count())
    except Exception:
        return identity
    devices: list[dict[str, object]] = []
    for index in range(max(0, count)):
        try:
            properties = cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "uuid": (
                        None
                        if getattr(properties, "uuid", None) is None
                        else str(properties.uuid)
                    ),
                    "name": (
                        None
                        if getattr(properties, "name", None) is None
                        else str(properties.name)
                    ),
                }
            )
        except Exception:
            # An unavailable device is still represented by the count and the
            # availability bit; never let diagnostics prevent S1 execution.
            devices.append({"index": index, "uuid": None, "name": None})
    identity["devices"] = devices
    return identity


def _stack_runtime_identity(callback: object | None) -> dict[str, object]:
    """Build the canonical runtime identity used by all Stack resume stages."""
    compiler = sysconfig.get_config_var("CC")
    compiler_name = None
    if compiler:
        compiler_name = Path(str(compiler).split()[0]).name
    return {
        "faninsar_git_revision": _faninsar_git_revision(),
        "versions": {
            "faninsar": _distribution_version(("FanInSAR", "faninsar")),
            "nisar": _distribution_version(("nisar", "nisar-products"))
            or _loaded_module_version(("nisar", "nisar.products")),
            "isce3": _distribution_version(("isce3", "isce3-python"))
            or _loaded_module_version(("isce3",)),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "compiler": {
            "python": platform.python_compiler(),
            "python_build": list(platform.python_build()),
            "c_compiler": compiler_name,
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
        },
        "runtime_image": _runtime_image_identity(),
        "cuda": _cuda_runtime_identity(),
        "callback": _callable_identity(callback),
    }


def _runtime_fingerprint(callback: object | None) -> str:
    """Return the canonical digest of one Stack execution runtime."""
    payload = _stack_runtime_identity(callback)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        logger.exception("Stack runtime identity cannot be canonicalized")
        reject_invalid_state(f"Stack runtime identity cannot be canonicalized: {error}")
    return hashlib.sha256(encoded).hexdigest()


def _phase_screen_lineage(
    primary_store: CoregisteredSceneStore,
    secondary_store: CoregisteredSceneStore,
    flatten_stage: str,
) -> tuple[str | None, dict[str, dict[str, str]]]:
    """Derive explicit phase-screen lineage from persisted scene units.

    Parameters
    ----------
    primary_store, secondary_store : CoregisteredSceneStore
        Source scene generations for the IFG pair.
    flatten_stage : str
        Validated flattening stage shared by both source generations.

    Returns
    -------
    tuple[str or None, dict[str, dict[str, str]]]
        Phase-screen model and role/unit payload digests.  Coregistration
        artifacts deliberately report no model or screen digest.

    """
    if flatten_stage == "coregistration":
        return None, {}

    stores = (("primary", primary_store), ("secondary", secondary_store))
    models: set[str] = set()
    digests: dict[str, dict[str, str]] = {}
    for role, store in stores:
        role_digests = {
            unit.tag: unit.phase_screen_digest
            for unit in store.units
            if unit.phase_screen_digest is not None
        }
        if role_digests:
            digests[role] = {
                tag: digest
                for tag, digest in sorted(role_digests.items())
                if digest is not None
            }
        for unit in store.units:
            state = unit.phase_state or {}
            model = state.get("range_offset_flatten", state.get("phase_screen_model"))
            if model is not None:
                models.add(str(model))
    if len(models) > 1:
        reject_invalid_state("scene units mix phase-screen models")
    model = next(iter(models), None)
    if model is None and digests:
        reject_invalid_state("phase-screen digests have no declared model")
    return model, digests


def _reclaim_after_stage(
    method: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Run reclaim_checkpoint(kind=stage) when a public Stack method returns."""

    @wraps(method)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        self = args[0]
        try:
            return method(*args, **kwargs)
        finally:
            self._reclaim_accelerator("stage")

    return wrapped


def _atomic_write_array(path: Path, array: np.ndarray) -> None:
    """Publish one binary array only after its bytes are durable."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.asarray(array).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish one text manifest through a same-directory durable rename."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _date_to_yyyymmdd(value: object) -> str:
    """Normalize timestamps / strings to ``YYYYMMDD``."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")  # type: ignore[union-attr]
    text = str(value).replace("-", "")[:8]
    if len(text) != 8 or not text.isdigit():
        reject_invalid_state(f"cannot parse date id from {value!r}")
    return text


def _pairs_from_factory(
    dates: Sequence[str],
    *,
    max_interval: int = 3,
    max_days: int = 72,
) -> Pairs:
    """Build short-baseline pairs from date ids."""
    from faninsar.core.pairs import PairsFactory

    factory = PairsFactory(list(dates))
    return factory.from_interval(max_interval=max_interval, max_days=max_days)


@dataclass
class Stack(NetworkContract):
    """Multi-scene InSAR session with Reference-relative coregistration.

    Parameters
    ----------
    catalog : SceneCatalog
        Date → product paths.
    config : StackConfig
        Session defaults (work_dir, coreg_mode, multilook, …).
    pairs : Pairs, optional
        Interferogram network. Default: short-baseline auto.
    misreg_pairs : Pairs, optional
        Misregistration measurement network. Default: shorter auto subset.
    reference : str, optional
        Reference date id. Default: earliest catalog date.
    acquisitions : Acquisition, optional
        Optional domain Acquisition index (informational).

    Notes
    -----
    ``grid`` is the authoritative shared output grid for DEMs, masks, and
    geographic products. An explicit ``StackConfig.grid`` always wins. With
    ``grid="auto"``, ROI precedence is an explicit ``resolve_grid(roi=...)``
    override, then ``StackConfig.roi``, then the deterministic union of
    selected acquisition/swath/burst footprints. The centre chooses UTM or
    UPS. Cross-zone, UTM/UPS-boundary, antimeridian, and large projected
    extents warn and continue; an explicit seam-crossing ROI fails before
    provider planning or allocation. Resource limits still fail closed.

    """

    catalog: SceneCatalog
    config: StackConfig
    pairs: Pairs
    misreg_pairs: Pairs
    reference: str
    acquisitions: Acquisition | None = None
    dask_client: Any | None = field(default=None, repr=False)
    scene_provider: StackSceneProvider | None = field(default=None, repr=False)
    arcs: list[MisregArc] = field(default_factory=list)
    date_misreg: DateMisreg | None = None
    coreg_paths: dict[str, Path] = field(default_factory=dict)
    pair_states: dict[str, ProductionPairState] = field(default_factory=dict)
    ifg_dirs: list[Path] = field(default_factory=list)
    timeseries: TimeSeriesResult | None = None
    unwrap_result: StackUnwrapResult | None = None
    _prepared: bool = False
    _generation: StackResultGeneration | None = field(default=None, repr=False)
    _unwrap_generation: Any | None = field(default=None, repr=False)
    _network_generation_id: str | None = field(default=None, repr=False)
    _network_product_index: NetworkProductIndex | None = field(default=None, repr=False)
    _network_view: Network | None = field(default=None, repr=False)
    _effective_roi_resolution: _EffectiveRoi | None = field(default=None, repr=False)
    _materialized_masks: dict[tuple[tuple[str, ...], str], object] = field(
        default_factory=dict, repr=False
    )
    _radar_projection_context: dict[str, object] | None = field(
        default=None, repr=False
    )
    _radar_projected_masks: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False
    )
    _radar_mask_identities: dict[tuple[str, tuple[int, int], tuple[int, int]], str] = (
        field(default_factory=dict, repr=False)
    )

    def __post_init__(self) -> None:
        """Initialize the inherited Network analysis surface lazily.

        A raw Stack has no on-disk Network products until interferogram
        formation commits them.  The Dataset-backed members are therefore
        initialized as empty views and are populated only by a future
        generation refresh; this keeps raw SAFE discovery out of Network.
        """
        self._root = self.config.work_dir
        self._geometry = None
        self._interferograms = None
        self._timeseries = None
        self._product_index = None

    @property
    def network(self) -> Network | None:
        """Return this Stack's validated Network view.

        The Stack and Network share one generation-scoped product index;
        before publication the view is intentionally unavailable.
        """
        if not self.analysis_ready:
            return None
        if (
            self._network_view is None
            or self._network_view.network_generation_id != self._network_generation_id
        ):
            view = object.__new__(Network)
            view._root = self._root
            view._geometry = self._geometry
            view._interferograms = self._interferograms
            view._timeseries = self._timeseries
            view._product_index = self._network_product_index
            view._network_generation_id = self._network_generation_id
            view._network_product_index = self._network_product_index
            view.manifest = {"generation_id": self._network_generation_id}
            view.generation_root = self._root
            self._network_view = view
        return self._network_view

    @property
    def grid(self) -> GridSpec:
        """Return the authoritative explicit or automatically selected grid.

        ``StackConfig.grid`` always wins when it contains a :class:`GridSpec`.
        Automatic selection uses the configured ROI, or the deterministic
        union of selected acquisition/swath/burst footprints, and performs no
        provider I/O.  Center-based UTM/UPS selection continues with a warning
        when a footprint crosses a seam; callers should provide an explicit
        grid for a multi-zone or polar-boundary study.
        """
        return self.resolve_grid()

    def resolve_grid(self, roi: object | None = None) -> GridSpec:
        """Resolve the Stack output grid from config or a WGS84 ROI.

        Parameters
        ----------
        roi : object, optional
            Footprint/ROI override.  It is treated as caller-explicit and an
            antimeridian crossing therefore fails before source I/O.

        Notes
        -----
        When both ``roi`` and ``StackConfig.roi`` are absent, selected
        acquisition, swath, or burst footprints supplied by the adapter are
        unioned deterministically from ``StackConfig.extra``.

        """
        from faninsar.stack.grid import resolve_stack_grid

        selected_roi = self.config.roi if roi is None else roi
        if selected_roi is None:
            selected_roi = self._selected_footprint_roi()
        return resolve_stack_grid(
            self.config.grid,
            roi=selected_roi,
            resolution_m=self.config.resolution_m,
            budget=self.config.resource_budget,
            explicit_roi=roi is not None or self.config.roi is not None,
        )

    def _selected_footprint_roi(self) -> object | None:
        """Return the deterministic union of selected acquisition footprints.

        Adapters may expose selected acquisition, swath, or burst footprints
        through ``StackConfig.extra``.  This discovery is intentionally
        read-only and happens after adapter selection; an explicit ``roi``
        remains authoritative.  The result is a WGS84 Shapely geometry so the
        automatic UTM/UPS resolver can apply its center and seam policy.
        """
        values: object | None = None
        for key in (
            "selected_footprints",
            "burst_footprints",
            "swath_footprints",
            "acquisition_footprints",
            "footprints",
        ):
            candidate = self.config.extra.get(key)
            if candidate is not None:
                values = candidate
                break
        if values is None:
            return None
        from shapely.geometry import Polygon, box, shape
        from shapely.ops import unary_union

        if isinstance(values, (str, bytes)):
            values = (values,)
        if not isinstance(values, Iterable) or isinstance(values, dict):
            values = (values,)
        geometries: list[Any] = []
        for value in values:
            geometry = getattr(value, "geometry", value)
            if hasattr(value, "footprint") and value.footprint is not None:
                geometry = value.footprint
            if hasattr(geometry, "__geo_interface__"):
                geometry = shape(geometry.__geo_interface__)
            elif isinstance(geometry, dict) and "type" in geometry:
                geometry = shape(geometry)
            elif isinstance(geometry, (tuple, list)):
                numbers = tuple(geometry)
                if len(numbers) == 4 and all(
                    isinstance(item, (int, float)) for item in numbers
                ):
                    geometry = box(*map(float, numbers))
                else:
                    geometry = Polygon(numbers)
            if geometry is not None and not geometry.is_empty:
                geometries.append(geometry)
        if not geometries:
            return None
        return unary_union(sorted(geometries, key=lambda geometry: geometry.wkb))

    def materialize_dem(self, dem: DEM | None = None) -> RasterDEM:
        """Materialize one DEM directly on the authoritative Stack grid.

        The source recipe remains inert until this method is called.  Stack
        supplies its work-directory cache when a source recipe needs it.
        """
        from faninsar.processing import dem as dem_api

        selected = dem or self.config.dem
        if selected is None:
            selected = dem_api.DEM.from_source("auto")
        cache_dir = self.config.dem_cache_dir or (self.config.work_dir / "dem-cache")
        if hasattr(selected, "cache_dir") and selected.cache_dir is None:
            selected.cache_dir = cache_dir  # type: ignore[attr-defined]
        return selected.to_raster(
            self.grid,
            vertical_datum="ellipsoidal",
            budget=self.config.resource_budget,
        )

    @classmethod
    def from_safes(
        cls,
        paths: Sequence[str | Path],
        *,
        reference: str | None = None,
        **kwargs: Any,
    ) -> Stack:
        """Construct the concrete Sentinel-1 adapter from SAFE sources.

        Raw-source discovery is an adapter concern.  Keeping this compatibility
        spelling on ``Stack`` lets older callers migrate without giving the
        mission-neutral session a production implementation of its own.
        """
        from faninsar.stack.s1 import S1Stack

        if "master" in kwargs:
            from faninsar.processing.errors import reject_pair_configuration

            reject_pair_configuration(
                "Stack.from_safes no longer accepts 'master'; use 'reference'"
            )
        if reference is not None:
            kwargs["reference"] = reference
        return S1Stack.from_safes(paths, **kwargs)

    @classmethod
    def _from_safes(
        cls,
        paths: Sequence[str | Path],
        *,
        work_dir: str | Path,
        pairs: Pairs | None = None,
        misreg_pairs: Pairs | None = None,
        reference: str | None = None,
        dem: DEM | None = None,
        geo_grid: GeoGridSpec | None = None,
        grid: GridSpec | Literal["auto"] = "auto",
        resolution_m: float = 30.0,
        dem_cache_dir: str | Path | None = None,
        roi: BoundingBox | Polygons | None = None,
        mask_plan: MaskPlan | None = None,
        coreg_mode: CoregMode = "pair",
        flatten_stage: FlattenStage = "coregistration",
        coregistration_grid: CoregistrationGrid = "radar",
        multilook: tuple[int, int] = (5, 2),
        goldstein_alpha: float = 0.5,
        esd_method: EsdMethod = "auto",
        swaths: tuple[str, ...] = ("IW1",),
        bursts: BurstSelection | None = None,
        executor: str = "torch",
        device: str = "auto",
        dask_client: Any | None = None,
        invert_device: str = "cpu",
        pair_max_interval: int = 3,
        pair_max_days: int = 72,
        misreg_max_interval: int = 2,
        misreg_max_days: int = 36,
        activation_mode: ActivationMode = "reference",
        activation_binding: StackActivationBinding | None = None,
        activation_token: ActivationToken | None = None,
        activation_authority_root: str | Path | None = None,
        retain_pair_states: bool = False,
        record_scientific_lineage: bool = False,
        gpu_memory_reclaim: GpuMemoryReclaim = "adaptive",
        resource_budget: ResourceBudget | None = None,
    ) -> Stack:
        """Construct a Stack from SAFE paths and optional pair graphs."""
        catalog = SceneCatalog.from_paths(list(paths))
        dates = list(catalog.dates)
        reference_id = dates[0] if reference is None else _date_to_yyyymmdd(reference)
        if reference_id not in catalog.paths:
            reject_invalid_state(f"reference {reference_id} not in catalog")
        ifg_pairs = pairs or _pairs_from_factory(
            dates,
            max_interval=pair_max_interval,
            max_days=pair_max_days,
        )
        m_pairs = misreg_pairs or _pairs_from_factory(
            dates,
            max_interval=misreg_max_interval,
            max_days=misreg_max_days,
        )
        from faninsar.core.acquisition import Acquisition

        acq = Acquisition(
            sorted({_date_to_yyyymmdd(d) for d in dates}),
        )
        config = StackConfig(
            work_dir=Path(work_dir),
            coreg_mode=coreg_mode,
            flatten_stage=flatten_stage,
            coregistration_grid=coregistration_grid,
            esd_method=esd_method,
            multilook=multilook,
            goldstein_alpha=goldstein_alpha,
            executor=executor,
            device=device,
            invert_device=invert_device,
            dem=dem,
            geo_grid=geo_grid,
            grid=grid,
            resolution_m=resolution_m,
            dem_cache_dir=(Path(dem_cache_dir) if dem_cache_dir is not None else None),
            roi=roi,
            mask_plan=mask_plan or MaskPlan(),
            swaths=swaths,
            bursts=bursts,
            activation_mode=activation_mode,
            activation_binding=activation_binding,
            activation_token=activation_token,
            activation_authority_root=(
                Path(activation_authority_root)
                if activation_authority_root is not None
                else None
            ),
            retain_pair_states=retain_pair_states,
            record_scientific_lineage=record_scientific_lineage,
            gpu_memory_reclaim=gpu_memory_reclaim,
            resource_budget=resource_budget,
        )
        return cls(
            catalog=catalog,
            config=config,
            pairs=ifg_pairs,
            misreg_pairs=m_pairs,
            reference=_date_to_yyyymmdd(reference_id),
            acquisitions=acq,
            dask_client=dask_client,
        )

    @_reclaim_after_stage
    def prepare_scenes(self) -> Self:
        """Create work directories and validate the Reference acquisition."""
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        (self.config.work_dir / "coreg").mkdir(exist_ok=True)
        (self.config.work_dir / "misreg").mkdir(exist_ok=True)
        (self.config.work_dir / "ifg").mkdir(exist_ok=True)
        (self.config.work_dir / "pairs").mkdir(exist_ok=True)
        if self.reference not in self.catalog.paths:
            reject_invalid_state(f"reference {self.reference} missing from catalog")
        self._prepared = True
        logger.info(
            "Stack prepared reference=%s n_scenes=%s n_pairs=%s "
            "n_misreg_pairs=%s mode=%s",
            self.reference,
            len(self.catalog),
            len(self.pairs),
            len(self.misreg_pairs),
            self.config.coreg_mode,
        )
        return self

    def _ensure_prepared(self) -> None:
        if not self._prepared:
            self.prepare_scenes()

    def _write_qualified_activation_record(self) -> None:
        """Publish the typed P19 activation record after scene preparation."""
        if self.config.activation_mode != "qualified":
            return
        binding = self.config.activation_binding
        token = self.config.activation_token
        if binding is None:
            reject_invalid_state("qualified Stack activation binding is missing")
        if token is None:
            reject_invalid_state("qualified Stack activation token is missing")
        authority_root = self.config.activation_authority_root
        if authority_root is None:
            reject_invalid_state("qualified Stack activation authority is missing")
        from faninsar.stack.activation import LocalActivationAuthority

        authority = LocalActivationAuthority.open(authority_root)
        _ = authority.verify_token(token)
        if binding.activation_token_digest != token.digest():
            reject_invalid_state("activation token does not match binding digest")
        if token.parent_id != binding.stack_generation_id:
            reject_invalid_state(
                "activation token parent does not match Stack generation"
            )
        if token.provider_receipt_digest != binding.qualification_receipt_digest:
            reject_invalid_state("activation token receipt does not match binding")
        if token.p18_stack_gate_event_id != binding.p18_stack_gate_event_id:
            reject_invalid_state("activation token Stack gate does not match binding")
        if token.p19_qualified_event_ids != binding.p19_qualified_event_ids:
            reject_invalid_state(
                "activation token event collection does not match binding"
            )
        record = {
            "schema_version": "scene_artifact_v1",
            "activation_mode": "qualified",
            "stack_generation_id": binding.stack_generation_id,
            "binding": asdict(binding),
            "activation_token": asdict(token),
            "activation_token_digest": token.digest(),
            "scene_dates": list(self.catalog.dates),
            "reference": self.reference,
        }
        path = self.config.work_dir / "activation" / "scene_artifact_v1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(record, sort_keys=True, indent=2) + "\n")

    def _require_qualified_activation_record(self) -> None:
        """Validate the immutable activation record before production IFGs."""
        if self.config.activation_mode != "qualified":
            return
        binding = self.config.activation_binding
        token = self.config.activation_token
        path = self.config.work_dir / "activation" / "scene_artifact_v1.json"
        if binding is None or token is None or not path.is_file():
            reject_invalid_state("qualified Stack activation record is missing")
        authority_root = self.config.activation_authority_root
        if authority_root is None:
            reject_invalid_state("qualified Stack activation authority is missing")
        from faninsar.stack.activation import LocalActivationAuthority

        authority = LocalActivationAuthority.open(authority_root)
        _ = authority.verify_token(token)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            reject_invalid_state(
                f"qualified Stack activation record is invalid: {error}"
            )
        if record.get("schema_version") != "scene_artifact_v1":
            reject_invalid_state("unsupported Stack activation record schema")
        expected_binding = json.loads(json.dumps(asdict(binding)))
        expected_token = json.loads(json.dumps(asdict(token)))
        if record.get("binding") != expected_binding:
            reject_invalid_state("Stack activation binding does not match config")
        if record.get("activation_token") != expected_token:
            reject_invalid_state("Stack activation token does not match config")
        if record.get("activation_token_digest") != token.digest():
            reject_invalid_state("Stack activation token digest is invalid")
        if record.get("reference") != self.reference:
            reject_invalid_state("Stack activation Reference does not match config")
        if record.get("scene_dates") != list(self.catalog.dates):
            reject_invalid_state("Stack activation scene set does not match catalog")

    def _effective_roi_with_mask(self) -> _EffectiveRoi:
        """Resolve the explicit ROI mask stage once per Stack run."""
        if self._effective_roi_resolution is None:
            self._effective_roi_resolution = _resolve_effective_roi(
                self.config.roi, mask_plan=self.config.mask_plan
            )
        return self._effective_roi_resolution

    def _mask_lineage_record(self) -> dict[str, object]:
        """Return normalized mask-plan lineage for run manifests."""
        return self._effective_roi_with_mask().lineage

    def _ifg_geo_grid(
        self, looks: tuple[int, int]
    ) -> tuple[Any, tuple[int, int]] | None:
        """Return the multilooked ``(transform, shape)`` of the geo IFG grid.

        The shared study-area :attr:`StackConfig.geo_grid` is the only
        geo-referenced grid the Stack session carries; multilooking scales
        the pixel size by the look factors and keeps the grid origin (the
        scene-grid alignment convention of ``form_interferograms``).
        ``None`` when no geo grid is configured or its CRS is not EPSG:4326
        (the v1 water pipeline is geographic-only, fail-closed).
        """
        from affine import Affine

        grid = self.config.geo_grid
        if grid is None:
            return None
        if _grid_crs_epsg(grid) != 4326:
            return None
        gdal_x0, gdal_dx, gdal_rx, gdal_y0, gdal_ry, gdal_dy = grid.transform
        az_looks, rg_looks = (int(looks[0]), int(looks[1]))
        transform = Affine(
            float(gdal_dx) * rg_looks,
            float(gdal_rx),
            float(gdal_x0),
            float(gdal_ry),
            float(gdal_dy) * az_looks,
            float(gdal_y0),
        )
        shape = (
            -(-int(grid.height) // az_looks),
            -(-int(grid.width) // rg_looks),
        )
        return transform, shape

    def _stage_mask_plane(
        self,
        *,
        stage: StageName,
        domain: str,
        looks: tuple[int, int],
        expected_shape: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        """Return one stage's canonical mask plane on its target domain/grid."""
        if not self.config.mask_plan.references(stage):
            return None
        if domain == "radar":
            from faninsar.processing.masking.mask import GridSpec
            from faninsar.processing.masking.radar_projection import (
                project_mask_to_radar,
            )

            context = self._radar_projection_context
            if context is None:
                configured = self.config.extra.get("radar_projection_context")
                context = configured if isinstance(configured, dict) else None
            master = None if context is None else context.get("master_grid")
            if master is None:
                master = self.config.geo_grid
            if master is None:
                message = (
                    "radar mask projection requires the authoritative geographic "
                    "mask GridSpec (configure StackConfig.geo_grid)"
                )
                logger.error(message)
                raise ValueError(message)
            if isinstance(master, dict):
                if not all(key in master for key in ("crs", "transform", "shape")):
                    message = "radar mask projection master GridSpec is incomplete"
                    logger.error(message)
                    raise ValueError(message)
                target = GridSpec(
                    master["crs"], master["transform"], shape=tuple(master["shape"])
                )
            elif all(hasattr(master, name) for name in ("crs", "transform", "shape")):
                target = GridSpec(master.crs, master.transform, shape=master.shape)
            else:
                message = "radar mask projection master GridSpec is incomplete"
                logger.error(message)
                raise ValueError(message)
            materialized = self._materialize_stage_mask(stage, target)
            if materialized is None:
                return None
            if context is None:
                message = (
                    "radar mask projection requires authoritative Stack radar "
                    "geometry or LUT; run coregister_scenes first"
                )
                logger.error(message)
                raise ValueError(message)
            geometry = context.get("geometry")
            lut = context.get("lut")
            full_shape = context.get("full_radar_shape")
            if full_shape is None:
                message = "radar mask projection context is missing full_radar_shape"
                logger.error(message)
                raise ValueError(message)
            if (geometry is None) == (lut is None):
                message = (
                    "radar mask projection context must provide exactly one "
                    "authoritative geometry or LUT"
                )
                logger.error(message)
                raise ValueError(message)
            from faninsar.processing.masking.radar_projection import (
                radar_projection_cache_key,
            )

            projection_key = radar_projection_cache_key(
                source_mask_identity=materialized.identity,
                master_grid=target,
                target_grid={
                    "shape": tuple(int(value) for value in full_shape),
                    "multilook": looks,
                },
                reference_scene=context.get("reference_scene", self.reference),
                lut=lut,
                geometry=geometry,
                dem=self.config.dem,
            )
            if projection_key not in self._radar_projected_masks:
                self._radar_projected_masks[projection_key] = project_mask_to_radar(
                    materialized,
                    full_radar_shape=tuple(int(value) for value in full_shape),
                    lut=lut,
                    geometry=geometry,
                    mask_transform=target.transform,
                    dem=self.config.dem,
                    device=self.config.device,
                    multilook=looks,
                    master_grid=target,
                    reference_scene=context.get("reference_scene", self.reference),
                )
            projected_shape = tuple(
                expected_shape or self._radar_projected_masks[projection_key].shape
            )
            self._radar_mask_identities[(stage, tuple(looks), projected_shape)] = (
                projection_key
            )
            projected = self._radar_projected_masks[projection_key]
            if expected_shape is not None and tuple(expected_shape) != projected.shape:
                message = (
                    f"projected radar mask shape {projected.shape} does not match "
                    f"IFG shape {expected_shape}"
                )
                logger.error(message)
                raise ValueError(message)
            return projected
        if domain != "geo":
            message = f"unsupported mask projection domain {domain!r}"
            logger.error(message)
            raise ValueError(message)
        grid = self._ifg_geo_grid(looks)
        if grid is None:
            raise ValueError("interferogram masks require an EPSG:4326 geo_grid")
        transform, shape = grid
        if expected_shape is not None and tuple(expected_shape) != shape:
            reject_invalid_state("mask target grid shape does not match IFG shape")
        from faninsar.processing.masking.mask import GridSpec

        target = GridSpec("EPSG:4326", transform, shape=shape)
        materialized = self._materialize_stage_mask(stage, target)
        return None if materialized is None else materialized.data

    def _ifg_mask_plane(
        self,
        *,
        domain: str,
        looks: tuple[int, int],
        expected_shape: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        """Return the interferogram-stage mask on its target grid."""
        return self._stage_mask_plane(
            stage="interferogram",
            domain=domain,
            looks=looks,
            expected_shape=expected_shape,
        )

    def _materialize_stage_mask(
        self, stage: StageName, target: object
    ) -> object | None:
        """Materialize, union, and cache one normalized stage mask exactly once."""
        refs = self.config.mask_plan.references(stage)
        if not refs:
            return None
        target_identity = hashlib.sha256(
            json.dumps(
                {
                    "crs": str(target.crs),
                    "transform": tuple(float(value) for value in target.transform),
                    "shape": tuple(int(value) for value in target.shape),
                    "validity": (
                        None
                        if target.validity is None
                        else np.asarray(target.validity, dtype=bool).tobytes().hex()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        # The identity is the normalized mask set plus target grid.  Stage is
        # deliberately excluded: the same declared set must materialize once
        # and be reusable when two stages share it.
        key = (refs, target_identity)
        if key in self._materialized_masks:
            return self._materialized_masks[key]
        from faninsar.processing.masking.mask import GridSpec, Mask

        if not isinstance(target, GridSpec):
            raise TypeError("mask targets must be GridSpec instances")

        def _water_mask(provider: str = "auto", policy: object | None = None) -> Mask:
            """Bind a deferred water mask to this Stack target grid."""
            bounds = target.bounds
            if target.crs != "EPSG:4326":
                import pyproj

                transformer = pyproj.Transformer.from_crs(
                    target.crs, "EPSG:4326", always_xy=True
                )
                corners = [
                    transformer.transform(x, y)
                    for x, y in (
                        (bounds[0], bounds[1]),
                        (bounds[0], bounds[3]),
                        (bounds[2], bounds[1]),
                        (bounds[2], bounds[3]),
                    )
                ]
                bounds = (
                    min(point[0] for point in corners),
                    min(point[1] for point in corners),
                    max(point[0] for point in corners),
                    max(point[1] for point in corners),
                )
            return Mask.from_water(
                bounds=bounds,
                provider=provider,
                policy=policy if policy is not None else None,
            )

        masks = []
        for definition in self.config.mask_plan.for_stage(stage):
            # Python callers may register an already normalized Mask (including
            # the deferred VectorMask returned by Mask.from_water()).  Check
            # this before looking at recipe-only fields such as ``kind``.
            if isinstance(definition, Mask):
                recipe = getattr(definition, "_water_recipe", None)
                if recipe is not None and recipe.get("bounds") is None:
                    masks.append(
                        _water_mask(
                            provider=str(recipe.get("provider", "auto")),
                            policy=recipe.get("policy"),
                        )
                    )
                else:
                    masks.append(definition)
            elif definition.kind == "raster":
                import rasterio

                with rasterio.open(definition.path) as source:
                    data = source.read(1)
                    source_grid = GridSpec(
                        source.crs,
                        source.transform,
                        shape=(source.height, source.width),
                    )
                masks.append(Mask.from_raster(data, grid=source_grid))
            elif definition.kind == "vector":
                import geopandas as gpd

                frame = gpd.read_file(definition.path)
                masks.append(Mask.from_vector(frame))
            else:
                masks.append(_water_mask(provider=definition.provider or "auto"))
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined + mask
        result = combined.to_raster(target)
        self._materialized_masks[key] = result
        return result

    def _ionosphere_mask_plane(
        self, ion_multilook: tuple[int, int] | None
    ) -> np.ndarray | None:
        """Return the explicit ionosphere-stage mask plane."""
        if not self.config.mask_plan.references("ionosphere"):
            return None
        looks = tuple(int(value) for value in (ion_multilook or self.config.multilook))
        grid = self._ifg_geo_grid(looks)
        if grid is None:
            raise ValueError("ionosphere masks require an EPSG:4326 geo_grid")
        from faninsar.processing.masking.mask import GridSpec

        transform, shape = grid
        return self._materialize_stage_mask(
            "ionosphere", GridSpec("EPSG:4326", transform, shape=shape)
        ).data

    def _unwrap_mask_plane(self, shape: tuple[int, int]) -> np.ndarray | None:
        """Materialize the explicit unwrap-stage mask on the IFG grid.

        Radar-domain unwrap uses the same authoritative master grid and
        projection cache as interferogram formation.  Geographic unwrap keeps
        the shared geo grid directly.  Keeping both paths in the common stage
        helper prevents the two consumers from silently materializing masks on
        different grids.
        """
        return self._stage_mask_plane(
            stage="unwrap",
            domain=self.config.coregistration_grid,
            looks=self.config.multilook,
            expected_shape=shape,
        )

    def _unwrap_mask_identity(self, shape: tuple[int, int]) -> str | None:
        """Return the materialized unwrap-mask identity for generation binding."""
        if not self.config.mask_plan.references("unwrap"):
            return None
        # Resolve through the exact same target-domain path as
        # ``_unwrap_mask_plane`` so identity and payload cannot diverge.
        plane = self._stage_mask_plane(
            stage="unwrap",
            domain=self.config.coregistration_grid,
            looks=self.config.multilook,
            expected_shape=shape,
        )
        if plane is None:
            return None
        if self.config.coregistration_grid == "geo":
            grid = self._ifg_geo_grid(self.config.multilook)
            if grid is None:
                reject_invalid_state("unwrap masks require an EPSG:4326 geo_grid")
            transform, target_shape = grid
            if tuple(shape) != target_shape:
                reject_invalid_state("unwrap mask target grid shape does not match IFG")
            from faninsar.processing.masking.mask import GridSpec

            target = GridSpec("EPSG:4326", transform, shape=target_shape)
            materialized = self._materialize_stage_mask("unwrap", target)
            return None if materialized is None else str(materialized.identity)
        key = self._radar_mask_identities.get(
            ("unwrap", tuple(self.config.multilook), tuple(shape))
        )
        if key is None:
            reject_invalid_state(
                "radar unwrap mask identity requires a projected mask result"
            )
        return key

    def _burst_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        bursts = cfg.bursts
        if bursts is None and cfg.swaths:
            # Default: first burst of each listed swath.
            bursts = {sw: [0] for sw in cfg.swaths}
        return {
            "swaths": cfg.swaths,
            "bursts": bursts,
            "dem": cfg.dem,
            "geo_grid": cfg.geo_grid,
            "executor": cfg.executor,
            "device": cfg.device,
            "dask_client": self.dask_client,
            "record_scientific_lineage": cfg.record_scientific_lineage,
            "coregistration_grid": cfg.coregistration_grid,
            "roi": self._effective_roi_with_mask().roi,
            "control_spacing": cfg.control_spacing,
            "n_jobs": cfg.n_jobs,
        }

    def _radar_projection_context_record(self, state: Any) -> dict[str, object] | None:
        """Return a durable descriptor for the authoritative radar context."""
        scene = getattr(state, "primary", None)
        geometry = getattr(scene, "geometry", None)
        shape = getattr(getattr(scene, "array", None), "shape", None)
        if shape is None:
            shape = getattr(getattr(scene, "array", None), "samples", None)
            shape = getattr(shape, "shape", None)
        if geometry is None or shape is None:
            logger.debug(
                "scene provider did not publish radar projection context; "
                "persisting a context-free radar marker"
            )
            return None
        burst = getattr(scene, "burst", None)
        swath = getattr(getattr(scene, "swath", None), "swath", None)
        orbit_paths = self.config.extra.get("orbit_paths")
        orbit_path = (
            orbit_paths.get(self.reference) if isinstance(orbit_paths, dict) else None
        )
        return {
            "source_path": str(
                getattr(scene, "path", self.catalog.paths_for(self.reference))
            ),
            "swath": str(
                swath or (self.config.swaths[0] if self.config.swaths else "IW1")
            ),
            "burst_index": int(getattr(burst, "index", 0)),
            "orbit_path": None if orbit_path is None else str(orbit_path),
            "full_range": True,
            "full_radar_shape": [int(value) for value in shape],
            "reference_scene": self.reference,
            "master_grid": self._radar_master_grid_metadata(),
        }

    def _radar_master_grid_metadata(self) -> dict[str, object]:
        """Return serializable metadata for the canonical radar mask master."""
        grid = self.config.geo_grid
        if grid is None:
            return {}
        return {
            "crs": str(grid.crs),
            "transform": [float(value) for value in grid.transform],
            "shape": [int(value) for value in grid.shape],
        }

    def _restore_radar_projection_context(
        self, marker_data: Mapping[str, object]
    ) -> None:
        """Restore radar geometry needed by masks after a coreg resume."""
        configured = self.config.extra.get("radar_projection_context")
        if isinstance(configured, dict) and configured.get("geometry") is not None:
            self._radar_projection_context = dict(configured)
            return
        record = marker_data.get("radar_projection_context")
        if not isinstance(record, dict):
            # No context is allowed to degrade silently.  Keep the error at the
            # projection boundary so callers without mask stages can still
            # resume ordinary radar scenes.
            self._radar_projection_context = None
            return
        source_path = record.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            reject_invalid_state("persisted radar projection context lacks source_path")
        try:
            from faninsar.processing.stages import load_production_scene

            scene = load_production_scene(
                source_path,
                swath=str(record.get("swath", "IW1")),
                burst_index=int(record.get("burst_index", 0)),
                orbit_path=record.get("orbit_path"),
                coregistration_grid="radar",
                full_range=bool(record.get("full_range", True)),
            )
        except Exception as error:
            logger.exception("failed to restore radar projection geometry")
            reject_invalid_state(
                f"persisted radar projection context cannot be restored: {error}"
            )
        shape = tuple(int(value) for value in record.get("full_radar_shape", ()))
        if len(shape) != 2 or any(value <= 0 for value in shape):
            reject_invalid_state("persisted radar projection context shape is invalid")
        if tuple(scene.array.shape) != shape:
            reject_invalid_state(
                "restored radar projection geometry shape differs from persisted "
                "coregistration shape"
            )
        self._radar_projection_context = {
            "geometry": scene.geometry,
            "full_radar_shape": shape,
            "reference_scene": record.get("reference_scene", self.reference),
            "master_grid": self.config.geo_grid,
        }

    def _reclaim_accelerator(self, kind: str) -> None:
        """Evaluate gpu_memory_reclaim on the CUDA-owning process."""
        from faninsar._core.device import reclaim_checkpoint
        from faninsar.backends.dask_gpu import run_reclaim_checkpoint

        policy = self.config.gpu_memory_reclaim
        device = self.config.device
        client = self.dask_client
        if client is not None:
            try:
                run_reclaim_checkpoint(client, device, policy, kind=kind)
            except Exception:
                logger.exception(
                    "reclaim_checkpoint on Dask GPU workers failed (kind=%s)",
                    kind,
                )
            return
        try:
            reclaim_checkpoint(device, policy, kind=kind)
        except Exception:
            logger.exception("reclaim_checkpoint failed (kind=%s)", kind)

    def _produce_pair(
        self,
        primary_path: SourceHandle | Path | tuple[Path, ...],
        secondary_path: SourceHandle | Path | tuple[Path, ...],
        *,
        output_dir: Path,
        **options: Any,
    ) -> Any:
        """Produce one pair through the admitted mission provider.

        Mission adapters supply a normalized callback; an unconfigured Stack
        fails closed rather than guessing how to open or produce a source.
        """
        if self.scene_provider is None:
            from faninsar.stack.provider import (
                UnsupportedStackCapabilityError,
            )

            mission = "Stack"
            capability = "scene-production"
            reason = "no mission scene provider was admitted"
            raise UnsupportedStackCapabilityError(
                mission,
                capability,
                reason,
            )
        options = dict(options)
        # A Stack owns exactly one materialized DEM on its authoritative
        # output grid.  Materialize a source recipe at the first production
        # boundary and reuse that raster for every scene/LUT callback.
        has_extent = (
            self.config.roi is not None or self._selected_footprint_roi() is not None
        )
        if self.config.dem is None:
            # A provider-only unit test (or another deliberately flat
            # callback) may not declare an ROI or explicit grid.  There is no
            # target grid on which an automatic DEM could be materialized in
            # that case; defer DEM ownership until a concrete Stack grid is
            # available.  Real projected/ROI runs always take the automatic
            # source branch here.
            if self.config.grid == "auto" and not has_extent:
                logger.debug(
                    "Stack DEM omitted without ROI or explicit grid; "
                    "deferring automatic DEM materialization"
                )
            else:
                self.config.dem = self.materialize_dem()
        elif isinstance(self.config.dem, SourceDEM) or not isinstance(
            self.config.dem, RasterDEM
        ):
            # Legacy bounded provider tests may intentionally use a constant
            # geometry height without a geographic extent.  Keep that sampler
            # intact; all source DEMs and extent-bearing runs materialize on
            # the authoritative Stack grid.
            if has_extent or type(self.config.dem).__name__ != "ConstantDEM":
                self.config.dem = self.materialize_dem(self.config.dem)
        elif (
            self.config.dem.grid != self.grid
            or self.config.dem.vertical_datum != "ellipsoidal"
        ):
            self.config.dem = self.config.dem.to_raster(
                self.grid,
                vertical_datum="ellipsoidal",
                budget=self.config.resource_budget,
            )
        if self.config.dem is not None:
            options.setdefault("dem", self.config.dem)
        return self.scene_provider(
            SourceHandle._from_source(primary_path),
            SourceHandle._from_source(secondary_path),
            output_dir=output_dir,
            options=options,
        )

    def release_accelerator(self) -> None:
        """Explicitly return unused accelerator slabs before yielding the GPU."""
        self._reclaim_accelerator("explicit")

    def _coreg_resume_identity(
        self,
        date_id: str,
        *,
        misreg_az_px: float,
        misreg_rg_px: float,
    ) -> str:
        """Return the canonical identity of one coregistration request."""

        def source_identity(path: Path | Sequence[Path]) -> object:
            if not isinstance(path, Path):
                return [source_identity(item) for item in path]
            resolved = path.resolve()
            try:
                stat = resolved.stat()
            except OSError:
                # Configuration paths such as an optional cache directory do
                # not have to exist yet.  Source paths still carry their full
                # stat identity when available, while an absent path remains
                # deterministic and is validated by the production call.
                return {"path": str(resolved)}
            return {
                "path": str(resolved),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }

        def semantic_value(value: object) -> object:
            if value is None or isinstance(value, (str, int, float, bool)):
                result = value
            elif isinstance(value, Path):
                result = source_identity(value)
            elif isinstance(value, np.generic):
                result = value.item()
            elif isinstance(value, dict):
                result = {
                    str(key): semantic_value(item)
                    for key, item in sorted(
                        value.items(), key=lambda item: str(item[0])
                    )
                }
            elif isinstance(value, (tuple, list)):
                result = [semantic_value(item) for item in value]
            elif is_dataclass(value) and not isinstance(value, type):
                result = {
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "fields": {
                        item.name: semantic_value(getattr(value, item.name))
                        for item in fields(value)
                        if not item.name.startswith("_")
                    },
                }
            elif hasattr(value, "identity") and not callable(value.identity):
                # Public RasterDEM uses slots, so its stable content identity
                # is the canonical resume key rather than ``__dict__`` state.
                result = {
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "identity": str(value.identity),
                }
            else:
                public_state = {
                    name: semantic_value(item)
                    for name, item in getattr(value, "__dict__", {}).items()
                    if not name.startswith("_") and not callable(item)
                }
                result = {
                    "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "state": public_state,
                }
            return result

        def dem_identity(dem: object | None) -> object:
            return None if dem is None else semantic_value(dem)

        def roi_identity(roi: object | None) -> object:
            if roi is None:
                return None
            geo_interface = getattr(roi, "__geo_interface__", None)
            if geo_interface is not None:
                return geo_interface
            bounds = getattr(roi, "bounds", None)
            if bounds is not None:
                return {"type": type(roi).__qualname__, "bounds": list(bounds)}
            return {"type": type(roi).__qualname__, "value": str(roi)}

        def digest_payload(value: object) -> str:
            """Hash one JSON-compatible identity payload."""
            try:
                encoded = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            except (TypeError, ValueError) as error:
                reject_invalid_state(
                    f"coregistration identity cannot be canonicalized: {error}"
                )
            return hashlib.sha256(encoded).hexdigest()

        def object_metadata(value: object) -> dict[str, object]:
            """Extract stable source metadata without serializing object reprs."""
            metadata: dict[str, object] = {}
            for name in (
                "source_id",
                "source_path",
                "source_uri",
                "uri",
                "filename",
                "path",
                "content_digest",
                "source_digest",
                "content_hash",
                "sha256",
                "digest",
            ):
                item = getattr(value, name, None)
                if item is not None and not callable(item):
                    metadata[name] = semantic_value(item)
            native = getattr(value, "mission_native", None)
            if isinstance(native, dict):
                for name, item in native.items():
                    key = str(name).lower()
                    if (
                        "id" in key
                        or "digest" in key
                        or "hash" in key
                        or key in {"source", "uri", "path"}
                    ):
                        metadata[f"mission_native:{name}"] = semantic_value(item)
            samples = getattr(value, "samples", None)
            if isinstance(samples, np.ndarray):
                # Reader products are normally lazy.  For an in-memory source,
                # include its bytes when reasonably sized so content changes
                # cannot reuse a prior marker without forcing a full read of a
                # large mission product.
                if samples.nbytes <= 32 * 1024 * 1024:
                    metadata["samples_sha256"] = hashlib.sha256(
                        np.ascontiguousarray(samples).tobytes()
                    ).hexdigest()
                else:
                    metadata["samples_descriptor"] = {
                        "shape": list(samples.shape),
                        "dtype": str(samples.dtype),
                        "nbytes": int(samples.nbytes),
                    }
            for name in ("content", "source_content", "data", "raw_bytes"):
                content = getattr(value, name, None)
                if isinstance(content, (bytes, bytearray, memoryview)):
                    metadata[f"{name}_sha256"] = hashlib.sha256(
                        bytes(content)
                    ).hexdigest()
            return metadata

        def source_object_identity(acquisition_id: str) -> dict[str, object] | None:
            """Return provider-owned identity metadata for one source, if any."""
            objects: list[object] = []
            results = getattr(self, "_nisar_results", None)
            if isinstance(results, dict) and acquisition_id in results:
                objects.append(results[acquisition_id])
                product = getattr(results[acquisition_id], "product", None)
                if product is not None:
                    objects.append(product)
            handles = getattr(self, "_nisar_handles", None)
            lineage = getattr(self, "_nisar_lineage", None)
            if isinstance(handles, dict) and isinstance(lineage, dict):
                source_path = lineage.get(acquisition_id)
                if source_path is not None:
                    handle = handles.get(Path(source_path))
                    if handle is not None:
                        objects.append(handle)
            metadata: dict[str, object] = {}
            for item in objects:
                metadata.update(object_metadata(item))
            return metadata or None

        def provider_identity() -> dict[str, object] | None:
            """Collect the optional mission-provider identity contract."""
            provider = self.scene_provider
            if provider is None:
                return None
            payload: dict[str, object] = {
                "name": str(getattr(provider, "name", type(provider).__qualname__)),
                "capability": str(
                    getattr(
                        provider,
                        "capability",
                        getattr(provider, "unsupported_capability", "scene-production"),
                    )
                ),
            }
            capabilities = getattr(provider, "capabilities", None)
            if capabilities is not None:
                payload["capabilities"] = semantic_value(capabilities)
            callback = _provider_callback(provider)
            callback_identity = _callable_identity(callback)
            if callback_identity is not None:
                payload["callback"] = callback_identity
            # Providers may expose either a plain metadata mapping or a
            # zero-argument identity hook.  This keeps Stack mission-neutral
            # while allowing adapters to bind windows, channels, and source
            # lineage without Stack knowing their implementation details.
            for name in ("identity_payload", "identity_metadata"):
                hook = getattr(provider, name, None)
                if hook is None:
                    continue
                try:
                    metadata = hook() if callable(hook) else hook
                except Exception as error:
                    logger.exception("Stack provider identity hook failed")
                    reject_invalid_state(
                        f"Stack provider identity hook failed: {error}"
                    )
                payload[name] = semantic_value(metadata)
            return payload

        geo_grid = self.config.geo_grid
        geo_grid_identity = (
            None
            if geo_grid is None
            else {
                "crs": geo_grid.crs,
                "transform": list(geo_grid.transform),
                "shape": list(geo_grid.shape),
                "resolution_m": list(geo_grid.resolution_m),
            }
        )
        bursts = self.config.bursts
        if bursts is None and self.config.swaths:
            bursts = {swath: [0] for swath in self.config.swaths}
        # The coreg identity binds to the ROI the run actually consumed: the
        # mask-aware effective ROI when the water mask reshaped it, otherwise
        # the configured ROI unchanged (a degraded unmasked run resumes).
        effective_roi = self._effective_roi_with_mask()
        identity_roi = (
            effective_roi.geometry
            if effective_roi.geometry is not None
            else self.config.roi
        )
        provider_payload = provider_identity()
        provider_metadata: dict[str, object] = {}
        if provider_payload is not None:
            provider_metadata["provider"] = provider_payload
            channel = getattr(self, "_nisar_channel", None)
            if channel is None:
                channel = getattr(self, "channel", None)
            if channel is not None:
                provider_metadata["channel"] = semantic_value(channel)
            provider_metadata["stack_metadata"] = {
                name: semantic_value(getattr(self, name))
                for name in ("_nisar_channel", "_nisar_lineage")
                if hasattr(self, name)
            }
            provider_metadata["source_objects"] = {
                "reference": source_object_identity(self.reference),
                "secondary": source_object_identity(date_id),
            }
        configuration = {
            "coreg_mode": self.config.coreg_mode,
            "flatten_stage": self.config.flatten_stage,
            "coregistration_grid": self.config.coregistration_grid,
            "esd_method": self.config.esd_method,
            "multilook": list(self.config.multilook),
            "goldstein_alpha": self.config.goldstein_alpha,
            "executor": self.config.executor,
            "device": self.config.device,
            "invert_device": self.config.invert_device,
            "control_spacing": self.config.control_spacing,
            "n_jobs": self.config.n_jobs,
            "swaths": list(self.config.swaths),
            "bursts": semantic_value(bursts),
            "extra": semantic_value(self.config.extra),
        }
        callback = _provider_callback(self.scene_provider)
        runtime = _stack_runtime_identity(callback)
        payload = {
            "schema": "stack_coreg_request_v1",
            "reference_id": self.reference,
            "date_id": date_id,
            "reference_source": source_identity(self.catalog.paths_for(self.reference)),
            "secondary_source": source_identity(self.catalog.paths_for(date_id)),
            "coreg_mode": self.config.coreg_mode,
            "flatten_stage": self.config.flatten_stage,
            "coregistration_grid": self.config.coregistration_grid,
            "esd_method": self.config.esd_method,
            "swaths": list(self.config.swaths),
            "bursts": {
                str(swath): (
                    indices
                    if isinstance(indices, str)
                    else [int(index) for index in indices]
                )
                for swath, indices in sorted((bursts or {}).items())
            },
            "roi": roi_identity(identity_roi),
            "dem": dem_identity(self.config.dem),
            "geo_grid": geo_grid_identity,
            "control_spacing": self.config.control_spacing,
            "executor": self.config.executor,
            "device": self.config.device,
            "misreg_az_px": float(misreg_az_px),
            "misreg_rg_px": float(misreg_rg_px),
            "provider_metadata": provider_metadata,
            "configuration": configuration,
            "configuration_fingerprint": digest_payload(configuration),
            "runtime_fingerprint": digest_payload(runtime),
        }
        return digest_payload(payload)

    @_reclaim_after_stage
    def measure_misreg(
        self,
        *,
        pairs: Pairs | None = None,
        esd_method: EsdMethod | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Measure per-pair misreg arcs (no full IFG products).

        For ``coreg_mode != "network"`` this is a no-op unless forced by
        calling with pairs while mode is network. Arcs are written under
        ``work_dir/misreg/arcs.json``.
        """
        self._ensure_prepared()
        if self.config.coreg_mode == "geometry":
            logger.info("measure_misreg skipped (coreg_mode=geometry)")
            return self
        if self.config.coreg_mode == "pair":
            # Pair mode applies residuals inside coregister; arcs optional.
            logger.info(
                "measure_misreg: pair mode — arcs recorded for diagnostics only",
            )

        method = esd_method or self.config.esd_method
        use_pairs = pairs or self.misreg_pairs
        arcs_path = self.config.work_dir / "misreg" / "arcs.json"
        if arcs_path.is_file() and not overwrite:
            self.arcs = _load_arcs(arcs_path)
            logger.info("Loaded %s misreg arcs from cache", len(self.arcs))
            return self

        arcs: list[MisregArc] = []
        for primary, secondary in _iter_pair_dates(use_pairs):
            if primary not in self.catalog.paths or secondary not in self.catalog.paths:
                continue
            out = self.config.work_dir / "misreg" / "measure" / f"{primary}_{secondary}"
            esd_on = method != "auto" or True
            # Measure-only: coreg + ESD/Ampcor; no unwrap; no IFG write required.
            state = self._produce_pair(
                self.catalog.paths_for(primary),
                self.catalog.paths_for(secondary),
                output_dir=out,
                multilook=self.config.multilook,
                goldstein_alpha=self.config.goldstein_alpha,
                esd_enabled=esd_on,
                amplitude_refinement_enabled=True,
                unwrap=False,
                overwrite=overwrite,
                **self._burst_kwargs(),
            )
            az = float(state.esd_azimuth_shift_px or 0.0)
            # Ampcor exposes the range residual in pixel units.  Keep the
            # network arc on the same semantic field used by the production
            # state; silently defaulting to zero would discard a measured
            # range correction and make the later ESD/network solve diverge.
            amp_rg = float(getattr(state, "amplitude_residual_rg_px", 0.0) or 0.0)
            arcs.append(
                MisregArc(
                    primary=primary,
                    secondary=secondary,
                    azimuth_shift_px=az,
                    range_shift_px=amp_rg,
                    azimuth_sigma_px=0.05 if abs(az) > 1e-9 else 1.0,
                    range_sigma_px=0.1 if abs(amp_rg) > 1e-9 else 1.0,
                    method=str(method),
                    n_valid=1,
                ),
            )
        self.arcs = arcs
        _save_arcs(arcs_path, arcs)
        logger.info("Measured %s misreg arcs (method=%s)", len(arcs), method)
        return self

    @_reclaim_after_stage
    def invert_misreg(
        self,
        *,
        reference: str | None = None,
        min_n_valid: int = 0,
        max_sigma_px: float = 1e3,
    ) -> Self:
        """Invert arcs to per-date az/rg with Reference fixed at 0."""
        self._ensure_prepared()
        if self.config.coreg_mode != "network":
            logger.info("invert_misreg skipped (coreg_mode=%s)", self.config.coreg_mode)
            return self
        if not self.arcs:
            self.measure_misreg()
        ref = reference or self.reference
        try:
            self.date_misreg = invert_pair_misregistration(
                self.arcs,
                reference=ref,
                dates=list(self.catalog.dates),
                min_n_valid=min_n_valid,
                max_sigma_px=max_sigma_px,
            )
        except Exception:
            # Qualified Stack execution never silently changes registration
            # semantics after a failed network inversion.  The configured
            # policy is validated as ``error`` at construction, so this path
            # remains fail-closed for missing/invalid residual solutions.
            logger.exception("misreg network inversion failed; aborting Stack")
            raise
        out = self.config.work_dir / "misreg" / "date_misreg.json"
        out.write_text(
            json.dumps(
                {
                    "reference": self.date_misreg.reference,
                    "azimuth_px": dict(self.date_misreg.azimuth_px),
                    "range_px": dict(self.date_misreg.range_px),
                    "metadata": dict(self.date_misreg.metadata),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return self

    @_reclaim_after_stage
    def coregister_scenes(
        self,
        *,
        dates: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Coregister each non-Reference date onto the Reference grid and cache.

        This stage does **not** form interferograms.
        """
        self._ensure_prepared()
        target_dates = (
            list(dates)
            if dates is not None
            else [d for d in self.catalog.dates if d != self.reference]
        )
        primary_path = self.catalog.paths_for(self.reference)
        # Cache Reference identity product path (no self-coreg).
        reference_dir = self.config.work_dir / "coreg" / self.reference
        reference_dir.mkdir(parents=True, exist_ok=True)
        self.coreg_paths[self.reference] = reference_dir

        # PROPOSAL-0017: pair = dense geometry + Ampcor range + ESD azimuth.
        # geometry skips residual measure; network apply uses date constants only.
        esd_on = self.config.coreg_mode == "pair"
        amp_on = self.config.coreg_mode in {"pair", "network"}
        if self.config.coreg_mode == "network":
            esd_on = False
            amp_on = False
        orbit_paths = self.config.extra.get("orbit_paths")
        if orbit_paths is not None and not isinstance(orbit_paths, dict):
            reject_invalid_state(
                "Stack extra orbit_paths must be a date-to-path mapping"
            )

        for date_id in target_dates:
            out = self.config.work_dir / "coreg" / date_id
            marker = out / "coreg_done.json"
            misreg_az = 0.0
            misreg_rg = 0.0
            if self.date_misreg is not None:
                misreg_az = float(self.date_misreg.azimuth_px.get(date_id, 0.0))
                misreg_rg = float(self.date_misreg.range_px.get(date_id, 0.0))
            coreg_identity = self._coreg_resume_identity(
                date_id,
                misreg_az_px=misreg_az,
                misreg_rg_px=misreg_rg,
            )
            if marker.is_file() and not overwrite:
                try:
                    marker_data = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, ValueError) as error:
                    reject_invalid_state(
                        f"coregistration marker cannot be validated: {error}"
                    )
                store = CoregisteredSceneStore.open(out / "scenes")
                if (
                    marker_data.get("reference") != self.reference
                    or marker_data.get("date") != date_id
                    or marker_data.get("coreg_identity") != coreg_identity
                    or store.reference_id != self.reference
                    or store.date_id != date_id
                    or store.domain != self.config.coregistration_grid
                ):
                    reject_invalid_state(
                        "persisted coregistration scene or marker does not match "
                        "the current date, Reference, or coordinate domain"
                    )
                self.coreg_paths[date_id] = out
                if self.config.coregistration_grid == "radar":
                    self._restore_radar_projection_context(marker_data)
                if date_id == target_dates[0]:
                    copy_reference_units(out / "scenes", reference_dir / "scenes")
                continue
            pair_kwargs = dict(self._burst_kwargs())
            if isinstance(orbit_paths, dict):
                reference_orbit = orbit_paths.get(self.reference)
                date_orbit = orbit_paths.get(date_id)
                if reference_orbit is not None:
                    pair_kwargs["primary_orbit_path"] = reference_orbit
                if date_orbit is not None:
                    pair_kwargs["secondary_orbit_path"] = date_orbit
            for extra_key in (
                "geoid_correction",
                "geo_lut_cache_dir",
                "geo_footprint_mask_enabled",
            ):
                if extra_key in self.config.extra:
                    pair_kwargs[extra_key] = self.config.extra[extra_key]
            geo_work = self.config.extra.get("geo_work_dir")
            if geo_work is not None:
                pair_kwargs["geo_work_dir"] = Path(geo_work) / date_id
            state = self._produce_pair(
                primary_path,
                self.catalog.paths_for(date_id),
                output_dir=out,
                multilook=self.config.multilook,
                goldstein_alpha=0.0,
                esd_enabled=esd_on,
                amplitude_refinement_enabled=amp_on,
                unwrap=False,
                overwrite=overwrite,
                misreg_az_px=misreg_az,
                misreg_rg_px=misreg_rg,
                scene_store_dir=out / "scenes",
                **pair_kwargs,
            )
            if self.config.coregistration_grid == "radar":
                primary = getattr(state, "primary", None)
                if primary is not None:
                    array = getattr(primary, "array", None)
                    shape = getattr(array, "shape", None)
                    if shape is None:
                        samples = getattr(array, "samples", None)
                        shape = getattr(samples, "shape", None)
                    geometry = getattr(primary, "geometry", None)
                    if shape is not None and geometry is not None:
                        self._radar_projection_context = {
                            "geometry": geometry,
                            "full_radar_shape": tuple(int(v) for v in shape),
                            "reference_scene": self.reference,
                            "master_grid": self.config.geo_grid,
                        }
                    else:
                        reject_invalid_state(
                            "radar coregistration did not produce projection geometry"
                        )
                else:
                    # Lightweight providers may publish scene units without
                    # retaining a full in-memory primary scene.  The persisted
                    # scene store is sufficient for radar coregistration; only
                    # projection-context restoration needs these optional fields.
                    logger.debug(
                        "scene provider did not retain primary scene; "
                        "skipping optional radar projection context"
                    )
            if self.config.retain_pair_states:
                self.pair_states[f"{self.reference}_{date_id}"] = state
            else:
                logger.info(
                    "Released in-memory ProductionPairState for %s after scene "
                    "publication",
                    date_id,
                )
            self.coreg_paths[date_id] = out
            if date_id == target_dates[0]:
                copy_reference_units(out / "scenes", reference_dir / "scenes")
            marker.write_text(
                json.dumps(
                    {
                        "reference": self.reference,
                        "date": date_id,
                        "coreg_identity": coreg_identity,
                        "misreg_az_px": misreg_az,
                        "misreg_rg_px": misreg_rg,
                        "geometric_phase_removed": (
                            "topo" if self.config.dem is not None else "flat"
                        ),
                        "esd_azimuth_shift_px": state.esd_azimuth_shift_px,
                        "range_shift_px": state.range_shift_px,
                        "azimuth_shift_px": state.azimuth_shift_px,
                        "esd_enabled": esd_on,
                        "amplitude_refinement_enabled": amp_on,
                        "stage_timings_s": getattr(state, "stage_timings_s", {}),
                        "coregistration_timings_s": getattr(
                            state, "coregistration_timings_s", {}
                        ),
                        "radar_projection_context": (
                            self._radar_projection_context_record(state)
                            if self.config.coregistration_grid == "radar"
                            else None
                        ),
                        **self._mask_lineage_record(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            logger.info("Coregistered %s → Reference %s", date_id, self.reference)
            if not self.config.retain_pair_states:
                # Assignment evaluates the next provider call before
                # replacing this local. Drop the completed state's full-burst
                # arrays now so adjacent dates cannot overlap in memory.
                del state
                gc.collect()
            self._reclaim_accelerator("persist")
        self._write_qualified_activation_record()
        return self

    @_reclaim_after_stage
    def form_interferograms(
        self,
        *,
        pairs: Pairs | None = None,
        multilook: tuple[int, int] | list[tuple[int, int]] | None = None,
        coherence_window: tuple[int, int] | None = (5, 5),
        phase_filter: PhaseFilter | None = _DEFAULT_PHASE_FILTER,
        goldstein_alpha: float | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Form wrapped interferograms from persisted Reference-aligned scenes.

        Parameters
        ----------
        pairs : Pairs, optional
            Pair graph to form. ``None`` uses the Stack graph fixed at
            construction. Pair endpoints are acquisition dates; their order is
            canonicalized by :class:`~faninsar.core.pairs.Pairs`.
        multilook : tuple[int, int] or list[tuple[int, int]], optional
            One or more true boxcar look factors in ``(azimuth, range)``
            pixels. ``None`` uses :attr:`StackConfig.multilook`, whose default
            is ``(5, 2)``. Each result is aligned to the common scene-grid
            origin. Scene-grid edge coverage and incomplete look support are
            represented by the validity mask rather than interpolation.
        coherence_window : tuple[int, int] or None, default=(5, 5)
            Odd, centered ``(azimuth, range)`` MLE support, with each axis at
            least three pixels. This selects the ISCE2-like first stage: a
            stride-one, clipped local coherence image followed by multilook
            averaging. ``None`` instead estimates coherence directly in each
            output look block, matching a direct-multilook workflow. The value
            is validated before this method reads scenes or opens stores.
        phase_filter : PhaseFilter or None, default=GoldsteinWerner()
            Runtime wrapped-phase strategy applied after multilooking.
            ``GoldsteinWerner(alpha=0.5, patch_size=32)`` is the default.
            ``None`` preserves the unfiltered complex interferogram. A custom
            strategy is a trusted in-memory object, is not persisted for later
            restoration, and must return an equal-shape, same-device, same
            complex-dtype tensor with a boolean support mask that is a subset
            of its input support. Valid returned samples must be finite.
        goldstein_alpha : float, optional
            Deprecated internal compatibility path for legacy callers. New
            callers should use ``phase_filter=GoldsteinWerner(alpha=...)``.
            An explicitly supplied ``phase_filter`` (including ``None``)
            takes precedence.
        output_dir : path-like, optional
            Root directory for immutable per-pair IFG artifacts. ``None`` uses
            ``config.work_dir / "ifg"``.
        overwrite : bool, default=False
            Replace an existing complete artifact at the selected output path.

        Returns
        -------
        Stack
            This Stack, with complete IFG artifact directories appended to its
            product set.

        Raises
        ------
        ValueError
            If ``coherence_window`` is not ``None`` or two odd integers at
            least three, or if other public numeric options are invalid.
        InvalidProcessingStateError
            If prerequisite scene generations are absent, incomplete, mixed in
            domain or grid, or a phase-filter result violates its runtime
            contract.

        Notes
        -----
        Invalid, nonfinite, and uncovered samples remain invalid through
        formation and filtering; filters may never resurrect masked support.
        The stored coherence is clamped to ``[0, 1]`` where it is defined.
        This method deliberately has no SAFE-path or pair-runner fallback.
        Missing, incomplete, mixed-domain, or multi-unit generations fail
        closed until the provider supplies a complete scene manifest.
        With an active PROPOSAL-0039 mask, the mask is a support input
        intersected into the persisted ``valid_mask`` after formation and
        filtering (``valid_mask &= ~mask``); it is never routed through the
        :class:`~faninsar.processing.interferometry.phase_filter.PhaseFilter`
        and never touches the complex/phase numerics.

        References
        ----------
        The centered two-stage coherence layout follows the TOPS Stack
        convention used by ISCE2. Goldstein-Werner filtering follows
        Goldstein and Werner (1998), *Radar interferogram filtering for
        geophysical applications*.

        """
        coherence_window = validate_coherence_window(coherence_window)
        self._ensure_prepared()
        self._require_qualified_activation_record()

        from faninsar.processing.resources import (
            ResourceAdmissionLedger,
            estimate_formation_resources,
            reserve_estimate,
        )

        use_pairs = pairs or self.pairs
        looks_list = _normalize_multilook(multilook or self.config.multilook)
        # ``goldstein_alpha`` remains an internal transition path for existing
        # callers.  The public MVP API is ``phase_filter``; an explicit filter
        # (including None) always takes precedence over the old scalar option.
        legacy_filter = (
            phase_filter is _DEFAULT_PHASE_FILTER and goldstein_alpha is not None
        )
        alpha = float(goldstein_alpha) if legacy_filter else 0.0
        selected_filter = None if legacy_filter else phase_filter
        selected_coherence_window = None if legacy_filter else coherence_window
        filter_name, filter_parameters = _phase_filter_metadata(selected_filter)
        base_out = (
            Path(output_dir)
            if output_dir is not None
            else (self.config.work_dir / "ifg")
        )
        formation_ledger = (
            ResourceAdmissionLedger(self.config.resource_budget, self.config.work_dir)
            if self.config.resource_budget is not None
            else None
        )

        for primary, secondary in _iter_pair_dates(use_pairs):
            if primary not in self.catalog.paths or secondary not in self.catalog.paths:
                continue
            for looks in looks_list:
                az_l, rg_l = looks
                sub = base_out / f"ml_{az_l}x{rg_l}" / f"{primary}_{secondary}"
                if sub.exists() and overwrite:
                    shutil.rmtree(sub)
                if primary not in self.coreg_paths or secondary not in self.coreg_paths:
                    reject_invalid_state(
                        "Stack scene generation missing; run coregister_scenes first"
                    )
                primary_store = CoregisteredSceneStore.open(
                    self.coreg_paths[primary] / "scenes"
                )
                secondary_store = CoregisteredSceneStore.open(
                    self.coreg_paths[secondary] / "scenes"
                )
                if primary_store.flatten_stage != secondary_store.flatten_stage:
                    reject_invalid_state("scene artifacts mix flattening stages")
                flatten_stage = primary_store.flatten_stage
                phase_screen_model, phase_screen_digests = _phase_screen_lineage(
                    primary_store,
                    secondary_store,
                    flatten_stage,
                )
                expected_sources = {
                    "primary": primary_store.manifest_digest,
                    "secondary": secondary_store.manifest_digest,
                    # The IFG manifest is also a runtime admission boundary:
                    # reusing bytes produced under another binary, Python,
                    # CUDA, or provider callback identity is unsafe.
                    "runtime": _runtime_fingerprint(
                        _provider_callback(self.scene_provider)
                    ),
                }
                mask_plan_identity = self.config.mask_plan.identity
                mask_identity = None
                if self.config.mask_plan.references("interferogram"):
                    mask_grid = self._ifg_geo_grid(looks)
                    if mask_grid is not None:
                        from faninsar.processing.masking.mask import GridSpec

                        mask_identity = self._materialize_stage_mask(
                            "interferogram",
                            GridSpec("EPSG:4326", mask_grid[0], shape=mask_grid[1]),
                        ).identity
                expected_filter_name = filter_name
                expected_filter_parameters = filter_parameters
                if (sub / "manifest.json").exists():
                    from faninsar.stack.ifg_store import (
                        InterferogramArtifactStore,
                    )

                    existing_store = InterferogramArtifactStore.open(sub)
                    if (
                        existing_store.pair != (primary, secondary)
                        or existing_store.looks != looks
                        or existing_store.domain != primary_store.domain
                        or existing_store.wavelength_m != primary_store.wavelength_m
                        or existing_store.grid_identity != primary_store.grid_identity
                        or existing_store.filter_name != expected_filter_name
                        or existing_store.filter_parameters
                        != expected_filter_parameters
                        or existing_store.source_manifest_digests != expected_sources
                        or existing_store.flatten_stage != flatten_stage
                        or existing_store.phase_screen_model != phase_screen_model
                        or existing_store.phase_screen_digests != phase_screen_digests
                        or existing_store.phase_screen_domain != primary_store.domain
                        or existing_store.phase_screen_grid_identity
                        != primary_store.grid_identity
                        or existing_store.mask_plan_identity != mask_plan_identity
                        or existing_store.mask_identity != mask_identity
                    ):
                        reject_invalid_state(
                            "persisted IFG artifact does not match current scene "
                            "lineage or processing parameters"
                        )
                    existing_store.read()
                    self.ifg_dirs.append(sub)
                    continue
                if sub.exists():
                    reject_invalid_state(
                        "partial IFG artifact directory exists without a manifest"
                    )
                reservation = None
                if formation_ledger is not None:
                    estimate = estimate_formation_resources(
                        shape=primary_store.grid_shape,
                        multilook=looks,
                        coherence_window=selected_coherence_window,
                        phase_filter=selected_filter,
                    )
                    reservation = reserve_estimate(formation_ledger, estimate)
                try:
                    product = form_merged_scene_interferogram(
                        primary_store,
                        secondary_store,
                        primary_role=(
                            "primary" if primary == self.reference else "secondary"
                        ),
                        secondary_role=(
                            "primary" if secondary == self.reference else "secondary"
                        ),
                        multilook=looks,
                        goldstein_alpha=alpha,
                        coherence_window=selected_coherence_window,
                        phase_filter=selected_filter,
                        device=self.config.device,
                        dask_client=self.dask_client,
                        flatten_stage=flatten_stage,
                    )
                finally:
                    if reservation is not None:
                        reservation.release()
                from faninsar.stack.ifg_store import write_ifg_artifact

                # PROPOSAL-0039 (AC-5): the active mask is a support input
                # intersected into the persisted valid_mask AFTER formation
                # and filtering. It never routes through the PhaseFilter and
                # never touches the complex/phase numerics.
                valid_mask = product.valid_mask
                mask_plane = self._ifg_mask_plane(
                    domain=primary_store.domain,
                    looks=looks,
                    expected_shape=(
                        valid_mask.shape
                        if valid_mask is not None
                        else product.complex_ifg.shape
                    ),
                )
                if mask_plane is not None:
                    if valid_mask is None:
                        valid_mask = np.isfinite(product.complex_ifg)
                    valid_mask = _apply_mask_to_valid_mask(valid_mask, mask_plane)

                write_ifg_artifact(
                    sub,
                    pair=(primary, secondary),
                    looks=looks,
                    domain=primary_store.domain,
                    wavelength_m=primary_store.wavelength_m,
                    grid_identity=primary_store.grid_identity,
                    filter_name=expected_filter_name,
                    filter_parameters=expected_filter_parameters,
                    source_manifest_digests=expected_sources,
                    mask_plan_identity=mask_plan_identity,
                    mask_identity=mask_identity,
                    flatten_stage=flatten_stage,
                    phase_screen_model=phase_screen_model,
                    phase_screen_digests=phase_screen_digests,
                    phase_screen_domain=primary_store.domain,
                    phase_screen_grid_identity=primary_store.grid_identity,
                    complex_ifg=product.complex_ifg,
                    coherence=product.coherence,
                    wrapped_phase=product.wrapped_phase,
                    amplitude=product.amplitude,
                    valid_mask=valid_mask,
                )
                self.ifg_dirs.append(sub)
        return self

    def _refresh_network_from_ifg_dirs(self) -> None:
        """Refresh inherited Network state from complete IFG artifacts.

        Product records are created only from manifest-validated artifact
        stores.  Validation completes before the immutable Network index is
        replaced, so a partial or mixed generation leaves the previous index
        untouched.
        """
        from faninsar.stack.ifg_store import InterferogramArtifactStore

        if not self.ifg_dirs:
            reject_invalid_state(
                "Stack cannot refresh Network products without IFG artifacts"
            )

        dimensions = self.config.extra
        expected_pairs = tuple(_iter_pair_dates(self.pairs))
        expected_counts = Counter(expected_pairs)
        frame = str(dimensions.get("frame_id", "stack"))
        swath = str(dimensions.get("swath", "merged"))
        channel = str(dimensions.get("channel", "merged"))
        polarization = str(dimensions.get("polarization", "merged"))
        products: list[NetworkProduct] = []
        manifest_digests: list[str] = []
        unwrap_manifest_digests: list[str] = []
        seen_paths: set[Path] = set()
        actual_pairs: list[tuple[str, str]] = []
        unique_ifg_paths = tuple(
            dict.fromkeys(Path(raw_path) for raw_path in self.ifg_dirs)
        )
        unwrap_flags = [
            (path / "unwrap_manifest.json").is_file() for path in unique_ifg_paths
        ]
        if any(unwrap_flags) and not all(unwrap_flags):
            reject_invalid_state(
                "Stack IFG generation contains a partial unwrap product set"
            )
        has_unwrapped_products = bool(unwrap_flags) and all(unwrap_flags)
        unwrapped_artifacts: list[Any] = []
        for raw_path in self.ifg_dirs:
            path = Path(raw_path)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            store = InterferogramArtifactStore.open(path)
            try:
                primary, secondary = store.pair
                actual_pairs.append(_canonical_pair_strings((primary, secondary)))
                primary_key = AcquisitionKey(
                    actual_pairs[-1][0], frame, swath, channel, polarization
                )
                secondary_key = AcquisitionKey(
                    actual_pairs[-1][1], frame, swath, channel, polarization
                )
                kind = AssetKind.COMPLEX_INTERFEROGRAM
                products.append(
                    NetworkProduct(
                        key=NetworkProductKey(primary_key, secondary_key, kind),
                        asset_location=str(store.generation_root / "complex_ifg.npy"),
                        geometry_identity=store.grid_identity,
                        source_software="faninsar",
                        phase_convention=PhaseConvention.PRIMARY_MINUS_SECONDARY,
                        asset_transform=AssetTransform.for_convention(
                            kind, PhaseConvention.PRIMARY_MINUS_SECONDARY
                        ),
                        content_digest=store.manifest_digest,
                        lineage=(store.manifest_digest,),
                    )
                )
                manifest_digests.append(store.manifest_digest)
                if has_unwrapped_products:
                    unwrapped = store.read_unwrapped()
                    unwrap_generation_id, unwrap_digest = (
                        _read_unwrap_manifest_identity(store)
                    )
                    unwrap_manifest_digests.append(unwrap_digest)
                    unwrapped_artifacts.append(unwrapped)
                    unwrap_kind = AssetKind.UNWRAPPED_PHASE
                    products.append(
                        NetworkProduct(
                            key=NetworkProductKey(
                                primary_key,
                                secondary_key,
                                unwrap_kind,
                            ),
                            asset_location=str(
                                store.root
                                / ".unwrap_generations"
                                / unwrap_generation_id
                                / "unwrapped_phase.npy"
                            ),
                            geometry_identity=store.grid_identity,
                            source_software="faninsar",
                            phase_convention=PhaseConvention.PRIMARY_MINUS_SECONDARY,
                            asset_transform=AssetTransform.for_convention(
                                unwrap_kind,
                                PhaseConvention.PRIMARY_MINUS_SECONDARY,
                            ),
                            content_digest=unwrap_digest,
                            lineage=(store.manifest_digest, unwrap_digest),
                        )
                    )
            finally:
                store.close()

        if not products:
            reject_invalid_state("Stack IFG generation contains no products")
        actual_counts = Counter(actual_pairs)
        if actual_counts != expected_counts:
            reject_invalid_state(
                "Stack IFG product pair set does not exactly match the configured "
                f"Pair network: expected={dict(expected_counts)!r}, "
                f"discovered={dict(actual_counts)!r}"
            )
        if has_unwrapped_products:
            expected_pair_ids = [
                f"{primary}_{secondary}" for primary, secondary in expected_pairs
            ]
            expected_parameters = unwrapped_artifacts[0].method_parameters
            for artifact in unwrapped_artifacts:
                if (
                    artifact.method_parameters != expected_parameters
                    or artifact.method_parameters.get("pair_ids") != expected_pair_ids
                ):
                    reject_invalid_state(
                        "Stack unwrap products do not form one consistent Pair network"
                    )
        generation_id = _network_generation_digest(
            manifest_digests, unwrap_manifest_digests
        )
        self.refresh_generation(generation_id, tuple(products))

    @property
    def network_generation_id(self) -> str | None:
        """Return the committed Network generation visible to analysis."""
        return self._network_generation_id

    @property
    def network_product_index(self) -> NetworkProductIndex | None:
        """Return the immutable Network product index, if refreshed."""
        return self._network_product_index

    @property
    def analysis_ready(self) -> bool:
        """Whether a complete Network generation is available for analysis."""
        return (
            self._network_generation_id is not None
            and self._network_product_index is not None
        )

    def _refresh_network_generation(
        self,
        generation_id: str,
        products: (
            NetworkProductIndex | tuple[NetworkProduct, ...] | list[NetworkProduct]
        ),
    ) -> Self:
        """Atomically refresh the inherited Network product index."""
        NetworkContract.refresh_generation(self, generation_id, products)
        view = getattr(self, "_network_view", None)
        if view is not None:
            # Keep an already-obtained view current without exposing a partial
            # index: NetworkContract.refresh_generation validated both fields
            # before this synchronization point.
            view._network_generation_id = self._network_generation_id
            view._network_product_index = self._network_product_index
            view._product_index = self._network_product_index
            view.manifest = {"generation_id": self._network_generation_id}
            view.generation_root = self._root
        return self

    def _refresh_network_from_unwrap_generation(self, generation: Any) -> None:
        """Refresh Network products from one committed spatial unwrap root.

        The root generation is the durable authority for unwrapped products.
        Existing IFG product records supply the shared acquisition and grid
        metadata; this method only adds the newly committed spatial layers.
        """
        from faninsar.stack.stack_generation import UnwrapResultGeneration

        if not isinstance(generation, UnwrapResultGeneration):
            logger.error("Stack unwrap refresh received an invalid generation")
            message = "generation must be an UnwrapResultGeneration"
            raise TypeError(message)
        index = self.network_product_index
        if index is None:
            self._refresh_network_from_ifg_dirs()
            index = self.network_product_index
        if index is None:
            reject_invalid_state("Stack unwrap refresh has no IFG Network products")
        dimensions = self.config.extra
        frame = str(dimensions.get("frame_id", "stack"))
        swath = str(dimensions.get("swath", "merged"))
        channel = str(dimensions.get("channel", "merged"))
        polarization = str(dimensions.get("polarization", "merged"))
        products = [
            product
            for product in index.products
            if product.key.product_kind is not AssetKind.UNWRAPPED_PHASE
        ]
        for pair_id in generation.pair_ids:
            try:
                primary, secondary = pair_id.split("_", 1)
            except ValueError as error:
                reject_invalid_state(f"Stack unwrap Pair id is invalid: {pair_id!r}")
                raise AssertionError from error
            source = next(
                (
                    product
                    for product in products
                    if product.primary.acquisition_id == primary
                    and product.secondary.acquisition_id == secondary
                    and product.key.product_kind is AssetKind.COMPLEX_INTERFEROGRAM
                ),
                None,
            )
            if source is None:
                reject_invalid_state(
                    f"Stack unwrap has no IFG Network product for Pair {pair_id}"
                )
            key = NetworkProductKey(
                AcquisitionKey(primary, frame, swath, channel, polarization),
                AcquisitionKey(secondary, frame, swath, channel, polarization),
                AssetKind.UNWRAPPED_PHASE,
            )
            products.append(
                NetworkProduct(
                    key=key,
                    asset_location=str(
                        generation.generation_root / pair_id / "unwrapped_phase.npy"
                    ),
                    geometry_identity=source.geometry_identity,
                    source_software="faninsar",
                    phase_convention=PhaseConvention.PRIMARY_MINUS_SECONDARY,
                    asset_transform=AssetTransform.for_convention(
                        AssetKind.UNWRAPPED_PHASE,
                        PhaseConvention.PRIMARY_MINUS_SECONDARY,
                    ),
                    content_digest=generation.manifest_digest,
                    lineage=(
                        source.content_digest or source.canonical,
                        generation.manifest_digest,
                    ),
                )
            )
        generation_id = _network_generation_digest(
            [product.content_digest or product.canonical for product in products],
            [generation.manifest_digest],
        )
        self._refresh_network_generation(generation_id, products)

    def refresh_unwrap_generation(self) -> Self:
        """Rebuild the Network cache from the durable ``UNWRAP_CURRENT`` root."""
        from faninsar.stack.stack_generation import open_unwrap_generation

        generation = open_unwrap_generation(self.config.work_dir)
        previous = self._unwrap_generation
        try:
            expected_plan = self.config.mask_plan.identity
            expected_mask = None
            if self.config.mask_plan.references("unwrap"):
                stores = self._pair_artifact_stores(
                    looks=self.config.multilook, ifg_root=None
                )
                try:
                    expected_mask = self._unwrap_mask_identity(stores[0].shape)
                finally:
                    for store in stores:
                        store.close()
            if (
                generation.mask_plan_identity != expected_plan
                or generation.mask_identity != expected_mask
            ):
                message = (
                    "persisted unwrap generation mask plan or materialized mask "
                    "identity does not match the current Stack configuration"
                )
                logger.error(message)
                reject_invalid_state(message)
            self._refresh_network_from_unwrap_generation(generation)
        except Exception:
            generation.close()
            raise
        self._unwrap_generation = generation
        if previous is not None:
            previous.close()
        return self

    @_reclaim_after_stage
    def unwrap(
        self,
        unwrapper: SpatialUnwrapper = _DEFAULT_UNWRAPPER,
    ) -> Self:
        """Spatially unwrap every persisted interferogram in this Stack.

        ``Stack`` owns pair iteration, Dataset materialization, and one atomic
        root-generation publication.  ``unwrapper`` owns the numerical solve
        for one two-dimensional pair.  The complete ordered Pair snapshot is
        frozen at call entry; this method has no temporal/network options and
        does not perform temporal reconciliation or time-series inversion.

        Parameters
        ----------
        unwrapper : SpatialUnwrapper, default=SpatialIRLS()
            Trusted runtime strategy for one pair.  The strategy receives
            Torch tensors in ``(azimuth, range)`` order.  Its result must be a
            same-shape, same-device :class:`SpatialUnwrapResult` whose valid
            output is finite and is a subset of the Dataset support.

        Returns
        -------
        Stack
            This Stack after ``UNWRAP_CURRENT`` has been atomically advanced
            and its Network cache rebuilt from that durable generation.

        Raises
        ------
        ValueError
            If the Pair set, IFG Dataset, or unwrapper result violates the
            scientific shape, dtype, mask, or finite-value contract.
        NoValidSupportError, ResourceAdmissionError
            Propagated from the selected unwrapper before numerical work.
        UnwrapFailedError
            If an admitted strategy fails, returns ``converged=False``, or
            publication cannot complete.  The failed result, when available,
            is attached as ``error.result`` and is never published.

        Notes
        -----
        IFG stores are pinned before Dataset reads and remain open for the
        whole call.  The result generation contains all expected Pairs under
        one Stack-root ``UNWRAP_CURRENT`` pointer; no partial Pair result is
        visible.  Normal Dataset reads do not recompute payload hashes.
        With an active PROPOSAL-0039 mask, the inverted mask is the
        caller-supplied mask of the PROPOSAL-0038 support rule: it is
        intersected with the persisted IFG mask before the strategy is
        invoked (finite phase and finite coherence remain the strategy's
        rule) and is converted to :attr:`StackConfig.device` exactly once.

        """
        import torch

        from faninsar._core.device import parse_device
        from faninsar.data.datasets.ifg import StackInterferogramDataset
        from faninsar.processing.resources import (
            ResourceAdmissionError,
            ResourceAdmissionLedger,
            estimate_spatial_irls_resources,
            estimate_unwrap_decode_resources,
            reserve_estimate,
        )
        from faninsar.processing.unwrap.common import (
            SpatialUnwrapper,
            SpatialUnwrapResult,
        )
        from faninsar.processing.unwrap.errors import (
            NoValidSupportError,
            UnwrapFailedError,
        )
        from faninsar.stack.stack_generation import (
            publish_unwrap_generation,
        )

        if not isinstance(unwrapper, SpatialUnwrapper):
            logger.error("Stack unwrap requires a SpatialUnwrapper instance")
            message = "unwrapper must be a SpatialUnwrapper"
            raise TypeError(message)

        expected_pairs = tuple(_iter_pair_dates(self.pairs))
        if not expected_pairs or len(set(expected_pairs)) != len(expected_pairs):
            reject_invalid_state(
                "Stack unwrap requires a non-empty, unique Pair network"
            )
        pair_ids = tuple(
            f"{primary}_{secondary}" for primary, secondary in expected_pairs
        )
        stores = self._pair_artifact_stores(
            looks=self.config.multilook,
            ifg_root=None,
        )
        device = parse_device(self.config.device)
        products: dict[str, dict[str, np.ndarray]] = {}
        failed_result: SpatialUnwrapResult | None = None
        # PROPOSAL-0039 (AC-5): the active mask is the caller-supplied mask of
        # the PROPOSAL-0038 authoritative support rule. The plane is resolved
        # once per call and converted to the Stack device exactly once with
        # the other arrays; per pair it intersects the persisted IFG mask
        # (finite phase and finite coherence remain the unwrapper's rule).
        mask_plane = self._unwrap_mask_plane(stores[0].shape)
        mask_plan_identity = self.config.mask_plan.identity
        mask_identity = self._unwrap_mask_identity(stores[0].shape)
        mask_removed: Any = None
        if mask_plane is not None:
            mask_removed = torch.as_tensor(
                np.asarray(mask_plane) != 0,
                dtype=torch.bool,
                device=device,
            )
        unwrap_ledger = (
            ResourceAdmissionLedger(self.config.resource_budget, self.config.work_dir)
            if self.config.resource_budget is not None
            else None
        )
        try:
            for pair_id, store in zip(pair_ids, stores, strict=True):
                decode_reservation = None
                solver_reservation = None
                if unwrap_ledger is not None:
                    decode_reservation = reserve_estimate(
                        unwrap_ledger,
                        estimate_unwrap_decode_resources(
                            shape=store.shape,
                            coherence_present="coherence" in store._payloads,
                        ),
                    )
                try:
                    dataset = StackInterferogramDataset.from_generation(store)
                    wrapped_phase = torch.as_tensor(
                        dataset.wrapped_phase,
                        dtype=torch.float32,
                        device=device,
                    )
                    coherence = None
                    if dataset.coherence is not None and bool(
                        np.any(np.isfinite(dataset.coherence))
                    ):
                        coherence = torch.as_tensor(
                            dataset.coherence,
                            dtype=torch.float32,
                            device=device,
                        )
                    valid_mask = torch.as_tensor(
                        dataset.valid_mask,
                        dtype=torch.bool,
                        device=device,
                    )
                    if mask_removed is not None:
                        valid_mask = valid_mask & ~mask_removed
                    if unwrap_ledger is not None and isinstance(unwrapper, SpatialIRLS):
                        support = torch.isfinite(wrapped_phase) & valid_mask
                        if coherence is not None:
                            support &= torch.isfinite(coherence)
                        labels, anchors, horizontal, vertical = unwrapper._graph(
                            support, coherence
                        )
                        if int(anchors.numel()) > 0:
                            areas: list[int] = []
                            for component in range(int(anchors.numel())):
                                rows, columns = torch.where(labels == component)
                                if int(rows.numel()) == 0:
                                    continue
                                height = int(rows.max().item() - rows.min().item() + 1)
                                width = int(
                                    columns.max().item() - columns.min().item() + 1
                                )
                                areas.append(height * width)
                            active_edges = int(
                                horizontal[2].sum().item() + vertical[2].sum().item()
                            )
                            solver_reservation = reserve_estimate(
                                unwrap_ledger,
                                estimate_spatial_irls_resources(
                                    shape=tuple(wrapped_phase.shape),
                                    active_edges=active_edges,
                                    component_bbox_areas=tuple(areas),
                                    max_iter=unwrapper.max_iter,
                                    cg_max_iter=unwrapper.cg_max_iter,
                                    phase_itemsize=wrapped_phase.element_size(),
                                ),
                            )
                    try:
                        result = unwrapper.unwrap(
                            wrapped_phase,
                            coherence=coherence,
                            valid_mask=valid_mask,
                        )
                    except (NoValidSupportError, ResourceAdmissionError):
                        raise
                    except Exception as error:
                        logger.exception("Stack spatial unwrap failed for %s", pair_id)
                        message = f"spatial unwrap failed for Pair {pair_id}"
                        raise UnwrapFailedError(message) from error
                finally:
                    if solver_reservation is not None:
                        solver_reservation.release()
                    if decode_reservation is not None:
                        decode_reservation.release()
                failed_result = (
                    result if isinstance(result, SpatialUnwrapResult) else None
                )
                if not isinstance(result, SpatialUnwrapResult):
                    logger.error(
                        "spatial unwrapper returned an invalid result for %s",
                        pair_id,
                    )
                    message = (
                        "spatial unwrapper returned an invalid result for Pair "
                        f"{pair_id}"
                    )
                    _raise_unwrap_failed(message)
                support = torch.isfinite(wrapped_phase) & valid_mask
                if coherence is not None:
                    support &= torch.isfinite(coherence)
                if result.phase.shape != wrapped_phase.shape:
                    _raise_unwrap_failed(
                        f"spatial unwrapper changed shape for Pair {pair_id}", result
                    )
                if result.phase.device != device or result.valid_mask.device != device:
                    _raise_unwrap_failed(
                        f"spatial unwrapper changed device for Pair {pair_id}", result
                    )
                if bool(torch.any(result.valid_mask & ~support)):
                    _raise_unwrap_failed(
                        f"spatial unwrapper expanded support for Pair {pair_id}",
                        result,
                    )
                if bool(torch.any(~torch.isfinite(result.phase[result.valid_mask]))):
                    _raise_unwrap_failed(
                        "spatial unwrapper returned nonfinite valid phase for "
                        f"Pair {pair_id}",
                        result,
                    )
                if not result.converged or result.failure_reason is not None:
                    _raise_unwrap_failed(
                        f"spatial unwrap did not converge for Pair {pair_id}",
                        result,
                    )
                products[pair_id] = {
                    "unwrapped_phase": result.phase.detach().cpu().numpy(),
                    "valid_mask": result.valid_mask.detach().cpu().numpy(),
                    "component_labels": result.component_labels.detach().cpu().numpy(),
                    "reference_values": result.reference_values.detach().cpu().numpy(),
                }
            generation = publish_unwrap_generation(
                self.config.work_dir,
                pair_ids=pair_ids,
                products=products,
                mask_plan_identity=mask_plan_identity,
                mask_identity=mask_identity,
            )
            previous = self._unwrap_generation
            self._unwrap_generation = generation
            if previous is not None:
                previous.close()
            self._refresh_network_from_unwrap_generation(generation)
        except (NoValidSupportError, ResourceAdmissionError, UnwrapFailedError):
            raise
        except Exception as error:
            logger.exception("Stack unwrap generation failed")
            message = "Stack unwrap generation failed"
            raise UnwrapFailedError(message, failed_result) from error
        finally:
            for store in stores:
                store.close()
        return self

    @_reclaim_after_stage
    def _legacy_temporal_unwrap(
        self,
        *,
        multilook: tuple[int, int] | None = None,
        ifg_root: str | Path | None = None,
        do_spatial: bool = True,
        spatial_executor: SpatialExecutor = "serial",
        spatial_device: str | None = None,
        temporal_device: str | None = None,
        spatial_kwargs: dict[str, Any] | None = None,
        temporal_kwargs: dict[str, Any] | None = None,
        quality_criteria: StackQualityCriteria | None = None,
    ) -> Self:
        """Spatially unwrap and temporally reconcile persisted pair artifacts.

        The method consumes one complete, common-grid IFG artifact for every
        pair in :attr:`pairs`. It never reopens SAFE products or reruns
        co-registration. Missing pairs, additional pairs, mixed look factors,
        and mixed grid shapes fail closed before numerical processing begins.

        Parameters
        ----------
        multilook : tuple[int, int], optional
            Artifact view to consume. Defaults to the configured look factors.
        ifg_root : path, optional
            Root containing pair artifact directories. Defaults to
            ``work_dir/ifg/ml_<az>x<rg>``.
        do_spatial : bool, optional
            Run spatial IRLS before temporal reconciliation. ``False`` is
            intended for already spatially unwrapped test or import products.
        spatial_executor : {"serial", "dask"}, optional
            Pair-level spatial scheduling mode.
        spatial_device, temporal_device : str, optional
            Explicit numerical devices. Defaults to the Stack inversion device.
        spatial_kwargs, temporal_kwargs : dict, optional
            Numerical options forwarded to the existing Stack unwrap
            orchestrator.
        quality_criteria : StackQualityCriteria, optional
            Explicit physical quality limits. Exact algebraic publication
            invariants are always enforced. A resumed artifact must have been
            produced under the identical criteria.

        Returns
        -------
        Stack
            This session with :attr:`unwrap_result` populated.

        """
        from faninsar.processing.unwrap.quality import (
            MetricDistribution,
            StackQualityCriteria,
            StackQualityReport,
        )
        from faninsar.processing.unwrap.stack import unwrap_stack
        from faninsar.stack.ifg_store import write_unwrapped_artifact

        requested_quality_criteria = asdict(quality_criteria or StackQualityCriteria())
        looks = multilook or self.config.multilook
        stores = self._pair_artifact_stores(looks=looks, ifg_root=ifg_root)
        existing = [(store.root / "unwrap_manifest.json").exists() for store in stores]
        if any(existing):
            if not all(existing):
                reject_invalid_state(
                    "partial Stack unwrap generation detected; refusing to mix "
                    "old and new temporal solutions"
                )
            unwrapped = self._qualified_unwrapped_artifacts(stores)
            phase = np.stack(
                [artifact.unwrapped_phase for artifact in unwrapped], axis=0
            )
            persisted_parameters = unwrapped[0].method_parameters
            persisted_quality_criteria = {
                key: value
                for key, value in persisted_parameters["quality_criteria"].items()
                if key
                not in {
                    "max_modulo_closure_p95_rad",
                    "max_sbas_residual_p95_rad",
                }
            }
            if persisted_quality_criteria != requested_quality_criteria:
                reject_invalid_state(
                    "persisted unwrap quality criteria do not match the requested "
                    "quality policy"
                )
            persisted_quality = persisted_parameters["quality_report"]
            quality_report = StackQualityReport(
                passed=bool(persisted_quality["passed"]),
                failures=tuple(str(value) for value in persisted_quality["failures"]),
                observed_pixels=int(persisted_quality["observed_pixels"]),
                full_rank_pixels=int(persisted_quality["full_rank_pixels"]),
                rank_coverage_fraction=float(
                    persisted_quality["rank_coverage_fraction"]
                ),
                published_pixels=int(persisted_quality["published_pixels"]),
                published_full_rank_fraction=float(
                    persisted_quality["published_full_rank_fraction"]
                ),
                converged_fraction=float(persisted_quality["converged_fraction"]),
                cycle_count=int(persisted_quality["cycle_count"]),
                modulo_closure_abs_rad=MetricDistribution(
                    **persisted_quality["modulo_closure_abs_rad"]
                ),
                sbas_residual_abs_rad=MetricDistribution(
                    **persisted_quality["sbas_residual_abs_rad"]
                ),
                integer_correction_max_error=persisted_quality[
                    "integer_correction_max_error"
                ],
                phase_reconstruction_max_error_rad=persisted_quality[
                    "phase_reconstruction_max_error_rad"
                ],
            )
            if not quality_report.passed:
                reject_invalid_state("persisted unwrap quality report did not pass")
            persisted_mask = np.all(np.isfinite(phase), axis=0)
            base_result = unwrap_stack(
                phase,
                _iter_pair_dates(self.pairs),
                do_spatial=False,
                do_temporal=False,
                do_invert=False,
            )
            self.unwrap_result = replace(
                base_result,
                phase_2d_unw=None,
                connected_components=None,
                phase_1d_unw=phase,
                temporal_applied=True,
                temporal_iterations=int(persisted_parameters["temporal_iterations"]),
                temporal_converged=bool(persisted_parameters["temporal_converged"]),
                temporal_converged_mask=persisted_mask,
                temporal_converged_pixels=int(
                    persisted_parameters["temporal_converged_pixels"]
                ),
                temporal_unconverged_pixels=int(
                    persisted_parameters["temporal_unconverged_pixels"]
                ),
                temporal_converged_fraction=float(
                    persisted_parameters["temporal_converged_fraction"]
                ),
                quality_report=quality_report,
            )
            try:
                self.ifg_dirs = [store.root for store in stores]
                self._refresh_network_from_ifg_dirs()
            finally:
                for store in stores:
                    store.close()
            return self

        wrapped_phase = np.stack(
            [store.read().wrapped_phase for store in stores], axis=0
        )
        pair_dates = _iter_pair_dates(self.pairs)
        result = unwrap_stack(
            wrapped_phase,
            pair_dates,
            do_spatial=do_spatial,
            do_temporal=True,
            do_invert=False,
            spatial_executor=spatial_executor,
            spatial_device=spatial_device or self.config.invert_device,
            temporal_device=temporal_device or self.config.invert_device,
            spatial_kwargs=spatial_kwargs,
            temporal_kwargs=temporal_kwargs,
            quality_criteria=quality_criteria,
        )
        if result.phase_1d_unw is None:
            reject_invalid_state("temporal Stack reconciliation produced no phase")
        if result.temporal_converged_pixels < 1:
            reject_invalid_state(
                "temporal Stack reconciliation produced no converged pixels; "
                "no unwrap artifact was published"
            )
        if result.connected_components is None:
            reject_invalid_state("spatial Stack unwrap produced no component labels")
        if result.quality_report is None or not result.quality_report.passed:
            reject_invalid_state("temporal Stack produced no passing quality report")
        method_parameters: dict[str, Any] = {
            "pair_ids": list(result.pair_ids),
            "spatial_method": result.spatial_method,
            "spatial_applied": do_spatial,
            "temporal_applied": result.temporal_applied,
            "temporal_iterations": result.temporal_iterations,
            "temporal_converged": result.temporal_converged,
            "temporal_converged_pixels": result.temporal_converged_pixels,
            "temporal_unconverged_pixels": result.temporal_unconverged_pixels,
            "temporal_converged_fraction": result.temporal_converged_fraction,
            "temporal_unconverged_masked": True,
            "quality_criteria": requested_quality_criteria,
            "quality_report": asdict(result.quality_report),
        }
        for index, store in enumerate(stores):
            phase = np.asarray(result.phase_1d_unw[index], dtype=np.float32)
            connected_components = np.asarray(
                result.connected_components[index],
                dtype=np.int32,
            ).copy()
            connected_components[~np.isfinite(phase)] = 0
            write_unwrapped_artifact(
                store.root,
                unwrapped_phase=phase,
                connected_components=connected_components,
                method="stack_irls",
                method_parameters=method_parameters,
                ifg_manifest_digest=store.manifest_digest,
            )
        self.unwrap_result = result
        try:
            self.ifg_dirs = [store.root for store in stores]
            self._refresh_network_from_ifg_dirs()
        finally:
            for store in stores:
                store.close()
        return self

    @_reclaim_after_stage
    def invert_timeseries(
        self,
        *,
        pair_phases: dict[str, np.ndarray] | None = None,
        device: str | None = None,
        multilook: tuple[int, int] | None = None,
        ifg_root: str | Path | None = None,
        _artifact_stores: Sequence[InterferogramArtifactStore] | None = None,
    ) -> TimeSeriesResult:
        """Invert persisted, temporally reconciled pair phases with SBAS.

        ``pair_phases`` remains available for backwards compatibility. When it
        is omitted, the exact configured pair network is loaded from immutable
        unwrap artifacts without rerunning any upstream processing.
        """
        from faninsar.processing.timeseries.inversion import invert_unwrapped_pairs

        if pair_phases is None:
            stores = (
                list(_artifact_stores)
                if _artifact_stores is not None
                else self._pair_artifact_stores(
                    looks=multilook or self.config.multilook,
                    ifg_root=ifg_root,
                )
            )
            active_unwrap = self.unwrap_result
            if active_unwrap is not None and active_unwrap.phase_1d_unw is not None:
                expected_pair_ids = tuple(
                    f"{store.pair[0]}_{store.pair[1]}" for store in stores
                )
                if active_unwrap.pair_ids != expected_pair_ids:
                    reject_invalid_state(
                        "in-memory temporal result does not match IFG pair order"
                    )
                qualified_phase_stack = active_unwrap.phase_1d_unw
                pair_phases = {
                    pair_id: np.asarray(qualified_phase_stack[index])
                    for index, pair_id in enumerate(expected_pair_ids)
                }
            else:
                unwrapped = self._qualified_unwrapped_artifacts(stores)
                pair_phases = {
                    f"{store.pair[0]}_{store.pair[1]}": artifact.unwrapped_phase
                    for store, artifact in zip(stores, unwrapped, strict=True)
                }
            wavelengths = {store.wavelength_m for store in stores}
            if len(wavelengths) != 1:
                reject_invalid_state("Stack IFG artifacts use mixed wavelengths")
            wavelength_m = next(iter(wavelengths))
        else:
            wavelength_m = None
        if self.unwrap_result is not None:
            self.unwrap_result = replace(
                self.unwrap_result,
                phase_2d_unw=None,
                connected_components=None,
                phase_1d_unw=None,
                corrections_k=None,
                temporal_converged_mask=None,
                timeseries=None,
            )
        active_unwrap = None
        self.timeseries = invert_unwrapped_pairs(
            pair_phases,
            device=device or self.config.invert_device,
            wavelength_m=wavelength_m,
        )
        return self.timeseries

    def analyze_time_series(
        self,
        *,
        solver: str = "sbas",
        model: Any | None = None,
        **kwargs: Any,
    ) -> TimeSeriesResult:
        """Analyze the Stack's inherited Network products after unwrapping.

        ``Stack`` owns the SLC-to-interferogram lifecycle.  Once the unwrap
        stage has committed a complete product set, this method schedules the
        existing SBAS time-series solver over that same pair network.  No
        second Network instance or Dataset injection path is created.

        Parameters
        ----------
        solver : str, default="sbas"
            Currently the shared Stack inversion is the SBAS solver.
        model : object, optional
            Reserved for a future NSBAS model adapter.
        **kwargs : Any
            Options forwarded to :meth:`invert_timeseries`.

        Returns
        -------
        TimeSeriesResult
            Inverted per-date time series.

        Raises
        ------
        ValueError
            If analysis is requested before unwrapped products are ready or
            an unsupported solver is selected.

        """
        del model
        if solver.lower() != "sbas":
            message = f"unsupported Stack time-series solver {solver!r}"
            logger.error(message)
            raise ValueError(message)
        if hasattr(self, "unwrap_result") and self.unwrap_result is None:
            message = (
                "Stack Network analysis is unavailable before committed "
                "interferograms are unwrapped"
            )
            logger.error(message)
            raise ValueError(message)
        return NetworkContract.analyze_time_series(self, **kwargs)

    def _analyze_network_products(
        self,
        _products: NetworkProductIndex,
        *,
        generation_id: str,
        **kwargs: Any,
    ) -> TimeSeriesResult:
        """Run Stack's existing inversion after Network generation admission."""
        # Keep the IFG generation pinned for the complete analysis call.  The
        # solver reads the unwrap payloads before doing its numerical solve;
        # closing the stores immediately after those reads would allow a
        # concurrent collector to reclaim the generation while the solver is
        # still consuming the resulting arrays.  Pure in-memory Network tests
        # intentionally have no Stack config/catalog and retain their light
        # weight inversion seam.
        if not hasattr(self, "config") or not hasattr(self, "pairs"):
            result = self.invert_timeseries(**kwargs)
            return (
                replace(result, revision_id=generation_id)
                if is_dataclass(result)
                else result
            )
        stores = self._pair_artifact_stores(
            looks=kwargs.get("multilook") or self.config.multilook,
            ifg_root=kwargs.get("ifg_root"),
        )
        try:
            observed_generation_id = _observed_network_generation_digest(
                stores, _products
            )
            if observed_generation_id != generation_id:
                reject_invalid_state(
                    "Stack IFG or unwrap artifacts changed after Network generation "
                    "refresh"
                )
            for store in stores:
                store._lease.heartbeat()
            result = self.invert_timeseries(
                _artifact_stores=stores,
                **kwargs,
            )
            # A heartbeat after the solve verifies that the reader remained
            # valid for the whole call.  If renewal fails, fail closed rather
            # than returning a result that was computed from an unpinned view.
            for store in stores:
                store._lease.heartbeat()
            if _observed_network_generation_digest(stores, _products) != generation_id:
                reject_invalid_state(
                    "Stack IFG or unwrap artifacts changed during Network analysis"
                )
            return (
                replace(result, revision_id=generation_id)
                if is_dataclass(result)
                else result
            )
        finally:
            for store in stores:
                store.close()

    def estimate_ionosphere(self, **kwargs: Any) -> list[Any]:
        """Estimate per-pair ionospheric screens (PROPOSAL-0036).

        See
        :func:`faninsar.stack.stack_api.estimate_ionosphere`
        for the full parameter contract.

        The explicit ``ionosphere`` mask stage feeds the estimator's existing
        ``valid_mask`` seam unless the caller supplies one.
        """
        from faninsar.stack.stack_api import estimate_ionosphere

        if (
            self.config.mask_plan.references("ionosphere")
            and kwargs.get("valid_mask") is None
        ):
            mask_plane = self._ionosphere_mask_plane(kwargs.get("ion_multilook"))
            if mask_plane is not None:
                kwargs = {**kwargs, "valid_mask": mask_plane}
        return estimate_ionosphere(self, **kwargs)

    def apply_ionosphere_correction(self, **kwargs: Any) -> dict[str, Path]:
        """Subtract qualified ion screens from unwrapped pair phases.

        See
        :func:`faninsar.stack.stack_api.apply_ionosphere_correction`
        for the full parameter contract.
        """
        from faninsar.stack.stack_api import apply_ionosphere_correction

        return apply_ionosphere_correction(self, **kwargs)

    def invert_ionosphere_dates(self, **kwargs: Any) -> Any:
        """Invert published pair ion screens into per-date screens.

        See
        :func:`faninsar.stack.stack_api.invert_ionosphere_dates`
        for the full parameter contract.
        """
        from faninsar.stack.stack_api import invert_ionosphere_dates

        return invert_ionosphere_dates(self, **kwargs)

    def publish_generation(
        self,
        timeseries_root: str | Path,
        *,
        multilook: tuple[int, int] | None = None,
        ifg_root: str | Path | None = None,
    ) -> StackResultGeneration:
        """Atomically publish the complete IFG, unwrap, and SBAS result set.

        Parameters
        ----------
        timeseries_root : str or pathlib.Path
            Immutable time-series transaction root produced by
            :func:`~faninsar.processing.timeseries.write_timeseries_zarr`.
        multilook : tuple[int, int], optional
            Artifact view to bind. Defaults to the configured look factors.
        ifg_root : str or pathlib.Path, optional
            Root containing the exact pair artifact set.

        Returns
        -------
        StackResultGeneration
            Pinned, hash-validated parent generation. Call :meth:`close` when
            the generation is no longer needed.

        """
        from faninsar.stack.stack_generation import (
            publish_stack_generation,
        )

        stores = self._pair_artifact_stores(
            looks=multilook or self.config.multilook,
            ifg_root=ifg_root,
        )
        try:
            self._qualified_unwrapped_artifacts(stores)
            expected_pair_ids = tuple(
                f"{primary}_{secondary}"
                for primary, secondary in _iter_pair_dates(self.pairs)
            )
            if (
                self.timeseries is None
                or len(self.timeseries.pair_ids) != len(expected_pair_ids)
                or set(self.timeseries.pair_ids) != set(expected_pair_ids)
            ):
                reject_invalid_state(
                    "Stack time-series result does not match the exact pair network"
                )
            return publish_stack_generation(
                self.config.work_dir,
                expected_pair_ids=expected_pair_ids,
                stores=stores,
                timeseries_root=timeseries_root,
            )
        finally:
            for store in stores:
                store.close()

    def open_generation(self) -> StackResultGeneration:
        """Open the current complete derived-result generation for this Stack."""
        from faninsar.stack.stack_generation import open_stack_generation

        generation = open_stack_generation(self.config.work_dir)
        expected_pair_ids = tuple(
            f"{primary}_{secondary}"
            for primary, secondary in _iter_pair_dates(self.pairs)
        )
        if generation.pair_ids != expected_pair_ids:
            generation.close()
            reject_invalid_state(
                "current Stack generation does not match the configured pair network"
            )
        return generation

    def refresh_generation(
        self,
        generation_id: str | None = None,
        products: NetworkProductIndex
        | tuple[NetworkProduct, ...]
        | list[NetworkProduct]
        | None = None,
    ) -> StackResultGeneration | Self:
        """Refresh the pinned derived-result generation from durable storage.

        A caller that keeps a Stack object alive across a new publication can
        use this hook to release the previous lease and atomically observe the
        newest complete generation.  Validation remains delegated to
        :meth:`open_generation`, so incomplete or mismatched generations fail
        closed.
        """
        if generation_id is not None or products is not None:
            if generation_id is None or products is None:
                reject_invalid_state(
                    "Network generation refresh requires id and products together"
                )
            return self._refresh_network_generation(generation_id, products)

        previous = self._generation
        if previous is not None:
            previous.close()
        current = self.open_generation()
        self._generation = current
        return current

    def _qualified_unwrapped_artifacts(
        self,
        stores: Sequence[InterferogramArtifactStore],
    ) -> list[UnwrappedArtifact]:
        """Load one exact, temporally qualified unwrap network."""
        expected_pair_ids = tuple(
            f"{primary}_{secondary}"
            for primary, secondary in _iter_pair_dates(self.pairs)
        )
        artifacts = [store.read_unwrapped() for store in stores]
        if not artifacts:
            reject_invalid_state("qualified temporal artifact network is empty")
        expected_parameters = artifacts[0].method_parameters
        required_numeric = (
            "temporal_iterations",
            "temporal_converged_pixels",
            "temporal_unconverged_pixels",
            "temporal_converged_fraction",
        )
        for artifact in artifacts:
            parameters = artifact.method_parameters
            quality_report = parameters.get("quality_report")
            quality_criteria = parameters.get("quality_criteria")
            if (
                artifact.method != "stack_irls"
                or parameters != expected_parameters
                or parameters.get("pair_ids") != list(expected_pair_ids)
                or parameters.get("temporal_unconverged_masked") is not True
                or parameters.get("temporal_applied") is not True
                or any(
                    not isinstance(parameters.get(name), (int, float))
                    for name in required_numeric
                )
                or int(parameters["temporal_converged_pixels"]) < 1
                or not isinstance(quality_report, dict)
                or quality_report.get("passed") is not True
                or not isinstance(quality_criteria, dict)
            ):
                reject_invalid_state(
                    "persisted unwrap artifact is not a qualified temporal network"
                )
        finite_pixels = int(
            np.count_nonzero(
                np.all(
                    np.isfinite(
                        np.stack(
                            [artifact.unwrapped_phase for artifact in artifacts],
                            axis=0,
                        )
                    ),
                    axis=0,
                )
            )
        )
        if finite_pixels != int(expected_parameters["temporal_converged_pixels"]):
            reject_invalid_state(
                "persisted unwrap convergence count does not match payload bytes"
            )
        return artifacts

    def _pair_artifact_stores(
        self,
        *,
        looks: tuple[int, int],
        ifg_root: str | Path | None,
    ) -> list[InterferogramArtifactStore]:
        """Open the exact ordered common-grid artifact set for this Stack."""
        from faninsar.stack.ifg_store import InterferogramArtifactStore

        azimuth_looks, range_looks = (int(looks[0]), int(looks[1]))
        root = (
            Path(ifg_root)
            if ifg_root is not None
            else self.config.work_dir / "ifg" / f"ml_{azimuth_looks}x{range_looks}"
        )
        expected_pairs = _iter_pair_dates(self.pairs)
        expected_ids = [
            f"{primary}_{secondary}" for primary, secondary in expected_pairs
        ]
        if not root.is_dir():
            reject_invalid_state(f"Stack IFG artifact view is missing: {root}")
        discovered_ids = sorted(
            path.parent.name for path in root.glob("*/manifest.json")
        )
        if sorted(expected_ids) != discovered_ids:
            reject_invalid_state(
                "Stack IFG artifact pair set does not exactly match the configured "
                f"network: expected={sorted(expected_ids)!r}, "
                f"discovered={discovered_ids!r}"
            )
        stores = [
            InterferogramArtifactStore.open(root / pair_id) for pair_id in expected_ids
        ]
        shapes = {store.shape for store in stores}
        artifact_looks = {store.looks for store in stores}
        artifact_domains = {store.domain for store in stores}
        artifact_wavelengths = {store.wavelength_m for store in stores}
        artifact_grid_identities = {store.grid_identity for store in stores}
        if len(shapes) != 1:
            reject_invalid_state("Stack IFG artifacts do not share one common grid")
        if artifact_looks != {(azimuth_looks, range_looks)}:
            reject_invalid_state("Stack IFG artifacts use mixed or unexpected looks")
        if artifact_domains != {self.config.coregistration_grid}:
            reject_invalid_state("Stack IFG artifacts use mixed or unexpected domains")
        if len(artifact_wavelengths) != 1:
            reject_invalid_state("Stack IFG artifacts use mixed radar wavelengths")
        if len(artifact_grid_identities) != 1:
            reject_invalid_state("Stack IFG artifacts use mixed coordinate grids")
        for expected_pair, store in zip(expected_pairs, stores, strict=True):
            if store.pair != expected_pair:
                reject_invalid_state(
                    "Stack IFG artifact pair order differs from the configured network"
                )
        return stores


def _iter_pair_dates(pairs: Pairs) -> list[tuple[str, str]]:
    """Yield chronological ``(primary, secondary)`` IDs from ``Pairs``.

    ``Pair`` is the single authority for role direction.  Reconstructing the
    roles from a filename would duplicate (and eventually drift from) the
    FanInSAR ordering contract, so each edge is normalized through ``Pair``
    before it enters Stack persistence or analysis.
    """
    from faninsar.core.pairs import Pair

    out: list[tuple[str, str]] = []
    for values in pairs.values:
        pair = Pair(values)
        out.append((pair.primary_string(), pair.secondary_string()))
    return out


def _canonical_pair_strings(values: tuple[str, str]) -> tuple[str, str]:
    """Normalize persisted pair roles through the FanInSAR :class:`Pair` type."""
    from faninsar.core.pairs import Pair

    try:
        pair = Pair(
            tuple(
                datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
                for value in values
            )
        )
    except (TypeError, ValueError):
        reject_invalid_state(f"Stack IFG artifact has an invalid Pair: {values!r}")
    return pair.primary_string(), pair.secondary_string()


def _read_unwrap_manifest_identity(store: Any) -> tuple[str, str]:
    """Read the validated unwrap generation and manifest identities."""
    try:
        manifest = json.loads(
            (store.root / "unwrap_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as error:
        reject_invalid_state(f"Stack unwrap manifest cannot be read: {error}")
    if not isinstance(manifest, dict):
        reject_invalid_state("Stack unwrap manifest must be a JSON object")
    generation_id = manifest.get("generation_id")
    manifest_digest = manifest.get("manifest_digest")
    if not isinstance(generation_id, str) or not isinstance(manifest_digest, str):
        reject_invalid_state("Stack unwrap manifest identity is incomplete")
    return generation_id, manifest_digest


def _network_generation_digest(
    ifg_manifest_digests: Iterable[str],
    unwrap_manifest_digests: Iterable[str] = (),
) -> str:
    """Hash one unambiguous IFG-plus-unwrap Network generation identity."""
    unwrap_digests = tuple(unwrap_manifest_digests)
    if not unwrap_digests:
        # Keep the lightweight in-memory Stack test seam deterministic while a
        # persisted Stack always supplies unwrap identities after unwrap.
        return hashlib.sha256(
            "|".join(sorted(ifg_manifest_digests)).encode("utf-8")
        ).hexdigest()
    components = [
        *(f"ifg:{digest}" for digest in ifg_manifest_digests),
        *(f"unwrap:{digest}" for digest in unwrap_digests),
    ]
    return hashlib.sha256("|".join(sorted(components)).encode("utf-8")).hexdigest()


def _observed_network_generation_digest(
    stores: Sequence[InterferogramArtifactStore],
    products: NetworkProductIndex,
) -> str:
    """Validate current stores against products and return their generation ID."""
    ifg_manifest_digests = [store.manifest_digest for store in stores]
    indexed_unwrapped = {
        (
            product.primary.acquisition_id,
            product.secondary.acquisition_id,
        ): product
        for product in products.products
        if product.key.product_kind is AssetKind.UNWRAPPED_PHASE
    }
    if not indexed_unwrapped:
        roots = [getattr(store, "root", None) for store in stores]
        if roots and all(
            isinstance(root, Path) and (root / "unwrap_manifest.json").is_file()
            for root in roots
        ):
            reject_invalid_state(
                "Stack Network index is missing current unwrapped phase products"
            )
        return _network_generation_digest(ifg_manifest_digests)
    if len(indexed_unwrapped) != len(stores):
        reject_invalid_state(
            "Stack Network index does not contain one unwrapped product per IFG"
        )

    unwrap_manifest_digests: list[str] = []
    for store in stores:
        # read_unwrapped validates CURRENT, its manifest digest, payload hashes,
        # and the binding back to this exact IFG before the digest is admitted.
        store.read_unwrapped()
        _generation_id, unwrap_digest = _read_unwrap_manifest_identity(store)
        pair = _canonical_pair_strings(store.pair)
        product = indexed_unwrapped.get(pair)
        if product is None or product.content_digest != unwrap_digest:
            reject_invalid_state(
                "Stack Network unwrapped product does not match current unwrap"
            )
        if product.lineage != (store.manifest_digest, unwrap_digest):
            reject_invalid_state(
                "Stack Network unwrapped product lineage does not match current "
                "artifacts"
            )
        unwrap_manifest_digests.append(unwrap_digest)
    return _network_generation_digest(ifg_manifest_digests, unwrap_manifest_digests)


def _normalize_multilook(
    multilook: tuple[int, int] | list[tuple[int, int]] | Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Accept one look pair or a list of look pairs."""
    if (
        isinstance(multilook, tuple)
        and len(multilook) == 2
        and not isinstance(multilook[0], (list, tuple))
    ):
        return [(int(multilook[0]), int(multilook[1]))]
    return [(int(a), int(b)) for a, b in multilook]  # type: ignore[misc]


def _save_arcs(path: Path, arcs: list[MisregArc]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "primary": a.primary,
            "secondary": a.secondary,
            "azimuth_shift_px": a.azimuth_shift_px,
            "range_shift_px": a.range_shift_px,
            "azimuth_sigma_px": a.azimuth_sigma_px,
            "range_sigma_px": a.range_sigma_px,
            "method": a.method,
            "n_valid": a.n_valid,
        }
        for a in arcs
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_arcs(path: Path) -> list[MisregArc]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [MisregArc(**row) for row in data]

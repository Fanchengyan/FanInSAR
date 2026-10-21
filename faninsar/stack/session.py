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
import re
import sys
import sysconfig
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, ParamSpec, Self, TypeVar

import numpy as np

from faninsar.io.storage.ifg_store import UNWRAP_KIND
from faninsar.io.storage.product import open_product
from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    form_merged_scene_interferogram,
    validate_reference_binding,
)
from faninsar.logging import setup_logger
from faninsar.network.network import Network, NetworkContract
from faninsar.network.products import (
    ASSET_KIND_GEOCODE_KERNELS,
    AcquisitionKey,
    AssetKind,
    AssetTransform,
    NetworkProduct,
    NetworkProductIndex,
    NetworkProductKey,
    PhaseConvention,
)
from faninsar.processing.coregistration.misreg_network import (
    DateMisreg,
    MisregArc,
    invert_optional_pair_misregistration,
)
from faninsar.processing.errors import (
    InvalidProcessingStateError,
    reject_invalid_state,
)
from faninsar.processing.geometry import RasterDEM, SourceDEM
from faninsar.processing.interferometry.pair import validate_coherence_window
from faninsar.processing.interferometry.phase_filter import (
    FilterProvenance,
    GoldsteinWerner,
    PhaseFilter,
)
from faninsar.processing.unwrapping.errors import UnwrapFailedError
from faninsar.processing.unwrapping.irls import SpatialIRLS
from faninsar.processing.unwrapping.snaphu_backend import Snaphu
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.config import (
    ActivationMode,
    CoregMode,
    FlattenStage,
    StackConfig,
)
from faninsar.stack.mask_plan import MaskPlan, StageName
from faninsar.stack.provider import SourceHandle
from faninsar.stack.search import (
    MissionPreparedInputs,
    MissionSearchProfile,
    SelectionRequest,
    merge_profile_filters,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from faninsar.core.acquisition import Acquisition
    from faninsar.core.pair import Pairs
    from faninsar.data.query import BoundingBox, Points, Polygons
    from faninsar.io.storage.ifg_store import InterferogramArtifactStore
    from faninsar.io.storage.product import Product
    from faninsar.missions.s1.processing import (
        BurstSelection,
        CoregistrationGrid,
        ProductionPairState,
    )
    from faninsar.processing.geocoding import GeocodeableProduct
    from faninsar.processing.geocoding.geocoder import GeocodedRaster
    from faninsar.processing.geometry import DEM, GridSpec
    from faninsar.processing.geometry.prepared import (
        ActivationToken,
        StackActivationBinding,
    )
    from faninsar.processing.mosaicking.grid import GeoGridSpec
    from faninsar.processing.runtime.device import GpuMemoryReclaim
    from faninsar.processing.runtime.resources import ResourceBudget
    from faninsar.processing.unwrapping.common import SpatialUnwrapper
    from faninsar.remote.records import CatalogItem, RemoteResourceBudget
    from faninsar.stack.misreg import MisregRecord
    from faninsar.stack.provider import StackSceneProvider
    from faninsar.timeseries.results import TimeSeries

logger = setup_logger(__name__)

GEOCODED_KIND = "geocoded"


def _scene_array_shape(scene: object) -> tuple[int, ...] | None:
    """Return the raster shape exposed by a production scene.

    Production Sentinel-1 scenes wrap the NumPy samples in ``BurstArray``;
    lightweight providers may expose an array-like object directly.  Keeping
    this normalization at the Stack boundary prevents resume metadata from
    depending on one provider's representation.
    """
    array = getattr(scene, "array", None)
    shape = getattr(array, "shape", None)
    if shape is None:
        shape = getattr(getattr(array, "samples", None), "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        return None


_DEFAULT_PHASE_FILTER = GoldsteinWerner(alpha=0.5, patch_size=32)
_DEFAULT_UNWRAPPER = Snaphu()


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

    Mirrors ``faninsar.missions.s1.processing._roi_geometry`` so the
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


_GIT_REVISION_CACHE: dict[str, str | None] = {}
_GIT_REVISION_SENTINEL: object = object()


def _read_git_revision(repository: Path) -> str | None:
    """Read the checked-out revision of a git repository without subprocess.

    Only plain checkouts (``.git/HEAD`` pointing at a ref file, or a detached
    ``.git/HEAD``) are resolved; anything else returns ``None``.  The result
    is deterministic for a given checkout, unlike a ``git`` subprocess whose
    availability depends on the calling environment.
    """
    git_dir = repository / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8", errors="replace").strip()
        marker = "gitdir: "
        if not text.startswith(marker):
            return None
        git_dir = repository / text[len(marker) :].strip()
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8", errors="replace").strip()
    if re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    if not value.startswith("ref: "):
        return None
    return _read_git_ref(git_dir, value[len("ref: ") :].strip())


def _read_git_ref(git_dir: Path, ref: str) -> str | None:
    """Resolve one git ref name to a commit id, loose file or packed."""
    if not re.fullmatch(r"[A-Za-z0-9/._-]+", ref):
        return None
    for candidate in (git_dir, _common_git_dir(git_dir)):
        if candidate is None:
            continue
        ref_file = candidate / ref
        if ref_file.is_file():
            resolved = ref_file.read_text(encoding="utf-8", errors="replace").strip()
            if resolved:
                return resolved
        packed = candidate / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.strip().endswith(f" {ref}"):
                    return line.split()[0]
    return None


def _common_git_dir(git_dir: Path) -> Path | None:
    """Return the common git directory of a possible worktree git dir."""
    commondir = git_dir / "commondir"
    if not commondir.is_file():
        return None
    value = commondir.read_text(encoding="utf-8", errors="replace").strip()
    if not value:
        return None
    common = Path(value)
    return common if common.is_absolute() else (git_dir / common).resolve()


def _faninsar_git_revision() -> str | None:
    """Return the enclosing FanInSAR Git revision, when the checkout has one.

    The revision is resolved once per process: it cannot change while that
    process runs, and resume identities must not depend on a transient
    subprocess failure.
    """
    cached = _GIT_REVISION_CACHE.get("revision", _GIT_REVISION_SENTINEL)
    if cached is not _GIT_REVISION_SENTINEL:
        return cached  # type: ignore[return-value]
    repository = Path(__file__).resolve().parents[2]
    revision = None
    try:
        revision = _read_git_revision(repository)
    except OSError:
        revision = None
    _GIT_REVISION_CACHE["revision"] = revision
    return revision


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
        # The IFG binds each scene by write id, so a screen is identified by
        # its scene product and unit; the write id stands in for its digest.
        role_digests = {
            unit.tag: store.write_id
            for unit in store.units
            if unit.phase_screen_path is not None
        }
        if role_digests:
            digests[role] = dict(sorted(role_digests.items()))
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
    from faninsar.core.pair import PairsFactory

    factory = PairsFactory(list(dates))
    return factory.from_interval(max_interval=max_interval, max_days=max_days)


_GEOCODE_KINDS = ("complex", "phase", "continuous", "label")
_GEOCODE_ROWS_PER_CHUNK = 256
_GEOCODE_LAYOUT_SCHEMA = 1


def _geocode_product_name(product: object) -> str:
    """Return the validated persisted file stem of a geocode product."""
    name = getattr(product, "name", None)
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        message = (
            "geocode product must expose a 'name' that is one safe path "
            f"component; got {name!r}"
        )
        logger.error(message)
        raise ValueError(message)
    return name


def _geocode_int_pair(product: object, attribute: str) -> tuple[int, int]:
    """Return a validated two-integer product attribute."""
    value = getattr(product, attribute, None)
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(entry, bool) or not isinstance(entry, int) for entry in value)
    ):
        message = (
            f"geocode product attribute {attribute!r} must be a two-integer "
            f"tuple; got {value!r}"
        )
        logger.error(message)
        raise ValueError(message)
    return int(value[0]), int(value[1])


def _geocode_resolve_kind(product: object, kind: str | None) -> tuple[str, str | None]:
    """Resolve the resampling kernel from ``kind`` or the asset vocabulary.

    Returns
    -------
    resolved_kind, asset_tag : tuple[str, str or None]
        The kernel name and the normalized asset vocabulary tag used in the
        persisted product identity.

    """
    raw = getattr(product, "asset_kind", None)
    asset_tag = None if raw is None else str(raw)
    if kind is not None:
        if kind not in _GEOCODE_KINDS:
            message = f"kind must be one of {_GEOCODE_KINDS}; got {kind!r}"
            logger.error(message)
            raise ValueError(message)
        return kind, asset_tag
    if raw is None:
        message = (
            "geocode could not infer a resampling kernel: the product carries "
            "no asset_kind; pass kind= explicitly"
        )
        logger.error(message)
        raise ValueError(message)
    try:
        asset_kind = AssetKind(raw)
    except ValueError as error:
        message = (
            "geocode could not infer a resampling kernel from asset_kind "
            f"{raw!r}; pass kind= explicitly"
        )
        logger.exception(message)
        raise ValueError(message) from error
    kernel = ASSET_KIND_GEOCODE_KERNELS.get(asset_kind)
    if kernel is None:
        message = (
            f"asset_kind {asset_kind!r} has no geocoding kernel mapping; "
            "pass kind= explicitly"
        )
        logger.error(message)
        raise ValueError(message)
    return str(kernel), asset_tag


def _geocode_product_identity(
    *,
    name: str,
    kind: str,
    asset_tag: str | None,
    grid: GridSpec,
    full_radar_shape: tuple[int, int],
    looks: tuple[int, int],
    origin: tuple[int, int],
    values: np.ndarray,
) -> str:
    """Return the qualified identity digest of one geocode request.

    The digest covers the persisted name, the kernel, the destination grid at
    full precision, the radar frame and its crop/multilook mapping, and the
    source payload content — everything that would change the committed
    product. The LUT cache identity is derived separately inside the builder.
    """
    array = np.ascontiguousarray(values)
    payload = {
        "schema": _GEOCODE_LAYOUT_SCHEMA,
        "name": name,
        "kind": kind,
        "asset_kind": asset_tag,
        "grid": {
            "crs": str(grid.crs),
            "transform": [float(value) for value in grid.transform],
            "shape": [int(grid.height), int(grid.width)],
        },
        "full_radar_shape": [int(full_radar_shape[0]), int(full_radar_shape[1])],
        "multilook": [int(looks[0]), int(looks[1])],
        "origin": [int(origin[0]), int(origin[1])],
        "values": {
            "dtype": str(array.dtype),
            "shape": [int(array.shape[0]), int(array.shape[1])],
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_geocoded_product(
    target_dir: Path,
    name: str,
    identity: str,
    grid: GridSpec,
) -> GeocodedRaster | None:
    """Return the committed geocoded child when its identity matches.

    A present child with a mismatched identity fails with an actionable
    error; only a matching one is reused. A missing child returns ``None``.
    """
    from faninsar.io.storage.product import open_product

    item_path = target_dir / "item.json"
    if not item_path.is_file():
        return None
    product = open_product(target_dir, kind=GEOCODED_KIND)
    recorded = {
        "identity": str(product.properties.get("faninsar:identity", "")),
        "name": str(product.properties.get("faninsar:name", "")),
    }
    if recorded["identity"] != identity:
        message = (
            f"persisted geocode product {name!r} exists with a different "
            "identity (grid, kernel, or source content changed); pass "
            "overwrite=True to replace it"
        )
        logger.error(message)
        raise ValueError(message)
    import rasterio

    from faninsar.processing.geocoding.geocoder import GeocodedRaster

    with rasterio.open(product.asset_path("values")) as source:
        values = source.read(1)
    with rasterio.open(product.asset_path("valid")) as source:
        valid = source.read(1).astype(bool)
    return GeocodedRaster(values=values, valid=valid, grid=grid)


def _commit_geocoded_product(
    target_dir: Path,
    name: str,
    identity: str,
    raster: GeocodedRaster,
    *,
    overwrite: bool,
) -> None:
    """Commit one geocoded child product (COG payloads and ``item.json``).

    Payloads are written into the kernel's staging directory and the Item
    last, so an interrupted commit leaves the previous product intact.
    """
    from affine import Affine

    from faninsar.io.storage.product import commit_product
    from faninsar.network.readers.raster_io import write_cog
    from faninsar.processing.geometry.grids import GeoGrid

    values = np.asarray(raster.values)
    if values.dtype == np.bool_:
        values = values.astype(np.uint8)
    if np.issubdtype(values.dtype, np.complexfloating):
        nodata = None
    elif np.issubdtype(values.dtype, np.floating):
        nodata = float("nan")
    else:
        nodata = 0
    geogrid = GeoGrid(
        Affine(*raster.grid.transform),
        raster.grid.shape,
        raster.grid.crs,
    )
    if (target_dir / "item.json").is_file() and not overwrite:
        from faninsar.io.storage.product import open_product

        existing = open_product(target_dir, kind=GEOCODED_KIND)
        if str(existing.properties.get("faninsar:identity")) == identity:
            logger.info("geocode: product %r already committed; keeping it", name)
            return
        message = (
            f"persisted geocode product {name!r} exists with a different "
            "identity; pass overwrite=True to replace it"
        )
        logger.error(message)
        raise ValueError(message)

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        values_name = f"{name}.tif"
        valid_name = f"{name}_valid.tif"
        write_cog(
            values,
            staging / values_name,
            geogrid,
            nodata=nodata,
            dtype=str(values.dtype),
            overwrite=True,
        )
        write_cog(
            raster.valid.astype(np.uint8),
            staging / valid_name,
            geogrid,
            nodata=None,
            dtype="uint8",
            overwrite=True,
        )
        return {
            "values": {
                "href": values_name,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
            "valid": {
                "href": valid_name,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
        }

    commit_product(
        target_dir,
        kind=GEOCODED_KIND,
        properties={
            "faninsar:name": name,
            "faninsar:identity": identity,
            "faninsar:grid": {
                "crs": str(raster.grid.crs),
                "transform": [float(v) for v in raster.grid.transform],
                "shape": [int(raster.grid.height), int(raster.grid.width)],
            },
        },
        write_payloads=write_payloads,
        overwrite=overwrite,
    )


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
    _nesd_skipped_no_windows: bool = False
    pair_states: dict[str, ProductionPairState] = field(default_factory=dict)
    ifg_dirs: list[Path] = field(default_factory=list)
    timeseries: TimeSeries | None = None
    _prepared: bool = False
    _mission_prepared: MissionPreparedInputs | None = field(default=None, repr=False)
    _network_generation_id: str | None = field(default=None, repr=False)
    _network_product_index: NetworkProductIndex | None = field(default=None, repr=False)
    _network_view: Network | None = field(default=None, repr=False)
    _effective_roi_resolution: _EffectiveRoi | None = field(default=None, repr=False)
    _materialized_masks: dict[tuple[tuple[str, ...], str], object] = field(
        default_factory=dict, repr=False
    )
    _radar_projected_masks: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False
    )
    _radar_mask_identities: dict[tuple[str, tuple[int, int], tuple[int, int]], str] = (
        field(default_factory=dict, repr=False)
    )
    _dem_source: SourceDEM | None = field(default=None, repr=False)
    _dem_wgs84: RasterDEM | None = field(default=None, repr=False)

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
        self._capture_dem_source()

    def _scene_dir(self, date_id: str) -> Path:
        """Return the scene product directory of one acquisition."""
        return self.config.work_dir / "coreg" / date_id

    @property
    def scenes(self) -> dict[str, CoregisteredSceneStore]:
        """Return the committed scene products by acquisition date.

        Discovered from ``coreg/<date>/item.json``; the Reference is one of
        them (see :attr:`reference_scene`).
        """
        from faninsar.io.storage.product import discover_products
        from faninsar.io.storage.scene_store import SCENE_KIND

        found: dict[str, CoregisteredSceneStore] = {}
        for product in discover_products(
            self.config.work_dir / "coreg", kind=SCENE_KIND
        ):
            store = CoregisteredSceneStore.open(product.path)
            found[store.date_id] = store
        return dict(sorted(found.items()))

    @property
    def reference_scene(self) -> CoregisteredSceneStore | None:
        """Return the committed Reference scene product, if any."""
        path = self._scene_dir(self.reference)
        if not (path / "item.json").is_file():
            return None
        store = CoregisteredSceneStore.open(path)
        if not store.is_reference:
            reject_invalid_state(f"{path} does not hold the Stack Reference scene")
        return store

    @property
    def coreg_paths(self) -> dict[str, Path]:
        """Return the scene product directory of every coregistered date."""
        return {date_id: store.root for date_id, store in self.scenes.items()}

    @property
    def misreg(self) -> MisregRecord | None:
        """Return the committed misregistration product, if any."""
        from faninsar.stack.misreg import read_misreg

        return read_misreg(self.config.work_dir / "misreg")

    @property
    def interferograms(self) -> Any:
        """Return the read-only IFG product collection of this Stack.

        Raises
        ------
        InvalidProcessingStateError
            If the planned Pair network has no complete IFG product set.

        """
        from faninsar.network.readers.stack import StackInterferogramCollection

        return StackInterferogramCollection(self)

    @property
    def unwrapped(self) -> Any:
        """Return the unwrapped-phase cube of every planned Pair.

        Returns
        -------
        xarray.DataArray
            ``(pair, y, x)`` cube read from the per-Pair unwrap products.

        Raises
        ------
        InvalidProcessingStateError
            If any planned Pair has no current unwrap product.

        """
        return self.interferograms.open_stack("unw_phase")

    def status(self) -> dict[str, object]:
        """Compare the planned networks with the committed products.

        Returns
        -------
        dict
            Per-stage ``planned``, ``committed``, ``missing``, and ``stale``
            identifiers. A stale product is bound to a replaced upstream;
            only ``overwrite=True`` recomputes it.

        """
        from faninsar.io.storage.ifg_store import (
            InterferogramArtifactStore,
            unwrap_root,
        )
        from faninsar.io.storage.product import open_product

        work = self.config.work_dir
        looks = self.config.multilook
        report: dict[str, object] = {
            "work_dir": str(work),
            "reference": self.reference,
            "activation_mode": self.config.activation_mode,
        }
        planned_dates = sorted(self.catalog.paths)
        scene_stale: list[str] = []
        scene_committed: list[str] = []
        try:
            reference_id = self.reference_scene.write_id
        except InvalidProcessingStateError:
            reference_id = None
        for date, store in sorted(self.scenes.items()):
            scene_committed.append(date)
            if date != self.reference and store.inputs.get("reference") != reference_id:
                scene_stale.append(date)
        report["scenes"] = {
            "planned": planned_dates,
            "committed": scene_committed,
            "missing": [d for d in planned_dates if d not in scene_committed],
            "stale": scene_stale,
        }
        misreg_record = self.misreg
        report["misreg"] = {
            "committed": misreg_record is not None,
        }

        ifg_root = work / "ifg" / f"ml_{looks[0]}x{looks[1]}"
        planned_pairs = [
            f"{first}_{second}" for first, second in _iter_pair_dates(self.pairs)
        ]
        ifg_committed: list[str] = []
        ifg_stale: list[str] = []
        scene_write_ids: dict[str, str] = {}
        for date, store in sorted(self.scenes.items()):
            scene_write_ids[date] = store.write_id
        if ifg_root.is_dir():
            for path in sorted(ifg_root.glob("*/item.json")):
                pair_id = path.parent.name
                ifg_committed.append(pair_id)
                try:
                    store = InterferogramArtifactStore.open(path.parent)
                except InvalidProcessingStateError:
                    ifg_stale.append(pair_id)
                    continue
                expected = {
                    "primary": scene_write_ids.get(store.pair[0]),
                    "secondary": scene_write_ids.get(store.pair[1]),
                }
                if any(
                    store.source_manifest_digests.get(role) != write_id
                    for role, write_id in expected.items()
                ):
                    ifg_stale.append(pair_id)
        report["interferograms"] = {
            "planned": planned_pairs,
            "committed": ifg_committed,
            "missing": [p for p in planned_pairs if p not in ifg_committed],
            "stale": ifg_stale,
        }

        unwrap_committed: list[str] = []
        unwrap_stale: list[str] = []
        for pair_id in ifg_committed:
            has_item = (ifg_root / pair_id).exists() and (
                unwrap_root(ifg_root / pair_id) / "item.json"
            ).is_file()
            if pair_id in ifg_stale:
                # Transitive staleness: the unwrap consumes the stale IFG.
                if has_item:
                    unwrap_stale.append(pair_id)
                continue
            store = InterferogramArtifactStore.open(ifg_root / pair_id)
            try:
                if store.has_unwrapped():
                    unwrap_committed.append(pair_id)
            except InvalidProcessingStateError:
                if has_item:
                    unwrap_stale.append(pair_id)
            finally:
                store.close()
        report["unwrap"] = {
            "planned": planned_pairs,
            "committed": unwrap_committed,
            "missing": [p for p in planned_pairs if p not in unwrap_committed],
            "stale": unwrap_stale,
        }

        timeseries_path = work / "timeseries"
        timeseries_committed = (timeseries_path / "item.json").is_file()
        timeseries_stale = False
        if timeseries_committed:
            try:
                item = open_product(timeseries_path)
                current = self.interferograms.unwrap_write_ids()
                timeseries_stale = any(
                    item.inputs.get(pair) != write_id
                    for pair, write_id in current.items()
                )
            except Exception:
                logger.warning("timeseries product could not be checked", exc_info=True)
                timeseries_stale = True
        report["timeseries"] = {
            "committed": timeseries_committed,
            "stale": timeseries_stale,
        }
        report["analysis_ready"] = self.analysis_ready
        return report

    def verify(self) -> list[str]:
        """Re-hash every committed product payload in this work directory.

        Normal reads skip hashing; this method checks every ``file:checksum``
        recorded in every ``item.json`` below the work directory (``cache/``
        is skipped).

        Returns
        -------
        list of str
            One entry per damaged asset, ``"<product>: <asset>"``; empty
            means every payload matches its write-time checksum.

        """
        from faninsar.io.storage.product import verify_product

        problems: list[str] = []
        for item_path in sorted(self.config.work_dir.rglob("item.json")):
            relative = item_path.relative_to(self.config.work_dir)
            if relative.parts[0] == "cache" or any(
                part.startswith(".") for part in relative.parts
            ):
                continue
            problems.extend(
                f"{item_path.parent}: {asset}"
                for asset in verify_product(item_path.parent)
            )
        return problems

    def _reference_radar_geometry(self) -> tuple[object, tuple[int, int]]:
        """Return the persisted Reference radar geometry and scene-grid shape.

        Raises
        ------
        ValueError
            If no radar Reference scene with a geometry record is committed.

        """
        from faninsar.processing.geometry.record import radar_geometry_from_record

        reference = self.reference_scene
        record = None if reference is None else reference.radar_geometry
        if reference is None or reference.domain != "radar" or record is None:
            message = (
                "the Reference radar geometry is not available; run "
                "coregister_scenes with coregistration_grid='radar' first"
            )
            logger.error(message)
            raise ValueError(message)
        return radar_geometry_from_record(record), reference.grid_shape

    @property
    def dem(self) -> RasterDEM | object | None:
        """Return the live DEM after preparation.

        A source recipe stays private until it is materialized onto the Stack
        grid. Constant and raster DEMs are returned as given.
        """
        current = self.config.dem
        if current is None or isinstance(current, SourceDEM):
            return None
        return current

    @property
    def network(self) -> Network | None:
        """Return the Network view over this Stack's committed products.

        The Stack and Network share one product index; until every planned
        Pair has a current IFG and unwrap product the view is unavailable.
        """
        if not self.analysis_ready:
            return None
        if (
            self._network_view is None
            or self._network_view.network_generation_id != self._network_generation_id
        ):
            self._network_view = Network._from_stack(self)
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
        from faninsar.processing.geometry import api as dem_api

        selected = dem or self.config.dem
        if selected is None:
            selected = dem_api.DEM.from_source("auto")
        if isinstance(selected, dem_api.RasterDEM):
            dem_api.ensure_dem_covers_grid(selected, self.grid)
        cache_dir = self.config.dem_cache_dir or (self.config.work_dir / "dem-cache")
        if hasattr(selected, "cache_dir") and selected.cache_dir is None:
            selected.cache_dir = cache_dir  # type: ignore[attr-defined]
        return selected.to_raster(
            self.grid,
            vertical_datum="ellipsoidal",
            budget=self.config.resource_budget,
        )

    def geocode(
        self,
        product: GeocodeableProduct,
        *,
        grid: GridSpec | None = None,
        resolution_m: float | None = None,
        bounds: object | None = None,
        kind: str | None = None,
        overwrite: bool = False,
    ) -> GeocodedRaster | GeocodeableProduct:
        """Geocode one radar-domain product onto the Stack grid on demand.

        This is the single entry point that turns any radar-domain result of
        any stage into a geographic raster. It composes the shared geocoding
        engine (:class:`~faninsar.processing.geocoding.Geocoder`) with the
        authoritative Stack grid, the reference-scene radar geometry from the
        coregistration context, and the Stack DEM. The kernel is fixed by the
        physical ``kind`` — never by dtype — following
        :data:`~faninsar.network.products.ASSET_KIND_GEOCODE_KERNELS`.

        Parameters
        ----------
        product : GeocodeableProduct
            Radar-domain product structurally exposing ``name``, ``values``
            (2-D radar raster), ``asset_kind``, ``full_radar_shape``,
            ``multilook``, and ``origin``; see
            :class:`~faninsar.processing.geocoding.GeocodeableProduct` and
            :class:`~faninsar.processing.geocoding.RadarDomainProduct`.
            ``values.shape`` must match the layer frame implied by the
            crop/multilook mapping (see
            :func:`~faninsar.processing.geocoding.product.layer_frame_shape`).
        grid : GridSpec, optional
            Explicit destination grid. When omitted, the Stack grid resolves
            through :meth:`resolve_grid` precedence (``StackConfig.grid``,
            then ``bounds``/``StackConfig.roi``, then selected footprints).
        resolution_m : float, optional
            Automatic-grid spacing override; ignored with an explicit grid.
        bounds : object, optional
            WGS84 ROI used when the destination grid resolves automatically.
            An explicit ``grid`` always wins.
        kind : {"complex", "phase", "continuous", "label"}, optional
            Resampling kernel. When omitted it is inferred from the product
            ``asset_kind`` (complex interferograms keep Lanczos-4, wrapped
            phase rides ``exp(i*phi)`` + Lanczos-4, smooth real quantities
            are bilinear, hard labels nearest).
        overwrite : bool, optional
            Persistence flag shared by every persisting stage
            (PROPOSAL-0033). The product commits to
            ``<work_dir>/geocode/<name>/`` — a product directory with COG
            payloads and an ``item.json`` recording the request identity. A
            committed child with a matching identity is reused and returned
            without recomputing; a missing or mismatched one fails
            actionably. ``overwrite=True`` replaces it.

        Returns
        -------
        GeocodedRaster or GeocodeableProduct
            Values, validity, and destination grid of the geocoded frame.
            When ``product.crs`` is already geographic, the product is
            returned unchanged after a warning.

        Raises
        ------
        ValueError
            If the product contract is malformed, the kernel cannot be
            resolved, the destination grid cannot be resolved, the reference
            radar geometry is unavailable, or a persisted target conflicts
            with the request identity.

        Notes
        -----
        The geo2rdr lookup table is content-addressed under
        ``<work_dir>/cache/geo-lut/``; the readable namespace never enters
        the on-disk identity. The returned raster always lives on the
        resolved destination grid: product ``origin`` and ``multilook`` only
        fold the source sampling coordinates into the product layer frame
        through a single
        :meth:`~faninsar.processing.geocoding.coordinates.RadarCoordinates.to_layer_frame`
        mapping. Unwrapped phase geocodes with the bilinear
        (``continuous``) kernel, which is locally unreliable across
        connected-component boundaries where the field jumps by ``2*pi``.
        Export never geocodes: geocoding happens here, on demand.

        """
        name = _geocode_product_name(product)
        crs = getattr(product, "crs", None)
        if crs is not None:
            from pyproj import CRS

            parsed = CRS.from_user_input(crs)
            if parsed.is_geographic:
                logger.warning(
                    "geocode: product %r is already georeferenced in %s; "
                    "returning it unchanged",
                    name,
                    parsed.to_string(),
                )
                return product
            message = (
                f"geocode: product {name!r} carries projected CRS "
                f"{parsed.to_string()}; geocode expects a radar-domain "
                "product (crs=None)"
            )
            logger.error(message)
            raise ValueError(message)
        resolved_kind, asset_tag = _geocode_resolve_kind(product, kind)
        values = np.asarray(product.values)
        full_radar_shape = _geocode_int_pair(product, "full_radar_shape")
        looks = _geocode_int_pair(product, "multilook")
        origin = _geocode_int_pair(product, "origin")
        if any(value < 1 for value in looks):
            message = f"multilook must contain positive integers; got {looks}"
            logger.error(message)
            raise ValueError(message)
        if any(value < 0 for value in origin):
            message = f"origin must contain non-negative integers; got {origin}"
            logger.error(message)
            raise ValueError(message)
        from faninsar.processing.geocoding.product import layer_frame_shape

        layer_shape = layer_frame_shape(
            full_radar_shape, multilook=looks, origin=origin
        )
        if values.shape != layer_shape:
            message = (
                f"geocode product {name!r} values shape {values.shape} does not "
                "match the layer frame implied by full_radar_shape="
                f"{full_radar_shape}, origin={origin}, multilook={looks} "
                f"(expected {layer_shape})"
            )
            logger.error(message)
            raise ValueError(message)
        destination = self._geocode_destination_grid(grid, resolution_m, bounds)
        identity = _geocode_product_identity(
            name=name,
            kind=resolved_kind,
            asset_tag=asset_tag,
            grid=destination,
            full_radar_shape=full_radar_shape,
            looks=looks,
            origin=origin,
            values=values,
        )
        target_dir = self.config.work_dir / "geocode" / name
        if not overwrite:
            existing = _load_geocoded_product(target_dir, name, identity, destination)
            if existing is not None:
                logger.info("geocode: reusing persisted product %r", name)
                return existing
        geometry = self._geocode_geometry(full_radar_shape)
        from faninsar.processing.geocoding import Geocoder

        engine = Geocoder(
            geometry,
            destination,
            dem=self._geometry_dem(),
            full_radar_shape=full_radar_shape,
            device=self.config.device,
            cache_dir=self.config.work_dir / "cache" / "geo-lut",
            cache_key=f"stack_{self.reference}",
            storage_dir=self.config.work_dir / "cache" / "geo-lut" / "storage",
        )

        def correction(coords: Any) -> Any:
            """Map full-radar coordinates into the product layer frame."""
            return coords.to_layer_frame(origin=origin, looks=looks)

        raster = engine.apply(
            values,
            kind=resolved_kind,
            correction=correction,
            rows_per_chunk=_GEOCODE_ROWS_PER_CHUNK,
        )
        _commit_geocoded_product(
            target_dir, name, identity, raster, overwrite=overwrite
        )
        return raster

    def _geocode_destination_grid(
        self,
        grid: GridSpec | None,
        resolution_m: float | None,
        bounds: object | None,
    ) -> GridSpec:
        """Resolve the destination grid with :meth:`resolve_grid` precedence."""
        from faninsar.processing.geometry import GridSpec as GridSpecClass
        from faninsar.stack.grid import resolve_stack_grid

        effective_grid = self.config.grid if grid is None else grid
        effective_resolution = (
            self.config.resolution_m if resolution_m is None else float(resolution_m)
        )
        selected_roi = self.config.roi if bounds is None else bounds
        if selected_roi is None and effective_grid == "auto":
            selected_roi = self._selected_footprint_roi()
        if selected_roi is None and not isinstance(effective_grid, GridSpecClass):
            message = (
                "geocode requires an explicit grid, a configured "
                "StackConfig.grid, or bounds/roi/footprints to resolve the "
                "automatic grid"
            )
            logger.error(message)
            raise ValueError(message)
        return resolve_stack_grid(
            effective_grid,
            roi=selected_roi,
            resolution_m=effective_resolution,
            budget=self.config.resource_budget,
            explicit_roi=bounds is not None or self.config.roi is not None,
        )

    def _geocode_geometry(self, full_radar_shape: tuple[int, int]) -> object:
        """Return the persisted Reference radar geometry for ``geocode``."""
        geometry, grid_shape = self._reference_radar_geometry()
        if tuple(full_radar_shape) != tuple(grid_shape):
            message = (
                "geocode product declares full_radar_shape "
                f"{tuple(full_radar_shape)} but the Reference scene grid is "
                f"{tuple(grid_shape)}"
            )
            logger.error(message)
            raise ValueError(message)
        return geometry

    def _capture_dem_source(self, dem: object | None = None) -> None:
        """Remember a source recipe so reopen can rematerialize it."""
        selected = self.config.dem if dem is None else dem
        if isinstance(selected, SourceDEM):
            self._dem_source = selected

    def _has_stack_extent(self) -> bool:
        """Return whether the Stack has an ROI or selected footprints."""
        return self.config.roi is not None or self._selected_footprint_roi() is not None

    def _stack_dem_raster_path(self) -> Path:
        """Return the durable raster path owned by this Stack."""
        return self.config.work_dir / "dem" / "stack.tif"

    def _stack_dem_sidecar_path(self) -> Path:
        """Return the sidecar that records source and raster identity."""
        return self.config.work_dir / "dem" / "stack.json"

    def _stack_wgs84_raster_path(self) -> Path:
        """Return the geographic DEM used by rdr2geo."""
        return self.config.work_dir / "dem" / "stack_wgs84.tif"

    def _geometry_dem(self) -> RasterDEM | object | None:
        """Return the DEM passed to radar geometry.

        A native EPSG:4326 crop is preferred when Stack materialized one from
        the source tiles.  Otherwise the public product raster is returned and
        geometry backends build a private lon/lat view.
        """
        if self._dem_wgs84 is not None:
            return self._dem_wgs84
        return self.config.dem

    def _source_record(self) -> dict[str, object] | None:
        """Return the serializable source recipe, if one was captured."""
        source = self._dem_source
        if source is None:
            return None
        return {
            "product": source.product,
            "provider": source.provider,
            "cache_dir": None if source.cache_dir is None else str(source.cache_dir),
        }

    def _persist_raster_dem(self, raster: RasterDEM) -> Path:
        """Write the live raster when this Stack owns a source recipe."""
        path = self._stack_dem_raster_path()
        sidecar = self._stack_dem_sidecar_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **(self._source_record() or {"kind": "raster"}),
            "identity": raster.identity,
            "vertical_datum": raster.vertical_datum,
            "path": str(path),
        }
        if path.exists():
            try:
                existing = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
            if existing.get("identity") != raster.identity:
                path.unlink()
        if not path.is_file():
            raster.save(path)
        self._persist_wgs84_dem()
        if self._dem_wgs84 is not None and self._dem_wgs84 is not raster:
            record["wgs84_path"] = str(self._stack_wgs84_raster_path())
            record["wgs84_identity"] = self._dem_wgs84.identity
        sidecar.write_text(json.dumps(record, indent=2) + chr(10), encoding="utf-8")
        return path

    def _persist_wgs84_dem(self) -> None:
        """Write the geographic crop when it is distinct from the product DEM."""
        wgs84 = self._dem_wgs84
        if wgs84 is None or wgs84 is self.config.dem:
            return
        path = self._stack_wgs84_raster_path()
        sidecar = self._stack_dem_sidecar_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        source_path = getattr(wgs84, "path", None)
        if (
            path.is_file()
            and source_path is not None
            and Path(source_path).resolve() == path.resolve()
        ):
            return
        if path.exists():
            try:
                existing = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
            if existing.get("wgs84_identity") == wgs84.identity and path.is_file():
                return
            path.unlink()
        wgs84.save(path)

    def _load_persisted_raster_dem(self) -> RasterDEM | None:
        """Reload a previously materialized Stack raster, if it still matches."""
        path = self._stack_dem_raster_path()
        sidecar = self._stack_dem_sidecar_path()
        if not path.is_file():
            return None
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = {}
        source = self._source_record()
        if source is not None and (
            recorded.get("product") != source["product"]
            or recorded.get("provider") != source["provider"]
        ):
            return None
        datum = recorded.get("vertical_datum") or "ellipsoidal"
        if not isinstance(datum, str):
            datum = "ellipsoidal"
        return RasterDEM(path=path, vertical_datum=datum)

    def _load_persisted_wgs84_dem(self) -> RasterDEM | None:
        """Reload the geographic crop when the source recipe still matches.

        Content identity is not compared.  RasterDEM identity includes
        path provenance, so a saved crop never hashes to the in-memory
        identity recorded at materialization.
        """
        path = self._stack_wgs84_raster_path()
        sidecar = self._stack_dem_sidecar_path()
        if not path.is_file():
            return None
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = {}
        source = self._source_record()
        if source is not None and (
            recorded.get("product") != source["product"]
            or recorded.get("provider") != source["provider"]
        ):
            return None
        datum = recorded.get("vertical_datum") or "ellipsoidal"
        if not isinstance(datum, str):
            datum = "ellipsoidal"
        raster = RasterDEM(path=path, vertical_datum=datum)
        from faninsar.processing.geometry.dem_view import is_geodetic_crs

        if not is_geodetic_crs(raster.crs):
            logger.warning(
                "Persisted geographic DEM %s is not lon/lat (crs=%s); ignoring",
                path,
                raster.crs,
            )
            return None
        return raster

    def _align_raster_dem(self, raster: RasterDEM) -> RasterDEM:
        """Regrid a raster onto the Stack grid when one exists."""
        if self.config.grid == "auto" and not self._has_stack_extent():
            self.config.dem = raster
            return raster
        from faninsar.processing.geometry import api as dem_api

        if raster.grid != self.grid or raster.vertical_datum != "ellipsoidal":
            dem_api.ensure_dem_covers_grid(raster, self.grid)
            raster = raster.to_raster(
                self.grid,
                vertical_datum="ellipsoidal",
                budget=self.config.resource_budget,
            )
        self.config.dem = raster
        self._ensure_wgs84_dem(raster)
        if self._dem_source is not None:
            self._persist_raster_dem(raster)
        return raster

    def _ensure_wgs84_dem(self, raster: RasterDEM) -> None:
        """Materialize a native-spacing geographic crop for rdr2geo."""
        from faninsar.processing.geometry.dem_view import (
            geodetic_crop_grid,
            is_geodetic_crs,
            native_spacing_deg,
        )

        if is_geodetic_crs(raster.crs):
            self._dem_wgs84 = raster
            return
        loaded = self._load_persisted_wgs84_dem()
        if loaded is not None:
            self._dem_wgs84 = loaded
            logger.info("Reusing geographic DEM %s", self._stack_wgs84_raster_path())
            return
        source = self._dem_source
        if source is None:
            self._dem_wgs84 = None
            return
        cache_dir = self.config.dem_cache_dir or (self.config.work_dir / "dem-cache")
        if source.cache_dir is None:
            source.cache_dir = cache_dir
        product = raster.provenance.get("product") or source.product
        grid = geodetic_crop_grid(self.grid, native_spacing_deg(product))
        logger.info("Materializing geographic DEM from source tiles onto %s", grid.crs)
        self._dem_wgs84 = source.to_raster(
            grid,
            vertical_datum="ellipsoidal",
            budget=self.config.resource_budget,
        )

    def _ensure_raster_dem(self) -> RasterDEM | object | None:
        """Make the live Stack DEM a raster or leave a constant sampler.

        Source recipes are materialized onto the authoritative grid and
        persisted under ``work_dir/dem``.  ``ConstantDEM`` stays a sampler so
        tests and flat-geometry campaigns do not allocate a scene-sized
        raster.  Pair production may still promote a constant when it has an
        extent.
        """
        self._capture_dem_source()
        current = self.config.dem
        if type(current).__name__ == "ConstantDEM":
            return current
        if isinstance(current, RasterDEM):
            return self._align_raster_dem(current)
        loaded = self._load_persisted_raster_dem()
        if loaded is not None:
            return self._align_raster_dem(loaded)
        if (
            current is None
            and self.config.grid == "auto"
            and not self._has_stack_extent()
        ):
            return None
        raster = self.materialize_dem(current)
        self.config.dem = raster
        self._ensure_wgs84_dem(raster)
        if self._dem_source is not None or isinstance(current, SourceDEM):
            self._capture_dem_source(current)
            self._persist_raster_dem(raster)
        return raster

    @classmethod
    def _search_profile(cls) -> MissionSearchProfile:
        """Return the mission profile used by :meth:`search`.

        Concrete remote-capable mission adapters override this protected
        method.  Keeping the default closed prevents a bare mission-neutral
        ``Stack`` from guessing a provider or mission.
        """
        message = f"{cls.__name__} does not provide a remote search profile"
        logger.error(message)
        raise NotImplementedError(message)

    @classmethod
    def search(
        cls,
        spatial: Points | BoundingBox | Polygons,
        *,
        datetime_range: tuple[datetime, datetime] | None = None,
        flight_direction: Literal["ASCENDING", "DESCENDING"] | None = None,
        filters: Mapping[str, object] | None = None,
        auth_profile: str | None = None,
        limit: int = 100,
        budget: RemoteResourceBudget | None = None,
    ) -> list[CatalogItem]:
        """Search a mission catalog through the provider-neutral remote API.

        Parameters
        ----------
        spatial : Points, BoundingBox, or Polygons
            FanInSAR spatial query.
        datetime_range : tuple of datetime, optional
            Inclusive acquisition interval.
        flight_direction : {"ASCENDING", "DESCENDING"}, optional
            Common orbit direction filter.
        filters : mapping, optional
            Advanced provider filters accepted by the mission profile.
        auth_profile : str, optional
            Registered authentication and delivery profile.
        limit : int, default=100
            Candidate limit after complete catalog traversal and mission
            normalization.
        budget : RemoteResourceBudget, optional
            Bound on the remote operation.

        Returns
        -------
        list of CatalogItem
            Mission-normalized selected catalog records.

        """
        profile = cls._search_profile()
        if not isinstance(profile, MissionSearchProfile):
            message = "mission search profile must be MissionSearchProfile"
            logger.error(message)
            raise TypeError(message)
        merged_filters = merge_profile_filters(profile.default_filters, filters)
        from faninsar.remote import search as remote_search

        items = remote_search(
            spatial,
            catalog=profile.catalog,
            collections=profile.collections or None,
            datetime_range=datetime_range,
            flight_direction=flight_direction,
            filters=merged_filters,
            auth_profile=auth_profile,
            limit=limit,
            budget=budget,
        )
        normalized = profile.normalize(items)
        request = SelectionRequest(
            limit=limit,
            datetime_range=datetime_range,
            flight_direction=flight_direction,
            filters=merged_filters,
        )
        return profile.select(normalized, request)

    def _prepare_mission(self) -> MissionPreparedInputs:
        """Return mission-owned local inputs for shared preparation.

        Local-only stacks have no additional orbit or calibration inputs. A
        mission adapter may override this hook after constructing a valid
        local Stack; the result is validated before shared state is published.
        """
        selected = self.config.extra.get("selected_footprints", ())
        if isinstance(selected, (str, bytes)):
            selected = (selected,)
        return MissionPreparedInputs(selected_footprints=tuple(selected))

    def _validate_mission_prepared(
        self,
        prepared: MissionPreparedInputs,
    ) -> MissionPreparedInputs:
        """Validate one immutable mission preparation result."""
        if not isinstance(prepared, MissionPreparedInputs):
            message = "_prepare_mission must return MissionPreparedInputs"
            logger.error(message)
            raise TypeError(message)
        paths = (
            *prepared.orbit_paths_by_acquisition.values(),
            *prepared.calibration_paths,
        )
        for path in paths:
            if not Path(path).is_file():
                message = f"mission preparation path is not a file: {path}"
                logger.error(message)
                raise FileNotFoundError(message)
        return prepared

    def prepare(self) -> Self:
        """Prepare shared Stack inputs and save the Stack identity.

        Mission preparation resolves local orbit and calibration bindings
        (downloading missing orbits is allowed here, never on reopen), the
        output grid is resolved and recorded, and the DEM recipe is
        materialized onto it. The identity is then saved as ``stack.json``;
        reopen goes through :meth:`Stack.open`, which never touches the
        network.
        """
        self._prepared = False
        self._mission_prepared = None
        try:
            prepared = self._validate_mission_prepared(self._prepare_mission())
            self._mission_prepared = prepared
            if prepared.selected_footprints:
                self.config.extra["selected_footprints"] = prepared.selected_footprints

            self.prepare_scenes()

            if (
                self.config.grid != "auto"
                or self.config.roi is not None
                or prepared.selected_footprints
            ):
                # Persist the resolved grid as the science value so a reopened
                # Stack resolves the same geometry without the footprints.
                self.config.grid = self.resolve_grid()
            self._ensure_raster_dem()
            from faninsar.stack.identity import save_stack_identity

            save_stack_identity(self)
        except Exception:
            self._prepared = False
            self._mission_prepared = None
            raise
        return self

    @classmethod
    def open(cls, work_dir: str | Path, **runtime: Any) -> Stack:
        """Reopen a Stack from ``stack.json`` in an existing work directory.

        The method performs no remote search. Local inputs are checked by
        stat (the recorded size must match); runtime values (device, dask
        client, activation token and binding, provider callback) are supplied
        as keyword arguments. Committed products are reused; a stage resumes
        from the first Pair without one.

        Parameters
        ----------
        work_dir : str or pathlib.Path
            Work directory holding ``stack.json``.
        **runtime : Any
            Runtime-only :class:`StackConfig` values
            (``device``, ``invert_device``, ``executor``, ``n_jobs``,
            ``gpu_memory_reclaim``, ``resource_budget``,
            ``activation_binding``, ``activation_token``,
            ``activation_authority_root``).

        Returns
        -------
        Stack
            The reopened Stack with its science configuration restored and
            every committed product discoverable through the views.

        Raises
        ------
        InvalidProcessingStateError
            If ``stack.json`` is missing or is not a FanInSAR Stack identity
            (rebuild the work directory), a local input is missing or its
            size changed, the recorded science configuration cannot be
            restored, or an unknown runtime keyword is supplied.

        """
        from faninsar.stack.catalog import SceneCatalog
        from faninsar.stack.identity import (
            RUNTIME_FIELDS,
            decode_value,
            read_stack_identity,
        )

        root = Path(work_dir)
        document = read_stack_identity(root)
        mission = str(document["mission"])
        stack_cls = cls if mission == cls.__name__ else _mission_class(mission)
        unknown = set(runtime) - RUNTIME_FIELDS - {"scene_provider"}
        if unknown:
            message = (
                "Stack.open accepts runtime values only; science options are "
                "fixed by stack.json: " + ", ".join(sorted(unknown))
            )
            logger.error(message)
            raise TypeError(message)

        sources: dict[str, list[Path]] = {}
        for date, entries in document["inputs"]["sources"].items():
            sources[date] = [_check_input(entry) for entry in entries]
        orbits = {
            date: _check_input(entry)
            for date, entry in document["inputs"].get("orbits", {}).items()
        }
        calibrations = [
            _check_input(entry) for entry in document["inputs"].get("calibrations", [])
        ]

        science = {
            name: decode_value(value) for name, value in document["science"].items()
        }
        science["work_dir"] = root
        science["activation_mode"] = document["activation_mode"]
        science.update(
            {name: value for name, value in runtime.items() if name != "scene_provider"}
        )
        extra = {
            key: decode_value(value)
            for key, value in document.get("science_extra", {}).items()
        }
        if orbits:
            extra["orbit_paths"] = orbits
        if calibrations:
            extra["calibration_paths"] = calibrations
        science["extra"] = extra
        config = StackConfig(**science)

        from faninsar import Pairs

        stack = stack_cls(
            catalog=SceneCatalog.from_paths(
                [path for date in sorted(sources) for path in sources[date]]
            ),
            config=config,
            pairs=Pairs.from_names(document["pairs"]),
            misreg_pairs=Pairs.from_names(document["misreg_pairs"]),
            reference=str(document["reference"]),
        )
        stack.scene_provider = (
            runtime.get("scene_provider") or stack_cls._default_scene_provider()
        )
        stack.prepare_scenes()
        stack._capture_dem_source()
        stack._ensure_raster_dem()
        stack._prepared = True
        stack._refresh_network()
        return stack

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
        coreg_mode: CoregMode = "nesd",
        flatten_stage: FlattenStage = "coregistration",
        coregistration_grid: CoregistrationGrid = "radar",
        multilook: tuple[int, int] = (5, 2),
        goldstein_alpha: float = 0.5,
        esd_coherence_threshold: float = 0.85,
        ampcor_snr_threshold: float = 6.0,
        swaths: tuple[str, ...] | None = None,
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
            esd_coherence_threshold=esd_coherence_threshold,
            ampcor_snr_threshold=ampcor_snr_threshold,
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

    @classmethod
    def _default_scene_provider(cls) -> Any:
        """Return the mission's default scene provider, if it defines one."""
        return None

    @_reclaim_after_stage
    def prepare_scenes(self) -> Self:
        """Create work directories and validate the Reference acquisition."""
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        # Product roots appear as their first product is committed.
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

    def _ensure_precise_orbits(self) -> None:
        """Bind covering precise orbits before geometry stages.

        The base Stack has no orbit network dependency. Mission adapters may
        override this hook so the first precise-orbit stage can download
        covering files after local SAFE inputs already exist.
        """
        return

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
        aff_a, aff_b, aff_c, aff_d, aff_e, aff_f = (
            float(value) for value in tuple(grid.transform)[:6]
        )
        az_looks, rg_looks = (int(looks[0]), int(looks[1]))
        transform = Affine(
            aff_a * rg_looks,
            aff_b,
            aff_c,
            aff_d,
            aff_e * az_looks,
            aff_f,
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
            from faninsar.processing.masking.radar_projection import (
                radar_projection_cache_key,
            )

            geometry, full_shape = self._reference_radar_geometry()
            lut = None

            projection_key = radar_projection_cache_key(
                source_mask_identity=materialized.identity,
                master_grid=target,
                target_grid={
                    "shape": tuple(int(value) for value in full_shape),
                    "multilook": looks,
                },
                reference_scene=self.reference,
                lut=lut,
                geometry=geometry,
                dem=self._geometry_dem(),
            )
            if projection_key not in self._radar_projected_masks:
                self._radar_projected_masks[projection_key] = project_mask_to_radar(
                    materialized,
                    full_radar_shape=tuple(int(value) for value in full_shape),
                    lut=lut,
                    geometry=geometry,
                    mask_transform=target.transform,
                    dem=self._geometry_dem(),
                    device=self.config.device,
                    multilook=looks,
                    master_grid=target,
                    reference_scene=self.reference,
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
        return {
            "swaths": cfg.swaths,
            "bursts": cfg.bursts,
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

    def _reclaim_accelerator(self, kind: str) -> None:
        """Evaluate gpu_memory_reclaim on the CUDA-owning process."""
        from faninsar.processing.runtime.dask_gpu import run_reclaim_checkpoint
        from faninsar.processing.runtime.device import reclaim_checkpoint

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
        # One live DEM on the Stack grid. Source recipes are materialized
        # by _ensure_raster_dem; a constant sampler is promoted only when
        # this production callback has a geographic extent.
        self._ensure_raster_dem()
        has_extent = self._has_stack_extent()
        if type(self.config.dem).__name__ == "ConstantDEM" and has_extent:
            self.config.dem = self.materialize_dem(self.config.dem)
        if self.config.dem is not None:
            options.setdefault("dem", self._geometry_dem())
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
        swaths = None if self.config.swaths is None else list(self.config.swaths)
        serialized_bursts: object
        if bursts is None:
            serialized_bursts = None
        elif isinstance(bursts, dict):
            serialized_bursts = {
                str(swath): (
                    indices
                    if isinstance(indices, str)
                    else [int(index) for index in indices]
                )
                for swath, indices in sorted(bursts.items())
            }
        else:
            serialized_bursts = semantic_value(bursts)
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
            "esd_coherence_threshold": self.config.esd_coherence_threshold,
            "ampcor_snr_threshold": self.config.ampcor_snr_threshold,
            "multilook": list(self.config.multilook),
            "goldstein_alpha": self.config.goldstein_alpha,
            "executor": self.config.executor,
            "device": self.config.device,
            "invert_device": self.config.invert_device,
            "control_spacing": self.config.control_spacing,
            "n_jobs": self.config.n_jobs,
            "swaths": swaths,
            "bursts": serialized_bursts,
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
            "esd_coherence_threshold": self.config.esd_coherence_threshold,
            "ampcor_snr_threshold": self.config.ampcor_snr_threshold,
            "swaths": swaths,
            "bursts": serialized_bursts,
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

    def _misreg_dir(self) -> Path:
        """Return the misreg product directory."""
        return self.config.work_dir / "misreg"

    def _persist_misreg(self) -> None:
        """Commit arcs, constants, and the NESD skip outcome as one product."""
        from faninsar.stack.misreg import write_misreg

        write_misreg(
            self._misreg_dir(),
            arcs=list(self.arcs),
            date_misreg=self.date_misreg,
            skipped_no_windows=self._nesd_skipped_no_windows,
        )

    def _restore_misreg(self) -> bool:
        """Load the persisted misreg product into the Stack, if present."""
        from faninsar.stack.misreg import read_misreg

        record = read_misreg(self._misreg_dir())
        if record is None:
            return False
        self.arcs = list(record.arcs)
        self.date_misreg = record.date_misreg
        self._nesd_skipped_no_windows = record.skipped_no_windows
        return True

    @property
    def _misreg_write_id(self) -> str | None:
        """Return the misreg product write id NESD secondaries bind."""
        if self.config.coreg_mode != "nesd":
            return None
        from faninsar.stack.misreg import read_misreg

        record = read_misreg(self._misreg_dir())
        return None if record is None else record.write_id

    def _zero_date_misreg(self, *, reason: str) -> DateMisreg:
        """Return geometry-zero constants for every catalog date."""
        dates = list(self.catalog.dates)
        return DateMisreg(
            reference=self.reference,
            azimuth_px=dict.fromkeys(dates, 0.0),
            range_px=dict.fromkeys(dates, 0.0),
            metadata={"skipped": reason, "n_arcs": 0},
        )

    def _apply_misreg_constants(self, date_id: str) -> tuple[float, float]:
        """Return apply-time az/rg constants and reject omitted NESD axes."""
        if self.date_misreg is None:
            return 0.0, 0.0
        if date_id not in self.date_misreg.azimuth_px:
            reject_invalid_state(
                f"azimuth misregistration is omitted for {date_id}; "
                "apply cannot substitute a numeric zero"
            )
        if date_id not in self.date_misreg.range_px:
            reject_invalid_state(
                f"range misregistration is omitted for {date_id}; "
                "apply cannot substitute a numeric zero"
            )
        return (
            float(self.date_misreg.azimuth_px[date_id]),
            float(self.date_misreg.range_px[date_id]),
        )

    def _requires_inverted_nesd_constants(self) -> bool:
        """Return whether apply must have inverted NESD constants.

        Sentinel-1 overrides this to ``True`` in ``nesd`` mode. Other missions
        still coregister through the pair engine without a prior invert.
        """
        return False

    def _reject_nesd_apply_without_constants(self) -> None:
        """Fail closed when nesd apply has no inverted constants and did not skip."""
        if (
            self._requires_inverted_nesd_constants()
            and self.date_misreg is None
            and not self._nesd_skipped_no_windows
        ):
            reject_invalid_state(
                "coreg_mode='nesd' requires date_misreg from invert_misreg "
                "before coregister_scenes"
            )

    def _orbit_path_for(self, date_id: str) -> Path | None:
        """Return the extra orbit path bound to ``date_id``, if any."""
        orbit_paths = self.config.extra.get("orbit_paths")
        if not isinstance(orbit_paths, dict):
            return None
        value = orbit_paths.get(date_id)
        return None if value is None else Path(value)

    def _measure_pair_misreg(self, primary: str, secondary: str) -> MisregArc:
        """Measure one misreg pair through the mission scene provider.

        Sentinel-1 overrides this hook with
        :func:`~faninsar.missions.s1.measure.measure_s1_pair_misreg`.
        """
        out = self.config.work_dir / "misreg" / "measure" / f"{primary}_{secondary}"
        state = self._produce_pair(
            self.catalog.paths_for(primary),
            self.catalog.paths_for(secondary),
            output_dir=out,
            multilook=self.config.multilook,
            goldstein_alpha=self.config.goldstein_alpha,
            esd_enabled=True,
            amplitude_refinement_enabled=True,
            unwrap=False,
            overwrite=True,
            **self._burst_kwargs(),
        )
        azimuth = _optional_shift(getattr(state, "esd_azimuth_shift_px", None))
        range_shift = _optional_shift(getattr(state, "amplitude_residual_rg_px", None))
        n_valid = int(getattr(state, "n_valid", 0) or 0)
        if n_valid <= 0:
            n_valid = int(azimuth is not None or range_shift is not None)
        return MisregArc(
            primary=primary,
            secondary=secondary,
            azimuth_shift_px=azimuth,
            range_shift_px=range_shift,
            azimuth_sigma_px=0.05 if azimuth is not None else 1.0,
            range_sigma_px=0.1 if range_shift is not None else 1.0,
            method="overlap",
            n_valid=n_valid,
        )

    def _prepare_reference_scene(self, *, overwrite: bool) -> object | None:
        """Compute the Reference scene product once for this run.

        Missions that split Reference preparation from secondary resampling
        (Sentinel-1) return a Reference context reused by every
        :meth:`_coregister_one_scene` call. The default provider path
        publishes the Reference inside its first pair call and returns
        ``None``.
        """
        del overwrite
        return None

    def _coregister_one_scene(
        self,
        date_id: str,
        *,
        output_dir: Path,
        misreg_az_px: float,
        misreg_rg_px: float,
        overwrite: bool,
        reference_context: object | None = None,
        inputs: Mapping[str, str] | None = None,
        properties: Mapping[str, object] | None = None,
    ) -> Any:
        """Resample one secondary date onto the Reference scene product.

        The default implementation calls the mission pair provider with
        ``scene_dir`` (the secondary product, ``coreg/<date>``) and
        ``reference_scene_dir`` (``coreg/<reference>``). The provider writes
        the Reference product once if it is missing, then the secondary
        product bound to it with ``inputs`` and ``properties``. Sentinel-1
        overrides this hook.
        """
        del reference_context
        pair_kwargs = dict(self._burst_kwargs())
        orbit_paths = self.config.extra.get("orbit_paths")
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
        return self._produce_pair(
            self.catalog.paths_for(self.reference),
            self.catalog.paths_for(date_id),
            output_dir=output_dir,
            multilook=self.config.multilook,
            goldstein_alpha=0.0,
            unwrap=False,
            overwrite=overwrite,
            misreg_az_px=misreg_az_px,
            misreg_rg_px=misreg_rg_px,
            scene_dir=output_dir,
            reference_scene_dir=self._scene_dir(self.reference),
            scene_inputs=dict(inputs or {}),
            scene_properties=dict(properties or {}),
            **pair_kwargs,
        )

    @_reclaim_after_stage
    def measure_misreg(
        self,
        *,
        pairs: Pairs | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Measure per-pair NESD misreg arcs (no full IFG products).

        ``coreg_mode="geometry"`` is a no-op. ``nesd`` measures each misreg
        pair through :meth:`_measure_pair_misreg`. A campaign whose every
        component is omitted warns and skips NESD so apply does not see
        ``date_misreg is None``.
        """
        self._ensure_prepared()
        if self.config.coreg_mode == "geometry":
            logger.info("measure_misreg skipped (coreg_mode=geometry)")
            return self
        self._ensure_precise_orbits()
        self._ensure_raster_dem()
        use_pairs = pairs or self.misreg_pairs
        if not overwrite and self._restore_misreg():
            logger.info("Loaded %s misreg arcs from the misreg product", len(self.arcs))
            return self

        arcs: list[MisregArc] = []
        for primary, secondary in _iter_pair_dates(use_pairs):
            if primary not in self.catalog.paths or secondary not in self.catalog.paths:
                continue
            arcs.append(self._measure_pair_misreg(primary, secondary))
        skipped = not arcs or all(
            arc.azimuth_shift_px is None and arc.range_shift_px is None for arc in arcs
        )
        if skipped:
            logger.warning(
                "No burst-overlap windows in the resolved NESD selection; "
                "skipping NESD (geometry constants)"
            )
        self._nesd_skipped_no_windows = skipped
        self.arcs = arcs
        self.date_misreg = None
        self._persist_misreg()
        logger.info("Measured %s misreg arcs (method=overlap)", len(arcs))
        return self

    @_reclaim_after_stage
    def invert_misreg(
        self,
        *,
        reference: str | None = None,
        min_n_valid: int = 0,
        max_sigma_px: float = 1e3,
    ) -> Self:
        """Invert optional az/rg arcs to per-date constants (Reference = 0).

        ``coreg_mode="geometry"`` is a no-op. A measure campaign that warned
        and skipped NESD writes geometry zeros for every date.
        """
        self._ensure_prepared()
        if self.config.coreg_mode == "geometry":
            logger.info("invert_misreg skipped (coreg_mode=%s)", self.config.coreg_mode)
            return self
        _ = min_n_valid, max_sigma_px
        if not self.arcs and not self._nesd_skipped_no_windows:
            self.measure_misreg()
        if self._nesd_skipped_no_windows:
            self.date_misreg = self._zero_date_misreg(reason="no_overlap_windows")
            self._persist_misreg()
            return self
        ref = reference or self.reference
        try:
            self.date_misreg = invert_optional_pair_misregistration(
                self.arcs,
                reference=ref,
                dates=list(self.catalog.dates),
            )
        except Exception:
            logger.exception("misreg network inversion failed; aborting Stack")
            raise
        self._persist_misreg()
        return self

    @_reclaim_after_stage
    def coregister_scenes(
        self,
        *,
        dates: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Coregister each non-Reference date onto the Reference scene.

        The Reference scene product (``coreg/<reference>``) is computed once;
        every secondary product (``coreg/<date>``) binds its ``write_id``. A
        persisted secondary is reused only while it is bound to the current
        Reference and its coregistration identity matches this request.

        This stage does **not** form interferograms and never measures NESD.
        ``coreg_mode="nesd"`` requires :attr:`date_misreg` unless measure
        warned and skipped because the selection had no overlap windows.
        ``geometry`` applies zero date constants.
        """
        self._ensure_prepared()
        self._ensure_precise_orbits()
        self._ensure_raster_dem()
        target_dates = (
            list(dates)
            if dates is not None
            else [d for d in self.catalog.dates if d != self.reference]
        )
        if self.reference in target_dates:
            reject_invalid_state(
                "coregister_scenes dates must not include the Reference; its "
                "scene product is computed once for the whole Stack"
            )
        orbit_paths = self.config.extra.get("orbit_paths")
        if orbit_paths is not None and not isinstance(orbit_paths, dict):
            reject_invalid_state(
                "Stack extra orbit_paths must be a date-to-path mapping"
            )
        reference_context = (
            self._prepare_reference_scene(overwrite=overwrite) if target_dates else None
        )
        for date_id in target_dates:
            out = self._scene_dir(date_id)
            misreg_az, misreg_rg = self._apply_misreg_constants(date_id)
            coreg_identity = self._coreg_resume_identity(
                date_id,
                misreg_az_px=misreg_az,
                misreg_rg_px=misreg_rg,
            )
            if not overwrite and self._reuse_scene(date_id, coreg_identity):
                continue
            self._reject_nesd_apply_without_constants()
            properties: dict[str, object] = {
                "faninsar:coreg_identity": coreg_identity,
                "faninsar:coreg_mode": self.config.coreg_mode,
                "faninsar:geometric_phase_removed": (
                    "topo" if self.config.dem is not None else "flat"
                ),
                "faninsar:mask_lineage": self._mask_lineage_record(),
            }
            inputs = self._scene_misreg_inputs()
            state = self._coregister_one_scene(
                date_id,
                output_dir=out,
                misreg_az_px=misreg_az,
                misreg_rg_px=misreg_rg,
                overwrite=True,
                reference_context=reference_context,
                inputs=inputs,
                properties=properties,
            )
            reference = self.reference_scene
            if reference is None:
                reject_invalid_state(
                    "scene provider did not publish the Reference scene product"
                )
            validate_reference_binding(reference, CoregisteredSceneStore.open(out))
            if state is not None and self.config.retain_pair_states:
                self.pair_states[f"{self.reference}_{date_id}"] = state
            logger.info("Coregistered %s → Reference %s", date_id, self.reference)
            del state
            gc.collect()
            self._reclaim_accelerator("persist")
        self._write_qualified_activation_record()
        return self

    def _scene_misreg_inputs(self) -> dict[str, str]:
        """Return the misreg product binding of a secondary scene, if any."""
        misreg = self._misreg_write_id
        return {} if misreg is None else {"misreg": misreg}

    def _reuse_scene(self, date_id: str, coreg_identity: str) -> bool:
        """Return whether the persisted secondary scene is current."""
        out = self._scene_dir(date_id)
        if not (out / "item.json").is_file():
            return False
        reference = self.reference_scene
        if reference is None:
            return False
        store = CoregisteredSceneStore.open(out)
        current = {"reference": reference.write_id, **self._scene_misreg_inputs()}
        if (
            store.date_id != date_id
            or store.reference_id != self.reference
            or store.domain != self.config.coregistration_grid
            or store.properties.get("faninsar:coreg_identity") != coreg_identity
            or store.stale_inputs(current)
        ):
            logger.info("Scene %s is stale; recoregistering", date_id)
            return False
        validate_reference_binding(reference, store)
        logger.info("Reusing coregistered scene %s", date_id)
        return True

    @_reclaim_after_stage
    def form_interferograms(
        self,
        *,
        pairs: Pairs | None = None,
        multilook: tuple[int, int] | list[tuple[int, int]] | None = None,
        coherence_window: tuple[int, int] | None = None,
        phase_filter: PhaseFilter | None = _DEFAULT_PHASE_FILTER,
        goldstein_alpha: float | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Form wrapped interferograms from persisted Reference-aligned scenes.

        Parameters
        ----------
        pairs : Pairs, optional
            Pair graph to form. ``None`` uses the Stack graph fixed at
            construction. Pair endpoints are acquisition dates; their order is
            canonicalized by :class:`~faninsar.core.pair.Pairs`.
        multilook : tuple[int, int] or list[tuple[int, int]], optional
            One or more true boxcar look factors in ``(azimuth, range)``
            pixels. ``None`` uses :attr:`StackConfig.multilook`, whose default
            is ``(5, 2)``. Each result is aligned to the common scene-grid
            origin. Scene-grid edge coverage and incomplete look support are
            represented by the validity mask rather than interpolation.
        coherence_window : tuple[int, int] or None, default=None
            ``None`` (the default) estimates coherence per multilook block
            from that block's own samples (pure block MLE). A tuple enables
            the P0038 two-stage sliding window instead.
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
        overwrite : bool, default=False
            Replace existing IFG products (``work_dir/ifg/ml_AxR/<pair>``).
            Without it a current product is reused and a stale or
            differently configured one is rejected.

        Returns
        -------
        Stack
            This Stack, with complete IFG product directories appended to its
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

        from faninsar.processing.runtime.resources import (
            ResourceAdmissionLedger,
            estimate_formation_resources,
            reserve_estimate,
        )

        # A subset of the plan forms only the requested Pairs; Pairs outside
        # the plan extend it (and rewrite stack.json) first.
        if pairs is not None:
            self._extend_plan(pairs)
            use_pairs = pairs
        else:
            use_pairs = self.pairs
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
        base_out = self.config.work_dir / "ifg"
        formation_ledger = (
            ResourceAdmissionLedger(self.config.resource_budget, self.config.work_dir)
            if self.config.resource_budget is not None
            else None
        )

        scenes = self.scenes
        for primary, secondary in _iter_pair_dates(use_pairs):
            if primary not in self.catalog.paths or secondary not in self.catalog.paths:
                continue
            for looks in looks_list:
                az_l, rg_l = looks
                sub = base_out / f"ml_{az_l}x{rg_l}" / f"{primary}_{secondary}"
                if primary not in scenes or secondary not in scenes:
                    reject_invalid_state(
                        f"scene product missing for {primary}_{secondary}; run "
                        "coregister_scenes first"
                    )
                primary_store = scenes[primary]
                secondary_store = scenes[secondary]
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
                if (sub / "item.json").exists() and not overwrite:
                    from faninsar.io.storage.ifg_store import (
                        InterferogramArtifactStore,
                    )

                    existing_store = InterferogramArtifactStore.open(sub)
                    observed = {
                        "pair": existing_store.pair,
                        "looks": existing_store.looks,
                        "domain": existing_store.domain,
                        "wavelength_m": existing_store.wavelength_m,
                        "grid_identity": existing_store.grid_identity,
                        "filter_name": existing_store.filter_name,
                        "filter_parameters": existing_store.filter_parameters,
                        "sources": existing_store.source_manifest_digests,
                        "flatten_stage": existing_store.flatten_stage,
                        "phase_screen_model": existing_store.phase_screen_model,
                        "phase_screen_digests": existing_store.phase_screen_digests,
                        "phase_screen_domain": existing_store.phase_screen_domain,
                        "phase_screen_grid_identity": (
                            existing_store.phase_screen_grid_identity
                        ),
                        "mask_plan_identity": existing_store.mask_plan_identity,
                        "mask_identity": existing_store.mask_identity,
                    }
                    expected = {
                        "pair": (primary, secondary),
                        "looks": looks,
                        "domain": primary_store.domain,
                        "wavelength_m": primary_store.wavelength_m,
                        "grid_identity": primary_store.grid_identity,
                        "filter_name": expected_filter_name,
                        "filter_parameters": expected_filter_parameters,
                        "sources": expected_sources,
                        "flatten_stage": flatten_stage,
                        "phase_screen_model": phase_screen_model,
                        "phase_screen_digests": phase_screen_digests,
                        "phase_screen_domain": primary_store.domain,
                        "phase_screen_grid_identity": primary_store.grid_identity,
                        "mask_plan_identity": mask_plan_identity,
                        "mask_identity": mask_identity,
                    }
                    mismatched = [
                        name
                        for name, expected_value in expected.items()
                        if observed[name] != expected_value
                    ]
                    if mismatched:
                        reject_invalid_state(
                            "persisted IFG artifact does not match current scene "
                            "lineage or processing parameters: "
                            f"{', '.join(sorted(mismatched))} differ"
                        )
                    existing_store.read()
                    self.ifg_dirs.append(sub)
                    continue
                if sub.exists() and not overwrite:
                    reject_invalid_state(
                        f"{sub} exists without item.json; it is not an IFG product "
                        "of this FanInSAR version (pass overwrite=True to replace it)"
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
                from faninsar.io.storage.ifg_store import write_ifg_artifact

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
                    crs=primary_store.crs,
                    transform=_scale_geo_artifact_transform(
                        primary_store.transform, looks
                    ),
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
                    replace_existing=overwrite,
                )
                self.ifg_dirs.append(sub)
        return self

    def _extend_plan(self, pairs: Pairs) -> None:
        """Add Pairs outside the plan to the planned Pair network.

        A subset of the plan never shrinks it; the identity is rewritten so
        the extended plan survives a reopen.
        """
        from faninsar import Pairs as PairsType

        current = set(_iter_pair_dates(self.pairs))
        requested = _iter_pair_dates(pairs)
        added = [pair for pair in requested if pair not in current]
        if not added:
            return
        merged = sorted(current | set(requested))
        self.pairs = PairsType.from_names([f"{a}_{b}" for a, b in merged])
        logger.info(
            "Extended the planned Pair network with %s", [f"{a}_{b}" for a, b in added]
        )
        from faninsar.stack.identity import save_stack_identity

        save_stack_identity(self)

    def _refresh_network(self) -> None:
        """Rebuild the Network index from the committed products.

        The index exists only when every planned Pair has a current IFG and a
        current unwrap product (bound to that IFG's ``write_id``), all from
        one unwrap method. Otherwise the index is cleared: a missing, stale,
        or mixed Pair blocks analysis instead of shrinking the network.
        """
        try:
            stores = self._pair_artifact_stores(looks=self.config.multilook)
        except InvalidProcessingStateError:
            self._clear_network()
            return
        if not stores or not all(store.has_unwrapped() for store in stores):
            self._clear_network()
            return
        unwraps = [store.unwrap_product() for store in stores]
        methods = {
            json.dumps(product.properties.get("faninsar:method"), sort_keys=True)
            for product in unwraps
        }
        if len(methods) != 1:
            logger.warning(
                "Stack unwrap products mix methods; rerun unwrap(overwrite=True)"
            )
            self._clear_network()
            return
        dimensions = self.config.extra
        frame = str(dimensions.get("frame_id", "stack"))
        swath = str(dimensions.get("swath", "merged"))
        channel = str(dimensions.get("channel", "merged"))
        polarization = str(dimensions.get("polarization", "merged"))
        products: list[NetworkProduct] = []
        for store, unwrap in zip(stores, unwraps, strict=True):
            primary, secondary = _canonical_pair_strings(store.pair)
            for kind, location, digest, lineage in (
                (
                    AssetKind.COMPLEX_INTERFEROGRAM,
                    store.root / "complex_ifg.zarr",
                    store.write_id,
                    (store.write_id,),
                ),
                (
                    AssetKind.UNWRAPPED_PHASE,
                    store.unwrap_root / "unwrapped_phase.zarr",
                    unwrap.write_id,
                    (store.write_id, unwrap.write_id),
                ),
            ):
                products.append(
                    NetworkProduct(
                        key=NetworkProductKey(
                            AcquisitionKey(
                                primary, frame, swath, channel, polarization
                            ),
                            AcquisitionKey(
                                secondary, frame, swath, channel, polarization
                            ),
                            kind,
                        ),
                        asset_location=str(location),
                        geometry_identity=store.grid_identity,
                        source_software="faninsar",
                        phase_convention=PhaseConvention.PRIMARY_MINUS_SECONDARY,
                        asset_transform=AssetTransform.for_convention(
                            kind, PhaseConvention.PRIMARY_MINUS_SECONDARY
                        ),
                        content_digest=digest,
                        lineage=lineage,
                    )
                )
        self.refresh_generation(_network_snapshot_id(stores), tuple(products))

    def _clear_network(self) -> None:
        """Drop the Network index; the view is unavailable until a refresh."""
        object.__setattr__(self, "_network_generation_id", None)
        object.__setattr__(self, "_network_product_index", None)
        self._network_view = None

    @property
    def network_generation_id(self) -> str | None:
        """Return the id of the product snapshot visible to analysis."""
        return self._network_generation_id

    @property
    def network_product_index(self) -> NetworkProductIndex | None:
        """Return the immutable Network product index, if refreshed."""
        return self._network_product_index

    @property
    def analysis_ready(self) -> bool:
        """Whether every planned Pair has current IFG and unwrap products."""
        return (
            self._network_generation_id is not None
            and self._network_product_index is not None
        )

    def refresh_generation(
        self,
        generation_id: str,
        products: (
            NetworkProductIndex | tuple[NetworkProduct, ...] | list[NetworkProduct]
        ),
    ) -> Self:
        """Replace the Network index and rebind an already obtained view."""
        NetworkContract.refresh_generation(self, generation_id, products)
        view = getattr(self, "_network_view", None)
        if view is not None:
            view._bind_stack(self)
        return self

    @_reclaim_after_stage
    def unwrap(
        self,
        unwrapper: SpatialUnwrapper = _DEFAULT_UNWRAPPER,
        *,
        overwrite: bool = False,
    ) -> Self:
        """Spatially unwrap every persisted interferogram in this Stack.

        ``Stack`` owns pair iteration, Dataset materialization, and one unwrap
        product per Pair (``work_dir/unwrap/ml_AxR/<pair>``, bound to the IFG
        ``write_id``).  ``unwrapper`` owns the numerical solve for one
        two-dimensional pair.  This method has no temporal/network options and
        does not perform temporal reconciliation or time-series inversion.

        Parameters
        ----------
        unwrapper : SpatialUnwrapper, default=Snaphu()
            Trusted runtime strategy for one pair.  The default is the
            production SNAPHU backend.  ``SpatialIRLS`` remains available as
            an explicit developer/experimental strategy.  The strategy
            receives Torch tensors in ``(azimuth, range)`` order.  Its result
            must be a same-shape, same-device :class:`SpatialUnwrapResult`
            whose valid output is finite and is a subset of the Dataset
            support.
        overwrite : bool, default=False
            Recompute every Pair. Without it a Pair whose unwrap product is
            bound to the current IFG with the same method and mask is reused,
            and a stale or differently configured product is rejected.

        Returns
        -------
        Stack
            This Stack after every Pair's unwrap product is committed and its
            Network index rebuilt from them.

        Raises
        ------
        ValueError
            If the Pair set, IFG Dataset, or unwrapper result violates the
            scientific shape, dtype, mask, or finite-value contract.
        NoValidSupportError, ResourceAdmissionError
            Propagated from the selected unwrapper before numerical work.
        UnwrapFailedError
            If an admitted strategy fails, returns ``converged=False``, or
            the product cannot be committed.  The failed result, when
            available, is attached as ``error.result`` and is never committed.
        InvalidProcessingStateError
            If an existing unwrap product is stale or was produced by another
            method or mask and ``overwrite`` is false.

        Notes
        -----
        Pairs are committed one at a time, so an interrupted call resumes from
        the first Pair without a current product. The Network index exists
        only when every planned Pair has one; a partial set never becomes
        visible to analysis. Normal Dataset reads do not recompute payload
        hashes. With an active PROPOSAL-0039 mask, the inverted mask is the
        caller-supplied mask of the PROPOSAL-0038 support rule: it is
        intersected with the persisted IFG mask before the strategy is
        invoked (finite phase and finite coherence remain the strategy's
        rule) and is converted to :attr:`StackConfig.device` exactly once.

        """
        import torch

        from faninsar.data.datasets.ifg import StackInterferogramDataset
        from faninsar.io.storage.ifg_store import write_unwrapped_artifact
        from faninsar.processing.runtime.device import parse_device
        from faninsar.processing.runtime.resources import (
            ResourceAdmissionError,
            ResourceAdmissionLedger,
            estimate_spatial_irls_resources,
            estimate_unwrap_decode_resources,
            reserve_estimate,
        )
        from faninsar.processing.unwrapping.common import (
            SpatialUnwrapper,
            SpatialUnwrapResult,
        )
        from faninsar.processing.unwrapping.errors import (
            NoValidSupportError,
            UnwrapFailedError,
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
        stores = self._pair_artifact_stores(looks=self.config.multilook)
        device = parse_device(self.config.device)
        method_name, method_parameters = _unwrapper_metadata(unwrapper)
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
        lineage = {
            "faninsar:mask_plan_identity": mask_plan_identity,
            "faninsar:mask_identity": mask_identity,
        }
        try:
            for pair_id, store in zip(pair_ids, stores, strict=True):
                if (store.unwrap_root / "item.json").is_file() and not overwrite:
                    existing = open_product(store.unwrap_root, kind=UNWRAP_KIND)
                    recorded = existing.properties
                    if (
                        existing.inputs.get("ifg") == store.write_id
                        and recorded.get("faninsar:method")
                        == {"name": method_name, "parameters": method_parameters}
                        and all(
                            recorded.get(key) == value for key, value in lineage.items()
                        )
                    ):
                        continue
                    reject_invalid_state(
                        f"unwrap product for {pair_id} is stale or was produced by "
                        "another unwrapper or mask; pass overwrite=True to recompute"
                    )
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
                    dataset = StackInterferogramDataset.from_product(store)
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
                # Compare against the materialized input tensor rather than the
                # requested device.  ``torch.device("cuda")`` and
                # ``torch.device("cuda:0")`` are equivalent execution targets,
                # but they are not equal values; tensors materialized from the
                # former carry the latter as their concrete device identity.
                if (
                    result.phase.device != wrapped_phase.device
                    or result.valid_mask.device != wrapped_phase.device
                ):
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
                write_unwrapped_artifact(
                    store.root,
                    unwrapped_phase=result.phase.detach().cpu().numpy(),
                    connected_components=result.component_labels.detach().cpu().numpy(),
                    method=method_name,
                    method_parameters=method_parameters,
                    ifg_manifest_digest=store.write_id,
                    extra_layers={
                        "valid_mask": result.valid_mask.detach().cpu().numpy(),
                        "reference_values": result.reference_values.detach()
                        .cpu()
                        .numpy(),
                    },
                    properties=lineage,
                    replace_existing=overwrite,
                )
        except (
            NoValidSupportError,
            ResourceAdmissionError,
            UnwrapFailedError,
            InvalidProcessingStateError,
        ):
            raise
        except Exception as error:
            logger.exception("Stack unwrap failed")
            message = "Stack unwrap failed"
            raise UnwrapFailedError(message, failed_result) from error
        finally:
            # A failed call may have committed some Pairs; the index is
            # rebuilt only from a complete set.
            self._clear_network()
        self._refresh_network()
        return self

    @_reclaim_after_stage
    def invert_timeseries(
        self,
        *,
        pair_phases: dict[str, np.ndarray] | None = None,
        device: str | None = None,
        multilook: tuple[int, int] | None = None,
        ionosphere_corrected: bool = False,
        pairs: Pairs | None = None,
        overwrite: bool = False,
    ) -> TimeSeries:
        """Invert the persisted unwrapped Pair network with SBAS.

        Parameters
        ----------
        pair_phases : dict[str, numpy.ndarray], optional
            In-memory Pair phases to invert directly; nothing is persisted.
        device : str, optional
            Solver device. Defaults to :attr:`StackConfig.invert_device`.
        multilook : tuple[int, int], optional
            Product view to consume. Defaults to the configured look factors.
        ionosphere_corrected : bool, default=False
            Consume the ionosphere-corrected unwrap of every Pair instead of
            the raw unwrap.
        pairs : Pairs, optional
            Pair subset to invert. Defaults to every planned Pair. The
            canonical time-series product always reflects the latest call.
        overwrite : bool, default=False
            Replace an existing time-series product bound to other inputs.

        Returns
        -------
        TimeSeries
            The result, committed as ``work_dir/timeseries`` whose Item
            records the ``write_id`` of every consumed unwrap product.

        Raises
        ------
        InvalidProcessingStateError
            If a planned Pair has no current (or corrected) unwrap, or the
            existing time-series product is bound to other inputs and
            ``overwrite`` is false.

        """
        from faninsar.timeseries.io import read_timeseries, write_timeseries_product
        from faninsar.timeseries.results import SpatialMetadata, invert_unwrapped_pairs

        solver_device = device or self.config.invert_device
        if pair_phases is not None:
            self.timeseries = invert_unwrapped_pairs(pair_phases, device=solver_device)
            return self.timeseries
        looks = multilook or self.config.multilook
        stores = self._pair_artifact_stores(looks=looks)
        if pairs is not None:
            selected = set(_iter_pair_dates(pairs))
            stores = [store for store in stores if store.pair in selected]
            if not stores:
                reject_invalid_state("requested Pair subset matches no IFG product")
        sources = _pair_phase_sources(
            self.config.work_dir, stores, ionosphere_corrected=ionosphere_corrected
        )
        inputs = {pair_id: source.write_id for pair_id, source in sources.items()}
        phase_source = "ion_corrected" if ionosphere_corrected else "unwrap"
        target = self.config.work_dir / "timeseries"
        complete_network = pairs is None
        if complete_network and (target / "item.json").is_file() and not overwrite:
            existing = open_product(target)
            if (
                existing.inputs == inputs
                and existing.properties.get("faninsar:phase_source") == phase_source
            ):
                self.timeseries = read_timeseries(target)
                return self.timeseries
            reject_invalid_state(
                "the time-series product is bound to other unwrap products; "
                "pass overwrite=True to recompute it"
            )
        wavelengths = {store.wavelength_m for store in stores}
        if len(wavelengths) != 1:
            reject_invalid_state("Stack IFG products use mixed wavelengths")
        result = invert_unwrapped_pairs(
            {
                pair_id: source.read("unwrapped_phase")
                for pair_id, source in sources.items()
            },
            device=solver_device,
            wavelength_m=next(iter(wavelengths)),
        )
        first = stores[0]
        result = replace(
            result,
            spatial=SpatialMetadata(
                domain=first.domain,
                shape=first.shape,
                grid_identity=first.grid_identity,
                crs=first.crs,
                transform=first.transform,
            ),
        )
        path = write_timeseries_product(
            result,
            target,
            inputs=inputs,
            properties={
                "faninsar:phase_source": phase_source,
                "faninsar:looks": [int(looks[0]), int(looks[1])],
            },
            overwrite=overwrite or (target / "item.json").is_file(),
        )
        self.timeseries = replace(result, canonical_path=path)
        if self._network_view is not None:
            self._network_view._timeseries = self.timeseries
        return self.timeseries

    def analyze_time_series(
        self,
        *,
        solver: str = "sbas",
        model: Any | None = None,
        **kwargs: Any,
    ) -> TimeSeries:
        """Analyze the Stack's Network products after unwrapping.

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
        TimeSeries
            Inverted per-date time series.

        Raises
        ------
        ValueError
            If analysis is requested before every planned Pair is unwrapped
            or an unsupported solver is selected.

        """
        del model
        if solver.lower() != "sbas":
            message = f"unsupported Stack time-series solver {solver!r}"
            logger.error(message)
            raise ValueError(message)
        if not self.analysis_ready:
            message = (
                "Stack Network analysis is unavailable before every planned Pair "
                "has a current interferogram and unwrap product"
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
    ) -> TimeSeries:
        """Run the Stack inversion after checking the product snapshot.

        Pure in-memory Network tests have no Stack config and keep the light
        inversion seam.
        """
        if not hasattr(self, "config") or not hasattr(self, "pairs"):
            return self.invert_timeseries(**kwargs)
        stores = self._pair_artifact_stores(
            looks=kwargs.get("multilook") or self.config.multilook
        )
        try:
            if _network_snapshot_id(stores) != generation_id:
                reject_invalid_state(
                    "Stack IFG or unwrap products changed after the Network index "
                    "was built; reopen the Stack"
                )
            return self.invert_timeseries(**kwargs)
        finally:
            for store in stores:
                store.close()

    def estimate_ionosphere(self, **kwargs: Any) -> list[Any]:
        """Estimate per-pair ionospheric screens (PROPOSAL-0036).

        See
        :func:`faninsar.stack.analysis.estimate_ionosphere`
        for the full parameter contract.

        The explicit ``ionosphere`` mask stage feeds the estimator's existing
        ``valid_mask`` seam unless the caller supplies one.
        """
        from faninsar.stack.analysis import estimate_ionosphere

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
        :func:`faninsar.stack.analysis.apply_ionosphere_correction`
        for the full parameter contract.
        """
        from faninsar.stack.analysis import apply_ionosphere_correction

        return apply_ionosphere_correction(self, **kwargs)

    def invert_ionosphere_dates(self, **kwargs: Any) -> Any:
        """Invert published pair ion screens into per-date screens.

        See
        :func:`faninsar.stack.analysis.invert_ionosphere_dates`
        for the full parameter contract.
        """
        from faninsar.stack.analysis import invert_ionosphere_dates

        return invert_ionosphere_dates(self, **kwargs)

    def _pair_artifact_stores(
        self,
        *,
        looks: tuple[int, int],
    ) -> list[InterferogramArtifactStore]:
        """Open the exact planned IFG product set of one multilook view.

        Raises
        ------
        InvalidProcessingStateError
            If a planned Pair has no IFG product, an unplanned one exists, or
            the products do not share one grid.

        """
        from faninsar.io.storage.ifg_store import InterferogramArtifactStore

        azimuth_looks, range_looks = (int(looks[0]), int(looks[1]))
        root = self.config.work_dir / "ifg" / f"ml_{azimuth_looks}x{range_looks}"
        expected_pairs = _iter_pair_dates(self.pairs)
        expected_ids = [
            f"{primary}_{secondary}" for primary, secondary in expected_pairs
        ]
        if not root.is_dir():
            reject_invalid_state(f"Stack IFG product view is missing: {root}")
        discovered_ids = sorted(path.parent.name for path in root.glob("*/item.json"))
        if sorted(expected_ids) != discovered_ids:
            reject_invalid_state(
                "Stack IFG product Pair set does not exactly match the planned "
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
    from faninsar.core.pair import Pair

    out: list[tuple[str, str]] = []
    for values in pairs.values:
        pair = Pair(values)
        out.append((pair.primary_string(), pair.secondary_string()))
    return out


def _canonical_pair_strings(values: tuple[str, str]) -> tuple[str, str]:
    """Normalize persisted pair roles through the FanInSAR :class:`Pair` type."""
    from faninsar.core.pair import Pair

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


def _unwrapper_metadata(unwrapper: object) -> tuple[str, dict[str, Any]]:
    """Return the persisted identity of a spatial unwrapper.

    The name is the class name; the parameters are its public attributes
    that have a strict-JSON form (dataclass configurations included), so a
    resumed unwrap reuses only products of the same method.
    """
    parameters: dict[str, Any] = {}
    for name, value in sorted(getattr(unwrapper, "__dict__", {}).items()):
        if name.startswith("_"):
            continue
        candidate = (
            asdict(value)
            if is_dataclass(value) and not isinstance(value, type)
            else value
        )
        try:
            parameters[name] = json.loads(json.dumps(candidate, allow_nan=False))
        except (TypeError, ValueError):
            continue
    return type(unwrapper).__name__, parameters


def _pair_phase_sources(
    work_dir: Path,
    stores: Sequence[InterferogramArtifactStore],
    *,
    ionosphere_corrected: bool,
) -> dict[str, Product]:
    """Return the current unwrap (or corrected unwrap) product of each Pair.

    Raises
    ------
    InvalidProcessingStateError
        If a Pair has no current unwrap, or its ionosphere correction is
        missing or bound to a replaced unwrap or ion product.

    """
    from faninsar.io.storage.ion_store import (
        ION_CORRECTED_NAME,
        IonosphereArtifactStore,
        ion_dir,
        open_ion_correction,
    )

    sources: dict[str, Product] = {}
    for store in stores:
        pair_id = f"{store.pair[0]}_{store.pair[1]}"
        if not store.has_unwrapped():
            reject_invalid_state(
                f"Pair {pair_id} has no unwrap product bound to its current IFG; "
                "run unwrap() first"
            )
        if not ionosphere_corrected:
            sources[pair_id] = store.unwrap_product()
            continue
        if not (store.unwrap_root / ION_CORRECTED_NAME / "item.json").is_file():
            reject_invalid_state(
                f"Pair {pair_id} has no ionosphere correction; run "
                "apply_ionosphere_correction() first"
            )
        corrected = open_ion_correction(store.root)
        looks = tuple(
            int(value) for value in corrected.properties["faninsar:ion_looks"]
        )
        ion = IonosphereArtifactStore.open(ion_dir(work_dir, pair_id, looks))
        if corrected.inputs.get("ion") != ion.write_id:
            reject_invalid_state(
                f"ionosphere correction of {pair_id} is bound to a replaced ion "
                "product; rerun apply_ionosphere_correction(overwrite=True)"
            )
        sources[pair_id] = corrected
    return sources


def _network_snapshot_id(stores: Sequence[InterferogramArtifactStore]) -> str:
    """Return the identity of one IFG-plus-unwrap product snapshot."""
    components = [
        *(f"ifg:{store.write_id}" for store in stores),
        *(f"unwrap:{store.unwrap_product().write_id}" for store in stores),
    ]
    return hashlib.sha256("|".join(sorted(components)).encode("utf-8")).hexdigest()


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


def _scale_geo_artifact_transform(
    transform: object | None, looks: tuple[int, int]
) -> tuple[float, ...] | None:
    """Scale one geo artifact transform by (azimuth, range) look factors.

    Block multilooking multiplies the pixel sizes by the look factors and
    keeps the grid origin, matching the scene-grid alignment convention of
    ``form_interferograms``.  Radar-domain artifacts carry no transform, so
    ``None`` passes through unchanged.  The stored convention is the
    six-coefficient affine order of the persisted manifest.
    """
    if transform is None:
        return None
    from affine import Affine

    aff_a, aff_b, aff_c, aff_d, aff_e, aff_f = (
        float(value) for value in tuple(transform)[:6]
    )
    az_looks, rg_looks = (int(looks[0]), int(looks[1]))
    scaled = Affine(
        aff_a * rg_looks,
        aff_b,
        aff_c,
        aff_d,
        aff_e * az_looks,
        aff_f,
    )
    return tuple(float(value) for value in scaled)[:6]


def _optional_shift(value: object) -> float | None:
    """Return a finite float observation, else ``None`` (F17).

    JSON ``null`` and non-finite sentinels stay omitted. A measured ``0.0``
    is preserved; it is never used to fill a missing component.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _check_input(entry: object) -> Path:
    """Stat-check one recorded Stack input and return its path.

    The recorded size must match; the file is never re-hashed.

    Raises
    ------
    InvalidProcessingStateError
        If the input is missing or its size changed.

    """
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        reject_invalid_state("stack.json input record is invalid")
    path = Path(str(entry["path"]))
    if not path.exists():
        message = (
            f"Stack input is missing: {path}; restore it or rebuild the work directory"
        )
        logger.error(message)
        reject_invalid_state(message)
    recorded = entry.get("size")
    if isinstance(recorded, int) and path.stat().st_size != recorded:
        message = (
            f"Stack input changed on disk: {path} recorded {recorded} bytes, "
            f"found {path.stat().st_size}"
        )
        logger.error(message)
        reject_invalid_state(message)
    return path


def _mission_class(mission: str) -> type[Stack]:
    """Return the Stack subclass recorded as the Stack mission.

    Raises
    ------
    InvalidProcessingStateError
        If the mission name does not name a known Stack subclass.

    """
    if mission == "Stack":
        return Stack
    if mission == "S1Stack":
        from faninsar.missions.s1 import S1Stack

        return S1Stack
    if mission == "NISARStack":
        from faninsar.missions.nisar import NISARStack

        return NISARStack
    message = (
        f"stack.json mission {mission!r} is not a known Stack subclass; rebuild "
        "this work directory with the FanInSAR version that created it"
    )
    logger.error(message)
    raise InvalidProcessingStateError(message)

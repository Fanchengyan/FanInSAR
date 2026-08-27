"""Mission-neutral Stack session (PROPOSAL-0017).

Orchestrates Reference-relative coregistration and pair products. Co-registration
and interferogram formation are separate stages: coreg caches per-date SLCs;
``form_interferograms`` only reads those products.
"""

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
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, Self, TypeVar

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
from faninsar.datasets.network import Network
from faninsar.logging import setup_logger
from faninsar.processing.coreg.misreg_network import (
    DateMisreg,
    MisregArc,
    invert_pair_misregistration,
)
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.stack.catalog import SceneCatalog
from faninsar.processing.stack.config import (
    ActivationMode,
    CoregMode,
    EsdMethod,
    StackConfig,
)
from faninsar.processing.stack.provider import SourceHandle
from faninsar.processing.stack.scene_store import (
    CoregisteredSceneStore,
    copy_reference_units,
    form_merged_scene_interferogram,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from faninsar._core.device import GpuMemoryReclaim
    from faninsar.core.acquisition import Acquisition
    from faninsar.core.pairs import Pairs
    from faninsar.processing.contracts.prepared_geometry import (
        ActivationToken,
        StackActivationBinding,
    )
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.merge.grid import GeoGridSpec
    from faninsar.processing.pipeline.production import (
        BurstSelection,
        CoregistrationGrid,
        ProductionPairState,
    )
    from faninsar.processing.stack.ifg_store import (
        InterferogramArtifactStore,
        UnwrappedArtifact,
    )
    from faninsar.processing.stack.provider import StackSceneProvider
    from faninsar.processing.stack.stack_generation import StackResultGeneration
    from faninsar.processing.timeseries.inversion import TimeSeriesResult
    from faninsar.processing.unwrap.quality import StackQualityCriteria
    from faninsar.processing.unwrap.stack import SpatialExecutor, StackUnwrapResult
    from faninsar.query import BoundingBox, Polygons

logger = setup_logger(__name__)

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
class Stack(Network):
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
    _network_generation_id: str | None = field(default=None, repr=False)
    _network_product_index: NetworkProductIndex | None = field(default=None, repr=False)

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
        from faninsar.processing.stack.s1 import S1Stack

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
        dem: DEMSampler | None = None,
        geo_grid: GeoGridSpec | None = None,
        roi: BoundingBox | Polygons | None = None,
        coreg_mode: CoregMode = "pair",
        coregistration_grid: CoregistrationGrid = "radar",
        multilook: tuple[int, int] = (2, 10),
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
            coregistration_grid=coregistration_grid,
            esd_method=esd_method,
            multilook=multilook,
            goldstein_alpha=goldstein_alpha,
            executor=executor,
            device=device,
            invert_device=invert_device,
            dem=dem,
            geo_grid=geo_grid,
            roi=roi,
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
        from faninsar.processing.stack.activation import LocalActivationAuthority

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
        from faninsar.processing.stack.activation import LocalActivationAuthority

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
            "roi": cfg.roi,
            "control_spacing": cfg.control_spacing,
            "n_jobs": cfg.n_jobs,
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
            from faninsar.processing.stack.provider import (
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
            "roi": roi_identity(self.config.roi),
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
        goldstein_alpha: float | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Form interferograms from persisted Reference-aligned scene artifacts.

        This method deliberately has no SAFE-path or pair-runner fallback.
        Missing, incomplete, mixed-domain, or multi-unit generations fail
        closed until the provider supplies a complete scene manifest.
        """
        self._ensure_prepared()
        self._require_qualified_activation_record()

        use_pairs = pairs or self.pairs
        looks_list = _normalize_multilook(multilook or self.config.multilook)
        alpha = (
            self.config.goldstein_alpha
            if goldstein_alpha is None
            else float(goldstein_alpha)
        )
        base_out = (
            Path(output_dir)
            if output_dir is not None
            else (self.config.work_dir / "ifg")
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
                expected_filter_name = "goldstein" if alpha > 0.0 else "none"
                expected_filter_parameters = (
                    {"alpha": alpha, "window": 32} if alpha > 0.0 else {}
                )
                if (sub / "manifest.json").exists():
                    from faninsar.processing.stack.ifg_store import (
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
                    device=self.config.device,
                    dask_client=self.dask_client,
                )
                from faninsar.processing.stack.ifg_store import write_ifg_artifact

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
                    complex_ifg=product.complex_ifg,
                    coherence=product.coherence,
                    wrapped_phase=product.wrapped_phase,
                    amplitude=product.amplitude,
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
        from faninsar.processing.stack.ifg_store import InterferogramArtifactStore

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
        seen_paths: set[Path] = set()
        actual_pairs: list[tuple[str, str]] = []
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
        generation_id = hashlib.sha256(
            "|".join(sorted(manifest_digests)).encode("utf-8")
        ).hexdigest()
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
        return self

    @_reclaim_after_stage
    def unwrap(
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
        from faninsar.processing.stack.ifg_store import write_unwrapped_artifact
        from faninsar.processing.unwrap.quality import (
            MetricDistribution,
            StackQualityCriteria,
            StackQualityReport,
        )
        from faninsar.processing.unwrap.stack import unwrap_stack

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
        del _products
        # Keep the IFG generation pinned for the complete analysis call.  The
        # solver reads the unwrap payloads before doing its numerical solve;
        # closing the stores immediately after those reads would allow a
        # concurrent collector to reclaim the generation while the solver is
        # still consuming the resulting arrays.  Pure in-memory Network tests
        # intentionally have no Stack config/catalog and retain their light
        # weight inversion seam.
        if not hasattr(self, "config") or not hasattr(self, "pairs"):
            return self.invert_timeseries(**kwargs)
        stores = self._pair_artifact_stores(
            looks=kwargs.get("multilook") or self.config.multilook,
            ifg_root=kwargs.get("ifg_root"),
        )
        try:
            observed_generation_id = hashlib.sha256(
                "|".join(sorted(store.manifest_digest for store in stores)).encode()
            ).hexdigest()
            if observed_generation_id != generation_id:
                reject_invalid_state(
                    "Stack IFG artifacts changed after Network generation refresh"
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
            return result
        finally:
            for store in stores:
                store.close()

    def estimate_ionosphere(self, **kwargs: Any) -> list[Any]:
        """Estimate per-pair ionospheric screens (PROPOSAL-0036).

        See
        :func:`faninsar.processing.stack.stack_api.estimate_ionosphere`
        for the full parameter contract.
        """
        from faninsar.processing.stack.stack_api import estimate_ionosphere

        return estimate_ionosphere(self, **kwargs)

    def apply_ionosphere_correction(self, **kwargs: Any) -> dict[str, Path]:
        """Subtract qualified ion screens from unwrapped pair phases.

        See
        :func:`faninsar.processing.stack.stack_api.apply_ionosphere_correction`
        for the full parameter contract.
        """
        from faninsar.processing.stack.stack_api import apply_ionosphere_correction

        return apply_ionosphere_correction(self, **kwargs)

    def invert_ionosphere_dates(self, **kwargs: Any) -> Any:
        """Invert published pair ion screens into per-date screens.

        See
        :func:`faninsar.processing.stack.stack_api.invert_ionosphere_dates`
        for the full parameter contract.
        """
        from faninsar.processing.stack.stack_api import invert_ionosphere_dates

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
        from faninsar.processing.stack.stack_generation import (
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
        from faninsar.processing.stack.stack_generation import open_stack_generation

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
        from faninsar.processing.stack.ifg_store import InterferogramArtifactStore

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

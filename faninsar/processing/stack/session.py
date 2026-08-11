"""Mission-neutral Stack session (PROPOSAL-0017).

Orchestrates master-centric coregistration and pair products. Co-registration
and interferogram formation are separate stages: coreg caches per-date SLCs;
``form_interferograms`` only reads those products.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

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
from faninsar.processing.stack.scene_store import (
    CoregisteredSceneStore,
    copy_reference_units,
    form_scene_interferograms,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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
    from faninsar.processing.timeseries.inversion import TimeSeriesResult

logger = setup_logger(__name__)


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
class Stack:
    """Multi-scene InSAR session with master-centric coregistration.

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
    master : str, optional
        Master date id. Default: earliest catalog date.
    acquisitions : Acquisition, optional
        Optional domain Acquisition index (informational).

    """

    catalog: SceneCatalog
    config: StackConfig
    pairs: Pairs
    misreg_pairs: Pairs
    master: str
    acquisitions: Acquisition | None = None
    arcs: list[MisregArc] = field(default_factory=list)
    date_misreg: DateMisreg | None = None
    coreg_paths: dict[str, Path] = field(default_factory=dict)
    pair_states: dict[str, ProductionPairState] = field(default_factory=dict)
    ifg_dirs: list[Path] = field(default_factory=list)
    timeseries: TimeSeriesResult | None = None
    _prepared: bool = False

    @classmethod
    def from_safes(
        cls,
        paths: Sequence[str | Path],
        *,
        work_dir: str | Path,
        pairs: Pairs | None = None,
        misreg_pairs: Pairs | None = None,
        master: str | None = None,
        dem: DEMSampler | None = None,
        geo_grid: GeoGridSpec | None = None,
        coreg_mode: CoregMode = "pair",
        coregistration_grid: CoregistrationGrid = "radar",
        multilook: tuple[int, int] = (2, 10),
        goldstein_alpha: float = 0.5,
        esd_method: EsdMethod = "auto",
        swaths: tuple[str, ...] = ("IW1",),
        bursts: BurstSelection | None = None,
        executor: str = "torch",
        device: str = "auto",
        invert_device: str = "cpu",
        pair_max_interval: int = 3,
        pair_max_days: int = 72,
        misreg_max_interval: int = 2,
        misreg_max_days: int = 36,
        activation_mode: ActivationMode,
        activation_binding: StackActivationBinding | None = None,
        activation_token: ActivationToken | None = None,
        activation_authority_root: str | Path | None = None,
        retain_pair_states: bool = False,
    ) -> Stack:
        """Construct a Stack from SAFE paths and optional pair graphs."""
        catalog = SceneCatalog.from_paths(list(paths))
        dates = list(catalog.dates)
        master_id = master or dates[0]
        if master_id not in catalog.paths:
            reject_invalid_state(f"master {master_id} not in catalog")
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
        )
        return cls(
            catalog=catalog,
            config=config,
            pairs=ifg_pairs,
            misreg_pairs=m_pairs,
            master=master_id,
            acquisitions=acq,
        )

    def prepare_scenes(self) -> Self:
        """Create work directories and validate catalog/master."""
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        (self.config.work_dir / "coreg").mkdir(exist_ok=True)
        (self.config.work_dir / "misreg").mkdir(exist_ok=True)
        (self.config.work_dir / "ifg").mkdir(exist_ok=True)
        (self.config.work_dir / "pairs").mkdir(exist_ok=True)
        if self.master not in self.catalog.paths:
            reject_invalid_state(f"master {self.master} missing from catalog")
        self._prepared = True
        logger.info(
            "Stack prepared master=%s n_scenes=%s n_pairs=%s n_misreg_pairs=%s mode=%s",
            self.master,
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
            "master": self.master,
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
            "coregistration_grid": cfg.coregistration_grid,
            "roi": cfg.roi,
            "control_spacing": cfg.control_spacing,
            "n_jobs": cfg.n_jobs,
        }

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

        from faninsar.processing.pipeline.production import run_pair

        arcs: list[MisregArc] = []
        for primary, secondary in _iter_pair_dates(use_pairs):
            if primary not in self.catalog.paths or secondary not in self.catalog.paths:
                continue
            out = self.config.work_dir / "misreg" / "measure" / f"{primary}_{secondary}"
            esd_on = method != "auto" or True
            # Measure-only: coreg + ESD/Ampcor; no unwrap; no IFG write required.
            state = run_pair(
                self.catalog.path_for(primary),
                self.catalog.path_for(secondary),
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

    def invert_misreg(
        self,
        *,
        master: str | None = None,
        min_n_valid: int = 0,
        max_sigma_px: float = 1e3,
    ) -> Self:
        """Invert arcs to per-date az/rg with master fixed at 0."""
        self._ensure_prepared()
        if self.config.coreg_mode != "network":
            logger.info("invert_misreg skipped (coreg_mode=%s)", self.config.coreg_mode)
            return self
        if not self.arcs:
            self.measure_misreg()
        ref = master or self.master
        try:
            self.date_misreg = invert_pair_misregistration(
                self.arcs,
                master=ref,
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
                    "master": self.date_misreg.master,
                    "azimuth_px": dict(self.date_misreg.azimuth_px),
                    "range_px": dict(self.date_misreg.range_px),
                    "metadata": dict(self.date_misreg.metadata),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return self

    def coregister_scenes(
        self,
        *,
        dates: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Coregister each non-master date once onto the master grid and cache.

        This stage does **not** form interferograms.
        """
        self._ensure_prepared()
        from faninsar.processing.pipeline.production import run_pair

        target_dates = list(dates) if dates is not None else [
            d for d in self.catalog.dates if d != self.master
        ]
        master_path = self.catalog.path_for(self.master)
        # Cache master identity product path (no self-coreg).
        master_dir = self.config.work_dir / "coreg" / self.master
        master_dir.mkdir(parents=True, exist_ok=True)
        self.coreg_paths[self.master] = master_dir

        esd_on = self.config.coreg_mode == "pair"
        amp_on = self.config.coreg_mode in {"pair", "network"}
        # network: measure-only residuals already inverted; do not re-apply pair ESD
        if self.config.coreg_mode == "network":
            esd_on = False
            amp_on = False

        for date_id in target_dates:
            out = self.config.work_dir / "coreg" / date_id
            marker = out / "coreg_done.json"
            if marker.is_file() and not overwrite:
                CoregisteredSceneStore.open(out / "scenes")
                self.coreg_paths[date_id] = out
                if date_id == target_dates[0]:
                    copy_reference_units(out / "scenes", master_dir / "scenes")
                continue
            misreg_az = 0.0
            misreg_rg = 0.0
            if self.date_misreg is not None:
                misreg_az = float(self.date_misreg.azimuth_px.get(date_id, 0.0))
                misreg_rg = float(self.date_misreg.range_px.get(date_id, 0.0))
            state = run_pair(
                master_path,
                self.catalog.path_for(date_id),
                output_dir=out,
                multilook=self.config.multilook,
                goldstein_alpha=self.config.goldstein_alpha,
                esd_enabled=esd_on,
                amplitude_refinement_enabled=amp_on,
                unwrap=False,
                overwrite=overwrite,
                misreg_az_px=misreg_az,
                misreg_rg_px=misreg_rg,
                scene_store_dir=out / "scenes",
                **self._burst_kwargs(),
            )
            if self.config.retain_pair_states:
                self.pair_states[f"{self.master}_{date_id}"] = state
            else:
                logger.info(
                    "Released in-memory ProductionPairState for %s after scene "
                    "publication",
                    date_id,
                )
            self.coreg_paths[date_id] = out
            if date_id == target_dates[0]:
                copy_reference_units(out / "scenes", master_dir / "scenes")
            marker.write_text(
                json.dumps(
                    {
                        "master": self.master,
                        "date": date_id,
                        "misreg_az_px": misreg_az,
                        "misreg_rg_px": misreg_rg,
                        "geometric_phase_removed": (
                            "topo" if self.config.dem is not None else "flat"
                        ),
                        "esd_azimuth_shift_px": state.esd_azimuth_shift_px,
                        "range_shift_px": state.range_shift_px,
                        "azimuth_shift_px": state.azimuth_shift_px,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Coregistered %s → master %s", date_id, self.master)
        self._write_qualified_activation_record()
        return self

    def form_interferograms(
        self,
        *,
        pairs: Pairs | None = None,
        multilook: tuple[int, int]
        | list[tuple[int, int]]
        | None = None,
        goldstein_alpha: float | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Form interferograms from persisted master-aligned scene artifacts.

        This method deliberately has no SAFE-path or ``run_pair`` fallback.
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
        base_out = Path(output_dir) if output_dir is not None else (
            self.config.work_dir / "ifg"
        )

        for primary, secondary in _iter_pair_dates(use_pairs):
            if primary not in self.catalog.paths or secondary not in self.catalog.paths:
                continue
            for looks in looks_list:
                az_l, rg_l = looks
                sub = base_out / f"ml_{az_l}x{rg_l}" / f"{primary}_{secondary}"
                if (sub / "pair.zarr").exists() and not overwrite:
                    self.ifg_dirs.append(sub)
                    continue
                if primary not in self.coreg_paths or secondary not in self.coreg_paths:
                    reject_invalid_state(
                        "Stack scene generation missing; run coregister_scenes first"
                    )
                reference_store = CoregisteredSceneStore.open(
                    self.coreg_paths[primary] / "scenes"
                )
                secondary_store = CoregisteredSceneStore.open(
                    self.coreg_paths[secondary] / "scenes"
                )
                outputs = form_scene_interferograms(
                    reference_store,
                    secondary_store,
                    reference_role=(
                        "reference" if primary == self.master else "secondary"
                    ),
                    secondary_role=(
                        "reference" if secondary == self.master else "secondary"
                    ),
                )
                sub.mkdir(parents=True, exist_ok=True)
                for tag, ifg in outputs.items():
                    _atomic_write_array(
                        sub / f"{tag}.complex64",
                        ifg.astype(np.complex64, copy=False),
                    )
                metadata = {
                    "schema_version": "stack_ifg_v1",
                    "primary": primary,
                    "secondary": secondary,
                    "master": self.master,
                    "domain": self.config.coregistration_grid,
                    "multilook": [az_l, rg_l],
                    "goldstein_alpha": alpha,
                    "source_scene_manifest_digests": [
                        reference_store.manifest_digest,
                        secondary_store.manifest_digest,
                    ],
                }
                _atomic_write_text(
                    sub / "result.json",
                    json.dumps(metadata, indent=2) + "\n",
                )
                self.ifg_dirs.append(sub)
        return self

    def unwrap(self, **kwargs: Any) -> Self:
        """Reserved: unwrap pair products (thin wrap of production unwrap)."""
        _ = kwargs
        logger.warning(
            "Stack.unwrap is reserved; use run_pair(..., unwrap=True) for now",
        )
        return self

    def invert_timeseries(
        self,
        *,
        pair_phases: dict[str, np.ndarray] | None = None,
        device: str | None = None,
    ) -> TimeSeriesResult:
        """Invert unwrapped pair phases when provided."""
        from faninsar.processing.timeseries.inversion import invert_unwrapped_pairs

        if pair_phases is None:
            reject_invalid_state(
                "invert_timeseries requires pair_phases mapping in V1 "
                "(load unwrapped products then pass them in)",
            )
        self.timeseries = invert_unwrapped_pairs(
            pair_phases,
            device=device or self.config.invert_device,
        )
        return self.timeseries


def _iter_pair_dates(pairs: Pairs) -> list[tuple[str, str]]:
    """Yield (primary, secondary) YYYYMMDD pairs from a Pairs object."""
    out: list[tuple[str, str]] = []
    for name in pairs.names:
        a, b = str(name).split("_")
        out.append((a, b))
    return out


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

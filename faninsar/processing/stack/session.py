"""Mission-neutral Stack session (PROPOSAL-0017).

Orchestrates master-centric coregistration and pair products. Co-registration
and interferogram formation are separate stages: coreg caches per-date SLCs;
``form_interferograms`` only reads those products.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
from faninsar.processing.stack.config import CoregMode, EsdMethod, StackConfig

if TYPE_CHECKING:
    from faninsar.core.acquisition import Acquisition
    from faninsar.core.pairs import Pairs
    from faninsar.processing.geometry.dem import DEMSampler
    from faninsar.processing.pipeline.production import (
        BurstSelection,
        CoregistrationGrid,
        ProductionPairState,
    )
    from faninsar.processing.timeseries.inversion import TimeSeriesResult

logger = setup_logger(__name__)


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
            swaths=swaths,
            bursts=bursts,
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
            # Range residual is not yet exposed on ProductionPairState; record 0
            # until amp residual is plumbed (network still solves rg when arcs set).
            amp_rg = float(getattr(state, "amplitude_residual_rg", 0.0) or 0.0)
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
            if self.config.on_network_failure == "degrade_to_pair":
                logger.warning(
                    "misreg network invert failed; degrading coreg_mode to pair",
                )
                self.config.coreg_mode = "pair"
                self.date_misreg = None
                return self
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
                self.coreg_paths[date_id] = out
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
                **self._burst_kwargs(),
            )
            self.pair_states[f"{self.master}_{date_id}"] = state
            self.coreg_paths[date_id] = out
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
        """Form interferograms from master-aligned pair products.

        Reads already-coregistered products by re-running the pair engine with
        master as reference when needed. Does **not** recompute geometric phase
        beyond what coreg already applied (InSAR.dev-like).
        """
        self._ensure_prepared()
        from faninsar.processing.pipeline.production import run_pair

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

        esd_on = self.config.coreg_mode == "pair"
        amp_on = self.config.coreg_mode == "pair"

        for primary, secondary in _iter_pair_dates(use_pairs):
            if primary not in self.catalog.paths or secondary not in self.catalog.paths:
                continue
            for looks in looks_list:
                az_l, rg_l = looks
                sub = base_out / f"ml_{az_l}x{rg_l}" / f"{primary}_{secondary}"
                if (sub / "pair.zarr").exists() and not overwrite:
                    self.ifg_dirs.append(sub)
                    continue
                # Prefer master-centric when one leg is master; else run_pair.
                ref_id, sec_id = primary, secondary
                misreg_az = 0.0
                misreg_rg = 0.0
                if self.date_misreg is not None and self.config.coreg_mode == "network":
                    # Relative date constants for non-master pair (approx).
                    misreg_az = float(
                        self.date_misreg.azimuth_px.get(sec_id, 0.0)
                        - self.date_misreg.azimuth_px.get(ref_id, 0.0),
                    )
                    misreg_rg = float(
                        self.date_misreg.range_px.get(sec_id, 0.0)
                        - self.date_misreg.range_px.get(ref_id, 0.0),
                    )
                state = run_pair(
                    self.catalog.path_for(ref_id),
                    self.catalog.path_for(sec_id),
                    output_dir=sub,
                    multilook=looks,
                    goldstein_alpha=alpha,
                    esd_enabled=esd_on and self.config.coreg_mode != "network",
                    amplitude_refinement_enabled=amp_on
                    and self.config.coreg_mode != "network",
                    unwrap=False,
                    overwrite=overwrite,
                    misreg_az_px=misreg_az,
                    misreg_rg_px=misreg_rg,
                    **self._burst_kwargs(),
                )
                self.pair_states[f"{ref_id}_{sec_id}"] = state
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

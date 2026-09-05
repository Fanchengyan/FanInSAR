"""Thin NISAR RSLC adapter for the mission-neutral Stack (PROPOSAL-0035)."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from faninsar.core.acquisition import Acquisition
from faninsar.logging import setup_logger
from faninsar.missions.nisar import NisarSensor, _normalize_channel
from faninsar.stack.catalog import SceneCatalog
from faninsar.stack.config import ActivationMode, StackConfig
from faninsar.stack.nisar_provider import (
    make_nisar_scene_provider,
)
from faninsar.stack.provider import (
    StackSceneProvider,
)
from faninsar.stack.session import Stack, _pairs_from_factory

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from faninsar.core.pairs import Pairs
    from faninsar.processing.contracts import SLCProduct
    from faninsar.processing.readers import SLCReadResult

logger = setup_logger(__name__)
_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?:T\d{6}(?:\.\d+)?)?")


def _date_id(value: object) -> str | None:
    """Return a canonical acquisition date id when ``value`` is date-like."""
    if isinstance(value, datetime | date):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    match = _DATE_RE.search(text)
    if match is not None:
        return match.group(1)
    try:
        return datetime.fromisoformat(text).strftime("%Y%m%d")
    except ValueError:
        return None


def _acquisition_id(result: SLCReadResult, path: Path) -> str:
    """Resolve one acquisition id from the source name or normalized grid."""
    # NISAR file names carry the acquisition date in normal operational data.
    # Prefer it over a reader-specific acquisition label so catalog identity is
    # stable across reader versions.
    source_id = _date_id(path.name)
    if source_id is not None:
        return source_id
    grid_id = _date_id(result.product.grid.sensing_start)
    if grid_id is not None:
        return grid_id
    product_id = _date_id(result.product.acquisition_id)
    if product_id is not None:
        return product_id
    message = f"cannot discover NISAR acquisition date from {path}"
    logger.error(message)
    raise ValueError(message)


def _reference_id(value: object, dates: Sequence[str]) -> str:
    """Normalize an optional reference date and require catalog membership."""
    result = _date_id(value)
    if result is None or result not in dates:
        message = f"NISAR reference {value!r} is not an acquisition"
        logger.error(message)
        raise ValueError(message)
    return result


class NISARStack(Stack):
    """NISAR RSLC Stack with explicit B/HH channel semantics.

    The adapter owns optional-reader opening, normalized SLC metadata, source
    lineage, pair topology, and bounded source-scene publication. Geometry,
    flattening, multilooking, coherence, and geocoding remain the shared Stack
    pipeline's responsibility.
    """

    @classmethod
    def from_rslc(
        cls,
        paths: Sequence[str | Path],
        *,
        work_dir: str | Path,
        frequency: str = "B",
        polarization: str = "HH",
        reference: object | None = None,
        pairs: Pairs | None = None,
        misreg_pairs: Pairs | None = None,
        pair_max_interval: int = 3,
        pair_max_days: int = 72,
        misreg_max_interval: int = 2,
        misreg_max_days: int = 36,
        activation_mode: ActivationMode = "reference",
        **config_kwargs: Any,
    ) -> Self:
        """Construct a mission-neutral Stack from admitted NISAR RSLCs.

        Parameters
        ----------
        paths : sequence of path-like
            NISAR RSLC HDF5 sources.  Every source must represent a unique
            acquisition date.
        work_dir : path-like
            Stack artifact directory, retained for shared lifecycle methods.
        frequency, polarization : str, default="B", "HH"
            Explicit NISAR channel selection.  The adapter never silently
            chooses another channel.
        reference : date-like, optional
            Reference acquisition.  Defaults to the earliest source date, as in
            :meth:`Stack.from_safes`.
        pairs, misreg_pairs : Pairs, optional
            Explicit pair graphs.  If omitted, shared short-baseline graphs
            are generated from the discovered dates.
        pair_max_interval, pair_max_days : int, optional
            Maximum temporal graph interval and baseline for the IFG pairs.
        misreg_max_interval, misreg_max_days : int, optional
            Maximum temporal graph interval and baseline for misregistration.
        activation_mode : {"reference", "qualified"}, default="reference"
            Shared Stack activation namespace.  Qualified mode still requires
            its normal binding, token, and authority configuration.
        **config_kwargs : Any
            Additional :class:`StackConfig` options.

        Returns
        -------
        NISARStack
            Prepared metadata seam; raster data remains lazy.

        Raises
        ------
        ValueError
            If paths, dates, channels, or pair setup is unsupported.
        InvalidProcessingStateError
            If an existing RSLC path is opened without explicit admission.
        ImportError
            If the optional NISAR reader is unavailable.

        """
        source_paths = tuple(Path(path) for path in paths)
        admitted_frequency = _normalize_channel(frequency, field="frequency")
        admitted_polarization = _normalize_channel(
            polarization,
            field="polarization",
        )
        if len(source_paths) < 2:
            message = "NISARStack requires at least two RSLC paths"
            logger.error(message)
            raise ValueError(message)
        if len(set(source_paths)) != len(source_paths):
            message = "NISARStack RSLC paths must be unique"
            logger.error(message)
            raise ValueError(message)

        extra = dict(config_kwargs.pop("extra", {}) or {})
        configured_admission = config_kwargs.pop("nisar_admission", None)
        if configured_admission is None:
            configured_admission = config_kwargs.pop("admission", None)
        if configured_admission is None:
            configured_admission = extra.get(
                "nisar_admission", extra.get("admission_policy")
            )
        sensor = NisarSensor()
        handles: dict[Path, Any] = {}
        results: dict[str, SLCReadResult] = {}
        lineage: dict[str, str] = {}
        admission_lineage: dict[str, dict[str, object]] = {}
        for path in source_paths:
            if configured_admission is None:
                handle = sensor.open_product(str(path))
            else:
                handle = sensor.open_product(
                    str(path),
                    admission=configured_admission,
                )
            result = sensor.to_slc_product(
                handle,
                frequency=admitted_frequency,
                polarization=admitted_polarization,
                source_path=path,
            )
            native_frequency = _normalize_channel(
                result.mission_native.get("frequency"),
                field="frequency",
            )
            native_polarization = _normalize_channel(
                result.mission_native.get("polarization"),
                field="polarization",
            )
            if (native_frequency, native_polarization) != (
                admitted_frequency,
                admitted_polarization,
            ):
                message = (
                    "NISAR reader admitted a channel different from the requested "
                    f"{admitted_frequency}/{admitted_polarization}: "
                    f"{native_frequency}/{native_polarization}"
                )
                logger.error(message)
                raise ValueError(message)
            if result.source_path != str(path):
                message = (
                    "NISAR reader source lineage does not match the admitted RSLC "
                    f"path {path}: {result.source_path}"
                )
                logger.error(message)
                raise ValueError(message)
            acquisition_id = _acquisition_id(result, path)
            if acquisition_id in results:
                message = (
                    "NISARStack requires one RSLC per acquisition; duplicate "
                    f"date {acquisition_id} appears in {path} and "
                    f"{lineage[acquisition_id]}"
                )
                logger.error(message)
                raise ValueError(message)
            # Keep the normalized product's acquisition identity aligned with
            # the Stack catalog, while preserving every other reader field.
            result = replace(
                result,
                product=replace(result.product, acquisition_id=acquisition_id),
            )
            handles[path] = handle
            results[acquisition_id] = result
            lineage[acquisition_id] = str(path)
            metadata = sensor.admission_metadata(handle)
            if metadata is None:
                native = result.mission_native
                if native.get("source_id") and native.get("source_digest"):
                    metadata = {
                        "source_id": native["source_id"],
                        "source_digest": native["source_digest"],
                        "source_size_bytes": native.get("source_size_bytes", 0),
                        "policy": native.get("admission_policy"),
                    }
            if metadata is not None:
                admission_lineage[acquisition_id] = metadata

        dates = tuple(sorted(results))
        catalog = SceneCatalog(
            paths={date_id: Path(lineage[date_id]) for date_id in dates}
        )
        reference_id = (
            dates[0] if reference is None else _reference_id(reference, dates)
        )
        ifg_pairs = pairs or _pairs_from_factory(
            dates,
            max_interval=pair_max_interval,
            max_days=pair_max_days,
        )
        network_pairs = misreg_pairs or _pairs_from_factory(
            dates,
            max_interval=misreg_max_interval,
            max_days=misreg_max_days,
        )
        extra.update(
            {
                "mission": "NISAR",
                "product": "RSLC",
                "frequency": admitted_frequency,
                "polarization": admitted_polarization,
                "source_lineage": dict(lineage),
                "source_admission": dict(admission_lineage),
            }
        )
        config = StackConfig(
            work_dir=Path(work_dir),
            activation_mode=activation_mode,
            swaths=(),
            extra=extra,
            **config_kwargs,
        )
        configured_window = extra.get("nisar_window", extra.get("rslc_window"))
        configured_tile_shape = extra.get("nisar_tile_shape")
        provider_callback = make_nisar_scene_provider(
            sensor=sensor,
            handles=handles,
            products={date_id: item.product for date_id, item in results.items()},
            lineage=lineage,
            # The provider callback is an internal seam; the Stack API uses
            # the canonical Reference terminology above.
            reference=reference_id,
            channel=(admitted_frequency, admitted_polarization),
            configured_window=configured_window,
            configured_tile_shape=configured_tile_shape,
            configured_dem=config.dem,
            configured_height=extra.get("height_m", extra.get("height")),
            admission_lineage=admission_lineage,
            flatten_stage=config.flatten_stage,
        )
        stack = cls(
            catalog=catalog,
            config=config,
            pairs=ifg_pairs,
            misreg_pairs=network_pairs,
            reference=reference_id,
            acquisitions=Acquisition(list(dates)),
            scene_provider=StackSceneProvider(
                name="NISAR RSLC",
                produce_pair=provider_callback,
            ),
        )
        stack._nisar_sensor = sensor
        stack._nisar_handles = handles
        stack._nisar_results = results
        stack._nisar_lineage = lineage
        stack._nisar_admission = admission_lineage
        stack._nisar_channel = (
            admitted_frequency,
            admitted_polarization,
        )
        return stack

    @property
    def channel(self) -> tuple[str, str]:
        """Return the selected ``(frequency, polarization)`` channel."""
        return self._nisar_channel

    @property
    def source_lineage(self) -> Mapping[str, str]:
        """Return date-to-source lineage for the admitted RSLCs."""
        return dict(self._nisar_lineage)

    @property
    def source_admission(self) -> Mapping[str, Mapping[str, object]]:
        """Return digest, source id, and trusted-open policy by acquisition."""
        return {date_id: dict(item) for date_id, item in self._nisar_admission.items()}

    @property
    def products(self) -> Mapping[str, SLCProduct]:
        """Return normalized lazy SLC products keyed by acquisition date."""
        return {date_id: item.product for date_id, item in self._nisar_results.items()}

    @property
    def reader_handles(self) -> Mapping[str, Any]:
        """Return opened native reader handles without reading raster arrays."""
        return {
            date_id: self._nisar_handles[Path(source)]
            for date_id, source in self._nisar_lineage.items()
        }


__all__ = ["NISARStack"]

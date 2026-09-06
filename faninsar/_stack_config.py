"""Public Stack-oriented ``run(config)`` front door."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_pair_configuration
from faninsar.processing.runtime.compute.numpy_backend import NumpyBackend
from faninsar.stack.mask_plan import MaskPlan

if TYPE_CHECKING:
    from faninsar.processing.runtime.protocols import ComputeBackend

logger = setup_logger(__name__)


def run(
    config: str | Path | dict[str, Any],
    *,
    client: Any | None = None,
    store: Any | None = None,
    backend: str | ComputeBackend = "numpy",
    fmt: str = "cog",
) -> Any:
    """Run a configured Stack workflow from a mapping or YAML path.

    Parameters
    ----------
    config : str, Path, or dict
        Stack configuration. ``paths`` (or ``sources``) must contain every
        acquisition path and ``output`` identifies the Stack artifact root.
        ``reference`` optionally selects the Stack-wide Reference acquisition.
        Pair-shaped ``reference`` plus ``secondary`` configurations are
        rejected with a migration error; the removed ``master`` key is also
        rejected rather than silently translated.

    client : optional
        Injected Dask Client (never constructed here).
    store : optional
        Injected product store.
    backend : str or ComputeBackend, optional
        ``"numpy"``, ``"dask_torch"``, or a ComputeBackend instance.
    fmt : str, optional
        Output format tag (``cog``, ``zarr``).

    Returns
    -------
    Any
        The prepared and processed :class:`~faninsar.stack.Stack`.

    Raises
    ------
    PairConfigurationMigrationError
        If a removed pair-shaped configuration is supplied.
    ValueError
        If required Stack fields are missing or unsupported options are used.

    """
    del client, store, fmt  # reserved for full YAML wiring (Phase 7)
    cfg = _load_config(config)
    mask_plan = _load_mask_plan(config, cfg)
    paths = cfg.get("paths") or cfg.get("sources")
    output = cfg.get("output") or cfg.get("output_dir")
    _reject_legacy_pair_config(cfg, has_stack_paths=paths is not None)
    if not paths or not output:
        message = "run() Stack config requires 'paths' (or 'sources') plus 'output'"
        logger.error(message)
        raise ValueError(message)

    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    source_paths: list[str | Path] = []
    for item in raw_paths:
        if isinstance(item, (str, Path)):
            source_paths.append(item)
        else:
            try:
                source_paths.extend(item)
            except TypeError as exc:
                message = "run() Stack config paths must be path-like values"
                logger.exception(message)
                raise TypeError(message) from exc
    if len(source_paths) < 2:
        message = "run() Stack config requires at least two acquisition paths"
        logger.error(message)
        raise ValueError(message)

    unsupported = sorted(
        {
            key
            for key in (
                "dead_pixel_amp_threshold",
                "esd_enabled",
                "amplitude_refinement_enabled",
                "geoid_correction",
                "unwrap_method",
            )
            if key in cfg
        }
    )
    if unsupported:
        message = "run() Stack config contains unsupported stage options: " + ", ".join(
            unsupported
        )
        logger.error(message)
        raise ValueError(message)

    # Resolve the requested backend only after the shape and required fields
    # have been admitted. The Stack remains the sole public execution seam.
    compute = _resolve_backend(backend)
    del compute

    from faninsar.stack import Stack

    kwargs: dict[str, Any] = {}
    if "swaths" in cfg:
        kwargs["swaths"] = tuple(cfg["swaths"])
    elif "swath" in cfg:
        kwargs["swaths"] = (cfg["swath"],)
    if "bursts" in cfg:
        kwargs["bursts"] = cfg["bursts"]
    elif "burst_index" in cfg:
        swath = cfg.get("swath", "IW1")
        kwargs["bursts"] = {swath: [cfg["burst_index"]]}
    if "mask_plan" in cfg:
        kwargs["mask_plan"] = mask_plan
    for key in (
        "roi",
        "dem",
        "multilook",
        "goldstein_alpha",
        "control_spacing",
        "executor",
        "device",
        "orbit_paths",
        "coreg_mode",
        "flatten_stage",
        "coregistration_grid",
        "geo_grid",
        "invert_device",
        "activation_binding",
        "activation_token",
        "activation_authority_root",
        "retain_pair_states",
        "record_scientific_lineage",
    ):
        if key in cfg:
            kwargs[key] = cfg[key]
    stack = Stack.from_safes(
        source_paths,
        work_dir=output,
        reference=cfg.get("reference"),
        activation_mode=cfg.get("activation_mode", "reference"),
        **kwargs,
    )
    overwrite = bool(cfg.get("overwrite", False))
    stack.prepare_scenes().coregister_scenes().form_interferograms(
        overwrite=overwrite,
    )
    if cfg.get("unwrap"):
        stack.unwrap()
    return stack


def _reject_legacy_pair_config(
    cfg: dict[str, Any],
    *,
    has_stack_paths: bool,
) -> None:
    """Reject removed pair-shaped fields before Stack or backend dispatch."""
    secondary_keys = {
        "secondary",
        "secondary_path",
        "secondary_orbit_path",
        "reference_orbit_path",
    } & cfg.keys()
    if "master" in cfg:
        reject_pair_configuration(
            "run() no longer accepts the removed 'master' key; use the Stack "
            "'reference' field"
        )
    reference_path = "reference_path" in cfg
    if secondary_keys or reference_path or ("reference" in cfg and not has_stack_paths):
        fields = sorted(
            {
                key
                for key in (
                    "reference",
                    "secondary",
                    "reference_path",
                    "reference_orbit_path",
                    "secondary_path",
                    "secondary_orbit_path",
                )
                if key in cfg
            }
        )
        detail = ", ".join(fields) or "reference/secondary"
        reject_pair_configuration(
            "run() no longer accepts pair-shaped configuration fields "
            f"({detail}); provide all acquisitions under 'paths' or 'sources' "
            "and select an optional Stack 'reference' instead"
        )


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        data = dict(config)
        _reject_legacy_mask_config(data)
        return data
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            message = "PyYAML required for YAML configs"
            raise ImportError(message) from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            message = "YAML config must be a mapping"
            raise ValueError(message)
        _reject_legacy_mask_config(data)
        return data
    # JSON
    import json

    data = json.loads(text)
    if not isinstance(data, dict):
        message = "JSON config must be a mapping"
        raise TypeError(message)
    _reject_legacy_mask_config(data)
    return data


def _load_mask_plan(
    config: str | Path | dict[str, Any], cfg: dict[str, Any]
) -> MaskPlan:
    """Normalize the explicit ``mask_plan`` section of a Stack config."""
    if "mask_plan" not in cfg:
        return MaskPlan()
    value = cfg["mask_plan"]
    if not isinstance(value, dict):
        message = "'mask_plan' must be a mapping"
        logger.error(message)
        raise TypeError(message)
    base_dir = Path(config).resolve().parent if not isinstance(config, dict) else None
    return MaskPlan.from_mapping(value, base_dir=base_dir)


def _reject_legacy_mask_config(cfg: dict[str, Any]) -> None:
    """Reject removed mask spellings at the public Stack config boundary."""
    legacy = {
        "mask",
        "mask_source",
        "mask_on_failure",
        "mask_apply_ionosphere",
        "water_mask",
        "auto_water_mask",
    } & cfg.keys()
    if legacy:
        message = (
            "legacy mask configuration is not supported; use explicit "
            "'mask_plan' definitions and stage references: " + ", ".join(sorted(legacy))
        )
        logger.error(message)
        raise ValueError(message)


def _resolve_backend(backend: str | ComputeBackend) -> ComputeBackend:
    if isinstance(backend, str):
        if backend == "numpy":
            return NumpyBackend()
        if backend in {"dask_torch", "dask"}:
            from faninsar.processing.runtime.compute.dask_torch import DaskTorchBackend

            return DaskTorchBackend()
        message = f"unknown backend {backend!r}"
        raise ValueError(message)
    return backend


__all__ = ["run"]

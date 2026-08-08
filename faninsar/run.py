"""Public ``run(config)`` front door."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.compute.numpy_backend import NumpyBackend

if TYPE_CHECKING:
    from faninsar.ports.compute import ComputeBackend


def run(
    config: str | Path | dict[str, Any],
    *,
    client: Any | None = None,
    store: Any | None = None,
    backend: str | ComputeBackend = "numpy",
    fmt: str = "cog",
) -> Any:
    """Run a pair or stack workflow from a config mapping or YAML path.

    Parameters
    ----------
    config : str, Path, or dict
        Workflow configuration. Required keys for the pair path:

        - ``reference`` / ``secondary``: SAFE URIs
        - ``output``: output URI/directory
    - optional ``swaths``, ``bursts``, ``roi``, ``multilook``, …

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
        Production pair state or workflow result.

    """
    del client, store, fmt  # reserved for full YAML wiring (Phase 7)
    cfg = _load_config(config)
    compute = _resolve_backend(backend)

    reference = cfg.get("reference") or cfg.get("reference_path")
    secondary = cfg.get("secondary") or cfg.get("secondary_path")
    output = cfg.get("output") or cfg.get("output_dir")
    if not reference or not secondary or not output:
        message = (
            "run() pair config requires 'reference', 'secondary', and 'output' keys"
        )
        raise ValueError(message)

    from faninsar.processing.pipeline import run_pair

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
    for key in (
        "roi",
        "dem",
        "multilook",
        "goldstein_alpha",
        "dead_pixel_amp_threshold",
        "esd_enabled",
        "amplitude_refinement_enabled",
        "control_spacing",
        "executor",
        "device",
        "reference_orbit_path",
        "secondary_orbit_path",
        "geoid_correction",
    ):
        if key in cfg:
            kwargs[key] = cfg[key]
    if cfg.get("unwrap") or cfg.get("unwrap_method") is not None:
        kwargs["unwrap"] = True
    # backend reserved for stage-level dispatch; production uses torch executor today
    del compute
    return run_pair(reference, secondary, output_dir=output, **kwargs)


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
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
        return data
    # JSON
    import json

    data = json.loads(text)
    if not isinstance(data, dict):
        message = "JSON config must be a mapping"
        raise TypeError(message)
    return data


def _resolve_backend(backend: str | ComputeBackend) -> ComputeBackend:
    if isinstance(backend, str):
        if backend == "numpy":
            return NumpyBackend()
        if backend in {"dask_torch", "dask"}:
            from faninsar.compute.dask_torch import DaskTorchBackend

            return DaskTorchBackend()
        message = f"unknown backend {backend!r}"
        raise ValueError(message)
    return backend


__all__ = ["run"]

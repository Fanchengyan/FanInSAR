"""Public ``run(config)`` front door."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faninsar.compute.numpy_backend import NumpyBackend
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
        - optional ``swath``, ``burst_index``, ``coregistration_grid``, …

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
        raise ValueError(
            "run() pair config requires 'reference', 'secondary', and 'output' keys"
        )

    from faninsar.processing.pipeline import run_production_pair

    kwargs = {
        k: cfg[k]
        for k in (
            "swath",
            "scope",
            "burst_index",
            "multilook",
            "unwrap_method",
            "esd_enabled",
            "coregistration_grid",
            "device",
            "executor",
        )
        if k in cfg
    }
    # backend reserved for stage-level dispatch; production uses torch executor today
    del compute
    return run_production_pair(reference, secondary, output_dir=output, **kwargs)


def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML required for YAML configs") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("YAML config must be a mapping")
        return data
    # JSON
    import json

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON config must be a mapping")
    return data


def _resolve_backend(backend: str | ComputeBackend) -> ComputeBackend:
    if isinstance(backend, str):
        if backend == "numpy":
            return NumpyBackend()
        if backend in {"dask_torch", "dask"}:
            from faninsar.compute.dask_torch import DaskTorchBackend

            return DaskTorchBackend()
        raise ValueError(f"unknown backend {backend!r}")
    return backend


__all__ = ["run"]

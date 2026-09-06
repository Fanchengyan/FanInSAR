"""CompileManager and COMPILE_TARGETS registry for hot Torch kernels."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from faninsar.processing.runtime.compute.cache import apply_compile_cache_env
from faninsar.processing.runtime.compute.compile_specs import DEFAULT_CHUNK

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class CompileTarget:
    """A named kernel that may be torch.compiled."""

    name: str
    factory: Callable[..., Callable[..., Any]]
    description: str = ""


@dataclass
class CompileManager:
    """Lazy ``torch.compile`` with process cache and eager fallback."""

    recompile_counts: dict[str, int] = field(default_factory=dict)
    _cache: dict[tuple[str, str, str, int], Callable[..., Any]] = field(
        default_factory=dict, repr=False
    )
    _eager_fallback: bool = True

    def get(
        self,
        name: str,
        *,
        device: str,
        dtype: Any = None,
        chunk_size: int = DEFAULT_CHUNK,
    ) -> Callable[..., Any]:
        """Return a compiled or eager kernel for *name*.

        Parameters
        ----------
        name : str
            Key in ``COMPILE_TARGETS``.
        device : str
            Torch device string (``cpu``, ``cuda``, ``mps``).
        dtype : torch.dtype, optional
            Preferred dtype; defaults to complex64 when torch is available.
        chunk_size : int, optional
            Fixed chunk size for shape specialization.

        """
        apply_compile_cache_env()
        if name not in COMPILE_TARGETS:
            message = f"unknown compile target {name!r}"
            raise KeyError(message)

        try:
            import torch
        except ImportError:
            return COMPILE_TARGETS[name].factory()

        if dtype is None:
            dtype = torch.complex64
        dtype_key = str(dtype)
        cache_key = (name, device, dtype_key, chunk_size)
        if cache_key in self._cache:
            return self._cache[cache_key]

        eager = COMPILE_TARGETS[name].factory()
        try:
            compiled = torch.compile(eager, fullgraph=False)

            def _safe(*args: Any, **kwargs: Any) -> Any:
                try:
                    return compiled(*args, **kwargs)
                except Exception as run_exc:
                    logger.warning(
                        "compiled %s runtime failed (%s); eager fallback",
                        name,
                        run_exc,
                    )
                    return eager(*args, **kwargs)

        except Exception as exc:
            logger.warning(
                "torch.compile failed for %s (%s); using eager", name, exc
            )
            if not self._eager_fallback:
                raise
            self._cache[cache_key] = eager
            return eager
        else:
            self.recompile_counts[name] = self.recompile_counts.get(name, 0) + 1
            self._cache[cache_key] = _safe
            return _safe


def _identity_kernel() -> Callable[..., Any]:
    """Create an identity kernel until real kernels are registered."""

    def _fn(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs:
            return args[0]
        return args, kwargs

    return _fn


def _multilook_factory() -> Callable[..., Any]:
    """Boxcar multilook factory (compile target)."""

    def multilook(arr: Any, looks: tuple[int, int] = (2, 2)) -> Any:
        import torch

        if not isinstance(arr, torch.Tensor):
            arr = torch.as_tensor(arr)
        az, rg = looks
        if torch.is_complex(arr):
            real = torch.nn.functional.avg_pool2d(
                arr.real.unsqueeze(0).unsqueeze(0), (az, rg)
            )
            imag = torch.nn.functional.avg_pool2d(
                arr.imag.unsqueeze(0).unsqueeze(0), (az, rg)
            )
            return torch.complex(real.squeeze(), imag.squeeze())
        return torch.nn.functional.avg_pool2d(
            arr.unsqueeze(0).unsqueeze(0).float(), (az, rg)
        ).squeeze()

    return multilook


def _form_interferogram_factory() -> Callable[..., Any]:
    """Complex interferogram formation factory."""

    def form_interferogram(ref: Any, sec: Any) -> Any:
        import torch

        if not isinstance(ref, torch.Tensor):
            ref = torch.as_tensor(ref)
        if not isinstance(sec, torch.Tensor):
            sec = torch.as_tensor(sec)
        return ref * torch.conj(sec)

    return form_interferogram


def _coherence_factory() -> Callable[..., Any]:
    """Global coherence statistic factory."""

    def coherence(ref: Any, sec: Any, window: int = 5) -> Any:
        import torch

        del window
        if not isinstance(ref, torch.Tensor):
            ref = torch.as_tensor(ref)
        if not isinstance(sec, torch.Tensor):
            sec = torch.as_tensor(sec)
        cross = ref * torch.conj(sec)
        num = torch.abs(cross.mean())
        den = torch.sqrt((ref.abs() ** 2).mean() * (sec.abs() ** 2).mean())
        return num / den.clamp_min(1e-12)

    return coherence


def _register_default_targets() -> dict[str, CompileTarget]:
    factories: dict[str, Callable[..., Callable[..., Any]]] = {
        "knab_resample": _identity_kernel,
        "lanczos_resample": _identity_kernel,
        "deramp": _identity_kernel,
        "reramp": _identity_kernel,
        "form_interferogram": _form_interferogram_factory,
        "flatten": _identity_kernel,
        "multilook": _multilook_factory,
        "coherence": _coherence_factory,
    }
    return {
        k: CompileTarget(name=k, factory=fn, description=k)
        for k, fn in factories.items()
    }


COMPILE_TARGETS: dict[str, CompileTarget] = _register_default_targets()

_MANAGER: CompileManager | None = None


def get_compile_manager() -> CompileManager:
    """Return the process-wide CompileManager singleton."""
    global _MANAGER  # noqa: PLW0603
    if _MANAGER is None:
        _MANAGER = CompileManager()
    return _MANAGER


def register_kernel(
    name: str,
    factory: Callable[..., Callable[..., Any]],
    *,
    description: str = "",
) -> None:
    """Register or replace a compile target factory."""
    COMPILE_TARGETS[name] = CompileTarget(
        name=name, factory=factory, description=description or name
    )


__all__ = [
    "COMPILE_TARGETS",
    "CompileManager",
    "CompileTarget",
    "get_compile_manager",
    "register_kernel",
]

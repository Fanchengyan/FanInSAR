"""``faninsar warmup`` — pre-compile hot kernels and seed the inductor cache."""

from __future__ import annotations

import time

from faninsar.compute.cache import apply_compile_cache_env
from faninsar.compute.compile import COMPILE_TARGETS, get_compile_manager
from faninsar.compute.profiles import get_profile
from faninsar.logging import setup_logger

logger = setup_logger(__name__)


def run_warmup(*, device: str = "cpu", profile: str = "sentinel1") -> int:
    """Compile registered kernels for *profile* on *device*.

    Returns
    -------
    int
        Exit code (0 on success).

    """
    cache_dir = apply_compile_cache_env()
    prof = get_profile(profile if "-" in profile else f"{profile}-{device}")
    if device == "auto":
        device = prof.device
    elif device != "cpu":
        # honour CLI device override
        pass
    else:
        device = "cpu"

    logger.info("faninsar warmup")
    logger.info("device: %s | profile: %s | cache: %s", device, prof.name, cache_dir)

    try:
        import torch
    except ImportError:
        logger.exception("torch not available; nothing to compile")
        return 1

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to cpu")
        device = "cpu"
    mps_missing = device == "mps" and not torch.backends.mps.is_available()
    if mps_missing:
        logger.warning("MPS not available; falling back to cpu")
        device = "cpu"

    manager = get_compile_manager()
    targets = prof.kernel_groups or tuple(COMPILE_TARGETS)
    t0 = time.perf_counter()
    for i, name in enumerate(targets, 1):
        if name not in COMPILE_TARGETS:
            logger.warning("[%s/%s] skip unknown %s", i, len(targets), name)
            continue
        logger.info("[%s/%s] compile %s", i, len(targets), name)
        started = time.perf_counter()
        fn = manager.get(name, device=device)
        # Tiny synthetic trigger so inductor may specialize
        try:
            x = torch.zeros(64, 64, dtype=torch.complex64)
            if device != "cpu":
                x = x.to(device)
            _ = fn(x)
        except Exception:
            # identity kernels accept any args; ignore runtime shape issues
            pass
        logger.info("%.2fs", time.perf_counter() - started)
    logger.info("done in %.2fs", time.perf_counter() - t0)
    return 0


__all__ = ["run_warmup"]

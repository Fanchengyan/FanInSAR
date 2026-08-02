"""``faninsar warmup`` — pre-compile hot kernels and seed the inductor cache."""

from __future__ import annotations

import sys
import time

from faninsar.compute.cache import apply_compile_cache_env
from faninsar.compute.compile import COMPILE_TARGETS, get_compile_manager
from faninsar.compute.profiles import get_profile


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

    print("faninsar warmup")
    print(f"  device:  {device}")
    print(f"  profile: {prof.name}")
    print(f"  cache:   {cache_dir}")

    try:
        import torch
    except ImportError:
        print("torch not available; nothing to compile", file=sys.stderr)
        return 1

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to cpu")
        device = "cpu"
    if (device == "mps" and not getattr(torch.backends, "mps", None)) or (device == "mps" and not torch.backends.mps.is_available()):
        print("MPS not available; falling back to cpu")
        device = "cpu"

    manager = get_compile_manager()
    targets = prof.kernel_groups or tuple(COMPILE_TARGETS)
    t0 = time.perf_counter()
    for i, name in enumerate(targets, 1):
        if name not in COMPILE_TARGETS:
            print(f"  [{i}/{len(targets)}] skip unknown {name}")
            continue
        print(f"  [{i}/{len(targets)}] compile {name} …", end=" ", flush=True)
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
        print(f"{time.perf_counter() - started:.2f}s")
    print(f"done in {time.perf_counter() - t0:.2f}s")
    return 0


__all__ = ["run_warmup"]

"""Pure torch/numpy kernels (compile targets).

Hot kernels are registered in :mod:`faninsar.compute.compile` via
``COMPILE_TARGETS``. Production stages may call
``get_compile_manager().get(name)`` for the hot path.
"""

from __future__ import annotations

__all__: list[str] = []

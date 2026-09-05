"""Physical data access grouped under one discoverable package.

The package contains reusable dataset readers, spatial/temporal query values,
and PyTorch-compatible samplers.  These are data primitives rather than
processing workflows.
"""

from __future__ import annotations

from . import datasets, query, samplers

__all__ = ["datasets", "query", "samplers"]

"""Product validation helpers (manifest / provenance).

Oracle and multi-stack residual comparison live outside this package:
``/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/lib/validation/``.
"""

from __future__ import annotations

from faninsar.validation.provenance import validate_pipeline_rebuild_manifest

__all__ = [
    "validate_pipeline_rebuild_manifest",
]

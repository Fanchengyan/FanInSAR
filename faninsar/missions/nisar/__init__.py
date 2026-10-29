"""NISAR mission adapter public surface.

The implementation lives in :mod:`faninsar.missions.nisar.adapter`; this
package keeps the mission-level import surface small while leaving the
optional RSLC reader details in one implementation module.
"""

from __future__ import annotations

from typing import Any

from faninsar.missions.nisar.adapter import (
    NisarAdmissionPolicy,
    NisarSensor,
    admit_nisar_source,
    normalize_channel,
)


def __getattr__(name: str) -> Any:
    """Load the concrete NISAR Stack adapter and GUNW datasets lazily."""
    if name == "NISARStack":
        from faninsar.missions.nisar.stack import NISARStack

        return NISARStack
    if name in {"NISAR", "NISARIonosphere", "NisarCoherenceDataset"}:
        from faninsar.missions.nisar import gunw

        return getattr(gunw, name)
    raise AttributeError(name)


__all__ = [
    "NISAR",
    "NISARIonosphere",
    "NISARStack",
    "NisarAdmissionPolicy",
    "NisarCoherenceDataset",
    "NisarSensor",
    "admit_nisar_source",
    "normalize_channel",
]

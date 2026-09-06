"""NISAR mission adapter public surface.

The implementation lives in :mod:`faninsar.missions.nisar.adapter`; this
package keeps the mission-level import surface small while leaving the
optional RSLC reader details in one implementation module.
"""

from __future__ import annotations

from faninsar.missions.nisar.adapter import (
    NisarAdmissionPolicy,
    NisarSensor,
    admit_nisar_source,
)

__all__ = [
    "NisarAdmissionPolicy",
    "NisarSensor",
    "admit_nisar_source",
]

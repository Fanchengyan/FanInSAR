"""Enhanced Spectral Diversity (ESD) azimuth shift estimation.

Compatibility re-export of the dual-look split-spectrum estimator. The
implementation lives in
:mod:`faninsar.processing.coregistration.esd_splitband`; shared result
types live in :mod:`faninsar.processing.coregistration.esd_common`.
Burst-overlap ESD is :mod:`faninsar.processing.coregistration.esd_overlap`.

"""

from __future__ import annotations

from faninsar.processing.coregistration.esd_common import ESDResult
from faninsar.processing.coregistration.esd_splitband import estimate_azimuth_shift_esd

__all__ = ["ESDResult", "estimate_azimuth_shift_esd"]

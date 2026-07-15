"""Poison test: multilook must use boxcar complex mean, never Lanczos/sinc.

Scientific rule (OF-1 / L7 / OQ-5, sar-resampling-kernels skill)
-----------------------------------------------------------------
Multilooking is **coherent spatial averaging** for speckle reduction:
``g[n] = (1/L²) · Σ z[i,j]`` over non-overlapping look windows, then
``arg`` of the averaged complex. It is **not** bandlimited interpolation.

Lanczos / sinc kernels reconstruct bandlimited signals on a new grid
(resampling / geocoding / coregistration). Using them for multilook:

* injects negative side-lobes (ringing) into look averages,
* underestimates effective number of looks,
* can bias phase and coherence.

This contract test fails CI if the multilook path ever switches to a
resampling kernel.
"""

from __future__ import annotations

import inspect

import numpy as np

from faninsar.processing.interferometry.pair import form_interferogram


def test_multilook_uses_boxcar_not_lanczos_on_impulse() -> None:
    """Impulse response of multilook must match boxcar mean, not Lanczos.

    A single non-zero complex sample in a look window is spread uniformly
    by boxcar averaging (value / L² in that output cell, zeros elsewhere).
    Lanczos/sinc resampling of the same impulse produces ringing side-lobes
    in neighboring output cells. Asserting uniformity therefore poisons any
    future Lanczos-based multilook implementation.
    """
    az_looks, rg_looks = 2, 2
    height, width = 4, 4
    impulse_value = 4.0 + 0.0j

    primary = np.zeros((height, width), dtype=np.complex64)
    # Secondary = 1 so ifg = primary * conj(secondary) == primary.
    secondary = np.ones((height, width), dtype=np.complex64)
    # Impulse in the top-left 2×2 look window (row 0, col 0).
    primary[0, 0] = impulse_value

    product = form_interferogram(primary, secondary, multilook=(az_looks, rg_looks))
    ifg = product.complex_ifg

    assert ifg.shape == (height // az_looks, width // rg_looks)

    # Boxcar: only the (0, 0) output cell holds impulse_value / (L_az * L_rg).
    expected_peak = impulse_value / float(az_looks * rg_looks)
    assert np.isclose(ifg[0, 0], expected_peak, rtol=0.0, atol=1e-6)

    # All other output cells must be exactly zero (no Lanczos side-lobes).
    others = ifg.copy()
    others[0, 0] = 0
    max_sidelobe = float(np.max(np.abs(others)))
    assert max_sidelobe < 1e-6, (
        f"Multilook produced non-zero side-lobes (max |z|={max_sidelobe:.3e}); "
        "this indicates a resampling kernel (Lanczos/sinc) instead of boxcar "
        "complex averaging (OF-1 / L7)."
    )

    # Uniform energy: peak magnitude equals total input energy / looks.
    assert np.isclose(np.abs(ifg[0, 0]), np.abs(expected_peak), atol=1e-6)


def test_multilook_source_forbids_resampling_kernels() -> None:
    """Source of form_interferogram must not reference Lanczos/sinc kernels.

    Complements the numerical impulse test: even if a future kernel were
    tuned to pass a single impulse case, the multilook implementation must
    remain explicit boxcar averaging (mean / sum over look blocks).
    """
    source = inspect.getsource(form_interferogram)
    forbidden = ("lanczos", "sinc", "map_coordinates", "ndimage", "resample")
    lowered = source.lower()
    for token in forbidden:
        assert token not in lowered, (
            f"form_interferogram multilook path references forbidden token "
            f"{token!r}; multilook must be boxcar complex mean only (OF-1 / L7)."
        )

    # Positive contract: implementation must average over look blocks.
    assert "inv_looks" in source or ".mean(" in source or ".sum(" in source, (
        "form_interferogram multilook path does not appear to average over "
        "look blocks (expected inv_looks, .mean, or .sum)."
    )

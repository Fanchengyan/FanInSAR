# Resampling, multilooking, and downsampling

This page is the FanInSAR user's reference for choosing the correct kernel
when moving SAR data between grids. It distills the
`sar-resampling-kernels` agent skill into a developer-facing guide with
runnable examples based on the FanInSAR API.

If you only read one section, read [three-operations](#the-three-operations): the most common
SAR "resampling" bug is conflating *resampling* with *multilooking* and
*decimation*, which have fundamentally different mathematics.

:::{note}
This page reflects consensus across ISCE2, ISCE3/NISAR, ESA SNAP,
ESA Sentinel-1 IPF, GAMMA, and the canonical theoretical reference
[Hanssen & Bamler 1999](https://doi.org/10.1109/36.739168). Full source
citations are in the skill's `references/kernel_evidence.md`.
:::

:::{admonition} TL;DR
:class: tip

1. **Identify the operation first**: resampling, multilooking, or decimation?
2. **Resampling a complex SLC / wrapped ifg** → sinc/Lanczos (`a=4`). **Never** bilinear.
3. **Multilooking** → boxcar complex average. **Never** sinc. **Never** on phase alone.
4. **Downsampling** → low-pass anti-alias filter, then subsample.
5. **Intensity** → resample on `sqrt(intensity)`, then square.
:::

## The three operations

(three-operations)=

### 1. Resampling — interpolation to a new grid

**Goal**: move a signal from one grid to another at comparable resolution
(geocoding, coregistration, reprojection).

**Math**: bandlimited interpolation

$$
f_\text{out}(x) = \sum_k f_\text{in}[k] \cdot h(x - k)
$$

where $h$ is the reconstruction kernel (sinc, Lanczos, bilinear, B-spline).

**Kernel choice**: keyed on the data's physical type. See [](decision-table).

### 2. Multilooking — incoherent averaging for speckle reduction

**Goal**: reduce speckle noise by averaging $L \times L$ complex samples,
producing an equivalent number of looks $L^2$.

**Math**: boxcar (or Gaussian-weighted) complex average

$$
g[n] = \frac{1}{L^2} \sum_{(i,j) \in \text{block}_n} z[i, j]
$$

**Kernel choice**: **boxcar** (default) or **Gaussian** weighted average.
**Never** sinc/Lanczos as a multilook kernel — they have negative
side-lobes that decorrelate the averaged result.

:::{warning}
For a complex interferogram, **always average the complex samples**
$z = I \cdot e^{i\varphi}$ (or equivalently I/Q together), then take
$\arg(\cdot)$. Averaging the phase $\varphi$ alone produces a bell-shaped
histogram artifact. See [SNAP `MultilookOp` documentation](https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.sar.processing.ui/operators/MultilookOp.html).
:::

### 3. Downsampling (decimation) — coarser grid with anti-aliasing

**Goal**: produce a coarser-resolution version of a raster, typically for
storage, preview, or to match a coarse DEM grid.

**Math**: low-pass anti-alias filter, then integer-stride (or rational)
subsample

$$
g[n] = \sum_k f_\text{in}[nL + k] \cdot h_\text{lowpass}[k]
$$

**Kernel choice for the anti-alias filter**:

| Scenario | Filter |
|----------|--------|
| Complex SLC, also want speckle reduction | Boxcar complex average (this is just multilooking) |
| Complex SLC, preserve phase statistics | Taylor-windowed sinc ($a=4$) or raised-cosine sinc |
| Real smooth field (DEM, incidence) | Gaussian or B-spline order 4-5 |
| Integer label mask | **Mode** of $L \times L$ block, never mean |

:::{important}
Plain sinc interpolation to fractional coarser coordinates is **wrong**
for downsampling — it does not pre-filter, so out-of-band energy folds
back as aliasing. Always apply an explicit low-pass first.
:::

## Decision table

(decision-table)=

For the **resampling** operation (case 1 above), choose the kernel by
data type:

| Data type | Production kernel | Acceptable alt | Never use |
|-----------|-------------------|----------------|-----------|
| **SLC, wrapped ifg** (complex, full BW) | Lanczos $a=4$ | sinc (windowed), Lanczos $a=6$ | bilinear, nearest, bicubic(Keys) |
| **Geocoded SLC / GSLC** (NISAR, CSLC) | sinc (raised-cosine windowed) | Lanczos | bilinear |
| **Coherence (already estimated)** | bilinear | B-spline 4, nearest | sinc (overkill) |
| **Coherence (re-estimation on new grid)** | re-estimate $\gamma$ on resampled SLC pair | — | resample the $\gamma$ estimate |
| **Amplitude** ($\sqrt{\text{intensity}}$) | Lanczos $a=4$ or B-spline 4 | bilinear | nearest |
| **Intensity / backscatter** | **resample on $\sqrt{\text{intensity}}$, then square** | — | direct bilinear on intensity |
| **Unwrapped phase** | bilinear or complex-average | nearest | — |
| **DEM / height** | B-spline order 4-5 | bilinear, cubic | nearest |
| **Incidence / azimuth / heading angles** | bilinear | B-spline, nearest | — |
| **Hard label mask** (conncomp, layover, water) | nearest | — | bilinear, sinc |
| **Soft classification probabilities** | bilinear (renormalize after) | B-spline | nearest |
| **Already-multilooked complex** (low BW) | bilinear OK | Lanczos, B-spline | nearest |
| **Polarimetric C3/C4/T3/T4** | component-wise Lanczos + PD check | — | naive component-wise |

## Using FanInSAR's resamplers

### Lanczos for complex SLC / ifg

`faninsar.processing.resampling.lanczos_resample` is a phase-preserving
replacement for `scipy.ndimage.map_coordinates(order=1)` on complex SAR
data. It applies a separable Lanczos-$a$ kernel along rows then columns,
accumulating in float64 / complex128 to avoid precision bias:

```python
import numpy as np
from faninsar.processing.resampling import lanczos_resample

# slc: 2D complex64 SLC burst, shape (az, rg)
# coords: output coordinates of shape (2, N), coords[0]=rows, coords[1]=cols
# as in scipy.ndimage.map_coordinates
resampled = lanczos_resample(slc, coords, a=4, mode="constant", cval=0.0)
```

For wrapped phase, convert to complex first, resample, then take `arg`:

```python
z = np.exp(1j * wrapped_phase)            # complex on the unit circle
z_resampled = lanczos_resample(z, coords, a=4)
phase_resampled = np.angle(z_resampled)   # wrap-invariant result
```

### Real-valued smooth fields

Use `scipy.ndimage.map_coordinates` directly:

```python
from scipy.ndimage import map_coordinates

# DEM, incidence angle, already-estimated coherence
dem_resampled = map_coordinates(dem, coords, order=1, mode="nearest")  # bilinear
dem_bspline = map_coordinates(dem, coords, order=4, mode="nearest")    # B-spline order 4
```

For label masks, use `order=0` (nearest).

### Raster-to-raster reprojection

`faninsar.processing.geometry.raster_ops.match_to_raster` wraps
`rasterio.warp.reproject` and currently defaults to
`Resampling.nearest`. **This default is conservative** — appropriate
for label masks but suboptimal for DEM, amplitude, and coherence. Pass
the algorithm explicitly:

```python
from rasterio.warp import Resampling
from faninsar.processing.geometry.raster_ops import match_to_raster

# For DEM or coherence:
dem_matched = match_to_raster(dem, src_profile, dst_profile,
                              algorithm=Resampling.bilinear)

# For complex ifg/SLC being moved to a new geographic grid:
# split into real/imag, resample each as bilinear (rasterio limitation),
# recombine. For phase fidelity, use lanczos_resample at explicit coords.
```

:::{todo}
`match_to_raster` should default to a dtype-aware kernel: complex →
lanczos (split real/imag), real → bilinear, integer → nearest. This is
tracked as a follow-up to the skill review.
:::

### Multilooking an interferogram

For an $L \times L$ complex multilook:

```python
from scipy.ndimage.uniform_filter

def multilook_complex(z: np.ndarray, L: tuple[int, int]) -> np.ndarray:
    """Boxcar-average complex `z` by (L_az, L_rg). Phase-safe."""
    # uniform_filter computes the mean over the window — equivalent to
    # incoherent complex averaging when applied to complex input.
    real = uniform_filter(z.real.astype(np.float64), size=L, mode="reflect")
    imag = uniform_filter(z.imag.astype(np.float64), size=L, mode="reflect")
    return (real + 1j * imag).astype(z.dtype)

# Correct: average the complex ifg, then derive phase/intensity
z_ml = multilook_complex(ifg_complex, L=(2, 8))
phase_ml = np.angle(z_ml)
intensity_ml = np.abs(z_ml) ** 2

# WRONG — averaging phase directly produces bell-shaped histogram artifacts
# phase_ml_wrong = uniform_filter(np.angle(ifg_complex), size=(2, 8))
```

### Sub-pixel coregistration via FFT

When the operation is "apply a known integer + fractional shift `(dx, dy)`
to a complex image" — the canonical sub-pixel coregistration step — the
FFT-based shift is exact for bandlimited signals:

```python
def fft_shift(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ny, nx = z.shape
    fx = np.fft.fftfreq(nx)
    fy = np.fft.fftfreq(ny)
    ramp = np.exp(-2j * np.pi * (fx[None, :] * dx + fy[:, None] * dy))
    return np.fft.ifft2(np.fft.fft2(z) * ramp).astype(z.dtype)
```

This is preferred over spatial-domain Lanczos when the shift is uniform
across the image. For spatially-varying shifts (e.g., from
`dense_offsets`), use `lanczos_resample` at per-pixel fractional
coordinates.

## Common mistakes

1. **"Bilinear is fine — it's faster."** → On complex SLC/ifg, bilinear
   produces a sub-pixel-offset-dependent phase bias that appears as burst
   seams and cm-level phantom deformation after unwrapping.

2. **"Multilooking is just bilinear decimation."** → Multilook is boxcar
   **complex** average. For ifg, average I/Q (or $z = I e^{i\varphi}$),
   then `arg`. Averaging phase alone is wrong.

3. **"Sinc is the right kernel for downsampling."** → Plain sinc to
   fractional coarser coordinates does not pre-filter; aliasing results.
   Use sinc low-pass → decimate, or multilook.

4. **"Resample the intensity field directly."** → Resample on
   `sqrt(intensity)` (amplitude), then square. Direct intensity
   interpolation biases undersampled data ([GAMMA TR](https://www.gamma-rs.ch/uploads/media/2017-2_TR_Interpolation_and_resampling.pdf)).

5. **"Resample the coherence estimate to a new grid."** → Re-estimate
   $\gamma$ over a window on the resampled SLC pair instead. Resampling
   the estimate loses spatial-support information.

6. **"Apply different kernels to different ifgs in a stack."** → Stack
   inconsistency. For time-series that will be differenced or inverted,
   all members must use the same kernel and weights, otherwise bias
   differences masquerade as deformation.

7. **"Use bicubic for higher quality on the SLC."** → Bicubic (Keys
   $\alpha=-0.5$) has non-flat group delay and produces phase bias on
   complex SAR. (Cubic is acceptable for real-valued smooth fields; not
   for complex.)

8. **"Nearest is safest."** → Aliases everything except hard integer-class
   labels.

## Advanced topics

### Mask-aware resampling

Sinc/Lanczos have negative side-lobes that pull energy from valid samples
into masked regions and vice versa. For rasters with NoData/NaN:

```python
def lanczos_resample_masked(data, mask, coords, a=4):
    masked_data = np.where(mask, data, 0)
    weight = np.where(mask, 1.0, 0.0)
    num = lanczos_resample(masked_data, coords, a=a)
    den = lanczos_resample(weight, coords, a=a)
    threshold = 0.5 * (2 * a) ** 2  # require ≥ 50% valid tap coverage
    return np.where(den > threshold,
                    num / np.where(den == 0, 1, den),
                    np.nan)
```

### Polarimetric matrices (C3/C4/T3/T4)

These matrices are Hermitian positive-definite. Naive component-wise
interpolation can break positive-definiteness:

1. Resample each element with Lanczos (off-diagonal complex) or B-spline
   (diagonal real).
2. Recompute $C = (C + C^\dagger) / 2$ to enforce Hermitian symmetry.
3. If any eigenvalue $\le 0$, clip to a small positive floor and
   reconstruct: `C = V @ diag(clipped_eigvals) @ V.conj().T`.

### M1 vs M2 path

Two equivalent paths from SLC pair to geocoded ifg:

- **M1**: geocode each SLC → form ifg on geo grid.
- **M2**: form ifg on radar grid → geocode the ifg.

M1 is the gold standard (Lanczos on each SLC), but ~2× the compute. M2
is preferred when the ifg has reduced bandwidth after conjugate
multiplication, making it more tolerant to kernel choice. **Pick one and
use it consistently across the stack** — mixing M1 and M2 within a stack
introduces path-inconsistent bias.

## References

- Hanssen, R. F., & Bamler, R. (1999). *Evaluation of interpolation kernels
  for SAR interferometry.* IEEE Transactions on Geoscience and Remote
  Sensing, 37(1), 318-321.
  [DOI: 10.1109/36.739168](https://doi.org/10.1109/36.739168) — canonical
  peer-reviewed evaluation of nearest, bilinear, 4-/6-point cubic, and
  truncated sinc kernels for SAR interferometry.

- GAMMA Remote Sensing (2017). *Interpolation and resampling — Technical
  Report.* [PDF](https://www.gamma-rs.ch/uploads/media/2017-2_TR_Interpolation_and_resampling.pdf) —
  recommends B-spline order 4-5 for MLI/DEM (partition-of-unity), Lanczos
  order 4+ for SLC, FFT-based resampling for sub-pixel coregistration,
  and interpolation on `sqrt(intensity)` rather than intensity.

- Slacikova, R., et al. (2011). *The effect of resampling methods on the
  interferogram phase quality and InSAR DEM accuracy.* EARSeL Symposium
  2011, Prague.
  [PDF](https://www.earsel.org/symposia/2011-symposium-Prague/Proceedings/PDF/Radar%20Remote%20Sensing/63%20ok25-a2521-slacikova.pdf) —
  empirical ERS-tandem comparison showing 6-point cubic convolution is
  statistically indistinguishable from truncated sinc for DEM generation
  at 40 m resolution.

- ESA SNAP Toolbox. *`MultilookOp` documentation.*
  [Link](https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.sar.processing.ui/operators/MultilookOp.html) —
  confirms multilook = "space-domain averaging of a single look image
  with a small sliding window" and warns against averaging phase alone.

- Kvasniy, S., et al. (2022). *Interpolation Methods with Phase Control
  for Backprojection of Complex-Valued SAR Data.* Sensors, 22(13), 4941.
  [DOI: 10.3390/s22134941](https://doi.org/10.3390/s22134941) — phase-control
  procedure extending linear/cubic/sinc interpolators for complex SAR
  backprojection, especially at THz frequencies.

- Internal skill: `~/.agents/skills/sar-resampling-kernels/SKILL.md` and
  `references/kernel_evidence.md` — full evidence chain with ISCE2 source
  citations and the failure-mode case study.

# Coregistration

## Intuition

An interferogram compares the complex return from the **same ground
scatterers**. Equal array indices in two SLCs generally observe different
points before registration. Coregistration estimates a mapping from reference
coordinates to secondary coordinates and resamples the secondary SLC onto the
reference or common geographic grid.

The geometric objective is shared across missions. A mission adapter can add a
stricter phase-sensitive constraint; for example, Sentinel-1 TOPS uses ESD to
refine azimuth timing.

### Ampcor executor contract

Torch Ampcor is an explicit, fail-closed opt-in. Its source-input allowlist is
limited to C-contiguous ``complex64`` and ``float32`` arrays; ``complex128``
and ``float64`` inputs remain supported by the permissive NumPy executor.
Switching back to the portable path requires ``executor="numpy"`` together
with ``device="cpu"`` or ``device="auto"``. A contradictory request such as
``executor="numpy", device="cuda"`` is rejected before tile or device work.

The CUDA path requires a visible CUDA device and Torch. Recorded parity and
speedup claims in the linked benchmark artifacts apply to the A100 fixture
that produced them; they are not a runtime admission filter. Callers that
require NumPy-equivalent cull membership must select NumPy. The boundary
oracle uses the existing NumPy/SciPy runtime and can reduce the CUDA speedup
for boundary-heavy workloads. The measured production claim is eager CUDA;
``torch.compile`` is only a fixed-shape audit path.

The deprecated ``stage_coregister`` route keeps its historical default
``search_radius=32``. The portable rollback is Ampcor-only: selecting
``executor="numpy"`` does not change the downstream phase-preserving resampler,
which remains Torch-owned and follows its own device policy.

## Mapping model

For a reference sample $(a,r)$, geometry predicts the ground point

$$
\mathbf{x}=\operatorname{rdr2geo}_1(a,r;\mathcal{O}_1,h),
$$

then maps it to the secondary radar grid

$$
(a_2,r_2)=\operatorname{geo2rdr}_2(\mathbf{x};\mathcal{O}_2).
$$

The offset field is

$$
\Delta a(a,r)=a_2-a,\qquad \Delta r(a,r)=r_2-r.
$$

The coregistered secondary is evaluated with a complex, phase-preserving
interpolator:

$$
\widetilde{s}_2(a,r)=
\sum_{m,n}s_2[m,n]h_a(a_2-m)h_r(r_2-n).
$$

The kernels $h_a,h_r$ are normally windowed sinc or a high-quality Lanczos
approximation. If a mission has a strong steering or focusing carrier, its
adapter transforms the SLC to a suitable baseband before interpolation and
restores the carrier at the mapped output coordinates.

## Common registration levels

### 1. Geometric registration

Orbit and DEM range-Doppler transforms provide a dense, smooth offset field.
This handles large shifts and terrain parallax and supplies the initial mapping
for every pixel.

### 2. Amplitude-correlation refinement

Patches of $|s_1|$ and $|s_2|$ are cross-correlated. For normalized
cross-correlation,

$$
\rho(\Delta a,\Delta r)=
\frac{\sum (A_1-\bar A_1)(A_2^{\Delta}-\bar A_2^{\Delta})}
{\sqrt{\sum(A_1-\bar A_1)^2\sum(A_2^{\Delta}-\bar A_2^{\Delta})^2}}.
$$

The peak is oversampled for subpixel offsets, outliers are rejected using SNR
or robust residuals, and a polynomial or spline correction is fitted.
Amplitude correlation may be unreliable over water, snow, vegetation, or long
temporal baselines.

### 3. Phase-sensitive mission refinement

Amplitude correlation does not measure every phase bias. Scanning modes,
segmented acquisitions, or frequency-dependent products can add constraints
that must be estimated before final resampling. Keep these estimators behind a
mission-aware interface and combine their correction with the shared offset
field. Sentinel-1's ESD equations and acceptance tests are documented in
[TOPS processing](../missions/sentinel1/tops.md).

## Registration and coherence

For an ideal band-limited signal, a fractional offset reduces correlation by
the normalized autocorrelation of the impulse response. Useful diagnostics
therefore include both correlation-peak SNR and residual offset relative to the
signal bandwidth. A mission-specific carrier or overlap constraint can be
stricter than the amplitude-coherence constraint.

## Quality-control gates

- Geometry offsets are smooth and mapped coordinates remain inside valid
  source support.
- Correlation tie points cover range and azimuth, not only one high-contrast
  corner.
- Robust-fit residuals and rejected-point fractions are reported.
- Range and azimuth residuals are evaluated separately.
- Required mission-specific phase residuals are centered near zero after their
  correction.
- Final interferometric coherence improves; a sharp amplitude image alone is
  not proof of phase registration.

## Common mistakes

- Using bilinear interpolation on full-band complex SLCs.
- Applying one offset everywhere when terrain creates a spatially varying map.
- Estimating only amplitude offsets when the mission requires phase-sensitive
  refinement.
- Fitting a flexible polynomial to sparse or noisy tie points without
  validation.
- Resampling reference and secondary repeatedly instead of combining transforms
  so each complex SLC is interpolated once.

See the existing [resampling guide](../../processing/resampling.md) for kernel
selection by physical data type.

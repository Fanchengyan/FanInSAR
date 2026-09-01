# TOPS Processing

## Intuition

Sentinel-1 IW uses Terrain Observation by Progressive Scans (TOPS). During each
burst the antenna beam sweeps in azimuth. The focused SLC therefore contains a
strong, range-dependent azimuth phase carrier. That carrier is valid signal,
but it makes fractional-pixel interpolation unsafe unless it is temporarily
removed.

**Deramp** removes the modeled carrier, interpolation/coregistration acts on the
baseband signal, and **reramp** restores the carrier at the output coordinates.

## Carrier model

A common form of the TOPS azimuth carrier is

$$
\phi_c(\eta,\tau)
= -\pi k_t(\tau)[\eta-\eta_{\mathrm{ref}}(\tau)]^2
-2\pi f_{dc}(\tau)\eta,
$$

where $k_t$ is the effective Doppler/FM rate, $f_{dc}$ is Doppler centroid,
$\eta$ is azimuth time relative to the burst, and $\tau$ is range time. Exact
signs and reference times differ among processors. The safe implementation rule
is a round trip:

$$
s_d=s\,e^{-j\phi_c},\qquad
s_{out}=\mathcal{I}(s_d)\,e^{+j\phi_c(\eta_{out},\tau_{out})}.
$$

If a package defines $\phi_c$ with the opposite sign, the two exponent signs
reverse together.

The effective steering rate is often derived from the azimuth FM rate $k_a$ and
the scanning rate $k_s$:

$$
k_t=\frac{k_a k_s}{k_a-k_s}.
$$

Both $k_a$ and $f_{dc}$ vary with range and are represented by annotation
polynomials.

## Why ordinary complex interpolation is not enough

Suppose a carrier has instantaneous azimuth frequency $f_c$. A residual
azimuth shift $\Delta\eta$ introduces approximately

$$
\Delta\phi \approx 2\pi f_c\Delta\eta.
$$

TOPS sweeps across a wide Doppler band, so a tiny timing error can become a
large phase discontinuity. Interpolating the ramped signal with a short kernel
also violates the kernel's band-limited assumption. Deramping reduces the local
bandwidth; reramping preserves the physically correct phase after resampling.

## TOPS azimuth refinement with ESD

The shared [coregistration](../../processing/coregistration.md) stage estimates
the geometric mapping, but Sentinel-1 TOPS normally needs a stricter azimuth
timing refinement. Adjacent bursts image their overlap with different Doppler
centroids. Form overlap interferograms from the two looks and take their double
difference:

$$
\Delta\phi_{\mathrm{ESD}}
=\arg\!\left(I_{\mathrm{upper}}I_{\mathrm{lower}}^*\right)
\approx 2\pi\Delta f_{\mathrm{DC}}\Delta\eta.
$$

Therefore

$$
\Delta\eta=
\frac{\Delta\phi_{\mathrm{ESD}}+2\pi k}
{2\pi\Delta f_{\mathrm{DC}}},
$$

where the integer ambiguity $k$ must be chosen consistently. Coherence masks,
multilooking, robust circular statistics, or a network over acquisitions reduce
noise. The correction updates secondary azimuth timing before final
phase-preserving resampling. Network ESD is described by
[Fattahi et al. (2017)](https://doi.org/10.1109/TGRS.2016.2614925).

## Implementation sequence

1. Parse burst start time, azimuth interval, range sampling, azimuth FM-rate
   polynomial, Doppler centroid polynomial, and steering rate.
2. Evaluate the carrier at **pixel centers** using one documented index origin.
3. Multiply by the conjugate carrier to deramp.
4. Interpolate the deramped complex SLC to the target radar or geographic grid.
5. Evaluate the acquisition-specific carrier at the source coordinates sampled
   by each output pixel and reramp.
6. Mask invalid source support and burst-edge zero padding.
7. Where required, estimate the ESD residual and update the secondary azimuth
   timing before final resampling.

:::{important}
Reference and secondary bursts have slightly different timing and polynomial
parameters. Do not reuse the reference carrier for the secondary acquisition.
:::

## Tests that catch real bugs

- **Identity transform:** deramp → identity resample → reramp reproduces the
  original valid SLC to numerical tolerance.
- **Known fractional shift:** a synthetic point target retains amplitude and
  phase after deramp/resample/reramp.
- **Pixel-origin test:** switching between 0-based indices and 0.5-centered
  coordinates does not introduce a constant or linear phase ramp.
- **Burst continuity:** overlap phase after coregistration has no deterministic
  burst-dependent range ramp.
- **Metadata sensitivity:** perturbing FM-rate/Doppler coefficients produces the
  expected phase error, proving they are actually used.

## Common mistakes

- Interpolating the wrapped phase or the ramped SLC directly.
- Applying deramp without matching reramp.
- Reramping at output array indices rather than mapped source radar coordinates.
- Mixing seconds, lines, PRF, and zero/half-based pixel centers.
- Treating deramp as a calibration correction and permanently removing it.

The carrier is a signal-processing coordinate transform. It is not the
flat-Earth or topographic phase correction discussed later.

After shared interferometric processing, use
[burst and subswath merge](burst_merge.md) when products must be assembled.

# Multilooking, Coherence, and Filtering

## Intuition

A single-look interferogram preserves the highest resolution but contains
strong speckle and noisy phase. Multilooking trades spatial resolution for a
more stable complex estimate. Coherence describes local phase consistency.
Adaptive filtering suppresses broadband phase noise so unwrapping sees fewer
false residues.

These are three distinct operations. Calling all of them "smoothing" leads to
incorrect kernels and misleading quality measures.

## Complex multilooking

For a look window $\Omega$ with weights $w_p$,

$$
\bar z=\frac{\sum_{p\in\Omega}w_p s_{1,p}s_{2,p}^*}
{\sum_{p\in\Omega}w_p},\qquad
\phi_{ML}=\arg(\bar z).
$$

Average the complex interferogram, not its angles. The number of independent
looks is smaller than the pixel count when samples are oversampled or the
window overlaps.

Choose range and azimuth looks to produce a useful ground-resolution aspect
ratio, not merely a square array block. Window support must exclude invalid
mission-segment edges and no-data samples.

## Coherence

The sample complex coherence is

$$
\hat\gamma=
\frac{\sum_{p\in\Omega}w_p s_{1,p}s_{2,p}^*}
{\sqrt{\left(\sum w_p|s_{1,p}|^2\right)
\left(\sum w_p|s_{2,p}|^2\right)}}.
$$

Its magnitude lies in $[0,1]$ in exact arithmetic. It measures local complex
similarity, not correctness. High coherence can accompany an orbit ramp or an
incorrect topographic model; low coherence can be caused by temporal change,
thermal noise, volume scattering, geometry, registration error, or insufficient
looks.

Always record the coherence window, weights, edge policy, and whether the SLCs
were filtered or multilooked first. Coherence estimates with different support
are not directly comparable.

## Goldstein adaptive spectral filter

In overlapping interferogram patches, let $Z(u,v)$ be the 2-D Fourier
transform. A common Goldstein-family filter is

$$
Z_f(u,v)=Z(u,v)\left[S\{|Z(u,v)|\}\right]^\alpha,
$$

where $S$ smooths the spectral magnitude and $\alpha\in[0,1]$ controls filter
strength. The inverse transform is overlap-added to form the filtered complex
interferogram. Strong spectral peaks associated with coherent fringes are
retained relative to broadband noise. See
[Goldstein and Werner (1998)](https://doi.org/10.1029/1998GL900033).

Implementations differ in exponent normalization, spectral smoothing, patch
size, overlap, taper, and whether $\alpha$ is fixed or coherence-adaptive.
"Goldstein filter" is therefore not a complete reproducibility statement.

## Other useful filters

- **Boxcar/Gaussian complex filters:** predictable low-pass behavior, but blur
  discontinuities and narrow deformation gradients.
- **Non-local filters:** exploit similar patches and can retain resolution, but
  are computationally expensive and can introduce selection bias.
- **Common-band filters:** remove non-overlapping SLC spectrum before
  cross-multiplication; they solve a different problem.
- **Median filters on phase:** generally unsafe because wrapped angles are
  circular; use circular/complex statistics if required.

## Ordering

A typical pair workflow forms a complex interferogram, multilooks and estimates
coherence from matched support, then phase-filters the complex product before
unwrapping. Some processors filter at single-look resolution and look later.
Whichever order is used, retain an unfiltered product and document the exact
support so science products can be audited.

## Quality-control gates

- Coherence is finite and within $[0,1]$ after numerical clipping only at tiny
  round-off excursions.
- A self-pair gives coherence near one on valid support.
- Low-coherence water or changed vegetation behaves plausibly.
- Filtering reduces phase-gradient outliers/residues without shifting broad
  fringes or erasing known discontinuities.
- The filtered phase rewraps consistently and no tile seams are introduced.
- Looks, filter window, overlap, exponent, masks, and output spacing are stored.

## Common mistakes

- Averaging phase angles instead of complex phasors.
- Resampling an already estimated coherence field and treating it as a new
  coherence estimate.
- Using coherence as a universal probability of correctness.
- Filtering so aggressively that deformation gradients or fault discontinuities
  disappear.
- Reporting nominal pixel count as the number of independent looks.

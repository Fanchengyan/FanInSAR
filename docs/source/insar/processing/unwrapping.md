# Phase Unwrapping

## Intuition

The complex interferogram gives phase only modulo $2\pi$. Phase unwrapping
chooses an integer number of cycles at every valid pixel:

$$
\phi_u(p)=\phi_w(p)+2\pi k(p),\qquad k(p)\in\mathbb{Z}.
$$

This is not a simple `numpy.unwrap` operation in two dimensions. Noise, masks,
true discontinuities, steep gradients, and disconnected regions make many
integer fields locally plausible.

## Wrapped gradients and residues

Define the wrapped difference along an edge $(p,q)$ as

$$
g_{pq}=\mathcal{W}[\phi_w(q)-\phi_w(p)].
$$

In a noise-free, sufficiently sampled field, summing these gradients around a
closed loop gives zero. A non-zero multiple of $2\pi$ is a phase **residue**.
Residues reveal inconsistent local gradients caused by noise, undersampling,
decorrelation, or true discontinuities. An unwrapper must decide where to place
discontinuities or how to distribute correction globally.

## Main algorithm families

### Path following and branch cuts

Connect positive and negative residues with branch cuts, then integrate without
crossing those cuts. These methods are intuitive and fast but sensitive to
residue pairing and masks.

### Minimum-cost flow / statistical network methods

Represent integer corrections or phase gradients as flows on a graph and
minimize a cost. SNAPHU formulates a statistical maximum-a-posteriori problem
with topography, deformation, and smooth-surface modes, initialized with MST or
MCF and refined by network optimization. See the
[official SNAPHU description](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/).

### Least squares and robust IRLS

Solve for a phase surface whose gradients best match wrapped observations:

$$
\min_{\phi_u}\sum_{(p,q)}w_{pq}
\rho\!\left(\phi_u(q)-\phi_u(p)-g_{pq}\right).
$$

With $\rho(e)=e^2$, this becomes a weighted Poisson/least-squares problem that
can be solved efficiently with DCT/FFT methods on regular grids. Iteratively
reweighted least squares (IRLS) approximates an $L_1$ or other robust loss by
repeated weighted least-squares solves. It suppresses isolated bad gradients
but must still enforce congruence and handle disconnected components carefully.

## Masks and connected components

Low-coherence masking removes unreliable edges but may split the image. Each
connected component then has an independent additive $2\pi n$ ambiguity. Store
component labels and link components only with defensible overlap, reference
points, or external constraints. A seamless color map created by arbitrary
component shifts is not scientific evidence.

## Preparing the interferogram

Before unwrapping:

1. remove flat-Earth and intended topographic phase;
2. correct any mission-specific segment or mosaic phase offsets;
3. estimate coherence on documented support;
4. filter enough to reduce noise without erasing discontinuities;
5. mask invalid, layover/shadow, or unusably incoherent pixels;
6. choose topography, deformation, or smooth cost assumptions consistently.

## Validation

### Rewrap consistency

The most basic invariant is

$$
\mathcal{W}(\phi_u)-\phi_w\approx0
$$

on every valid pixel. This catches format, sign, and gross congruence errors but
cannot prove that $k(p)$ is correct.

### Independent checks

- compare connected components and residue density;
- inspect phase-gradient residuals, not only phase values;
- test pair-loop closure after unwrapping;
- compare against a second algorithm (for example SNAPHU versus robust IRLS);
- evaluate known stable reference areas and external GNSS if available;
- inspect discontinuities across faults, coastlines, water, and acquisition
  seams;
- perturb mask/filter parameters and check solution stability.

## Common mistakes

- Treating unwrapping as noise removal or atmospheric correction.
- Using `np.unwrap` row-by-row on a 2-D interferogram.
- Dropping connected-component labels.
- Choosing a high coherence threshold that disconnects the image, then silently
  assigning component offsets.
- Using a smoothness prior across a real coseismic rupture.
- Validating only by rewrap consistency; a wrong integer-cycle field also
  rewraps perfectly.

# Burst and Subswath Merge

## Intuition

Sentinel-1 IW coverage is segmented into bursts and three subswaths. Adjacent
pieces overlap, but "just averaging" them is safe only after their geometry and
phase references agree. A seam can originate in coregistration, carrier
handling, simulated phase, unwrap ambiguity, or mosaicking. Blending weights
cannot repair a wrong phase model.

## Align first, merge second

For two overlapping wrapped products $z_i$ and $z_j$, estimate the overlap
phase difference with circular statistics:

$$
\widehat{\Delta\phi}_{ij}
=\arg\left(\sum_{p\in\Omega_{ij}}w_p z_i(p)z_j^*(p)\right).
$$

For multiple bursts, solve a network problem

$$
c_j-c_i\approx\widehat{\Delta\phi}_{ij}
$$

with one reference node fixed. Robust methods reject low-support/outlier rows,
handle disconnected components, and may include a range ramp
$c_i(x)=m_ix+b_i$. After applying corrections, compute residual overlap phase
before mosaicking.

For unwrapped products, overlap differences can include integer $2\pi$ cycle
offsets. Resolve those component labels before arithmetic blending.

## Reference-system strategies

| System | Phase preparation | Overlap merge |
|---|---|---|
| ISCE2 | ESD corrects secondary azimuth timing before burst interferograms | `top`, `bot`, or $0.5(z_1+z_2)$; normal VRT path is source-order selection |
| InSAR.dev | overlap network fit; optional three-step offset → range-ramp → offset | equal/fractional arithmetic mean; circular mean for wrapped phase |
| GMTSAR | geometry/xcorr and optional spectral diversity | hard midpoint cut for along-track TOPS stitch |
| ISCE3 | no Sentinel-1 TOPS burst workflow in the pinned source | no Sentinel-1 burst merge implementation |

These statements are verified against the source snapshots listed in the
[software guides](../../software/index.md). None of the four reference systems uses a
Hann/cosine feather as its standard Sentinel-1 burst-overlap solution.

## Complex versus phase mosaics

For coregistered complex interferograms, equal-weight merging is

$$
z_{out}=\frac{\sum_i w_i z_i}{\sum_i w_i},
\qquad \phi_{out}=\arg(z_{out}).
$$

For wrapped angle arrays, first map to the unit circle:

$$
\phi_{out}=\arg\left(\sum_iw_i e^{j\phi_i}\right).
$$

For unwrapped phase, arithmetic averaging is valid only after integer-cycle
and continuous ramp alignment. Coherence can be used as a quality mask or
carefully defined weight, but weighting changes estimator support and must be
recorded.

## Burst validity and multilook boundaries

Burst zero padding and invalid-sample masks must never enter the merge. In a
radar-domain processor, valid boxes may be snapped to multilook boundaries so a
look window does not mix two bursts with different carrier histories. In a
geocoded processor, all overlapping bursts must share exactly compatible map
coordinates or be reindexed with explicit no-data handling.

## Quality-control gates

- Per-edge overlap pixel count, coherence, fitted offset/ramp, and robust
  residual are recorded.
- The overlap graph is connected or disconnected components are explicitly
  identified.
- Phase residuals are checked **before** applying any visual blend.
- No-data footprints and valid sample masks survive reprojection.
- Wrapped merges use complex/circular statistics.
- Unwrapped merges pass a rewrap test and have no unresolved $2\pi$ jumps.
- Seam diagnostics compare gradients inside and outside overlap, not only a
  color-stretched image.

## Common mistakes

- Inventing a smooth feather to hide a carrier or registration error.
- Averaging wrapped phase as ordinary real values.
- Unwrapping each burst independently and merging without component linking.
- Assuming ISCE2's seam quality comes from sophisticated merge weights; its ESD
  and final SLC resampling do the critical alignment work.
- Claiming an ISCE3 Sentinel-1 merge algorithm when the project does not provide
  that workflow.

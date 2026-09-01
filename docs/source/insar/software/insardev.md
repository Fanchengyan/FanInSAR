# InSAR.dev: geocode-first, per-burst processing

## Scope and source snapshot

This page describes the InSAR.dev core repository at commit
[`190ea67`](https://github.com/InSARdev/core/tree/190ea67e9a3a0f3e3c2e3b794657b05dfe4bd1db).
The ecosystem is split into:

- `insardev_pygmtsar` for Sentinel-1 SLC preprocessing and geocoded Zarr;
- `insardev` for interferograms, filtering, burst alignment/merge, unwrapping,
  and time-series analysis;
- `insardev_toolkit` for data, orbit, and DEM access.

The current design is not simply a Python wrapper around the legacy GMTSAR
pair pipeline. It processes bursts independently onto a common geographic grid.

## Actual tutorial path

The repository's Sentinel-1 notebooks follow this shape:

1. download/select Sentinel-1 bursts, orbit files, and DEM;
2. construct `S1`, choose a reference date, and call `transform(...)` to write
   per-burst geocoded SLC Zarr;
3. load with `Stack().load(...)` and optionally align stack elevation/burst
   phase;
4. choose pairs and call `interferogram(...)` with Gaussian look/filter support;
5. optionally apply Goldstein filtering;
6. align overlapping burst interferograms and `dissolve()` them;
7. unwrap with DCT+IRLS or SNAPHU and convert to LOS displacement.

## Sentinel-1 preprocessing and coregistration

[`S1_align.py`](https://github.com/InSARdev/core/blob/190ea67e9a3a0f3e3c2e3b794657b05dfe4bd1db/insardev_pygmtsar/insardev_pygmtsar/S1_align.py)
uses orbit/DEM radar-to-geographic geometry to estimate reference-to-repeat
offsets and fits a bilinear offset model. Optional amplitude xcorr refines
**range**, while azimuth remains orbit-based in the pinned code to avoid noisy
TOPS range-dependent reramp slopes.

The repeat SLC is deramped but not independently shifted and reramped at this
stage. Alignment and geocoding are fused in
[`S1_transform.py`](https://github.com/InSARdev/core/blob/190ea67e9a3a0f3e3c2e3b794657b05dfe4bd1db/insardev_pygmtsar/insardev_pygmtsar/S1_transform.py):

1. compute a merged output→secondary-radar transform;
2. remap the deramped complex SLC once to the projected grid;
3. analytically reramp using the repeat burst's own FM-rate/Doppler parameters;
4. apply flat-Earth/topographic phase and optional tidal/radiometric corrections;
5. store complex data and geometry in per-burst Zarr groups.

This is the geocode-first counterpart to ISCE2's radar-domain `Geo2rdr` +
`Resamp_slc` path.

## Interferogram, looks, and coherence

[`Batch.interferogram`](https://github.com/InSARdev/core/blob/190ea67e9a3a0f3e3c2e3b794657b05dfe4bd1db/insardev/insardev/Batch.py#L2603-L2679)
computes

$$
z=s_{ref}s_{rep}^*.
$$

An optional complex phase model can be multiplied/subtracted. When a wavelength
parameter is supplied, Gaussian spatial filtering is applied to the
interferogram and both powers, then coherence is computed as

$$
\gamma=\frac{|\langle z\rangle_G|}
{\sqrt{\langle|s_{ref}|^2\rangle_G
\langle|s_{rep}|^2\rangle_G}}.
$$

An optional Goldstein filter is implemented separately and can use correlation
as a weight.

## Burst phase alignment

Per-pair burst-interferogram alignment is implemented by
[`BatchCore.fit/align`](https://github.com/InSARdev/core/blob/190ea67e9a3a0f3e3c2e3b794657b05dfe4bd1db/insardev/insardev/BatchCore.py#L5395-L6054).
For overlap pairs it computes row-wise phase statistics, rejects outliers with a
median absolute deviation threshold, estimates weighted offsets, optionally
fits a range ramp, and solves per-connected-component burst corrections with a
sparse incidence matrix and `scipy.sparse.lsqr`.

`align(degree=1)` is a three-step procedure:

1. fit and remove constant offsets;
2. fit and remove range ramps;
3. refit residual constant offsets and combine corrections.

This is an overlap-network alignment of already geocoded products. It is not
the same estimator as ISCE2's Doppler-based per-pair ESD timing correction.
`Stack.align()` also contains a stack-level interferometric double-difference
method for decomposing burst jumps across dates.

## Dissolve/merge

[`BatchCore.dissolve`](https://github.com/InSARdev/core/blob/190ea67e9a3a0f3e3c2e3b794657b05dfe4bd1db/insardev/insardev/BatchCore.py#L6056-L6120)
finds spatially overlapping burst datasets. With no explicit weight it uses
equal weights; an optional fractional current-burst weight divides the remainder
among neighbors. Wrapped phase is merged with a circular mean, while unwrapped
phase, correlation, and scalar fields use arithmetic means. There is no
Hann/cosine feather.

## Unwrapping

The default native route in
[`Stack_unwrap2d.py`](https://github.com/InSARdev/core/blob/190ea67e9a3a0f3e3c2e3b794657b05dfe4bd1db/insardev/insardev/Stack_unwrap2d.py)
is a PyTorch DCT-initialized, correlation-weighted IRLS solver approximating an
$L_1$ gradient objective. It can use CPU, CUDA, or Apple MPS, unwrap connected
components independently, return labels, or link components with globally
optimized integer $2\pi$ offsets. The same module includes a SNAPHU wrapper.

## What to learn from InSAR.dev

- Fuse alignment and geocoding to avoid double interpolation.
- Store per-burst complex SLCs on an explicit common grid.
- Separate overlap phase alignment from overlap averaging.
- Use circular statistics for wrapped phase and arithmetic statistics only for
  already aligned unwrapped/scalar fields.
- Treat connected-component linking as an integer optimization problem with
  explicit labels and diagnostics.

## Limitations and cautions

- This source snapshot is under active development; pin versions and re-check
  notebook/source behavior.
- The geocode-first route makes output grid, remap kernel, phase correction, and
  coordinate agreement scientifically consequential.
- Source-available `insardev` and BSD `insardev_pygmtsar` have different license
  terms; algorithm study does not imply code-copy permission.
- Pair-level overlap alignment and stack-level ESD-like alignment must not be
  described as the same algorithm.


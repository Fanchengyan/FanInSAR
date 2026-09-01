# GMTSAR: Sentinel-1 TOPS with geometry grids and GMT rasters

## Scope and source snapshot

This page describes GMTSAR commit
[`46143a9`](https://github.com/gmtsar/gmtsar/tree/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f).
GMTSAR combines compiled SAR programs with C-shell workflows and Generic
Mapping Tools rasters. Its architecture is radar-domain and file-oriented.

The official project background and citations are listed in the
[GMTSAR documentation](https://gmtsar.github.io/documentation/Citation_and_Funding_Information.html).

## Main Sentinel-1 TOPS routes

For stacks, `preproc_batch_tops.csh` or `preproc_batch_tops_esd.csh` prepares
and aligns acquisitions. `intf_tops.csh` loops over pairs to form/filter
interferograms, optionally unwrap with SNAPHU, and geocode. For full frames,
`p2p_S1_TOPS_Frame.csh` and `merge_unwrap_geocode_tops.csh` coordinate the
subswaths.

The core path is:

1. parse annotation/TIFF with `make_s1a_tops` and install precise orbit data;
2. estimate geometry-based range/azimuth offsets from a DEM;
3. fit offset surfaces, generate aligned SLCs, and optionally refine with ESD;
4. stitch along-track frames/bursts as required;
5. create radar-coordinate topography and projection lookup data;
6. run `phasediff` with optional `topo_ra.grd` removal;
7. Gaussian-filter/decimate, estimate correlation, and Goldstein-filter phase;
8. merge subswath grids with hard stitch positions;
9. unwrap with SNAPHU and geocode with `trans.dat`.

## Geometry and coregistration

[`align_tops.csh`](https://github.com/gmtsar/gmtsar/blob/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f/gmtsar/csh/align_tops.csh)
downsamples/filter the DEM, maps DEM points into master and repeat radar
coordinates using `SAT_llt2rat`, differences the coordinates, block-medians the
offset samples, and fits regular GMT surfaces for range/azimuth shifts. Those
grids are passed to `make_s1a_tops`, after which `resamp` handles remaining
integer/fractional alignment represented in PRM parameters.

The ESD stack route
[`preproc_batch_tops_esd.csh`](https://github.com/gmtsar/gmtsar/blob/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f/gmtsar/csh/preproc_batch_tops_esd.csh)
creates upper/lower sub-aperture products with `make_s1a_tops` mode 2 and calls
`spectral_diversity` to estimate residual azimuth shift. This is optional and
distinct from ordinary amplitude cross-correlation.

## Stitching

Along-track
[`stitch_tops.c`](https://github.com/gmtsar/gmtsar/blob/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f/preproc/S1A_preproc/src_stitch/stitch_tops.c)
uses amplitude cross-correlation for a small integer offset and writes data up
to a hard midpoint between overlapping frames. There is no phase-weighted or
cosine feather.

Cross-subswath
[`merge_swath.c`](https://github.com/gmtsar/gmtsar/blob/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f/preproc/S1A_preproc/src_stitch/merge_swath.c)
selects a hard stitch column from PRM/range geometry or the valid-data boundary.
`merge_unwrap_geocode_tops.csh` applies this to filtered phase, correlation, and
mask grids before unwrapping/geocoding the merged product.

## Interferogram and topographic phase

[`intf.csh`](https://github.com/gmtsar/gmtsar/blob/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f/gmtsar/csh/intf.csh)
updates baseline metadata then calls the compiled `phasediff` program. Passing
`-topo topo_ra.grd` subtracts simulated radar-coordinate topographic phase
during interferogram formation. `-model` can similarly remove a supplied phase
model.

This is GMTSAR's expression of

$$
z_{corr}=s_{ref}s_{rep}^*e^{-j\phi_{topo/model}}.
$$

## Filtering and coherence

[`filter.csh`](https://github.com/gmtsar/gmtsar/blob/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f/gmtsar/csh/filter.csh)
does more than one operation:

- builds wavelength-dependent Gaussian filters and decimation factors;
- filters the two amplitude/power supports;
- filters real and imaginary interferogram grids;
- computes correlation from filtered complex magnitude and amplitudes;
- forms wrapped phase with `atan2`;
- calls `phasefilt` with 32-pixel patches for Werner/Goldstein filtering.

Consequently GMTSAR's `filter` parameter is a physical wavelength and output-
resolution control, while `phasefilt` is the later adaptive spectral phase
filter. They should not be collapsed into one generic smoothing parameter.

## Unwrapping and geocoding

[`snaphu.csh`](https://github.com/gmtsar/gmtsar/blob/46143a94f222aaca9b42a3bf7ea5f48ba241ff9f/gmtsar/csh/snaphu.csh)
masks by correlation/land options, writes binary phase/correlation input, and
runs SNAPHU. A zero maximum-discontinuity setting selects smooth mode; otherwise
the script uses deformation mode with configured `DEFOMAX_CYCLE`. It writes an
unwrapped grid and can preserve connected-component output during the run.

`geocode.csh` uses `trans.dat`, created by mapping DEM points through
`SAT_llt2rat`, to transform radar grids into geographic products.

## What to learn from GMTSAR

- Geometry-offset grids and phase-model grids are inspectable first-class
  artifacts.
- Physical filter wavelength and raster decimation are separate controls.
- Simple hard seams can work when alignment and valid footprints are correct.
- Shell workflows make intermediate products easy to inspect but require
  disciplined filename, PRM, grid-registration, and sign provenance.

## Limitations and cautions

- Multiple scripts cover pair, stack, ESD, and full-frame cases; do not combine
  their branches into one imaginary universal sequence.
- GMT grid registration/axis flips and PRM shift fields are algorithmically
  significant.
- A hard stitch does not estimate phase continuity. Verify overlap residuals
  before and after merging.
- GMTSAR bundles/calls SNAPHU, but SNAPHU is an independent algorithm with its
  own statistical assumptions and configuration.


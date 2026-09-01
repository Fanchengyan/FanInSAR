# ISCE2 `topsApp`: Sentinel-1 TOPS in the radar domain

## Scope and source snapshot

This page describes ISCE2 commit
[`a492b8d`](https://github.com/isce-framework/isce2/tree/a492b8d76fc91fa82a100458b1714120b0fae090).
The authoritative application driver is
[`applications/topsApp.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/applications/topsApp.py).

`topsApp` is the clearest complete reference among the four projects for a
Sentinel-1 SAFE pair processed burst-by-burst in radar coordinates.

## Actual stage order

The driver's `main()` method executes, in simplified form:

1. preprocess reference and secondary SAFE products;
2. compute baselines and verify the DEM;
3. run reference `topo` geometry;
4. extract burst overlaps;
5. compute coarse offsets and coarse-resample overlaps;
6. form overlap interferograms, prepare ESD, and estimate ESD correction;
7. estimate a residual range correction with amplitude correlation;
8. compute final dense geometry offsets and fine-resample the secondary bursts;
9. form flattened burst interferograms and coherence;
10. merge bursts/subswaths and multilook as configured;
11. Goldstein-Werner filter and estimate phase-sigma coherence;
12. unwrap and geocode.

The concrete call sequence is visible in
[`topsApp.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/applications/topsApp.py#L982-L1051).

## Input and geometry

`runPreprocessor` parses Sentinel-1 TOPS burst metadata and SLCs. `runTopo`
solves the reference radar-to-ground problem with the DEM, generating latitude,
longitude, height, LOS, and shadow/layover products. `runCoarseOffsets` maps
those ground coordinates into the secondary acquisition with `Geo2rdr`, giving
dense range and azimuth offset rasters.

This is the reference-grid mapping derived in the
[geometry chapter](../processing/geometry.md):

$$
(a_1,r_1)\xrightarrow{rdr2geo_1}\mathbf{x}
\xrightarrow{geo2rdr_2}(a_2,r_2).
$$

CPU and GPU geometry paths exist. The algorithmic contract is the same even
when numerical implementation and tiling differ.

## Coarse and fine TOPS coregistration

ISCE2 separates three corrections:

### Geometry offsets

`runCoarseOffsets` computes geometry for burst overlaps, and `runFineOffsets`
computes final per-burst offset rasters. These capture orbit/DEM-dependent
spatial variation.

### ESD azimuth correction

[`runESD.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runESD.py)
forms a coherence mask and estimates azimuth misregistration from overlap
interferogram phase and Doppler-frequency separation. Its implemented core is
equivalent to

$$
\Delta\eta\propto
\operatorname{median}\left(
\frac{\arg(I_{ESD})+2\pi k}{\Delta\omega_D}
\right).
$$

The result is a **single secondary timing correction**, converted to lines and
applied in final resampling. ESD occurs before burst interferograms and merge;
that ordering is why a simple overlap selection/average can still be seamless.

### Residual range correction

[`runRangeCoreg.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runRangeCoreg.py)
uses magnitude-only `Ampcor` patches (64 × 32 windows, 16-pixel searches, 32×
peak oversampling in this snapshot), filters by SNR and plausible magnitude,
then estimates residual range misregistration.

Final resampling uses `Resamp_slc`, geometry offset rasters, timing/range
corrections, and burst carrier polynomials. The secondary is resampled in
complex form with TOPS carrier handling rather than interpolating phase angles.

## Interferogram and flattening

[`runBurstIfg.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runBurstIfg.py)
computes

$$
z=s_{ref}s_{sec}^*
$$

and, when flattening is enabled, multiplies by a phase derived from the final
range-offset raster and wavelength. Thus "interferogram" and "flatten" occur in
one routine. The routine also estimates per-burst coherence and adjusts valid
sample/line support.

## Burst merge

[`runMergeBursts.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runMergeBursts.py)
supports `top`, `bot`, and `avg`; `avg` is exactly

$$
z_{overlap}=0.5(z_{top}+z_{bottom}).
$$

There is no Hann/cosine feather. The virtual merge path selects sources by VRT
ordering. `adjustValidWithLooks` trims/snaps burst validity so multilook windows
do not mix incompatible burst edges.

## Filtering and unwrapping

[`runFilter.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runFilter.py)
applies `goldsteinWerner(alpha=filterStrength)` to the flattened merged complex
interferogram. ICU is then run with unwrapping disabled to generate a phase-sigma
correlation layer.

The default SNAPHU wrapper in
[`runUnwrapSnaphu.py`](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runUnwrapSnaphu.py)
uses deformation cost mode, MST initialization, coherence, effective looks,
wavelength, Earth radius, altitude, and connected-component output. An MCF
initialization variant and ICU/Grass alternatives are available through
factories/configuration.

## What to learn from ISCE2

- Keep radar geometry, TOPS carrier, and ESD corrections explicit.
- A sophisticated blend is not a substitute for phase-accurate coregistration.
- Flattening belongs to the complex phase model and can be fused with IFG
  formation.
- Valid burst support and multilook boundaries are part of correctness.
- Preserve ESD, Ampcor, geometry, coherence, and unwrap diagnostics separately.

## Limitations and cautions

`topsApp` has many configurable branches, including ionosphere processing,
dense offsets, alternate unwrappers, and geocode lists. The sequence above is
the main pair-InSAR path, not a claim that every option executes on every run.
File formats and implicit conventions require careful provenance when results
are compared with geocode-first systems.

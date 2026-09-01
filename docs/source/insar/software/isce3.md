# ISCE3: reusable algorithms and the NISAR InSAR workflow

## Scope and the Sentinel-1 boundary

This page describes ISCE3 commit
[`bdf1f6f`](https://github.com/isce-framework/isce3/tree/bdf1f6fb9175ffe9e53b61839c797e69947dd268).

ISCE3 provides modern CPU/GPU geometry, resampling, cross-multiplication,
geocoding, filtering, and unwrapping components. The complete application
workflow in this source tree is built around **NISAR RSLC products**, not
Sentinel-1 SAFE/TOPS bursts. The pinned tree has no complete Sentinel-1 SAFE
reader, TOPS pair driver, ESD stage, or Sentinel-1 burst merge.

Therefore ISCE3 is included for algorithm learning and component reuse, not as
evidence of a ready-made Sentinel-1 SLC→unwrapped pipeline.

## The actual NISAR workflow order

[`nisar/workflows/insar.py`](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/insar.py)
executes the main path:

1. optional common-band processing;
2. reference `rdr2geo`;
3. secondary `geo2rdr` offset generation;
4. coarse SLC resampling;
5. optional dense offsets, rubbersheeting, and fine resampling;
6. `crossmul` with multilooking, coherence, and optional flattening;
7. optional interferogram filtering;
8. phase unwrapping;
9. optional ionosphere correction;
10. geocode wrapped/unwrapped products;
11. optional troposphere, solid-Earth tide, and baseline products.

This is a strong reference for how modular InSAR stages exchange typed raster
products and how restart/persistence is organized.

## Geometry and registration

ISCE3 exposes iterative range-Doppler `rdr2geo` and `geo2rdr` algorithms. The
NISAR workflow writes range and azimuth offset rasters on the reference radar
grid. Coarse resampling uses those geometric offsets; optional dense offset
correlation and rubbersheeting refine the mapping before a fine resample.

The generic helper
[`rdr2rdr.py`](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/isce3/geometry/rdr2rdr.py)
expresses the same composition used throughout this guide:

$$
(t_1,R_1)\rightarrow\mathbf{x}\rightarrow(t_2,R_2).
$$

To adapt these components to Sentinel-1, an external layer must supply SAFE
parsing, burst timing/valid masks, Doppler/FM-rate carrier models, ESD or an
equivalent TOPS refinement, and burst/subswath assembly.

## SLC resampling

The CPU implementation
[`ResampSlc.cpp`](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/image/ResampSlc.cpp)
loads offset tiles, evaluates carrier/Doppler information, and performs sinc
interpolation of complex SLC chips. It can also apply flattening terms when the
required reference information is provided. CUDA counterparts implement the
same class of operation for GPU execution.

ISCE3 also provides
[`geocode_slc`](https://isce-framework.github.io/isce3/api/python/isce3/geocode/geocode_slc.html),
which maps one or more SLC arrays to a geographic grid with `geo2rdr`, optional
azimuth/range corrections, reramping, and flattening. This is a useful primitive
for a geocode-first architecture, but it is not itself a Sentinel-1 workflow.

## Crossmul, flattening, and coherence

[`crossmul.py`](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/crossmul.py)
configures CPU or CUDA `Crossmul`, Doppler LUTs, range/azimuth looks, oversampling,
and the geo2rdr range-offset raster used for flattening.

The C++
[`Crossmul.cpp`](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/signal/Crossmul.cpp)
uses the range offset to evaluate

$$
\phi_{flat}=\frac{4\pi}{\lambda}\,\Delta R
$$

(represented in the code by range-pixel spacing times range offset),
cross-multiplies the SLCs, multilooks, and writes coherence. Configuration can
enable common-band range and azimuth filters.

## Filtering

[`filter_interferogram.py`](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/filter_interferogram.py)
supports `no_filter`, boxcar, and separable Gaussian filtering in this snapshot.
The default run configuration is `no_filter`. This NISAR workflow does not use
ISCE2's Goldstein-Werner step by default.

## Unwrapping

[`unwrap.py`](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/unwrap.py)
can preprocess/mask/fill wrapped phase and dispatch to:

- **SNAPHU** (default), with smooth/deformation cost, MST/MCF initialization,
  effective looks, masks, tiling, overlaps, and connected components;
- **ICU**, a tree/region-growth style ISCE unwrapper with correlation and phase-
  gradient controls;
- **PHASS**, with correlation thresholds and minimum unwrap areas.

The workflow stores both unwrapped phase and connected-component labels and can
bridge components with optional postprocessing.

## What to learn from ISCE3

- Separate geometry, offsets, resampling, crossmul, filtering, unwrap, and
  geocode through explicit product contracts.
- Offer CPU/GPU implementations behind the same numerical contract.
- Choose interpolation and geocoding by physical data type.
- Make unwrapping algorithm, cost mode, looks, masks, tiling, and connected
  components explicit configuration.
- Do not infer mission support from the existence of generic SAR kernels.

## What a Sentinel-1 adapter still needs

1. SAFE/annotation/calibration/noise parsing;
2. IW burst/subswath inventory and valid-sample masks;
3. TOPS carrier derivation, deramp, and acquisition-specific reramp;
4. overlap construction and ESD/network phase refinement;
5. burst/subswath alignment and mosaic policy;
6. Sentinel-1-specific provenance and acceptance tests.


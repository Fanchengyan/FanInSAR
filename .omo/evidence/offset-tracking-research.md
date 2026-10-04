# Production SAR offset tracking: primary-source evidence

**Scope.** Design evidence relevant to FanInSAR dense SAR offset tracking, gathered from
official ISCE2/ISCE3 source trees and the ESA/Copernicus Sentinel-1 guide. Local source
snapshots used were ISCE2 `a492b8d76fc91fa82a100458b1714120b0fae090` (2025-05-01)
and ISCE3 `bdf1f6fb9175ffe9e53b61839c797e69947dd268` (2026-06-29).

## 1. Correlation contract and parameterization

- ISCE2 `DenseAmpcor` exposes separate reference chip (`windowSizeWidth/Height`) and
  search-window sizes, integer gross range/azimuth offsets, independent range/azimuth
  sampling skips, FFT oversampling, image interpretation (`complex`, `real`, `mag`),
  SNR/covariance thresholds, and an edge margin. It rejects odd chip dimensions,
  non-power-of-two zoom/oversampling factors, and search windows at least twice the
  chip size. [ISCE2 DenseAmpcor.py](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/mroipac/ampcor/DenseAmpcor.py#L81-L121), [#L214-L275](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/mroipac/ampcor/DenseAmpcor.py#L214-L275), [#L620-L648](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/mroipac/ampcor/DenseAmpcor.py#L620-L648).
- The grid is not simply “every N pixels”: ISCE2 computes valid starts from gross
  offsets, search/chip halos, and margins, then records the actual first/last sample
  locations and output shape. A production API should persist grid origin, spacing,
  chip/search geometry, and gross-offset convention with the raster. [ISCE2 DenseAmpcor.py](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/mroipac/ampcor/DenseAmpcor.py#L341-L376).
- ISCE3 makes the two-stage sub-pixel process explicit: correlate over a larger raw
  search range, extract a smaller neighborhood around the coarse maximum, then
  oversample the correlation surface. The final offset is coarse pixel offset plus
  zoomed maximum divided by `(correlation-surface oversampling * raw-data
  oversampling)`. [ISCE3 cuOffset.cpp](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuOffset.cpp#L53-L73), [#L74-L91](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuOffset.cpp#L74-L91).
- Gross offsets may be constant or per-window varying. ISCE3 computes secondary
  starts as reference start + gross offset - half search range, and separately stores
  whether gross offsets are merged into the reported result. [ISCE3 cuAmpcorParameter.cpp](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuAmpcorParameter.cpp#L136-L188), [ISCE3 cuAmpcorParameter.h](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuAmpcorParameter.h#L117-L127).

## 2. Sentinel-1 radar geometry

- For TOPS processing, offsets are estimated on geometry-coregistered/merged SLCs,
  not arbitrary image arrays. ISCE2's TOPS dense-offset runner uses the merged
  reference/secondary SLC, passes PRF/range-spacing scale factors, and uses magnitude
  for complex TOPS SLCs; the GPU path explicitly selects deramping and frequency-domain
  correlation. [ISCE2 runDenseOffsets.py](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runDenseOffsets.py#L32-L130), [#L187-L286](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runDenseOffsets.py#L187-L286).
- Sentinel-1 geocoding obtains the reference burst's azimuth time interval, starting
  slant range, range pixel size, radar wavelength, merged orbit, and right-looking
  geometry. Offset-mode geocoding adjusts the radar-grid origin using the offset grid's
  first row/column and the skip factors. [ISCE2 runGeocode.py](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runGeocode.py#L82-L141).
- The mission-level primary reference is the [ESA/Copernicus Sentinel-1 SAR User
  Guides](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar)
  and the [Sentinel-1 Product Specification](https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Specification),
  which define the SLC radar coordinates, TOPS burst structure, timing, orbit and
  calibration metadata on which those ISCE geometry fields depend. (URLs may redirect
  as the Copernicus documentation portal evolves.)

## 3. Outputs, uncertainty, and quality

- The established dense-offset output packet is more than two displacement bands:
  ISCE2 writes two-channel offsets (azimuth/down, then range/across), one-band SNR,
  and three covariance components (`cov1`, `cov2`, `cov3`); invalid lanes are initialized
  with sentinel offsets and large covariance/zero SNR. [ISCE2 DenseAmpcor.py](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/mroipac/ampcor/DenseAmpcor.py#L384-L400), [#L526-L596](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/mroipac/ampcor/DenseAmpcor.py#L526-L596).
- ISCE3's ROFF product preserves raw, unculled, unfiltered along-track and slant-range
  offsets, SNR, normalized correlation peak, along-track variance, slant-range
  variance, and cross covariance. This is a strong production pattern: retain raw
  observables and publish a separate validity/quality mask rather than irreversibly
  replacing measurements during the correlation step. [ISCE3 ROFF_writer.py](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/products/insar/ROFF_writer.py#L279-L359).
- SNR is computed from the peak squared divided by the mean squared correlation away
  from the peak. Covariance is estimated from the local 3x3 correlation-surface
  curvature; a peak on the surface edge or near-zero Gaussian curvature receives a
  sentinel covariance. Therefore quality logic should test finite values, peak
  interior/curvature validity, SNR, correlation peak, and geometric/image-range
  validity together—not SNR alone. [ISCE3 cuEstimateStats.cu](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuEstimateStats.cu#L20-L35), [#L38-L110](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuEstimateStats.cu#L38-L110).

## 4. Geocoding to physical displacement and rates

- ISCE3 converts slant-range pixel offsets to meters by multiplying by reference
  slant-range pixel spacing. Along-track pixel offsets are converted using the local
  ground-track velocity times the reference zero-Doppler azimuth-time spacing. This
  velocity is a geometry scale factor (m/s), not a deformation rate estimated by
  Ampcor. [ISCE3 offsets_product.py](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/offsets_product.py#L394-L468), [#L471-L523](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/offsets_product.py#L471-L523).
- Covariance must undergo the matching Jacobian transform: along-track variance is
  multiplied by `(ground-track velocity * azimuth-time spacing)^2`, range variance by
  `slant-range spacing^2`, and cross covariance by the product of the two scales.
  Physical displacement and its uncertainty must therefore be transformed together.
  [ISCE3 offsets_product.py](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/offsets_product.py#L526-L624).
- A temporal displacement velocity (for example mm/year) is a downstream multi-date
  estimation problem: use geocoded displacement observations, their covariance/quality
  masks, and acquisition times in a robust temporal model. Do not label the
  ground-track conversion factor itself as deformation velocity.

## 5. CPU/GPU, memory, and lazy execution implications

- ISCE2 selects CPU or GPU based on configuration and availability, but its CPU path
  materializes/opens files and its GPU path requires explicit CUDA device, stream count,
  chunk dimensions, memory-map size, and output interleave. [ISCE2 runDenseOffsets.py](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runDenseOffsets.py#L15-L29), [#L187-L237](https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/components/isceobj/TopsProc/runDenseOffsets.py#L187-L237).
- ISCE3's controller batches windows into chunks and uses CUDA streams to overlap
  transfers and kernels; both CPU and CUDA bindings expose the same parameter surface.
  Memory mapping is a first-class input strategy, not an incidental optimization. [ISCE3 cuAmpcorController.h](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuAmpcorController.h#L1-L12), [ISCE3 cuAmpcorController.cpp](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuAmpcorController.cpp#L33-L89), [#L103-L177](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/cxx/isce3/matchtemplate/pycuampcor/cuAmpcorController.cpp#L103-L177).
- The ISCE3 workflow explicitly uses memory-mappable ENVI copies, configurable
  lines-per-block, window batches, and CUDA streams, then writes HDF5 outputs blockwise.
  A FanInSAR implementation should preserve the same public-call boundary for lazy
  execution: chunk I/O and compute, but return a stable grid plus all quality/uncertainty
  bands, and never silently change device/backend after admission. [ISCE3 dense_offsets.py](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/dense_offsets.py#L40-L124), [#L131-L282](https://github.com/isce-framework/isce3/blob/bdf1f6fb9175ffe9e53b61839c797e69947dd268/python/packages/nisar/workflows/dense_offsets.py#L131-L282).

## Design conclusions for FanInSAR

1. Make the coordinate contract explicit: output axis order, sign, grid origin/spacing,
   chip/search windows, gross-offset merge, units, and Sentinel-1 radar-grid metadata.
2. Treat offsets as a result packet: displacement bands, SNR, correlation peak,
   covariance (including cross term), finite/geometric validity, and provenance.
3. Keep raw measurements immutable; derive filtered products and masks separately.
4. Convert radar pixels to meters with the correct local geometry Jacobian and transform
   covariance with the same Jacobian; estimate temporal velocity only downstream.
5. Validate all windows against both images before ABI/kernel access, including margins,
   gross and varying offsets, and boundary peaks. Chunk/mmap lazily, but make backend,
   device, precision, and failure behavior explicit and reproducible.
6. Use CPU/GPU golden fixtures that compare the complete packet (offsets, SNR, peak,
   covariance, masks), not only offsets or compute-only timings.

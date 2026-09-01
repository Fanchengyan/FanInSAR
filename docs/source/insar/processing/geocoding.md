# Geocoding and Validation

## Intuition

Radar coordinates are indexed by sensing time and slant range. Users normally
need a map grid. Geocoding maps each output map pixel through the DEM and orbit
to the radar grid, samples the appropriate product, and writes an explicit CRS,
transform, resolution, and mask.

## Inverse-map geocoding

For output map coordinate $(x_g,y_g)$:

1. transform to geodetic latitude/longitude;
2. sample ellipsoidal DEM height and form ECEF point $\mathbf{x}$;
3. solve secondary/reference `geo2rdr` for $(t,R)$;
4. convert $(t,R)$ to fractional radar indices;
5. interpolate the radar product using a kernel chosen by physical type.

Inverse mapping avoids holes that occur when source radar pixels are simply
forward-projected.

## Kernel depends on the product

| Product | Appropriate treatment |
|---|---|
| Complex SLC or wrapped interferogram | deramp if required; phase-preserving sinc/Lanczos complex interpolation |
| Unwrapped phase | bilinear or a documented complex-gradient-aware method |
| Coherence | preferably re-estimate on geocoded SLC support; otherwise bilinear with clear provenance |
| DEM/incidence/continuous geometry | bilinear or higher-order smooth interpolation |
| Connected component / layover / water mask | nearest neighbor or block mode |

Do not apply one raster-library default to every layer.

## Convert phase to displacement

After removing unwanted phase terms and selecting a reference,

$$
d_{LOS}=-\frac{\lambda}{4\pi}\phi_{defo}
$$

for the convention used throughout this guide. State whether positive means
toward or away from the satellite. A single LOS measurement cannot uniquely
recover vertical, east-west, and north-south motion. Combining ascending,
descending, GNSS, or physical-model constraints requires compatible look
vectors and uncertainty propagation.

## Product metadata

A scientific product should include:

- reference and secondary acquisition IDs/times;
- wavelength, polarization, orbit direction, relative orbit, and acquisition
  segmentation identifiers;
- complex conjugation and LOS sign conventions;
- orbit and DEM sources, checksums, datums, and interpolation methods;
- coregistration, looks, coherence, filtering, merge, and unwrap parameters;
- CRS, affine transform, pixel registration, no-data, and valid masks;
- connected components and uncertainty/quality layers;
- software version, source commit, and processing timestamp.

## End-to-end validation ladder

### Geometry

- forward/inverse transform closure;
- coastlines and stable bright targets align with independent maps;
- no half-pixel shift or north/south axis reversal;
- terrain displacement pattern is consistent with look direction.

### Interferometry

- conjugate pair reverses phase;
- coherence, valid fraction, and mission-specific overlap residuals are
  plausible;
- DEM-correlated residual and long-wavelength orbit ramp are quantified;
- no phase seam is hidden by a plotting wrap or color-cycle choice.

### Unwrapping

- rewrap residual is near numerical zero;
- connected components are stored;
- loop closure and independent-unwrapper comparisons are evaluated;
- reference-pixel choice and component offsets are reproducible.

### Output

- CRS/transform round-trip correctly in another GIS library;
- no-data and label masks use appropriate resampling;
- quicklooks are generated from the written product, not an in-memory array;
- provenance is sufficient to reproduce every phase correction.

## A developer's definition of done

A pipeline is not complete when it writes `unwrapped.tif`. It is complete when
the output can be traced to input samples and metadata, its sign and units are
unambiguous, quality gates are machine-readable, and an independent reader can
open the artifact and reproduce the key diagnostics.

Continue with the [software implementation guides](../software/index.md) to see
how InSAR.dev, ISCE2, ISCE3, and GMTSAR realize these concepts differently.

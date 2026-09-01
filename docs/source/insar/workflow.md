# Workflow

## The complete path

```{mermaid}
flowchart TD
    subgraph replaceable[Replaceable per mission]
        product[mission product]
        adapter[mission adapter<br>decode complex samples, timing, carriers, valid support]
        product --> adapter
    end
    geometry[orbit + DEM range-Doppler geometry]
    coreg[coarse and fine coregistration]
    resample[phase-preserving complex resampling]
    cross[complex cross-multiplication]
    removal[flat-Earth and topographic phase removal]
    looks[multilooking, coherence, and filtering]
    mosaic[mission-specific segment continuity or mosaic, if required]
    unwrap[masking and phase unwrapping]
    geocode[geocoding, validation, and LOS displacement]

    adapter --> geometry --> coreg --> resample --> cross
    cross --> removal --> looks --> mosaic --> unwrap --> geocode
```

The first box is intentionally replaceable. Sentinel-1 IW supplies a SAFE/TOPS
adapter with bursts, a steering carrier, and ESD. A future NISAR adapter will
decode its RSLC structure without inheriting Sentinel-1 burst assumptions.
Both then use the same physical objectives in the core path.

## Mission and core responsibilities

| Layer | Owns | Must not assume |
|---|---|---|
| Mission adapter | product schema, timing, wavelength, orbit source, Doppler/carrier model, valid support, segment identity | that every mission uses TOPS or bursts |
| Core geometry | radar-to-ground and ground-to-radar transforms, DEM height convention, look vectors, baselines | a particular SAFE, HDF5, or annotation layout |
| Core registration | mapping, complex resampling, residual estimation, acceptance metrics | one mission-specific refinement such as ESD is universal |
| Core interferometry | cross-multiplication, modeled phase removal, looks, coherence, filtering | one execution order or filename convention |
| Mission continuity hook | burst/frame/swath alignment and mosaic when the acquisition mode requires it | that all SLCs are segmented |
| Core unwrapping/output | masks, connected components, unwrapping, geocoding, LOS convention, validation | mission metadata can be discarded after preprocessing |

## Why the shared stages exist

| Stage | The problem it removes | Observable failure if skipped |
|---|---|---|
| Geometry | Acquisitions observe the ground from different positions | Large offsets and topography-dominated phase |
| Coregistration | Equal array indices do not initially represent the same scatterer | Decorrelation and phase bias |
| Interferogram | The observable is phase difference, not either absolute SLC phase | No deformation-sensitive measurement |
| Phase correction | Deterministic geometry dominates the smaller residual signals | Dense fringes and DEM-correlated residuals |
| Multilook/coherence/filter | Speckle and decorrelation create unreliable gradients | Residues and unstable unwrapping |
| Unwrapping | Complex phase is observed modulo $2\pi$ | No continuous range-change estimate |
| Geocoding and QC | Radar coordinates are not a map and plausible errors are common | Mislocated or scientifically invalid output |

Mission steps solve different problems. For example, Sentinel-1 TOPS carrier
handling protects phase during interpolation, while burst alignment prevents
seams between acquisition segments. Neither operation is a universal InSAR
stage.

## The phase budget

The wrapped interferometric phase is

$$
\phi_w = \mathcal{W}\!\left(
\phi_{\mathrm{defo}} + \phi_{\mathrm{topo}} + \phi_{\mathrm{flat}}
+ \phi_{\mathrm{tropo}} + \phi_{\mathrm{iono}} + \phi_{\mathrm{orbit}}
+ \phi_{\mathrm{proc}} + \phi_{\mathrm{noise}}
\right),
$$

where $\mathcal{W}(x)=\operatorname{atan2}(\sin x,\cos x)$ wraps phase to
$(-\pi,\pi]$. Unwrapping removes only the modulo ambiguity. It does not
separate deformation from atmosphere, orbit error, DEM error, or a
mission-adapter mistake.

For $z_{12}=s_1s_2^*$, line-of-sight displacement is commonly written

$$
d_{\mathrm{LOS}}=-\frac{\lambda}{4\pi}\phi_{\mathrm{defo}},
$$

when positive displacement means motion toward the satellite. Packages can
reverse the interferogram or LOS convention, so provenance must record both.

## Two valid execution architectures

### Radar-domain

The reference SLC grid remains the computational grid through unwrapping. The
secondary SLC is resampled into it and products are geocoded near the end. This
keeps native Doppler and acquisition geometry explicit and is the mature ISCE2
and GMTSAR pattern.

### Geocode-first

Each acquisition or acquisition segment is transformed to a common map grid
before interferogram formation. Registration, geocoding, carrier handling, and
simulated-phase removal can be fused into one phase-aware remap. This simplifies
xarray/Dask mosaics but makes the geocoder responsible for every phase term.

## Production acceptance gates

Record at least:

1. mission/product identity, wavelength, orbit source, DEM source and datum;
2. coarse and fine range/azimuth residuals;
3. mission-specific carrier, overlap, or segment-continuity diagnostics;
4. coherence distribution and valid-pixel fraction;
5. unwrap connected components, rewrap residual, and pair-loop closure;
6. geolocation residuals and exact phase/LOS sign conventions.

Continue with a [mission guide](missions/index.md) for input-specific behavior
and [core processing](processing/index.md) for the shared derivations.

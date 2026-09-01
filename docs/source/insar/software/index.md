# Software Studies

The preceding chapters describe processing invariants. This section traces four
real implementations so readers can see where algorithms agree, where they
differ, and where a project does not provide a Sentinel-1 workflow.

## Comparison at a glance

| Topic | InSAR.dev | ISCE2 `topsApp` | ISCE3 / NISAR workflow | GMTSAR |
|---|---|---|---|---|
| Sentinel-1 SAFE reader | `insardev_pygmtsar` | Yes | No complete Sentinel-1 TOPS application | Yes |
| Main domain | Per-burst common geographic grid | Reference radar grid | NISAR reference radar grid, then geocode | Reference radar grid |
| Coarse registration | Orbit/DEM transform | `topo` + `geo2rdr` | `rdr2geo` + `geo2rdr` | `SAT_llt2rat` geometry |
| Refinement | Range xcorr; stack/burst phase alignment | Amplitude range xcorr + ESD | optional dense offsets/rubbersheet | polynomial geometry/xcorr; optional spectral diversity |
| Complex resampling | Fused deramp → map remap → reramp | `Resamp_slc` with TOPS carrier | sinc-based `ResampSlc` CPU/GPU | `make_s1a_tops` + `resamp` |
| Flatten/topography | During geocoded SLC transform by default | range-offset phase during burst IFG | crossmul from geo2rdr range offsets | `phasediff -topo topo_ra.grd` |
| Filtering | Gaussian look + optional Goldstein | Goldstein-Werner | no filter, boxcar, or Gaussian | Gaussian/decimation + Goldstein-Werner |
| Burst merge | robust overlap alignment + circular/arithmetic dissolve | selection or equal complex average after ESD | no Sentinel-1 burst merge | hard cuts/stitch positions |
| Unwrapping | DCT+IRLS $L_1$; optional SNAPHU | SNAPHU; ICU/Grass alternatives | SNAPHU default; ICU or PHASS | SNAPHU |

## How to read the implementation pages

Each page states a pinned source commit. The pages describe that snapshot, not
an eternal product promise. Source links are pinned where possible, and
algorithm names are taken from actual calls rather than marketing summaries.

```{toctree}
:maxdepth: 1

insardev
isce2
isce3
gmtsar
```

## The most important architectural lesson

Matching output filenames does not imply matching algorithms. ISCE2 can apply
flattening during burst-interferogram formation; ISCE3 can apply it inside
crossmul; InSAR.dev can remove it while geocoding each SLC; GMTSAR passes a
radar-coordinate topography grid to `phasediff`. Compare the modeled phase and
sign convention, not the apparent stage label.

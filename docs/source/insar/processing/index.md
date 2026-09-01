# Core Processing

These chapters describe the mission-independent InSAR path. They begin after a
mission adapter has exposed complex samples, timing, wavelength, orbit state,
valid support, and any carrier model required for phase-preserving resampling.

```{mermaid}
flowchart TD
    slcs[mission-adapted complex SLCs]
    geometry[orbit and DEM geometry]
    coreg[coregistration and complex resampling]
    ifg[interferogram formation]
    removal[flat-Earth and topographic phase removal]
    looks[multilooking, coherence, and filtering]
    unwrap[masking and phase unwrapping]
    output[geocoding, displacement conversion, and validation]

    slcs --> geometry --> coreg --> ifg --> removal --> looks
    looks --> unwrap --> output
```

## What remains mission-specific

The mathematical objective of each stage is shared, but adapters still provide
mission parameters. For example, wavelength changes phase-to-range conversion;
orbit and Doppler metadata drive geometry; and a TOPS mission may add ESD to
the common coregistration objective. Those additions belong in the mission
subsection and plug into the shared stage rather than replacing it.

```{toctree}
:maxdepth: 1

geometry
coregistration
interferogram
phase_corrections
multilooking
unwrapping
geocoding
```

# Missions

The mission layer converts a focused SAR product into a phase-aware complex
data contract that the shared InSAR algorithms can consume. It owns what is
specific to a satellite, instrument, acquisition mode, or product family.

## Mission adapter boundary

A mission subsection should define:

1. product layout and complex-sample decoding;
2. acquisition timing, wavelength, polarization, and orbit metadata;
3. Doppler, focusing, or steering carriers that affect interpolation;
4. valid-sample, swath, burst, or frame support;
5. mission-specific registration refinements and continuity checks;
6. the handoff contract to [core processing](../processing/index.md).

It should not repeat general range-Doppler geometry, interferogram formation,
topographic-phase removal, filtering, or unwrapping theory.

## Available mission guides

| Mission | Acquisition structure | Guide status | Mission-specific focus |
|---|---|---|---|
| Sentinel-1 IW | TOPS bursts in three subswaths | Available | SAFE SLC, TOPS carrier, ESD, burst/subswath continuity |
| NISAR | RSLC swaths without the Sentinel-1 TOPS burst model | Planned | RSLC metadata, frequency/polarization structure, mission-specific preprocessing |

Adding NISAR later requires a new mission subsection, not a copy of the shared
processing chapters.

```{toctree}
:maxdepth: 2

sentinel1/index
```

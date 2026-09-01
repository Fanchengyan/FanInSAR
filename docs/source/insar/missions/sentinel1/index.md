# Sentinel-1

This subsection contains only the parts of the workflow that are specific to
Sentinel-1 Interferometric Wide Swath (IW) Single Look Complex data. Sentinel-1
uses TOPS bursts, so phase-preserving interpolation requires carrier handling
and azimuth registration is unusually strict.

## Sentinel-1 handoff to shared processing

```{mermaid}
flowchart TD
    safe[SAFE SLC]
    decode[decode measurement and annotation files]
    identity[preserve burst/subswath identity and valid support]
    carrier[model the TOPS carrier]
    reramp[use deramp/resample/reramp where interpolation is required]
    esd[refine TOPS azimuth timing with ESD when applicable]
    shared[join the shared geometry, coregistration, interferogram,<br>phase-correction, filtering, and unwrapping stages]
    merge[align and merge burst/subswath products<br>at the appropriate product stage]

    safe --> decode --> identity --> carrier --> reramp --> esd
    esd --> shared --> merge
```

The [SLC page](slc.md) defines the input contract. The [TOPS page](tops.md)
explains carrier handling and ESD. The [burst merge page](burst_merge.md)
explains the mission-specific continuity and mosaic problem. General InSAR
operations remain in [core processing](../../processing/index.md).

:::{note}
Burst merge is shown here because it is mission-specific, even though an
actual processor may perform it after interferogram formation, filtering, or
unwrapping. Information architecture and execution order are different
concerns.
:::

```{toctree}
:maxdepth: 1

slc
tops
burst_merge
```

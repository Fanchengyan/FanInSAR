(insar-guide)=

# InSAR Guide

This guide explains how focused complex SAR data becomes a geocoded,
unwrapped interferogram. It separates two kinds of knowledge that should not
be duplicated:

- **Mission adapters** explain product formats, acquisition modes, timing,
  carriers, valid-data masks, and mission-specific continuity rules.
- **Core processing** explains the geometry and signal-processing operations
  shared by repeat-pass InSAR missions.

Sentinel-1 is the first mission subsection. Future NISAR or other mission
subsections can supply their own front end and then join the same core
processing path.

:::{important}
A diagram is a dependency graph, not necessarily a command list. A
geocode-first processor can fuse registration, geocoding, carrier handling,
and simulated-phase removal into one interpolation. A radar-domain processor
can expose them as separate stages. Correctness is defined by the phase model
and data contract being preserved.
:::

## Choose a path

**Learn the complete workflow**
: Start with the [workflow map](workflow.md), then follow the mission adapter
  and core-processing chapters.

**Process Sentinel-1 IW data**
: Read the [Sentinel-1 subsection](missions/sentinel1/index.md) for SAFE SLC,
  TOPS, ESD, and burst continuity, then continue with
  [core processing](processing/index.md).

**Implement or compare algorithms**
: Read the equations and quality gates in core processing, then inspect the
  source-pinned [software studies](software/index.md).

## Shared convention

Unless a page says otherwise, the interferogram convention is

$$
z_{12}=s_1s_2^*,
$$

where acquisition 1 is the reference and acquisition 2 is the secondary.
Software may use the opposite conjugation order, so every product must record
its phase and line-of-sight sign conventions.

```{toctree}
:maxdepth: 2
:caption: Overview

workflow
```

```{toctree}
:maxdepth: 3
:caption: Missions

missions/index
```

```{toctree}
:maxdepth: 2
:caption: Core processing

processing/index
```

```{toctree}
:maxdepth: 2
:caption: Implementations

software/index
```

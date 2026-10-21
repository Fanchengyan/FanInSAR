# Processing overview

Stack processing stages (explicit, not a black box):

1. **READ** — open SAFE, build TOPS carrier from annotation, read complex burst window  
2. **DERAMP** — remove TOPS azimuth carrier  
3. **COREG** — geometry coarse shift (orbits) + amplitude correlation refine + resample + reramp  
4. **IFG** — complex interferogram, multilook, Goldstein filter  
5. **UNWRAP** — SNAPHU (the default spatial unwrapper)
6. **GEOCODE** — `rdr2geo` (+ DEM if provided) for unwrapped phase and coherence  
7. **WRITE** — Zarr (radar layers + `geocoded/` group) and STAC item  

## Stack API

```python
from faninsar.missions import S1Stack

stack = S1Stack.from_safes(
    ["acquisitions/20240101.SAFE", "acquisitions/20240113.SAFE"],
    work_dir="out/stack",
    reference="20240101",
)

# Each stage is explicit and resumable.
stack.prepare_scenes()
stack.coregister_scenes()
stack.form_interferograms(multilook=(2, 8))
stack.unwrap()
result = stack.network.invert()
```

`Stack.unwrap()` uses SNAPHU by default. The Torch-native `SpatialIRLS`
backend remains available as an explicit developer/experimental strategy:

```python
from faninsar.processing.unwrapping import SpatialIRLS

stack.unwrap(SpatialIRLS())
```

With no model, `Network.invert()` performs SBAS. Pass a model class to select
NSBAS; the class is instantiated internally for the selected acquisitions:

```python
from faninsar.timeseries import LinearModel

result = stack.network.invert(model=LinearModel)
```

Inversion writes the canonical STAC+Zarr product automatically, and
`result.canonical_path` identifies its root. Export is an optional conversion:

```python
result.export("out/timeseries")  # STAC+COG; requires complete geo metadata
radar_result.export("out/timeseries-radar", exporter="stac-zarr")
radar_result.export("out/timeseries-netcdf", exporter="netcdf")
```

There is no publish step in this workflow: every stage commits products
(``ifg/``, ``unwrap/``, ``timeseries/``) directly into the work directory,
and ``stack.network`` is a live view over them. ``Network.open(path)``
reopens a completed work directory from its ``stack.json``.

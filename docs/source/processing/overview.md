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
report = stack.prepare_data()
stack.coregister_scenes()
stack.form_interferograms(multilook=(2, 8))
stack.unwrap()
result = stack.network.invert()
```

Burst products stay unmerged on disk. A committed `merge_layout` lets collection
views select native samples at read time, independently per date or pair:

```python
merged = stack.scenes.merge_bursts()
slc = merged["20240113"].layers.slc
slc.read(window=(2, 6, 4, 12))  # source-grid window, sliced before load
slc.plot()            # crop=True: tight valid bbox, holes kept
slc.plot(crop=False)  # original scientific extent

ifgs = stack.interferograms.merge_bursts()
ifgs["20240101_20240113"].complex.read()
unmerged = stack.interferograms.sel(representation="burst", burst=0)
```

All mission constructors (`S1Stack.from_safes`, `S1Stack.from_search`, and
`NISARStack.from_rslc`) configure requests only. Call `prepare_data()` explicitly
before raw-input scientific stages; those stages never download or auto-prepare.
`prepare_data()` returns requested, ready and failed membership with an outcome
of `complete`, `partial`, or `failed`. The default `on_error="warn"` keeps
independent qualified targets usable. Use `on_error="raise"` for strict
completeness; `DataPreparationError.report` retains the partial report and the
previous committed state remains installed. `offline=True` requires local inputs.
A complete matching request is a local no-op; a partial request retries missing
inputs. `Stack.open(work_dir)` is local and read-only, including request-only
recovery and product inspection after original raw sources disappear.

### CDSE burst transfer concurrency

Pass runtime resource guards to `prepare_data()` to control measurement tasks,
Range requests per exact object URL, and their shared HTTP request ceiling:

```python
from faninsar.remote.records import RemoteResourceBudget

report = stack.prepare_data(
    budget=RemoteResourceBudget(
        max_active_files=4,
        max_ranges_per_file=4,
        max_workers=4,
        max_metadata_files=2,
    ),
)
```

These are client policy ceilings, not provider quota claims. Per-file limits do
not multiply the shared `max_workers` ceiling. For eligible CDSE burst requests
with one selected item per acquisition date, independent acquisition tasks share
one provider-owned RangeFetcher, authentication refresh, adaptive window and
cooldown for the entire preparation call. Other acquisition shapes retain their
existing serial scheduling. The runtime budget does not change scientific Stack
identity. Choose an HTTP ceiling allowed by your account; CDSE documents four
concurrent connections for General Users in its
[quota table](https://documentation.dataspace.copernicus.eu/Quotas.html).

`max_ranges_per_file=None` preserves global-only Range admission. Metadata
reservation is opt-in: with `max_metadata_files=2`, pending annotation XML work
uses at most two of the four shared file slots while payload files use the other
two. When pending metadata drains, payloads may use all file slots. Existing
transfers are not interrupted when new metadata arrives. XML and payload Range
requests still share the same HTTP ceiling and provider backoff; this is a
client-side scheduling policy, not a claim about CDSE's internal file accounting.

CDSE burst preparation permits ten shared 429 retries per exact object URL.
Exhaustion defers that acquisition until every regular acquisition task in the
preparation call has drained. Deferred acquisitions run serially with one shared
HTTP request and no further 429 retry; another 429 becomes a reported failure.
This final pass retains the same session, operation ledger and server cooldown.
Authentication, integrity and other failures are not reclassified as rate limits.
Requested acquisitions and failed membership remain in the preparation report;
strict `on_error="raise"` does not silently drop dates or interferogram pairs.

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

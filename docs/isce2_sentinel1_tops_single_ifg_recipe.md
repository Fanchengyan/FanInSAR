# ISCE2 Sentinel-1 TOPS → Single Interferogram (wrapped + coherence, geocoded to UTM 40 m)

> Research note grounded in the **official ISCE2 source code** (`applications/topsApp.py`,
> `components/isceobj/TopsProc/*`, `contrib/stack/*`) and the conda-forge package metadata.
> Verified against `isce-framework/isce2` branch `main`, conda-forge `isce2` 2.6.5.
>
> **Important:** the URLs cited in the request — `https://isce-framework.github.io/isce2/index.html`
> and `https://isce-framework.github.io/isce2/tutorials/s1TOPS/TOPS.html` — currently return **404**.
> That GitHub-Pages documentation site is not hosted anymore. The maintained, authoritative
> references are the **source code in the repo** (linked below) and the in-repo `README.md`
> (`contrib/stack/README.md`, `contrib/stack/topsStack/README.md`). Everything in this note is
> traced back to that source.

---

## 0. Terminology — two different ISCE2 S1 workflows (do not confuse)

| | **topsApp (single pair)** | **topsStack (stack processor)** |
|---|---|---|
| Entry point | `topsApp.py` (official Application) | `stackSentinel.py` → generates `run_*` scripts; modular `topo.py`, `geocodeGdal.py`, `mergeBursts.py`, … |
| Purpose | ONE reference/secondary interferogram | a **stack**/time-series of many acquisitions |
| Config | one `topsApp.xml` | `-s/-o/-a/-d` + auto `configs/` |
| Steps | built-in `--steps` | numbered `run_*` shell files |
| Geocoding | built-in `geocode` step | separate `geocodeGdal.py`/`geocodeIsce.py` |

The user's goal (ONE pair, wrapped phase + coherence, geocoded) is exactly the **topsApp**
single-pair workflow. The `topo.py` / `geocode.py` / `geocodeGdal.py` names the user mentions
are the separate **stack** scripts — I cover both so you can pick the one that matches your
intent, but the primary recipe below is **topsApp**.

---

## 1. Installation (Ubuntu x86_64, no sudo)

### 1.1 Recommended: conda-forge (simplest reliable, no root)

conda-forge ships a pre-built binary `isce2` **2.6.5** for `linux-64` (and `osx-64`). This is a
proper precompiled package — no compiler toolchain, no `scons`/`cmake` build, works fully
unprivileged. Check: https://anaconda.org/conda-forge/isce2

```bash
# install micromamba (or use conda/mamba — same commands)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash

micromamba create -n isce2 -c conda-forge isce2
micromamba activate isce2
```

- **Python version:** isce2 2.6.5 is built against **Python 3.11** (conda-forge pins the
  interpreter as part of the package; it resolves 3.11 automatically). The package also pulls
  `numpy`, `scipy`, `gdal`/`libgdal`, `h5py`, `fftw`, `hdf4`/`hdf5`, `opencv`, `scons`,
  `cython`, `gfortran` as runtime deps. ISCE2 is not pure-Python — it ships compiled C++/Fortran
  extension modules, so you must use the conda env's interpreter.
- After activation, set up the runtime environment:
```bash
export ISCE_HOME=$(python3 -c "import isce; import os; print(os.path.dirname(isce.__file__))")
source $ISCE_HOME/isce/components/isce_defaults.sh  # optional, sets PATH/PYTHONPATH
export PATH=$PATH:$ISCE_HOME/applications
```
Verify: `python3 -c "import isce, isceobj; print(isce.__version__)"` and `topsApp.py --help`.

### 1.2 pixi

Same binaries as conda-forge, but declarative and reproducible (pixi uses the conda-forge
channel under the hood):

```bash
pixi init
pixi add isce2
pixi shell
```

### 1.3 Docker (`isce-framework/isce2`)

The repo does ship dockerfiles (`docker/Dockerfile`, `docker/Dockerfile.cuda`), but they build
**from source** on an old base (`ubuntu:20.04`, Python 3.8) and are not the fastest or simplest
path, and running Docker without sudo is itself extra setup. **Recommendation: skip Docker** on
this box; use the conda-forge binary (1.1) or pixi (1.2). Note also: the current
`isce-framework/isce2` Docker images are not actively published to Docker Hub / GHCR under that
exact tag, and the GPU path in `Dockerfile.cuda` is ancient (ISCE2 2.3 era). GPU for TOPS is
not needed — see §7.

**Bottom line:** use **conda-forge `isce2` via micromamba (or pixi), Python 3.11, unprivileged.**
Do not build from source unless you must.

---

## 2. The `topsApp.xml` for a Sentinel-1 TOPS run

The root tag is `<topsApp>` and the mandatory component is `<component name="topsinsar">`.
Reference/secondary are Facility components `<component name="reference">` / `<component
name="secondary">` (sensor catalogs). All property **public names** below are confirmed in
`applications/topsApp.py` (`Component.Parameter(..., public_name=...)`) and the in-repo template
`examples/input_files/topsApp.xml` + `examples/input_files/reference_TOPS_SENTINEL1.xml`.

### 2.1 REQUIRED properties

**In `<topsinsar>`:**
- `sensor name` → `SENTINEL1` (mandatory).
- `range looks` (integer) — **yes, the XML property name is `range looks`** and `azimuth looks`
  is `azimuth looks`. (Not `numberOfRangeLooks`.) Confirmed: `public_name='range looks'`,
  `public_name='azimuth looks'` in `topsApp.py`.
- `azimuth looks` (integer).

**In `<reference>` and `<secondary>`** (S1 TOPS sensor catalog, see
`reference_TOPS_SENTINEL1.xml`):
- `safe` — path to the `.SAFE` directory **or** the `.zip` (filename must end in `.zip` to be
  recognized as a zip). Multiple slices → provide as a list.
- `swath number` — a single swath, or use the `swaths` list at the topsinsar level to override.
- `output directory` — where the unpacked SLC product goes (files: `IW<n>` + `.xml`).
- `orbit directory` — folder containing the precise/restituted `.EOF` files (see §6).
- `auxiliary data directory` — folder with the S1 AUX_CAL antenna-pattern files (see §6).

### 2.2 Optional properties we set for this goal

At the `<topsinsar>` level:
- `demFilename` — path to a WGS84 DEM (or leave unset and ISCE downloads SRTM automatically via
  `verifyDEM`). We supply one.
- `swaths` — list of swaths to process, e.g. `[1,2,3]` (overrides per-`swath number`). If a
  `region of interest` is given, only swaths covering it are kept.
- `region of interest` — SNWE (S,N,W,E) lat/lon cropping box for unpacking.
- `filter strength` — Goldstein power-spectral filter alpha; default 0.5. Typical 0.4–0.6.
- `geocode bounding box` — SNWE box the geocode step projects onto.
- `do unwrap` — **`False`** (this achieves your "stop before unwrap" requirement; the
  `unwrap`/`unwrap2stage` steps become no-ops but the `geocode` step still runs).
- `do ESD` — default True; leave it (needed for accurate TOPS azimuth coreg).
- `use virtual files` — default True (saves disk by using GDAL VRTs).
- `geocode list` — optional; by default ISCE geocodes `['phsig.cor','topophase.cor',
  'filt_topophase.unw','los.rdr','topophase.flat','filt_topophase.flat','filt_topophase_2stage.unw']`
  (see `TopsProc.py` `GEOCODE_LIST` default). To force only wrapped phase + coherence:
  `<property name="geocode list">['filt_topophase.flat', 'phsig.cor']</property>`.
- `unwrapper name` — irrelevant once `do unwrap` is False, but can be set to `snaphu_mcf`.

**NUANCE for §2.1/§2.2:** the `range looks`/`azimuth looks` in topsApp are applied **twice**:
(1) per-burst in the `burstifg` step (`takeLooks`), and (2) again in `mergebursts` to multilook
the merged product (see §3). The "looks" the user names — `azimuthLooks=5, rangeLooks=2` — map to
`azimuth looks = 5` and `range looks = 2` in the XML. These are NOT `numberRangeLooks`-style
internal names; the XML public names are as above.

### 2.3 Complete working example `topsApp.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<topsApp>
  <component name="topsinsar">
    <property name="sensor name">SENTINEL1</property>

    <!-- Multilooking (applied to the merged interferogram & geometry) -->
    <property name="azimuth looks">5</property>
    <property name="range looks">2</property>

    <!-- Whichever swaths/region you want (SNWE). Overrides per-safe swath number. -->
    <property name="swaths">[1,2,3]</property>
    <property name="region of interest">[19.10, 19.45, -155.55, -154.70]</property>

    <!-- DEM (WGS84). If omitted, ISCE downloads SRTM. -->
    <property name="demFilename">../DEM/dem_lat_19_lon_-155.dem.wgs84</property>

    <!-- Goldstein filter strength -->
    <property name="filter strength">0.4</property>

    <!-- Geocode a wrapped+coherence-only list onto a UTM grid -->
    <property name="geocode bounding box">[19.10, 19.45, -155.55, -154.70]</property>
    <property name="geocode list">['filt_topophase.flat', 'phsig.cor']</property>

    <!-- STOP before unwrapping, but still run the geocode step -->
    <property name="do unwrap">False</property>

    <component name="reference">
      <property name="safe">/data/S1A_IW_SLC__1SDV_20230101T010101_20230101T010131_000001_000001_AAAA.SAFE</property>
      <property name="swath number">2</property>
      <property name="output directory">reference</property>
      <property name="orbit directory">/data/orbits/S1A</property>
      <property name="auxiliary data directory">/data/aux_cal</property>
    </component>

    <component name="secondary">
      <property name="safe">/data/S1A_IW_SLC__1SDV_20230113T010101_20230113T010131_000007_000008_BBBB.SAFE</property>
      <property name="swath number">2</property>
      <property name="output directory">secondary</property>
      <property name="orbit directory">/data/orbits/S1A</property>
      <property name="auxiliary data directory">/data/aux_cal</property>
    </component>
  </component>
</topsApp>
```

Run from the directory where you want outputs created:
```bash
topsApp.py topsApp.xml
```

---

## 3. Command sequence and steps (`--steps`)

`topsApp.py` supports the `--steps` flow (shown in `topsApp.py`, `_steps()`); without it,
`main()` runs everything in sequence. Step names and order (authoritative from `_steps()`):

```
startup
preprocess        -> unpack ref/sec SAFE -> reference/ and secondary/ (IW<n> SLCs)
computeBaselines  -> baseline + number of common bursts (find overlap between dates)
verifyDEM         -> load/download DEM (runTopo calls this internally too)
topo              -> per-burst geometry: lat_XX.rdr, lon_XX.rdr, hgt_XX.rdr, los_XX.rdr
subsetoverlaps    -> extract burst-overlap strips for ESD
coarseoffsets     -> coarse azimuth offsets (geometry-based)
coarseresamp      -> coarse resample of secondary
overlapifg        -> overlap interferograms
prepesd           -> prepare ESD inputs
esd               -> Enhanced Spectral Diversity azimuth misregistration (the TOPS killer step)
rangecoreg        -> range coregistration (common range offset)
fineoffsets       -> refined (dense) offset estimation
fineresamp        -> precise resample of secondary bursts
ion               -> ionospheric phase estimation (skipped if do ionosphere correction=False)
burstifg          -> per-burst interferogram + burst-level coherence (multilooked with az/range looks)
mergebursts       -> MERGE bursts/swaths into merged/ products (topophase.flat etc.)
filter            -> Goldstein filter -> filt_topophase.flat; computes phsig.cor coherence
unwrap            -> skipped when do unwrap=False
unwrap2stage      -> skipped when do unwrap 2 stage=False
geocode           -> geocode the geocode list to the geocode bounding box
denseoffsets / filteroffsets / geocodeoffsets  -> only if dense offsets enabled (default off)
```

To run step by step and stop right after geocoding (no unwrap):

```bash
# Produces everything up through the wrapped phase + filtering + geocoding
topsApp.py topsApp.xml --steps=startup,preprocess,computeBaselines,verifyDEM,topo,subsetoverlaps,coarseoffsets,coarseresamp,overlapifg,prepesd,esd,rangecoreg,fineoffsets,fineresamp,burstifg,mergebursts,filter,geocode
```

Or run fully in one shot (it will skip unwrap for you):
```bash
topsApp.py topsApp.xml
```

### Which step/files give wrapped phase + coherence

- **`burstifg`** → `fine_interferogram/IW<n>/burst_XX.int` (flattened, per-burst, already
  multi-looked at `azimuth looks × range looks`) and `burst_XX.cor` (burst-level coherence).
- **`mergebursts`** → **`merged/topophase.flat`** (wrapped, flattened interferogram on the
  multilooked grid) and **`merged/topophase.cor`** (burst-merged coherence), plus the merged
  geometry **`merged/lat.rdr`, `merged/lon.rdr`, `merged/z.rdr`, `merged/los.rdr`** and merged
  SLCs `merged/reference.slc`, `merged/secondary.slc`.
- **`filter`** → **`merged/filt_topophase.flat`** (Goldstein-Werner filtered wrapped phase) and
  **`merged/phsig.cor`** (coherence computed by the ICU correlator on the filtered product).
  These two are the ones to geocode.

(The `.flat` is complex `cfloat` data; the wrapped phase is `atan2(imag,real)`. `phsig.cor` is
`FLOAT` coherence in [0,1].)

---

## 4. Geocoding the wrapped phase + coherence to a UTM map at ~40 m

There are **two** ways; pick based on whether you stay inside topsApp or go manual:

### 4.1 (Recommended) Inside topsApp — the built-in `geocode` step

The `geocode` step (`TopsProc/runGeocode.py`) uses **`getMergedOrbit(frames)` from the
`fine_coreg` products** plus the DEM and, for every file in `geocode list`, runs `createGeozero()`
(`zerodop/geozero`). The products in the geocode list are looked up in `merged/`, and outputs are
written as `<file>.geo` (with matching `.geo.xml` metadata). The geocode routine uses the
**same `numberAzimuthLooks`/`numberRangeLooks`, the `geocode bounding box` (SNWE), and your DEM**,
so all geocoded products share one grid. `runGeocode.py` prints the exact South/North/West/East
and range/azimuth looks it used.

This is the simplest path to a geocoded wrapped phase: it already handles the DEM crop and the
UTM/strict grid. It is **not** literally a real-valued-grounded EPSG reprojection — it is an
ellipsoid-geodetic lat/lon grid with a `geocode bounding box`, which is what topsApp produces.
If you specifically need **EPSG:326XX UTM / exact 40 m in metres**, use the manual path below.

### 4.2 Manual/stack-style geocoding (lat/lon/los in radar coord → UTM GeoTIFF)

If you run topsApp to the `filter` step only (`--steps=... ,mergebursts,filter`), you have
`merged/outputs` (filt_topophase.flat, phsig.cor) **and** the merged geometry
`merged/lat.rdr`, `merged/lon.rdr`, `merged/z.rdr`, `merged/los.rdr` (all already multilooked —
the outputs of `mergebursts`). Then geocode them with **`geocodeGdal.py`** (the stack-style
script, `contrib/stack/topsStack/geocodeGdal.py`) or `geocodeIsce.py`. These are the scripts
the docs refer to for geocoding merged products. Add `contrib/stack/topsStack` to `$PATH`.

**`geocodeGdal.py`** (2D geolocation via GDAL VRT — this is the one that gives you a metre-sized
grid):

```bash
# -b  is SNWE; -x/-y are output pixel sizes in DEGREES (lon/lat step).
# Convert your 40 m goal to degrees: ~40 m / (111320 m/deg) ≈ 3.6e-4 deg.
geocodeGdal.py -l merged/lat.rdr -L merged/lon.rdr \
  -f "merged/filt_topophase.flat merged/phsig.cor" \
  -b "19.10 19.45 -155.55 -154.70" \
  -x 0.00036 -y 0.00036 -r bilinear -t
# -t => GeoTIFF; outputs are written alongside the input as geo_<name>.tif (actually: outFile = geo_<basename>)
```

> The script writes `geo_<basename>` next to the source (per `runGeo`: `outFile =
> os.path.join(dirname(infile), "geo_" + basename(infile))`); with `-t` it's a GeoTIFF with
> lat/lon (EPSG:4326) georeferencing. To get an **EPSG:326XX UTM raster**, warp the result:
>
> ```bash
> gdalwarp -t_srs EPSG:32604 -tr 40 40 -r bilinear -overwrite \
>   geo_filt_topophase.flat.tif filt_topophase_UTM40.tif
> gdalwarp -t_srs EPSG:32604 -tr 40 40 -r bilinear -overwrite \
>   geo_phsig.cor.tif phsig_cor_UTM40.tif
> ```
> (adjust the EPSG zone to your QGIS location).

**`geocodeIsce.py`** — the range-Doppler/ellipsoid geocode that mirrors topsApp's internal
`geocode` step (uses the orbit + DEM rather than only lat/lon lookup). It needs the reference +
secondary dirs and the looks count:

```bash
geocodeIsce.py -f "merged/filt_topophase.flat merged/phsig.cor" \
  -b "19.10 19.45 -155.55 -154.70" \
  -d ../DEM/dem_lat_19_lon_-155.dem.wgs84 \
  -m reference -s secondary -r 2 -a 5
```

**Important caveat about `topo.py`/`geocode.py`:** The `topo.py` and `geocode.py` under
`contrib/stack/*` are the **stack** processor's modular scripts (and `stripmapStack/geocode.py`
is explicitly marked a UNAVCO-workshop demo, "not official Applications"). In the topsApp
workflow you do **not** run `topo.py`/`geocode.py` manually — topsApp runs `topo`/`geocode`
itself. The geometry files topsApp's internal geocode uses are the **per-burst** ones produced by
`topo` (`geom_reference/IW<n>/lat_XX.rdr, lon_XX.rdr, los_XX.rdr, hgt_XX.rdr`) which
`mergebursts` then merges into `merged/lat.rdr, lon.rdr, z.rdr, los.rdr`. If you prefer the
modular stack scripts, use `topo.py` + `geocodeGdal.py`/`geocodeIsce.py` with the merged
products as shown above.

### topo outputs used by geocode (as requested)

- `topo` writes per-burst: `geom_reference/IW<n>/lat_%02d.rdr`, `lon_%02d.rdr`, `hgt_%02d.rdr`,
  `los_%02d.rdr` (see `TopsProc/runTopo.py`; `los` is 2-band FLOAT: incidence + azimuth).
- `mergebursts` merges these into `merged/lat.rdr`, `merged/lon.rdr`, `merged/z.rdr`,
  `merged/los.rdr` (all multilooked). Those `merged/lat.rdr`/`lon.rdr` are the inputs to
  `geocodeGdal.py` (`-l/-L`); `merged/z.rdr` is the DEM crop reference; `merged/los.rdr` is
  what you geocode if you want the LOS map. topsApp's own `geocode` step does **not** consume
  `lat.rdr/lon.rdr` — it re-computes geometry from orbit+DEM (`runGeocode.py` uses
  `getMergedOrbit` + `createGeozero`). Only the modular `geocodeGdal.py` approach uses the
  lat/lon rasters directly.

---

## 5. Pixel spacing and looks for a ~40 m product

Sentinel-1 IW SLC native pixel spacing (authoritative values):

- **Range (slant):** computed by ISCE as `rangePixelSize = c / (2 * rangeSampleRate)` with the
  IW range sample rate (~64.35 MHz → **≈ 2.33 m slant range**). Ground-range spacing is larger
  by `1/sin(inc)` (≈ 3–5 m depending on incidence).
- **Azimuth:** `azimuthTimeInterval` from the manifest → **≈ 13.99–14.1 m** (typical 13.9–14.1 m).

**Looks math for ~40 m (range) goal:**

- **5 azimuth looks × 2 range looks:** az spacing ≈ 5 × 14.1 ≈ **70 m**; slant-range ≈ 2 × 2.33 ≈
  **4.7 m** (≈ 5–8 m ground). This gives a very anisotropic, ~70 m (az) × ~5 m (range) pixel —
  **not** a 40 m × 40 m pixel.
- **2 range looks × 5 azimuth looks** is the same thing (5 az × 2 rg); the ordering in the XML
  is just `azimuth looks`/`range looks`.

To actually reach **~40 m × 40 m** you need roughly **3 azimuth looks × ~10–16 range looks**
(3 × 14.1 ≈ 42 m az; 12 × 2.33 ≈ 28 m slant ≈ 40+ m ground — tune to your incidence angle), or
use the geocode grid (`-x/-y`, or `gdalwarp -tr 40 40`) to **resample** to 40 m regardless of
the multilooking. **The 5×2 the user proposed does not yield 40 m on the ground — it yields a
~70 m × ~5 m radar-look product.** If your real intent is "roughly 40 m ground pixel," use
`3 az × 12 rg` (closer to isotropic ~40 m) and then force the exact 40 m via the geocoding
grid/`gdalwarp -tr 40 40`.

**What ISCE's own defaults/tutorials use:** topsApp defaults are `azimuth looks = 7`,
`range looks = 19` (a ~100 m × ~44 m slant product, the classic 20-ish-metre stack look). There is
no "official 40 m" combination in the source; the tutorials/README use these defaults or what the
user supplies. For a nominal ~40 m product most practitioners use something like
`azimuth looks = 3, range looks = 12` plus geocoding to a 40 m grid.

---

## 6. Precise orbit (.EOF) handling

- topsApp locates the precise/restituted orbit automatically by **scanning the `orbit directory`
  property** and matching the correct `.EOF` file to your acquisition time/satellite
  (`Sentinel1.py`, `extractPreciseOrbit`). You do **not** need to name the exact file.
- **Where:** set `orbit directory` = a folder holding all the `S1A_OPER_AUX_POEORB_...EOF` /
  `RESORB` files. ISCE picks the right one. (Precise POEORB are available ~3 weeks after
  acquisition; RESORB sooner.)
- **Aux (antenna pattern) files:** set `auxiliary data directory` = folder with the
  AUX_CAL `.SAFE`/`.zip` files, used to correct the IPF-002.36 elevation-antenna-pattern phase
  (mostly pre-March-2015 data). Recommended to set it anyway. See `reference_TOPS_SENTINEL1.xml`
  notes and https://sar-mpc.eu/ipf-adf/aux_cal/.
- If you omit both, ISCE will try to proceed without precise orbit/aux, which degrades accuracy;
  recommend supplying at least `orbit directory`.

---

## 7. Hardware notes (64 cores, 2× A100 GPU, no sudo)

- **The A100 GPUs are essentially unused by ISCE2 TOPS.** topsApp's GPU path
  (`use GPU` + `zerodop.GPU*` modules) is experimental, ancient, and only accelerates
  topo/geo2rdr; the conda-forge build does not ship the CUDA extensions, and the TOPS coreg
  (ESD, resampling) is CPU. **Leave `use GPU` False.** The 2× A100 can be ignored.
- The **64 cores** are useful: `burstifg`/`mergebursts` and the internal `geocode` are
  single-threaded by default, but S1 IW processing is embarrassingly parallel across bursts/
  swaths. You can parallelize by running topsApp step-by-step or by splitting swaths (one
  `--steps` run per swath) across cores. If you go the stack route, the `run.py` script
  (`run.py -i runfiles -p 64`) parallelizes the `run_*` commands.
- Disk: each IW pair roughly needs 10s of GB (SLCs + ifgs + geometry); with `use virtual files`
  True, disk is modest (~tens of GB). Keep `reference`/`secondary`/`merged` on fast storage.

---

## 8. Verification checklist

1. `topsApp.py --steps` completes through `filter`.
2. `merged/filt_topophase.flat` and `merged/phsig.cor` exist (wrapped phase + coherence).
3. Unwrap did NOT run (`$ grep -i "unwrap" topsinsar.log` should show the step skipped).
4. After geocode: `merged/filt_topophase.flat.geo` + `.geo.xml` (or the UTM GeoTIFFs from §4.2)
   open in QGIS, and coherence looks sensible ([0,1]).

---

## Primary sources cited

- ISCE2 repo (branch `main`): https://github.com/isce-framework/isce2
  - `applications/topsApp.py` — steps, all property public names, geocode call
    https://github.com/isce-framework/isce2/blob/main/applications/topsApp.py
  - `components/isceobj/TopsProc/TopsProc.py` — merged output filenames & geocode list default
    https://github.com/isce-framework/isce2/blob/main/components/isceobj/TopsProc/TopsProc.py
  - `components/isceobj/TopsProc/runTopo.py` — geometry file names
    https://github.com/isce-framework/isce2/blob/main/components/isceobj/TopsProc/runTopo.py
  - `components/isceobj/TopsProc/runBurstIfg.py` — per-burst ifg + takeLooks (looks applied)
    https://github.com/isce-framework/isce2/blob/main/components/isceobj/TopsProc/runBurstIfg.py
  - `components/isceobj/TopsProc/runMergeBursts.py` — merged/topophase.flat, lat/lon/z/los, multilook
    https://github.com/isce-framework/isce2/blob/main/components/isceobj/TopsProc/runMergeBursts.py
  - `components/isceobj/TopsProc/runFilter.py` — filt_topophase.flat + phsig.cor
    https://github.com/isce-framework/isce2/blob/main/components/isceobj/TopsProc/runFilter.py
  - `components/isceobj/TopsProc/runGeocode.py` — geocode step internals
    https://github.com/isce-framework/isce2/blob/main/components/isceobj/TopsProc/runGeocode.py
  - `components/isceobj/Sensor/TOPS/Sentinel1.py` — rangePixelSize / azimuthTimeInterval
    https://github.com/isce-framework/isce2/blob/main/components/isceobj/Sensor/TOPS/Sentinel1.py
  - `examples/input_files/topsApp.xml`, `reference_TOPS_SENTINEL1.xml` — XML templates
  - `contrib/stack/README.md`, `contrib/stack/topsStack/README.md` — stack scripts, geocodeGdal
  - `docker/Dockerfile`, `docker/Dockerfile.cuda` — docker build from source (not recommended)
- conda-forge `isce2` 2.6.5 (linux-64, Python 3.11): https://anaconda.org/conda-forge/isce2
- Sentinel-1 AUX_CAL antenna-pattern / orbit data notes: https://sar-mpc.eu/ipf-adf/aux_cal/

> The requested documentation URLs `https://isce-framework.github.io/isce2/index.html` and
> `.../tutorials/s1TOPS/TOPS.html` currently return **404** — that GitHub-Pages site is not
> currently published. All conclusions above are verified directly against the isce2 source tree
> instead.

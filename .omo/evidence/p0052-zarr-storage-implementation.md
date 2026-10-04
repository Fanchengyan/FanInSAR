# P0052 non-time-series storage implementation evidence

Date: 2026-09-10

## Scope

Durable aligned-scene, IFG, and Stack unwrap payloads now use self-contained
Zarr directories and local STAC Item records. Manifests carry the shared
`domain`, `shape`, and `grid_identity` fields. Geo products require CRS and a
six-value transform; radar products reject geographic CRS metadata rather than
fabricating it. Existing generation transactions remain in use.

The collection reader exposes `StackInterferogramCollection.spatial_metadata`
with `domain`, `shape`, `grid_identity`, and optional `crs`/`transform`.

## Verification

Command:

```text
uv run --offline ruff check faninsar/io/storage/_zarr_stac.py \
  faninsar/io/storage/scene_store.py faninsar/io/storage/ifg_store.py \
  faninsar/io/storage/stack_generation.py faninsar/io/storage/artifact_transaction.py \
  faninsar/data/datasets/ifg.py faninsar/network/readers/stack.py \
  faninsar/missions/s1/processing.py faninsar/missions/nisar/provider.py \
  faninsar/stack/session.py
```

Observable result: `All checks passed!`.

Command:

```text
uv run --offline pytest tests/io/storage/test_scene_store.py \
  tests/io/storage/test_ifg_store.py tests/io/storage/test_stack_generation.py \
  tests/stack/test_spatial_unwrap_api.py -q
```

Observable result: `62 passed`.

Command:

```text
uv run --offline python <Zarr/STAC smoke scenario>
```

Scenario: publish one radar scene, one IFG, one IFG-bound unwrap, and one
Stack-root unwrap containing both a spatial layer and a one-dimensional
`reference_values` diagnostic; reopen each product and assert STAC records,
Zarr directories, and zero `*.npy` files.

Captured artifact: `.omo/evidence/p0052-zarr-smoke-20260910/`

Observable output:

```text
scene_stac True
ifg_stac True
unwrap_stac True True
npy_count 0
```

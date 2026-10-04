# Proposal 0052 documentation verification

Date: 2026-09-10

Changed documentation:

- `docs/source/processing/overview.md`
- `docs/source/api/workflows.rst`

The processing overview now shows `stack.network.invert()` after unwrapping,
SBAS when `model` is omitted, NSBAS with a model class, automatic
`result.canonical_path`, and explicit radar export formats. It also states that
`publish_generation()` and manual `Network.open()` are not part of the live
Stack workflow. The API workflow page keeps `Network.open()` only for products
persisted elsewhere and uses `network.invert()`.

Verification:

- Invocation: `cd docs && uv run --offline make html`
- Observable: `build succeeded, 78 warnings`; no build error was introduced by
  the edited pages.
- Invocation: `rg -n "canonical_path|publish_generation|STAC\\+COG|stac-zarr|netcdf" docs/build/html/processing/overview.html`
- Observable: the rendered overview contains `canonical_path`, explicit
  `STAC+COG`, `stac-zarr`, `netcdf`, and the no-`publish_generation` guidance.
- Captured rendered artifacts:
  - `docs/build/html/processing/overview.html`
  - `docs/build/html/api/workflows.html`

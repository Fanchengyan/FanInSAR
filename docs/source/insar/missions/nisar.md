# NISAR

This subsection covers only the parts of the workflow that are specific to
NISAR Range SLC (RSLC) products. NISAR scenes are frequency/polarization
swaths without the Sentinel-1 TOPS burst model. Shared geometry,
interferogram formation, flattening, and unwrapping remain in
[core processing](../processing/index.md).

## Source identity

The adapter records a canonical local path, the selected frequency and
polarization, and the declared processing configuration. Those records are
provenance, not a claim that source bytes were hashed or that they cannot
change.

Normal admission, `open_product`, window reads, and pair execution do not
scan an entire RSLC for a self digest. Repeated window reads keep the same
numerical samples while reading only the requested native selection.
Shape, grid, and channel checks still fail closed.

`expected_sha256` on :class:`~faninsar.missions.nisar.NisarAdmissionPolicy`
is an inventory of declared checksums. FanInSAR records it as metadata. It
is not compared with local file bytes at open, window read, or pair
execution.

## Stable inputs and explicit reruns

A run treats local RSLC paths as stable, trusted inputs. Same-path byte
edits after admission are not a processing reject. There is no source
stat watcher and no generation graph for this purpose.

Same-input pair reuse keeps the existing product write identity. A
deliberate scientific rerun uses the existing `overwrite=True` /
`replace_existing=True` controls and publishes a fresh identity. The
default is not destructive overwrite.

## Phase metadata

Range-offset flattening still records its scientific fields: sign, model,
RMS, coverage, and tile configuration identity. New writes do not attach
a whole phase-array payload hash. Legacy records that already contain
those optional fields remain readable.

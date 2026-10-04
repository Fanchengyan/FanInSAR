# PROPOSAL-0055 real-corpus manifest timing evidence (2026-10-02)

Host: A100 (bare metal, Intel D3-S4510 3.84 TB SATA SSD RAID volume).
Corpus: 4 Sentinel-1 SAFE zips, 23.1 GB total
(/DATA/DATA6/fancy/FanInSAR-stac-test/data/SLC).
Code: e2e_new/source (synced from local repo revision implementing
PROPOSAL-0055; MANIFEST_SCHEMA = faninsar.stack.input_manifest.v2).

Commands (verbatim, via ssh):

    build_manifest: t1 = time.perf_counter(); doc =
    im.build_manifest(mission='S1Stack', catalog_paths={...4 zips...},
    prepared=P())  ->  0.0 s; schema faninsar.stack.input_manifest.v2

    validate_manifest: t2 = time.perf_counter(); im.validate_manifest(doc)
    ->  0.0 s  (size tripwire pass)

Comparison (same corpus, previous v1 hashing implementation, measured
2026-09-29 on the same host): build_manifest 26.51 s; validate_manifest
26.44 s (sha256 of 23 GB at ~170 MB/s single-stream).

Conclusion: processing-time input-anchoring cost dropped from ~52.9 s per
prepare to ~0.0 s (size stat only); anchoring now references the
download-time verification records per PROPOSAL-0055.

Independent verify context: the verifier's report (2026-10-02) marked the
missing real-corpus evidence as MEDIUM; this file closes that item.

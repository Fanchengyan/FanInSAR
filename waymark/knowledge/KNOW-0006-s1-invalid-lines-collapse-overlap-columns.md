---
kind: knowledge
n: 6
title: "Sentinel-1 invalid lines (-1) must stay out of overlap column intersections"
status: active
type: architecture
scope: project
authors:
  - agent
tags:
  - sentinel-1
  - tops
  - nesd
  - overlap
  - coregistration
  - valid-samples
created: 2026-10-16
updated: 2026-10-16
derived_from:
  - PROPOSAL-0056
  - PROPOSAL-0058
  - NOTE-0366
related_notes:
  - NOTE-0366
related_knowledge: []
maturity: confirmed
---

# Summary

Sentinel-1 annotation marks burst azimuth lines that carry no valid samples
with `first_valid_sample = -1` and `last_valid_sample = -1`. Feeding those
stored values into a min/max column intersection collapses the result to
empty and silently skips the pair. Trim the window to the longest run of
lines that are valid on every date first; only then intersect columns over
lines whose first valid sample is non-negative.

# Problem pattern

The common-valid clip (topsStack adjustCommonValidRegion analogue) collected
`first`/`last` from every line of every date and computed
`col0 = max(firsts)`, `col1 = min(lasts) + 1`. One trailing invalid line
stores `last = -1`, so `min(lasts) = -1` and the window becomes empty.
The pair was skipped with "no common valid-sample crop across dates" on CPU
and CUDA alike, because annotation parsing precedes any device work. On the
Etna T124 campaign this dropped the 20240611_20240705 and 20240623_20240705
pairs; 20240705 burst 0 ends the overlap band with three such lines.

# Recommended approach

- Treat `first_valid_sample < 0` as "no data on this line", never as a
  numeric bound of -1.
- Before intersecting columns across dates, find the longest azimuth run
  where every date's burst line is valid, and shift top/bottom strips
  together to that run so alignment is preserved.
- Compute column bounds only from lines inside that run.
- Keep the array-envelope clip (loaded full-range burst widths) as a second,
  separate intersection step.

# Verification checklist

1. Reproduce the skip by reading annotation only (no GPU, no SLC reads).
2. Confirm the fix yields a non-empty window with lines trimmed by the count
   of invalid lines and identical column bounds to an all-valid baseline.
3. Re-run the full NESD measure on one CPU and one CUDA host with the same
   code and confirm both recover the previously skipped pairs.


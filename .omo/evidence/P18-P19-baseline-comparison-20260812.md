# P18/P19 fresh baseline comparison

Date: 2026-08-12

## Direct Pair baseline

The same two real SAFE inputs (20161207/20161231, IW1 bursts 0--2,
control spacing 8, Torch CPU, `(16,40)` looks, no ESD/Ampcor) were run from
the pre-P18 commit `397d0a6` and from the current tree.

| Run | Commit | Elapsed | Peak RSS | Output shape |
| --- | --- | ---: | ---: | --- |
| old direct Pair | `397d0a6` | 72.1626 s | 7,094,468,608 B | `(261,534)` |
| current direct Pair | current tree | 69.9482 s | 6,029,230,080 B | `(261,534)` |

The current direct path is 3.06% faster in this single run. The old and new
`complex_ifg`, `coherence`, and `wrapped_phase` arrays are exactly equal on
the common output, including shape and invalid-pixel placement. This is a
control-flow baseline only; it does not represent the main two-pass reuse
case because this invocation has no residual-measurement/product second pass.

## Stack control-flow comparison

The three-date/three-burst Radar Stack was also run with the same SAFE set and
settings through the pre-persisted Stack implementation (`4ba0e5a`) and the
current persisted IFG path. The older control-flow run took 253.514 s and
8,785,903,616 B peak RSS; the current run took 275.489 s and 9,165,783,040 B.
These numbers are not an optimization pass/fail result: `4ba0e5a` already
contains an early prepared-field implementation, and the current run includes
the manifest-last transaction writer. They are recorded to prevent treating a
transaction overhead measurement as the coregistration reuse gain.

## Decision

The qualified performance claim remains the isolated prepared-vs-reference
three-burst Pair packet (`+32.3036%` median total, exact hashes, bounded RSS).
There is no honest complete old-vs-current Radar+Geo Stack performance packet
yet; the remaining matrix item is therefore left open rather than inferred
from these incompatible control-flow runs.

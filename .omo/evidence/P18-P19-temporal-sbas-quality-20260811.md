# P18/P19 temporal and SBAS quality qualification — 2026-08-11

## Outcome

The product now computes and persists an independent temporal/SBAS quality
report before inversion. The default gate fails closed on exact implementation
invariants:

- at least one temporal pixel is publishable;
- every published pixel has a full-rank finite SBAS subnetwork;
- every temporal correction is an integer cycle count;
- published phase equals the pre-temporal phase plus `2π * correction`;
- persisted criteria and reports must match exactly on resume.

Physical closure and SBAS-residual thresholds are intentionally optional. No
universal threshold is derivable from network algebra, and no campaign error
budget or independent scientific oracle currently supplies one. Selecting those
limits is the remaining human scientific-policy decision; implementation safety
does not depend on that choice.

## Synthetic qualification

- Exact planted `2π` ambiguity: recovered with zero network closure and zero
  SBAS residual to floating-point tolerance.
- Rank-deficient published pixel: rejected.
- Configured closure/residual/convergence limits: rejected when exceeded.
- Resume under different quality criteria: rejected.
- Focused suite: 36 passed.
- Targeted Ruff: passed.
- `git diff --check`: passed.

## Fresh real Geo network

Evidence:
`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/real-full-chain-20260811/runs/20260811T074035.462652Z/summary.json`

SHA-256:
`3f26d76a9c47fd1b8b2aa25769898b3015b3bba205b7592db1315f5baaf619fa`

- 6,796 observed pixels; 6,777 have full temporal rank (99.7204%).
- 5,279 pixels converged and were published (77.8958% of full-rank pixels).
- Published full-rank fraction: 100%.
- Integer-correction error: 0.
- Pre/post-temporal reconstruction error: 0 rad.
- Modulo-network-closure absolute p95: 2.989213 rad.
- SBAS residual absolute p95: 2.058356 rad.
- Default algebraic gate: PASS.

## Fresh real Radar network

Evidence:
`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/real-radar-three-burst-full-chain-20260811/runs/20260811T074140.337494Z/summary.json`

SHA-256:
`aac7d94d796e08a62f7d6eba7f3a062f8a64125f9cad2e612a8282f4154395ec`

- 133,120 observed pixels; 132,608 have full temporal rank (99.6154%).
- 54,849 pixels converged and were published (41.3618% of full-rank pixels).
- Published full-rank fraction: 100%.
- Integer-correction error: 0.
- Pre/post-temporal reconstruction error: 0 rad.
- Modulo-network-closure absolute p95: 2.815553 rad.
- SBAS residual absolute p95: 2.003449 rad.
- Default algebraic gate: PASS.

## Interpretation

The new default is executable and defensible: partially converged rasters may
continue only after unsafe pixels are masked, while every published pixel must
pass exact algebra and rank checks. This does not claim that the observed
closure or residual distributions meet a scientific accuracy target. The real
data show that global convergence is incomplete, especially in Radar mode.
P18/P19 therefore remain scientifically unqualified until a human-approved
error budget or independent oracle defines physical closure/residual limits and
the measured distributions satisfy them.

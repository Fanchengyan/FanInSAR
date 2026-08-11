# P18/P19 raw same-frame Geo Stack → SBAS qualification

Date: 2026-08-12

## Scope

Fresh local run of the current public Stack pipeline from the same three
same-frame Sentinel-1 SAFE ZIPs used by the Radar run. The run used IW1
bursts 0/1/2, an automatically constructed EPSG:32647 grid at 40 m,
`coregistration_grid="geo"`, Pair mode, `(8, 20)` multilook, CPU Torch,
spatial unwrap, temporal reconciliation, and SBAS inversion.

## Results

- Status: `PASS` for the raw same-frame Geo functional scope.
- 3 scenes, 3 bursts, 3 pair IFGs, all pair artifacts on the common Geo grid.
- Geo grid: EPSG:32647, shape `(1857, 2422)`, 40 m resolution.
- Timeseries shape `(3, 232, 121)` after `(8, 20)` multilook; 12,932 finite pixels.
- Elapsed product time: `161.042361 s`.
- Peak RSS from `/usr/bin/time -l`: `8,791,343,104` bytes.
- Product commit: `ea0a3dcb045c1ec6983e607177e9290673c3aed7`.
- Temporal closure was not used as a hard acceptance gate; the persisted
  unwrapped/SBAS result is evaluated through the configured valid-pixel and
  reference/ISCE2 comparison policy.

Pair manifest digests:

- `20161207_20161231`: `4e676bd2c83896b0c4b3f02e63c9bc9c2870ec6c327fb8310cb2ba58630ff3bf`
- `20161207_20170124`: `0ec4265102b96961592fc7138011472a8e1dec40e009387a175abd09239e1541`
- `20161231_20170124`: `7f4d8cbcd286c460868f9f1ec8f0ad551cb42b2575f325e2e57d8a36a0be0bce`

A fresh `open_stack_generation()` verified the complete parent and all child
bindings after the process exited:

- Stack generation: `8ad8c95a33ec460687581ccba7f294fd`
- Stack manifest: `09173c6fd47e95699fe17b67c6accf60b7947ba50a6b55d13969b2e1c8d31879`
- Timeseries generation: `3ab38ed99ac24b259898555131fbcac6`
- Timeseries manifest: `2742623a8d36dd47e7f76a6ba2d1e0d717aab225709d16be12a84df515bc6343`

## Interpretation

This closes the fresh raw same-frame Geo 3-date/3-burst functional path and
parent/child reopen evidence. It is not a claim that the temporal phase must
close to zero after multilooking; the scientific acceptance comparison is the
same-crop/reference/ISCE2 product policy recorded separately.


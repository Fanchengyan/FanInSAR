# P18/P19 raw same-frame Radar Stack → SBAS qualification

Date: 2026-08-12

## Scope

Fresh local run of the current product tree from three same-frame Sentinel-1
IW1 SAFE ZIPs, using bursts 0/1/2, Pair coregistration, `(16, 40)` multilook,
CPU Torch, spatial unwrap, temporal reconciliation, and SBAS inversion.

The run was written under the repository-owned real directory
`.qualification-runs/p18-p19-firstframe-20260812`; the earlier attempt under
`/tmp` was correctly rejected because the macOS `/tmp` path contains a
symbolic-link component.

## Inputs

| Date | SAFE SHA-256 |
| --- | --- |
| 20161207 | `a5bdf079edf55e656b3f0598887afc587138613d380f39d345bc72dc8854ccfb` |
| 20161231 | `253c407d1a1f697d9c6da5fd1d3c91ce3eb15eb31f522d23f9d7166f3e4722c2` |
| 20170124 | `bc6b44bfe33a7aeffd243dd2b959a361b18bf1d990f6ea84b8a178ca73a5beec` |

## Results

- Status: `PASS` for the raw same-frame Radar functional scope.
- 3 scenes, 3 bursts, 3 pair IFGs, all shape `(261, 534)`.
- Timeseries shape `(3, 261, 534)` with 52,274 finite pixels.
- Elapsed product time: `279.272442 s`.
- Product commit: `ea0a3dcb045c1ec6983e607177e9290673c3aed7`.
- Pair manifest digests:
  - `20161207_20161231`: `fab357181f139e79944a6c9e45a1270bb4557e52125077cc80087b33723f44e6`
  - `20161207_20170124`: `66d7ff89159ded5cc52c53142517880bbc8cad4d00a6db8ab684554c8272a1ba`
  - `20161231_20170124`: `74a35d2e1bc2cb43d7260d29b0e5bafba46f1ecf4125fb9c9809dd7b9ac4f355`

The run published three immutable IFG artifacts, three IFG-bound unwrap
artifacts, a timeseries Zarr generation, and a complete Stack parent
generation. A fresh `open_stack_generation()` verified the exact pair order,
all child bindings, and the timeseries manifest digest:

- Stack generation: `55919738de534960adad2fdc2851ae23`
- Stack manifest: `e2fcae4a8867408398f63a9d2abcfa06111ffa623272388ba9793a2e15c8756a`
- Timeseries generation: `369a8f4b23e047eaa5d30f7e13adc5e9`
- Timeseries manifest: `6fefeb5ffda8e69fce71e47fd4ba673aabcdb79f496748cdd83da958840fb0be`

## Interpretation

This closes the fresh raw same-frame Radar functional path and parent/child
reopen evidence. It is not, by itself, the official Waymark activation event,
the dual-domain old/current performance packet, or the separate ISCE2 product
oracle; those remain governed by their own evidence files.


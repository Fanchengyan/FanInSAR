# P18/P19 real-generation transaction fault matrix

Date: 2026-08-12

The three-date, three-burst Radar generation from
`qualified-raw-safe-sbas-20260812/output` was copied into an isolated DATA2
work directory before fault injection. The source generation was not changed.

| Case | Expected result | Result |
| --- | --- | --- |
| fresh-process reopen | accept | PASS; generation `a793508db00941ccba3a2f5b776255c7` |
| tampered `STACK_CURRENT` control digest | reject | PASS |
| removed parent generation manifest | reject | PASS |
| one-byte time-series chunk mutation | reject | PASS |
| symlinked `STACK_CURRENT` | reject | PASS |

Machine-readable results:

`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/transaction-fault-matrix-20260812.2wOzVx/fault-matrix.json`

The existing focused transaction suite additionally covers lease pinning across
CURRENT replacement, hardlink/non-regular entries, quota preflight, and legacy
direct-layout rejection. This packet does not claim that a process crash,
source mutation during acquisition, or a concurrent stale writer has been
replayed against a full raw SAFE run; those scenarios remain open evidence
items.

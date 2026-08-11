# P18/P19 real Radar and Geo ROI matrix

Date: 2026-08-11

## Result

**PASS for the bounded two-date public-pipeline ROI and Stack-generation gate.**

The current public Stack wrapper ran from two real Sentinel-1 SAFE ZIPs through
coregistration, persisted scene generation, common-grid IFG, spatial and
temporal unwrap, SBAS inversion, transactional time-series publication, and a
complete Stack parent generation in all four combinations:

| Case | Result | Elapsed | Time-series shape | Valid pixels |
|---|---:|---:|---:|---:|
| Radar, no ROI | PASS | 30.983 s | 2 x 93 x 534 | 47,104 |
| Radar, ROI | PASS | 51.831 s | 2 x 177 x 1,707 | 21,882 |
| Geo, no ROI | PASS | 24.329 s | 2 x 117 x 112 | 6,916 |
| Geo, ROI | PASS | 36.806 s | 2 x 117 x 112 | 2,475 |

The ROI was `BoundingBox(98.3, 38.95, 98.7, 39.1, EPSG:4326)`. The Geo cases
used a frozen EPSG:32647 grid at 40 m. ROI selection was exercised through the
public `run_stack_pipeline` / `Stack.from_safes` surface rather than by editing
the internal configuration after construction.

The process completed with exit code zero. `/usr/bin/time -l` reported maximum
resident set size 7,267,057,664 bytes (about 6.77 GiB). All four products were
published through immutable transactional time-series generations and Stack
parent generations. A fresh interpreter reopened all four parent generations,
validated every IFG/unwrap/SBAS child binding, and released the reader pins:

| Case | Stack parent generation | SBAS generation |
|---|---|---|
| Radar, no ROI | `dc4ccba005eb46b69a48cf2721073383` | `804a6f80acb240d59dbba13f0e0e1223` |
| Radar, ROI | `9409e6a5c1214c68b80567e213ba00c3` | `bbce9a9658c14f92ad1893c390cd82c9` |
| Geo, no ROI | `cc273d0f06104ab9ada560ac5af980a3` | `4215eba603584ddf989244cbf1cbbaad` |
| Geo, ROI | `0ae6b646668045c39f21d4e9cca52c4f` | `2361ca58f6584a8e8d33a32bf3fd06ff` |

## Frozen inputs and artifacts

- Product commit: `225d8c4e340abcdfbdb42da5d6b73ca7ffcfa98a`
- Reference SAFE SHA-256:
  `e741ade308d94e3bd841136b20a192b977bd7d6201a445093f07f02529606ba2`
- Secondary SAFE SHA-256:
  `2e27f9eb9420a0a59a27e1563d0002883a35cabcc51512391dff37907afe0e8a`
- Summary:
  `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/PROPOSAL-0018/roi-matrix-stack-generation-retry-20260811/summary.json`
- Summary SHA-256:
  `56b65af390a3bdbc111f58ad6e2b994bb342c5a2ecfbb1e5540aa26f692f75a`

## Scope boundary

This closes the real public-surface ROI/no-ROI requirement in both Radar and
Geo domains and demonstrates the new complete-result transaction on real
artifacts. It does not replace the proposal's remaining three-date,
three-burst Geo breadth, complete old/new Stack performance repetitions, Linux
cgroup hard-limit evidence, or human-approved physical temporal error budget.

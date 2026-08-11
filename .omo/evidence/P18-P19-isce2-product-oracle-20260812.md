# P18/P19 ISCE2 product oracle

Date: 2026-08-12

## Scope

This packet closes the product-level ISCE2 comparison item for the existing
same-grid campaign. It is an oracle comparison, not a claim of byte identity:
ISCE2 and FanInSAR use different internal FFT/window and invalid-pixel
handling, so the acceptance comparison is made on common finite pixels after
the same crop, look factors, and Goldstein exponent are applied.

## Evidence

Source campaign:

- `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/campaigns/fullframe-parity`
- 20 common Sentinel-1 bursts across IW1/IW2/IW3
- common output shape `(5651, 6829)` with zero row/column shift
- multilook `(2, 10)` and Goldstein `alpha=0.5`
- no ESD or unwrap in the wrapped-product oracle

The frozen metrics in `output/fullframe_final_metrics.json` report:

| Comparison | Result |
| --- | ---: |
| raw complex correlation | 0.98150 |
| filtered complex correlation | 0.98493 |
| raw phase RMSE | 0.2295 rad |
| filtered phase RMSE | 0.2049 rad |
| filtered core phase RMSE | 0.1581 rad |
| filtered core complex correlation | 0.99010 |
| common valid pixels | 29,746,579 |

The per-swath filtered complex correlations are 0.97613 (IW1), 0.98380
(IW2), and 0.99494 (IW3). The independent two-date, two-look unwrapped
comparison in
`/Volumes/DATA2/TEST_sentinel-1/faninsar-radar-parity-20260718/diagnostics/fan_vs_isce2_2x10_unwrapped_metrics.json`
reports residual RMSE `0.3034 rad` after a global (2\pi) offset and valid
fraction `0.86697`.

## Decision

PASS for the ISCE2 parity criterion at product level: the same-crop,
same-look, same-filter wrapped products have high complex correlation and
sub-quarter-radian phase RMSE on the common core. The unwrapped residual is
recorded as a diagnostic, not a closure gate; multi-looking and independent
unwrapping legitimately leave non-zero residual phase.

Displacement-level byte equality is not required and is not claimed. Any
future displacement comparison must bind wavelength, sign convention,
reference pixel, and unwrapping offset before computing metres.

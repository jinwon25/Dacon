# Exact OOF threshold calibration sprint - 2026-07-17

## Decision

Promote one controlled submission candidate:
`submissions/blend_best_meta_g3_thr10_off575.csv`.

The policy adds 575 kWh to group-3 predictions that are at least 10% of capacity. It was selected
only from Q1/Q2 development consistency, before inspecting locked H2. The candidate then improved
locked H2 score, 1-NMAE, and FICR, and passed the predeclared promotion checks. Groups 1 and 2 are
byte-for-byte unchanged from the publicly confirmed best submission `1494307`.

This is a controlled public probe, not evidence that 0.65 has been reached. Scaling the locked
group-3 score delta by the competition's three-group macro average gives a rough public delta of
`+0.001660`, or about `0.643316` from the current public `0.641655`. Public/private distribution
shift and the negative December result make that estimate uncertain.

## Selection protocol

Three calibration families were searched on the frozen exact meta-gate OOF vector:

1. prediction-thresholded constant offset;
2. global affine scale and offset;
3. one-breakpoint piecewise constant offset.

A candidate had to improve score, 1-NMAE, and FICR in both Q1 and Q2. Family selection used the
minimum of the two quarterly score gains and a fixed simplicity tolerance of `0.00125`; H2 was not
used for parameter or family selection. Within the threshold-offset family, a fixed `0.00025`
near-peak tolerance chose the policy that changed fewer rows.

| family | selected policy | min Q1/Q2 delta | locked H2 score | locked 1-NMAE | locked FICR |
|---|---|---:|---:|---:|---:|
| threshold offset | ratio >= 0.10, +575 kWh | +0.005584 | +0.004981 | +0.000855 | +0.009108 |
| affine | 1.022 x prediction + 300 kWh | +0.006628 | +0.002130 | +0.000889 | +0.003370 |
| piecewise offset | <40%: +450; >=40%: +575 kWh | +0.006477 | +0.003857 | +0.001007 | +0.006707 |

The simpler threshold-offset policy was within the predefined tolerance of the strongest
development family and had the best locked score of the three family winners. The latter is
reported as a diagnostic only and did not drive selection.

## Locked validation

The selected policy raised group-3 locked H2 score from `0.610700` to `0.615681`.

| month | score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|
| July | +0.014011 | +0.005404 | +0.022617 |
| August | +0.013064 | +0.002120 | +0.024007 |
| September | +0.020703 | +0.004048 | +0.037358 |
| October | +0.013238 | +0.004289 | +0.022187 |
| November | +0.007320 | +0.005661 | +0.008979 |
| December | -0.011109 | -0.007593 | -0.014626 |

Five of six months improved. A 2,000-sample day-block bootstrap was positive in `89.0%` of draws
with median `+0.004686`, but its 95% interval `[-0.002661, +0.012230]` crossed zero. This is the
main reason to treat the CSV as one public probe rather than a new structural solution.

## Submission guard

- rows: 8,760;
- changed group-3 rows: 7,964 (`90.91%`);
- mean / p95 / maximum movement: `522.75 / 575.00 / 575.00 kWh`;
- source-cache parity maximum error: `0.00098 kWh`;
- groups 1/2 unchanged: yes;
- schema, finite-value, and capacity-bound checks: passed.

The broad changed-row coverage is intentional: the exact OOF evidence identifies a systematic
underprediction offset rather than another sparse member-selection opportunity. It also increases
public risk compared with the earlier meta-gate, so no second calibration CSV was emitted.

## Reproduction

```bash
python -m experiments.threshold_piecewise_calibration
python -m pytest -q
```

Machine-readable output:
`artifacts_final/threshold_calibration/threshold_calibration_report.json`.

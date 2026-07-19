# Issue-time / season blocked rolling validation

## Outcome

The current fine meta-gate is a verified public improvement, but it is not a
robust route to 0.65. It must not be extrapolated or strengthened automatically.
The next score attempt must add a genuinely new predictive signal.

The audit compares the public-best fine policy (`p >= .545`, alpha `.50`) with
the previous public-proven policy (`p >= .55`, alpha `.25`) on the locked H2
exact OOF lineage.

| Check | Result | Guard |
|---|---:|---:|
| H2 group-3 score delta | +0.0006389 | positive |
| H2 1-NMAE delta | +0.0000216 | non-negative |
| H2 FICR delta | +0.0012561 | non-negative |
| Worst month (2024-10) | -0.0004071 | non-negative |
| Positive month fraction | 66.7% | diagnostic |
| Winter rolling block | -0.0004018 | non-negative |
| Issue-block bootstrap positive fraction | 87.65% | at least 90% |
| Issue-block bootstrap q05 | -0.0002784 | non-negative |

The candidate therefore fails the worst-month and bootstrap guards.

## Why the new split is stricter

`data_available_kst_dtm` is treated as the atomic dependency unit. All target
hours originating from one GFS issue remain in the same block, even when the
forecast horizon crosses midnight, a month boundary, or a season boundary.
Issue cycles are assigned by the centre timestamp of their target horizon.

Rolling folds use expanding complete meteorological seasons:

1. JJA -> SON: score delta +0.0011271
2. JJA + SON -> DJF: score delta -0.0004018

Bootstrap sampling also resamples complete issue cycles within seasons. It does
not treat correlated hourly horizons as independent observations.

## Public-transfer audit

Two real leaderboard probes are connected to their locked local estimates.

| Probe | Locked macro delta | Public delta | Transfer |
|---|---:|---:|---:|
| Fine meta-gate | +0.0002130 | +0.0000918 | +0.431x |
| Broad settlement composite | +0.0036790 | -0.0038893 | -1.057x |

Only 50% of observed probes preserve the local direction. The broad composite
has a complete sign reversal, so automatic public-score projection is disabled
until at least three direction-consistent probes exist.

The public best is `0.6417471627`; the gap to `0.65` is `0.0082528373`, or about
89.9 times the last verified public improvement. Calibration-only iteration is
therefore both statistically unsafe and too small in expected effect.

## Promotion contract

A calibration candidate can be promoted only when all of the following hold:

- locked score, 1-NMAE, and FICR deltas are non-negative;
- the worst complete issue-month score delta is non-negative;
- issue-cycle bootstrap q05 is non-negative;
- bootstrap positive fraction is at least 90%;
- public-transfer evidence has at least three direction-consistent probes.

Machine-readable evidence is stored in
`artifacts_final/validation/blocked_rolling_validation_report.json`. The runner
is `experiments/blocked_rolling_validation.py`.

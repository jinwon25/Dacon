# Exact OOF meta-gate fine sweep - 2026-07-17

## Decision

Submission `1494670`, `blend_best_crossg3_traj_meta_finesweep.csv`, is the selected public best at
`0.6417471627` (`1-NMAE 0.8754733572`, `FICR 0.4080209682`). It improved submission `1494307` by
only `+0.0000917901`; the gain was positive in both components but remains far below the local
extrapolation. The policy was chosen without reading the locked H2 result: train the meta classifier
on Q1, select threshold/alpha on Q2, retrain on H1, and evaluate the selected policy once on H2.

The selected policy is probability threshold `0.545` and additional cross-group alpha `0.50`.
The search covered 527 combinations: thresholds `0.500..0.650` in steps of `0.005` and alphas
`0.100..0.500` in steps of `0.025`.

## Validation result

| policy | H2 changed rows | score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|---:|
| reference `p55/a25` | 323 | +0.000232 | +0.000092 | +0.000372 |
| selected `p545/a50` | 345 | +0.000871 | +0.000114 | +0.001628 |
| selected minus reference | - | +0.000639 | +0.000022 | +0.001256 |

All five seed-specific H2 deltas are positive (`+0.000734..+0.000897`) and all six complete H2
months improve. The 2,000-sample day-block bootstrap positive fraction rises from `89.9%` to
`95.2%`; its 5% quantile rises from `-0.000063` to `+0.000014`.

Those figures compare each policy with the pre-gate trajectory. For promotion against the current
public incumbent, the paired selected-minus-`p55/a25` result is the relevant contract:

| paired incremental result | value |
|---|---:|
| group-3 locked score delta | +0.000639 |
| expected competition macro delta | +0.000213 |
| improved months | 4 / 6 |
| day-bootstrap positive fraction | 87.35% |
| day-bootstrap q05 | -0.000313 |

The paired q05 is below the current automatic-promotion floor of `-0.00025`. The service therefore
correctly rejects automatic submission while retaining the CSV and compact evidence for manual
review. No current run or selection row was changed during adaptation.

## Candidate integrity

- 8,760 rows, expected five columns, no missing/non-finite values or duplicate keys;
- groups 1 and 2 exactly unchanged;
- 1,045 material group-3 changes;
- mean / p95 / maximum absolute movement: `25.40 / 234.75 / 480.57 kWh`;
- file size: `789,134 bytes`;
- current candidate SHA256: `c00273dd4836f134547c43e0e6dcd38e57f864dfc4e10688913d143c5f2b6345`.

The fine sweep transferred as a credible incremental probe, not a route to `0.65` by itself. Its
realized gain was about 43% of the `+0.000213` locked macro estimate. The larger target still
requires a new source of group-3 predictive signal.

Machine-readable evidence is in
`artifacts_final/meta_gate_sweep/meta_gate_policy_sweep_report.json`; the DB-free standard
evaluation is in `artifacts_final/meta_gate_sweep/agent_evaluation.json`.

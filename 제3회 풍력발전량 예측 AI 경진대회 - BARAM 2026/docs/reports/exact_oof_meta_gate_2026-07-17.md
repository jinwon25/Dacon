# Exact OOF driver and settlement meta-gate sprint - 2026-07-17

## Decision

Select public submission `1494307`, `blend_best_crossg3_traj_meta25_p55.csv`.

The complete group-1/group-2 driver lineage is now reconstructed, closing the validation
infrastructure bottleneck identified on 2026-07-14. A settlement-aware meta-gate is the first new
method on this exact surface to improve both the Q1-to-Q2 development split and the locked H1-to-H2
split with both official score components positive. Its proxy and bootstrap evidence is not strong
enough to replace the confirmed best without public feedback, so it was submitted as a controlled
probe. The public result confirmed it as the new best:

- file: `submissions/blend_best_crossg3_traj_meta25_p55.csv`;
- submitted title: `submission24 edit`;
- submission ID / time: `1494307` / `2026-07-17 12:35:35`;
- groups 1/2: exactly unchanged;
- group 3 changed rows: 969 / 8,760;
- mean / p95 / maximum absolute movement: `11.65 / 106.24 / 240.28 kWh`.

## Public result

| submission | score | 1-NMAE | FICR |
|---|---:|---:|---:|
| previous best `1491284` | 0.6414800049 | 0.8754395685 | 0.4075204413 |
| meta-gate `1494307` | 0.6416553726 | 0.8754552834 | 0.4078554618 |
| delta | +0.0001753677 | +0.0000157149 | +0.0003350205 |

The gain is real but remains small. Its composition matches the method's intent: nearly all of the
improvement came from selecting rows that preserve or enter FICR settlement bands, rather than from
reducing broad average error. This submission is the new selected best, while the remaining work
must target a larger structural gain instead of another micro-weight sweep.

## Exact group-1/group-2 lineage

The reconstructed path is:

`blend_v1 -> over115 against cal125 -> SCADA 5% -> weighted member 2% with 4% agreement gate`.

The missing historical SCADA 5% member is recovered algebraically from the archived SCADA 15%
submission, then the rebuilt validation member is affinely aligned to that test lineage. The final
2025 vectors reproduce `artifacts_final/lineage_inputs/base_pre_cross.csv` to numerical precision:

| target | test parity MAE kWh | maximum error kWh | validation weighted-gate rows |
|---|---:|---:|---:|
| group 1 | 0.00000329 | 0.0000310 | 2,772 |
| group 2 | 0.00000234 | 0.0000200 | 4,368 |
| group 3 | effectively zero | 0.000000000011 | unchanged |

The group-3 stages also reproduce the earlier exact cache: blend/calibration/over are identical and
the SCADA 5% stage differs only by float32 serialization noise.

## Alternative methods tested

### Exact-domain residual stacking

ExtraTrees residual models were trained directly on exact 2024 OOF group-1/group-2/current-group-3
predictions, eliminating the old actual-driver-to-predicted-driver mismatch. A broad leaf-30 policy
improved locked H2 by `+0.001125`, but only 3/6 H2 months improved, the day-bootstrap 95% interval
crossed zero, and weighted/final-pool proxy H2 signs were negative. A temporal-consensus version
reversed sign under seed ensembling. This family is rejected.

### Errors-in-variables driver calibration

Affine, ridge, and isotonic mappings calibrated exact group-1/group-2 OOF predictions toward their
actual values before cross-group transfer. Isotonic 75% calibration improved Q2 by `+0.001879`,
mostly through FICR, but reversed to `-0.001206` on locked H2. Every leading linear/ridge policy was
also H2-negative. This family is rejected.

### Settlement-aware meta-gate

The final method leaves the confirmed 25% cross-group and 5% trajectory policies intact. For each
eligible/actionable OOF row, it labels whether a small additional move toward the cross-group member
raises that row's correctly normalized official NMAE/FICR contribution. Twelve features describe
the exact group-1/group-2 drivers, pre-cross/current/member group-3 predictions, disagreement, and
hour of day.

Five ExtraTrees classifiers use `min_samples_leaf=50`. The policy selected wholly inside H1 applies
an additional 25% of the remaining member-current difference when mean benefit probability is at
least 0.55.

| split | changed rows | score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|---:|
| Q1 train -> Q2 evaluate | 153 | +0.001635 | +0.000165 | +0.003105 |
| H1 train -> locked H2 evaluate | 323 | +0.000232 | +0.000092 | +0.000372 |

All five locked-H2 seed deltas were positive, ranging from `+0.000172` to `+0.000233`.

## Robustness limits

Four of six locked-H2 months improved. The 2,000-sample day-block bootstrap had median
`+0.000229` and a positive fraction of `89.65%`, but its 95% interval was
`[-0.000158, +0.000619]`.

Applying the fixed H2 correction vector as a stress test to the old broad proxy surfaces gave H2
score deltas of `-0.000206`, `-0.000880`, and `+0.000016` for weighted/global/final-pool. These
surfaces no longer match the public lineage, so they are not allowed to override the exact OOF
result; they do show that threshold settlement remains fragile. This is why the CSV is a controlled
public probe rather than the selected submission.

## Reproduction

```bash
python -m experiments.exact_driver_oof
python -m experiments.exact_oof_meta_gate
python -m pytest -q
```

Machine-readable outputs after artifact consolidation:

- `artifacts_final/lineage/exact_driver_oof.npz`;
- `artifacts_final/lineage/exact_driver_oof_report.json`;
- `artifacts_final/meta_gate/meta_gate_cache.npz`;
- `artifacts_final/meta_gate/meta_gate_report.json`.

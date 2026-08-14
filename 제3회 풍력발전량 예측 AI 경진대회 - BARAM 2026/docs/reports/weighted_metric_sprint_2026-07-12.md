# Weighted Metric Sprint - 2026-07-12

## Public Result

- Submission ID: `1488084`
- File: `blend_stack5_weighted_g12_2_agree4.csv`
- Submitted at: `2026-07-12 23:22:15`
- Score: `0.6403102237`
- 1-NMAE: `0.8753191781`
- FICR: `0.4053012693`
- Gain over `blend_over115_scada_stack5.csv`: `+0.0000449963`

The safest probe improved both score components and is now the public best. The other weighted probe
CSVs were removed after this result; only the submitted CSV remains in the active submission folder.

## Objective

The official FICR is not an unweighted hit-rate average. Each eligible hour's settlement is
weighted by actual generation before division by the theoretical maximum settlement. The local
metric and tests were corrected first, then a new LightGBM family was trained only on eligible rows
with sample weights that interpolate between uniform and generation-proportional weighting.

Candidates used generation fractions of 0%, 35%, 70%, and 100%. All validation uses the 2024
time holdout. The single 2025-01-01 boundary row is retained in the full metric but excluded from
monthly-stability summaries because it does not form a meaningful month.

## Corrected Holdout Results

| target | uniform score | best weighted single | selected blend | blend 1-NMAE | blend FICR | score gain vs uniform |
|---|---:|---:|---:|---:|---:|---:|
| `kpx_group_1` | 0.668884 | 0.671762 (35%) | 0.675164 | 0.885921 | 0.464406 | +0.006279 |
| `kpx_group_2` | 0.670727 | 0.672460 (35%) | 0.674375 | 0.882318 | 0.466431 | +0.003647 |
| `kpx_group_3` | 0.579177 | 0.586960 (100%) | 0.587982 | 0.862053 | 0.313912 | +0.008805 |

Selected weights:

- Group 1: 24.15% uniform, 69.36% generation35, 6.47% generation70.
- Group 2: 6.73% uniform, 79.74% generation35, 0.68% generation70, 12.84% generation100.
- Group 3: 19.00% generation35, 6.95% generation70, 74.05% generation100.

The standalone weighted family should not replace the current submission. Group 3 remains much
weaker in 1-NMAE than the existing CatBoost-based route, so every new blend keeps group 3 equal to
the public-best base. The new family is used only as a small groups 1/2 correction member.

## Monthly Stability

Relative to the uniform eligible-only candidate:

| target | improved months | mean monthly score delta | median monthly score delta |
|---|---:|---:|---:|
| `kpx_group_1` | 8 / 12 | +0.005502 | +0.006054 |
| `kpx_group_2` | 7 / 12 | +0.003489 | +0.007521 |
| `kpx_group_3` | 9 / 12 | +0.005328 | +0.004754 |

The groups 1/2 improvement is not confined to one month, but several losing months remain. This is
why submission candidates use only 2-3% injection with a base/member agreement gate.

## Generated Submission Candidates

All candidates use `blend_over115_scada_stack5.csv` as the base and leave group 3 unchanged up to
CSV floating-point serialization noise (maximum absolute difference below `4e-12`).

| file | injection | change profile |
|---|---|---|
| `blend_stack5_weighted_g12_2_agree4.csv` | G1/G2 2%, disagreement <= 4% | safest; abs-mean delta 3.38 / 4.78 kWh |
| `blend_stack5_weighted_g12_3_agree3.csv` | G1/G2 3%, disagreement <= 3% | narrower gate; abs-mean delta 2.80 / 4.26 kWh |
| `blend_stack5_weighted_g1_3_g2_2_agree5.csv` | G1 3%, G2 2%, disagreement <= 5% | broader probe; abs-mean delta 7.86 / 6.54 kWh |
| `blend_stack5_weighted_g12_2_high20_agree6.csv` | G1/G2 2%, base >= 20%, disagreement <= 6% | high-generation probe; abs-mean delta 5.64 / 5.11 kWh |

Recommended first submission: `blend_stack5_weighted_g12_2_agree4.csv`. It has the smallest broad
change, matches the local improvement direction, and keeps the known weak group 3 member out.

## Validation and Next Step

- Metric unit tests: 3 passed.
- All five generated CSVs: 8,760 rows, exact sample columns and keys, no missing values, and all
  predictions within group capacities.
- Prediction cache diagnostics exclude incomplete months from stability aggregation by default.

The next modeling run should rebuild `artifacts_blend_metric/prediction_cache.npz` with the corrected
metric and formally optimize the weighted member together with the GBDT/CatBoost base candidates.
Until that cache exists, avoid increasing the public-base injection above the conservative probes.

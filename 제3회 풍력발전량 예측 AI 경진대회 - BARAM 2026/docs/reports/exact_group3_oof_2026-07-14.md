# Exact group-3 OOF reconstruction - 2026-07-14

## Outcome

The group-3 test lineage before cross-group correction has been reconstructed as an OOF surface:

`blend_v1 -> over115 against cal125 -> SCADA 5% -> cross-group 25% -> trajectory 5%`.

This does not create a new submission. It replaces the least reliable part of model research:
selecting corrections against broad OOF proxies that differ materially from the public base.

## Recovered components

The old SCADA Hist stack was retrained after fixing its cache writer. The new run now retains the
selected validation/test vectors, indices, truth, weights, and official competition metric in
`artifacts_scada_stack_hist/prediction_cache.npz`.

The rebuilt group-3 test vector has `0.9999215` correlation with the historical SCADA member
algebraically recovered from the archived submissions. The remaining difference is almost entirely
the historical calibration:

`historical_scada = 0.925644762 * rebuilt_scada - 269.420969`.

After alignment, mean absolute test difference is `16.12 kWh` and the 95th percentile is
`33.83 kWh`. The aligned validation member scores `0.56618` under the corrected official metric.

The two CatBoost components of `blend_v1` and the eligible CatBoost component of `cal125` were
retrained with their original seeds, iterations, feature surface, and calibration settings. Their
NMAE values reproduce the historical report to numerical precision. Historical FICR values do not
match because that report predates the audit that changed local FICR from unweighted hit rate to
actual-generation-weighted settlement; current metrics use the corrected official implementation.

## Reconstructed metric surface

| stage | H1 score | H2 score | full score | full 1-NMAE | full FICR |
|---|---:|---:|---:|---:|---:|
| `blend_v1` | 0.57185 | 0.60995 | 0.59047 | 0.86883 | 0.31211 |
| `cal125` | 0.56963 | 0.60123 | 0.58508 | 0.86739 | 0.30277 |
| `over115` | 0.57091 | 0.61200 | 0.59100 | 0.86869 | 0.31330 |
| aligned SCADA | 0.55317 | 0.57978 | 0.56618 | 0.85603 | 0.27634 |
| exact pre-cross 95/5 | 0.57161 | 0.61033 | 0.59054 | 0.86868 | 0.31239 |
| cross-group 25% | 0.57204 | 0.61003 | 0.59060 | 0.86897 | 0.31224 |
| trajectory 5% | 0.57217 | 0.60998 | 0.59065 | 0.86899 | 0.31231 |

Cross-group 25% versus exact pre-cross:

| period | score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|
| H1 | +0.000431 | +0.000303 | +0.000559 |
| H2 | -0.000308 | +0.000278 | -0.000894 |
| full | +0.000068 | +0.000291 | -0.000155 |

Trajectory versus cross-group 25%:

| period | score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|
| H1 | +0.000135 | +0.000028 | +0.000242 |
| H2 | -0.000049 | +0.000015 | -0.000113 |
| full | +0.000045 | +0.000022 | +0.000068 |

The trajectory direction is positive on the exact full-year OOF and on the public leaderboard,
which strengthens its interpretation. The much larger public cross-group gain is not reproduced
locally; this confirms a real 2024-to-2025/public-sample shift and argues against choosing another
large correction from 2024 alone.

## Remaining limitation

The pre-cross group-3 base follows the exact archived test lineage. Cross-group validation still
uses the retained weighted OOF forecasts as group-1/group-2 drivers, because the complete public
group-1/group-2 OOF lineage has not yet been rebuilt. This limitation is recorded in the report and
is why no new candidate is generated from this reconstruction.

## Reproduction

```bash
python -m experiments.scada_proxy_stack_hist \
  --artifact-dir artifacts_scada_stack_hist \
  --output artifacts_scada_stack_hist/scada_proxy_stack_hist.csv
python -m experiments.oof_lineage_audit
python -m experiments.exact_group3_oof
python -m pytest -q
```

Retained generated artifacts:

- `artifacts_scada_stack_hist/prediction_cache.npz`;
- `artifacts_oof_lineage/scada_group3_aligned_cache.npz`;
- `artifacts_oof_lineage/exact_group3_oof.npz`;
- JSON reports beside those caches.

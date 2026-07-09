# BARAM 2026 Wind Power Forecasting

Working repository for the DACON BARAM 2026 wind-power forecasting competition.

Goal: predict hourly 2025 wind-power generation for three KPX groups from LDAPS/GFS day-ahead NWP forecasts. The score is `0.5 * (1-NMAE) + 0.5 * FICR`, evaluated only when actual generation is at least 10% of installed capacity.

## Current Best

| submission_id | file | score | 1-NMAE | FICR |
|---:|---|---:|---:|---:|
| 1480959 | `blend_over115_scada_stack5.csv` | 0.6402652274 | 0.8753153766 | 0.4052150782 |

`blend_over115_scada_stack4.csv` dropped slightly to `0.6402123189`, and `blend_over115_scada_stack10.csv` dropped to `0.6397287088`, so the global SCADA stack weight is currently best at 5%.

## Active Submission Candidates

Active candidates are kept in `submissions/` root. Older generated CSVs are moved to `submissions/archive/`.

Recommended next order:

1. `blend_over115_scada_g12_5_g3_3.csv`
2. `blend_over115_scada_g12_6_g3_3.csv`
3. `blend_over115_scada_stack6.csv`
4. `blend_stack5_antipowercurve2_g12_t06.csv`

Suggested titles:

- `Blend over115 scada g12 5 g3 3`
- `Blend over115 scada g12 6 g3 3`
- `Blend over115 scada stack6`
- `Blend stack5 antipowercurve2 g12 t06`

## Project Layout

```text
.
|-- README.md
|-- requirements.txt
|-- train.py                 # baseline/hybrid model training
|-- inference.py             # submission generation from saved models
|-- src/
|   |-- features.py          # NWP feature engineering
|   `-- metrics.py           # local 1-NMAE/FICR implementation
|-- experiments/
|   |-- blend_experiment.py
|   |-- make_submission_blends.py
|   |-- power_curve_residual.py
|   |-- selective_member_blend.py
|   |-- scada_proxy_stack_hist.py
|   |-- scada_proxy_direct.py
|   |-- recent_specialist.py
|   `-- analog_experiment.py
|-- docs/
|   |-- modeling_strategy.md
|   `-- reports/
|-- data/                    # ignored raw competition data
|-- artifacts*/              # ignored trained models/reports
`-- submissions/
    |-- results.csv          # tracked leaderboard log
    |-- *.csv                # active ignored submission candidates
    `-- archive/             # older ignored candidates
```

## Main Commands

Hybrid LightGBM/CatBoost model:

```bash
python train.py --data-dir data --artifact-dir artifacts_hybrid --feature-set base --catboost-targets kpx_group_3
python inference.py --data-dir data --artifact-dir artifacts_hybrid --output submissions/hybrid_lgbm_cat_g3_full_cal.csv --calibration-strength 1.0
```

Metric-optimized blend:

```bash
python -m experiments.blend_experiment --data-dir data --artifact-dir artifacts_blend --output submissions/blend_v1.csv
```

SCADA proxy stack:

```bash
python -m experiments.scada_proxy_stack_hist --data-dir data --artifact-dir artifacts_scada_stack_hist --output submissions/scada_proxy_stack_hist.csv
```

Manual member injection:

```bash
python -m experiments.make_submission_blends --output submissions/blend_over115_scada_stack5.csv --weights 0.05
python -m experiments.make_submission_blends --output submissions/blend_over115_scada_g12_5_g3_3.csv --weights kpx_group_1=0.05,kpx_group_2=0.05,kpx_group_3=0.03
python -m experiments.make_submission_blends --output submissions/blend_over115_scada_g12_6_g3_3.csv --weights kpx_group_1=0.06,kpx_group_2=0.06,kpx_group_3=0.03
```

Power-curve residual member:

```bash
python -m experiments.power_curve_residual --data-dir data --artifact-dir artifacts_power_curve --output submissions/power_curve_residual.csv
python -m experiments.selective_member_blend --base submissions/blend_over115_scada_stack5.csv --member submissions/power_curve_residual.csv --output submissions/blend_stack5_powercurve_sel5_g12_t06.csv --weights kpx_group_1=0.05,kpx_group_2=0.05,kpx_group_3=0.0 --max-disagreement 0.06
```

## Modeling Notes

- The strongest baseline is not a single model but a blend of LightGBM/CatBoost candidates and SCADA proxy stack members.
- `blend_v1` moved the public score from `0.6367` to `0.6395`.
- Manual extrapolation from `cal125` toward `blend_v1` peaked around `over115`.
- SCADA proxy stack at 5% injection moved the score to `0.6402652274`.
- `stack4` and `stack10` showed that the global optimum is narrow around 5%; next tests should be group-wise, not more global weight tuning.
- Power-curve residual selective injection was tested in `blend_stack5_powercurve_sel5_g12_t06.csv` and dropped to `0.6400608956`; do not expand this family unless a stronger local validation signal is found.
- Because the failed power-curve candidate lowered G1/G2 and hurt FICR, `blend_stack5_antipowercurve2_g12_t06.csv` is kept only as a tiny reverse-direction probe.

## Rule Compliance

- Test-period actual generation and test-period SCADA are not used.
- All SCADA usage is restricted to train-period proxy modeling.
- No external weather data or remote inference API is used.
- Generated model artifacts and large submission CSVs remain ignored.

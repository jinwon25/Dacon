# BARAM 2026 Wind Power Forecasting

Working repository for the DACON BARAM 2026 wind-power forecasting competition.

Goal: predict hourly 2025 wind-power generation for three KPX groups from LDAPS/GFS day-ahead NWP forecasts. The score is `0.5 * (1-NMAE) + 0.5 * FICR`, evaluated only when actual generation is at least 10% of installed capacity.

## Current Best

| submission_id | file | score | 1-NMAE | FICR |
|---:|---|---:|---:|---:|
| 1480959 | `blend_over115_scada_stack5.csv` | 0.6402652274 | 0.8753153766 | 0.4052150782 |

`blend_over115_scada_stack10.csv` dropped to `0.6397287088`, so the SCADA stack member is useful but should be injected lightly.

## Active Submission Candidates

Active candidates are kept in `submissions/` root. Older generated CSVs are moved to `submissions/archive/`.

Recommended next order:

1. `blend_over115_scada_stack4.csv`
2. `blend_over115_scada_stack6.csv`
3. `blend_over115_scada_g12_6_g3_3.csv`

Suggested titles:

- `Blend over115 scada stack4`
- `Blend over115 scada stack6`
- `Blend over115 scada g12 6 g3 3`

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
python -m experiments.make_submission_blends --output submissions/blend_over115_scada_g12_6_g3_3.csv --weights kpx_group_1=0.06,kpx_group_2=0.06,kpx_group_3=0.03
```

## Modeling Notes

- The strongest baseline is not a single model but a blend of LightGBM/CatBoost candidates and SCADA proxy stack members.
- `blend_v1` moved the public score from `0.6367` to `0.6395`.
- Manual extrapolation from `cal125` toward `blend_v1` peaked around `over115`.
- SCADA proxy stack at 5% injection moved the score to `0.6402652274`.
- `stack10` showed that more SCADA stack is not automatically better; FICR is the limiting term.

## Rule Compliance

- Test-period actual generation and test-period SCADA are not used.
- All SCADA usage is restricted to train-period proxy modeling.
- No external weather data or remote inference API is used.
- Generated model artifacts and large submission CSVs remain ignored.

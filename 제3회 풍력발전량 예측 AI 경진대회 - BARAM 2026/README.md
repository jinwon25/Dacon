# BARAM 2026 Wind Power Forecasting

Working repository for the DACON BARAM 2026 wind-power forecasting competition.

Goal: predict hourly 2025 wind-power generation for three KPX groups from LDAPS/GFS day-ahead NWP forecasts. The score is `0.5 * (1-NMAE) + 0.5 * FICR`, evaluated only when actual generation is at least 10% of installed capacity.

## Current Best

| submission_id | file | score | 1-NMAE | FICR |
|---:|---|---:|---:|---:|
| 1494670 | `blend_best_crossg3_traj_meta_finesweep.csv` | 0.6417471627 | 0.8754733572 | 0.4080209682 |

The fine meta-gate sweep improved submission `1494307` by `+0.0000917901`. Most of the gain came
from FICR (`+0.0001655064`) while 1-NMAE increased by only `+0.0000180738`, confirming that the
threshold-stable direction transfers but is saturated. Reaching `0.65` still requires
`+0.0082528373`.

## Active Submission Candidate

The only retained manual comparison candidate is
`submissions/blend_best_g2_component_safe_affine.csv`. It changes group 2 by the
small affine policy `0.996 * prediction + 50 kWh`, leaves groups 1/3 byte-equivalent,
and passed Q1/Q2 all-component checks plus the locked H2 score/component checks.
Its locked group-2 gain is only `+0.0009313` (`~+0.0003104` macro before transfer),
so it is not evidence of a path to `0.65` and is intentionally not auto-selected:
all 8,760 group-2 rows change, exceeding the service's 25% coverage guard.

Submission `1494986`
(`blend_best_spatiotemporal_multitask20.csv`) scored `0.6415388286`, or
`-0.0002083341` versus the public best. Its 1-NMAE improved by `+0.0000796343`, but
FICR fell by `-0.0004963026`; service run 7 is publicly rejected and archived.

The next independent research track is a leakage-safe independent operational
forecast. NOAA GEFS spread and mean/disagreement screens were rejected before any
2025 collection. The current priority is KMA UM global N128, because it is an
independent operational model and its historical 2024 forecasts remain queryable
from the official KMA APIHub. The collector is implemented and fails closed without
a user-issued `KMA_API_KEY`; no key is accepted on the command line or written to
artifacts. No experiment may be promoted until
the exact run publication time, raw files, checksums, license, and per-row causal
join pass the external-data manifest guard. Retrospective Open-Meteo history is
research-only and blocked from submission use. See
`docs/reports/external_data_pretrained_audit_2026-07-18.md`.

The GEFS operational-archive implementation collected and audited the
2023–2024 screen period (6,579 source objects, 2.5 GB, zero timing violations) and
decoded 157,896 time/grid rows. Its first 10 m component-spread residual model was
rejected at Q2 before opening H2: the best policy gained score/FICR but lost
`-0.00001087` 1-NMAE, and no all-component policy passed. No 2025 GEFS data or
submission CSV was produced from the failed family. Its rejected raw GRIB payloads
were pruned after preserving decoded features, source URLs/checksums, request plans,
and diagnostic reports.

NOAA CFSv2 operational forecasts were then tested as a second, independently
initialized public forecast source. The 2024 H1 screen retained 4,368 hourly
targets across nine grids with zero timing violations under a conservative
30-hour publication bound. Direct residual correction failed the all-component
Q2 contract, and the CFSv2 meta-risk gate produced no seed-stable policy beyond
an identical no-weather control. H2 and 2025 therefore remained unopened and no
submission CSV was produced. The rejected 130,786,197-byte raw payload was pruned;
decoded features, exact source URLs/byte ranges, hashes, plans, and reports remain
for audit and reproducible redownload.

The broad settlement composite scored `0.6377660509`, falling `-0.0038893217` below the previous
best with both components lower; it is rejected and archived. The related broad `+575 kWh`
group-3 candidate is also archived without submission because the public result invalidated the
shared wide-calibration assumption. The phase/regime and selective issue-lag families are rejected.
The structural candidate has meaningful local evidence but projects to roughly a `+0.00169`
macro gain before public-transfer uncertainty, so it is a step toward `0.65`, not evidence that
the target has already been reached.

Future submission candidates must be written under `submissions/`; retained caches and reports
live under the single `artifacts_final/` tree. Rejected experiment artifacts should not be retained.

From this sprint onward, each completed experiment should leave only one clearly recommended
submission CSV. Intermediate predictions and caches belong under `artifacts_final/`, and rejected
probe CSVs should be removed rather than accumulated in the active submission directory.

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
|   |-- prediction_cache_diagnostics.py
|   |-- regime_member_blend.py
|   |-- combine_prediction_caches.py
|   |-- global_capacity_model.py
|   |-- turbine_scada_model.py
|   |-- build_feature_cache.py
|   |-- ficr_distribution_model.py
|   |-- cross_group_transfer.py
|   |-- cross_group_trajectory_smoothing.py
|   |-- bottleneck_eda.py
|   |-- oof_lineage_audit.py
|   |-- exact_group3_oof.py
|   |-- exact_driver_oof.py
|   |-- exact_oof_meta_gate.py
|   |-- phase_regime_cross_group.py
|   |-- selective_member_blend.py
|   |-- scada_proxy_stack_hist.py
|   |-- scada_proxy_direct.py
|   |-- weighted_metric_member.py
|   |-- recent_specialist.py
|   `-- analog_experiment.py
|-- docs/
|   |-- agent_service.md      # Competition Scientist control plane
|   |-- modeling_strategy.md
|   `-- reports/
|-- agent_service/           # generic experiment tree, governance, and submission service
|-- .agents/                 # competition plug-in, policies, roles, and examples
|-- data/                    # ignored raw competition data
|-- artifacts_final/         # retained feature cache, lineage inputs, OOF, and final reports
`-- submissions/
    |-- results.csv          # tracked leaderboard log
    |-- *.csv                # active ignored submission candidates
    `-- archive/             # older ignored candidates
```

## Main Commands

Competition Scientist control plane:

```bash
python -m agent_service init
python -m agent_service status
python -m agent_service competition-show
python -m agent_service tree
python -m agent_service auto-cycle
```

The service now keeps an approved validation strategy, parent/child experiment lineage,
`local_best` and `submission_candidate` as separate states, and guarded DACON submission. See
`docs/agent_service.md` for the full workflow. External submission requires the local explicit
execute flag plus environment credentials; the HTTP service cannot submit.

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

Regime-gated SCADA injection:

```bash
python -m experiments.regime_member_blend --output submissions/blend_stack5_scada_extra2_agree4.csv --weights 0.02 --max-disagreement 0.04 --min-base-ratio 0.10
python -m experiments.regime_member_blend --output submissions/blend_stack5_scada_extra2_agree6_mid.csv --weights 0.02 --max-disagreement 0.06 --min-base-ratio 0.10 --max-base-ratio 0.75
python -m experiments.regime_member_blend --output submissions/blend_stack5_scada_g12_extra3_agree5.csv --weights kpx_group_1=0.03,kpx_group_2=0.03,kpx_group_3=0.0 --max-disagreement 0.05 --min-base-ratio 0.10
```

FICR-oriented generation-weighted member and diagnostics:

```bash
python -m experiments.weighted_metric_member --data-dir data --artifact-dir artifacts_weighted_metric --output artifacts_weighted_metric/weighted_metric_member.csv
python -m experiments.prediction_cache_diagnostics --cache artifacts_weighted_metric/prediction_cache.npz --baseline eligible_uniform --output-json artifacts_weighted_metric/monthly_diagnostics.json --output-markdown artifacts_weighted_metric/monthly_diagnostics.md
python -m experiments.regime_member_blend --base submissions/blend_over115_scada_stack5.csv --member artifacts_weighted_metric/weighted_metric_member.csv --output submissions/blend_stack5_weighted_g12_2_agree4.csv --weights kpx_group_1=0.02,kpx_group_2=0.02,kpx_group_3=0.0 --max-disagreement 0.04 --min-base-ratio 0.10
```

Cross-group group-3 transfer candidate:

```bash
python -m experiments.build_feature_cache --data-dir data --cache-dir artifacts_feature_cache
python -m experiments.cross_group_transfer --data-dir data --base artifacts_cross_group/base_pre_cross.csv --artifact-dir artifacts_cross_group --output submissions/blend_best_crossg3_45_agree8_delta8.csv --alpha 0.45 --max-group-disagreement 0.08 --max-member-disagreement 0.08
```

Cross-group trajectory-consensus smoothing:

```bash
python -m experiments.cross_group_trajectory_smoothing
```

Reproducible bottleneck EDA:

```bash
python -m experiments.bottleneck_eda
```

Exact group-3 OOF lineage:

```bash
python -m experiments.oof_lineage_audit
python -m experiments.exact_group3_oof
```

Exact group-1/group-2 driver lineage and settlement meta-gate:

```bash
python -m experiments.exact_driver_oof
python -m experiments.exact_oof_meta_gate
```

Lead-phase/weather-regime cross-sectional candidate:

```bash
python -m experiments.phase_regime_cross_group
```

Power-curve residual member:

```bash
python -m experiments.power_curve_residual --data-dir data --artifact-dir artifacts_power_curve --output submissions/power_curve_residual.csv
python -m experiments.selective_member_blend --base submissions/blend_over115_scada_stack5.csv --member submissions/power_curve_residual.csv --output submissions/blend_stack5_powercurve_sel5_g12_t06.csv --weights kpx_group_1=0.05,kpx_group_2=0.05,kpx_group_3=0.0 --max-disagreement 0.06
```

Causal KMA ASOS state and group-3 six-hour router:

```powershell
# Set the user-issued key only in the local process; never commit or pass it as an argument.
$env:KMA_API_KEY = "<your KMA APIHub key>"
python -m experiments.fetch_kma_asos_observations
python -m experiments.kma_observation_block_router
```

The exact provider request scope can be inspected without a key via
`python -m experiments.fetch_kma_asos_observations --plan-only`.

The collector defaults to Taebaek ASOS 216, retains redacted source URLs and raw-file
checksums, and applies a conservative two-hour observation publication lag. The router is
diagnostic-only: it compares ASOS features with an otherwise identical no-observation
control, opens H2 only after strict Q2 component/month/seed gates, and never creates a
submission. See `docs/reports/kma_observation_router_2026-07-19.md`.

The real-data run retained 8,803 causal observation joins with zero timing violations.
Although locked H2 group-3 score improved `+0.002404` and bootstrap q05 was positive,
November FICR and incremental NMAE versus the no-observation control failed the frozen
promotion contract. The ASOS router is therefore rejected without a submission; do not tune
its policy on H2.

## Modeling Notes

- The strongest baseline is not a single model but a blend of LightGBM/CatBoost candidates and SCADA proxy stack members.
- `blend_v1` moved the public score from `0.6367` to `0.6395`.
- Manual extrapolation from `cal125` toward `blend_v1` peaked around `over115`.
- SCADA proxy stack at 5% injection moved the score to `0.6402652274`.
- `stack4` and `stack10` showed that the global optimum is narrow around 5%; next tests should be group-wise, not more global weight tuning.
- Regime-gated SCADA injection now replaces broad global sweeps: keep `stack5` as the base and add tiny extra SCADA movement only where base/member disagreement is small.
- The official FICR implementation weights hourly settlement by actual generation. Generation-weighted LightGBM members improved corrected 2024 holdout scores for groups 1 and 2, so the safest new probe injects 2% only on agreement rows and leaves group 3 unchanged.
- That probe was publicly confirmed at `0.6403102237`, improving both score components and becoming the new best submission.
- The strongest new local architecture is a group-3 blend of a capacity-normalized global model (23.8%) and a nominal-operation turbine-level SCADA model (76.2%). Its corrected 2024 holdout score is `0.61165` for group 3 and the full local pool score is `0.65406`; these local values are directional and are not estimates of the public score.
- The positive 10% group-3 injection scored `0.6401243372`, losing `0.0001858865` versus the best; both 1-NMAE and FICR fell. This public result overrides the positive local signal.
- The reverse-direction probe scored `0.6400644769`: 1-NMAE improved by `0.0001072756`, but FICR fell by `0.0005987692`, producing a net loss of `0.0002457468`. Both directions are now rejected; do not continue the turbine group-3 family.
- A FICR distribution layer improved its own weak quantile baseline but did not transfer reliably to the strongest ensembles; it was rejected without submission.
- The cross-group transfer model exploits the stable normalized correlation between group 2 and group 3 (`0.9138` in 2023 and `0.9447` in 2024). Its gated 25% correction improved both NMAE and FICR on weighted, global, and final-pool OOF proxies, including annual group-3 score gains of `+0.00096`, `+0.00061`, and `+0.00051` respectively.
- The 25%/6% gate was publicly confirmed at `0.6414690556`, improving score by `0.0011588319`, 1-NMAE by `0.0001062905`, and FICR by `0.0022113733`.
- The next fixed-family expansion uses 45% weight and an 8% member-disagreement gate. Across the same three proxies, annual group-3 gains are `+0.00199`, `+0.00152`, and `+0.00130`, with both metric components positive in every proxy.
- Publicly, the 45%/8% expansion scored `0.6414589725`: 1-NMAE improved by `0.0001869875`, but FICR fell by `0.0002071538`, leaving score `0.0000100831` below submission20. Broad expansion is therefore rejected.
- Simple selective strengthening could not improve score and FICR across every proxy and both validation halves, so regime micro-tuning was rejected.
- A full audit against the user-provided official metric notebook found exact local metric parity. The new neural loss was corrected to macro-average groups, train/test tensor caches were physically separated, the issue-cycle H1/H2 boundary was corrected, and non-finite eligible predictions now fail closed. See `docs/reports/training_evaluation_audit_2026-07-13.md`.
- The corrected two-seed spatial-temporal model improved group-3 NMAE but not FICR robustly enough to modify the publicly confirmed cross-group member. It was rejected without submission.
- Cross-group trajectory-consensus smoothing is the first post-audit method to pass the full/H2 proxy checks. It leaves groups 1/2 unchanged and makes a maximum group-3 movement of only `20.99 kWh`; treat it as a controlled public probe because the weighted-proxy bootstrap score sign is still less certain than the global/pool proxies.
- Submission `1491284` confirmed that trajectory smoothing was directionally correct but saturated: the score gain was only `+0.0000109493`.
- Bottleneck EDA found that the weighted group-3 proxy underpredicts true 90-100% output by about `4,597 kWh`, with `98.6%` of those rows outside the 8% settlement band. At the same time, strong-wind high-power frequency changes sharply by direction and month, so direct upward correction is unsafe.
- The 2025 NWP period is modestly but consistently windier than 2024 (group-3 LDAPS hub-wind mean `10.055` vs `9.230 m/s`). High-power classifiers, residual distributions, importance weighting, and seasonal experts all failed the multi-seed/proxy H2 checks; no new CSV was created. See `docs/reports/bottleneck_eda_2026-07-14.md`.
- A SCADA-derived nominal/output-limited state classifier reached only `0.653` AUC on the relevant high-potential split, and its mixture-of-experts correction reduced all proxy scores. A nine-threshold ordinal distribution also failed to transfer consistently. Both are rejected as post-processing families.
- A shared three-group ordinal base improved weighted H1 but sharply worsened H2 and did not transfer to global/final-pool. The next bottleneck is exact OOF parity: retrain and cache the historical blend/SCADA components, then reconstruct the complete submission23 lineage before selecting another correction.
- The historical SCADA stack has been recached and aligned to its archived test lineage; aligned/recovered test MAE is `16.12 kWh`. The exact group-3 pre-cross OOF now scores `0.59054`, and trajectory smoothing adds `+0.000045` on that surface. Cross-group 25% is H2-negative locally despite its large public gain, so no further weight expansion is justified. See `docs/reports/exact_group3_oof_2026-07-14.md`.
- The complete group-1/group-2 driver lineage reproduces the archived 2025 public-base vectors within `3.1e-5 kWh` maximum error. Exact-OOF residual stacking and driver calibration were rejected after temporal/proxy instability. The settlement-aware meta-gate improved Q1->Q2 by `+0.001635`, locked H1->H2 by `+0.000232`, and the public score by `+0.0001753677`; submission `1494307` became the public best before the fine sweep. See `docs/reports/exact_oof_meta_gate_2026-07-17.md`.
- A literature-driven cross-sectional ensemble separated the forecast trajectory into four lead phases and four NWP weather regimes. Its H2 exact-OOF gain did not transfer publicly: submission `1494535` improved 1-NMAE but lost `-0.0011923392` FICR and `-0.0005174729` total score versus the best. Broad structural injection is rejected; see `docs/reports/phase_regime_cross_group_2026-07-17.md`.
- Fine meta-gate sweep submission `1494670` scored `0.6417471627`, improving the previous best by `+0.0000917901`; the direction transferred through FICR but remains a micro-gain.
- The broad settlement composite scored `0.6377660509`, losing `-0.0038893217` versus submission `1494307` with both metric components lower. Broad cross-group calibration is publicly rejected.
- A two-seed spatial-temporal graph multitask model improved locked H2 locally, but submission `1494986` scored `0.6415388286`: 1-NMAE rose slightly while FICR fell `-0.0004963026`. The global 20% blend is publicly rejected and archived; its OOF/model diagnostics may be retained only for diversity analysis.
- Power-curve residual selective injection was tested in `blend_stack5_powercurve_sel5_g12_t06.csv` and dropped to `0.6400608956`; do not expand this family unless a stronger local validation signal is found.
- Rejected power-curve, PCA/lead-lag, strict-aggregation, and aggregate CatBoost probe outputs were removed from the active workspace.

## Rule Compliance

- Test-period actual generation and test-period SCADA are not used.
- All SCADA usage is restricted to train-period proxy modeling.
- No external weather data is present in an active submission. External-data runs require an operational-source manifest, exact publication-time audit, raw-file checksums, and reproducible license/provenance before promotion.
- Retrospective Open-Meteo historical/previous-run data is explicitly ineligible for submissions unless its original public-availability evidence can be independently established.
- Pretrained weights must have been officially public by 2026-07-05 and permit use, modification, distribution, redistribution, and commercial use; dynamic inputs must independently satisfy the prediction-time cutoff.
- No remote inference API is used.
- Generated model artifacts and large submission CSVs remain ignored.

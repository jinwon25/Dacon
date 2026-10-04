# 상세 연구 기록 (English)

아래의 current/next는 작성 당시 연구 상태를 뜻합니다. 최종 결과는 상위 README를 따릅니다.


아래는 실험별 감사·판정의 원본 기록이다.

## Final Evaluation Audit

The local arithmetic is not the bottleneck. `src/metrics.py` matches the
supplied official notebook on the 10% eligibility mask, capacity-normalized
within-group MAE, actual-generation-weighted FiCR, inclusive 6%/8% settlement
cliffs, group macro averaging, and final 50:50 component average. Randomized
and boundary tests pass exactly.

The faulty layer was promotion governance. Full-year positive deltas were able
to pass even when the exact 40% public-sized and complementary 60%
private-sized lower tails were negative. For submission `1504383`, 10,000
local complementary splits put the observed public pair-score delta at the
0.25th percentile (IID) and 0.32nd percentile (month-stratified). More
importantly, the simulated public q05 was already negative for score
(`-0.00013977` IID) and FiCR (`-0.00042350` IID), so a subset-safe gate would
have rejected it before submission.

Policy `baram-public-v4-subset-safe` now requires non-negative q05 for score,
1-NMAE, and FiCR on both 40% and complementary 60% samples, in addition to a
non-negative issue-block bootstrap q05. Re-screening rejected both remaining
materialized paths: the G1/G2 reconciliation and the pooled G2 weight-0.10
expansion. No new submission CSV was created; incumbent `1502437` remains
frozen. See `docs/reports/evaluation_promotion_final_065_2026-07-29.md`.

A final new-core screen replaced independently fitted residual quantiles with
a pooled, structurally non-crossing conditional quantile MLP for groups 1 and
2. A follow-up audit found that its first report selected the action around the
distribution median in 2023 but overlaid that delta on a different incumbent
baseline in 2024. Because FiCR depends on absolute 6%/8% error-cliff position,
that asymmetric comparison is invalid as a transfer contract.

The corrected baseline-symmetric run shows that the action itself transfers:
the affected-pair 2024 internal deltas are `+0.01040158` score,
`+0.00287221` 1-NMAE, and `+0.01793096` FiCR, with positive IID and
month-stratified complementary 40/60 q05. July remains negative. More
decisively, the completed standalone core is far weaker than the exact
incumbent: `-0.02807739` score, `-0.01581153` 1-NMAE, and `-0.04034326`
FiCR. The incumbent-delta overlay is now prohibited, the family is rejected
fail-closed, and no candidate was written. The corrected bottleneck is absolute
core accuracy relative to the strong ensemble, not year-shifting action
asymmetry. See
`docs/reports/noncrossing_baseline_symmetry_audit_2026-07-29.md`.

A final baseline-matched follow-up then modeled the exact incumbent residual
instead of replacing the incumbent. Its non-crossing distribution model passed
Q2 selection at coverage `0.05`, weight `0.30`, and score delta
`+0.0020063333`, but locked H2 FiCR reversed by `-0.0000163270`. The causal
full-year gain was only `+0.0004267741`, projecting to `0.6465518655` and
leaving `0.0034481345` to 0.65. A separate direct official-utility value model
also failed Q2 because April score was negative. This isolates calendar
transfer of incumbent residual rankings as the remaining bottleneck; neither
path writes a candidate. The action selector now stays nearest the incumbent
on expected-utility plateaus, and short monthly slices safely handle farms
with no eligible rows. See
`docs/reports/incumbent_residual_decision_followup_2026-07-29.md`.

## Active Submission Candidate

The selected public incumbent is
`artifacts_final/candidates/public_positive_g1w1375_g2w1825_g3frozen_20260802.csv`
(SHA-256 `8056206176d12f21f72fda14fba8fa3b19bcc8a6da7d8a6e389b9e28a40c4902`).
Its observed Public result is `0.6474704399`, with 1-NMAE `0.8761125755` and
FICR `0.4188283043`. The matched 2024 lineage OOF is `0.6491632057 /
0.8827530123 / 0.4155733992`; these local values are validation evidence, not
an estimate of the Private score. All later broad replacement candidates are
withheld after submission `1511408` failed both score components.

The zero-sum G1/G2 structure was
`artifacts_final/candidates/group12_difference_reconciliation_unanimous_w10_20260728.csv`
(SHA-256 `fab745cbed1a1d60c3d15af15b9590a399fd62f8bf63ec16a6dfa3dd9c4a047a`).
It predicts only the capacity-normalized G1-G2 difference from paired
LDAPS/GFS site features, applies a weight-0.10 correction when all three seeds
agree on its direction, preserves the incumbent G1+G2 sum row by row, and
keeps G3 exactly unchanged. Its 2024 G1/G2-average deltas are `+0.00065385`
score, `+0.00014709` 1-NMAE, and `+0.00116062` FICR; Q1, Q2, H2, and every
seed-period score are positive. A 10,000-draw issue bootstrap has 95.46%
positive draws and q05 `+0.00001783`. The movement is orthogonal to the
retained public factors (cosine `0.18679`) and averages only 0.1255% of
capacity, but just 7/12 months are positive. It was submitted as the isolated
controlled probe `1504383` and scored `0.6456906139`, with 1-NMAE
`0.8758649682` and FICR `0.4155162596`. Relative to incumbent `1502437`,
score fell `-0.0004344775` even though 1-NMAE improved `+0.0000807205`,
because FICR fell `-0.0009496756`. The implied G1/G2 affected-group mean
delta is `-0.0006517163`, almost the exact negative of the local
`+0.0006538534`. Component transfer was asymmetric: the 1-NMAE transfer ratio
was `+0.8232`, while the FICR transfer ratio was `-1.2274`. Broad G1/G2
differential post-processing is closed and its weight must not be retuned from
this public result.

A one-sided selective-gain gate then tried to retain only rows whose predicted
lower quantile of absolute-error improvement was positive. The 10% and 20%
lower-quantile models selected no rows; the 30% model selected only a few rows
with actual positive-gain precision of 26.7% for G1 and 18.2% for G2. No March
policy passed the preregistered pair/group/coverage/precision gates, Q2/H2
remained closed, and no CSV was generated. This also closes sparse row
selection over the same rejected action. See
`docs/reports/public_factor_and_group12_reconciliation_2026-07-28.md`.

The JMA MSM plateau successor was
`artifacts_final/candidates/jma_msm_g3_plateau_mean_w040625_20260728.csv`
(SHA-256 `da0bbb9fba76189be89a95d3b99c6e736403acb8078fbc21747a380f36253900`).
It keeps incumbent groups 1 and 2 exactly unchanged and applies only the
Q1-near-best JMA MSM group-3 plateau-mean weight `0.040625`. Group-3 local
deltas are `+0.00215100` score, `+0.00054100` 1-NMAE, and `+0.00376101`
FICR; 10/12 months and every seed/split component are positive, while the
issue bootstrap has 99.1% positive draws and q05 `+0.00063262`. The simple
full-transfer projection is `0.64684209`. It was initially withheld because
the plateau rule was proposed after inspecting the earlier 2024 confirmation,
then used as controlled G3-only submission `1504352`, which
scored `0.6455966798`, 1-NMAE `0.8757843690`, and FICR `0.4154089906`.
Relative to incumbent `1502437`, score fell `-0.0005284116`; 1-NMAE was
essentially unchanged at `+0.0000001213`, while FICR fell `-0.0010569446`.
The implied full group-3 deltas are score `-0.0015852348`, 1-NMAE
`+0.0000003639`, and FICR `-0.0031708338`. This publicly rejects the JMA MSM
plateau and closes JMA external-weather group-3 post-processing; do not retune
its weight against the public result or combine it with another unconfirmed
factor.
See `docs/reports/scenario_analog_and_plateau_breakthrough_2026-07-28.md`.

The isolated JMA GSM group-3 probe was
`artifacts_final/candidates/public_factor_g1g2_kma_jma_gsm_g3_strict_20260726.csv`
(SHA-256 `30a1229ecacd083a06bf1ca26593f39c5c473565b4090944728574c500d31ed5`).
It differs from the incumbent only in G3 and uses the independent KMA+JMA GSM
pooled expert. Despite a 2024 G3 gain of `+0.0032619155` and passing every
strict local gate, submission `1502447` scored `0.6449277253`: `-0.0011973661`
score, `-0.0003654917` 1-NMAE, and `-0.0020292405` FICR versus the incumbent.
The JMA GSM G3 production family is rejected and must not be blended or retuned.
The full audit and submission protocol are in
`docs/reports/public_factor_gsm_breakthrough_2026-07-26.md`.

The exact two-probe directional FiCR experiment was publicly rejected.
Submission `1503281` (G1 only) scored `0.6441230595`, or `-0.0020020319`
versus the incumbent, with both 1-NMAE and FiCR lower. Submission `1503282`
(G2/G3 only) scored `0.6451061773`, or `-0.0010189141`, again with both
components lower. Group-macro additivity identifies the unsubmitted target
exactly as `0.6431041454`, 1-NMAE `0.8732865899`, and FiCR `0.4129217008`.
`public_directional_ficr065_target_20260727.csv` must not be submitted.

The result invalidates aggressive post-hoc settlement calibration despite its
positive Q1/Q2/H2, monthly, and block-bootstrap diagnostics. The related safe
G2/G3 control is withheld because it shares the rejected directional family.
One daily submission slot remains and is intentionally preserved. Full evidence
is in `docs/reports/public_directional_ficr_2026-07-27.md`.

The expanding-origin 24-hour trajectory TCN probe for group 3 was publicly
rejected:
`artifacts_final/candidates/public_probe_issue_tcn_expanding_g3_q65w05_20260727.csv`
(SHA-256 `0c527afaad8fe67a367c1b1ac252cfab03d373c7a92c846763bd0e4b6f8d19d7`).
It preserves incumbent groups 1 and 2 exactly and changes only group 3. The
three-seed q=0.65, weight=0.05 policy improved score, 1-NMAE, and FICR in every
2024 expanding quarter; its full group-3 deltas were `+0.00293625`,
`+0.00106516`, and `+0.00480733`. Ten of twelve months were positive and the
H2 issue-block bootstrap q05 was `+0.00123757` with 99.8% positive draws.
September and October were negative and the policy was selected after repeated
inspection of 2024, so this is a controlled exploratory public probe rather
than a strict promotion. Its simple full-transfer projection is only
`0.64710384`, not evidence of a 0.65 submission.

Submission `1503349` scored `0.6454576156`, with 1-NMAE `0.8759443235` and
FiCR `0.4149709077`. Relative to incumbent submission `1502437`, score fell
`-0.0006674758`: the 1-NMAE gain of `+0.0001600758` was outweighed by a FiCR
loss of `-0.0014950275`. Because only group 3 changed, the implied full
group-3 deltas are three times those macro deltas: score `-0.0020024274`,
1-NMAE `+0.0004802274`, and FiCR `-0.0044850825`. The trajectory TCN family is
closed and its quantile or blend weight must not be retuned against this public
result.

The lower-priority public-positive factor control is
`artifacts_final/candidates/public_positive_pooled_g2_w10_probe_20260727.csv`
(SHA-256 `139a6d4e93d9eaeec3731352f55c9f5e4b145805a5eb14ff0fdcbf978f6192a2`).
It doubles only the already-public-positive pooled group-2 movement from model
weight 0.05 to 0.10 and projects to `0.64634108` under its observed transfer
ratio. Its incremental issue-bootstrap evidence is weaker, so it remains
second in the queue and must not be combined with the TCN probe before both
single factors are publicly confirmed. Full evidence and the submission
protocol are in `docs/reports/top10_trajectory_followup_2026-07-27.md`.

The G2 weight-0.10 control was deterministically regenerated on 2026-07-28 and
kept byte-identical. It remains withheld: incremental issue-cycle score is
`-0.00001063`, FiCR is `-0.00024059`, bootstrap q05 is `-0.00151194`, and only
49.06% of bootstrap draws are positive. Its projected public gain is too small
relative to the current top-10 gap to justify treating a failed robustness
surface as today's primary probe.

Two new structural follow-ups were rejected without a submission. Fitting
separate KMA/ECMWF/DWD/GEM pooled models and taking their robust median still
reversed in Q4: group-2 score changed from `+0.0044744` in Q3 to `-0.0022817`
in Q4, while group 3 changed from `+0.0049079` to `-0.0079622`. This closes the
joint/median multi-NWP family rather than opening mean aggregation or weight
micro-tuning.

JMA MSM 2022 H2 was then collected as 4,344 causal forecast hours over the
same 3x3 stencil, with zero timing violations and a 30-minute minimum
availability margin. A two-year pooled model trained on 2022 H2 plus 2023 did
not beat the 2023-only control. Its closest result was group 3: full score
`+0.0018537`, all Q1/Q2/H2 components and seeds positive, but only 9/12
positive months and 96.85% positive bootstrap draws. No target passed strict
promotion and no CSV was generated. See
`docs/reports/multisource_year_forward_followup_2026-07-28.md`.

The archived manual probe expanded the same fixed group-2 gate from alpha
`0.10` to `0.20`:
`submissions/archive/kma_group2_overlay_alpha20_20260725.csv`.
Relative to submission `1501476`, locked H2 improved score by `+0.0010148836`,
1-NMAE by `+0.0000901354`, and FICR by `+0.0019396318`; all five locked months
were positive and issue-block bootstrap q05 was `+0.0002790199`. Incremental
p95 movement is 0.977% of capacity over 12.89% of target cells. Q2 nevertheless
preferred alpha `0.10`, but submission `1501483` publicly confirmed the expansion
with both components improving.

The next bounded manual probe is alpha `0.2375`:
`artifacts_final/candidates/kma_group2_overlay_alpha2375_20260725.csv`.
Relative to alpha `0.20`, Q2 score improved `+0.0004403252` and locked H2 score
improved `+0.0002409091`; locked H2 component deltas were both positive.
Bootstrap q05 is `-0.0001706419` with 82.8% positive draws, and one locked month
is weak. Cumulative p95 and maximum movement reach 2.320% and 5.993% of capacity,
so this is the final expansion retained under the 2.5%/6.0% movement caps.

With the alpha `0.2375` core frozen, the archived sparse probe added only 889
previously untouched upward-disagreement rows:
`submissions/archive/kma_group2_alpha2375_up_supplement_20260725.csv`.
Its Q2 and locked H2 incremental scores are `+0.0001388568` and
`+0.0003581130`; issue-block bootstrap q05 is `+0.0000746069` with 98.8%
positive draws. Total target-cell coverage remains 16.27%, while cumulative
p95/maximum movement stays at the existing 2.320%/5.993% caps.
Submission `1501731` rejected this probe at `0.6439398345`, a score change of
`-0.0001599771` versus submission `1501487`; 1-NMAE fell `-0.0000119700` and
FICR fell `-0.0003079842`. The alpha-0.2375 core remains selected and both
same-row strength expansion and non-core upward supplementation are closed.

The next group-3 experiment conditioned the publicly successful KMA monotone
power curve on direction, 10 m--850 hPa shear/alignment, and forecast-cycle
change. Its sparse direction-sector policy improved locked group-3 score by
`+0.0006017` and FiCR by `+0.0011979`, but October and November were negative,
all bootstrap q05 components were negative, and the effect missed the frozen
minimum. No CSV was created and this regime-curve family is closed. See
`docs/reports/group3_regime_power_curve_2026-07-26.md`.

Two literature-driven LDAPS spatial families were then tested instead of
continuing alpha/coverage micro-tuning. A bias-corrected 16-grid neighborhood
decision model passed Q2 but reversed on locked H2: group-3 score
`-0.0006798`, 1-NMAE `-0.0001605`, and FICR `-0.0011992`, with only 0.85% of
issue-block bootstrap draws positive on all components. A physics-guided
wind-aligned upwind kernel found a 3 km Q2 structure that improved score and
FICR, but 1-NMAE fell `-0.0004222`, so H2 was not opened. Both families are
closed and no CSV was created. A causal 2023+2024 incumbent OOF was audited as
the prerequisite for weather-conditioned wind smoothing; see
`docs/reports/ldaps_spatial_methods_2026-07-26.md`.

The requested causal 2023 incumbent OOF is not identifiable: group 3 has zero
labels in 2022, while the current cross-group lineage uses 2023 group-3 labels
to predict 2024. Back-casting that model would be target leakage. A distinct
year-forward experiment therefore collected 366 causal 2023 KMA UMRG issues
(8,760 targets and 1,464 source objects, zero timing violations) and used them
only to train an independent prior-year expert. Direct history pooling had zero
Q2 candidates that improved every component. A Q2-selected sparse 4.35%-coverage
expert blend gained `+0.0003434`, but reversed on locked H2 at `-0.0001844`
score and `-0.0003459` FICR; only 10.6% of issue-block bootstrap draws improved
all components. No CSV was created and the active submission remains selected.
See `docs/reports/kma_year_forward_blend_2026-07-26.md`.

As a separate leakage-safe fallback, the paper-specified within-issue
`t-1/t/t+1` LDAPS wind moving average was screened on Q2 with separate
April/May/June gates. None of 48 policies improved score, 1-NMAE, and FICR
together; H2 stayed closed and no CSV was created. See
`docs/reports/temporal_smoothing_cleanup_2026-07-26.md`.

Submission `1494986`
(`blend_best_spatiotemporal_multitask20.csv`) scored `0.6415388286`, or
`-0.0002083341` versus the public best. Its 1-NMAE improved by `+0.0000796343`, but
FICR fell by `-0.0004963026`; service run 7 is publicly rejected and archived.

The next independent research track is a leakage-safe independent operational
forecast. NOAA GEFS spread and mean/disagreement screens were rejected before any
2025 collection. A preliminary KMA UM global N128 single-point 10 m screen was also
rejected: through 2024-09-08 its locked score, 1-NMAE, and FiCR deltas were
`-0.00028103`, `-0.00013592`, and `-0.00042614`. The active KMA branch therefore
uses the latest/previous forecast-cycle change plus 850/700 hPa vertical context,
with a 12 km regional-model pilot only after the global request schema passes. The
collectors read a user-issued key from the process or ignored `.env.local`; no key
is accepted on the command line or written to artifacts. No experiment may be promoted until
the exact run publication time, raw files, checksums, license, and per-row causal
join pass the external-data manifest guard. Retrospective Open-Meteo history is
research-only and blocked from submission use. See
`docs/reports/external_data_pretrained_audit_2026-07-18.md` and
`docs/reports/kma_um_context_sprint_2026-07-19.md`.

The completed 12 km UMRG context archive retained 1,468 original responses for
8,784 forecast hours with zero checksum, coverage, timing, or secret-retention
violations. The direct residual correction lost `-0.00042926` score, and the KMA
harm-risk abstention model produced no seed-stable Q2 policy. Those two subfamilies
are closed. A separate bounded monotone power-curve path did pass: against the exact
public-best rolling OOF surface its H1-refit locked H2 gain was `+0.00699607` for
group 3, while the KMA increment beyond an identical GFS-850 control was
`+0.00233477`. All locked monthly total FiCR deltas were positive, the 2,000-issue
bootstrap lower 5% KMA increment was positive in score, 1-NMAE, and FiCR, and every
movement is capped at 5% capacity. This promotes causal 2025 UMRG collection, but no
submission is written until that archive and its manifest pass validation. See
`experiments/kma_um_power_curve_gate.py` and
`docs/reports/kma_um_context_sprint_2026-07-19.md`.

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
- The active KMA UM submission is backed by an operational-source manifest, exact publication-time audit, raw-file checksums, and reproducible license/provenance. Any further external-data run must satisfy the same guard before promotion.
- Retrospective Open-Meteo historical/previous-run data is explicitly ineligible for submissions unless its original public-availability evidence can be independently established.
- Pretrained weights must have been officially public by 2026-07-05 and permit use, modification, distribution, redistribution, and commercial use; dynamic inputs must independently satisfy the prediction-time cutoff.
- No remote inference API is used.
- Generated model artifacts and large submission CSVs remain ignored.

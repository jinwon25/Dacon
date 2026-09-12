# Sprint 0.660 Plan

Updated: 2026-08-05 KST

The target of 0.660 is aggressive and is not treated as a guaranteed result.
Promotion is based on time-ordered OOF evidence, not Public leaderboard fit.
The latest logged Public result (submission 1508386, Score 0.6474704399,
1-NMAE 0.8761125755, FICR 0.4188283043) is status-only.

## Completed and open work

| Area | State | Evidence or stop rule |
|---|---|---|
| Official metric | Complete | Actual-weighted FICR plus threshold/NaN/no-row unit tests |
| NWP cutoff audit | Complete | All retained rows available by prior-day 14:00 KST; actual lead 12-35 h |
| Expanding OOF | Complete | Groups 1/2 use 2022-to-2023 and 2022-2023-to-2024; group 3 uses 2023-to-2024 |
| Corrected baseline | Complete | Strict locked-2024 Score 0.623751 |
| Weighted and calibrated LGBM | Complete | Group-gated stable policy retained |
| Metric-aware Bayes | Complete | Direct quantile action improved raw baseline but not the stronger safe composite reliably |
| Trajectory ablation | Complete, closed | Negative locked macro delta and negative bootstrap lower bound |
| Site and physical features | Complete, closed | Broad blocks failed transfer; training-only SCADA was retained only for group 3 |
| Group-3 pooling | Complete, closed | NMAE improved, FICR declined, bootstrap lower bound negative |
| Frozen spatial-temporal diversity | Complete | Unit-correct 20% member retained as higher-risk candidate |
| Full train/inference | Complete | Full refit plus three 8,760-row validated candidates |
| Current-incumbent baseline match | Complete | Reconstructed 2024 lineage OOF 0.649163; stronger than every broad Sprint-066 replacement |
| Group-1 recency overlay | Complete, closed | H1/H2 mean positive, but locked bootstrap q05 -0.004537 and P(delta>0) 0.6315 |
| Group-3 frozen displacement | Complete, closed | Nearest-grid x 0.50 froze on 2023; 2024 H2 and bootstrap gates failed, so no CSV was written |

## Experiment schedule and terminal decisions

| ID | One hypothesis | Dependency | Actual runtime | Decision |
|---|---|---|---:|---|
| S066-00 | Stage-0 metric, availability and split paths remain valid | local data and tests | under 2 min | passed |
| S066-M1 | Eligible quantiles support an official-utility Bayes action | 11 quantile LGBMs | 1,195 s | signal retained; final group-2 gate is candidate-only |
| S066-M2 | Analog residuals improve distribution calibration | M1 | not run | stopped: M1 incremental bootstrap probability was only 0.687 |
| S066-B | A narrower trajectory block improves distribution prediction | prior P3 | not rerun | stopped after trajectory-family failures |
| S066-G1 | Frozen spatial-temporal member diversifies group 3 | existing two-seed OOF | 22 s | candidate-only |
| S066-FINAL | Frozen policies reproduce from train to inference | promoted policies | about 7 min total | passed, including serialized two-seed member |
| S066-R1 | Recent-12-month minus equal-history factor repairs group-1 transfer | incumbent-matched OOF and cached base features | 119 s | rejected; no candidate written |
| S066-G3D | A small, issue-local LDAPS spatial/temporal displacement increment repairs incumbent group 3 | incumbent lineage, 2023 KMA context, train LDAPS | 5.6 s full; 2.1 s smoke | rejected; frozen 2024 G3 delta +0.000179, H2 -0.000618, bootstrap P=0.5825 |

## S066-G3D preregistration (written before model execution)

The 2024 incumbent remains exactly fixed. Because no byte-identical 2023 OOF
surface exists for submission 1508386, the selection anchor is a causal
incumbent-family KMA-UM 10 m isotonic power curve refit separately in each
2023 fold. Every candidate is an incremental difference between a displaced
LDAPS power curve and the same-fold site-centred LDAPS power curve. The frozen
increment is then added to the exact 2024 incumbent G3 surface; G1/G2 are not
involved in selection or prediction.

Internal folds are fixed as follows (all boundaries use forecast KST):

- fold `2023_h1_to_q3`: train 2023-01-01--2023-06-30, validate Q3;
- fold `2023_q3_to_q4`: train 2023-01-01--2023-09-30, validate Q4.

The six and only six displacement families are:

| ID | Frozen transform inside one LDAPS issuance |
|---|---|
| D1 | nearest grid to the documented G3 turbine centroid |
| D2 | 1.5 km instantaneous upstream Gaussian kernel |
| D3 | 3.0 km instantaneous upstream Gaussian kernel |
| D4 | site-centred trajectory shifted by -1 forecast hour, edge-replicated |
| D5 | site-centred trajectory shifted by +1 forecast hour, edge-replicated |
| D6 | 1.5 km upstream trajectory followed by centred +/-1 h mean |

Only overlay weights `{0.10, 0.25, 0.50}` are evaluated. Incremental movement
is clipped at 1% of G3 capacity, then the prediction is clipped to physical
bounds. A rule is stable only if both forward folds have positive Score and
FICR deltas and neither fold loses more than 0.002 in 1-NMAE. Among stable
rules, select the largest mean fold Score delta, breaking ties by worst-fold
Score, then lower weight, then candidate ID. If no rule is stable, stop without
opening 2024. There is no post-2024 offset, weight, displacement, or gate
search.

After selection, refit the centred and selected displaced curves on all valid
2023 G3 labels and evaluate once against the exact incumbent 2024 OOF. H1/H2,
month, season, actual lead and paired target-date bootstrap are diagnostic
views only. A submission candidate is written only if every user-specified
2024 promotion gate passes.

Terminal result: D1 nearest grid at weight 0.50 was the frozen rule. It was
positive on both 2023 validation quarters, but its one-shot 2024 result was
only +0.000179 G3 Score (+0.000060 macro), with H2 -0.000618 and paired
target-date bootstrap P(delta > 0) 0.5825. The family is closed without a
submission file or any post-result retuning. The next permitted work is a
structural G3 label/NWP/site/capacity alignment audit.

## Validation contract

- Forecast day is 01:00 through next-day 00:00; midnight belongs to the
  preceding issue day.
- Groups 1 and 2: 2022 fit to 2023 selection/confirmation, then 2022-2023
  refit to locked 2024 evaluation.
- Group 3: 2023 H1 to H2 nested selection where possible, then full 2023
  refit to locked 2024 evaluation.
- Affine rules, Bayes shrinkage and blend weights are selected before their
  locked evaluation period.
- Bootstrap resamples complete forecast issue days.
- Test weather is opened only by inference. It is never used for selection,
  calibration or hyperparameter tuning.

## Stop policy

The direct-quantile Bayes family is closed after the group-gated incremental
gain failed the 0.0015 Score and 0.80 bootstrap-probability gates. Broad
trajectory/site/physical and pooling families are also closed after transfer
failures. The bounded recency overlay is also closed: despite positive H1 and
H2 point estimates, its locked issue-day q05 and positive probability failed.
The frozen G3 displacement family is now also closed after its H2 reversal and
bootstrap failure. The remaining high-value work is the predeclared structural
G3 label/NWP/site/capacity alignment audit, not more locked-2024 tuning.

# 목표 점수 개선 실험 보고서

한국어 요약: 목표 점수를 향한 실험 단계별 설계·지표 감사·후보 판정 기록입니다. 공식 평가 대상과 정산 경계를 먼저 맞추고, 시간 순서의 학습 외 예측으로 후보를 비교합니다. 보고서 제목의 목표값은 달성 성과가 아니며, 최종 공개 점수는 프로젝트 README를 따릅니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: BARAM 2026 Sprint 0.660 Experiment Report

Updated: 2026-08-05 KST

Leaderboard update: 2026-08-05 KST

Submission 1511408 (safe_cv_best) scored 0.6345216694 / 0.8679554758 /
0.4010878631. It is rejected. The deltas versus incumbent submission 1508386
are -0.0129487705 Score, -0.0081570997 1-NMAE, and -0.0177404412 FICR.
Consequently the Bayes and diverse derivatives are held back despite their
positive locked-2024 increments.

A nested calibration audit using only 2024 H1 selects group-1 scale 1.066 and
offset +700. It improves 2024 H2 Score from 0.638598 raw to 0.663399; the older
1.07/+900 policy scores 0.658497. Historical validation therefore still
supports a positive affine, so it will not be removed based only on one Public
result. The next offline hypothesis is recency-weighted group-1 refitting.

That follow-up is now complete. A 2023-only group-1 direct model beats the
2022-2023 equal-history direct model by +0.01860064 Score and +0.03728789
FICR on 2024, with -0.00008660 1-NMAE. Adding only this model difference to
the current incumbent selected alpha 0.25 on H1 and stayed positive on H2,
but locked H2 issue-day bootstrap q05 was -0.00453722 and P(delta > 0) was
0.6315. The overlay is rejected and no new submission file was generated.

## 1. Executive result

Within the broad Sprint-066 replacement family, the highest locked-2024 OOF
result is the diverse candidate:
Score 0.643821, 1-NMAE 0.878346, and official FICR 0.409297. This is an
OOF result, not a Public or Private leaderboard estimate. It does not establish
that 0.660 has been reached.

The selected production baseline is still submission 1508386. Its reconstructed
2024 lineage OOF is 0.649163 / 0.882753 / 0.415573, higher than every broad
Sprint-066 replacement. Its observed Public result is 0.6474704399 /
0.8761125755 / 0.4188283043 and was not used to select the recency alpha.

## 2. Stage-0 audit

### Official metric

Official FICR is actual-generation weighted:

    earned = sum(actual * unit_price)
    maximum = sum(actual * 4)
    FICR = earned / maximum

Only rows with actual at least 10% of group capacity are evaluated. Unit price
is 4 at error <= 6%, 3 at 6% < error <= 8%, and 0 otherwise. Tests cover the
difference from an hourly price mean, NaN actuals, no eligible rows, exact 6%
and 8%, and the 10% eligibility boundary. The corrected scorer changes the
preferred group-1 baseline variant from all-row to eligible-only training.

### Forecast availability

| Check | Result |
|---|---|
| Stored timezone | Naive timestamps, semantically Asia/Seoul per data documentation |
| LDAPS/GFS availability | 13:00 KST |
| Day-ahead cutoff | Prior day 14:00 KST |
| Actual lead | 12-35 hours from forecast minus availability |
| Cutoff violations | 0 |
| Exact duplicates | 0 |
| Multiple retained cycles per forecast/grid/source | 0 in supplied data |

Feature construction selects the latest legal cycle row by row and asserts
that every selected availability time is no later than the cutoff.

### Labels, capacity and SCADA

| Group | Capacity | Provided labels | Missing labels | Eligible rows | Eligible rate |
|---|---:|---:|---:|---:|---:|
| 1 | 21,600 kWh | 26,200 | 104 | 15,915 | 60.74% |
| 2 | 21,600 kWh | 26,201 | 103 | 15,891 | 60.65% |
| 3 | 21,000 kWh | 17,538 | 8,766 | 9,414 | 53.68% |

Groups 1 and 2 each map to six VESTAS V126 3.6 MW turbines. Group 3 maps to
five UNISON U136 4.2 MW turbines. Six 10-minute SCADA power values must be
summed to the hourly KPX label. The best alignments are VESTAS ceil-to-hour
and UNISON floor-plus-one-hour. Their correlations with KPX are 0.999731,
0.999708, and 0.999913 for groups 1-3, respectively. No test-period SCADA is
read or required.

## 3. Validation design

- Groups 1 and 2: 2022 fit to 2023 development, then 2022-2023 refit to
  locked 2024.
- Group 3: 2023 H1 to H2 nested selection for rules requiring calibration,
  then full 2023 refit to locked 2024.
- Forecast issue day, not hourly row, is the bootstrap unit.
- Selection/calibration and locked evaluation rows are separated.
- Test weather is used only after all feature/model/action decisions are fixed.

## 4. Corrected baseline and final candidates

| Policy | Score | 1-NMAE | FICR | Delta Score vs raw | Bootstrap P(delta > 0) vs parent |
|---|---:|---:|---:|---:|---:|
| Strict corrected raw | 0.623751 | 0.876778 | 0.370725 | 0.000000 | - |
| Current incumbent matched OOF | **0.649163** | **0.882753** | **0.415573** | +0.025412 | frozen production lineage |
| P2 safe direct | 0.640422 | 0.874759 | 0.406085 | +0.016670 | 1.000 |
| safe_cv_best | 0.642489 | 0.877149 | 0.407828 | +0.018737 | above 0.95; q05 +0.000684 vs P2 safe |
| ficr_bayes | 0.642975 | 0.876178 | 0.409772 | +0.019224 | 0.687 vs safe |
| diverse_ensemble | **0.643821** | **0.878346** | **0.409297** | **+0.020070** | **0.993 vs safe** |
| Group-1 recency overlay diagnostic | 0.649699 | 0.882621 | 0.416778 | +0.025948 | rejected: locked q05 -0.004537 |

The Bayes candidate fails the primary incremental promotion threshold. It is
retained only as a single-hypothesis submission probe. The diverse candidate
passes the alternative blend threshold but has a small H2 FICR reversal.
After the observed failure of the safe broad replacement, neither derivative
is recommended for submission. The recency row is a non-materialized OOF
diagnostic, not a candidate file.

## 5. Fold and group results

### Fixed direct-model folds

| Group | Fold | Selected raw path | Score | 1-NMAE | FICR |
|---|---|---|---:|---:|---:|
| 1 | 2022 to 2023 | eligible L1 | 0.571336 | 0.861033 | 0.281638 |
| 1 | 2022-2023 to 2024 | eligible L1 | 0.631541 | 0.885587 | 0.377495 |
| 2 | 2022 to 2023 | eligible L1 | 0.634836 | 0.870588 | 0.399083 |
| 2 | 2022-2023 to 2024 | eligible L1 | 0.669774 | 0.885014 | 0.454534 |
| 3 | 2023 to 2024 | generation-weighted eligible L1 | 0.585895 | 0.859361 | 0.312429 |

### Locked-2024 candidate results

| Candidate | Group | Score | 1-NMAE | FICR | Eligible rows |
|---|---|---:|---:|---:|---:|
| safe | 1 | 0.665597 | 0.879902 | 0.451292 | 4,990 |
| safe | 2 | 0.669774 | 0.885014 | 0.454534 | 4,977 |
| safe | 3 | 0.592095 | 0.866530 | 0.317660 | 4,567 |
| Bayes | 1 | 0.665597 | 0.879902 | 0.451292 | 4,990 |
| Bayes | 2 | 0.671234 | 0.882102 | 0.460365 | 4,977 |
| Bayes | 3 | 0.592095 | 0.866530 | 0.317660 | 4,567 |
| diverse | 1 | 0.665597 | 0.879902 | 0.451292 | 4,990 |
| diverse | 2 | 0.669774 | 0.885014 | 0.454534 | 4,977 |
| diverse | 3 | 0.596093 | 0.870122 | 0.322065 | 4,567 |

### Current-incumbent matched OOF

| Group | Score | 1-NMAE | FICR | Eligible rows |
|---|---:|---:|---:|---:|
| 1 | 0.673267 | 0.892021 | 0.454512 | 4,990 |
| 2 | 0.678466 | 0.885189 | 0.471744 | 4,977 |
| 3 | 0.595757 | 0.871049 | 0.320464 | 4,567 |

The exact reconstruction command is
`python -m experiments.audit_current_incumbent_oof`; it also revalidates the
8,760-row candidate and records source hashes in
`artifacts/current_incumbent_oof_audit.json`.

For diverse group 3, 2024 H1 Score/FICR deltas are +0.007090/+0.009504.
Locked H2 Score still improves by +0.000754, but FICR changes by -0.000876.

## 6. Metric-aware Bayes action

The implemented conditional action maximizes:

    E[-abs(p - Y) + C/(4 * mean_eligible_Y) * Y * price(abs(p - Y))]

for eligible outcomes, where price is 4 inside 6%, 3 inside 8%, and zero
outside. A unit test proves that utility differences are a positive constant
multiple of the prediction-dependent official Score difference.

Eleven eligible-only LightGBM quantiles from 0.05 through 0.95 form a discrete
conditional distribution. Crossing rows are sorted monotonically. The
pre-locked alpha choices are group 1 = 1.0, group 2 = 0.5, and group 3 = 0.5.

| Point rule | Score | 1-NMAE | FICR | Delta Score vs raw |
|---|---:|---:|---:|---:|
| Raw L1 median path | 0.623751 | 0.876778 | 0.370725 | 0.000000 |
| Quantile p50 | 0.623169 | 0.877718 | 0.368620 | -0.000582 |
| Unshrunk Bayes action | 0.627133 | 0.872977 | 0.381289 | +0.003382 |
| Selected shrinkage | 0.628768 | 0.875748 | 0.381789 | +0.005017 |

The action improves high-generation settlement hits but moves away from the
median. On top of the stronger safe policy, group-2 Bayes gives +0.005832 FICR
but -0.002911 1-NMAE, yielding only +0.001460 group Score and +0.000487 macro
Score. This is why it is not the safe default.

## 7. Where FICR changed

### Diverse group-3 member versus safe

| Slice | FICR delta |
|---|---:|
| Actual 10-20% | +0.024242 |
| Actual 20-40% | +0.008833 |
| Actual 40-60% | +0.010356 |
| Actual 60-80% | -0.004036 |
| Actual 80-100% | +0.002067 |
| Lead 12-17 h | +0.003945 |
| Lead 18-23 h | +0.000559 |
| Lead 24-29 h | +0.004753 |
| Lead 30-35 h | +0.007793 |
| DJF / MAM / JJA / SON | +0.000834 / +0.014922 / -0.000903 / +0.001839 |

### Bayes group-2 member versus safe

Bayes gains are concentrated at actual 60-80% (+0.025765), 80-100%
(+0.018506), lead 12-17 h (+0.016435), lead 18-23 h (+0.008014), DJF
(+0.011229), and MAM (+0.008532). It loses FICR for actual 10-60%, lead
24-35 h, JJA, and SON. This explains the negative bootstrap lower bound.

## 8. Adopted and rejected hypotheses

| Hypothesis | Decision | Locked evidence |
|---|---|---|
| Official actual-weighted scorer | adopted | changes baseline selection; tests pass |
| Eligible-only direct L1 | adopted | improves groups 1/2 over all-row path |
| Group-1 prior-OOF affine | adopted | stable group bootstrap |
| Generation-weighted group 3 | adopted | +0.015955 Score versus raw group 3 |
| Training-only SCADA for all groups | rejected | full composite unstable |
| Training-only SCADA only for group 3 | adopted | +0.006200 group-3 Score; positive q05 |
| Direct eligible quantile Bayes action | adopted as diagnostic | +0.005017 versus strict raw |
| Bayes on top of safe | candidate-only | +0.000487 macro; P(delta > 0) 0.687 |
| Broad trajectory | rejected | macro -0.001547; q05 negative |
| Broad site interpolation | rejected | macro -0.001583; q05 negative |
| Broad physical block | rejected | macro -0.002163; q05 negative |
| Capacity-normalized group-3 pooling | rejected | FICR -0.001922; q05 negative |
| Frozen 20% spatial-temporal member | candidate-only | macro +0.001333; P(delta > 0) 0.993 |
| Analog distribution and more trajectory search | stopped | predeclared family stop criterion reached |
| Group-1 recent-12-month direct factor | rejected | H1/H2 mean positive, but locked bootstrap P=0.6315 and q05=-0.004537 |

## 9. Submission candidates and risk

| Candidate | Construction | Risk |
|---|---|---|
| active incumbent 1508386 | Public-confirmed G1 residual and G2 pooled factors; G3 frozen | Lowest available. Still exposed to unknown Private subset shift |
| safe_cv_best | Group-1 eligible L1 plus prior affine; group-2 eligible L1; group-3 weighted L1 plus 27.5% SCADA physical | Rejected. Observed Public Score fell by 0.012949 versus the incumbent |
| ficr_bayes | Safe candidate with group-2 alpha-0.5 quantile Bayes action | Withheld. It inherits the rejected safe core and its incremental bootstrap probability is 0.687 |
| diverse_ensemble | Safe candidate with 20% two-seed spatial-temporal group-3 member | Withheld. It inherits the rejected safe core and locked-H2 FICR is slightly negative |

All three files contain exactly 8,760 rows, match sample keys and column order,
contain no NaN/inf, respect group capacity, and reload as UTF-8.

Operational decision after submission 1511408: only the active incumbent is
recommended. The three Sprint-066 files remain reproducible diagnostic
artifacts but are withheld; no recency CSV exists because its promotion gate
failed.

## 12. Frozen group-3 spatial-temporal displacement follow-up

This follow-up used submission 1508386 as the only production anchor. Its CSV
SHA-256 is
`8056206176d12f21f72fda14fba8fa3b19bcc8a6da7d8a6e389b9e28a40c4902`;
the primary, residual, and frozen-G3 source hashes all matched the incumbent
audit. The official macro invariant was also exercised on the reconstructed
surface: changing only G3 changed every macro component by exactly one third
of the corresponding G3 delta. The two often-confused leaderboard deltas are
kept separate: safe_cv_best OOF to its Public result is -0.007967, while its
Public result is -0.012949 below the incumbent Public result.

Before execution, six candidate families were fixed: nearest grid, 1.5 km
upstream, 3.0 km upstream, -1 h lead, +1 h lead, and 1.5 km upstream followed
by a centred +/-1 h mean. Only weights 0.10, 0.25, and 0.50 were permitted and
movement was capped at 1% of G3 capacity. All shifts stayed inside one LDAPS
issuance. KMA and LDAPS 2023/2024 audits found zero cutoff violations and
actual leads of 12--35 h.

### 2023-only selection and frozen rule

The selection anchor was a causal same-fold KMA-UM isotonic power curve. This
is an incumbent-family anchor, not a claim that a byte-identical 2023 OOF for
submission 1508386 exists. Candidate increments were displaced-minus-centred
LDAPS power curves.

| Fold | Frozen-rule Score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|
| 2023 H1 train -> Q3 validation | +0.000777 | +0.001549 | +0.000006 |
| 2023 through Q3 train -> Q4 validation | +0.002846 | +0.003360 | +0.002331 |

Three rule/weight pairs passed the predeclared two-fold sign gates. The single
frozen choice was `d1_nearest_grid`, weight 0.50, because it had the largest
mean forward Score delta (+0.001811). No rule or dose was changed after the
2024 result was opened.

### One-shot 2024 result versus exact incumbent G3

| Period | Incumbent Score | Candidate Score | Score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|---:|---:|
| Full | 0.595757 | 0.595935 | +0.000179 | -0.000075 | +0.000432 |
| H1 | 0.574177 | 0.575100 | +0.000922 | +0.000570 | +0.001274 |
| H2 | 0.618330 | 0.617712 | -0.000618 | -0.000795 | -0.000440 |

G1 and G2 arrays were exact copies of the incumbent. Consequently the macro
Score moved only +0.000060, from 0.649163 to 0.649223. The requested target
would require G3 near 0.628267; the frozen result is still about 0.032332
below that group score and is not evidence for reaching 0.660.

Paired target-date block bootstrap used 2,000 resamples across 367 forecast
days: P(delta > 0) = 0.5825, q05 = -0.001495, median = +0.000198, and mean =
+0.000198. The q05 floor passed, but the 0.80 probability gate did not.

| Slice | Score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|
| DJF | -0.000104 | -0.000584 | +0.000376 |
| MAM | +0.000218 | +0.000862 | -0.000427 |
| JJA | +0.001898 | -0.000225 | +0.004022 |
| SON | -0.002036 | -0.000535 | -0.003537 |
| Lead 12--17 h | +0.003062 | -0.000083 | +0.006208 |
| Lead 18--23 h | -0.003395 | -0.000324 | -0.006467 |
| Lead 24--29 h | -0.000415 | -0.000223 | -0.000606 |
| Lead 30--35 h | +0.000518 | +0.000281 | +0.000756 |

Monthly Score deltas for January through December were respectively +0.004196,
+0.001177, +0.000236, +0.002510, -0.001192, -0.004015, +0.004162,
+0.000827, -0.012389, -0.004231, +0.003174, and -0.003992. September is the
largest failure and explains much of the H2 reversal; it also violates no
extra post-hoc gate because the displacement and weight remain frozen.

The candidate failed three required gates: full G3 Score gain was below
+0.0045, H2 Score was not positive, and bootstrap probability was below 0.80.
It passed the NMAE, full FICR, q05, season-collapse, and unchanged-G1/G2 gates.
Per protocol, no submission CSV was generated and this family is closed.

The next audit priority is structural rather than another overlay sweep:

1. reconcile the 01:00--00:00 forecast-day convention, the single missing
   2023 G3 label, 2024 boundary rows, and every label/NWP timestamp;
2. verify G3 capacity, turbine membership, turbine coordinates, and SCADA/KPX
   aggregation units against `info.xlsx` and `data_description.md`;
3. verify LDAPS/KMA grid coordinate order, farm centroid, nearest-cell distance,
   and land/sea placement;
4. re-audit source/cycle coverage and issuance selection at the year boundary;
5. quantify any hour offset between 10-minute SCADA aggregation and KPX labels.

Reproduction:

    python -m pytest tests/test_group3_frozen_displacement.py -q
    python -m experiments.group3_frozen_displacement --smoke
    python -m experiments.group3_frozen_displacement --bootstrap-repetitions 2000

Measured runtime on the current machine was 5.4 s for the targeted tests,
2.1 s for smoke, and 5.6 s for the full experiment. Rerunning the full command
requires `--overwrite` because the experiment report is immutable by default.

## 10. Remaining bottlenecks

1. Group 3 remains far below groups 1 and 2: incumbent-matched OOF is 0.595757
   versus 0.673267 and 0.678466.
2. The selected incumbent OOF Score is still 0.010837 below 0.660.
3. Quantile crossing before repair is 74-84%, showing weak distributional
   coherence and limiting Bayes reliability.
4. Group-3 history provides only one full year-to-year evaluation fold.
5. Broad engineered feature blocks do not transfer; gains now depend on small,
   regime-specific effects that are easy to overfit.
6. The latest full-policy Public transfer failure is much larger than the
   remaining local increments, so broad replacements are not submission-safe.

## 11. Reproduction

Smoke:

    python train.py --pipeline sprint066 --data-dir data --artifact-dir artifacts/sprint066_smoke --direct-estimators 3 --quantile-estimators 3
    python inference.py --pipeline sprint066 --data-dir data --artifact-dir artifacts/sprint066_smoke --output-dir submissions/sprint066_smoke --diverse-member artifacts_final/spatiotemporal_final/spatiotemporal_member.csv

Full:

    python train.py --pipeline sprint066 --data-dir data --artifact-dir artifacts/sprint066_models --include-diverse-member
    python inference.py --pipeline sprint066 --data-dir data --artifact-dir artifacts/sprint066_models --output-dir submissions/sprint066
    python -m experiments.audit_current_incumbent_oof
    python -m experiments.group1_recency_overlay
    python -m pytest -q

Measured on the current machine: full tabular/Bayes/SCADA training 319.5 s,
the serialized two-seed spatial-temporal refit 81.4 s, artifact-only inference
and validation 13.1 s, the recency audit 119.1 s, and the complete test suite
about 25-31 s. Training does not open test data; inference rebuilds the
legal-cycle test tensor and requires neither train labels nor test SCADA.

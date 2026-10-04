# 정산 지표 보정과 강도별 공개 결과

그룹 1·2의 결합 보정이 공개 평가로 이전된 결과와 점수 구성요소를 분해한 기록입니다. 목표까지 남은 차이를 이미 달성한 성과로 읽지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Public 1508365 FiCR dose-response follow-up

## Outcome

Submission 1508365 validated that the joint G1/G2 public-positive expansion
transfers to the 2025 public set.

| Submission | Score | 1-NMAE | FiCR | Change vs 1502437 |
| --- | ---: | ---: | ---: | ---: |
| 1502437 incumbent | 0.6461250914 | 0.8757842477 | 0.4164659352 | - |
| 1508365 G1 0.1375 / G2 0.095 | 0.6470679857 | 0.8759467997 | 0.4181891717 | +0.0009428943 |

The score gain decomposes into `+0.0001625520` 1-NMAE and `+0.0017232365`
FiCR. At the current 1-NMAE, a 0.650 score requires FiCR 0.4240532003,
which is 0.0058640286 above submission 1508365. The user-provided live
leaderboard also places the visible rank-88 line near 0.65198, so 0.650 is an
intermediate milestone rather than a competitive endpoint.

## Selected next candidate

The selected candidate combines only factor directions already isolated as
positive on the public leaderboard:

- G1 residual factor remains at the scored weight 0.1375.
- G2 pooled-weather factor expands from 0.095 to 0.1825.
- G3 KMA/GFS factor expands from its scored dose to 1.25x.

Candidate:
`artifacts_final/candidates/public_positive_g1w1375_g2w1825_g3x125_20260802.csv`

Suggested DACON title: `Public Positive G1w1375 G2w1825 G3x125`

SHA-256:
`8020678b261a17727aa9ef3fd2d9d1c53b70f46f9d708e1479632351b869dcf9`

The candidate contains 8,760 rows, exactly matches the required identifiers
and columns, stays within every capacity bound, and passed `CandidateValidator`.
A second deterministic generation produced the same SHA-256.

## Validation evidence

### G1/G2 anchor

Relative to submission 1502437's OOF lineage, the G1 0.1375 / G2 0.1825
anchor produced:

| Period | Score delta | 1-NMAE delta | FiCR delta |
| --- | ---: | ---: | ---: |
| Q1 | +0.0000362 | -0.0003275 | +0.0003999 |
| Q2 | +0.0016661 | -0.0000404 | +0.0033725 |
| H2 | +0.0015264 | +0.0002438 | +0.0028090 |
| Full year | +0.0011516 | +0.0000214 | +0.0022818 |

Score improved in 10 of 12 months. The 5% score lower bounds were positive
for IID public-like subsets (`+0.0001178`), month-stratified public-like
subsets (`+0.0001246`), and H2 issue-cycle bootstrap (`+0.0001734`). The
component-specific transfer observed at submission 1508365 projects this
anchor to score 0.6489825 and FiCR 0.4221354.

### G3 dose response

Submissions 1494670 and 1501460 differed only in G3. The KMA/GFS treatment
improved public score by 0.0004896767 and FiCR by 0.0012305746 while reducing
1-NMAE by 0.0002512213. Scaling that same production factor gave the following
OOF response:

| G3 dose | Full incremental score | H2 incremental score | H2 worst monthly score | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1.10x | +0.0010589 | +0.0014999 | +0.0006222 | safe, lower utility |
| 1.25x | +0.0014400 | +0.0020955 | +0.0005262 | selected |
| 1.50x | +0.0011177 | +0.0022852 | -0.0003896 | reject |
| 2.50x | -0.0002689 | +0.0013237 | -0.0099356 | reject |

At 1.25x, all six H2 months improved score. The 5,000-repetition H2
issue-cycle bootstrap had positive 5% lower bounds for score (`+0.0009799`),
1-NMAE (`+0.0001503`), and FiCR (`+0.0016195`), with all components positive
in 99.34% of resamples. Production movement versus the G3 treatment is small:
53.30 kWh mean absolute, 241.45 kWh p95, and 262.50 kWh maximum.

The isolated-public-factor dose response adds a projected `+0.0001521` score
and `+0.0004538` FiCR over the G1/G2 anchor. The combined projection is:

| Metric | Projection |
| --- | ---: |
| Score | 0.6491346 |
| 1-NMAE | 0.8758168 |
| FiCR | 0.4225892 |
| Gap to 0.650 | 0.0008654 |

This is an extrapolation from scored factor points, not a guaranteed public
score. It is selected because every added direction is public-positive and
the locked G3 stress tests are materially stronger than the rejected options.

## Rejected paths

- Further G1 expansion was rejected: moving from 0.1375 to 0.1600 reduced
  local group score by 0.0002740.
- The G2 score response peaked near 0.1825. Extreme weights improved isolated
  FiCR slightly but reduced calibrated total score.
- A 1,715-rule power/movement router transferred from Q1 to Q2 but lost H2
  1-NMAE and failed closed.
- A 2,888-policy threshold, affine, and piecewise calibration search produced
  no Q2-transferable policy and failed closed.
- G3 doses at or above 1.50x introduced negative quarters or H2 months and
  were not written as submission candidates.

## Submission sequence

1. Submit `public_positive_g1w1375_g2w1825_g3x125_20260802.csv` first.
2. If attribution is needed after its score arrives, submit the G2-only anchor
   `public_positive_g1w1375_g2w1825_g3frozen_20260802.csv`.
3. Use `public_bridge_g1base_g2w095_g3frozen_20260802.csv` only when a daily
   slot can be spent on exact G1/G2 effect decomposition.

The first result determines the next dose update. A positive result below
0.650 supports another locally bounded factor step; a regression freezes the
new G3 dose and falls back to the G2-only anchor.

## Submission 1508378 observation

The selected joint candidate was submitted as 1508378 and established another
public best:

| Submission | Score | 1-NMAE | FiCR |
| --- | ---: | ---: | ---: |
| 1508365, G2 0.095 / G3 1.00x | 0.6470679857 | 0.8759467997 | 0.4181891717 |
| 1508378, G2 0.1825 / G3 1.25x | 0.6474693752 | 0.8759267969 | 0.4190119536 |
| Delta | +0.0004013895 | -0.0000200028 | +0.0008227819 |

The factor direction is confirmed, but the public score and FiCR response was
only about 18--19% of the corresponding full-year OOF response. The current
1-NMAE implies that score 0.650 now requires FiCR 0.4240732031, leaving a
FiCR gap of 0.0050612495. Repeated extrapolation of the same joint direction
cannot credibly close that gap without first separating its two components.

## Post-1508378 breakthrough audit

Three routes were checked without using the public score for policy selection.

1. A 180-candidate stabilizer grid mixed the legacy exact driver and the
   issue-trajectory TCN around the new G1/G2/G3 anchor. No candidate passed
   the Q1 component and monthly gates.
2. A 280-candidate mechanism-diversity grid tested the exact driver, SCADA
   stack, LDAPS multiresolution surface, and issue-trajectory TCN by group and
   dose. Five candidates passed Q1, none transferred through Q2. Six
   retrospective all-period nonnegative points existed only at tiny doses;
   their full-year gains were on the order of `1e-5`, below submission value.
3. The retained lineage was audited for a full-season, incumbent-matched 2023
   OOF surface. The nested Base-v2 cache starts at 2023-12-01 and all exact
   incumbent-matched caches cover 2024 only. A defensible conditional
   distribution or direct-utility retry therefore requires rebuilding an
   earlier year-forward baseline, not retuning the already exposed 2024 H2.

This agrees with external wind-forecasting evidence: weather ensembles and
probabilistic post-processing can improve point decisions, but the final power
ensemble must be calibrated, and a market objective should choose an action
from the conditional distribution rather than blindly optimizing MAE.

- Bruninx et al., [Probabilistic Wind Power Forecasting with Tree-Based
  Machine Learning and Weather Ensembles](https://arxiv.org/abs/2602.13010)
- Muñoz et al., [Feature-driven Improvement of Renewable Energy Forecasting
  and Trading](https://arxiv.org/abs/1907.07580)
- Phipps et al., [Evaluating Ensemble Post-Processing for Wind Power
  Forecasts](https://arxiv.org/abs/2009.14127)

## Next factorial probe after 1508378

Submit the already validated G2-only anchor:

`artifacts_final/candidates/public_positive_g1w1375_g2w1825_g3frozen_20260802.csv`

Suggested title: `Public Positive G1w1375 G2w1825 G3 Frozen`

SHA-256:
`8056206176d12f21f72fda14fba8fa3b19bcc8a6da7d8a6e389b9e28a40c4902`

Let `A` be submission 1508365, `B` submission 1508378, and `C` this G2-only
probe. Because G2 and G3 are different official macro groups, the exact public
component effects are:

- additional G2 dose: `C - A`
- additional G3 dose: `B - C`
- checksum: `(C - A) + (B - C) = B - A`

Only one submission is needed to identify both effects. If `C > B`, G3 1.25x
is removed. If `C < A`, the extra G2 dose is removed. If `A < C < B`, both
directions are positive and the stronger component receives the next bounded
dose. This preserves the remaining daily slots for an evidence-based update.

## Submission 1508386 factorial result

The G2-only probe was submitted as 1508386 and narrowly exceeded the joint
G2/G3 candidate:

| Submission | G2 dose | G3 dose | Score | 1-NMAE | FiCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1508365 (A) | 0.0950 | 1.00x | 0.6470679857 | 0.8759467997 | 0.4181891717 |
| 1508378 (B) | 0.1825 | 1.25x | 0.6474693752 | 0.8759267969 | 0.4190119536 |
| 1508386 (C) | 0.1825 | 1.00x | 0.6474704399 | 0.8761125755 | 0.4188283043 |

The exact public factor effects are therefore:

| Isolated effect | Score | 1-NMAE | FiCR |
| --- | ---: | ---: | ---: |
| Additional G2 dose, `C - A` | +0.0004024542 | +0.0001657758 | +0.0006391326 |
| G3 1.25x, `B - C` | -0.0000010647 | -0.0001857786 | +0.0001836493 |

G3 1.25x buys FiCR by giving back almost exactly the same amount of nMAE and
is net negative. G3 is frozen at 1.00x. The meaningful public gain comes from
G2, for which the nMAE response is also positive. At submission 1508386's
1-NMAE, score 0.650 requires FiCR `0.4238874245`; the remaining FiCR gap is
`0.0050591202`.

## Incremental G2 fail-closed gate

A dense locked-OOF grid searched G2 weights from 0.1850 to 0.3000 relative to
the scored 0.1825 anchor. The apparent full-year peak was 0.2350, but its
incremental gain was only `+0.0000537` locally and did not transfer by period:

- Q1 score improved approximately `+0.0005844`.
- Q2 score fell approximately `-0.0000311`.
- H2 score fell approximately `-0.0001875`.

No searched weight passed the Q2/H2/full/month gates. For weight 0.2350, the
minimum 5% lower bounds across 5,000 complementary subset draws and 2,000 H2
issue-block bootstraps were score `-0.0010557`, 1-NMAE `-0.0001371`, and FiCR
`-0.0020532`. The public component calibration was deliberately disabled for
selection because the local nMAE calibration delta and the observed public
nMAE delta have opposite signs. G2 is frozen at 0.1825.

## Independent prior-year OOF

To test whether the FiCR-oriented G2 quantile mechanism was repeatable, a new
causal group-2 OOF surface was built for the four 2023 issue seasons. Each
outer season was excluded from alpha, train-row policy, early-stopping, and
iteration selection. The diagnostic used a reduced base-feature profile with
alpha 0.65/0.70 and one fixed seed; it created neither test predictions nor a
submission.

| 2023 outer season | Quantile minus L1 score |
| --- | ---: |
| DJF | +0.027242 |
| MAM | +0.021643 |
| JJA | -0.007939 |
| SON | +0.016423 |
| Full prior year | +0.017050 |

The full-year component deltas were 1-NMAE `-0.007328` and FiCR `+0.041429`.
The season-stratified issue-block bootstrap was positive in 100% of 2,000
draws with a score 5% lower bound of `+0.007753`. This confirms the broad G2
quantile/FiCR mechanism in another year, but the negative JJA result explains
why unrestricted dose escalation is unstable.

## Seasonal policy transfer audit

Three increasingly narrow additions above 0.1825 were then checked on both
2023 and 2024 OOF:

- January--March weight 0.2350: positive annual mean in both years, but March
  2024 was negative and every required resampling lower bound failed.
- January--February weight 0.2350: positive annual mean in both years, but
  score and FiCR 5% lower bounds remained negative.
- February-only weight 0.2350: 2023 score `+0.000257` and 2024 score
  `+0.000096`, but the 2023 score/FiCR lower bounds and both-year FiCR lower
  bounds were negative.

A policy selected only on 2023 (`DJF/MAM/SON=0.35`, `JJA=0.05`) gained
`+0.004419` in 2023 but lost `-0.001521` on untouched 2024 validation. This
closes calendar-season routing: the apparent monthly optimum is not stable
across years.

## Final decision for 2026-08-02

Submission 1508386 remains the selected incumbent:

`artifacts_final/candidates/public_positive_g1w1375_g2w1825_g3frozen_20260802.csv`

SHA-256:
`8056206176d12f21f72fda14fba8fa3b19bcc8a6da7d8a6e389b9e28a40c4902`

No additional candidate passes the locked validation gates. The final daily
slot should be preserved rather than spent on a projected `1e-4` gain with a
negative resampling lower bound. The next model family must add genuinely new
conditional-distribution or weather-ensemble information; further G2/G3 dose
or calendar retuning is closed.

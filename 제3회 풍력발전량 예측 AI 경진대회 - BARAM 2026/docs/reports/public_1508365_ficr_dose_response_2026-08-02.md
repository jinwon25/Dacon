# Public 1508365 FiCR dose-response follow-up

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

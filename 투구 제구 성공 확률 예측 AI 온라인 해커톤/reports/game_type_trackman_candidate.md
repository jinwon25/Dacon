# game_type_trackman_f100_r10_v1

- Base: incumbent 0.35 engineered LightGBM + 0.65 official RF.
- `game_type=R`: replace only the RF component with the frozen half-life-1 recency RF, then blend Trackman at 10%.
- `game_type=F`: use the rolling-damped Trackman probability at 100%.
- Other game types: retain the candidate base (Trackman weight 0%).
- All weights are row-local constants; no test-batch statistic is used.

| year | incumbent Brier | candidate Brier | delta |
| ---: | ---: | ---: | ---: |
| 2021 | 0.246945046 | 0.246190484 | -0.000754562 |
| 2022 | 0.244169346 | 0.244102628 | -0.000066717 |
| 2023 | 0.251137482 | 0.249783946 | -0.001353536 |
| 2024 | 0.248460961 | 0.248370280 | -0.000090681 |

- Recency-weighted delta: **-0.000531133**
- Worst-fold delta: **-0.000066717**
- Combined pitcher-season bootstrap P(improve): **1.0000**
- This is a local OOF research candidate; the public score is not claimed until a DACON submission is made.

# Hybrid recency/Trackman candidate

- Frozen recipe: incumbent with half-life-1 RF substituted only for `game_type=R`, then 5% rolling-damped Trackman LGB.
- Trackman linkage and profiles use only seasons before each forecast origin.
- No validation target enters linkage, feature construction, drift forecasts, or row-level application.

| year | incumbent | candidate | delta |
| ---: | ---: | ---: | ---: |
| 2021 | 0.246945046 | 0.246865966 | -0.000079080 |
| 2022 | 0.244169346 | 0.244151170 | -0.000018176 |
| 2023 | 0.251137482 | 0.251047904 | -0.000089578 |
| 2024 | 0.248460961 | 0.248440533 | -0.000020428 |

- Recency-weighted delta: **-0.000046588**
- Latest-year delta: **-0.000020428**
- Worst-fold delta: **-0.000018176**
- Combined pitcher-season bootstrap P(improve): **1.0000**
- Fixed gate: **PASS**
- Selection caveat: all four outer years have already been reused extensively as development data; Public submission is a deployment probe, not independent proof.

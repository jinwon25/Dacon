# Cross-group transfer sprint — 2026-07-13

## Public feedback that changed the direction

The positive and negative turbine group-3 probes both lost public score:

| submission | score | 1-NMAE | FICR | delta vs best |
|---|---:|---:|---:|---:|
| current best `1488084` | 0.6403102237 | 0.8753191781 | 0.4053012693 | — |
| turbine +10% `1488631` | 0.6401243372 | 0.8751815837 | 0.4050670908 | -0.0001858865 |
| turbine -10% `1488632` | 0.6400644769 | 0.8754264537 | 0.4047025001 | -0.0002457468 |

The reverse probe improved NMAE but sharply reduced FICR. This ruled out further one-dimensional group-3 turbine adjustment.

## Bottleneck work

A reusable NWP feature cache was added. It stores 26,304 train rows × 1,089 columns (109MB) and 8,760 test rows × 1,089 columns (36MB), avoiding repeated raw LDAPS/GFS pivots.

A five-quantile conditional distribution model was then trained to maximize expected NMAE/FICR utility. On a strict 2024 first-half selection and second-half evaluation, it improved its own baseline by `+0.00309`, mainly through FICR. However, its correction did not transfer consistently to the strongest existing OOF models, so no submission was produced from that family.

Residual stacking was also rejected: it improved NMAE but reduced FICR for both groups 1 and 2.

## Cross-group transfer

Normalized concurrent generation is strongly related across the adjacent farms:

| year | corr(group 1, group 3) | corr(group 2, group 3) |
|---|---:|---:|
| 2023 | 0.8849 | 0.9138 |
| 2024 | 0.9238 | 0.9447 |

An ExtraTrees model maps normalized group-1/group-2 generation, their average and difference, hour, and season to group-3 generation. Validation trains on 2023 actual group relationships and predicts 2024 using OOF group-1/group-2 model predictions, matching the inference-time uncertainty more closely than using target-period actuals.

The final correction is applied only when:

- normalized group-1/group-2 prediction disagreement is at most 8%;
- cross-group member/base group-3 disagreement is at most 6% of capacity;
- base group-3 prediction is at least 10% of capacity;
- blend weight is 25%.

## Robust validation

The fixed gate was evaluated against three different group-3 OOF bases:

| OOF base | annual score delta | annual 1-NMAE delta | annual FICR delta | H2 score delta | months improved |
|---|---:|---:|---:|---:|---:|
| weighted | +0.000958 | +0.000615 | +0.001300 | +0.000980 | 8/12 |
| global | +0.000607 | +0.000829 | +0.000385 | +0.001381 | 5/12 |
| final pool | +0.000506 | +0.000344 | +0.000667 | +0.000423 | 8/12 |

All three bases improved in both annual and forward-half score, and both metric components improved annually. This is materially stronger evidence than the rejected turbine probes.

## Submission decision

- archived file: `submissions/archive/blend_best_crossg3_25_agree8_delta6.csv`
- DACON title: `Best crossgroup G3 25 agree8 delta6`
- groups 1/2: unchanged from public best
- group 3: 5,761/8,760 rows changed
- mean absolute group-3 movement: 99.64 kWh
- 95th percentile absolute movement: 281.17 kWh

Keep submission `1488084` selected until this candidate's public score is known.

## Public result

The candidate was submitted at `2026-07-13 07:44:04` and scored `0.6414690556` (`1-NMAE 0.8754254686`, `FICR 0.4075126426`). Relative to submission `1488084`, the gains were:

- score: `+0.0011588319`;
- 1-NMAE: `+0.0001062905`;
- FICR: `+0.0022113733`.

The public result confirms the cross-group transfer signal and the agreement gate. This submission is the new selected best.

## Next controlled expansion

After public confirmation, the same transfer family was re-evaluated without changing the model. Increasing the blend to 45% and the member-disagreement gate to 8% improved all three proxy bases in both annual and forward-half validation:

| OOF base | annual score delta | annual 1-NMAE delta | annual FICR delta | H2 score delta |
|---|---:|---:|---:|---:|
| weighted | +0.001990 | +0.001630 | +0.002351 | +0.001232 |
| global | +0.001520 | +0.002383 | +0.000657 | +0.002897 |
| final pool | +0.001299 | +0.000966 | +0.001632 | +0.001809 |

The next candidate is `submissions/blend_best_crossg3_45_agree8_delta8.csv`, titled `Best crossgroup G3 45 agree8 delta8`. It changes 6,612 group-3 rows with a mean absolute movement of 241.94 kWh. Keep the confirmed 25% candidate selected until the new public result is known.

## Expansion public result

Submission `1489614` scored `0.6414589725` (`1-NMAE 0.8756124561`, `FICR 0.4073054888`). Relative to submission20, 1-NMAE improved by `0.0001869875`, but FICR declined by `0.0002071538`; total score was `0.0000100831` lower. The broad 45%/8% expansion is rejected and submission20 remains selected. Further work must preserve the 25% confirmed base and increase correction only in regimes where FICR gains are stable.

## Smooth-member candidate

No simple direction, generation range, or disagreement sub-regime supported stronger correction across all proxies and both validation halves. Instead, the successful 25% policy was held fixed and only the ExtraTrees member was smoothed from `min_samples_leaf=30` to `120`, using all eight transfer features.

| OOF base | annual score delta | annual 1-NMAE delta | annual FICR delta | H2 score delta |
|---|---:|---:|---:|---:|
| weighted | +0.001304 | +0.000613 | +0.001994 | +0.001153 |
| global | +0.000496 | +0.000894 | +0.000099 | +0.001399 |
| final pool | +0.001259 | +0.000401 | +0.002116 | +0.000937 |

The candidate is `submissions/blend_best_crossg3_smoothsolo25_agree8_delta6.csv`, titled `Best crossgroup smooth G3 25 agree8 delta6`. It differs from submission20 by only 42.55 kWh mean absolute group-3 movement, making it a lower-risk structural refinement than another weight expansion.

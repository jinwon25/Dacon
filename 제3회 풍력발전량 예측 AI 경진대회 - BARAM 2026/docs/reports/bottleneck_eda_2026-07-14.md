# Group-3 bottleneck EDA - 2026-07-14

## Decision

Keep submission `1491284`, `blend_best_crossg3_traj5_consensus.csv`, selected. Its public gain
over submission20 was only `+0.0000109493`, so the trajectory and cross-group micro-blend surface
is considered saturated.

No new submission CSV was created in this sprint. Every tested high-power or distribution-shift
correction failed at least one forward-half or proxy-robustness check. A subsequent explicit
nominal/curtailed/offline state mixture failed as well, showing that the latent state is not
predictable enough from NWP alone to use as a post-processing correction.

## Data coverage and target distribution

Group 3 has no observed label at all in 2022. It has 8,759 observed hours in 2023 and 8,778 in
2024, compared with nearly complete three-year histories for groups 1 and 2.

| group-3 year | observed | eligible fraction | mean / median ratio | >=80% capacity |
|---|---:|---:|---:|---:|
| 2022 | 0 | - | - | - |
| 2023 | 8,759 | 55.34% | 0.269 / 0.139 | 8.01% |
| 2024 | 8,778 | 52.02% | 0.261 / 0.118 | 8.11% |

The high-power class is therefore real and stable in prevalence, but small enough that a few
false-positive corrections can erase a FICR gain.

## Current group-3 error surface

The publicly confirmed 25% cross-group correction plus the submitted 5% trajectory smoothing was
reconstructed on all three retained 2024 OOF surfaces.

| proxy | score | 1-NMAE | FICR |
|---|---:|---:|---:|
| weighted | 0.58893 | 0.86267 | 0.31519 |
| global | 0.60334 | 0.86713 | 0.33955 |
| final pool | 0.61241 | 0.86369 | 0.36112 |

The proxy spread is not negligible: its median is 3.52% of capacity and its 95th percentile is
6.81%. Spread has only `0.124` correlation with absolute weighted-proxy error, so disagreement is
a useful risk flag but not an adequate correction direction by itself.

The dominant weighted-proxy failure is concentrated at the top of the true output distribution:

| true capacity ratio | rows | bias kWh | 1-NMAE | FICR | error >8% |
|---|---:|---:|---:|---:|---:|
| 0.1-0.2 | 921 | +2,753 | 0.8649 | 0.4154 | 56.7% |
| 0.7-0.8 | 468 | -1,583 | 0.9015 | 0.5453 | 43.2% |
| 0.8-0.9 | 274 | -3,132 | 0.8483 | 0.2051 | 77.4% |
| 0.9-1.0 | 438 | -4,597 | 0.7811 | 0.0098 | 98.6% |

This is not a global scale error. The low-output region is overpredicted while the high-output
tail is sharply underpredicted. A broad upward calibration therefore improves one region and
damages another, matching the public behavior of the rejected 45% cross-group expansion.

## Physical regimes and distribution shift

The high-power signal exists in the NWP data and is stable across labeled years. The best
single-feature high-power classifiers are:

| feature | 2023 AUC | 2024 AUC |
|---|---:|---:|
| LDAPS hub-wind cubed spatial std | 0.8873 | 0.9219 |
| LDAPS hub-wind squared spatial std | 0.8822 | 0.9075 |
| LDAPS 10 m wind, grid 13 | 0.8804 | 0.9236 |
| LDAPS 117 m hub wind, grid 13 | 0.8799 | 0.9217 |

Wind strength alone does not identify the operating state. Among 2024 hours with modeled hub
wind at least 10 m/s, the actual high-power fraction varies greatly by direction. It is about
28-33% in the 0-90 degree sectors, 5.3% in 315-360 degrees, and zero in the sufficiently sampled
180-270 degree sectors. December is the clearest temporal counterexample: strong-wind actual
generation averages 57.2% of capacity, but only 2.0% of those hours reach 80% capacity.

The 2025 NWP period is also systematically windier than 2024:

| feature | 2024 mean | 2025 mean | PSI 2024->2025 |
|---|---:|---:|---:|
| LDAPS group-3 hub wind | 9.230 | 10.055 | 0.0420 |
| LDAPS hub wind grid 8 | 9.407 | 10.430 | 0.0467 |
| LDAPS hub wind grid 13 | 9.625 | 10.642 | 0.0462 |
| GFS group-3 hub wind | 4.001 | 4.492 | 0.0350 |
| GFS group-3 gust | 4.673 | 5.234 | 0.0383 |

The PSI values indicate a modest rather than catastrophic covariate shift, but the shift points
directly toward the regime where the current model is weakest. This makes blind extrapolation of
the average power curve especially risky.

## EDA-driven modeling experiments

All deltas below are group-3 deltas on the retained 2024 OOF surfaces.

1. A full-feature high-power classifier achieved 2024 AUC `0.9372`. One seed produced a weighted
   full-year gain of `+0.00162`, but two additional seeds and their consensus made weighted/global
   H2 worse. The apparent gain was seed-sensitive.
2. Adding the current group-3 prediction and group-1/group-2 driver forecasts reduced some false
   positives. At the strictest useful H1-to-H2 gate, weighted improved `+0.00118`, while global
   and pool fell `-0.00198` and `-0.00259`.
3. A 25-class conditional residual distribution optimized the official NMAE/FICR utility around
   the current forecast. At only 10% movement, H2 deltas were `-0.00015`, `-0.00213`, and
   `+0.00117` for weighted/global/pool.
4. Unsupervised 2023-to-2024 density-ratio weighting improved its standalone H2 score by
   `+0.00981`, but worsened H1 by `-0.00239` and did not transfer to the strong proxies.
5. Seasonal experts slightly improved the standalone full-year model at a 10% blend, but split
   into an H1 loss and H2 gain. Their correction also changed sign across proxy families.
6. Pretraining on cross-group pseudo-labels for missing 2022 group-3 rows improved the standalone
   full-year proxy, but its H2 NMAE/FICR and transfer direction were unstable.
7. A turbine-SCADA operating-state model separated low-potential, nominal, and limited states.
   Low-potential and nominal one-vs-rest AUCs were `0.961` and `0.905`, but the important limited
   state reached only `0.732`; nominal versus limited AUC within high-potential rows was `0.653`.
   The expert mixture reduced the standalone score and every proxy's full-year score.
8. Nine cumulative classifiers formed an ordinal output distribution from 10% through 90%
   capacity. Expected-utility movement improved its own coarse median in both halves, but the
   FICR displacement transferred with opposite H1/H2 signs. A 4% disagreement gate left only
   about 100 actionable rows and a negligible gain.
9. A shared multi-task ordinal model stacked all three groups with aligned group-location
   features and group-macro weights. Group-1/group-2 observations supplied 2022 auxiliary data
   without pseudo-labeling group 3. Its best utility blend scored `0.59716` in H1 but only
   `0.57167` in H2 (`0.58468` full), versus `0.56740 / 0.61145 / 0.58893` for the current weighted
   surface. H1-only injection improved weighted but reduced global FICR at every tested gate, so
   no common proxy policy exists.

These failures all support the same diagnosis: the remaining uncertainty is not merely the
conditional mean of output. It is a latent operating-state problem, and FICR makes a wrong state
choice much more expensive than a small mean error.

## Operating-state follow-up

The SCADA follow-up used a turbine-level 90th-percentile wind-power envelope learned on 2023.
For hours with complete turbine data, a potential output below 20% was labeled low-potential;
remaining hours were nominal when actual/potential output was at least 75%, and output-limited
otherwise.

| state | 2023 rows | 2024 rows | 2024 one-vs-rest AUC |
|---|---:|---:|---:|
| low potential | 4,061 | 4,329 | 0.961 |
| nominal | 2,836 | 2,796 | 0.905 |
| output limited | 1,557 | 1,164 | 0.732 |

The apparently strong nominal AUC is partly driven by separating low wind from high wind. Once
restricted to rows with at least 20% potential output, nominal versus limited AUC falls to
`0.653`. The mixture therefore cannot safely decide which high-wind branch to use, and even a 10%
injection lowered all three proxy scores. This family is rejected rather than promoted to a CSV.

## Multi-task ordinal follow-up

The final bounded base-model experiment used one shared LightGBM tree structure over 178 aligned
features. Three group-specific rows were formed per timestamp, group ID was retained, and each
group contributed equal total sample weight. Nine cumulative threshold classifiers represented
10-90% capacity.

The model did exactly what the EDA suggested in H1, raising both FICR and NMAE relative to its
coarse median, but the temporal sign reversed in H2. Restricting it to H1 protected H2, yet the
same member was helpful for weighted and harmful for global/final-pool at common gates. It fails
the raw two-half and multi-proxy requirements and is rejected.

## OOF parity audit and next action

The repeated proxy disagreement exposed an infrastructure bottleneck: none of the three retained
OOF surfaces exactly reproduces submission23's pre-cross-group group-3 base. Their prediction
spread has a 3.52% median and 6.81% 95th percentile, which is on the same scale as every attempted
FICR correction.

The exact historical OOF cannot be composed from retained artifacts:

- the old `artifacts_blend` run predates its current `prediction_cache.npz` writer;
- the historical `artifacts_scada_stack_hist` directory/cache was not retained;
- `scada_proxy_stack_hist.py` computed validation predictions but then replaced them with NaNs,
  so it could not be used as a lineage component.

The SCADA script is now fixed to retain the selected validation vector, test vector, timestamps,
truth, selected weight, and official competition metric in `prediction_cache.npz`. The group-3
`blend_v1 -> over115 -> SCADA 5%` pre-cross OOF lineage has since been retrained and frozen. See
`docs/reports/exact_group3_oof_2026-07-14.md`. New residual or regime models should be selected
against that exact group-3 surface; the three broad proxies should remain stress tests.

The remaining parity gap is the complete group-1/group-2 driver lineage used by the cross-group
member. Until that is rebuilt, further correction searches are not statistically identifiable.
Submission `1491284` remains the only recommended submission.

## Reproduction

```bash
python -m experiments.bottleneck_eda
python -m pytest -q
```

The machine-readable output is
`artifacts_bottleneck_eda/bottleneck_eda_report.json`. The final test suite passes with `25 passed`.

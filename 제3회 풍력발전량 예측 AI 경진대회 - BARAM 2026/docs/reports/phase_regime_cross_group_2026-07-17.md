# Lead-phase and weather-regime cross-group sprint - 2026-07-17

## Decision

Reject and archive `blend_best_phase_regime_crossg3.csv`. Submission `1494535` scored
`0.6411378997` (`1-NMAE 0.8756126769`, `FICR 0.4066631226`). Relative to the selected best
submission `1494307`, the score fell by `-0.0005174729`. The local NMAE direction transferred
(`+0.0001573935` publicly), but FICR fell by `-0.0011923392` and dominated the result.

Submission `1494307` therefore remains selected. The public result overrides the positive locked
H2 estimate below and invalidates broad phase/regime injection as a promotion family.

The candidate changes only group 3. It passed an untouched 2024-H2 test against the exact OOF
surface of the publicly confirmed meta-gate, improving group-3 score by `+0.0006886`, 1-NMAE by
`+0.0007404`, and FICR by `+0.0006368`. Because the official competition score macro-averages the
three groups and groups 1/2 are unchanged, the corresponding local total-score delta is about
`+0.0002295`.

## Literature-to-model transfer

This sprint deliberately moved beyond another calibration or blend-weight sweep.

1. The official KDD Cup 2022 rank-1 solution combined a global spatio-temporal branch with a
   spatio-partitioned, time-phased tree branch and data-driven ensembling:
   https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0518.pdf
2. The rank-2 solution combined LightGBM and GRU members over different multi-step schemes:
   https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1286.pdf
3. The GEFCom2014 wind winner used smoothed/bidirectional wind-energy inputs and a cross-sectional
   second layer across correlated farms:
   https://www.sciencedirect.com/science/article/pii/S0169207016000145
4. The GEFCom2012 wind winner used separate forecast-horizon models and weather-similarity
   clusters, with validation shaped like the competition forecast block:
   https://www.sciencedirect.com/science/article/pii/S0169207013000836

The transferable BARAM architecture therefore has three ExtraTrees branches:

- a global cross-sectional group-3 model;
- four six-hour lead-phase experts for forecast hours `+12` through `+35`;
- four KMeans NWP weather-regime experts.

All branches use normalized group-1/group-2 forecasts, group-specific/spatial NWP wind features,
and cycle-safe bidirectional smoothing, slope, and curvature features. Three independent seeds are
averaged. The selected component weights are `0.25 global / 0.50 lead phase / 0.25 weather regime`.

## Validation protocol

Models were trained on observed 2023 group-1/group-2/group-3 rows. The existing exact OOF
group-1/group-2 predictions were used as 2024 driver inputs. Hyperparameters and injection policy
were selected only on 2024-H1; the selected policy was then applied without change to H2 on top of
the current meta-gate exact OOF baseline.

| split | score delta | 1-NMAE delta | FICR delta |
|---|---:|---:|---:|
| H1 selection | +0.0031423 | +0.0011600 | +0.0051246 |
| H2 locked | +0.0006886 | +0.0007404 | +0.0006368 |

The injection uses `alpha=0.25`, an 8%-of-capacity member-disagreement limit, and no extra
group-1/group-2 agreement restriction. On the 2025 candidate it changes 7,514 of 8,760 rows,
with mean absolute movement `120.18 kWh` and 95th-percentile movement `351.33 kWh`.

## Robustness and risk

- Four of six locked months improved. October lost `-0.00108` and December lost `-0.00437`.
- Of the top 20 policies selected on H1, 19 improved H2 score and 17 improved score, NMAE, and
  FICR simultaneously. The gain is therefore not isolated to one exact mixture.
- A 2,000-sample day bootstrap was positive in `64.9%` of resamples. Its median was `+0.000656`,
  but the 5% quantile was `-0.00243`; public transfer remains uncertain.
- The 2025 correction changed 85.8% of rows. The public FICR loss confirms that this breadth was
  too aggressive despite positive local averages. Future automated promotion must penalize broad
  row coverage and must not promote a candidate whose bootstrap lower tail crosses zero without a
  much smaller public probe.

## Reproduction

```bash
python -m experiments.phase_regime_cross_group
python -B -m pytest -q -p no:cacheprovider
```

The rejected CSV is under `submissions/archive/`. The reusable code and compact diagnostic report
are retained; the prediction cache was removed.

# Local evaluation v2 — dependence and selection aware

## Fold evidence

| candidate | year | unclipped BSS-eq Δ | official BSS Δ | month + | worst month | min resampling p05 | min P(+) | leave-one-team min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v14 | 2022 | +3.0222 | +3.0222 | 57.1% | -6.1280 | -1.2489 | 87.7% | +0.0000 |
| v14 | 2023 | +21.5989 | +0.0000 | 100.0% | +6.4108 | +10.8854 | 99.9% | +0.0000 |
| v14 | 2024 | +4.2991 | +4.2991 | 75.0% | -24.6432 | +0.1966 | 95.8% | +0.0000 |
| v15 | 2022 | +1.9093 | +1.9093 | 57.1% | -13.0860 | -5.2758 | 67.5% | +0.0000 |
| v15 | 2023 | +43.6782 | +0.0000 | 100.0% | +12.9511 | +20.4654 | 99.9% | +0.0000 |
| v15 | 2024 | +5.8236 | +5.8236 | 75.0% | -68.0248 | -1.3078 | 91.0% | +0.0000 |

## Selection-aware gate

- 2024 registry reuse count (lower bound): **82**
- Final-family Reality Check p-value: **0.0180**
- Reality Check scope: v14/v15 final family only; earlier discarded trials are not corrected.
- Independent confirmation: **FAIL by construction** because 2024 is development-contaminated.

| candidate | statistical evidence | independent confirmation | promotion ready | failed checks |
|---|:---:|:---:|:---:|---|
| v14 | True | False | False | - |
| v15 | False | False | False | latest_all_resampling_p05_positive, latest_all_resampling_probability, post_break_resampling |

## Interpretation

- `unclipped BSS-eq Δ` preserves paired Brier information even when both historical official scores clip to zero.
- `official BSS Δ` applies the competition's zero floor and is shown separately.
- The minimum resampling figures combine pitcher, batter, crossed pitcher×batter, and 500/2,000/5,000-pitch circular block bootstraps.
- Bootstrap intervals describe sampling/dependence sensitivity, not uncertainty from trying many model recipes.
- A new Public result remains a deployment probe, not an independent scientific confirmation.
- A leave-one-team result of exactly zero is non-degradation: removing the only modified team/domain can make candidate and incumbent identical.

## Primary references

- DACON official evaluation: https://dacon.io/competitions/official/236743/overview/evaluation
- Murphy (1973), Brier reliability-resolution-uncertainty decomposition: https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2
- Tashman (2000), rolling-origin and multiple out-of-sample periods: https://doi.org/10.1016/S0169-2070(00)00065-0
- Politis & Romano (1994), stationary/block bootstrap for dependent observations: https://doi.org/10.1080/01621459.1994.10476870
- Owen (2007), pigeonhole bootstrap for crossed random effects: https://doi.org/10.1214/07-AOAS122
- Diebold & Mariano (1995), paired predictive-accuracy comparison under dependent errors: https://doi.org/10.1080/07350015.1995.10524599
- White (2000), Reality Check for data snooping: https://doi.org/10.1111/1468-0262.00152
- Hansen, Lunde & Nason (2011), Model Confidence Set: https://doi.org/10.3982/ECTA5771
- Cawley & Talbot (2010), model-selection overfitting and evaluation bias: https://www.jmlr.org/papers/v11/cawley10a.html

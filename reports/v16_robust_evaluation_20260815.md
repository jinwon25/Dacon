# v16 multi-season empirical-Bayes robust evaluation

- Selected recipe: `multi_pitcher_batter_hand_d0.5_a3200_w1`
- Correction: previous-season OOF residual by pitcher × batter hand, R_CORE fit/apply only.
- Historical weights: 0.5 per one-season lag; posterior denominator adds 3,200 pseudo-pitches.

| comparison | year | gain | month + | worst month | min p05 | min P(+) | leave-team min |
|---|---:|---:|---:|---:|---:|---:|---:|
| incremental_vs_v14 | 2023 | +10.2304 | 71.4% | -39.7136 | -2.7486 | 89.2% | +5.7759 |
| incremental_vs_v14 | 2024 | +7.3227 | 62.5% | -4.9037 | -3.0656 | 87.5% | +5.5791 |
| total_vs_v13 | 2023 | +31.8293 | 85.7% | -23.5597 | +14.3329 | 99.9% | +14.2920 |
| total_vs_v13 | 2024 | +11.6218 | 75.0% | -18.3936 | +0.5002 | 95.7% | +10.3952 |

- Final-shortlist (top five) Reality Check p-value: **0.0323**.
- Recipes screened before robust confirmation: **552**.
- Reality Check covers only the saved top-five multi-season shortlist; the larger search remains a source of selection bias.
- 2024 is development-contaminated, so this supports a deployment probe rather than independent confirmation.

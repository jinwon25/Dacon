# Score ladder

At base rate near 0.5, approximate score-equivalent improvements are:

| ΔBrier | Score gain |
|---:|---:|
| -0.00005 | +20 |
| -0.00010 | +40 |
| -0.00025 | +100 |
| -0.00050 | +200 |
| -0.00084 | +336 |

The 763.27→1100 gap therefore requires roughly `-0.00084` Brier, not a single calibration or blend-weight adjustment. The first promotion gate is fixed at 2024 Δ≤-0.00010, recency-weighted four-fold Δ≤-0.00010, worst fold≤+0.00005 and pitcher-season bootstrap negative probability≥0.95.

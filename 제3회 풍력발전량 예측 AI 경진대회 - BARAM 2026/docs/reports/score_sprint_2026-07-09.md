# Score Sprint - 2026-07-09

## Leaderboard State

Current public best:

- `blend_over115_scada_stack5.csv`
- Score: `0.6402652274`
- 1-NMAE: `0.8753153766`
- FICR: `0.4052150782`

`blend_over115_scada_stack10.csv` dropped to `0.6397287088`.

## Interpretation

The current plateau is no longer solved by global calibration or extrapolation. The useful new member is the SCADA proxy stack:

- `stack5` improved both 1-NMAE and FICR.
- `stack10` improved 1-NMAE but reduced FICR enough to lower total score.
- Therefore the useful injection region is around 3-6%, not 10%+.

This mirrors the mosquito competition lesson: once the main ensemble is saturated, the next gain comes from a decorrelated member injected manually at a small weight, not from chasing the best OOF member.

## Active Next Submissions

Recommended order:

1. `blend_over115_scada_stack4.csv`
2. `blend_over115_scada_stack6.csv`
3. `blend_over115_scada_g12_6_g3_3.csv`

Rationale:

- `stack4` probes just below the public-winning 5%.
- `stack6` probes just above it with low risk.
- `g12_6_g3_3` reflects local evidence that the SCADA stack is weaker for group 3.

## Methods Tried

Useful:

- LightGBM/CatBoost target-wise hybrid.
- Metric-optimized convex blend of GBDT candidates.
- Manual extrapolation from `cal125` toward `blend_v1`.
- SCADA proxy stack with small injection weight.

Low-priority:

- Analog/KNN ensemble: local 2024 score around `0.603`.
- Recent-year specialist: local 2024 score around `0.647`.
- Direct SCADA proxy: local 2024 score around `0.642-0.643`.

## Next Method Work

1. Cache OOF/test prediction matrices for every member.
2. Add SCADA stack as a formal member in the blend optimizer instead of manual CSV-level injection.
3. Search group-wise weights with constraints:
   - all-group SCADA stack weight: 0.03-0.07
   - group 3 stack weight capped below group 1/2
   - no extra extrapolation above `over115` unless public evidence changes
4. Add month/season diagnostics for public-like periods.

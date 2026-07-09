# Score Sprint - 2026-07-09

## Leaderboard State

Current public best:

- `blend_over115_scada_stack5.csv`
- Score: `0.6402652274`
- 1-NMAE: `0.8753153766`
- FICR: `0.4052150782`

`blend_over115_scada_stack10.csv` dropped to `0.6397287088`.
`blend_stack5_powercurve_sel5_g12_t06.csv` dropped to `0.6400608956`.
`blend_over115_scada_stack4.csv` dropped slightly to `0.6402123189`.

## Interpretation

The current plateau is no longer solved by global calibration or extrapolation. The useful new member is the SCADA proxy stack:

- `stack5` improved both 1-NMAE and FICR.
- `stack4` reduced both 1-NMAE and FICR versus stack5.
- `stack10` improved 1-NMAE but reduced FICR enough to lower total score.
- Therefore the global SCADA stack weight is narrow around 5%; the next useful search is group-wise, not another global sweep.

This mirrors the mosquito competition lesson: once the main ensemble is saturated, the next gain comes from a decorrelated member injected manually at a small weight, not from chasing the best OOF member.

## Active Next Submissions

Recommended order:

1. `blend_over115_scada_g12_5_g3_3.csv`
2. `blend_over115_scada_g12_6_g3_3.csv`
3. `blend_over115_scada_stack6.csv`
4. `blend_stack5_antipowercurve2_g12_t06.csv`

Rationale:

- `g12_5_g3_3` keeps G1/G2 exactly at the current best and only lowers G3 SCADA stack weight to 3%.
- `g12_6_g3_3` then tests whether G1/G2 can tolerate 6% while G3 stays lower.
- `stack6` is now lower priority because `stack4` confirmed that global weights around 5% are already near the peak.
- `powercurve_sel5_g12_t06` was tested as the first non-GBDT/SCADA-proxy injection candidate, but it reduced both 1-NMAE and FICR. Do not expand this route for now.
- `antipowercurve2_g12_t06` is a tiny reverse-direction probe based on that failed result. It changes only G1/G2 agreement rows and moves the mean prediction by less than 4 kWh.

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
- Pure power-curve residual standalone: local 2024 score around `0.612`, so do not submit it directly.
- Selective power-curve residual blend: public `0.6400608956`, below stack5 by `0.0002043318`.

## Next Method Work

1. Cache OOF/test prediction matrices for every member.
2. Add SCADA stack as a formal member in the blend optimizer instead of manual CSV-level injection.
3. Search group-wise weights with constraints:
   - all-group SCADA stack weight: keep 0.05 as anchor
   - group 3 stack weight capped below group 1/2
   - no extra extrapolation above `over115` unless public evidence changes
4. Add month/season diagnostics for public-like periods.
5. If SCADA stack 4/6 do not move the score, move to a new member family rather than expanding power-curve residual injection.

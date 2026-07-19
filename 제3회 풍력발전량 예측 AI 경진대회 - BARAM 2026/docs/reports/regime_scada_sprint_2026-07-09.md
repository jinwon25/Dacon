# Regime SCADA Sprint - 2026-07-09

## Why Direction Changed

The public plateau is FICR-limited. `blend_over115_scada_stack5.csv` improved the public score to
`0.6402652274`, but `stack10` improved `1-NMAE` while lowering FICR enough to lose score. This
means the SCADA proxy member is useful, but the global weight must be protected around settlement
thresholds.

The new direction is regime-gated SCADA injection:

- keep `blend_over115_scada_stack5.csv` as the base;
- add a small extra move toward `scada_proxy_stack_hist.csv`;
- only change rows where base/member disagreement is small enough to reduce FICR boundary risk;
- optionally restrict by power regime, direction, month, or hour.

This is consistent with wind-power forecasting practice: use NWP as the base signal, SCADA-derived
bias correction as a post-processing member, and regime-aware gating when atmospheric conditions or
error behavior differ.

## Added Tooling

`experiments/regime_member_blend.py`

Key controls:

- `--weights`: global or per-target extra alpha.
- `--max-disagreement`: absolute normalized base/member disagreement cap.
- `--min-base-ratio`, `--max-base-ratio`: power-regime gate.
- `--months`, `--hours`: seasonal/hour gates.
- `--direction`: `both`, `up`, or `down`.

Validation:

```bash
python -m py_compile experiments/regime_member_blend.py
```

## Generated Candidates

| file | intent | change profile |
|---|---|---|
| `blend_stack5_scada_extra2_agree4.csv` | safest extra SCADA probe | alpha 2%, disagreement <= 4%; absmean delta about 4.5-4.7 kWh |
| `blend_stack5_scada_extra2_agree6_mid.csv` | broader mid-power probe | alpha 2%, disagreement <= 6%, base ratio 10-75%; absmean delta about 6.5-8.1 kWh |
| `blend_stack5_scada_g12_extra3_agree5.csv` | group 1/2 focused probe | alpha 3% for G1/G2 only, disagreement <= 5%; G3 unchanged |
| `blend_stack5_scada_extra2_down_agree6.csv` | reverse-direction diagnostic | alpha 2%, only rows where SCADA is below base |

## Submission Order

1. `blend_stack5_scada_extra2_agree4.csv`
2. `blend_stack5_scada_extra2_agree6_mid.csv`
3. `blend_stack5_scada_g12_extra3_agree5.csv`
4. `blend_stack5_scada_extra2_down_agree6.csv`

If the first candidate beats `stack5`, continue this family with disagreement caps around `0.03-0.05`
and group-wise alpha. If it loses while `1-NMAE` rises, the next move should be explicit FICR-boundary
protection rather than more SCADA weight.

## Sources Used

- DACON evaluation page: score is `0.5 * (1-NMAE) + 0.5 * FICR`, evaluated only where actual
  generation is at least 10% of capacity.
  - https://dacon.io/competitions/official/236727/overview/evaluation
- DACON rules page: evaluation data can only be used for inference, so test-period SCADA leakage is
  still out of scope.
  - https://dacon.io/competitions/official/236727/overview/rules
- SCADA/NWP bias-correction work supports using historical SCADA to correct NWP-derived forecasts.
  - https://arxiv.org/abs/2402.13916
- GEFCom2014 wind winner supports zone/group-specific GBM modeling and post-processing rather than a
  single global model.
  - https://ideas.repec.org/a/eee/intfor/v32y2016i3p1061-1066.html
- Regime-aware wind forecasting literature supports gating forecasts by meteorological/power regimes.
  - https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2025.1686125/full

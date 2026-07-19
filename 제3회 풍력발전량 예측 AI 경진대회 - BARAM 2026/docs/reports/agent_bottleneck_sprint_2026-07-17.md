# Competition Scientist bottleneck sprint - 2026-07-17

## Decision

Keep submission `1494307`, `blend_best_crossg3_traj_meta25_p55.csv`, selected. No new
submission CSV was created and no DACON submission was executed.

The selected public score is `0.6416553726`; reaching `0.65` still requires
`+0.0083446274`. If groups 1 and 2 stay fixed, that is approximately a `+0.0250` group-3
score improvement. None of the locked experiments in this sprint approached that requirement
with acceptable stability.

## Agent-controlled experiment tree

All model runs used the approved rolling validation plan: Q2 selected iterations, calibration,
or policy; H2 remained locked. The service recorded hypothesis ancestry, inputs, manifests,
seed metrics, month metrics, day bootstrap, and steward action in SQLite.

| run | bounded hypothesis | locked group-3 score delta | result |
|---:|---|---:|---|
| 1 | group-specific IDW/hub CatBoost | +0.001671 | rejected: worse than its base-control sibling and seed-unstable |
| 2 | rolling CatBoost seed-consensus gate | +0.000194 | rejected: bootstrap probability 0.7405 and q05 -0.000354 |
| 3 | missing-target multi-output CatBoost | -0.002324 | rejected: Q2-to-H2 sign reversal and negative transfer |
| 4 | four direct lead-phase CatBoost experts | -0.016468 | rejected from Q2 onward; only one of four lead phases positive |
| 5 | XGBoost absolute-error diversity member | -0.014985 | rejected: weak and 0.992 correlated with CatBoost |

Run 1's monolithic base-control CatBoost improved H2 by `+0.002407`, but its individual seed
deltas were `+0.008750`, `-0.005205`, and `-0.002540`; December lost about `-0.0393`.
The mean-only bounded policy was therefore evaluated as run 2 rather than promoted directly.

## Research-stage failures

- A 24-hour Analog Ensemble residual had a `-0.000177` locked H2 delta and negative FICR.
- A no-unanimity CatBoost gate selected on Q2 improved H2 by `+0.000577`, but its day-bootstrap
  q05 was `-0.000376` and p95 movement exceeded policy.
- Selecting the CatBoost gate by Q2 bootstrap q05 produced a safer H2 `+0.000137`, with both
  components positive and q05 `-0.000100`; it still missed the service's minimum score/macro
  thresholds and had one negative seed.
- An 11-class direct settlement-action policy improved Q2 by `+0.001958` at 1.78% coverage but
  reversed to `-0.000237` on H2; all five seeds were negative.

The repeated failure signature is temporal operating-state drift, especially in December.
Post-hoc exclusion of December would leak the locked result and is not allowed. The earlier broad
phase/regime public submission already demonstrated that locally favorable seasonal selection can
destroy public FICR.

## Method sources and interpretation

- CatBoost natively supports missing-target multi-regression, which made run 3 a clean test of
  auxiliary-farm transfer without group-3 pseudo-labels:
  https://catboost.ai/docs/en/concepts/loss-functions-multiregression
- Multi-task wind forecasting for new farms motivates borrowing older farms' histories, but the
  local shared-tree implementation showed negative transfer:
  https://vbn.aau.dk/en/publications/probabilistic-wind-power-forecasting-for-newly-built-wind-farms-b/
- The GEFCom 2012 solution trained forecast-horizon-specific models; direct CatBoost replication
  failed here because BARAM's group-3 sample per expert was too small and FICR variance increased:
  https://www.sciencedirect.com/science/article/pii/S0169207013000836
- The KDD Cup 2022 88VIP solution combined heterogeneous GBDT/RNN models. XGBoost added neither
  sufficient accuracy nor useful diversity on the exact BARAM surface:
  https://arxiv.org/abs/2208.08952
- XGBoost's regularized tree formulation motivated the algorithm-diversity branch:
  https://arxiv.org/abs/1603.02754

## Artifact and submission stewardship

Only `artifacts_final/` remains. Closed multi-output, horizon, and XGBoost prediction caches were
removed after validation; compact reports, evaluations, manifests, and logs remain. Run 1 and run
2 caches are retained because they contain the only reproducible near-positive child evidence.

The active `submissions/` directory contains only:

- `blend_best_crossg3_traj_meta25_p55.csv`;
- `results.csv`.

## Selection semantics fix

The control plane formerly updated `local_best` before deterministic promotion, leaving rejected
run 1 selected. `local_best` now updates only for a `candidate` policy outcome. The stale selection
was deactivated with an audit event; `submission_candidate` was always empty, so no automatic
submission was possible.

## Next search boundary

The current exact surface is saturated for generic model replacement, horizon partitioning,
multi-farm shared trees, and direct residual action policies. Further automatic work should require
either a new exogenous operating-state signal allowed by the rules or a redesigned multi-period
validation asset. It should not spend submissions on another H2-tuned seasonal gate.

# Spatiotemporal multitask structural-model promotion

## Decision

The frozen two-seed graph multitask ensemble with a global `alpha=0.20` blend
passed the structural-model promotion contract. The guarded candidate is:

`submissions/blend_best_spatiotemporal_multitask20.csv`

Service family: `spatiotemporal_multitask_blend`.

The global policy was selected from Q1/Q2 only. H2 was opened once after the
policy was frozen. Seed-consensus and uncertainty/disagreement bounded gates
were evaluated as a small pre-declared comparison grid, but the retained final
test artifact contains the reproducible two-seed ensemble rather than two
separate seed predictions. Those non-deployable policies were excluded before
Q1/Q2 selection; H2 did not influence that decision.

## Locked evidence versus the finesweep lineage

| Period | Score delta | 1-NMAE delta | FICR delta | Worst month | Bootstrap positive | Bootstrap q05 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 development | +0.005800 | +0.004364 | +0.007235 | +0.002885 | 0.9795 | +0.001215 |
| Q2 development | +0.008667 | +0.004126 | +0.013208 | +0.002697 | 0.9975 | +0.003899 |
| H2 locked | +0.005061 | +0.002183 | +0.007939 | +0.001647 | 0.9805 | +0.000873 |

The H2 seasonal score deltas were +0.005682 (JJA), +0.007665 (SON), and
+0.001647 (DJF). All six H2 issue-centre months improved. Bootstrap statistics
use 2,000 season-stratified resamples of complete GFS
`data_available_kst_dtm` issue cycles.

Both independent validation seeds had positive score, 1-NMAE, and FICR deltas
in Q1, Q2, and H2. Individual-seed blocked bootstraps remain diagnostic because
the deployable object is the averaged ensemble; the complete monthly,
seasonal, and bootstrap contract is enforced on that ensemble.

## Candidate guard

- Base: `submissions/blend_best_crossg3_traj_meta_finesweep.csv`
- Seeds / final epochs: 17 / 12 and 29 / 13
- Rows: 8,760; schema and key order exactly match `sample_submission.csv`
- Group 1 and group 2: byte-value unchanged from the base
- Changed group-3 rows: 8,760 (structural global blend)
- Mean / p95 / maximum absolute movement: 191.59 / 534.12 / 1,250.77 kWh
- Candidate SHA-256: `f0da30ae9198c923219e734220a40ecbe97febf4774df579f0e02bfec360977d`
- `CandidateValidator`: valid, no schema/key/range/non-finite errors

The generic 25% changed-row cap applies to selective calibration patches. This
candidate instead uses the separately versioned structural-model family
override (`max_changed_ratio=1.0`, `max_p95_movement_ratio=0.04`); its observed
p95 movement is about 0.02543 of group-3 capacity.

The machine-readable audit is retained at
`artifacts_final/spatiotemporal_consensus/spatiotemporal_consensus_promotion_report.json`.

# Cross-group trajectory smoothing sprint - 2026-07-14

## Decision

Keep the publicly confirmed `submission20` selected. The next controlled probe is:

- archived file: `submissions/archive/blend_best_crossg3_traj5_consensus.csv`;
- DACON title: `Best crossgroup trajectory G3 5 cons2`;
- base: `blend_best_crossg3_25_agree8_delta6.csv`;
- groups 1/2: exactly unchanged;
- group 3: 5,750 materially changed rows;
- mean / 95th percentile / maximum absolute movement: `3.98 / 15.79 / 20.99 kWh`.

## Public result

Submission `1491284` scored `0.6414800049` (`1-NMAE 0.8754395685`, `FICR 0.4075204413`). Relative
to submission20, the deltas were only `+0.0000109493`, `+0.0000140999`, and `+0.0000077987`.
The direction was correct, but the magnitude confirms that this post-processing surface is
saturated. Submission23 is the new selected best; do not continue trajectory micro-tuning.

## Bottleneck diagnosis

The post-audit search confirmed that the remaining bottleneck is FICR stability rather than raw
MAE alone. Several methods reduced average error but changed which rows fell inside the 6% and 8%
settlement bands, causing score direction to reverse across time halves or validation proxies.

Rejected routes in this sprint:

1. Cross-group 24-hour trajectory features reduced neither member MAE nor FICR robustly. Adjacent
   and multi-offset features changed sign between H1/H2 and proxy families.
2. Cross-group quantile/FICR decisions improved conditional medians but sharply reduced H2 FICR.
3. Domain-aligned driver training reduced raw group-3 member MAE from about `2,610` to `2,027 kWh`,
   but its correction worsened NMAE/FICR stability when blended into the strong bases.
4. A two-member consensus gate, reciprocal group-1/group-2 transfer, and a learned settlement
   meta-gate all failed the worst-proxy H1/H2 checks.
5. The SCADA audit confirmed Vestas alignment at +50 minutes: raw SCADA-label MAE fell from about
   `211-225` to `118-124 kWh`. However, a direct NWP-to-SCADA LightGBM proxy became worse for group
   1 and effectively tied for group 2. The old +60-minute proxy's diversity appears to come partly
   from temporal smoothing, so the production alignment code was not changed retroactively.

## Selected method

The successful method extracts the useful smoothing effect directly from forecasts while keeping
issue-cycle boundaries explicit:

1. Within each 01:00-to-next-day-00:00 forecast cycle, compute a triangular value
   `(previous + 2 * current + next) / 4`; boundaries fall back to the current prediction.
2. Compute this delta for capacity-normalized group-1 and group-2 forecasts.
3. Allow a group-3 change only when the two driver deltas have the same sign.
4. Reject rows where the raw group-3 smoothing delta exceeds 2% of capacity.
5. Move group 3 only 5% toward the triangular value.

All inputs are test-period forecasts already present in the submission. No target, SCADA, future
observation, or cross-issue-cycle value is used.

## Validation

The 25% publicly confirmed cross-group correction was reconstructed on the three retained 2024
OOF proxy surfaces. H1 ends at `2024-07-01 00:00`; H2 starts at `2024-07-01 01:00`.

| proxy | full score delta | full 1-NMAE delta | full FICR delta | H2 score delta | months improved |
|---|---:|---:|---:|---:|---:|
| weighted | +0.000078 | +0.000021 | +0.000136 | +0.000159 | 6/12 |
| global | +0.000385 | +0.000011 | +0.000760 | +0.000686 | 10/12 |
| final pool | +0.000193 | +0.000020 | +0.000365 | +0.000118 | 8/12 |

Full-year score, 1-NMAE, and FICR improved on every proxy. Every H2 score delta was positive. H1
was positive for global/pool and numerically tied for weighted (`-0.00000013` score, with NMAE up
and FICR down by nearly equal amounts).

Five hundred random 40:60 timestamp splits were also evaluated. The probability of a positive
Private-60% score / 1-NMAE / FICR delta was:

| proxy | score positive | 1-NMAE positive | FICR positive |
|---|---:|---:|---:|
| weighted | 63.2% | 100.0% | 61.6% |
| global | 98.0% | 99.2% | 97.8% |
| final pool | 83.8% | 100.0% | 82.4% |

The weighted-proxy FICR uncertainty is why this remains a low-risk public probe rather than an
automatic replacement. Its movements are roughly two orders of magnitude smaller than the
rejected 45% cross-group expansion.

## Reproduction and checks

```bash
python -m experiments.cross_group_trajectory_smoothing
python -m pytest -q
```

The experiment fails closed unless full-year score/NMAE/FICR are positive on every proxy, every
H2 score is positive, and H1 scores are nonnegative within a `1e-6` numerical tolerance. The final
run passed all checks; the complete machine-readable report is
`artifacts_trajectory_smoothing/trajectory_smoothing_report.json`.

# Group 3 covariate-shift audit — 2026-07-18

## Decision

Reject this family as a submission candidate.  After repairing the validation
contract, the locked H2 gain is only `+0.00004656`, below the deterministic
`+0.00015` promotion floor, and the issue-cycle bootstrap positive fraction is
`0.6815`, below the required `0.80`.

## Defects repaired

1. The previous implementation clipped weights to `[0.25, 4.0]` and then
   divided by their mean.  That final division violated the clip contract:
   selection/H2 maximum weights were `7.72`/`5.95`.  The replacement solves for
   a scale *inside* the clipping operation, preserving both mean one and the
   exact bounds.  Corrected maximum weights are `4.0` in both periods.
2. `Q2_START=2024-04-01 00:00` split one LDAPS issue cycle between training and
   validation.  The boundary is now `2024-04-01 01:00`, the first target of the
   next complete issue cycle.  The training start is aligned in the same way.
3. Calendar-row month grouping split 24-hour NWP issue horizons at month
   boundaries.  Monthly diagnostics now assign every issue to the month of its
   median target time.
4. The day bootstrap was replaced by a season-stratified complete-issue-cycle
   bootstrap.  This preserves the dependence among forecast horizons from the
   same NWP run.

## Corrected locked evidence

| item | value |
|---|---:|
| Q2-selected policy | alpha `0.25`, disagreement `0.01–0.02`, uncertainty `0.008537` |
| locked score delta | `+0.00004656` |
| locked 1-NMAE delta | `+0.00003185` |
| locked FICR delta | `+0.00006127` |
| changed ratio | `0.08163` |
| positive issue-months | `4/6` |
| bootstrap positive fraction | `0.6815` |
| bootstrap q05 | `-0.00007591` |

August (`-0.0004374`) and September (`-0.0002553`) reverse sign.  The former
report's `+0.00010819` and `93.5%` bootstrap support therefore overstated the
evidence because the weight and dependency contracts were not satisfied.

## Bounded refinement

The development-only alpha neighbourhood was extended from `0.20` to `0.225`
and `0.25`.  Q2 selected `0.25` with all three issue-month blocks positive, but
the locked transfer remained tiny and unstable.  No H2-tuned policy was chosen,
no test prediction was fitted, and no file was created under `submissions/`.

Regression coverage is in `tests/test_group3_covariate_shift.py` for bounded
normalization, invalid ratios, issue-block month reporting, and rejection of a
tiny unstable locked gain.

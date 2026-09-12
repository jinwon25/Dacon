# P0-A season-state feature catalog

The audit supports a cumulative, pre-pitch sufficient-statistic interpretation for pitcher state. Candidate features are computed from the current row and train-origin snapshots only:

- pitcher/batter season exposure and season success count;
- career-before-season, prior-season and current-season posterior rates;
- current-versus-career/prior logit deltas and posterior uncertainty;
- newcomer, returner, long-gap, team-switch and role-change flags;
- current-season pitch-mix composition and career drift;
- prev1/3/5 game disagreement against season/career posterior.

Rate displays are rounded before integer reconstruction. If a denominator/rate pair is not within its feasible range, it is excluded from count features and retained only as a missingness diagnostic. Fixed smoothing constants are not treated as validated hyperparameters; later state models must choose discount/process variance on earlier inner folds.

# Sprint 0.660 Status

Updated: 2026-08-05 KST

Submission update: 2026-08-05 KST

All values below are local time-ordered OOF results unless explicitly labeled
as a logged leaderboard result.

## Current outcome

| Policy | Score | 1-NMAE | Official FICR | Decision |
|---|---:|---:|---:|---|
| Strict corrected raw baseline | 0.623751 | 0.876778 | 0.370725 | reference |
| Current Public incumbent, matched 2024 OOF | **0.649163** | **0.882753** | **0.415573** | selected production baseline |
| P2 safe direct policy | 0.640422 | 0.874759 | 0.406085 | retained |
| safe_cv_best with group-3 SCADA path | 0.642489 | 0.877149 | 0.407828 | Public-rejected replacement |
| ficr_bayes group-2 gate | 0.642975 | 0.876178 | 0.409772 | withheld derivative |
| diverse_ensemble | **0.643821** | **0.878346** | **0.409297** | withheld derivative |
| Group-1 recency overlay diagnostic | 0.649699 | 0.882621 | 0.416778 | rejected; unstable locked bootstrap |
| G3 frozen nearest-grid overlay diagnostic | 0.649223 | 0.882728 | 0.415717 | rejected; H2 and bootstrap gates failed |

Submission 1511408 tested safe_cv_best and scored 0.6345216694 / 0.8679554758 /
0.4010878631. Relative to incumbent submission 1508386, this is -0.0129487705
Score, -0.0081570997 1-NMAE, and -0.0177404412 FICR. The candidate is rejected.
The incumbent remains 0.6474704399 / 0.8761125755 / 0.4188283043.

The matched incumbent OOF is reconstructed from the frozen primary, residual,
and group-3 lineage at the exact production weights. It scores 0.6491632057 /
0.8827530123 / 0.4155733992 on 2024 and is the correct local production
baseline; the lower Sprint-066 raw/direct rows remain useful component
ablations rather than replacement candidates.

## Completed decisions

- Stage 0 passes: official actual-weighted FICR, inclusive 6%/8% boundaries,
  eligibility at 10%, NaN handling, and no-eligible-row behavior are tested.
- The official metric changes the preferred group-1 baseline variant from
  all-row training to eligible-only training.
- All retained LDAPS/GFS rows are issued at 13:00 KST and satisfy the
  prior-day 14:00 KST cutoff. Actual lead is 12-35 hours.
- Broad forecast-structure, trajectory, site-aware and physical feature blocks
  were rejected because their locked macro deltas or bootstrap lower bounds
  were negative.
- Capacity-normalized group-3 pooling was rejected: it improved NMAE but
  reduced FICR and was not stable by issue day.
- Training-only SCADA was retained only for group 3 at a frozen 27.5% weight.
  It does not require inference-time SCADA.
- The direct metric-aware action improves strict raw OOF by +0.005017 Score and
  +0.011064 FICR, with -0.001030 1-NMAE. On top of the stronger safe composite,
  however, the group-2 gate adds only +0.000487 Score and its bootstrap
  P(delta > 0) is 0.687.
- The unit-correct frozen spatial-temporal member adds +0.001333 Score to the
  safe composite. Bootstrap P(delta > 0) is 0.993, but locked-2024 H2 group-3
  FICR is -0.000876, so it remains the higher-risk candidate.
- A bounded group-1 recency factor was selected at alpha 0.25 on 2024 H1. It
  remained positive on H2 (+0.001189 group Score, +0.002760 FICR), but its H2
  day-block bootstrap had q05 -0.004537 and P(delta > 0) 0.6315. It was
  rejected without opening test weather or writing a submission candidate.
- The G3 displacement family was preregistered at six issue-local transforms
  and three bounded weights. Two 2023 forward folds froze nearest-grid x 0.50
  (Q3 +0.000777 Score, Q4 +0.002846). The one-shot 2024 evaluation improved
  G3 by only +0.000179 Score and +0.000432 FICR while losing -0.000075
  1-NMAE. H2 was -0.000618 Score and bootstrap P(delta > 0) was 0.5825
  (q05 -0.001495). It was rejected and no candidate CSV was created.

## Candidate files

| Candidate | Path | Rows | Validation |
|---|---|---:|---|
| active incumbent | artifacts_final/candidates/public_positive_g1w1375_g2w1825_g3frozen_20260802.csv | 8,760 | passed; selected |
| safe | submissions/sprint066/safe_cv_best.csv | 8,760 | passed |
| Bayes/FICR | submissions/sprint066/ficr_bayes.csv | 8,760 | passed |
| diverse | submissions/sprint066/diverse_ensemble.csv | 8,760 | passed |

Every file matches the sample key order and columns, contains no NaN/inf, is
inside group capacity bounds, and reloads as UTF-8.

The canonical diverse file was regenerated from serialized seed-17 and
seed-29 models without supplying a precomputed member CSV. Safe and Bayes were
identical to the prior run; the newly refit diverse group-3 prediction has
0.999806 correlation with the earlier full-refit candidate. No score is
inferred from this test-period difference.

After submission 1511408, ficr_bayes and diverse_ensemble are valid files but
are not recommended for submission: their OOF increments over safe are too
small to compensate for the observed full-policy transfer failure. Prediction
comparison against the incumbent shows mean shifts of +1,337.1, +91.6, and
+431.3 kWh for groups 1-3. Group 1 differs by more than 6% of capacity on
52.7% of test rows, making the prior-OOF group-1 affine the first component to
audit. Public aggregate metrics cannot prove group attribution.

The follow-up nested audit does not justify manually removing that affine:
2024 H1 alone selects scale 1.066 and offset +700, and on 2024 H2 it improves
Score from 0.638598 raw to 0.663399. The older 1.07/+900 policy scores 0.658497
on the same H2. Therefore the Public failure is treated as a 2025 transfer or
full-policy replacement failure, not as permission to tune the offset on the
leaderboard.

## Closed and next work

No analog distribution, additional trajectory grid, large hyperparameter
search or new deep model was launched after the predeclared stop criteria were
met. The recency follow-up and the later frozen G3 displacement family both
failed their locked bootstrap gates. The active submission remains 1508386.
None of the three Sprint-066 broad-replacement files is recommended after
1511408, and the displacement experiment produced no fourth file. The next
work is a structural G3 label/NWP/site/capacity alignment audit.

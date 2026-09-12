# Incumbent-matched residual decision follow-up

Date: 2026-07-29

## Verdict

The evaluation formula is not the remaining bottleneck. Two new,
incumbent-matched decision models were evaluated under a fail-closed
Q1-to-Q2 selection and H1-to-H2 confirmation contract. Neither qualifies for
promotion, and neither writes a submission.

The strongest new result is the non-crossing residual distribution model. Its
locked Q2 policy improves the development score by `+0.0020063333`, but the
same policy produces only `+0.0000023380` on H2 and reverses FiCR. The causal
full-year local gain is `+0.0004267741`, projecting the public score from
`0.6461250914` to only `0.6465518655`. The remaining gap to `0.65` is
`0.0034481345`.

## Evaluation and decision-logic corrections

Two edge cases were corrected before the formal runs.

1. When several candidate actions have numerically identical expected utility,
   the decision rule now chooses the action closest to the incumbent. The old
   ascending candidate order could select the most negative action on an FiCR
   utility plateau even though it had no modeled benefit.
2. A short calendar slice can contain no official evaluation row for one farm.
   Monthly diagnostics now evaluate the remaining valid farms with the same
   equal-group macro convention as the official scorer. When all three farms
   have valid rows, the helper delegates to and exactly matches
   `evaluate_competition`.

The official full-period metric remains unchanged: actual-only 10% eligibility,
capacity-normalized group MAE, actual-generation-weighted FiCR, inclusive
6%/8% cliffs, equal group macro averaging, and the final 50:50 component
average.

## Validation contract

- Development: fit on 2024 Q1, with January-February inner training and March
  early stopping; select one coverage/weight policy on Q2.
- Confirmation: refit on H1, with January-May inner training and June early
  stopping; apply the locked Q2 policy to H2.
- Causal surface: Q1 incumbent unchanged, Q2 development prediction, H2 locked
  confirmation prediction.
- Promotion components: score, 1-NMAE, and FiCR must all be positive.
- Stability: every interval-month score must be non-negative.
- Subset safety: all three component q05 values must be non-negative on both
  complementary 40%/60% IID and month-stratified samples.
- H2 additionally requires issue-cycle bootstrap q05 non-negativity.
- The public score was not used for model or policy selection.
- H2 has already been exposed to other research families, so these runs remain
  research evidence even if a gate had passed.

## Experiment A: non-crossing residual distribution

The model estimates `(truth - incumbent) / capacity` while conditioning on the
exact incumbent and pooled three-farm weather features. It emits structurally
non-crossing residual quantiles and evaluates bounded actions within 4% of
capacity against the official settlement utility.

The selected Q2 policy uses coverage `0.05` and weight `0.30`.

| period | score delta | 1-NMAE delta | FiCR delta |
|---|---:|---:|---:|
| Q2 selection | +0.0020063333 | +0.0001874797 | +0.0038251869 |
| H2 confirmation | +0.0000023380 | +0.0000210030 | -0.0000163270 |
| causal full year | +0.0004267741 | +0.0000574071 | +0.0007961411 |

Q2 passed both 500-repeat complementary subset screens. For example, the IID
public q05 values were `+0.0003176333` score, `+0.0000705454` 1-NMAE, and
`+0.0004824331` FiCR.

The transfer failed after July. H2 monthly score deltas were:

| interval month | score delta |
|---:|---:|
| July | +0.0016027142 |
| August | -0.0007179916 |
| September | -0.0002403247 |
| October | -0.0005481075 |
| November | -0.0006510503 |
| December | -0.0006481807 |

The H2 issue-cycle bootstrap q05 was `-0.0008100697` for score and
`-0.0015743990` for FiCR. The H2, full-year subset, month, and issue gates
therefore fail.

## Experiment B: direct official-utility value model

The second route removes the quantile-to-action approximation. For each
incumbent-relative action from -4% to +4% capacity, it calculates the exact
row-level official-utility gain and trains a pooled LightGBM value model with
the action ratio as an input. The model then ranks actions by predicted
incremental utility, including the unchanged incumbent action.

No Q2 policy passed selection. The best raw Q2 score policy used coverage
`0.20` and weight `0.30`:

- score `+0.0009458030`
- 1-NMAE `+0.0000087343`
- FiCR `+0.0018828717`
- April score `-0.0008259559`

The negative April transfer makes the policy ineligible before subset stress
and keeps H2 closed.

## Interpretation

The two models fail at different layers but point to the same bottleneck.
The non-crossing model can identify useful Q2 actions, while the direct value
model avoids distributional approximation entirely; neither learns an
incumbent residual ranking that is stable across the full calendar year.
Post-hoc action tuning, coverage expansion, or public-score weight search would
therefore be another seasonal overfit rather than a defensible route to 0.65.

The next credible route is to reconstruct incumbent-like year-forward OOF
predictions for earlier labeled years. That would make it possible to train on
a full seasonal cycle and require two independent year-forward transfers before
touching 2025. Without that additional baseline-matched history, another
high-dimensional residual gate is unlikely to close the remaining
`+0.0034481345` gap.

## Artifacts

- Formal non-crossing report:
  `artifacts_final/diagnostics/incumbent_residual_noncrossing_20260729.json`
- Formal non-crossing lineage:
  `artifacts_final/lineage/incumbent_residual_noncrossing_20260729.npz`
- Formal direct-utility report:
  `artifacts_final/diagnostics/incumbent_direct_utility_20260729.json`
- Formal direct-utility lineage:
  `artifacts_final/lineage/incumbent_direct_utility_20260729.npz`
- Implementation:
  `experiments/incumbent_residual_noncrossing.py`
- Tests:
  `tests/test_incumbent_residual_noncrossing.py`

## Method references

- [Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and
  Estimation](https://doi.org/10.1198/016214506000001437)
- [Wen et al., continuous and distribution-free probabilistic wind-power
  forecasting](https://arxiv.org/abs/2206.02433)
- [Donti, Amos, and Kolter, task-based end-to-end model learning in stochastic
  optimization](https://proceedings.neurips.cc/paper/2017/hash/3fc2c60b5782f641f76bcefc39fb2392-Abstract.html)
- [Mandi et al., decision-focused learning through learning to
  rank](https://proceedings.mlr.press/v162/mandi22a.html)

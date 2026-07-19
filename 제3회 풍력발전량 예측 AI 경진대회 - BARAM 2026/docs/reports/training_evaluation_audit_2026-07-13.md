# Training and evaluation audit — 2026-07-13

## Conclusion

The publicly confirmed best submission remains valid, but the audit found and corrected four
issues in the new spatial-temporal research pipeline. None of them contaminated submission20.
The unsubmitted spatial-temporal CSV was rejected and removed after corrected validation.

No evaluation-period target, SCADA, or post-forecast observation enters training. The official
test NWP files are now physically isolated from validation/training and opened only by inference.

## Official metric parity

The user-provided `평가_산식 코드.ipynb` was treated as the executable source of truth.
`src/metrics.py` was compared with that notebook over 100 randomized datasets including exact
10%, 6%, and 8% boundary cases. The maximum absolute differences for total score, 1-NMAE, and
FICR were all exactly `0.0`.

Confirmed aggregation order:

1. Keep rows where actual generation is at least 10% of that group's capacity.
2. Compute unweighted NMAE within each group.
3. For FICR, award 4 units at error `<= 6%`, 3 units at error `<= 8%`, otherwise 0; weight the
   settlement by actual generation within the group.
4. Macro-average the three group NMAEs and the three group FICRs.
5. Score = `0.5 * (1 - mean group NMAE) + 0.5 * mean group FICR`.

One defensive mismatch was fixed: the previous local evaluator could silently remove an eligible
row when its prediction was NaN. The official code would produce an invalid result. The local
evaluator now fails immediately on non-finite eligible predictions.

The hidden Public/Private masks cannot be reconstructed locally. Public uses a presampled 40% of
evaluation rows and Private uses the remaining 60%, so full-year local validation is a model
selection proxy rather than an estimate of either leaderboard score.

## Data and leakage audit

| check | result |
|---|---|
| train forecast timestamps | 26,304 unique, continuous hourly rows |
| test forecast timestamps | 8,760 unique, continuous hourly rows |
| train/test forecast cycles | 1,096 / 365 complete 24-hour cycles |
| forecast lead times | exactly 12–35 hours |
| issue times per 24-hour cycle | exactly one for LDAPS and GFS, train and test |
| LDAPS/GFS nodes per timestamp | exactly 16 / 9 |
| target/NWP timestamp alignment | exact; no duplicate or unmatched timestamps |
| target or SCADA columns in neural inputs | none |
| train/test feature columns | identical, 41 LDAPS and 49 GFS engineered channels |
| train/test grid coordinates | exactly identical |
| all-missing NWP channels | none |

The non-causal 24-hour temporal convolution is legal for this dataset: all 24 forecasts in a
cycle were published together before their target hours. It uses future forecast horizons from
the same issue, not future observations or targets.

Clean issue-cycle validation boundaries are now:

- fit: 729 cycles ending before `2024-01-01 00:00`;
- H1 selection: 182 cycles from `2024-01-01 01:00` through `2024-07-01 00:00`;
- H2 confirmation: 184 cycles from `2024-07-01 01:00` through `2025-01-01 00:00`.

There is no timestamp or issue-cycle overlap. A one-hour H1/H2 diagnostic boundary error that had
placed `2024-07-01 00:00` in H2 was corrected.

## Corrected model logic

Group 3 has 8,766 missing training targets, mainly because 2022 group-3 labels are unavailable;
groups 1 and 2 have only 104 and 103 missing values. The first neural loss pooled all eligible
rows, unintentionally giving group 3 much less influence than the official group-macro metric.
The loss now computes NMAE and smooth FICR surrogates separately per group and macro-averages only
the available groups in each batch.

The previous combined train/test tensor cache did not numerically leak test information into
normalization or fitting, but it unnecessarily read evaluation NWP during the training workflow.
It was replaced by physically separate train-only and test-only caches. The separated arrays were
verified bit-for-bit equal to their old counterparts before the old combined cache was removed.

Corrected two-seed validation selected epochs 12 and 13. Its H2 metric was `0.6423337`, with
`1-NMAE 0.8825114` and `FICR 0.4021560`. Although group-3 NMAE improved, the corrected hybrid did
not improve score, NMAE, and FICR simultaneously across weighted/global/final-pool proxies using
H1-only selection. Therefore it is not a submission candidate.

## Public/Private sampling robustness of submission20

The confirmed 25% cross-group correction was additionally tested with 500 random 40:60 timestamp
splits of the 2024 proxy period. The table reports the probability that the group-3 delta is
positive on the simulated Private 60% complement.

| OOF proxy | score positive | 1-NMAE positive | FICR positive | mean Private score delta |
|---|---:|---:|---:|---:|
| weighted | 96.4% | 100.0% | 90.4% | +0.000958 |
| global | 86.8% | 100.0% | 64.8% | +0.000641 |
| final pool | 84.8% | 100.0% | 77.2% | +0.000526 |

This supports keeping submission20, but it also shows why tiny Public gains should not drive
further tuning: the FICR sign remains sample-sensitive, especially on the global proxy.

## Verification and retained outputs

- Automated tests: `13 passed`.
- Retained historical submission, now archived at `submissions/archive/blend_best_crossg3_25_agree8_delta6.csv`.
- Retained neural research artifacts: separate train/test caches, validation predictions,
  corrected validation report, and hybrid validation report.
- Removed: smoother micro-tune CSV, invalid single-seed spatial CSV, its test predictions/member,
  obsolete combined tensor cache, and run logs.

Before a possible top-30 deliverable, the selected full ensemble must be frozen into explicitly
separate training and inference entry points with a reproducible environment. This packaging
requirement does not affect current leaderboard CSV validity.

## Official references

- Evaluation: https://dacon.io/competitions/official/236727/overview/evaluation
- Rules: https://dacon.io/competitions/official/236727/overview/rules

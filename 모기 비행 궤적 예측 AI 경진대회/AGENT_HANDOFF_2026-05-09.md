# Agent Handoff - 2026-05-09

## Project

- Competition: DACON mosquito flight trajectory prediction.
- Metric: R-Hit@1cm. Prediction is counted as hit when 3D Euclidean distance is `<= 0.01m`.
- Current working directory:
  `E:\학업\교외 활동\대외 활동\공모전\데이콘\모기 비행 궤적 예측 AI 경진대회`
- Main script:
  `src/pipeline.py`

## Current Public LB Results

Submitted on 2026-05-09:

| File | Public LB |
|---|---:|
| `outputs/selector_full/submission_selector_ensemble_soft.csv` | 0.6688 |
| `outputs/selector_full/submission_attn_gru_selector_soft.csv` | 0.6688 |
| `outputs/selector_full/submission_selector_ensemble_argmax.csv` | 0.6674 |
| `outputs/selector_full/submission_selector_ensemble_projected.csv` | 0.6686 |
| `outputs/boundary_1fold_inline_resmlp/submission_boundary_tiny_gate.csv` | 0.6834 |

Conclusion:

- Selector variants are not the source of the improvement.
- Boundary tiny correction is the productive direction.
- Do not submit `projected` again unless its generation logic changes materially.

## Best Candidate Prepared For Next Submission

Submit this first on the next available day:

```text
outputs/boundary_best_cap0p006_apply0p75_seed20260608/submission_boundary_tiny_gate.csv
```

Why:

- It is the same boundary correction family that produced `0.6834`, but with better fold0 OOF.
- Submitted boundary gate:
  - config: `cap=0.006`, `apply_scale=1.0`, `seed=20260606`
  - fold0 gate: `0.6722772277`
  - Public LB: `0.6834`
- New best candidate:
  - config: `cap=0.006`, `apply_scale=0.75`, `seed=20260608`
  - fold0 soft: `0.6737623762`
  - fold0 gate: `0.6762376238`
  - fold0 argmax: `0.6717821782`
- Difference from submitted boundary gate:
  - mean distance difference: about `0.6975mm`
  - changed rows: `10000 / 10000`
  - max distance difference: about `5.0577mm`

Secondary candidates in the same folder:

```text
outputs/boundary_best_cap0p006_apply0p75_seed20260608/submission_boundary_tiny_soft.csv
outputs/boundary_best_cap0p006_apply0p75_seed20260608/submission_boundary_tiny_argmax.csv
```

Submit `gate` first. Only submit `soft` after seeing whether `gate` improves.

## Important Generated Files

Selector outputs:

```text
outputs/selector_full/tcn_gru_selector_report.json
outputs/selector_full/oof_selector_scores.npz
outputs/selector_full/test_selector_scores.npz
outputs/selector_full/submission_selector_ensemble_soft.csv
outputs/selector_full/submission_selector_ensemble_projected.csv
outputs/selector_full/submission_selector_ensemble_gate.csv
outputs/selector_full/submission_selector_ensemble_argmax.csv
```

Boundary outputs:

```text
outputs/boundary_1fold_inline_resmlp/submission_boundary_tiny_gate.csv
outputs/boundary_1fold_inline_resmlp/boundary_tiny_correction_report.json
outputs/boundary_best_cap0p006_apply0p75_seed20260608/submission_boundary_tiny_gate.csv
outputs/boundary_best_cap0p006_apply0p75_seed20260608/boundary_tiny_correction_report.json
```

Sweep summaries:

```text
outputs/boundary_sweep_fold0_summary.json
outputs/boundary_seed_sweep_fold0_summary.json
```

## Code Changes Made

File changed:

```text
src/pipeline.py
```

Main changes:

- Added reusable selector post-processing helpers:
  - `physics_project_scores`
  - `argmax_soft_gate_select`
- Full selector training now saves:
  - `submission_selector_ensemble_projected.csv`
  - `submission_selector_ensemble_gate.csv`
  - `ens_prior` into `test_selector_scores.npz`
- Added CLI:
  - `--write-selector-variants`
  - `--run-boundary-only`
  - `--selector-out`
- Boundary test generation now writes:
  - `submission_boundary_tiny_soft.csv`
  - `submission_boundary_tiny_gate.csv`
  - `submission_boundary_tiny_argmax.csv`
- Boundary validation `.npz` now also stores gate predictions.

Validation:

```text
python -m py_compile .\src\pipeline.py
```

passed after edits.

## Useful Commands

Regenerate selector variants from existing selector output:

```powershell
python .\src\pipeline.py --write-selector-variants --selector-out .\outputs\selector_full
```

Run default 1-fold boundary from existing selector output:

```powershell
python .\src\pipeline.py --run-boundary-only --selector-out .\outputs\selector_full
```

Regenerate the current best boundary candidate:

```powershell
@'
import src.pipeline as p
selector = p.WORK_DIR / 'selector_full'
out = p.WORK_DIR / 'boundary_best_cap0p006_apply0p75_seed20260608'
p.call_main(p.BOUNDARY_MAIN, [
    '--root', p.DATA_ROOT,
    '--out-dir', out,
    '--fold', 0, '--folds', 5,
    '--score-bank', selector / 'oof_selector_scores.npz',
    '--make-test', '--test-score-bank', selector / 'test_selector_scores.npz',
    '--epochs', 1, '--fine-epochs', 1, '--min-epochs', 1, '--patience', 1,
    '--hidden', 64, '--batch', 8192,
    '--lr', 0.001, '--fine-lr-scale', 0.18,
    '--cap', 0.006, '--apply-scale', 0.75,
    '--device', 'cpu', '--seed', 20260608, '--save-val-pred',
])
'@ | python -
```

## Boundary Sweep Results

Config sweep on fold0, seed `20260606`:

| cap | apply_scale | soft | gate | argmax |
|---:|---:|---:|---:|---:|
| 0.006 | 0.75 | 0.672277 | 0.674257 | 0.672277 |
| 0.004 | 1.00 | 0.669307 | 0.673267 | 0.670297 |
| 0.006 | 1.00 | 0.669802 | 0.672277 | 0.667327 |
| 0.004 | 1.20 | 0.670297 | 0.671782 | 0.670792 |
| 0.004 | 0.75 | 0.668812 | 0.671782 | 0.668812 |
| 0.008 | 0.75 | 0.668317 | 0.671287 | 0.669802 |
| 0.006 | 1.20 | 0.671782 | 0.670792 | 0.666832 |
| 0.008 | 1.00 | 0.669307 | 0.670792 | 0.666337 |
| 0.008 | 1.20 | 0.665347 | 0.667822 | 0.664851 |

Seed sweep for `cap=0.006`, `apply_scale=0.75`:

| seed | soft | gate | argmax |
|---:|---:|---:|---:|
| 20260608 | 0.673762 | 0.676238 | 0.671782 |
| 20260609 | 0.670792 | 0.673762 | 0.673267 |
| 20260607 | 0.670297 | 0.673762 | 0.670792 |
| 20260610 | 0.671287 | 0.671782 | 0.668812 |

## Recommended Next Work

1. Submit:

   ```text
   outputs/boundary_best_cap0p006_apply0p75_seed20260608/submission_boundary_tiny_gate.csv
   ```

2. If it improves over `0.6834`, continue boundary sweep:

   - Seeds around `20260608`.
   - `cap=0.005, 0.006, 0.007`.
   - `apply_scale=0.65, 0.70, 0.75, 0.80, 0.85`.
   - Keep `epochs=1`, `fine_epochs=1` first. More epochs degraded in one tested run.

3. Start building more robust OOF:

   - Current boundary validation is only fold0.
   - Add a script to run boundary correction on folds 0-4 and aggregate OOF.
   - Use that for choosing `cap`, `apply_scale`, `seed`, `gate` threshold.

4. Larger model direction:

   - Candidate bank expansion is the next structural step.
   - Current candidate oracle is around `0.7188`; top LB is around 0.69, so there is room but selector/correction must stay stable.
   - Expand candidates conservatively around Frenet/latency candidates using local-frame offsets, then retrain selector.

## Notes

- Folder is not a git repository, so `git diff` does not work here.
- Python version seen: `Python 3.11.2`.
- Torch seen: `2.7.1+cpu`.
- CUDA unavailable, so long selector training runs on CPU.

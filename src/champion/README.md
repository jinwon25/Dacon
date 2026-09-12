# Champion lineage

Modules reachable from a version that was actually submitted. Everything here
imports only `src/core/` and other `src/champion/` modules, never
`src/archive/`, so a version can be read and run without loading unrelated
generations.

Concluded one-off experiments live in `src/archive/`.

## Deployed versions

| Version | Public | Role |
|---|---:|---|
| v148 | **1170.3014697177** | current champion, conservative v142→v138 bridge |
| v142 | 1169.6277932822 | independent H1 + sign-stable C3 |
| v124 | 1164.2949203402 | multi-axis public quadratic stack |
| v104 | 1162.6302840289 | source-stability majority-two mask |
| v84 | 1161.2020600422 | fixed v56 shared FM, F route |
| v26 | 1157.9736407889 | frozen signal weight 15% |
| v25 | 1155.8293405409 | post-break R_ANCHOR direct probability |
| v22 | 1153.0436023798 | low-variance domain calibration + row-local ASOF |

## Running the champion

`submit_v148.zip` is the official deliverable and is already standalone: the
evaluation server unpacks it and runs its own `script.py`. Nothing in this
repository is needed at evaluation time.

To rebuild the flattened package from the original archive:

```bash
python -m src.champion.v148_flat_build_package \
  --original-zip artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip \
  --runtime-script src/champion/v148_flat_runtime_script.py \
  --output-dir artifacts/v148_flat_20260823_01
```

`v148_flat_runtime_script.py` is the standalone runtime that ships inside the
flattened package. It has a single `main()` and static imports, replacing the
six-level chain of dynamically loaded parent scripts in the original build.
Its predictions match the original bit for bit within float noise; see
`research/reports/v148_flat_refactor_20260823.md`.

To verify parity yourself:

```bash
python -m src.champion.v148_flat_parity_audit \
  --original-zip artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip \
  --flat-zip artifacts/v148_flat_20260823_01/submit_v148_flat.zip \
  --train-csv data/train.csv --rows 100000
```

## Rebuilding earlier generations

Each build script takes its inputs explicitly on the command line, so no
script reaches back into a previous version's working directory.

```bash
# v142 — independent H1 and sign-stable C3
python -m src.champion.v142_build_submission_package \
  --parent-zip <v124 zip> --h1-zip <H1 zip> \
  --runtime-script src/champion/v142_runtime_script.py \
  --train-csv data/train.csv --oof-path <oof npz> \
  --data-dir data --config research/configs/v142_v141_submission_package.json \
  --output-dir artifacts/<run>

# v124 — public quadratic stack
python -m src.champion.v124_public_quadratic_stack \
  --parent-zip <v104 zip> --data-dir data \
  --config research/configs/v124_public_quadratic_stack.json \
  --output-dir artifacts/<run>

# v104 — source-stability mask
python -m src.champion.v104_source_stability_mask \
  --train-csv data/train.csv --contract-dir <contract dir> \
  --v103-dir <v103 run dir> --config research/configs/<v104 config> \
  --output-dir artifacts/<run>
```

The `<...>` inputs are frozen artifacts, not repository files. They are kept
out of Git and shared through the approved private team channel with their
SHA-256, as described in `docs/PROJECT_STATUS.md`.

## Rules

- Never overwrite a champion artifact in place. Build into a new
  `artifacts/<name>_<date>_NN/` directory.
- A new candidate must pass `research/configs/evaluation_v3.json` before it is
  packaged, and the champion pointer only moves after a Public result
  confirms it.
- Do not add an `src/archive/` import here. If a champion module needs
  something from an archived experiment, that symbol belongs in `src/core/`.

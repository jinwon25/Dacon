# Modeling agent

Implement exactly one approved hypothesis in a reproducible experiment module.

- Accept a registered `RunSpec`; do not invent output paths.
- Write intermediate files only under `artifacts_final/agent_service/runs/{run_id}/`.
- Write a promoted CSV only to the declared path under `submissions/`.
- Preserve exact OOF lineage and never use test labels or test-period SCADA.
- Fix seeds, log the command, and emit both a detailed report and standardized `Evaluation` JSON.
- Do not decide whether the candidate should be submitted.

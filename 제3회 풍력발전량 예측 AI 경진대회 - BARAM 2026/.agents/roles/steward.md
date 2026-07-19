# Artifact steward

Keep the active workspace small and preserve provenance.

- Leave only the selected public best and genuinely unsubmitted candidates in `submissions/`.
- Move public failures to `submissions/archive/`.
- Delete rejected prediction/model caches after exact targets are verified; retain compact reports.
- Keep reusable feature caches and exact lineage under `artifacts_final/`.
- Never overwrite an archive target; suffix it with the run ID when names collide.

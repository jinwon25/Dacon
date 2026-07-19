# Artifact consolidation - 2026-07-17

All retained generated files are consolidated under `artifacts_final/`.

## Retained structure

```text
artifacts_final/
|-- feature_cache/       # expensive reusable train/test NWP features
|-- lineage_inputs/      # minimal reports/caches needed to rebuild exact OOF
|-- lineage/             # exact group-1/group-2/group-3 OOF cache and report
|-- meta_gate/           # selected meta-gate cache and public-result report
|-- diagnostics/         # compact bottleneck, trajectory, and rejected-family JSON reports
`-- agent_service/       # SQLite control-plane state; run logs appear only when executed
```

The feature cache accounts for roughly 146MB of the retained footprint. It is kept because
rebuilding LDAPS/GFS pivot features is the most expensive preprocessing step.
The remaining non-feature artifacts total about 3.5MB; the agent-service database is about 0.1MB.
These are lineage or compact evidence, not duplicate prediction caches.

## Removed or reduced

- rejected spatial-temporal train/test tensor caches and validation predictions;
- rejected global and final-pool proxy caches/members;
- baseline and hybrid serialized models that are not used by the selected pipeline;
- duplicate SCADA, weighted-member, and cross-group prediction CSVs;
- legacy group-3-only OOF caches superseded by `exact_driver_oof.npz`;
- per-experiment directories after their minimal lineage inputs or diagnostic reports were moved;
- Python and pytest caches.

Historical human-readable experiment reports remain under `docs/reports/`. Previous submitted CSVs
remain recoverable under `submissions/archive/`.

Eighteen unreferenced calibration/weight-sweep CSVs in `submissions/archive/` were consolidated
into `legacy_sweeps_2026-07-17.zip`. The ZIP entry names were verified before the source CSVs were
removed. This reduced those files from 13.585MB to 5.309MB while keeping them recoverable. CSVs
used by exact OOF lineage code or explicitly referenced by reports/tests remain uncompressed.

## 2026-07-18 second cleanup

The two publicly rejected GEFS screens accounted for almost the entire new artifact
growth. Their 19,824 raw/temporary files (4,342,136,992 bytes) were removed after
retaining the decoded feature tables, request/join plans, source URLs, object hashes,
availability audit, and compact model reports. Those raw objects remain recoverable
from NOAA using the retained plans, but the pruned manifests are now explicitly
`competition_eligible: false` until the objects are redownloaded and revalidated.

The unverified retrospective Open-Meteo quarantine and obsolete GEFS pilot were also
removed. A duplicate group-2 candidate was removed only after its SHA-256 matched the
copy retained in `submissions/`. The artifact tree fell from 4.66 GB / 19,951 files to
0.295 GB / 126 files; these deletions are not locally recoverable, although NOAA raw
objects can be downloaded again.

The active `submissions/` root now contains the selected public best, the one isolated
manual group-2 comparison candidate, and `results.csv`. Superseded submitted files
remain under `submissions/archive/`.

## 2026-07-18 CFSv2 cleanup

The NOAA CFSv2 operational-forecast branch was rejected twice on 2024 H1: first as
a direct residual feature family and then as a harm-risk gate around the selected
meta correction. Locked H2 and the 2025 test period remained unopened. After changing
the retained manifest to `competition_eligible: false` and
`artifact_state: rejected_raw_pruned`, 364 H1 raw objects (130,040,923 bytes) and the
seven-file pilot directory (745,274 bytes) were removed. The 4,368-hour decoded
feature table, request/join plans, exact source URLs and byte ranges, checksums,
causality audit, and both model reports remain; the raw bytes are not locally
recoverable but can be redownloaded from the retained NOAA URLs.

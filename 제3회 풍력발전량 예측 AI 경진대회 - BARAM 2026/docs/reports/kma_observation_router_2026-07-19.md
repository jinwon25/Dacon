# KMA issue-time observation router — 2026-07-19

## Decision

The next controlled experiment targets the main group-3 bottleneck with a new
causal signal rather than another global calibration. It uses KMA ASOS weather
observed before each 13:00 KST prediction reference and routes only complete
six-hour lead blocks among existing OOF experts.

No submission is created by this experiment. The current public incumbent
remains unchanged until every locked promotion gate passes.

## Implemented pipeline

`experiments/fetch_kma_asos_observations.py`:

- downloads the official ASOS hourly period archive in at most 30-day chunks;
- defaults to Taebaek station 216 and supports multiple station IDs;
- accepts the API key only through `KMA_API_KEY`;
- stores raw responses, redacted URLs, retrieval times, and SHA-256 checksums;
- permits observations only through `issue time - 120 minutes`;
- builds 1/3/6/12/24-hour wind-vector, persistence, trend, gust, pressure,
  temperature, humidity, and rain features;
- emits an external-data manifest and exact observation-time causality audit.

The full 2024 group-3 OOF surface has 367 issue cycles. With one station and
the default 30-day chunks, the collector plans 13 provider requests.

`experiments/kma_observation_block_router.py`:

- preserves the existing issue-cycle and six-hour lead-phase dependency unit;
- learns each expert's official-metric utility with shallow CatBoost models;
- uses seeds 17 and 29 and averages their deployment surfaces;
- runs an identical model without ASOS features as the incremental control;
- routes only the top 5/10/15% positive-utility blocks and moves only
  5/10/15% toward the chosen expert;
- selects on Q1 -> Q2 and opens H1 -> H2 only for a qualifying frozen policy;
- bootstraps complete issue cycles rather than individual hourly rows.

## Promotion contract

A policy may open H2 only when its ensemble and both individual seeds have:

- positive total score delta;
- nonnegative 1-NMAE and FICR deltas;
- nonnegative score delta in every Q2 month;
- positive score and nonnegative component increments versus the otherwise
  identical no-observation control.

Locked promotion additionally requires issue-cycle bootstrap `q05 >= 0` and a
positive fraction of at least `0.90`. Failing any condition produces a rejected
diagnostic report and no CSV.

## Reproduction

In PowerShell, set the API key locally and run:

```powershell
$env:KMA_API_KEY = "<your KMA APIHub key>"
python -m experiments.fetch_kma_asos_observations
python -m experiments.kma_observation_block_router
```

The no-network request plan is reproducible without a key:

```powershell
python -m experiments.fetch_kma_asos_observations --plan-only
```

Expected retained inputs and outputs:

- `artifacts_final/external_weather/kma_asos_2024/raw/`
- `artifacts_final/external_weather/kma_asos_2024/issue_features.csv`
- `artifacts_final/external_weather/kma_asos_2024/manifest.json`
- `artifacts_final/diagnostics/kma_observation_block_router.json`

If ASOS passes its locked incremental gate, the already implemented KMA UM
N128 collector and disagreement screen become the next independent-source
addition. If ASOS fails, do not tune around H2: retain the rejection and move
to the full conditional-distribution/FICR action experiment.

## Current verification state

- The official period archive returned 8,803 causal station-hour joins for 367
  issue cycles. The generated feature table has 201 columns, all 13 raw files
  have retained SHA-256 checksums, and the two-hour availability audit has zero
  violations.
- Eight Q2 policies passed the development gate. The frozen selection was
  `shallow_a15_c15`: 15% movement on the top 15% six-hour blocks.
- On locked H2, the group-3 ensemble improved score by `+0.0024036`, 1-NMAE by
  `+0.0008390`, and FICR by `+0.0039682`. The complete-issue bootstrap was also
  positive (`q05=+0.0001669`, positive fraction `0.964`).
- Promotion nevertheless failed. November score fell `-0.0009520` with FICR
  `-0.0040365`, December 1-NMAE fell `-0.0001179`, and the incremental ASOS
  branch versus the identical no-observation control lost `-0.0000346`
  1-NMAE despite gaining `+0.0001284` FICR.
- The experiment is therefore retained as a promising but rejected diagnostic.
  No submission file was created or modified, and H2 must not be used to relax
  its policy or gates.
- Collector, causality, router, and existing-project regression tests all pass:
  `146 passed`.

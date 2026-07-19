# External Data and Pretrained Model Audit — 2026-07-18

## Decision

Submission 28 (`1494986`) is publicly rejected and archived. It scored
`0.6415388286`, which is `-0.0002083341` below submission 27. 1-NMAE rose by
`+0.0000796343`, but FICR fell by `-0.0004963026`; the broad 20% group-3
spatial-temporal blend therefore failed public transfer.

The next independent signal family is operational ensemble forecast uncertainty,
not another global structural blend. NOAA GEFS is the first pilot because it adds
ensemble spread and quantiles that are absent from the supplied deterministic GFS
features. It must pass the public-availability guard before any model result can be
promoted.

## Hard causality contract

- `data_available_kst_dtm` is the prediction reference time. In the supplied test
  GFS file it is 13:00 KST, or 04:00 UTC.
- A forecast row is eligible only when its retained public-availability timestamp
  is no later than that reference time.
- Valid time may be later than the reference time; valid time never proves that the
  forecast was available.
- Without a provider timestamp, the pilot uses a conservative publication delay of
  6 hours 10 minutes. At a 13:00 KST cutoff, this selects the previous day's 18Z
  cycle rather than same-day 00Z.
- Every raw file must retain source URL, retrieval time, SHA-256, provider, model
  cycle, variables, period, license, and preprocessing lineage.
- Retrospective observation/reanalysis products and post-reference forecast cycles
  are forbidden.

`agent_service.compliance` now enforces the run-selection rule. Any run tagged with
`external_data` is automatically rejected when it has no eligible manifest, a
causality violation, or a missing/checksum-mismatched raw file.

## Source triage

### Promote to pilot: NOAA GEFS operational archive

GEFS supplies multiple ensemble members and four daily cycles. The useful derived
features are ensemble mean/spread and wind-speed quantiles, joined by exact model
initialization and lead time. These target forecast uncertainty and should be most
relevant to FICR, the component that has caused recent public failures.

The first experiment is deliberately narrow:

1. Use only the latest run conservatively public before each issue time.
2. Aggregate members at the competition grid/nearby cells into wind vector mean,
   speed standard deviation, and p10/p50/p90.
3. Train a group-3 residual/uncertainty member on 2023, select on 2024 H1, and lock
   2024 H2.
4. Require positive score, 1-NMAE, and FICR for every seed; complete issue-cycle
   bootstrap q05 must be non-negative; every locked month FICR must be non-negative.
5. A candidate may change at most 5–10% of rows after the public structural failure.

### Low priority: additional deterministic GFS fields

NOAA's GFS archive is provenance-safe, but supplied competition GFS already includes
10/80/100 m winds, gusts, PBL wind, and pressure-level variables. Extra deterministic
GFS is likely redundant and is lower priority than GEFS spread.

### Blocked: retrospective Open-Meteo history

The existing Historical Forecast and Previous Runs collectors do not retain enough
evidence of the original run's public-availability time. Their CLIs now fail closed
unless explicitly run as unverified research-only downloads. Existing files are
quarantined and are not submission inputs.

## Pretrained weather-model audit

No pretrained weather foundation model currently passes all four gates: license,
publication cutoff, causal inputs, and practical incremental value.

| Model | Weight/license finding | Causal/runtime finding | Decision |
|---|---|---|---|
| GraphCast | Weight terms are CC BY-NC-SA | Non-commercial restriction conflicts with the competition rule | Exclude |
| Pangu-Weather | Official weights are BY-NC-SA | Non-commercial restriction conflicts with the competition rule | Exclude |
| NeuralGCM | Checkpoints are CC BY-SA and code is Apache-2.0 | Official workflow uses ERA5 inputs; reanalysis after the reference time is forbidden, and resolution is coarse | Do not use dynamically |
| Aurora | Official Hugging Face repository is tagged MIT | Exact weight-license scope needs written confirmation; global 0.25-degree inputs and large GPU cost make it a low-priority pilot | Hold |
| FourCastNet | Code license and checkpoint terms are not sufficiently aligned/clear | Input and compute burden is high | Exclude until clarified |

Pretrained weights must also have been officially public by 2026-07-05. Newer model
versions or checkpoints are excluded even when their family existed earlier.

## Promotion checklist

- Eligible manifest passes `validate_external_data_manifest` with raw-file checksum
  verification.
- Per-row availability audit has zero violations and non-negative minimum margin.
- No retrospective API, observation, reanalysis, or post-reference run is used.
- Full source/license/retrieval/preprocessing bundle is reproducible offline.
- 2023→2024 rolling evaluation, locked H2, issue-cycle bootstrap, worst-month FICR,
  and seed-component gates all pass.
- Only then may a CSV be written to `submissions/` and selected by the service.

## Implemented GEFS pilot and screen

The operational-archive pipeline was implemented and tested rather than left as a
research proposal.

- The 2023–2024 target period uses 731 prediction reference times and 6,579 exact
  previous-day 18Z GEFS objects. Range requests retain only the 10 m U/V ensemble
  standard-deviation GRIB messages.
- All 6,579 source objects have a URL, S3 `Last-Modified`, byte ranges, retrieval
  timestamp, local SHA-256, and atomic sidecar. Total retained raw size is about
  2.5 GB.
- The availability audit has zero violations. The minimum actual margin between
  S3 publication and `data_available_kst_dtm` is 30.77 minutes; this confirms why
  per-object timestamps are necessary even with a conservative cycle rule.
- ecCodes reproducibly decoded 157,896 rows (17,544 target hours × 9 grids), with
  no missing values. The feature-cache SHA-256 is
  `c237020911b92b1cf4db73145a5635c29f1dc0fb6cb89a04c1811f57cc168f50`.

The first causal residual screen used Q1 for training and Q2 only for policy
selection. A calendar/current-prediction control and the same model plus GEFS spread
were evaluated under identical sparse 2–10% gates. The GEFS model's best Q2 policy
improved score by `+0.00005188` and FICR by `+0.00011463`, but reduced 1-NMAE by
`-0.00001087`. Therefore no policy improved all three components and H2 remained
unopened. The strongest Spearman correlation between a spread feature and eligible
absolute error was only `0.0624`.

Decision: reject 10 m component-spread residual correction as a submission family.
Do not download the 2025 GEFS test period or generate a submission from this screen.
The decoded feature cache, request plan, source URLs/checksums, and reports are retained.
The 4.04 GB rejected raw GRIB payloads were pruned; both manifests now fail closed until
their raw objects are redownloaded and revalidated. Do not micro-tune this failed policy.

## KMA UM global operational-forecast branch

The next independent source is the official KMA APIHub UM global N128 archive. This is
not an observation or reanalysis: every row is keyed to its original model initialization
and lead. KMA documents N128 history from 2018-06-09 and explicitly states that historical
UM data remains queryable after new UM production ended on 2026-03-31.

`experiments/fetch_kma_um_global.py` now:

- selects the latest cycle public before each `data_available_kst_dtm` using a conservative
  12-hour assumed publication delay (the 13:00 KST cutoff therefore selects previous-day
  12Z, leaving a four-hour margin);
- requests only 10 m U/V operational forecasts at the configured public coordinates;
- brackets each hourly target with 3-hour leads and interpolates only within that bracket;
- retains redacted source URL, retrieval time, issue/valid/publication times, byte count,
  SHA-256, raw response, and a causality manifest;
- rejects missing cycles, incomplete variables/points, out-of-range interpolation, and any
  post-reference publication;
- reads a user-issued key only from `KMA_API_KEY`, never from CLI arguments or artifacts.

The parser, request selector, secret redaction, and causal feature join have synthetic tests;
the complete suite passes (`118 passed`). A real one-issue pilot remains blocked only on a
user-issued APIHub key. Do not use keys exposed by search indexes or third-party documents:
APIHub terms require each member to use their own key.

Once the pilot passes, collect only 2024 first. Train/select on Q1/Q2 and open locked H2 once;
download 2025 only if total score, 1-NMAE, and FICR all pass the locked external-signal gate.

## Official references

- NOAA GEFS: https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast
- NOAA GFS: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- Open-Meteo Single Runs: https://open-meteo.com/en/docs/single-runs-api
- Open-Meteo license: https://open-meteo.com/en/license
- GraphCast: https://github.com/google-deepmind/graphcast
- Pangu-Weather: https://github.com/198808xc/Pangu-Weather
- NeuralGCM: https://github.com/neuralgcm/neuralgcm
- NeuralGCM checkpoints: https://neuralgcm.readthedocs.io/en/stable/checkpoints.html
- Aurora: https://huggingface.co/microsoft/aurora
- KMA numerical-model API: https://apihub.kma.go.kr/apiList.do?seqApi=9
- KMA historical UM availability notice: https://apihub.kma.go.kr/notice.do?seqNotice=52
- KMA APIHub use/key policy: https://apihub.kma.go.kr/apiInfo.do

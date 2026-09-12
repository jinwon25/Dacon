# JMA MSM causal previous-run breakthrough — 2026-07-26

## Outcome

The strongest rules-compliant candidate produced in this sprint is:

- `artifacts_final/candidates/kma_jma_msm_stencil_g1g3_20260726.csv`
- SHA-256:
  `28c72d0c97893728a948a53e5553aa0d0b21677ec58e144eaf935e48dbf7d61c`
- rows: `8,760`
- encoding: UTF-8 with BOM
- validator: passed, with the exact five submission columns
- projected public score if the 2024 local delta transfers: `0.6476720663`

It leaves the incumbent group-2 forecast unchanged and replaces only groups 1
and 3 with bounded blends toward a JMA MSM/KMA year-forward expert.

## Why this is a genuinely new source

JMA's official MSM documentation describes:

- a 5 km surface grid over `22.4N–47.6N, 120E–150E`, which contains the BARAM
  sites;
- runs every three hours;
- 39/78-hour forecast ranges;
- approximate distribution `2 hours 30 minutes` after initialisation.

Sources:

- <https://www.jmbsc.or.jp/jp/online/file/f-online10200.html>
- <https://www.jma.go.jp/jma/en/Activities/nwp.html>

Open-Meteo's Previous Runs API provides JMA MSM from 2018 and defines
`previous_day1` and `previous_day2` as forecasts made 24 and 48 hours before
valid time:

- <https://open-meteo.com/en/docs/previous-runs-api>
- <https://open-meteo.com/en/licence>

This API was used only to retrieve archived weather-model output. It was not
used to infer wind-power generation. All power models ran locally. Every raw
JSON response, exact query URL, checksum, derived feature checksum, and
causality audit is retained in:

- `artifacts_final/external_weather/jma_msm_stencil_2023`
- `artifacts_final/external_weather/jma_msm_stencil_2024`
- `artifacts_final/external_weather/jma_msm_stencil_2025`

## Causal forecast policy

BARAM target leads are 12–35 hours after the daily 13:00 KST information
cutoff. With JMA's documented 150-minute distribution lag:

- leads 12–21 use the forecast made 24 hours before valid time;
- leads 22–35 use the forecast made 48 hours before valid time.

The conservative public-time upper bound is therefore:

`valid_time - selected_offset + 150 minutes`.

Across all 2023, 2024, and 2025 contexts:

- causality violations: `0`;
- minimum margin before the BARAM cutoff: `30 minutes`;
- maximum margin: `1,410 minutes`.

Five missing day-2 hours on 2024-11-12 and 2025-11-11 were interpolated in
u/v vector space from the adjacent day-2 forecast hours. Those adjacent
forecasts share the same safe, pre-cutoff information class; no day-1 value
was substituted into an unsafe late-lead row.

The Historical Forecast API was explicitly rejected even though it is easy to
retrieve: it stitches the newest model runs and can include information issued
after the BARAM cutoff.

## Model and validation contract

The fixed 3×3 MSM surface-wind stencil contains the nearest model cells around
the site. It is reduced without generation labels to 41 physical features:

- centre, mean, standard deviation, minimum, and maximum of safe speed/u/v;
- east-west and north-south contrasts;
- safe 48-hour backup fields;
- 24-vs-48-hour disagreement where both are causal;
- selected offset and availability-regime indicators.

The local quantile LightGBM expert follows the existing year-forward contract:

1. select quantile on 2023 Q4 only;
2. train on all 2023 and predict 2024;
3. select bounded blend weight on 2024 Q1 only;
4. confirm on Q2 and locked H2;
5. require all three seeds, score components, month fraction, and issue-block
   bootstrap gates;
6. refit on 2024 and predict 2025 with the frozen specification.

Every row is capped to an absolute movement of 5% of group capacity.

## Locked OOF result

| group | alpha | weight | Q1 | Q2 | H2 | full | positive months | bootstrap q05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| group 1 | 0.75 | 0.20 | +0.010628 | +0.004119 | +0.007908 | **+0.007851** | 10/12 | +0.002496 |
| group 3 | 0.80 | 0.05 | +0.003521 | +0.002474 | +0.002623 | **+0.002866** | 11/12 | +0.001175 |

Both groups passed:

- Q1/Q2/H2/full score, 1-NMAE, and FICR positivity;
- every seed on every split;
- at least 80% positive months;
- issue-block bootstrap q05 above zero;
- bootstrap positive fraction above 98%.

The expected macro delta is:

`(+0.0078512298 + 0 + +0.0028655343) / 3 = +0.0035722547`.

## Rejected branches

### NOAA GEFS spread

Adding compact GEFS spread summaries produced positive average deltas, but its
movement/error-gain correlations with the incumbent KMA experts were
approximately `0.90–0.92`. Q1 selected the existing experts, so row or group
blending added little genuine diversity.

### JMA single cell

The single-cell source was strong on average, but group 1 failed one seed's Q2
1-NMAE by `-0.000186`. The 3×3 stencil removed that instability.

### JMA group 2

JMA-only and KMA+JMA experts produced positive annual means, but Q2 FICR or
issue-block bootstrap remained negative. Group 2 is therefore unchanged.

### 2022 H2 expanding training

Open-Meteo coverage before 2022-07-04 was incomplete. Adding the available
2022 H2 rows reversed group-1 Q2 to `-0.002998`, so the expanding model was
rejected.

### Lead-regime weight expansion

The Q1 optimum increased the early-lead group-1 weight, but confirmation left
only 9/12 positive months. The uniform frozen weight `0.20` remains the
submission policy.

## Submission order

Upload the stable stencil candidate first. The public result is still required
to measure the local-to-public transfer ratio; no leaderboard score was used
for model or weight selection.

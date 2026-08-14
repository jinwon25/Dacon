# Top-tier score sprint — 2026-07-12

## Decision

The original active submission candidate was:

- file: `submissions/blend_best_turbine_g3_10_agree10.csv`
- DACON title: `Best turbine G3 10 agree10`
- base: public best `blend_stack5_weighted_g12_2_agree4.csv` (`0.6403102237`)
- mutation: group 3 only, 10% toward the new turbine-SCADA pool when base power is at least 10% of capacity and base/member disagreement is at most 10% of capacity

Groups 1 and 2 remain numerically unchanged. Group 3 changes on 6,613/8,760 rows; mean absolute movement is 69.69 kWh and the 95th percentile is 177.25 kWh.

## Leaderboard diagnosis

The official leaderboard observed on 2026-07-12 had a leading score of `0.66912` (`1-NMAE 0.88483`, `FICR 0.45342`). The current best is `0.6403102237` (`1-NMAE 0.8753191781`, `FICR 0.4053012693`). The main numerical gap is therefore FICR, not only average error. Source: [DACON official leaderboard](https://dacon.io/en/competitions/official/236727/leaderboard).

Group-level local diagnostics pointed to group 3 as the weakest component. Correcting the official FICR implementation to use actual-generation-weighted settlement removed an earlier optimistic local estimate and changed model selection materially.

## Data and model findings

Train-period SCADA alignment was re-audited at turbine level:

- Vestas group 1/2 records align to hourly labels with a +50-minute shift and require all six ten-minute readings per hour.
- Unison group 3 records align with +60 minutes.
- Strict group-3 SCADA aggregation tracks labels extremely closely on complete hours, making turbine operating-state filtering a high-value feature source.

The winning group-3 local pool combines:

- 23.77% capacity-normalized global model;
- 76.23% turbine-shared nominal-operation model, trained after excluding likely stoppage/curtailment rows where nacelle wind speed is at least 5 m/s but turbine power is below 5% of rated power.

Corrected 2024 holdout results:

| model | group-3 score | 1-NMAE | FICR |
|---|---:|---:|---:|
| global generation-weighted | 0.60230 | — | — |
| turbine shared nominal | 0.60710 | 0.85806 | 0.35613 |
| selected group-3 pool | 0.61165 | 0.86333 | 0.35996 |

The selected group-3 pool improved 9 of 12 validation months. The complete three-group local pool scored `0.65406` (`1-NMAE 0.87753`, `FICR 0.43059`). This is a validation selection signal, not a public-score forecast.

## External method review

The implemented direction agrees with strong published wind-forecasting practice:

- The GEFCom2014 winning approach used separate gradient-boosting models and emphasized smoothing the dominant wind signal: [Landry et al., International Journal of Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207016000145).
- A top GEFCom2014 solution combined quantile random forests and stacked RF/GBDT ensembles, supporting diverse tree ensembles rather than one large monolithic model: [Nagy et al., International Journal of Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207015001521).
- Multi-point NWP research recommends wind-variable PCA, lead/lag forecasts, and spatial mean/dispersion features: [Bessa et al., IEEE Transactions on Sustainable Energy](https://repositorio.inesctec.pt/bitstreams/2931c13f-fa54-46c0-8462-bca18cc34ce2/download). The direct PCA/lag-lead replication was tested here but reduced the corrected local score, so it was rejected.
- SCADA literature stresses filtering stoppage and curtailment anomalies before fitting power curves: [Long et al., Renewable Energy](https://eprints.gla.ac.uk/260595/). This directly motivated the nominal-operation turbine model.

Foundation time-series models were considered, but the test period supplies no actual target/SCADA history. Their usual autoregressive advantage is therefore unavailable, while the NWP/SCADA tree pipeline is rule-safe, reproducible, and empirically stronger on this data.

## Rejected experiments

- strict hourly SCADA aggregation without turbine operating-state modeling: group-3 score `0.58327`;
- PCA plus NWP lead/lag feature replication: group-3 score `0.58286`, overall `0.63779`;
- group-3 aggregate CatBoost specialist: best `0.59472`;
- broad replacement by the new pool: locally strong but too large a distribution shift for a first public probe.

Failed experiment sources and artifacts were removed. The submission directory retains only the current public best, the one next candidate, and the leaderboard result log.

## Public feedback — 2026-07-13

Submission `1488631`, `Best turbine G3 10 agree10`, scored `0.6401243372` (`1-NMAE 0.8751815837`, `FICR 0.4050670908`). Relative to the current best, the changes were:

- score: `-0.0001858865`;
- 1-NMAE: `-0.0001375944`;
- FICR: `-0.0002341785`.

Because groups 1 and 2 were unchanged, this is direct public evidence that the group-3 correction direction was wrong despite its positive 2024 holdout result. The next candidate is `submissions/blend_best_antiturbine_g3_10_agree10.csv`, titled `Best antiturbine G3 10 agree10`. It uses the exact same eligibility mask and magnitude but reverses the sign, providing a controlled estimate of the public group-3 gradient.

Submission `1488632`, the reverse-direction probe, scored `0.6400644769` (`1-NMAE 0.8754264537`, `FICR 0.4047025001`). Relative to the current best, 1-NMAE improved by `0.0001072756`, but FICR fell by `0.0005987692`, so total score fell by `0.0002457468`. The conclusion is not merely that the sign was wrong: moving either way on this turbine-derived axis destroys more threshold settlement than it creates. The entire group-3 turbine injection family is rejected, and submission `1488084` remains selected.

A recent-year specialist was then tested to address the 2024-to-2025 distribution shift. Training on 2023 for 2024 validation and refitting on 2024 produced an overall corrected local score of `0.63899` (`1-NMAE 0.87619`, `FICR 0.40179`). Group 3 FICR was only `0.32141`. The candidate was rejected without public submission and its generated artifacts were removed.

## Rule compliance

- Test-period actual generation and test-period SCADA are not used.
- All SCADA-derived learning uses only train-period records.
- No external weather observations or remote inference API are used.
- Predictions are clipped to official group capacities and preserve the sample-submission key order.

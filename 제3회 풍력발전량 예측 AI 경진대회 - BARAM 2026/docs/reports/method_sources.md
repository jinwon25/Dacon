# Method Sources

The next experiments are guided by competition practice and wind-power forecasting literature.

## Official Competition Sources

- The official DACON overview defines the task as wind-power generation forecasting from weather forecasts.
  - https://dacon.io/competitions/official/236727/overview/description
- The official rule page emphasizes that every prediction may only use information actually available at the forecast reference time. This keeps the current SCADA usage restricted to train-period proxy modeling, not test-period SCADA leakage.
  - https://dacon.io/competitions/official/236727/overview/rules
- DACON's evaluation code share confirms the competition score as `0.5 * (1 - NMAE) + 0.5 * FICR`.
  - https://dacon.io/competitions/official/236727/codeshare/14035
- The public leaderboard currently shows that the top teams are gaining mostly through FICR, not only 1-NMAE, so high-variance calibration that hurts FICR should be avoided.
  - https://dacon.io/competitions/official/236727/leaderboard

## Competition Practice

- The official KDD Cup 2022 rank-1 paper combined a deep spatio-temporal model with a
  spatio-partitioned, time-phased tree model and a data-driven ensemble. Its ablation shows that
  the complementary ensemble outperformed either branch alone. The transferable parts for BARAM
  are lead-phase experts, meteorological partitioning, and constrained heterogeneous ensembling.
  - https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0518.pdf
- The official KDD Cup 2022 rank-2 solution independently combined LightGBM and GRU forecasts and
  used local ensembling across multi-step schemes. This reinforces using structurally different
  time-horizon experts instead of another single monolithic tree fit.
  - https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1286.pdf
- The KDD Cup workshop ranking page records those solutions as ranks 1 and 2 in the regular track.
  - https://baidukddcup2022.github.io/

- GEFCom2014 organized wind, solar, load, and price forecasting tracks and is a useful reference for energy-forecasting competition workflows. The published summary emphasizes preprocessing, multiple modeling tracks, and post-processing/ensembling rather than relying on a single model.
  - https://robjhyndman.com/papers/gefcom2014.pdf
  - https://blog.drhongtao.com/2016/01/probabilistic-energy-forecasting-gefcom2014.html
- The winning GEFCom2014 wind method used gradient boosted machines fitted independently by wind zone and quantile. Even though BARAM is point forecasting, the key transferable idea is zone/group-specific tree models plus distribution-aware post-processing.
  It also smoothed and used bidirectional lags of the dominant wind-energy signal and added a
  cross-sectional second layer across correlated wind farms.
  - https://www.sciencedirect.com/science/article/pii/S0169207016000145
- The winning GEFCom2012 wind solution used feature engineering, separate farm/horizon models,
  weather-similarity clusters, and a linear combination/smoothing layer. Its validation matched
  the competition's forecast-block geometry rather than using random rows.
  - https://www.sciencedirect.com/science/article/pii/S0169207013000836
- CatBoost's `MultiRMSEWithMissingValues` was tested as a clean auxiliary-farm transfer mechanism
  for group 3's missing 2022 labels. It improved Q2 group 3 strongly but reversed on locked H2,
  so missing-target shared trees are now a recorded negative-transfer family.
  - https://catboost.ai/docs/en/concepts/loss-functions-multiregression
- XGBoost was tested as a direct algorithm-diversity member because the production lineage had
  only LightGBM and CatBoost direct learners. It was both weaker and 0.992 correlated with the
  CatBoost parent, so this diversity branch is rejected.
  - https://arxiv.org/abs/1603.02754
- Another GEFCom2014 wind/solar paper used a voted ensemble and stacked random forest/GBDT model with post-processing, which supports keeping heterogeneous tree members and small constrained stack weights.
  - https://ideas.repec.org/a/eee/intfor/v32y2016i3p1087-1093.html

## Wind Forecasting Methods

- Wind-power forecasting literature commonly separates physical/NWP information, SCADA measurements, power-curve behavior, and statistical post-processing.
- An NREL-led numerical-ensemble study defines hub-height wind uncertainty using normalized
  across-ensemble wind-speed variability and finds meaningful stability-regime differences. BARAM
  has LDAPS and GFS rather than WRF ensemble members, but their hub-vector disagreement is a direct,
  forecast-time-safe analogue worth testing as a regime feature.
  - https://doi.org/10.5194/wes-6-1363-2021
- A 2023 wind-farm experiment integrating several NWP sources reported error reductions from their
  combined use, supporting explicit cross-source features rather than only parallel raw columns.
  - https://doi.org/10.1016/j.heliyon.2023.e21479
- The NREL-led Power Curve Working Group assessment confirms that air density, turbulence, wind
  shear, and atmospheric stability affect real power-curve behavior. It documents the IEC-style
  density normalization `Vn = V * (rho / rho0)^(1/3)` and notes that density as an independent
  statistical-model input can outperform using only normalized speed.
  - https://doi.org/10.5194/wes-5-199-2020
- Recent SCADA/NWP bias-correction work supports the current direction: learn a proxy from train-period SCADA and inject it into the final forecast without using test-period SCADA.
  - https://arxiv.org/html/2402.13916
  - https://www.sciencedirect.com/science/article/pii/S1364032126002406

## Operational External Forecast Sources

- NOAA/NCEI documents CFSv2 operational forecasts from 2011 onward and retains the
  historical operational nine-month forecast archive. NCEP documents four daily
  cycles and the member-specific 10 m U/V time-series products used by the collector.
  This branch was causally eligible but failed both the direct-residual and risk-gate
  H1 selection contracts, so it was rejected before H2 or 2025 collection.
  - https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system
  - https://www.nco.ncep.noaa.gov/pmb/products/cfs/
- NOAA identifies its web information as public information unless otherwise noted;
  the retained manifest records this disclaimer together with all source URLs and
  checksums.
  - https://www.noaa.gov/disclaimer

- KMA APIHub officially exposes historical UM global N128 point forecasts by model
  initialization (`tmfc`), forecast lead (`hf`), variable code, and longitude/latitude.
  N128 history is documented from 2018-06-09 onward, so it covers the 2024 OOF period.
  - https://apihub.kma.go.kr/apiList.do?seqApi=9
- KMA ended new UM production on 2026-03-31 but explicitly states that past UM data
  remains queryable. This supports reproducible retrieval of the original operational
  forecast, not a later reanalysis.
  - https://apihub.kma.go.kr/notice.do?seqNotice=52
- APIHub requires a member's own authentication key. General members are automatically
  approved and currently receive 20,000 calls / 5 GB per day. Keys may not be loaned or
  transferred, so the collector reads only the local `KMA_API_KEY` environment variable.
  - https://apihub.kma.go.kr/apiInfo.do

## Deep and Spatial Models

- Deep wind forecasting surveys and KDD Cup wind papers support spatial/temporal models, but for this competition the train size and no-test-SCADA constraint make GBDT and proxy stacking more practical first.
  - https://link.springer.com/article/10.1007/s10462-024-10728-z
  - https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_5582.pdf

## Practical Takeaway

For this repository:

1. Keep the strong GBDT blend as the base.
2. Add decorrelated members only when they move public score.
3. Use SCADA proxy members at low weights because `stack5` improved public score but `stack10` degraded FICR.
4. Prefer constrained group-wise injection over unconstrained larger global injection.
5. Optimize against FICR sensitivity separately from NMAE; public evidence shows that better 1-NMAE can still lose total score when FICR drops.
6. Treat physical power-curve models as low-correlation ensemble members, not direct replacements for the GBDT blend.

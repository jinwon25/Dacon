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

- GEFCom2014 organized wind, solar, load, and price forecasting tracks and is a useful reference for energy-forecasting competition workflows. The published summary emphasizes preprocessing, multiple modeling tracks, and post-processing/ensembling rather than relying on a single model.
  - https://robjhyndman.com/papers/gefcom2014.pdf
  - https://blog.drhongtao.com/2016/01/probabilistic-energy-forecasting-gefcom2014.html
- The winning GEFCom2014 wind method used gradient boosted machines fitted independently by wind zone and quantile. Even though BARAM is point forecasting, the key transferable idea is zone/group-specific tree models plus distribution-aware post-processing.
  - https://ideas.repec.org/a/eee/intfor/v32y2016i3p1061-1066.html
- Another GEFCom2014 wind/solar paper used a voted ensemble and stacked random forest/GBDT model with post-processing, which supports keeping heterogeneous tree members and small constrained stack weights.
  - https://ideas.repec.org/a/eee/intfor/v32y2016i3p1087-1093.html

## Wind Forecasting Methods

- Wind-power forecasting literature commonly separates physical/NWP information, SCADA measurements, power-curve behavior, and statistical post-processing.
- Recent SCADA/NWP bias-correction work supports the current direction: learn a proxy from train-period SCADA and inject it into the final forecast without using test-period SCADA.
  - https://arxiv.org/html/2402.13916
  - https://www.sciencedirect.com/science/article/pii/S1364032126002406

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

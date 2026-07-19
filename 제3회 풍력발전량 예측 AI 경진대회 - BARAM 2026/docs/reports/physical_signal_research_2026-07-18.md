# BARAM physical-signal research sprint (2026-07-18)

## Scope and data contract

Only official competition NWP files are read, with `usecols` restricting the
local read to the required schema. No raw row, target value, SCADA value, or
derived record was sent to an external service. The external research used
paper and official-document text only.

## Ranked signals mapped to the BARAM schema

### 1. Cross-NWP hub-wind disagreement and agreement regimes

**Why first:** BARAM uniquely provides two structurally different forecast
sources, LDAPS and GFS. The across-member spread is therefore an available-at-
issue-time proxy for model uncertainty. NREL-led research quantifies NWP wind
uncertainty using normalized across-ensemble hub-height wind-speed variability
and finds that uncertainty changes by atmospheric regime. A 2023 wind-farm
study also reports that combining several NWP sources reduced forecast errors.

**BARAM mapping:**

- LDAPS: `heightAboveGround_10_10u/v` and mean of the 50 m max/min `u/v`
- GFS: `heightAboveGround_80_u/v`, `heightAboveGround_100_100u/v`
- derived at each target group's turbine-location IDW surface:
  `nwp_hub_ws_abs_diff`, `nwp_hub_ws_rel_spread`,
  `nwp_hub_vector_diff`, `nwp_direction_agreement`,
  `nwp_density_ws_abs_diff`, `nwp_power_density_log_ratio`, and
  `nwp_shear_abs_diff`

Primary sources:

- Bodini et al. (2021), *Wind Energy Science*:
  https://doi.org/10.5194/wes-6-1363-2021
- Yakoub et al. (2023), direct/indirect forecasts integrating several NWP
  sources: https://doi.org/10.1016/j.heliyon.2023.e21479

### 2. Density, shear, and gust power-deviation surface

**Why second:** wind power depends on air density, turbulence and vertical
shear in addition to hub-height wind speed. The NREL Power Curve Working Group
found that these conditions systematically move production away from a
one-dimensional reference power curve. Importantly, their assessment notes
that statistical models can benefit more from air density as an independent
input than from only replacing wind speed with a density-normalized speed.

**BARAM mapping:**

- density: `surface_0_sp`, 2 m temperature, and 2 m specific humidity from
  each NWP; `rho = p / (R_d * T * (1 + 0.61 q))`
- density-normalized speed: `Vn = V * (rho / 1.225)^(1/3)`
- shear: LDAPS 10--50 m and GFS 80--100 m power-law exponents
- gust proxy: GFS `surface_0_gust / hub_ws117`
- direct features: `air_density`, `density_ws117`, `wind_power_density`,
  `shear_alpha`, and `gust_factor`

Primary source:

- Lee et al. (2020), NREL-led Power Curve Working Group assessment:
  https://doi.org/10.5194/wes-5-199-2020

### 3. Directional turbine-layout wake alignment

**Why third:** group-level output changes when the forecast flow aligns turbine
pairs and downstream machines become waked. The KDD Cup 2022 rank-1 solution
explicitly used wake/location structure when partitioning turbines. BARAM has
the exact turbine coordinates and source-specific hub wind vectors, so a soft
layout-alignment index can be computed without test SCADA.

**BARAM mapping:** for every turbine pair, project the normalized hub wind
vector onto the pair axis, raise absolute alignment to a high even power, and
average with inverse-distance weights. Separate `ldaps_wake_alignment` and
`gfs_wake_alignment` features retain source disagreement.

Primary source:

- KDD Cup 2022 rank-1 paper, Li et al. (2022):
  https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0518.pdf

## Implemented experiment

`src/physical_signals.py` implements all three compact signal families.
`experiments/group3_nwp_disagreement_residual.py` tests the first-ranked family
as a small correction to the exact current group-3 OOF surface:

1. train residual model on 2024-01 through 2024-04;
2. select bounded correction/gating policy on 2024-05 through 2024-06;
3. refit only through 2024-H1;
4. evaluate once on locked 2024-H2;
5. require score, 1-NMAE, FICR, month stability, and day-bootstrap evidence
   before permitting a test-inference child experiment.

## Result

Selection improved by `+0.002844`, but locked H2 changed by `-0.000032`:

- 1-NMAE: `+0.000184`
- FICR: `-0.000247`
- positive months: `2 / 6`
- positive day-bootstrap fraction: `0.489`

The selected policy effectively removed the NWP-spread gate, which is direct
evidence that LDAPS--GFS disagreement did not add a stable correction boundary
on this residual setup. No submission was created. The signal generator remains
useful for the next, structurally different experiment: direct density/shear/
gust power-deviation modeling, with the wake index used only as a regime input.

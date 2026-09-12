# P0 Leakage and data-availability audit

Timestamps are stored without timezone offsets but are documented and interpreted as Asia/Seoul (KST).
The forecast day is the 24-hour block from 01:00 through next-day 00:00; its legal cutoff is the prior day at 14:00 KST.

## Weather availability

| Source | Split | Rows | Forecast range | Availability range | Grids | Exact duplicates | Multi-cycle pairs | Post-cutoff rows | Lead hours | Margin h |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| ldaps | train | 420,864 | 2022-01-01 01:00:00 – 2025-01-01 00:00:00 | 2021-12-31 13:00:00 – 2024-12-30 13:00:00 | 16 | 0 | 0 | 0 | 12–35 | 1–1 |
| gfs | train | 236,736 | 2022-01-01 01:00:00 – 2025-01-01 00:00:00 | 2021-12-31 13:00:00 – 2024-12-30 13:00:00 | 9 | 0 | 0 | 0 | 12–35 | 1–1 |
| ldaps | test | 140,160 | 2025-01-01 01:00:00 – 2026-01-01 00:00:00 | 2024-12-31 13:00:00 – 2025-12-30 13:00:00 | 16 | 0 | 0 | 0 | 12–35 | 1–1 |
| gfs | test | 78,840 | 2025-01-01 01:00:00 – 2026-01-01 00:00:00 | 2024-12-31 13:00:00 – 2025-12-30 13:00:00 | 9 | 0 | 0 | 0 | 12–35 | 1–1 |

## Labels

| Group | Provided range | Provided | Missing | Eligible >=10% | Eligible rate | Negative | Above capacity |
|---|---|---:|---:|---:|---:|---:|---:|
| kpx_group_1 | 2022-01-01 01:00:00 – 2025-01-01 00:00:00 | 26,200 | 104 | 15,915 | 60.74% | 0 | 0 |
| kpx_group_2 | 2022-01-01 01:00:00 – 2025-01-01 00:00:00 | 26,201 | 103 | 15,891 | 60.65% | 0 | 0 |
| kpx_group_3 | 2023-01-01 01:00:00 – 2025-01-01 00:00:00 | 17,538 | 8,766 | 9,414 | 53.68% | 0 | 38 |

## Turbine mapping

| Group | Manufacturer/model | Turbines | Capacity sum MW | Declared MW |
|---:|---|---:|---:|---:|
| 1 | VESTAS / V126 | 6 | 21.6 | 21.6 |
| 2 | VESTAS / V126 | 6 | 21.6 | 21.6 |
| 3 | UNISON / U136 | 5 | 21 | 21 |

## SCADA alignment

`sum` means summing six 10-minute per-turbine values into hourly energy. `kw_to_kwh` is the competing divide-by-six interpretation.

| Group | Best correlation candidate | Lag h | Corr | Best absolute-scale candidate | Lag h | NMAE | Bias kWh | Stop/curtail candidates |
|---|---|---:|---:|---|---:|---:|---:|---:|
| kpx_group_1 | ceil:sum | 0 | 0.999731 | ceil:sum | 0 | 0.007887 | 135.3 | 38,526 (4.07%) |
| kpx_group_2 | ceil:sum | 0 | 0.999708 | ceil:sum | 0 | 0.007212 | 111.2 | 34,722 (3.67%) |
| kpx_group_3 | floor:sum | 1 | 0.999913 | floor:sum | 1 | 0.002544 | -4.9 | 23,643 (4.54%) |

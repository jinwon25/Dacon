# v13 recent exact-ASOF submission validation

- Candidate / parent: `submit_v13_fixed.zip` / `submit_v11.zip`
- ZIP size: **25.906 MB** compressed, **65.854 MB** extracted
- Exact parent lineage: **True**
- Official sample smoke: **pass**, 8.974s, peak 273.5 MB
- Representative 245,789-row archive inference: **pass**, 32.193s, peak 1418.4 MB
- Split-batch invariance: **True**, max absolute difference 1.110e-16
- R_ANCHOR unchanged from v11: **True**
- R_CORE overlay active: **True**
- F overlay active: **True**
- Offline and row-local static audit: **True**
- Domain comparison: `{'R_CORE': {'rows': 64, 'mean_difference': -0.04711409366785897, 'mean_abs_difference': 0.04711409366785897, 'max_abs_difference': 0.06392963458762524}, 'R_ANCHOR': {'rows': 64, 'mean_difference': 0.0, 'mean_abs_difference': 1.734723475976807e-17, 'max_abs_difference': 1.1102230246251565e-16}, 'F': {'rows': 64, 'mean_difference': -0.024816964800865943, 'mean_abs_difference': 0.024816964800865943, 'max_abs_difference': 0.039756114857424474}}`
- All gates: **True**

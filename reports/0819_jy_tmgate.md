# 0819_jy TrackMan-ASOF Gate

## Summary

- Parent line: `submit_v25.zip` -> `src/package_v26_anchor_weight_probe.py`
- Candidate recipe: v26 15% R_ANCHOR probe plus a final R_ANCHOR-only TrackMan-ASOF shrink gate
- Submitted file: `0819_3_tmgate03.zip`
- Public score: **1158.0745556751**
- Previous confirmed v26 score: **1157.9736407889**
- Public delta vs v26: **+0.1009148862**

## Final Overlay

The final prediction is adjusted after all existing v26 overlays:

```python
prediction = apply_v25_postbreak_anchor_overlay(prediction, frame)
prediction = apply_trackman_asof_gate_overlay(
    prediction,
    frame,
    float(ensemble["global_rate"]),
)
```

The gate is row-local and uses only frozen training artifacts in `model/` plus the current test row:

```python
gate = 0.03 * tm_conf * pressure
```

- Apply domain: regular-season `R_ANCHOR`
- TrackMan confidence: `trackman_pitcher_profiles.csv` fields `tm_linked`, `tm_pitcher_n`
- ASOF prior: pitcher/batter success prior with `alpha=200`, `batter_weight=0.25`
- Pressure scale: `1.25` for three-ball or two-strike counts, `0.75` otherwise
- Gate cap: `0.03`

## Local Audit Notes

`0819_trackman_asof_shrink_gate_audit.py` screened the idea on temporal transitions:

| candidate | min_gain | mean_gain | note |
|---|---:|---:|---|
| `R_ANCHOR eta=0.01` boost | `+0.028589` | `+1.045667` | submitted as 0819_1, Public 1158.0201910845 |
| `R_ANCHOR eta=0.03` boost | `+0.058987` | `+3.096241` | submitted as 0819_3, Public 1158.0745556751 |
| `R_ANCHOR eta=0.03` flat pressure | `-0.140198` | `+3.293108` | rejected before submission |

## Rebuild

The branch now bakes this overlay into `src/package_v26_anchor_weight_probe.py`.

```powershell
python -m src.package_v26_anchor_weight_probe `
  --project . `
  --parent submit_v25.zip `
  --output 0819_jy_tmgate03.zip
```

The package writer uses forward-slash ZIP paths and then runs `verify_package`, so the required DACON top-level layout remains:

- `model/`
- `script.py`
- `requirements.txt`

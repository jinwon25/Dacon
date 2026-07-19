from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions" / "blend_best_crossg3_traj_meta_finesweep.csv"
OUT = ROOT / "submissions"


def main() -> None:
    base = pd.read_csv(BASE, encoding="utf-8-sig")
    specs = [
        ("g12_g3m1030p175", {"kpx_group_1": (0.99, 500.0), "kpx_group_2": (0.988, 450.0), "kpx_group_3": (1.03, 175.0)}),
        ("g12_g3m1055p300", {"kpx_group_1": (0.99, 500.0), "kpx_group_2": (0.988, 450.0), "kpx_group_3": (1.055, 300.0)}),
        ("g12_g3m1100p250", {"kpx_group_1": (0.99, 500.0), "kpx_group_2": (0.988, 450.0), "kpx_group_3": (1.10, 250.0)}),
        ("g1_g3m1055p300", {"kpx_group_1": (0.99, 500.0), "kpx_group_3": (1.055, 300.0)}),
        ("g2_g3m1055p300", {"kpx_group_2": (0.988, 450.0), "kpx_group_3": (1.055, 300.0)}),
    ]
    records = []
    for name, transforms in specs:
        out = base.copy()
        for group, (scale, offset) in transforms.items():
            out[group] = np.clip(out[group].to_numpy(dtype=float) * scale + offset, 0.0, CAPACITY_KWH[group])
        path = OUT / f"blend_{name}.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        records.append({"file": str(path.relative_to(ROOT)), "transforms": transforms, "means": {g: float(out[g].mean()) for g in CAPACITY_KWH}})

    # Seasonal group-3 policy selected on contiguous Q1/Q2 development:
    # preserve the public winner in Jan/Jun/Dec and apply 1.05*x+400 to the
    # remaining months.  This is deliberately emitted as a separate candidate
    # because its public transfer risk is higher than the global affine policy.
    seasonal = base.copy()
    months = pd.to_datetime(seasonal["forecast_kst_dtm"]).dt.month
    mask = ~months.isin([1, 6, 12])
    seasonal.loc[mask, "kpx_group_3"] = np.clip(
        seasonal.loc[mask, "kpx_group_3"].to_numpy(dtype=float) * 1.05 + 400.0,
        0.0,
        CAPACITY_KWH["kpx_group_3"],
    )
    path = OUT / "blend_g12_g3_season105p400_skip1_6_12.csv"
    seasonal["kpx_group_1"] = np.clip(seasonal["kpx_group_1"] * 0.99 + 500.0, 0.0, CAPACITY_KWH["kpx_group_1"])
    seasonal["kpx_group_2"] = np.clip(seasonal["kpx_group_2"] * 0.988 + 450.0, 0.0, CAPACITY_KWH["kpx_group_2"])
    seasonal.to_csv(path, index=False, encoding="utf-8-sig")
    records.append({"file": str(path.relative_to(ROOT)), "transforms": {"kpx_group_1": [0.99, 500.0], "kpx_group_2": [0.988, 450.0], "kpx_group_3": {"scale": 1.05, "offset": 400.0, "months": "2,3,4,5,7,8,9,10,11"}}, "means": {g: float(seasonal[g].mean()) for g in CAPACITY_KWH}})

    # Lower-risk SCADA-transfer month mask combined with the issue-block-safe
    # group-1/group-2 affine policy. This is the primary composite probe.
    monthmask = base.copy()
    monthmask["kpx_group_1"] = np.clip(monthmask["kpx_group_1"] * 0.99 + 500.0, 0.0, CAPACITY_KWH["kpx_group_1"])
    monthmask["kpx_group_2"] = np.clip(monthmask["kpx_group_2"] * 0.988 + 450.0, 0.0, CAPACITY_KWH["kpx_group_2"])
    monthmask_months = pd.to_datetime(monthmask["forecast_kst_dtm"]).dt.month
    monthmask_keep = ~monthmask_months.isin([1, 6, 12])
    monthmask.loc[monthmask_keep, "kpx_group_3"] = np.clip(
        monthmask.loc[monthmask_keep, "kpx_group_3"].to_numpy(dtype=float) * 1.055 + 300.0,
        0.0,
        CAPACITY_KWH["kpx_group_3"],
    )
    path = OUT / "blend_g12_g3_monthmask1055p300.csv"
    monthmask.to_csv(path, index=False, encoding="utf-8-sig")
    records.append({"file": str(path.relative_to(ROOT)), "transforms": {"kpx_group_1": [0.99, 500.0], "kpx_group_2": [0.988, 450.0], "kpx_group_3": {"scale": 1.055, "offset": 300.0, "months": "2,3,4,5,7,8,9,10,11"}}, "means": {g: float(monthmask[g].mean()) for g in CAPACITY_KWH}})
    report = ROOT / "artifacts_final" / "calibration" / "affine_composites.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"base": str(BASE.relative_to(ROOT)), "candidates": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

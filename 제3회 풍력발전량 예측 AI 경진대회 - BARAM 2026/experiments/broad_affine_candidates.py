from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions" / "blend_best_crossg3_traj_meta_finesweep.csv"
OOF = ROOT / "artifacts_final" / "lineage" / "exact_driver_oof.npz"
OUT = ROOT / "submissions"
REPORT = ROOT / "artifacts_final" / "calibration" / "broad_affine_candidates.json"


def _candidate(base: pd.DataFrame, group: str, scale: float, offset: float) -> pd.DataFrame:
    out = base.copy()
    cap = CAPACITY_KWH[group]
    out[group] = np.clip(out[group].to_numpy(dtype=float) * scale + offset, 0.0, cap)
    return out


def main() -> None:
    base = pd.read_csv(BASE, encoding="utf-8-sig")
    z = np.load(OOF, allow_pickle=True)
    idx = pd.DatetimeIndex(pd.to_datetime(z["kpx_group_3__valid_index_ns"]))
    truth = z["kpx_group_3__valid_truth"].astype(float)
    control = z["kpx_group_3__over115"].astype(float)
    rows: list[dict[str, object]] = []
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts_final" / "calibration").mkdir(parents=True, exist_ok=True)

    # The existing public winner contains a bounded cross-group/meta adjustment.
    # Apply only a small, explicit affine family to its group-3 output.
    for scale, offset in (
        (1.05, -250.0),
        (1.08, -350.0),
        (1.10, -450.0),
        (1.12, -500.0),
        (1.15, -600.0),
        (1.18, -650.0),
        (1.20, -700.0),
        (1.22, -750.0),
    ):
        name = f"blend_g3_affine_s{scale:.2f}_o{int(offset):+d}.csv".replace("+", "p").replace("-", "m")
        path = OUT / name
        out = _candidate(base, "kpx_group_3", scale, offset)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        local = evaluate_group(truth, np.clip(control * scale + offset, 0.0, CAPACITY_KWH["kpx_group_3"]), CAPACITY_KWH["kpx_group_3"])
        rows.append({"file": str(path.relative_to(ROOT)), "group": "kpx_group_3", "scale": scale, "offset": offset, "oof_control": local.to_dict(), "test_mean": float(out["kpx_group_3"].mean())})

    # Isolated group-1 and group-2 affine probes are kept separate so their
    # public transfer can be audited without mixing group-3 changes.
    for group, scale, offset in (("kpx_group_1", 0.99, 500.0), ("kpx_group_2", 0.988, 450.0)):
        name = f"blend_{group[-1]}_affine_s{scale:.3f}_op{int(offset)}.csv"
        path = OUT / name
        out = _candidate(base, group, scale, offset)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append({"file": str(path.relative_to(ROOT)), "group": group, "scale": scale, "offset": offset, "oof_control": None, "test_mean": float(out[group].mean())})

    REPORT.write_text(json.dumps({"base": str(BASE.relative_to(ROOT)), "oof_source": str(OOF.relative_to(ROOT)), "candidates": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

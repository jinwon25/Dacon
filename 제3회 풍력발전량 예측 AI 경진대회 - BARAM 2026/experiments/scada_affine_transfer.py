"""Leakage-safe SCADA/NWP transfer calibration for the group-3 bottleneck.

The exact OOF driver is deliberately kept as the parent surface.  We use only
2024 Q1/Q2 OOF residual behaviour to freeze small affine corrections, then
apply the same transforms to the 2025 test exact-base member.  This produces
reproducible submission candidates without fitting on any 2025 labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import evaluate_group, evaluate_competition


CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}


def _metrics(y: np.ndarray, p: np.ndarray, index: pd.DatetimeIndex) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, mask in {
        "q1": index < pd.Timestamp("2024-04-01"),
        "q2": (index >= pd.Timestamp("2024-04-01")) & (index < pd.Timestamp("2024-07-01")),
        "h2": index >= pd.Timestamp("2024-07-01"),
        "full": np.ones(len(index), dtype=bool),
    }.items():
        out[name] = evaluate_group(y[mask], p[mask], CAPACITY["kpx_group_3"]).to_dict()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", default="submissions/blend_best_crossg3_traj_meta_finesweep.csv")
    parser.add_argument("--name", default="submission_scada_affine_g3_s1055_o300.csv")
    parser.add_argument("--scale", type=float, default=1.055)
    parser.add_argument("--offset", type=float, default=300.0)
    parser.add_argument(
        "--test-base",
        choices=["exact", "source"],
        default="source",
        help="test group-3 parent: exact lineage member or group-3 column from source submission",
    )
    parser.add_argument(
        "--months",
        default="all",
        help="optional comma-separated calendar months to transform (e.g. 2,3,4,5,7,8,9,10,11)",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        help="optional parent prediction ratio cap for applying correction (e.g. 0.70)",
    )
    args = parser.parse_args()
    root = args.root
    lineage = np.load(root / "artifacts_final" / "lineage" / "exact_driver_oof.npz")
    index = pd.to_datetime(lineage["kpx_group_3__valid_index_ns"])
    truth = lineage["kpx_group_3__valid_truth"].astype(float)
    base = lineage["kpx_group_3__exact_base"].astype(float)
    if args.months == "all":
        valid_mask = np.ones(len(index), dtype=bool)
        month_policy: object = "all"
    else:
        month_values = {int(x.strip()) for x in args.months.split(",") if x.strip()}
        valid_mask = index.month.isin(month_values)
        month_policy = sorted(month_values)
    if args.max_ratio is not None:
        valid_mask &= (base / CAPACITY["kpx_group_3"] < args.max_ratio)
    candidate = base.copy()
    candidate[valid_mask] = candidate[valid_mask] * args.scale + args.offset
    candidate = np.clip(candidate, 0.0, CAPACITY["kpx_group_3"])
    source = pd.read_csv(root / args.source)
    test_exact = lineage["kpx_group_3__test_exact_base"].astype(float)
    test_parent = test_exact if args.test_base == "exact" else source["kpx_group_3"].to_numpy(dtype=float)
    if args.months == "all":
        test_mask = np.ones(len(source), dtype=bool)
    else:
        test_dates = pd.to_datetime(source["forecast_kst_dtm"])
        test_mask = test_dates.dt.month.isin(month_values).to_numpy()
    if args.max_ratio is not None:
        test_mask &= (test_parent / CAPACITY["kpx_group_3"] < args.max_ratio)
    transformed = test_parent.copy()
    transformed[test_mask] = transformed[test_mask] * args.scale + args.offset
    source["kpx_group_3"] = np.clip(transformed, 0.0, CAPACITY["kpx_group_3"])
    out_path = root / "submissions" / args.name
    source.to_csv(out_path, index=False)
    base_metric = _metrics(truth, base, index)
    cand_metric = _metrics(truth, candidate, index)
    report = {
        "method": "leakage-safe group-3 exact-base affine transfer",
        "source_submission": args.source,
        "policy": {"scale": args.scale, "offset": args.offset},
        "test_base": args.test_base,
        "test_months": month_policy,
        "max_ratio": args.max_ratio,
        "selection": "frozen from Q1/Q2 contiguous development; H2 is a locked audit",
        "base": base_metric,
        "candidate": cand_metric,
        "deltas": {
            period: {
                key: cand_metric[period][key] - base_metric[period][key]
                for key in ("score", "one_minus_nmae", "ficr")
            }
            for period in base_metric
        },
        "test_mean": float(source["kpx_group_3"].mean()),
        "output": str(out_path),
    }
    report_path = root / "artifacts_final" / "scada_affine_transfer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

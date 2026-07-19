from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.group12_settlement_calibration import apply_affine
from src.metrics import CAPACITY_KWH, evaluate_group


H2_START = pd.Timestamp("2024-07-01 01:00:00")
GROUP12_POLICIES = {
    "kpx_group_1": (1.026, 400.0),
    "kpx_group_2": (0.988, 450.0),
}


def _delta(truth: np.ndarray, base: np.ndarray, candidate: np.ndarray, capacity: float) -> dict[str, float]:
    before = evaluate_group(truth, base, capacity)
    after = evaluate_group(truth, candidate, capacity)
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def run(
    driver_path: Path,
    meta_path: Path,
    threshold_candidate_path: Path,
    output_path: Path,
    report_path: Path,
    n_bootstrap: int,
) -> dict[str, object]:
    driver = np.load(driver_path)
    meta = np.load(meta_path)
    output = pd.read_csv(threshold_candidate_path, encoding="utf-8-sig")
    records = []
    group_deltas: dict[str, dict[str, float]] = {}

    for group, capacity in CAPACITY_KWH.items():
        index = pd.DatetimeIndex(pd.to_datetime(driver[f"{group}__valid_index_ns"]))
        locked = index >= H2_START
        truth = driver[f"{group}__valid_truth"].astype(float)
        if group == "kpx_group_3":
            base = meta["valid_candidate"].astype(float)
            candidate = np.where(base >= 0.10 * capacity, np.clip(base + 575.0, 0.0, capacity), base)
        else:
            base = driver[f"{group}__exact_base"].astype(float)
            scale, offset = GROUP12_POLICIES[group]
            candidate = apply_affine(base, capacity, scale, offset)
            output[group] = apply_affine(output[group].to_numpy(dtype=float), capacity, scale, offset)
        group_deltas[group] = _delta(truth[locked], base[locked], candidate[locked], capacity)
        records.append((index[locked], truth[locked], base[locked], candidate[locked], capacity))

    days = records[0][0].normalize().unique()
    positions = [
        {day: np.flatnonzero(record[0].normalize() == day) for day in days}
        for record in records
    ]
    rng = np.random.default_rng(7_317)
    bootstrap = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(days, size=len(days), replace=True)
        deltas = []
        for record_i, (_, truth, base, candidate, capacity) in enumerate(records):
            rows = np.concatenate([positions[record_i][day] for day in sampled])
            deltas.append(_delta(truth[rows], base[rows], candidate[rows], capacity)["score"])
        bootstrap.append(float(np.mean(deltas)))
    values = np.asarray(bootstrap)
    macro_components = {
        key: float(np.mean([row[key] for row in group_deltas.values()]))
        for key in ("score", "one_minus_nmae", "ficr")
    }
    report: dict[str, object] = {
        "method": "composition of independently selected group1/2 affine and group3 threshold policies",
        "group_deltas": group_deltas,
        "locked_h2_macro_delta": macro_components,
        "bootstrap": {
            "positive_fraction": float((values > 0.0).mean()),
            "q025": float(np.quantile(values, 0.025)),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
            "q975": float(np.quantile(values, 0.975)),
        },
        "estimated_public_score": 0.6416553726 + macro_components["score"],
        "classification": "controlled_probe",
        "auto_submit_eligible": False,
        "auto_submit_blockers": [
            "bootstrap q05 is below the service policy threshold",
            "changed-row coverage exceeds the service policy maximum",
        ],
        "output": str(output_path),
    }
    if not all(np.isfinite(output[group]).all() for group in CAPACITY_KWH):
        raise ValueError("Composite candidate contains non-finite predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument("--meta", default="artifacts_final/meta_gate/meta_gate_cache.npz")
    parser.add_argument("--threshold-candidate", default="submissions/blend_best_meta_g3_thr10_off575.csv")
    parser.add_argument("--output", default="submissions/blend_best_settlement_composite.csv")
    parser.add_argument("--report", default="artifacts_final/calibration/settlement_composite_report.json")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.driver), Path(args.meta), Path(args.threshold_candidate), Path(args.output), Path(args.report), args.n_bootstrap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

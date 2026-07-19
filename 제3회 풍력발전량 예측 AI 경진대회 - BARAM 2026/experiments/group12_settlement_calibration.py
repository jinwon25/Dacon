from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, evaluate_group


GROUPS = ("kpx_group_1", "kpx_group_2")
Q1_END = pd.Timestamp("2024-04-01")
H1_END = pd.Timestamp("2024-07-01 01:00:00")


def apply_affine(prediction: np.ndarray, capacity: float, scale: float, offset: float) -> np.ndarray:
    return np.clip(np.asarray(prediction, dtype=float) * scale + offset, 0.0, capacity)


def select_affine(
    truth: np.ndarray,
    prediction: np.ndarray,
    capacity: float,
    train_mask: np.ndarray,
) -> tuple[float, float]:
    """Select once on Q1; the rest of 2024 remains untouched validation."""
    best: tuple[float, float, float] | None = None
    for scale in np.arange(0.96, 1.0401, 0.002):
        for offset in np.arange(-500.0, 500.1, 50.0):
            candidate = apply_affine(prediction, capacity, scale, offset)
            metric = evaluate_group(truth[train_mask], candidate[train_mask], capacity)
            row = (metric.score, float(scale), float(offset))
            if best is None or row[0] > best[0]:
                best = row
    if best is None:
        raise RuntimeError("No affine policy was evaluated")
    return best[1], best[2]


def comparison(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    capacity: float,
    mask: np.ndarray,
) -> dict[str, object]:
    before = evaluate_group(truth[mask], base[mask], capacity)
    after = evaluate_group(truth[mask], candidate[mask], capacity)
    return {
        "base": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": {
            "score": after.score - before.score,
            "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
            "ficr": after.ficr - before.ficr,
        },
    }


def bootstrap_days(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, pd.DatetimeIndex]],
    n_bootstrap: int,
    seed: int = 9127,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    days = records[0][4].normalize().unique()
    positions_by_record = [
        {day: np.flatnonzero(index.normalize() == day) for day in days}
        for _, _, _, _, index in records
    ]
    values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(days, size=len(days), replace=True)
        deltas = []
        for record_i, (truth, base, candidate, capacity, _) in enumerate(records):
            positions = positions_by_record[record_i]
            rows = np.concatenate([positions[day] for day in sampled])
            deltas.append(
                evaluate_group(truth[rows], candidate[rows], capacity).score
                - evaluate_group(truth[rows], base[rows], capacity).score
            )
        values.append(float(np.mean(deltas)))
    array = np.asarray(values)
    return {
        "positive_fraction": float((array > 0.0).mean()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def run(
    driver_path: Path,
    submission_path: Path,
    output_path: Path,
    report_path: Path,
    n_bootstrap: int,
) -> dict[str, object]:
    cache = np.load(driver_path)
    submission = pd.read_csv(submission_path, encoding="utf-8-sig")
    report: dict[str, object] = {"method": "Q1-selected group-1/2 settlement affine calibration", "groups": {}}
    locked_records = []
    test_updates: dict[str, np.ndarray] = {}
    all_pass = True

    for group in GROUPS:
        capacity = CAPACITY_KWH[group]
        index = pd.DatetimeIndex(pd.to_datetime(cache[f"{group}__valid_index_ns"]))
        truth = cache[f"{group}__valid_truth"].astype(float)
        base = cache[f"{group}__exact_base"].astype(float)
        q1 = index < Q1_END
        q2 = (index >= Q1_END) & (index < H1_END)
        h2 = index >= H1_END
        scale, offset = select_affine(truth, base, capacity, q1)
        candidate = apply_affine(base, capacity, scale, offset)
        q2_result = comparison(truth, base, candidate, capacity, q2)
        h2_result = comparison(truth, base, candidate, capacity, h2)
        parity = np.allclose(
            submission[group].to_numpy(dtype=float),
            cache[f"{group}__test_exact_base"].astype(float),
            atol=0.05,
            rtol=0.0,
        )
        passed = bool(
            parity
            and q2_result["delta"]["score"] > 0.0
            and h2_result["delta"]["score"] > 0.0
        )
        all_pass &= passed
        test_updates[group] = apply_affine(
            submission[group].to_numpy(dtype=float), capacity, scale, offset
        )
        h2_index = index[h2]
        locked_records.append((truth[h2], base[h2], candidate[h2], capacity, h2_index))
        report["groups"][group] = {
            "policy": {"scale": scale, "offset": offset},
            "q2": q2_result,
            "locked_h2": h2_result,
            "test_lineage_parity": parity,
            "passed": passed,
        }

    bootstrap = bootstrap_days(locked_records, n_bootstrap)
    report["locked_h2_macro_bootstrap"] = bootstrap
    report["expected_macro_score_delta"] = float(
        np.mean([report["groups"][g]["locked_h2"]["delta"]["score"] for g in GROUPS])
        * (len(GROUPS) / 3.0)
    )
    report["candidate_created"] = bool(all_pass and bootstrap["positive_fraction"] >= 0.80)
    if report["candidate_created"]:
        output = submission.copy()
        for group, values in test_updates.items():
            output[group] = values
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        report["output"] = str(output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument("--submission", default="submissions/blend_best_crossg3_traj_meta25_p55.csv")
    parser.add_argument("--output", default="submissions/blend_best_meta_g12_settlement.csv")
    parser.add_argument("--report", default="artifacts_final/calibration/group12_settlement_report.json")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.driver), Path(args.submission), Path(args.output), Path(args.report), args.n_bootstrap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

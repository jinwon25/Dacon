from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.group12_settlement_calibration import apply_affine, comparison
from src.metrics import CAPACITY_KWH, evaluate_group


GROUP = "kpx_group_2"
CAPACITY = CAPACITY_KWH[GROUP]
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")


@dataclass(frozen=True)
class Policy:
    scale: float
    offset: float

    def to_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "offset": self.offset}


def policy_grid() -> tuple[Policy, ...]:
    return tuple(
        Policy(float(round(scale, 3)), float(offset))
        for scale in np.arange(0.98, 1.0201, 0.002)
        for offset in np.arange(-200.0, 500.1, 50.0)
    )


def _record(
    truth: np.ndarray,
    base: np.ndarray,
    index: pd.DatetimeIndex,
    policy: Policy,
) -> dict[str, object]:
    candidate = apply_affine(base, CAPACITY, policy.scale, policy.offset)
    q1 = np.asarray(index < Q2_START)
    q2 = np.asarray((index >= Q2_START) & (index < H2_START))
    q1_result = comparison(truth, base, candidate, CAPACITY, q1)
    q2_result = comparison(truth, base, candidate, CAPACITY, q2)
    movement = np.abs(candidate - base)
    return {
        "policy": policy.to_dict(),
        "q1": q1_result,
        "q2": q2_result,
        "robust_score_delta": float(
            min(q1_result["delta"]["score"], q2_result["delta"]["score"])
        ),
        "mean_absolute_movement_kwh": float(movement.mean()),
    }


def select_development(records: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = []
    for record in records:
        deltas = (record["q1"]["delta"], record["q2"]["delta"])
        if all(
            delta["score"] > 0.0
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            for delta in deltas
        ):
            eligible.append(record)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            record["robust_score_delta"],
            np.mean(
                [record["q1"]["delta"]["score"], record["q2"]["delta"]["score"]]
            ),
            -record["mean_absolute_movement_kwh"],
        ),
    )


def bootstrap_days(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    mask: np.ndarray,
    n_bootstrap: int,
) -> dict[str, float]:
    masked_index = index[mask]
    masked_truth = truth[mask]
    masked_base = base[mask]
    masked_candidate = candidate[mask]
    days = masked_index.normalize().unique()
    positions = {
        day: np.flatnonzero(masked_index.normalize() == day) for day in days
    }
    rng = np.random.default_rng(20260718)
    values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([positions[day] for day in sampled])
        before = evaluate_group(masked_truth[rows], masked_base[rows], CAPACITY)
        after = evaluate_group(masked_truth[rows], masked_candidate[rows], CAPACITY)
        values.append(after.score - before.score)
    array = np.asarray(values, dtype=float)
    return {
        "n_bootstrap": n_bootstrap,
        "positive_fraction": float((array > 0.0).mean()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--driver", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--base-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--output", default="submissions/blend_best_g2_component_safe_affine.csv"
    )
    parser.add_argument(
        "--report",
        default="artifacts_final/calibration/group2_component_safe_report.json",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    cache = np.load(args.driver)
    index = pd.DatetimeIndex(pd.to_datetime(cache[f"{GROUP}__valid_index_ns"]))
    truth = cache[f"{GROUP}__valid_truth"].astype(float)
    base = cache[f"{GROUP}__exact_base"].astype(float)
    records = [_record(truth, base, index, policy) for policy in policy_grid()]
    selected = select_development(records)

    locked = None
    monthly: dict[str, dict[str, object]] = {}
    bootstrap = None
    qualified = False
    selected_policy = None
    if selected is not None:
        selected_policy = Policy(**selected["policy"])
        candidate = apply_affine(
            base, CAPACITY, selected_policy.scale, selected_policy.offset
        )
        h2 = np.asarray(index >= H2_START)
        locked = comparison(truth, base, candidate, CAPACITY, h2)
        for month in range(7, 13):
            month_mask = h2 & (index.month == month)
            monthly[str(month)] = comparison(
                truth, base, candidate, CAPACITY, month_mask
            )
        bootstrap = bootstrap_days(
            truth, base, candidate, index, h2, args.bootstrap
        )
        delta = locked["delta"]
        qualified = bool(
            delta["score"] > 0.0
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            and sum(
                value["delta"]["score"] > 0.0 for value in monthly.values()
            )
            >= 4
            and bootstrap["positive_fraction"] >= 0.90
            and bootstrap["q05"] >= -0.00025
        )

    submission = None
    if qualified and selected_policy is not None:
        source = pd.read_csv(args.base_submission, encoding="utf-8-sig")
        test_base = cache[f"{GROUP}__test_exact_base"].astype(float)
        source_values = source[GROUP].to_numpy(dtype=float)
        parity_error = float(np.max(np.abs(source_values - test_base)))
        if parity_error > 0.05:
            raise ValueError(
                f"Public-best group-2 lineage parity failed: {parity_error:.6f} kWh"
            )
        output = source.copy()
        output[GROUP] = apply_affine(
            source_values, CAPACITY, selected_policy.scale, selected_policy.offset
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        payload = output_path.read_bytes()
        movement = output[GROUP].to_numpy(dtype=float) - source_values
        submission = {
            "output": str(output_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": int(len(output)),
            "changed_rows": int((np.abs(movement) > 1e-9).sum()),
            "mean_absolute_movement_kwh": float(np.abs(movement).mean()),
            "p95_absolute_movement_kwh": float(np.quantile(np.abs(movement), 0.95)),
            "groups_1_3_unchanged": bool(
                np.array_equal(
                    output["kpx_group_1"].to_numpy(),
                    source["kpx_group_1"].to_numpy(),
                )
                and np.array_equal(
                    output["kpx_group_3"].to_numpy(),
                    source["kpx_group_3"].to_numpy(),
                )
            ),
            "lineage_parity_max_error_kwh": parity_error,
        }

    report = {
        "method": "component-safe group-2 affine calibration",
        "selection_contract": {
            "development": "select only policies positive in score, 1-NMAE, and FICR on both Q1 and Q2",
            "locked": "evaluate selected policy once on H2",
            "bootstrap_q05_floor": -0.00025,
            "policy_count": len(policy_grid()),
            "source_submission": args.base_submission,
        },
        "development_selected": selected,
        "development_top": sorted(
            records, key=lambda item: item["robust_score_delta"], reverse=True
        )[:10],
        "locked_h2": locked,
        "locked_monthly": monthly,
        "locked_day_bootstrap": bootstrap,
        "qualified": qualified,
        "submission": submission,
        "decision": (
            "create isolated group-2 submission candidate"
            if qualified
            else "reject; no submission candidate"
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

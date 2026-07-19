from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.group12_settlement_calibration import apply_affine, comparison
from experiments.group2_component_safe_calibration import (
    CAPACITY,
    GROUP,
    H2_START,
    Q2_START,
    bootstrap_days,
)


SCALE = 0.996
OFFSET = 50.0


@dataclass(frozen=True)
class Policy:
    minimum_ratio: float
    maximum_ratio: float


def policies() -> tuple[Policy, ...]:
    edges = tuple(float(round(value, 2)) for value in np.arange(0.10, 1.001, 0.05))
    return tuple(
        Policy(low, high)
        for low in edges[:-1]
        for high in edges[1:]
        if 0.05 <= high - low <= 0.30
    )


def apply_selective(base: np.ndarray, policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=float)
    ratio = base / CAPACITY
    gate = (ratio >= policy.minimum_ratio) & (ratio < policy.maximum_ratio)
    affine = apply_affine(base, CAPACITY, SCALE, OFFSET)
    candidate = base.copy()
    candidate[gate] = affine[gate]
    return candidate, gate


def period_record(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    gate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    return {
        "coverage": float((gate & mask).sum() / max(int(mask.sum()), 1)),
        "metrics": comparison(truth, base, candidate, CAPACITY, mask),
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
        "--output", default="submissions/blend_best_g2_selective_affine.csv"
    )
    parser.add_argument(
        "--report", default="artifacts_final/calibration/group2_selective_affine_report.json"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    cache = np.load(args.driver)
    index = pd.DatetimeIndex(pd.to_datetime(cache[f"{GROUP}__valid_index_ns"]))
    truth = cache[f"{GROUP}__valid_truth"].astype(float)
    base = cache[f"{GROUP}__exact_base"].astype(float)
    q1 = np.asarray(index < Q2_START)
    q2 = np.asarray((index >= Q2_START) & (index < H2_START))
    h2 = np.asarray(index >= H2_START)

    development = []
    for policy in policies():
        candidate, gate = apply_selective(base, policy)
        q1_record = period_record(truth, base, candidate, gate, q1)
        q2_record = period_record(truth, base, candidate, gate, q2)
        deltas = (q1_record["metrics"]["delta"], q2_record["metrics"]["delta"])
        passed = bool(
            q1_record["coverage"] <= 0.25
            and q2_record["coverage"] <= 0.25
            and all(
                delta["score"] > 0.0
                and delta["one_minus_nmae"] > 0.0
                and delta["ficr"] > 0.0
                for delta in deltas
            )
        )
        development.append(
            {
                "policy": asdict(policy),
                "q1": q1_record,
                "q2": q2_record,
                "robust_score_delta": float(
                    min(delta["score"] for delta in deltas)
                ),
                "passed": passed,
            }
        )
    eligible = [record for record in development if record["passed"]]
    selected = (
        max(
            eligible,
            key=lambda record: (
                record["robust_score_delta"],
                record["q1"]["metrics"]["delta"]["score"]
                + record["q2"]["metrics"]["delta"]["score"],
                -record["q1"]["coverage"] - record["q2"]["coverage"],
            ),
        )
        if eligible
        else None
    )

    locked = None
    monthly = {}
    bootstrap = None
    qualified = False
    candidate = None
    gate = None
    if selected is not None:
        policy = Policy(**selected["policy"])
        candidate, gate = apply_selective(base, policy)
        locked = period_record(truth, base, candidate, gate, h2)
        for month in range(7, 13):
            mask = h2 & (index.month == month)
            monthly[str(month)] = period_record(
                truth, base, candidate, gate, mask
            )
        bootstrap = bootstrap_days(
            truth, base, candidate, index, h2, args.bootstrap
        )
        delta = locked["metrics"]["delta"]
        qualified = bool(
            locked["coverage"] <= 0.25
            and delta["score"] >= 0.00015
            and delta["one_minus_nmae"] > 0.0
            and delta["ficr"] > 0.0
            and sum(
                item["metrics"]["delta"]["score"] > 0.0
                for item in monthly.values()
            )
            >= 3
            and bootstrap["positive_fraction"] >= 0.80
            and bootstrap["q05"] >= -0.00025
        )

    submission = None
    if qualified and selected is not None:
        policy = Policy(**selected["policy"])
        source = pd.read_csv(args.base_submission, encoding="utf-8-sig")
        source_values = source[GROUP].to_numpy(dtype=float)
        test_base = cache[f"{GROUP}__test_exact_base"].astype(float)
        parity = float(np.max(np.abs(source_values - test_base)))
        if parity > 0.05:
            raise ValueError("Public-best group-2 lineage parity failed")
        test_candidate, test_gate = apply_selective(source_values, policy)
        output = source.copy()
        output[GROUP] = test_candidate
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        payload = output_path.read_bytes()
        submission = {
            "output": str(output_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": int(len(output)),
            "changed_rows": int(test_gate.sum()),
            "changed_ratio": float(test_gate.mean()),
            "mean_absolute_movement_kwh": float(
                np.abs(test_candidate - source_values).mean()
            ),
            "groups_1_3_unchanged": bool(
                np.array_equal(output["kpx_group_1"], source["kpx_group_1"])
                and np.array_equal(output["kpx_group_3"], source["kpx_group_3"])
            ),
        }

    report = {
        "method": "bounded selective child of component-safe group-2 affine",
        "fixed_affine": {"scale": SCALE, "offset": OFFSET},
        "selection_contract": {
            "development": "Q1/Q2 all-component positive contiguous prediction-ratio interval",
            "locked": "selected interval opened once on H2",
            "maximum_changed_ratio": 0.25,
            "policy_count": len(policies()),
        },
        "development_selected": selected,
        "development_top": sorted(
            development,
            key=lambda record: record["robust_score_delta"],
            reverse=True,
        )[:10],
        "locked_h2": locked,
        "locked_monthly": monthly,
        "locked_bootstrap": bootstrap,
        "qualified": qualified,
        "submission": submission,
        "decision": (
            "create service-bounded group-2 candidate"
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

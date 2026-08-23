"""Audit leaderboard-inferred stage weights against frozen temporal OOF axes.

DACON explicitly allows leaderboard scores to select fixed model weights. This
module asks whether the public gains of the v19 -> v20 -> v21 -> v22 ladder can
identify a better stage scale, then rejects that scale unless it also beats v22
on every pre-registered local time axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss
from src.v24_semantic_signal_screen import _v22


PUBLIC_SCORES = {
    "v19": 1144.1518063753,
    "v20": 1151.4724287190,
    "v21": 1151.5138356157,
    "v22": 1153.0436023798,
}


def score_curvature(target: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Return Q for score gain ``linear @ x - x.T @ Q @ x``."""

    target = np.asarray(target, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    reference = float(target.mean() * (1.0 - target.mean()))
    if reference <= 0.0:
        raise ValueError("target variance must be positive")
    return 100000.0 * (deltas.T @ deltas) / (len(target) * reference)


def infer_linear_terms(
    curvature: np.ndarray, sequential_gains: np.ndarray
) -> np.ndarray:
    """Infer linear terms from gains observed after adding stages in order."""

    q = np.asarray(curvature, dtype=np.float64)
    gains = np.asarray(sequential_gains, dtype=np.float64)
    if q.shape != (len(gains), len(gains)):
        raise ValueError("curvature and gains have incompatible shapes")
    linear = np.empty(len(gains), dtype=np.float64)
    for index in range(len(gains)):
        linear[index] = (
            gains[index]
            + q[index, index]
            + 2.0 * q[index, :index].sum()
        )
    return linear


def optimal_scales(curvature: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Unconstrained optimum for the inferred public quadratic."""

    return 0.5 * np.linalg.solve(curvature, linear)


def _rows_for_axis(train: pd.DataFrame, axis: str) -> pd.DataFrame:
    if axis == "y2023_early_to_late":
        mask = train["season"].eq(2023) & train["game_month"].ge(8)
    elif axis == "y2023_to_y2024":
        mask = train["season"].eq(2024)
    elif axis == "y2024_early_to_late":
        mask = train["season"].eq(2024) & train["game_month"].ge(8)
    else:
        raise ValueError(f"unknown axis: {axis}")
    return train.loc[mask].reset_index(drop=True)


def _axis_audit(
    project: Path,
    train: pd.DataFrame,
    axis: str,
    scales: np.ndarray,
) -> dict[str, object]:
    path = project / "artifacts/champion_oof_20260817_01" / f"{axis}.npz"
    with np.load(path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v19 = saved["v19"].astype(np.float64)
        v20 = saved["v20"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    rows = _rows_for_axis(train, axis)
    if len(rows) != len(target):
        raise ValueError(f"row alignment failed for {axis}")
    v22 = _v22(rows, v21, domain)
    deltas = np.column_stack((v20 - v19, v21 - v20, v22 - v21))
    candidate = np.clip(v19 + deltas @ scales, 0.001, 0.999)
    domain_gains = {}
    for name in ("R_CORE", "R_ANCHOR", "F"):
        mask = domain == name
        domain_gains[name] = float(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], v22[mask])
        )
    return {
        "axis": axis,
        "rows": int(len(target)),
        "gain_vs_v22": float(_bss(target, candidate) - _bss(target, v22)),
        "domain_gains_vs_v22": domain_gains,
        "mean_shift_vs_v22": float(np.mean(candidate - v22)),
        "mean_abs_shift_vs_v22": float(np.mean(np.abs(candidate - v22))),
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project / "data/train.csv", low_memory=False)

    outer_axis = "y2023_to_y2024"
    outer_path = (
        project / "artifacts/champion_oof_20260817_01" / f"{outer_axis}.npz"
    )
    with np.load(outer_path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v19 = saved["v19"].astype(np.float64)
        v20 = saved["v20"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    rows = _rows_for_axis(train, outer_axis)
    v22 = _v22(rows, v21, domain)
    deltas = np.column_stack((v20 - v19, v21 - v20, v22 - v21))
    curvature = score_curvature(target, deltas)
    gains = np.asarray(
        [
            PUBLIC_SCORES["v20"] - PUBLIC_SCORES["v19"],
            PUBLIC_SCORES["v21"] - PUBLIC_SCORES["v20"],
            PUBLIC_SCORES["v22"] - PUBLIC_SCORES["v21"],
        ],
        dtype=np.float64,
    )
    linear = infer_linear_terms(curvature, gains)
    scales = optimal_scales(curvature, linear)
    public_gain_from_v19 = float(linear @ scales - scales @ curvature @ scales)
    estimated_public = float(PUBLIC_SCORES["v19"] + public_gain_from_v19)

    axes = [
        _axis_audit(project, train, axis, scales)
        for axis in (
            "y2023_early_to_late",
            "y2023_to_y2024",
            "y2024_early_to_late",
        )
    ]
    eligible = bool(
        estimated_public >= 1157.9594495591
        and all(item["gain_vs_v22"] > 0.0 for item in axes)
    )
    summary = {
        "protocol": "V24_LEADERBOARD_STAGE_SCALE_AUDIT_V1",
        "public_scores": PUBLIC_SCORES,
        "public_sequential_gains": gains.tolist(),
        "local_2024_curvature": curvature.tolist(),
        "inferred_public_linear_terms": linear.tolist(),
        "inferred_optimal_stage_scales": scales.tolist(),
        "estimated_public_score": estimated_public,
        "estimated_gain_vs_v22": estimated_public - PUBLIC_SCORES["v22"],
        "top10_cutoff_at_audit": 1157.9594495591,
        "axes": axes,
        "eligible_for_packaging": eligible,
        "decision": "reject" if not eligible else "promote",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v24_leaderboard_scale_audit_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

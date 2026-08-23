"""Low-variance domain calibration screen above v21.

The screen moves each of R_CORE, R_ANCHOR and F a small distance toward a
fixed probability anchor.  The transform simultaneously corrects mean bias
and shrinks excessive resolution, remains row-local, and adds no model files.
All combinations are compared on the three reconstructed champion axes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss


AXES = (
    "y2023_to_y2024",
    "y2023_early_to_late",
    "y2024_early_to_late",
)
DOMAINS = ("R_CORE", "R_ANCHOR", "F")
ANCHORS = (0.44, 0.45, 0.46, 0.47, 0.475, 0.48, 0.49, 0.50)
WEIGHTS = (0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10)


def _candidate(
    prediction: np.ndarray,
    domain: np.ndarray,
    anchor: float,
    weights: tuple[float, float, float],
) -> np.ndarray:
    output = prediction.copy()
    for name, weight in zip(DOMAINS, weights, strict=True):
        mask = domain == name
        output[mask] += weight * (anchor - output[mask])
    return np.clip(output, 0.001, 0.999)


def run(project: Path, champion_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    champion_dir = (project / champion_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = {}
    for axis in AXES:
        with np.load(champion_dir / f"{axis}.npz", allow_pickle=True) as saved:
            folds[axis] = {
                "target": saved["target"].astype(np.float64),
                "v21": saved["v21"].astype(np.float64),
                "domain3": saved["domain3"].astype(str),
                "game_month": saved["game_month"].astype(np.int16),
            }
    weight_grid = np.asarray(
        list(itertools.product(WEIGHTS, repeat=len(DOMAINS))), dtype=np.float64
    )
    rows = []
    for anchor in ANCHORS:
        gain_by_axis = {}
        for axis, fold in folds.items():
            target = fold["target"]
            prediction = fold["v21"]
            residual = target - prediction
            scale = 100000.0 / (target.mean() * (1.0 - target.mean()))
            linear = []
            square = []
            for domain in DOMAINS:
                correction = np.where(
                    fold["domain3"] == domain, anchor - prediction, 0.0
                )
                linear.append(scale * 2.0 * np.mean(residual * correction))
                square.append(scale * np.mean(np.square(correction)))
            gain_by_axis[axis] = (
                weight_grid @ np.asarray(linear)
                - np.square(weight_grid) @ np.asarray(square)
            )
        for index, weights_array in enumerate(weight_grid):
            weights = tuple(float(value) for value in weights_array)
            gains = {axis: float(value[index]) for axis, value in gain_by_axis.items()}
            rows.append(
                {
                    "anchor": anchor,
                    **{
                        f"weight_{domain.lower()}": weight
                        for domain, weight in zip(DOMAINS, weights, strict=True)
                    },
                    **gains,
                    "min_gain": min(gains.values()),
                    "mean_gain": float(np.mean(list(gains.values()))),
                    "max_gain": max(gains.values()),
                }
            )
    metrics = pd.DataFrame(rows).sort_values(
        ["min_gain", "mean_gain"], ascending=False
    ).reset_index(drop=True)
    best = metrics.iloc[0]
    diagnostics = []
    best_weights = tuple(float(best[f"weight_{d.lower()}"]) for d in DOMAINS)
    for axis, fold in folds.items():
        candidate = _candidate(
            fold["v21"], fold["domain3"], float(best["anchor"]), best_weights
        )
        month_rows = []
        for month in sorted(np.unique(fold["game_month"])):
            mask = fold["game_month"] == month
            month_rows.append(
                {
                    "month": int(month),
                    "rows": int(mask.sum()),
                    "gain": _bss(fold["target"][mask], candidate[mask])
                    - _bss(fold["target"][mask], fold["v21"][mask]),
                }
            )
        domain_rows = []
        for domain in DOMAINS:
            mask = fold["domain3"] == domain
            domain_rows.append(
                {
                    "domain": domain,
                    "rows": int(mask.sum()),
                    "gain": _bss(fold["target"][mask], candidate[mask])
                    - _bss(fold["target"][mask], fold["v21"][mask]),
                }
            )
        diagnostics.append(
            {
                "axis": axis,
                "gain": _bss(fold["target"], candidate)
                - _bss(fold["target"], fold["v21"]),
                "mean_shift": float(np.mean(candidate - fold["v21"])),
                "mean_abs_shift": float(np.mean(np.abs(candidate - fold["v21"]))),
                "positive_month_fraction": float(
                    np.mean([row["gain"] > 0.0 for row in month_rows])
                ),
                "worst_month_gain": float(min(row["gain"] for row in month_rows)),
                "months": month_rows,
                "domains": domain_rows,
            }
        )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "protocol": "V22_DOMAIN_ANCHOR_CALIBRATION_V1",
        "row_local_inference": True,
        "candidate_count": int(len(metrics)),
        "best": metrics.head(30).to_dict("records"),
        "best_diagnostics": diagnostics,
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
        "--champion-dir",
        type=Path,
        default=Path("artifacts/champion_oof_20260817_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v22_domain_calibration_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.champion_dir, args.output_dir)


if __name__ == "__main__":
    main()

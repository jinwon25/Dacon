"""Domain-specific anchor calibration with independent anchors per regime."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss
from src.champion.v22_domain_calibration_screen import AXES, DOMAINS


ANCHORS = (0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52)
WEIGHTS = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10)
TRANSFORMS = ((0.0, 0.50),) + tuple(
    (weight, anchor) for weight in WEIGHTS for anchor in ANCHORS
)


def _candidate(
    prediction: np.ndarray,
    domain: np.ndarray,
    transforms: tuple[tuple[float, float], ...],
) -> np.ndarray:
    output = prediction.copy()
    for name, (weight, anchor) in zip(DOMAINS, transforms, strict=True):
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

    contribution: dict[str, dict[str, np.ndarray]] = {}
    for axis, fold in folds.items():
        target = fold["target"]
        prediction = fold["v21"]
        residual = target - prediction
        scale = 100000.0 / (target.mean() * (1.0 - target.mean()))
        contribution[axis] = {}
        for domain in DOMAINS:
            mask = fold["domain3"] == domain
            values = []
            for weight, anchor in TRANSFORMS:
                correction = np.where(mask, weight * (anchor - prediction), 0.0)
                values.append(
                    scale
                    * (
                        2.0 * np.mean(residual * correction)
                        - np.mean(np.square(correction))
                    )
                )
            contribution[axis][domain] = np.asarray(values)

    rows = []
    for indices in itertools.product(range(len(TRANSFORMS)), repeat=len(DOMAINS)):
        gains = {
            axis: float(
                sum(
                    contribution[axis][domain][index]
                    for domain, index in zip(DOMAINS, indices, strict=True)
                )
            )
            for axis in AXES
        }
        transforms = [TRANSFORMS[index] for index in indices]
        rows.append(
            {
                **{
                    f"weight_{domain.lower()}": transforms[position][0]
                    for position, domain in enumerate(DOMAINS)
                },
                **{
                    f"anchor_{domain.lower()}": transforms[position][1]
                    for position, domain in enumerate(DOMAINS)
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
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    best = metrics.iloc[0]
    best_transforms = tuple(
        (
            float(best[f"weight_{domain.lower()}"]),
            float(best[f"anchor_{domain.lower()}"]),
        )
        for domain in DOMAINS
    )
    diagnostics = []
    for axis, fold in folds.items():
        candidate = _candidate(fold["v21"], fold["domain3"], best_transforms)
        months = []
        for month in sorted(np.unique(fold["game_month"])):
            mask = fold["game_month"] == month
            months.append(
                {
                    "month": int(month),
                    "rows": int(mask.sum()),
                    "gain": _bss(fold["target"][mask], candidate[mask])
                    - _bss(fold["target"][mask], fold["v21"][mask]),
                }
            )
        domains = []
        for domain in DOMAINS:
            mask = fold["domain3"] == domain
            domains.append(
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
                "positive_month_fraction": float(
                    np.mean([row["gain"] > 0.0 for row in months])
                ),
                "worst_month_gain": float(min(row["gain"] for row in months)),
                "months": months,
                "domains": domains,
            }
        )
    summary = {
        "protocol": "V22_DOMAIN_SPECIFIC_ANCHOR_CALIBRATION_V2",
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
        default=Path("artifacts/v22_domain_calibration_v2_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.champion_dir, args.output_dir)


if __name__ == "__main__":
    main()

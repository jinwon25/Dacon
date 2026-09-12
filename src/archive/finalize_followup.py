"""Append unlogged follow-up evaluations to the immutable experiment ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.archive.experiment_tracking import append_experiment


def run(project_dir: Path) -> int:
    experiments_path = project_dir / "research" / "reports" / "experiments.csv"
    walk_path = project_dir / "research" / "reports" / "walk_forward_results.csv"
    walk = pd.read_csv(walk_path, encoding="utf-8")
    existing = (
        set(
            pd.read_csv(experiments_path, usecols=["experiment_id"], encoding="utf-8")[
                "experiment_id"
            ].astype(str)
        )
        if experiments_path.exists()
        else set()
    )
    appended = 0
    for row in walk.itertuples(index=False):
        experiment_id = str(row.experiment_id)
        if experiment_id in existing:
            continue
        is_calibration = experiment_id.startswith("W1_calibration_")
        raw_brier = float(row.brier - row.delta_brier) if is_calibration else float(row.brier)
        append_experiment(
            experiments_path,
            {
                "experiment_id": experiment_id,
                "validation_split": (
                    f"train_before_{int(row.outer_validation_season)}"
                    f"_validate_{int(row.outer_validation_season)}"
                ),
                "feature_groups": str(row.prediction_source),
                "model_params": json.dumps(
                    {
                        "model": str(row.model),
                        "fit_seasons": str(row.fit_seasons)
                        if not pd.isna(row.fit_seasons)
                        else "",
                    },
                    sort_keys=True,
                ),
                "seed": 42,
                "raw_brier_score": f"{raw_brier:.9f}",
                "calibrated_brier_score": (
                    f"{float(row.brier):.9f}" if is_calibration else ""
                ),
                "local_brier_skill_score": f"{float(row.brier_skill):.3f}",
                "calibration": str(row.model) if is_calibration else "none/frozen train-only rule",
                "training_seconds": "",
                "inference_seconds": "",
                "model_size_mb": "",
                "peak_memory_mb": "",
                "leakage_risk_notes": (
                    "strict outer-year evaluation; target access limited to earlier OOF "
                    "for learned calibration/blend; validation targets used only for scoring"
                ),
            },
        )
        existing.add(experiment_id)
        appended += 1
    return appended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    count = run(args.project_dir.resolve())
    print(f"Appended {count} follow-up experiment rows")


if __name__ == "__main__":
    main()

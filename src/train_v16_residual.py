"""Fit the frozen 2025 v16 empirical-Bayes correction from historical OOF residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v16_residual_calibration_screen import _keys, load_v14_folds


SOURCE_SEASONS = (2022, 2023, 2024)
FORECAST_SEASON = 2025
DECAY = 0.5
ALPHA = 3200.0
ANCHOR_TEAM_ID = 13
GROUP_COLUMNS = ("pitcher_id", "batter_hand")


def run(
    project: Path,
    output_dir: Path,
    *,
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
    decay: float = DECAY,
    alpha: float = ALPHA,
    correction_weight: float = 1.0,
    candidate_name: str = "v16_multiseason_pitcher_batter_hand_eb_v1",
) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds = load_v14_folds(project, train)
    parts = []
    season_rows = {}
    for year in SOURCE_SEASONS:
        rows, target, prediction = folds[year]
        core = rows["domain3"].eq("R_CORE").to_numpy()
        weight = float(decay) ** (FORECAST_SEASON - 1 - year)
        season_rows[str(year)] = int(core.sum())
        parts.append(
            pd.DataFrame(
                {
                    "key": _keys(rows.loc[core].reset_index(drop=True), group_columns),
                    "weighted_residual": weight * (target[core] - prediction[core]),
                    "weight": weight,
                }
            )
        )
    history = pd.concat(parts, ignore_index=True)
    stats = history.groupby("key", observed=True).agg(
        weighted_residual=("weighted_residual", "sum"),
        effective_count=("weight", "sum"),
    )
    correction = stats["weighted_residual"] / (
        stats["effective_count"] + float(alpha)
    )
    deployed_correction = float(correction_weight) * correction
    effects = {str(key): float(value) for key, value in deployed_correction.items()}
    spec: dict[str, object] = {
        "candidate": candidate_name,
        "method": "smoothed_oof_residual_mean",
        "source_seasons": list(SOURCE_SEASONS),
        "forecast_season": FORECAST_SEASON,
        "season_decay": float(decay),
        "alpha": float(alpha),
        "correction_weight": float(correction_weight),
        "group_columns": list(group_columns),
        "apply_domain": "R_CORE",
        "anchor_team_id": ANCHOR_TEAM_ID,
        "separator": "\\u001f",
        "unseen_effect": 0.0,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "season_rows": season_rows,
        "n_effects": len(effects),
        "effect_summary": {
            "minimum": float(deployed_correction.min()),
            "maximum": float(deployed_correction.max()),
            "mean": float(deployed_correction.mean()),
            "mean_absolute": float(deployed_correction.abs().mean()),
            "weighted_mean_absolute": float(
                np.average(deployed_correction.abs(), weights=stats["effective_count"])
            ),
            "median_effective_count": float(stats["effective_count"].median()),
            "maximum_effective_count": float(stats["effective_count"].max()),
        },
        "effects": effects,
        "selection_rule": (
            "positive next-season gain on 2022->2023 and 2022+2023->2024; "
            "selected by maximum two-transition minimum gain"
        ),
        "selection_caveat": (
            "2024 and the candidate family are development-contaminated; Public "
            "submission is a deployment probe, not independent confirmation"
        ),
    }
    path = output_dir / "v16_residual_spec.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        key: value for key, value in spec.items() if key != "effects"
    }
    manifest["artifact"] = path.name
    manifest["artifact_size_bytes"] = path.stat().st_size
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v16_residual_final_20260815"),
    )
    parser.add_argument(
        "--group",
        choices=["pitcher_batter_hand", "pitcher_batter_hand_pressure"],
        default="pitcher_batter_hand",
    )
    parser.add_argument("--decay", type=float, default=DECAY)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--correction-weight", type=float, default=1.0)
    parser.add_argument(
        "--candidate-name", default="v16_multiseason_pitcher_batter_hand_eb_v1"
    )
    args = parser.parse_args()
    group_columns = (
        ("pitcher_id", "batter_hand", "pressure")
        if args.group == "pitcher_batter_hand_pressure"
        else GROUP_COLUMNS
    )
    run(
        args.project,
        args.output_dir,
        group_columns=group_columns,
        decay=args.decay,
        alpha=args.alpha,
        correction_weight=args.correction_weight,
        candidate_name=args.candidate_name,
    )


if __name__ == "__main__":
    main()

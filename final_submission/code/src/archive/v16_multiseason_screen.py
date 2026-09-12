"""Multi-season empirical-Bayes residual screen for the v16 R_CORE axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.archive.v16_residual_calibration_screen import EPSILON, _bss_gain, _keys, load_v14_folds


GROUPS = {
    "pitcher_batter_hand": ("pitcher_id", "batter_hand"),
    "pitcher_count": ("pitcher_id", "balls_before", "strikes_before"),
    "pitcher_batter_hand_count": (
        "pitcher_id",
        "batter_hand",
        "balls_before",
        "strikes_before",
    ),
    "pitcher_batter_hand_base": ("pitcher_id", "batter_hand", "base_state"),
    "pitcher_batter_hand_pressure": ("pitcher_id", "batter_hand", "pressure"),
    "pitcher_batter_hand_runners": (
        "pitcher_id",
        "batter_hand",
        "num_runners_on",
    ),
    "pitcher_batter_hand_half": ("pitcher_id", "batter_hand", "top_bottom"),
}


def _history_stats(
    folds: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    audit_year: int,
    columns: tuple[str, ...],
    decay: float,
) -> pd.DataFrame:
    parts = []
    for year in sorted(value for value in folds if value < audit_year):
        rows, target, prediction = folds[year]
        mask = rows["domain3"].eq("R_CORE").to_numpy()
        weight = float(decay) ** float(audit_year - 1 - year)
        parts.append(
            pd.DataFrame(
                {
                    "key": _keys(rows.loc[mask].reset_index(drop=True), columns),
                    "weighted_residual": weight * (target[mask] - prediction[mask]),
                    "weight": weight,
                }
            )
        )
    history = pd.concat(parts, ignore_index=True)
    return history.groupby("key", observed=True).agg(
        weighted_residual=("weighted_residual", "sum"),
        effective_count=("weight", "sum"),
    )


def _map_correction(
    stats: pd.DataFrame,
    audit_rows: pd.DataFrame,
    columns: tuple[str, ...],
    alpha: float,
) -> np.ndarray:
    effect = stats["weighted_residual"] / (stats["effective_count"] + float(alpha))
    return _keys(audit_rows, columns).map(effect).fillna(0.0).to_numpy(np.float64)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds = load_v14_folds(project, train)
    rows_out: list[dict[str, object]] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for audit_year in (2023, 2024):
        audit_rows, target, incumbent = folds[audit_year]
        core = audit_rows["domain3"].eq("R_CORE").to_numpy()
        for group_name, columns in GROUPS.items():
            for decay in (0.25, 0.50, 0.75, 1.0):
                stats = _history_stats(folds, audit_year, columns, decay)
                for alpha in (800.0, 1600.0, 3200.0, 6400.0, 12800.0):
                    correction = _map_correction(stats, audit_rows, columns, alpha)
                    for weight in (0.5, 1.0, 1.5):
                        name = f"multi_{group_name}_d{decay:g}_a{alpha:g}_w{weight:g}"
                        candidate = incumbent.copy()
                        candidate[core] += weight * correction[core]
                        candidate = np.clip(candidate, EPSILON, 1.0 - EPSILON)
                        rows_out.append(
                            {
                                "candidate": name,
                                "audit_year": audit_year,
                                "n_rows": len(target),
                                "n_changed": int(core.sum()),
                                "group": group_name,
                                "decay": decay,
                                "alpha": alpha,
                                "weight": weight,
                                "gain_vs_v14": _bss_gain(target, candidate, incumbent),
                            }
                        )
                        predictions[(name, audit_year)] = candidate
    metrics = pd.DataFrame(rows_out)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust = (
        metrics.groupby("candidate", observed=True)["gain_vs_v14"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust["positive_folds"] = (
        metrics.assign(positive=metrics["gain_vs_v14"].gt(0.0))
        .groupby("candidate", observed=True)["positive"]
        .sum()
        .reindex(robust["candidate"])
        .to_numpy()
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    for name in robust.head(5)["candidate"]:
        for year in (2023, 2024):
            rows, target, incumbent = folds[year]
            np.savez_compressed(
                output_dir / f"{name}_o{year}.npz",
                target=target,
                incumbent=incumbent,
                candidate=predictions[(name, year)],
            )
    passing = robust.loc[robust["min_gain"].gt(0.0)]
    summary = {
        "protocol": "V16_MULTI_SEASON_EB_FORWARD_ONLY_V1",
        "audit_years": [2023, 2024],
        "n_recipes": int(metrics["candidate"].nunique()),
        "n_strict_improvements": int(len(passing)),
        "best": robust.head(20).to_dict(orient="records"),
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
        default=Path("artifacts/v16_multiseason_20260815_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

"""Evaluate the frozen recency-RF plus Trackman hybrid candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.data import TARGET_COL, read_main
from src.metrics import brier_score, cluster_bootstrap_delta
from src.archive.rolling_drift import build_rolling_damped_ensemble


CANDIDATE_NAME = "hybrid_recency_r_trackman_w05_v1"
TRACKMAN_WEIGHT = 0.05
ROLLING_METHODS = ["damped_3_0.5", "damped_3_0.8"]
ROLLING_WEIGHTS = [0.5, 0.5]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def build_fold_prediction(
    project_dir: Path, train: pd.DataFrame, year: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    incumbent = _load_npz(
        project_dir
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    recency = _load_npz(
        project_dir
        / "artifacts"
        / "followup"
        / "models"
        / f"recency_h1p0_validate_{year}.npz"
    )
    trackman = _load_npz(
        project_dir
        / "artifacts"
        / "followup"
        / "models"
        / f"lgb_trackman_pitcher_v1_validate_{year}.npz"
    )
    target = incumbent["target"].astype(np.float64)
    valid_idx = incumbent["valid_idx"].astype(np.int64)
    if not np.array_equal(valid_idx, recency["valid_idx"]):
        raise ValueError(f"recency validation indices mismatch for {year}")
    if not np.array_equal(valid_idx, trackman["valid_idx"]):
        raise ValueError(f"Trackman validation indices mismatch for {year}")
    base = incumbent["incumbent"].astype(np.float64)
    recency_r = base.copy()
    regular = train.iloc[valid_idx]["game_type"].astype("string").eq("R").to_numpy()
    weighted_rf_blend = (
        0.35 * incumbent["lgb_trend"].astype(np.float64)
        + 0.65 * recency["weighted_rf"].astype(np.float64)
    )
    recency_r[regular] = weighted_rf_blend[regular]
    season_rates = (
        train.loc[train["season"] < year]
        .groupby("season", observed=True)[TARGET_COL]
        .mean()
    )
    trackman_rolling, _ = build_rolling_damped_ensemble(
        trackman["prediction"].astype(np.float64),
        season_rates,
        year,
        ROLLING_METHODS,
        ROLLING_WEIGHTS,
    )
    prediction = (
        (1.0 - TRACKMAN_WEIGHT) * recency_r
        + TRACKMAN_WEIGHT * trackman_rolling
    )
    return target, base, prediction, valid_idx


def run(project_dir: Path, resamples: int = 10_000) -> pd.DataFrame:
    project_dir = project_dir.resolve()
    train = read_main(project_dir / "data" / "train.csv")
    followup = json.loads(
        (project_dir / "research" / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    combined_target: list[np.ndarray] = []
    combined_candidate: list[np.ndarray] = []
    combined_incumbent: list[np.ndarray] = []
    combined_clusters: list[pd.Series] = []
    for year in (2021, 2022, 2023, 2024):
        target, incumbent, prediction, valid_idx = build_fold_prediction(
            project_dir, train, year
        )
        delta = brier_score(target, prediction) - brier_score(target, incumbent)
        rows.append(
            {
                "candidate": CANDIDATE_NAME,
                "outer_validation_season": year,
                "n_rows": len(target),
                "brier": brier_score(target, prediction),
                "incumbent_brier": brier_score(target, incumbent),
                "delta_brier": delta,
                "prediction_mean": float(prediction.mean()),
                "target_rate": float(target.mean()),
            }
        )
        fold = train.iloc[valid_idx].reset_index(drop=True)
        clusters = (
            fold["pitcher_id"].astype("string")
            + "-"
            + fold["season"].astype("string")
        )
        bootstrap_rows.append(
            {
                "scope": str(year),
                "cluster_type": "pitcher_season",
                **cluster_bootstrap_delta(
                    target,
                    prediction,
                    incumbent,
                    clusters,
                    n_resamples=resamples,
                    seed=int(followup["seed"]) + year,
                ),
            }
        )
        combined_target.append(target)
        combined_candidate.append(prediction)
        combined_incumbent.append(incumbent)
        combined_clusters.append(clusters)
    frame = pd.DataFrame(rows)
    weights = np.asarray(
        [float(followup["recency_weights"][str(year)]) for year in frame["outer_validation_season"]]
    )
    recency_delta = float(np.average(frame["delta_brier"], weights=weights))
    worst_delta = float(frame["delta_brier"].max())
    combined = cluster_bootstrap_delta(
        np.concatenate(combined_target),
        np.concatenate(combined_candidate),
        np.concatenate(combined_incumbent),
        pd.concat(combined_clusters, ignore_index=True),
        n_resamples=resamples,
        seed=int(followup["seed"]),
    )
    bootstrap_rows.append(
        {
            "scope": "2021-2024",
            "cluster_type": "pitcher_season",
            **combined,
        }
    )
    gate = followup["submission_gate"]
    latest_delta = float(
        frame.loc[frame["outer_validation_season"] == 2024, "delta_brier"].iloc[0]
    )
    checks = {
        "recency": recency_delta
        <= -float(gate["minimum_recency_weighted_improvement"]),
        "latest": latest_delta
        <= -float(gate["minimum_2024_improvement"]),
        "worst": worst_delta
        <= float(gate["maximum_single_fold_worsening"]),
        "bootstrap": float(combined["improvement_probability"])
        >= float(gate["minimum_cluster_improvement_probability"]),
    }
    frame["recency_weighted_delta"] = recency_delta
    frame["worst_fold_delta"] = worst_delta
    frame["combined_improvement_probability"] = float(
        combined["improvement_probability"]
    )
    frame["passes_statistical_gate"] = all(checks.values())
    frame.to_csv(
        project_dir / "research" / "reports" / "hybrid_candidate_results.csv",
        index=False,
        encoding="utf-8",
    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(
        project_dir / "research" / "reports" / "hybrid_candidate_bootstrap.csv",
        index=False,
        encoding="utf-8",
    )
    lines = [
        "# Hybrid recency/Trackman candidate",
        "",
        "- Frozen recipe: incumbent with half-life-1 RF substituted only for `game_type=R`, then 5% rolling-damped Trackman LGB.",
        "- Trackman linkage and profiles use only seasons before each forecast origin.",
        "- No validation target enters linkage, feature construction, drift forecasts, or row-level application.",
        "",
        "| year | incumbent | candidate | delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.outer_validation_season} | {row.incumbent_brier:.9f} | "
            f"{row.brier:.9f} | {row.delta_brier:+.9f} |"
        )
    lines.extend(
        [
            "",
            f"- Recency-weighted delta: **{recency_delta:+.9f}**",
            f"- Latest-year delta: **{latest_delta:+.9f}**",
            f"- Worst-fold delta: **{worst_delta:+.9f}**",
            f"- Combined pitcher-season bootstrap P(improve): **{combined['improvement_probability']:.4f}**",
            f"- Fixed gate: **{'PASS' if all(checks.values()) else 'FAIL'}**",
            "- Selection caveat: all four outer years have already been reused extensively as development data; Public submission is a deployment probe, not independent proof.",
        ]
    )
    (project_dir / "research" / "reports" / "hybrid_candidate_findings.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print("checks", checks)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    run(args.project_dir, args.resamples)


if __name__ == "__main__":
    main()

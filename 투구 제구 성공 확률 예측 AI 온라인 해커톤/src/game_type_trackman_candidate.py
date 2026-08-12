"""Evaluate a row-local game-type Trackman blend candidate on cached OOF data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import TARGET_COL, read_main
from src.metrics import brier_score, cluster_bootstrap_delta
from src.rolling_drift import build_rolling_damped_ensemble


CANDIDATE_NAME = "game_type_trackman_f100_r10_v1"
TRACKMAN_WEIGHTS = {"F": 1.0, "R": 0.10}
DEFAULT_TRACKMAN_WEIGHT = 0.0
ROLLING_METHODS = ["damped_3_0.5", "damped_3_0.8"]
ROLLING_WEIGHTS = [0.5, 0.5]
YEARS = (2021, 2022, 2023, 2024)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def build_fold_prediction(
    project_dir: Path, train: pd.DataFrame, year: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    incumbent = _load_npz(
        project_dir / "artifacts" / "followup" / "oof" / f"wave0_incumbent_validate_{year}.npz"
    )
    recency = _load_npz(
        project_dir / "artifacts" / "followup" / "models" / f"recency_h1p0_validate_{year}.npz"
    )
    trackman = _load_npz(
        project_dir / "artifacts" / "followup" / "models" / f"lgb_trackman_pitcher_v1_validate_{year}.npz"
    )
    valid_idx = incumbent["valid_idx"].astype(np.int64)
    if not np.array_equal(valid_idx, recency["valid_idx"]):
        raise ValueError(f"recency validation indices mismatch for {year}")
    if not np.array_equal(valid_idx, trackman["valid_idx"]):
        raise ValueError(f"Trackman validation indices mismatch for {year}")

    target = incumbent["target"].astype(np.float64)
    frame = train.iloc[valid_idx].reset_index(drop=True).copy()
    candidate_base = incumbent["incumbent"].astype(np.float64).copy()
    regular = frame["game_type"].astype("string").eq("R").to_numpy()
    candidate_base[regular] = (
        0.35 * incumbent["lgb_trend"].astype(np.float64)[regular]
        + 0.65 * recency["weighted_rf"].astype(np.float64)[regular]
    )

    season_rates = train.loc[train["season"] < year].groupby("season")[TARGET_COL].mean()
    trackman_probability, _ = build_rolling_damped_ensemble(
        trackman["prediction"].astype(np.float64),
        season_rates,
        year,
        ROLLING_METHODS,
        ROLLING_WEIGHTS,
    )
    weights = np.full(len(frame), DEFAULT_TRACKMAN_WEIGHT, dtype=np.float64)
    for game_type, weight in TRACKMAN_WEIGHTS.items():
        weights[frame["game_type"].astype("string").eq(game_type).to_numpy()] = weight
    prediction = (1.0 - weights) * candidate_base + weights * trackman_probability
    return target, incumbent["incumbent"].astype(np.float64), prediction, frame


def run(project_dir: Path, resamples: int = 2_000) -> pd.DataFrame:
    project_dir = project_dir.resolve()
    train = read_main(project_dir / "data" / "train.csv")
    rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    all_target: list[np.ndarray] = []
    all_candidate: list[np.ndarray] = []
    all_incumbent: list[np.ndarray] = []
    all_clusters: list[pd.Series] = []

    for year in YEARS:
        target, incumbent, candidate, frame = build_fold_prediction(project_dir, train, year)
        rows.append(
            {
                "candidate": CANDIDATE_NAME,
                "outer_validation_season": year,
                "n_rows": len(target),
                "brier": brier_score(target, candidate),
                "incumbent_brier": brier_score(target, incumbent),
                "delta_brier": brier_score(target, candidate) - brier_score(target, incumbent),
                "prediction_mean": float(candidate.mean()),
                "target_rate": float(target.mean()),
            }
        )
        clusters = frame["pitcher_id"].astype("string") + "-" + frame["season"].astype("string")
        bootstrap_rows.append(
            {
                "scope": str(year),
                "cluster_type": "pitcher_season",
                **cluster_bootstrap_delta(
                    target,
                    candidate,
                    incumbent,
                    clusters,
                    n_resamples=resamples,
                    seed=42 + year,
                ),
            }
        )
        all_target.append(target)
        all_candidate.append(candidate)
        all_incumbent.append(incumbent)
        all_clusters.append(clusters)

    target = np.concatenate(all_target)
    candidate = np.concatenate(all_candidate)
    incumbent = np.concatenate(all_incumbent)
    combined = cluster_bootstrap_delta(
        target,
        candidate,
        incumbent,
        pd.concat(all_clusters, ignore_index=True),
        n_resamples=resamples,
        seed=42,
    )
    bootstrap_rows.append({"scope": "2021-2024", "cluster_type": "pitcher_season", **combined})

    result = pd.DataFrame(rows)
    recency_weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    recency_delta = float(np.average(result["delta_brier"], weights=recency_weights))
    worst_delta = float(result["delta_brier"].max())
    result["recency_weighted_delta"] = recency_delta
    result["worst_fold_delta"] = worst_delta
    result["combined_improvement_probability"] = float(combined["improvement_probability"])
    result["passes_all_folds"] = result["delta_brier"].lt(0.0)
    result.to_csv(project_dir / "reports" / "game_type_trackman_candidate_results.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        project_dir / "reports" / "game_type_trackman_candidate_bootstrap.csv", index=False
    )

    report = [
        f"# {CANDIDATE_NAME}",
        "",
        "- Base: incumbent 0.35 engineered LightGBM + 0.65 official RF.",
        "- `game_type=R`: replace only the RF component with the frozen half-life-1 recency RF, then blend Trackman at 10%.",
        "- `game_type=F`: use the rolling-damped Trackman probability at 100%.",
        "- Other game types: retain the candidate base (Trackman weight 0%).",
        "- All weights are row-local constants; no test-batch statistic is used.",
        "",
        "| year | incumbent Brier | candidate Brier | delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in result.itertuples(index=False):
        report.append(
            f"| {row.outer_validation_season} | {row.incumbent_brier:.9f} | "
            f"{row.brier:.9f} | {row.delta_brier:+.9f} |"
        )
    report.extend(
        [
            "",
            f"- Recency-weighted delta: **{recency_delta:+.9f}**",
            f"- Worst-fold delta: **{worst_delta:+.9f}**",
            f"- Combined pitcher-season bootstrap P(improve): **{combined['improvement_probability']:.4f}**",
            "- This is a local OOF research candidate; the public score is not claimed until a DACON submission is made.",
        ]
    )
    (project_dir / "reports" / "game_type_trackman_candidate.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (project_dir / "reports" / "game_type_trackman_candidate_config.json").write_text(
        json.dumps(
            {
                "candidate": CANDIDATE_NAME,
                "trackman_weight_by_game_type": TRACKMAN_WEIGHTS,
                "trackman_default_weight": DEFAULT_TRACKMAN_WEIGHT,
                "rolling_methods": ROLLING_METHODS,
                "rolling_weights": ROLLING_WEIGHTS,
                "selection_scope": "cached 2021-2024 walk-forward OOF; no test rows",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print("combined bootstrap", combined)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--resamples", type=int, default=2_000)
    args = parser.parse_args()
    run(args.project_dir, args.resamples)


if __name__ == "__main__":
    main()

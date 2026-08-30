"""Screen a strict-forward Beta-Binomial complement above v290-equivalent axes.

The feature representation is inspired by a public competition repository, but
this implementation is independent and uses only the official train columns.
Career ASOF counts are differenced against an exact prior-season terminal
snapshot to recover row-local current-season pitcher and batter form.  Pooling
weights and concentration for prediction year Y are learned only on Y-1.

Blend weights are selected by maximin gain across full-2022 and late-2023.
Full-2024 is opened once as locked confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROTOCOL = "V321_STRICT_BETA_BINOMIAL_COMPLEMENT_V1"
TARGET = "control_success"
CONCENTRATIONS = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0)
BLEND_WEIGHTS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
CANDIDATE_NAMES = (
    "pitcher_season_posterior",
    "batter_season_posterior",
    "previous_season_game_type_prior",
    "pitcher_career_rate",
    "pitcher_prev5_game_rate",
)


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rate = float(target.mean())
    return float(
        100000.0
        * (1.0 - np.mean((target - prediction) ** 2) / (rate * (1.0 - rate)))
    )


def terminal_snapshot(
    history: pd.DataFrame,
    id_column: str,
    n_column: str,
    rate_column: str,
) -> dict[int, tuple[float, float]]:
    """Return exact post-pitch terminal (count, successes) from prior seasons."""
    if history.empty:
        return {}
    work = history[[id_column, n_column, rate_column, TARGET]].copy()
    work["_n"] = pd.to_numeric(work[n_column], errors="coerce").fillna(0.0)
    work["_successes_after"] = (
        work["_n"]
        * pd.to_numeric(work[rate_column], errors="coerce").fillna(0.0)
        + pd.to_numeric(work[TARGET], errors="raise")
    )
    work["_n_after"] = work["_n"] + 1.0
    latest = work.loc[work.groupby(id_column, sort=False)["_n"].idxmax()]
    return {
        int(identifier): (float(count), float(successes))
        for identifier, count, successes in latest[
            [id_column, "_n_after", "_successes_after"]
        ].itertuples(index=False, name=None)
    }


def current_season_state(
    frame: pd.DataFrame,
    snapshot: dict[int, tuple[float, float]],
    id_column: str,
    n_column: str,
    rate_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    identifiers = pd.to_numeric(frame[id_column], errors="raise").to_numpy(np.int64)
    opening_n = np.fromiter(
        (snapshot.get(int(value), (0.0, 0.0))[0] for value in identifiers),
        dtype=np.float64,
        count=len(frame),
    )
    opening_successes = np.fromiter(
        (snapshot.get(int(value), (0.0, 0.0))[1] for value in identifiers),
        dtype=np.float64,
        count=len(frame),
    )
    career_n = pd.to_numeric(frame[n_column], errors="coerce").fillna(0.0).to_numpy(np.float64)
    career_rate = (
        pd.to_numeric(frame[rate_column], errors="coerce").fillna(0.5).to_numpy(np.float64)
    )
    season_n = np.maximum(career_n - opening_n, 0.0)
    season_successes = np.clip(career_n * career_rate - opening_successes, 0.0, season_n)
    return season_n, season_successes


def beta_candidate_matrix(
    train: pd.DataFrame,
    prediction_year: int,
    concentration: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build candidates for Y without reading any target from season Y."""
    frame = train.loc[train["season"].eq(prediction_year)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no rows for prediction year {prediction_year}")
    prior_history = train.loc[train["season"].lt(prediction_year)].reset_index(drop=True)
    previous = train.loc[train["season"].eq(prediction_year - 1)]
    global_prior = float(prior_history[TARGET].mean())
    by_type = previous.groupby("game_type", observed=True)[TARGET].mean()
    prior = frame["game_type"].map(by_type).fillna(global_prior).to_numpy(np.float64)
    # The official train data has an abrupt F generation-process change in 2023.
    # A neutral prior is fixed only for that first new-regime season.
    if prediction_year == 2023:
        prior = np.where(frame["game_type"].astype(str).eq("F"), 0.5, prior)

    posteriors: list[np.ndarray] = []
    for id_column, n_column, rate_column in (
        ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
    ):
        snapshot = terminal_snapshot(prior_history, id_column, n_column, rate_column)
        season_n, season_successes = current_season_state(
            frame, snapshot, id_column, n_column, rate_column
        )
        posteriors.append((season_successes + concentration * prior) / (season_n + concentration))

    career = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        .fillna(pd.Series(prior, index=frame.index))
        .to_numpy(np.float64)
    )
    recent = (
        pd.to_numeric(frame["asof_pitcher_prev5_game_success_rate"], errors="coerce")
        .fillna(pd.Series(career, index=frame.index))
        .to_numpy(np.float64)
    )
    matrix = np.column_stack((posteriors[0], posteriors[1], prior, career, recent))
    return frame, np.clip(matrix, 0.001, 0.999)


def fit_previous_year_pool(train: pd.DataFrame, prediction_year: int) -> dict[str, Any]:
    """Select concentration and convex weights on Y-1, then freeze for Y."""
    calibration_year = prediction_year - 1
    best: tuple[float, float, np.ndarray] | None = None
    for concentration in CONCENTRATIONS:
        frame, matrix = beta_candidate_matrix(train, calibration_year, concentration)
        target = frame[TARGET].to_numpy(np.float64)
        result = minimize(
            lambda weights: float(np.mean((matrix @ weights - target) ** 2)),
            x0=np.full(matrix.shape[1], 1.0 / matrix.shape[1]),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * matrix.shape[1],
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(result.message)
        candidate = (float(result.fun), concentration, result.x.astype(np.float64))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("no Beta-Binomial pool fitted")
    loss, concentration, weights = best
    return {
        "calibration_year": calibration_year,
        "concentration": concentration,
        "weights": weights,
        "candidate_names": CANDIDATE_NAMES,
        "calibration_brier": loss,
    }


def predict_year(train: pd.DataFrame, prediction_year: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    fitted = fit_previous_year_pool(train, prediction_year)
    frame, matrix = beta_candidate_matrix(train, prediction_year, float(fitted["concentration"]))
    probability = np.clip(matrix @ np.asarray(fitted["weights"], dtype=np.float64), 0.001, 0.999)
    serializable = {**fitted, "weights": np.asarray(fitted["weights"]).tolist()}
    return frame, probability, serializable


def axis_metrics(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    target = frame[TARGET].to_numpy(np.float64)
    months = []
    for month in sorted(frame["game_month"].unique()):
        selected = frame["game_month"].eq(month).to_numpy()
        months.append(
            {
                "month": int(month),
                "rows": int(selected.sum()),
                "gain": bss(target[selected], candidate[selected])
                - bss(target[selected], baseline[selected]),
            }
        )
    shift = candidate - baseline
    return {
        "gain": bss(target, candidate) - bss(target, baseline),
        "positive_month_fraction": float(np.mean([row["gain"] > 0.0 for row in months])),
        "worst_month_gain": float(min(row["gain"] for row in months)),
        "rms_shift": float(np.sqrt(np.mean(shift**2))),
        "mean_abs_shift": float(np.mean(np.abs(shift))),
        "months": months,
    }


def run(train_csv: Path, v285_axes: Path, v288_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    with np.load(v285_axes, allow_pickle=False) as saved:
        baseline_2022 = saved["candidate_full_2022"].astype(np.float64)
    with np.load(v288_axes, allow_pickle=False) as saved:
        baseline_late_2023 = saved["candidate_late_2023"].astype(np.float64)
        baseline_2024 = saved["candidate_full_2024"].astype(np.float64)

    frames: dict[str, pd.DataFrame] = {}
    beta: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    for year in (2022, 2023, 2024):
        year_frame, prediction, recipe = predict_year(train, year)
        name = {2022: "full_2022", 2023: "late_2023", 2024: "full_2024"}[year]
        if year == 2023:
            selected = year_frame["game_month"].ge(8).to_numpy()
            year_frame = year_frame.loc[selected].reset_index(drop=True)
            prediction = prediction[selected]
        frames[name] = year_frame
        beta[name] = prediction
        fitted[name] = recipe
    baselines = {
        "full_2022": baseline_2022,
        "late_2023": baseline_late_2023,
        "full_2024": baseline_2024,
    }
    for name in frames:
        if len(frames[name]) != len(baselines[name]):
            raise ValueError(f"axis alignment mismatch: {name}")

    selected_weights: dict[str, float] = {}
    source_grids: dict[str, Any] = {}
    for group in ("R", "F"):
        grid_rows = []
        for weight in BLEND_WEIGHTS:
            gains = []
            for name in ("full_2022", "late_2023"):
                active = frames[name]["game_type"].astype(str).eq(group).to_numpy()
                candidate = baselines[name].copy()
                candidate[active] = (
                    (1.0 - weight) * candidate[active] + weight * beta[name][active]
                )
                gains.append(
                    bss(frames[name][TARGET].to_numpy(np.float64), candidate)
                    - bss(frames[name][TARGET].to_numpy(np.float64), baselines[name])
                )
            grid_rows.append(
                {
                    "weight": weight,
                    "full_2022_gain": gains[0],
                    "late_2023_gain": gains[1],
                    "source_min_gain": float(min(gains)),
                    "source_mean_gain": float(np.mean(gains)),
                }
            )
        selected = max(
            grid_rows,
            key=lambda row: (
                row["source_min_gain"], row["source_mean_gain"], -row["weight"]
            ),
        )
        selected_weights[group] = float(selected["weight"])
        source_grids[group] = {"selected": selected, "rows": grid_rows}

    candidates: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, frame in frames.items():
        route_weight = frame["game_type"].astype(str).map(selected_weights).fillna(0.0).to_numpy(np.float64)
        candidates[name] = np.clip(
            (1.0 - route_weight) * baselines[name] + route_weight * beta[name],
            0.001,
            0.999,
        )
        metrics[name] = axis_metrics(frame, baselines[name], candidates[name])

    source_pass = bool(
        metrics["full_2022"]["gain"] > 0.0 and metrics["late_2023"]["gain"] > 0.0
    )
    locked_pass = bool(
        metrics["full_2024"]["gain"] > 0.0
        and metrics["full_2024"]["positive_month_fraction"] >= 0.625
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if source_pass and locked_pass else "screen_reject",
        "selected_route_weights": selected_weights,
        "source_grids": source_grids,
        "fitted_pooling": fitted,
        "metrics": metrics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_followup": bool(source_pass and locked_pass),
        "restrictions": {
            "official_train_only": True,
            "external_predictions_used": False,
            "prediction_year_target_used": False,
            "pooling_fitted_on_previous_year_only": True,
            "blend_selected_on_full_2022_and_late_2023_only": True,
            "full_2024_locked_confirmation": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        baseline_full_2022=baselines["full_2022"],
        beta_full_2022=beta["full_2022"],
        candidate_full_2022=candidates["full_2022"],
        baseline_late_2023=baselines["late_2023"],
        beta_late_2023=beta["late_2023"],
        candidate_late_2023=candidates["late_2023"],
        baseline_full_2024=baselines["full_2024"],
        beta_full_2024=beta["full_2024"],
        candidate_full_2024=candidates["full_2024"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v288-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v285_axes, args.v288_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

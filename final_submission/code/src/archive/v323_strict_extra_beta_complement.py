"""Strict-forward ExtraTrees plus Beta-Binomial complement above v290.

This is an independent implementation of a public methodological idea.  It
uses no external predictions or data.  For every prediction season Y:

* engineered current-season features use ASOF values plus targets from <Y;
* ExtraTrees trains only on seasons <Y;
* the Beta-Binomial component is fitted on Y-1;
* groupwise logit calibration is fitted on raw predictions from Y-1;
* blend weights above v290 are selected on full-2022 and late-2023 only.

Full-2024 is opened once as locked confirmation.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.archive.v321_strict_beta_binomial_complement import (
    BLEND_WEIGHTS,
    TARGET,
    bss,
    current_season_state,
    predict_year,
    terminal_snapshot,
)


PROTOCOL = "V323_STRICT_EXTRA_BETA_COMPLEMENT_V1"
EXTRA_WEIGHT = 0.7590660319
EPSILON = 1e-6
CAT_COLUMNS = ("top_bottom", "game_type", "base_state")


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def add_time_safe_features(train: pd.DataFrame) -> pd.DataFrame:
    """Derive within-season form without reading a row's current-season target."""
    enriched = train.copy()
    seasons = sorted(int(value) for value in train["season"].unique())
    for entity, id_column, n_column, rate_column in (
        ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
    ):
        season_n_all = np.zeros(len(train), dtype=np.float64)
        season_successes_all = np.zeros(len(train), dtype=np.float64)
        history_rate_all = np.full(len(train), 0.5, dtype=np.float64)
        for year in seasons:
            selected = train["season"].eq(year).to_numpy()
            frame = train.loc[selected]
            history = train.loc[train["season"].lt(year)]
            snapshot = terminal_snapshot(history, id_column, n_column, rate_column)
            season_n, season_successes = current_season_state(
                frame, snapshot, id_column, n_column, rate_column
            )
            identifiers = pd.to_numeric(frame[id_column], errors="raise").to_numpy(np.int64)
            fallback = float(history[TARGET].mean()) if not history.empty else 0.5
            history_rate = np.fromiter(
                (
                    snapshot[int(value)][1] / snapshot[int(value)][0]
                    if int(value) in snapshot and snapshot[int(value)][0] > 0.0
                    else fallback
                    for value in identifiers
                ),
                dtype=np.float64,
                count=len(frame),
            )
            season_n_all[selected] = season_n
            season_successes_all[selected] = season_successes
            history_rate_all[selected] = history_rate
        enriched[f"derived_{entity}_history_success_rate"] = history_rate_all
        enriched[f"derived_{entity}_season_n"] = season_n_all
        raw_rate = np.divide(
            season_successes_all,
            season_n_all,
            out=history_rate_all.copy(),
            where=season_n_all > 0.0,
        )
        enriched[f"derived_{entity}_season_success_rate"] = raw_rate
        for strength in (10.0, 50.0, 200.0):
            enriched[f"derived_{entity}_season_success_rate_s{int(strength)}"] = (
                season_successes_all + strength * history_rate_all
            ) / (season_n_all + strength)

    enriched["derived_prev1_minus_career_success"] = (
        enriched["asof_pitcher_prev1_game_success_rate"]
        - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_prev3_minus_career_success"] = (
        enriched["asof_pitcher_prev3_game_success_rate"]
        - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_prev5_minus_career_success"] = (
        enriched["asof_pitcher_prev5_game_success_rate"]
        - enriched["asof_pitcher_success_rate"]
    )
    enriched["derived_recent_success_slope"] = (
        enriched["asof_pitcher_prev1_game_success_rate"]
        - enriched["asof_pitcher_prev5_game_success_rate"]
    )
    enriched["derived_two_strike"] = enriched["strikes_before"].eq(2).astype("int8")
    enriched["derived_three_ball"] = enriched["balls_before"].eq(3).astype("int8")
    enriched["derived_full_count"] = (
        enriched["balls_before"].eq(3) & enriched["strikes_before"].eq(2)
    ).astype("int8")
    enriched["derived_platoon_same_hand"] = (
        enriched["pitcher_hand"].astype(str) == enriched["batter_hand"].astype(str)
    ).astype("int8")
    enriched["derived_futures_new_regime"] = (
        enriched["game_type"].astype(str).eq("F") & enriched["season"].ge(2023)
    ).astype("int8")
    return enriched


def make_pipeline(feature_columns: list[str], n_estimators: int, workers: int, seed: int) -> Pipeline:
    categorical = [column for column in CAT_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        verbose_feature_names_out=False,
    )
    classifier = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=14,
        min_samples_leaf=100,
        max_features=0.8,
        n_jobs=workers,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def fit_brier_logit_calibration(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    target = np.asarray(target, dtype=np.float64)
    raw = logit(probability)
    result = minimize(
        lambda parameters: float(
            np.mean((sigmoid(parameters[0] * raw + parameters[1]) - target) ** 2)
        ),
        x0=np.asarray([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=((0.1, 3.0), (-2.0, 2.0)),
        options={"maxiter": 300, "ftol": 1e-15},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[0]), float(result.x[1])


def calibrate_from_previous(
    previous: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, list[float]]]:
    output = np.asarray(current["raw"], dtype=np.float64).copy()
    parameters: dict[str, list[float]] = {}
    for group in ("R", "F"):
        previous_mask = np.asarray(previous["game_type"]).astype(str) == group
        current_mask = np.asarray(current["game_type"]).astype(str) == group
        if not current_mask.any():
            continue
        fit_mask = previous_mask if previous_mask.sum() >= 100 else np.ones(len(previous_mask), bool)
        slope, intercept = fit_brier_logit_calibration(
            np.asarray(previous["target"], dtype=np.float64)[fit_mask],
            np.asarray(previous["raw"], dtype=np.float64)[fit_mask],
        )
        output[current_mask] = sigmoid(slope * logit(output[current_mask]) + intercept)
        parameters[group] = [slope, intercept]
    return np.clip(output, 0.001, 0.999), parameters


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


def run(
    train_csv: Path,
    v285_axes: Path,
    v288_axes: Path,
    output_dir: Path,
    n_estimators: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    enriched = add_time_safe_features(train)
    feature_columns = [
        column for column in enriched.columns if column not in {"row_id", TARGET}
    ]
    folds: dict[int, dict[str, np.ndarray]] = {}
    model_records: dict[str, Any] = {}
    for year in (2021, 2022, 2023, 2024):
        train_mask = enriched["season"].lt(year).to_numpy()
        valid_mask = enriched["season"].eq(year).to_numpy()
        pipeline = make_pipeline(feature_columns, n_estimators, workers, seed + year)
        pipeline.fit(
            enriched.loc[train_mask, feature_columns],
            enriched.loc[train_mask, TARGET].to_numpy(np.float64),
        )
        extra = pipeline.predict_proba(enriched.loc[valid_mask, feature_columns])[:, 1]
        beta_frame, beta_probability, beta_recipe = predict_year(train, year)
        year_frame = train.loc[valid_mask].reset_index(drop=True)
        if not np.array_equal(
            beta_frame["row_id"].astype(str).to_numpy(),
            year_frame["row_id"].astype(str).to_numpy(),
        ):
            raise ValueError(f"Beta/Extra row mismatch for {year}")
        raw = np.clip(
            EXTRA_WEIGHT * extra + (1.0 - EXTRA_WEIGHT) * beta_probability,
            0.001,
            0.999,
        )
        folds[year] = {
            "target": year_frame[TARGET].to_numpy(np.float64),
            "game_type": year_frame["game_type"].astype(str).to_numpy(),
            "extra": extra,
            "beta": beta_probability,
            "raw": raw,
        }
        model_path = output_dir / f"extra_{year}.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        model_records[str(year)] = {
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "model_path": str(model_path),
            "beta_recipe": beta_recipe,
        }
        del pipeline
        gc.collect()

    calibration: dict[str, Any] = {}
    for year in (2022, 2023, 2024):
        calibrated, parameters = calibrate_from_previous(folds[year - 1], folds[year])
        folds[year]["calibrated"] = calibrated
        calibration[str(year)] = {"fit_year": year - 1, "parameters": parameters}

    with np.load(v285_axes, allow_pickle=False) as saved:
        baseline_2022 = saved["candidate_full_2022"].astype(np.float64)
    with np.load(v288_axes, allow_pickle=False) as saved:
        baseline_late_2023 = saved["candidate_late_2023"].astype(np.float64)
        baseline_2024 = saved["candidate_full_2024"].astype(np.float64)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    baselines = {
        "full_2022": baseline_2022,
        "late_2023": baseline_late_2023,
        "full_2024": baseline_2024,
    }
    complement = {
        "full_2022": folds[2022]["calibrated"],
        "late_2023": folds[2023]["calibrated"][
            train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
        ],
        "full_2024": folds[2024]["calibrated"],
    }
    for name in frames:
        if len(frames[name]) != len(baselines[name]) or len(frames[name]) != len(complement[name]):
            raise ValueError(f"axis alignment mismatch: {name}")

    selected_weights: dict[str, float] = {}
    source_grids: dict[str, Any] = {}
    for group in ("R", "F"):
        rows = []
        for weight in BLEND_WEIGHTS:
            gains = []
            for name in ("full_2022", "late_2023"):
                active = frames[name]["game_type"].astype(str).eq(group).to_numpy()
                candidate = baselines[name].copy()
                candidate[active] = (
                    (1.0 - weight) * candidate[active]
                    + weight * complement[name][active]
                )
                target = frames[name][TARGET].to_numpy(np.float64)
                gains.append(bss(target, candidate) - bss(target, baselines[name]))
            rows.append(
                {
                    "weight": weight,
                    "full_2022_gain": gains[0],
                    "late_2023_gain": gains[1],
                    "source_min_gain": float(min(gains)),
                    "source_mean_gain": float(np.mean(gains)),
                }
            )
        selected = max(
            rows,
            key=lambda row: (
                row["source_min_gain"], row["source_mean_gain"], -row["weight"]
            ),
        )
        selected_weights[group] = float(selected["weight"])
        source_grids[group] = {"selected": selected, "rows": rows}

    candidates: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, frame in frames.items():
        weights = (
            frame["game_type"].astype(str).map(selected_weights).fillna(0.0).to_numpy(np.float64)
        )
        candidates[name] = np.clip(
            (1.0 - weights) * baselines[name] + weights * complement[name],
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
        "n_estimators": n_estimators,
        "extra_weight": EXTRA_WEIGHT,
        "feature_count": len(feature_columns),
        "selected_route_weights": selected_weights,
        "source_grids": source_grids,
        "calibration": calibration,
        "models": model_records,
        "metrics": metrics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_followup": bool(source_pass and locked_pass),
        "restrictions": {
            "official_train_only": True,
            "external_predictions_used": False,
            "each_extra_fold_trains_only_on_prior_seasons": True,
            "calibration_fitted_on_previous_fold_only": True,
            "blend_selected_on_full_2022_and_late_2023_only": True,
            "full_2024_locked_confirmation": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"baseline_{name}": baselines[name] for name in frames},
        **{f"complement_{name}": complement[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
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
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3230)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.v285_axes,
                args.v288_axes,
                args.output_dir,
                args.n_estimators,
                args.workers,
                args.seed,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()

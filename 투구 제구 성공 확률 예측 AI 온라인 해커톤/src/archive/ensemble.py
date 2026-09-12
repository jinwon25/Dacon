"""Nested temporal ensembles and fixed statistical submission gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.archive.calibration import apply_logit_offset, fit_constrained_blend, forecast_base_rate
from src.archive.data import TARGET_COL, read_main
from src.archive.followup import _load_all_caches, _metric_row, _upsert_csv
from src.metrics import brier_score, cluster_bootstrap_delta, probability_logit


def _model_prediction(project_dir: Path, model_name: str, year: int) -> np.ndarray:
    path = (
        project_dir
        / "artifacts"
        / "followup"
        / "models"
        / f"{model_name}_validate_{year}.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        return saved["prediction"].astype(np.float64)


def _fit_logit_stack(
    predictions: list[np.ndarray],
    target: np.ndarray,
    c_value: float,
) -> LogisticRegression:
    features = np.column_stack([probability_logit(values) for values in predictions])
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(features, target)
    return model


def _predict_logit_stack(
    model: LogisticRegression,
    predictions: list[np.ndarray],
) -> np.ndarray:
    features = np.column_stack([probability_logit(values) for values in predictions])
    return model.predict_proba(features)[:, 1]


def _candidate_predictions(
    project_dir: Path,
    train: pd.DataFrame,
    folds: dict[int, Any],
    caches: dict[int, dict[str, Any]],
    years: list[int],
) -> tuple[dict[str, dict[int, np.ndarray]], list[dict[str, Any]]]:
    all_years = sorted(caches)
    hierarchy = {
        year: _model_prediction(project_dir, "hier_backoff_a100_200", year)
        for year in years
    }
    hierarchy_history = {
        year: _model_prediction(project_dir, "hier_backoff_a100_200", year)
        for year in all_years
        if (
            project_dir
            / "artifacts"
            / "followup"
            / "models"
            / f"hier_backoff_a100_200_validate_{year}.npz"
        ).exists()
    }
    # The 2020 hierarchical cache was not needed in Wave 2. Use the incumbent
    # as a conservative anchor for the earliest nested fit instead of peeking.
    if 2020 not in hierarchy_history:
        hierarchy_history[2020] = caches[2020]["incumbent"]

    candidates: dict[str, dict[int, np.ndarray]] = {
        "incumbent": {},
        "raw_blend": {},
        "damped_3_0.8_fallback_raw": {},
    }
    detail_rows: list[dict[str, Any]] = []
    for year in years:
        candidates["incumbent"][year] = caches[year]["incumbent"]
        candidates["raw_blend"][year] = caches[year]["blend_raw"]
        rates = (
            train.iloc[folds[year].train_idx]
            .groupby("season", observed=True)[TARGET_COL]
            .mean()
            .sort_index()
        )
        if len(rates) >= 3:
            forecast = forecast_base_rate(rates, year, "damped_3_0.8")
            latest = float(rates.iloc[-1])
            offset = float(
                probability_logit(np.array([forecast]))[0]
                - probability_logit(np.array([latest]))[0]
            )
            candidates["damped_3_0.8_fallback_raw"][year] = apply_logit_offset(
                caches[year]["blend_raw"], offset
            )
        else:
            candidates["damped_3_0.8_fallback_raw"][year] = caches[year]["blend_raw"]

    blend_specs = {
        "nested_convex_trend": ("trend", [1e-4, 1e-3, 1e-2]),
        "nested_convex_regime": ("regime", [1e-4, 1e-3, 1e-2]),
    }
    for base_name, (kind, penalties) in blend_specs.items():
        for penalty in penalties:
            candidate_name = f"{base_name}_l2_{penalty:g}"
            candidates[candidate_name] = {}
            for year in years:
                history_years = [history for history in all_years if history < year]
                if kind == "trend":
                    history_bases = [
                        np.concatenate([caches[h]["lgb_trend"] for h in history_years]),
                        np.concatenate([caches[h]["rf_trend"] for h in history_years]),
                        np.concatenate([hierarchy_history[h] for h in history_years]),
                    ]
                    current_bases = [
                        caches[year]["lgb_trend"],
                        caches[year]["rf_trend"],
                        hierarchy[year],
                    ]
                    anchor = np.array([0.35, 0.65, 0.0])
                else:
                    history_bases = [
                        np.concatenate([caches[h]["blend_raw"] for h in history_years]),
                        np.concatenate([caches[h]["incumbent"] for h in history_years]),
                        np.concatenate([hierarchy_history[h] for h in history_years]),
                    ]
                    current_bases = [
                        caches[year]["blend_raw"],
                        caches[year]["incumbent"],
                        hierarchy[year],
                    ]
                    anchor = np.array([0.0, 1.0, 0.0])
                history_target = np.concatenate([caches[h]["target"] for h in history_years])
                weight, _ = fit_constrained_blend(
                    history_bases,
                    history_target,
                    l2=penalty,
                    anchor=anchor,
                )
                candidates[candidate_name][year] = sum(
                    value * prediction
                    for value, prediction in zip(weight, current_bases)
                )
                detail_rows.append(
                    {
                        "candidate": candidate_name,
                        "outer_validation_season": year,
                        "fit_seasons": ";".join(str(value) for value in history_years),
                        "parameters": json.dumps(
                            {"l2": penalty, "weights": weight.tolist()}, sort_keys=True
                        ),
                    }
                )

    for c_value in (0.001, 0.01, 0.1):
        candidate_name = f"nested_logit_stack_c_{c_value:g}"
        candidates[candidate_name] = {}
        for year in years:
            history_years = [history for history in all_years if history < year]
            history_bases = [
                np.concatenate([caches[h]["lgb_trend"] for h in history_years]),
                np.concatenate([caches[h]["rf_trend"] for h in history_years]),
                np.concatenate([hierarchy_history[h] for h in history_years]),
            ]
            current_bases = [
                caches[year]["lgb_trend"],
                caches[year]["rf_trend"],
                hierarchy[year],
            ]
            history_target = np.concatenate([caches[h]["target"] for h in history_years])
            model = _fit_logit_stack(history_bases, history_target, c_value)
            candidates[candidate_name][year] = _predict_logit_stack(model, current_bases)
            detail_rows.append(
                {
                    "candidate": candidate_name,
                    "outer_validation_season": year,
                    "fit_seasons": ";".join(str(value) for value in history_years),
                    "parameters": json.dumps(
                        {
                            "C": c_value,
                            "coefficients": model.coef_[0].tolist(),
                            "intercept": float(model.intercept_[0]),
                        },
                        sort_keys=True,
                    ),
                }
            )
    return candidates, detail_rows


def run(project_dir: Path) -> None:
    followup_config = json.loads(
        (project_dir / "research" / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    train = read_main(project_dir / "data" / "train.csv")
    folds, caches = _load_all_caches(project_dir, followup_config, train)
    years = [int(value) for value in followup_config["outer_validation_seasons"]]
    candidates, detail_rows = _candidate_predictions(
        project_dir, train, folds, caches, years
    )
    result_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    weights = np.array(
        [float(followup_config["recency_weights"][str(year)]) for year in years]
    )
    gate = followup_config["submission_gate"]
    incumbent_concat = np.concatenate([caches[year]["incumbent"] for year in years])
    target_concat = np.concatenate([caches[year]["target"] for year in years])
    frame_concat = pd.concat(
        [train.iloc[folds[year].valid_idx] for year in years],
        ignore_index=True,
    )
    pitcher_season = (
        frame_concat["pitcher_id"].astype("string")
        + "-"
        + frame_concat["season"].astype("string")
    )
    pitcher = frame_concat["pitcher_id"].astype("string")

    for candidate_name, predictions_by_year in candidates.items():
        deltas = []
        for year in years:
            prediction = predictions_by_year[year]
            target = caches[year]["target"]
            incumbent = caches[year]["incumbent"]
            result_rows.append(
                _metric_row(
                    f"W5_{candidate_name}_{year}",
                    year,
                    candidate_name,
                    prediction,
                    target,
                    incumbent,
                    diagnostic=False,
                    prediction_source="nested earlier-year OOF or fixed train-only rule",
                )
            )
            deltas.append(brier_score(target, prediction) - brier_score(target, incumbent))
        candidate_concat = np.concatenate(
            [predictions_by_year[year] for year in years]
        )
        bootstrap = {}
        for cluster_name, clusters in (
            ("pitcher_season", pitcher_season),
            ("pitcher", pitcher),
        ):
            values = cluster_bootstrap_delta(
                target_concat,
                candidate_concat,
                incumbent_concat,
                clusters,
                n_resamples=int(followup_config["bootstrap_resamples"]),
                seed=int(followup_config["seed"]) + 500,
            )
            bootstrap_rows.append(
                {
                    "experiment_id": f"W5_{candidate_name}_combined",
                    "outer_validation_season": "2021-2024",
                    "candidate": candidate_name,
                    "reference": "incumbent",
                    "cluster_type": cluster_name,
                    **values,
                }
            )
            bootstrap[cluster_name] = values
        recency_delta = float(np.average(deltas, weights=weights))
        delta_2024 = float(deltas[-1])
        worst_delta = float(max(deltas))
        probability = float(
            bootstrap["pitcher_season"]["improvement_probability"]
        )
        checks = {
            "recency_improvement": recency_delta
            <= -float(gate["minimum_recency_weighted_improvement"]),
            "latest_improvement": delta_2024
            <= -float(gate["minimum_2024_improvement"]),
            "worst_fold": worst_delta
            <= float(gate["maximum_single_fold_worsening"]),
            "bootstrap": probability
            >= float(gate["minimum_cluster_improvement_probability"]),
        }
        gate_rows.append(
            {
                "candidate": candidate_name,
                "delta_2021": deltas[0],
                "delta_2022": deltas[1],
                "delta_2023": deltas[2],
                "delta_2024": deltas[3],
                "mean_delta": float(np.mean(deltas)),
                "recency_weighted_delta": recency_delta,
                "worst_fold_delta": worst_delta,
                "pitcher_season_improvement_probability": probability,
                **{f"pass_{name}": value for name, value in checks.items()},
                "passes_statistical_gate": all(checks.values()),
            }
        )

    _upsert_csv(
        project_dir / "research" / "reports" / "walk_forward_results.csv",
        pd.DataFrame(result_rows),
        ["experiment_id"],
    )
    detail = pd.DataFrame(detail_rows)
    scores = pd.DataFrame(gate_rows)
    _upsert_csv(
        project_dir / "research" / "reports" / "ensemble_results.csv",
        scores,
        ["candidate"],
    )
    _upsert_csv(
        project_dir / "research" / "reports" / "ensemble_parameters.csv",
        detail,
        ["candidate", "outer_validation_season"],
    )
    _upsert_csv(
        project_dir / "research" / "reports" / "bootstrap_results.csv",
        pd.DataFrame(bootstrap_rows),
        ["experiment_id", "cluster_type"],
    )
    print(scores.sort_values("recency_weighted_delta").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.project_dir.resolve())


if __name__ == "__main__":
    main()

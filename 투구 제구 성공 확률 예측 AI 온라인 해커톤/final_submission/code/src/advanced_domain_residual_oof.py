"""Forward OOF screen for a richer, row-local baseball residual model.

The experiment learns ``target - V2`` exclusively from completed earlier
outer-season OOF rows.  Model/eta choices for an audit season are made only
from the immediately preceding outer season.  Evaluation-batch aggregates
are never used.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.advanced_domain_features import (
    META_COLUMNS,
    add_base_prediction_features,
    build_advanced_domain_features,
)
from src.data import read_main
from src.metrics import brier_score, cluster_bootstrap_delta


YEARS = (2021, 2022, 2023, 2024)
AUDIT_YEARS = (2022, 2023, 2024)
ETAS = (0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 1.0, 1.25)
CURRENT_ETA = {2022: 0.25, 2023: 0.50, 2024: 1.00}


@dataclass(frozen=True)
class Config:
    name: str
    feature_set: str
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int
    rounds: int
    fit_regular_only: bool = False
    recency_half_life: float | None = None


CONFIGS = (
    Config("compact_l15", "compact", 15, 5, 2000, 160),
    Config("expanded_l15", "expanded", 15, 5, 2000, 180),
    Config("expanded_l31", "expanded", 31, 6, 1000, 220),
    Config("expanded_l63", "expanded", 63, 8, 500, 240),
    Config("expanded_l31_r", "expanded", 31, 6, 1000, 220, True),
    Config("expanded_l31_h2", "expanded", 31, 6, 1000, 220, False, 2.0),
    Config("team_l31", "team", 31, 6, 1000, 220),
)


def _load_folds(train: pd.DataFrame, corrected_cb_dir: Path) -> dict[int, pd.DataFrame]:
    train_ids = pd.Index(train["row_id"].astype(str))
    if train_ids.has_duplicates:
        raise ValueError("train row_id is not unique")
    folds: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        path = corrected_cb_dir / f"corrected_cb_o{year}.npz"
        with np.load(path, allow_pickle=True) as bundle:
            row_id = bundle["row_id"].astype(str)
            target = bundle["target"].astype(np.float64)
            v2 = bundle["v2"].astype(np.float64)
            game_type = bundle["game_type"].astype(str)
        index = train_ids.get_indexer(row_id)
        if (index < 0).any():
            raise ValueError(f"{year} OOF row missing from train")
        expected = train.iloc[index]["control_success"].to_numpy(np.float64)
        if not np.array_equal(target, expected):
            raise ValueError(f"{year} OOF target mismatch")
        folds[year] = pd.DataFrame(
            {
                "row_id": row_id,
                "train_index": index,
                "target": target,
                "v2": v2,
                "game_type": game_type,
                "pitcher_id": train.iloc[index]["pitcher_id"].astype(str).to_numpy(),
            }
        )
    return folds


def _current_prediction(
    fold: pd.DataFrame,
    correction_dir: Path,
    year: int,
) -> np.ndarray:
    mode = "r_only" if year == 2022 else "all"
    path = correction_dir / f"correction_corrected_state_{mode}_o{year}.npz"
    with np.load(path, allow_pickle=True) as bundle:
        row_id = bundle["row_id"].astype(str)
        correction = bundle["correction"].astype(np.float64)
    regular = fold["game_type"].eq("R").to_numpy()
    expected = fold.loc[regular, "row_id"].astype(str).to_numpy()
    if not np.array_equal(row_id, expected):
        raise ValueError(f"existing correction row mismatch for {year}")
    prediction = fold["v2"].to_numpy(np.float64, copy=True)
    prediction[regular] = np.clip(
        prediction[regular] + CURRENT_ETA[year] * correction,
        1e-6,
        1.0 - 1e-6,
    )
    return prediction


def _feature_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    team_columns = {"pitcher_team_id", "batter_team_id", "team_matchup"}
    if feature_set == "team":
        return list(features.columns)
    if feature_set == "expanded":
        return [column for column in features.columns if column not in team_columns]
    if feature_set != "compact":
        raise ValueError(f"unknown feature set {feature_set}")
    wanted = {
        *META_COLUMNS,
        "season",
        "game_month",
        "inning",
        "balls_before",
        "strikes_before",
        "outs_before",
        "score_diff_pitcher_team",
        "num_runners_on",
        "li_log",
        "count_state",
        "platoon",
        "count_platoon",
        "game_type",
        "top_bottom",
        "base_state",
        "pitcher_hand",
        "batter_hand",
        "pitcher_season_n",
        "pitcher_prior_rate",
        "pitcher_season_rate",
        "pitcher_season_logit_delta",
        "batter_prior_rate",
        "batter_season_rate",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_batter_success_rate",
        "recent_success_mean",
        "recent_middle_mean",
        "recent_success_1_minus_5",
        "recent_success_3_minus_5",
        "recent_success_3_minus_career",
        "recent_middle_1_minus_5",
        "recent_middle_3_minus_career",
        "pitcher_batter_success_gap",
        "pitcher_batter_middle_gap",
        "command_risk_sum",
        "command_risk_balance",
        "pitcher_confidence",
        "pitcher_season_confidence",
        "season_pitch_pace",
        "count_balance",
        "two_strike",
        "three_ball",
        "full_count",
        "count_pressure",
        "three_ball_pressure",
        "bases_loaded",
        "absolute_score_diff",
    }
    return [column for column in features.columns if column in wanted]


def _align_categories(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fit = fit.copy()
    audit = audit.copy()
    categorical = list(
        fit.select_dtypes(include=["category", "object", "string"]).columns
    )
    for column in categorical:
        fit_values = fit[column].astype("string").fillna("__MISSING__")
        vocabulary = pd.Index(fit_values.unique()).astype(str)
        fit[column] = pd.Categorical(fit_values, categories=vocabulary)
        audit[column] = pd.Categorical(
            audit[column].astype("string").fillna("__MISSING__"),
            categories=vocabulary,
        )
    return fit, audit, categorical


def _fit_correction(
    config: Config,
    fit_features: pd.DataFrame,
    fit_residual: np.ndarray,
    audit_features: pd.DataFrame,
    fit_years: np.ndarray,
    audit_year: int,
) -> tuple[np.ndarray, float]:
    columns = _feature_columns(fit_features, config.feature_set)
    fit = fit_features[columns]
    audit = audit_features[columns]
    fit, audit, categorical = _align_categories(fit, audit)
    weights = None
    if config.recency_half_life is not None:
        latest = audit_year - 1
        weights = np.power(
            0.5,
            np.maximum(latest - fit_years, 0) / config.recency_half_life,
        ).astype(np.float32)
    dataset = lgb.Dataset(
        fit,
        label=fit_residual,
        weight=weights,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    started = time.perf_counter()
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.025,
            "num_leaves": config.num_leaves,
            "max_depth": config.max_depth,
            "min_data_in_leaf": config.min_data_in_leaf,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 75.0,
            "verbosity": -1,
            "seed": 20260821,
            "num_threads": 6,
            "deterministic": True,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=config.rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    correction = np.clip(
        np.asarray(model.predict(audit), dtype=np.float64), -0.05, 0.05
    )
    return correction, time.perf_counter() - started


def _candidate_prediction(
    fold: pd.DataFrame,
    correction: np.ndarray,
    eta: float,
) -> np.ndarray:
    prediction = fold["v2"].to_numpy(np.float64, copy=True)
    regular = fold["game_type"].eq("R").to_numpy()
    prediction[regular] = np.clip(
        prediction[regular] + eta * correction[regular],
        1e-6,
        1.0 - 1e-6,
    )
    return prediction


def _choice(metrics: pd.DataFrame, selection_year: int) -> pd.Series:
    candidates = metrics.loc[
        metrics["season"].eq(selection_year) & metrics["eta"].gt(0)
    ].copy()
    candidates["complexity"] = candidates["config"].map(
        {config.name: index for index, config in enumerate(CONFIGS)}
    )
    best = candidates["delta_vs_current"].min()
    candidates = candidates.loc[candidates["delta_vs_current"].le(best + 1e-15)]
    return candidates.sort_values(
        ["delta_vs_current", "complexity", "eta"], kind="mergesort"
    ).iloc[0]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(
            f"{value:.12g}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        )
        + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def run(
    data_project: Path,
    corrected_cb_dir: Path,
    correction_dir: Path,
    out_dir: Path,
    n_resamples: int = 5000,
) -> None:
    data_project = data_project.resolve()
    corrected_cb_dir = corrected_cb_dir.resolve()
    correction_dir = correction_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(train, corrected_cb_dir)
    domain = build_advanced_domain_features(train, train, include_teams=True)
    current = {
        year: _current_prediction(folds[year], correction_dir, year)
        for year in AUDIT_YEARS
    }
    corrections: dict[tuple[int, str], np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []

    for audit_year in AUDIT_YEARS:
        fit_year_list = tuple(year for year in YEARS if year < audit_year)
        fit_indices = np.concatenate(
            [folds[year]["train_index"].to_numpy(np.int64) for year in fit_year_list]
        )
        fit_probability = np.concatenate(
            [folds[year]["v2"].to_numpy(np.float64) for year in fit_year_list]
        )
        fit_residual = np.concatenate(
            [
                folds[year]["target"].to_numpy(np.float64)
                - folds[year]["v2"].to_numpy(np.float64)
                for year in fit_year_list
            ]
        )
        fit_years = np.concatenate(
            [np.full(len(folds[year]), year, dtype=np.int16) for year in fit_year_list]
        )
        audit_indices = folds[audit_year]["train_index"].to_numpy(np.int64)
        fit_features = add_base_prediction_features(
            domain.iloc[fit_indices].reset_index(drop=True), fit_probability
        )
        audit_features = add_base_prediction_features(
            domain.iloc[audit_indices].reset_index(drop=True),
            folds[audit_year]["v2"].to_numpy(np.float64),
        )
        target = folds[audit_year]["target"].to_numpy(np.float64)
        v2 = folds[audit_year]["v2"].to_numpy(np.float64)
        current_brier = brier_score(target, current[audit_year])
        regular_audit = folds[audit_year]["game_type"].eq("R").to_numpy()

        for config in CONFIGS:
            mask = np.ones(len(fit_residual), dtype=bool)
            if config.fit_regular_only:
                mask = np.concatenate(
                    [folds[year]["game_type"].eq("R").to_numpy() for year in fit_year_list]
                )
            correction, seconds = _fit_correction(
                config,
                fit_features.loc[mask].reset_index(drop=True),
                fit_residual[mask],
                audit_features,
                fit_years[mask],
                audit_year,
            )
            corrections[(audit_year, config.name)] = correction
            np.savez_compressed(
                out_dir / f"correction_{config.name}_o{audit_year}.npz",
                row_id=folds[audit_year]["row_id"].astype(str).to_numpy(),
                correction=correction,
            )
            for eta in ETAS:
                prediction = _candidate_prediction(folds[audit_year], correction, eta)
                brier = brier_score(target, prediction)
                metric_rows.append(
                    {
                        "season": audit_year,
                        "config": config.name,
                        "feature_set": config.feature_set,
                        "fit_regular_only": config.fit_regular_only,
                        "recency_half_life": config.recency_half_life,
                        "eta": eta,
                        "brier": brier,
                        "delta_vs_v2": brier - brier_score(target, v2),
                        "delta_vs_current": brier - current_brier,
                        "r_delta_vs_current": brier_score(
                            target[regular_audit], prediction[regular_audit]
                        )
                        - brier_score(
                            target[regular_audit], current[audit_year][regular_audit]
                        ),
                        "fit_rows": int(mask.sum()),
                        "fit_seconds": seconds,
                        "outer_target_used_for_fit": False,
                    }
                )
            print(
                f"audit={audit_year} config={config.name} "
                f"seconds={seconds:.1f}",
                flush=True,
            )
        del fit_features, audit_features
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "grid_metrics.csv", index=False)

    forward_rows = []
    forward_predictions: dict[int, np.ndarray] = {}
    for audit_year, selection_year in ((2023, 2022), (2024, 2023)):
        choice = _choice(metrics, selection_year)
        config = str(choice["config"])
        eta = float(choice["eta"])
        prediction = _candidate_prediction(
            folds[audit_year], corrections[(audit_year, config)], eta
        )
        forward_predictions[audit_year] = prediction
        target = folds[audit_year]["target"].to_numpy(np.float64)
        forward_rows.append(
            {
                "audit_season": audit_year,
                "selection_season": selection_year,
                "config": config,
                "eta": eta,
                "selection_delta_vs_current": float(choice["delta_vs_current"]),
                "audit_delta_vs_current": brier_score(target, prediction)
                - brier_score(target, current[audit_year]),
                "audit_delta_vs_v2": brier_score(target, prediction)
                - brier_score(target, folds[audit_year]["v2"]),
                "outer_target_used_for_selection": False,
            }
        )
    forward = pd.DataFrame(forward_rows)
    forward.to_csv(out_dir / "forward_selection.csv", index=False)

    final_choice = _choice(metrics, 2024)
    final = {
        "selection_season": 2024,
        "config": str(final_choice["config"]),
        "eta": float(final_choice["eta"]),
        "selection_delta_vs_current": float(final_choice["delta_vs_current"]),
        "selection_delta_vs_v2": float(final_choice["delta_vs_v2"]),
        "intended_audit_season": 2025,
        "outer_target_used_for_selection": False,
    }
    (out_dir / "final_choice.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )

    o24_prediction = forward_predictions[2024]
    bootstrap = cluster_bootstrap_delta(
        folds[2024]["target"].to_numpy(),
        o24_prediction,
        current[2024],
        folds[2024]["pitcher_id"].astype(str).to_numpy(),
        n_resamples=n_resamples,
        seed=20260821,
    )
    (out_dir / "forward_o24_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "configs": [config.__dict__ for config in CONFIGS],
        "etas": ETAS,
        "current_eta": CURRENT_ETA,
        "training_rule": "earlier outer OOF rows only",
        "selection_rule": "immediately previous outer season",
        "test_batch_aggregates_used": False,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report = (
        "# Advanced row-local domain residual OOF\n\n"
        "The candidate uses only the current row, official-training prior-season "
        "endpoints, and a frozen base probability.\n\n"
        "## Forward audit\n\n"
        + _markdown_table(forward)
        + "\n\n## O24 paired pitcher bootstrap versus current residual\n\n```json\n"
        + json.dumps(bootstrap, indent=2)
        + "\n```\n\n## 2025 choice from completed O24\n\n```json\n"
        + json.dumps(final, indent=2)
        + "\n```\n"
    )
    (out_dir / "decision.md").write_text(report, encoding="utf-8")
    print(forward.to_string(index=False), flush=True)
    print(json.dumps(bootstrap, indent=2), flush=True)
    print(json.dumps(final, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, default=Path("."))
    parser.add_argument("--corrected-cb-dir", type=Path, required=True)
    parser.add_argument("--correction-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    run(
        args.data_project,
        args.corrected_cb_dir,
        args.correction_dir,
        args.out_dir,
        args.n_resamples,
    )


if __name__ == "__main__":
    main()

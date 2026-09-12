"""R_CORE CatBoost screen with exact current-season ASOF state and player IDs."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.recent_shared_exact_asof import _feature_pair
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.archive.v14_component_audit import bss, load_folds


TRANSITIONS = ((2022, 2023), (2023, 2024))
CATEGORICAL = [
    "game_dayofweek",
    "top_bottom",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
    "count_state",
    "platoon",
    "pitcher_count",
    "pitcher_batter_hand",
    "batter_count",
    "team_matchup",
]


def _cat_frame(
    numeric: pd.DataFrame, raw: pd.DataFrame, base: np.ndarray | None
) -> pd.DataFrame:
    output = numeric.reset_index(drop=True).copy()
    count = raw["balls_before"].astype(str) + "-" + raw["strikes_before"].astype(str)
    derived = {
        "game_dayofweek": raw["game_dayofweek"],
        "top_bottom": raw["top_bottom"],
        "base_state": raw["base_state"],
        "pitcher_hand": raw["pitcher_hand"],
        "batter_hand": raw["batter_hand"],
        "pitcher_team_id": raw["pitcher_team_id"],
        "batter_team_id": raw["batter_team_id"],
        "pitcher_id": raw["pitcher_id"],
        "batter_id": raw["batter_id"],
        "count_state": count,
        "platoon": raw["pitcher_hand"].astype(str) + "-" + raw["batter_hand"].astype(str),
        "pitcher_count": raw["pitcher_id"].astype(str) + "-" + count,
        "pitcher_batter_hand": raw["pitcher_id"].astype(str)
        + "-"
        + raw["batter_hand"].astype(str),
        "batter_count": raw["batter_id"].astype(str) + "-" + count,
        "team_matchup": raw["pitcher_team_id"].astype(str)
        + "-"
        + raw["batter_team_id"].astype(str),
    }
    for column, values in derived.items():
        output[column] = (
            values.astype("string").fillna("__MISSING__").astype(str).to_numpy()
        )
    if base is not None:
        output["frozen_base_probability"] = np.asarray(base, dtype=np.float64)
    return output


def _common_params(seed: int) -> dict[str, object]:
    return {
        "iterations": 260,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 20.0,
        "random_strength": 0.7,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "random_seed": seed,
        "thread_count": 6,
        "allow_writing_files": False,
        "task_type": "CPU",
        "verbose": 100,
    }


def run(project: Path, exact_dir: Path, component_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    exact_dir = exact_dir.resolve()
    component_dir = component_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds = load_folds(train, exact_dir)
    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for source_year, audit_year in TRANSITIONS:
        source = train.loc[train["season"].eq(source_year)].reset_index(drop=True)
        audit = train.loc[train["season"].eq(audit_year)].reset_index(drop=True)
        source_fold = folds[source_year]
        audit_fold = folds[audit_year]
        source_core = source["domain3"].eq("R_CORE").to_numpy()
        audit_core = audit["domain3"].eq("R_CORE").to_numpy()
        numeric_source, numeric_audit = _feature_pair(
            train,
            source,
            audit,
            source_year,
            audit_year,
            include_categories=False,
        )
        fit_frame = _cat_frame(
            numeric_source.loc[source_core].reset_index(drop=True),
            source.loc[source_core].reset_index(drop=True),
            source_fold.base[source_core],
        )
        audit_frame = _cat_frame(
            numeric_audit.loc[audit_core].reset_index(drop=True),
            audit.loc[audit_core].reset_index(drop=True),
            audit_fold.base[audit_core],
        )
        y_source = source_fold.target[source_core]
        y_audit = audit_fold.target[audit_core]
        base_source = source_fold.base[source_core]
        base_audit = audit_fold.base[audit_core]
        with np.load(component_dir / f"v13_components_o{audit_year}.npz") as saved:
            if not np.array_equal(saved["train_index"], audit_fold.train_index):
                raise ValueError("v13 component order mismatch")
            v13 = saved["v13"].astype(np.float64)[audit_core]
        fit_pool_direct = Pool(
            fit_frame, label=y_source, cat_features=CATEGORICAL
        )
        audit_pool = Pool(audit_frame, cat_features=CATEGORICAL)
        predictions: dict[str, np.ndarray] = {}

        started = time.perf_counter()
        classifier = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="BrierScore",
            **_common_params(42),
        )
        classifier.fit(fit_pool_direct)
        predictions["catboost_logloss"] = classifier.predict_proba(audit_pool)[:, 1]
        classifier.save_model(str(output_dir / f"catboost_logloss_s{source_year}.cbm"))
        fit_rows.append(
            {
                "source_year": source_year,
                "audit_year": audit_year,
                "candidate": "catboost_logloss",
                "seconds": time.perf_counter() - started,
            }
        )
        del classifier

        started = time.perf_counter()
        direct = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            **_common_params(142),
        )
        direct.fit(fit_pool_direct)
        predictions["catboost_brier_direct"] = np.clip(
            direct.predict(audit_pool), 0.001, 0.999
        )
        direct.save_model(str(output_dir / f"catboost_brier_direct_s{source_year}.cbm"))
        fit_rows.append(
            {
                "source_year": source_year,
                "audit_year": audit_year,
                "candidate": "catboost_brier_direct",
                "seconds": time.perf_counter() - started,
            }
        )
        del direct

        started = time.perf_counter()
        residual_pool = Pool(
            fit_frame,
            label=y_source - base_source,
            cat_features=CATEGORICAL,
        )
        residual = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            **_common_params(242),
        )
        residual.fit(residual_pool)
        correction = np.clip(residual.predict(audit_pool), -0.08, 0.08)
        predictions["catboost_brier_residual"] = np.clip(
            base_audit + correction, 0.001, 0.999
        )
        residual.save_model(str(output_dir / f"catboost_brier_residual_s{source_year}.cbm"))
        fit_rows.append(
            {
                "source_year": source_year,
                "audit_year": audit_year,
                "candidate": "catboost_brier_residual",
                "seconds": time.perf_counter() - started,
            }
        )
        del residual, fit_pool_direct, residual_pool, audit_pool

        np.savez_compressed(
            output_dir / f"catboost_exact_o{audit_year}.npz",
            train_index=audit_fold.train_index[audit_core],
            target=y_audit,
            base=base_audit,
            v13=v13,
            exact_lgb=audit_fold.exact_lgb[audit_core],
            **predictions,
        )
        for name, prediction in predictions.items():
            recipes = {"standalone": prediction}
            for weight in (0.05, 0.10, 0.20, 0.35, 0.50, 0.65):
                recipes[f"blend_v13_{weight:.2f}"] = np.clip(
                    v13 + weight * (prediction - v13), 1e-6, 1.0 - 1e-6
                )
                recipes[f"add_from_base_{weight:.2f}"] = np.clip(
                    v13 + weight * (prediction - base_audit), 1e-6, 1.0 - 1e-6
                )
                recipes[f"replace_exact_{weight:.2f}"] = np.clip(
                    v13
                    + weight * (prediction - audit_fold.exact_lgb[audit_core]),
                    1e-6,
                    1.0 - 1e-6,
                )
            for recipe, candidate in recipes.items():
                metric_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "candidate": name,
                        "recipe": recipe,
                        "n_rows": len(y_audit),
                        "bss": bss(y_audit, candidate),
                        "gain_vs_base": bss(y_audit, candidate)
                        - bss(y_audit, base_audit),
                        "gain_vs_v13": bss(y_audit, candidate) - bss(y_audit, v13),
                    }
                )
        del fit_frame, audit_frame, numeric_source, numeric_audit, predictions
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "timings.csv", index=False)
    robust = (
        metrics.groupby(["candidate", "recipe"], observed=True)["gain_vs_v13"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=[False, False])
    )
    robust.to_csv(output_dir / "robust_vs_v13.csv", index=False)
    passing = robust.loc[robust["min_gain"].gt(0)]
    summary = {
        "protocol": "V14_CATBOOST_EXACT_PLAYER_STATE_SCREEN_V1",
        "transitions": [f"{a}->{b}" for a, b in TRANSITIONS],
        "n_strict_improvements": int(len(passing)),
        "best": robust.head(15).to_dict(orient="records"),
        "selection_eligible": bool(len(passing)),
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
        "--exact-dir",
        type=Path,
        default=Path("artifacts/recent_shared_exact_asof_20260815_02"),
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=Path("artifacts/v14_component_audit_20260815_01"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.exact_dir, args.component_dir, args.output_dir)


if __name__ == "__main__":
    main()

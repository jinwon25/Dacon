"""Strict next-season screen for low-variance v14 residual corrections.

The candidate for an audit season is fitted only on the immediately preceding
labelled season.  Corrections are intentionally simple: domain calibration or
smoothed group residual means.  This keeps the experiment row-local at
inference and makes unseen entities fall back to no change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.evaluate_v14_v15_robust import _candidate_fold
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.archive.v14_component_audit import Fold, V13_CORE, V13_F_WEIGHT, fit_stable_component, predict_recipe


TRANSITIONS = ((2022, 2023), (2023, 2024))
EPSILON = 1e-6


def _bss_gain(target: np.ndarray, candidate: np.ndarray, incumbent: np.ndarray) -> float:
    reference = float(target.mean() * (1.0 - target.mean()))
    improvement = np.square(incumbent - target) - np.square(candidate - target)
    return 100_000.0 * float(improvement.mean()) / reference


def _fold_from_cache(project: Path, train: pd.DataFrame, year: int) -> Fold:
    path = project / "artifacts" / "recent_shared_exact_asof_20260815_02" / f"recent_shared_o{year}.npz"
    with np.load(path) as saved:
        index = saved["train_index"].astype(np.int64)
        return Fold(
            year=year,
            train_index=index,
            target=saved["target"].astype(np.float64),
            base=saved["base"].astype(np.float64),
            exact_lgb=saved["exact_lgb"].astype(np.float64),
            exact_ridge=saved["exact_ridge"].astype(np.float64),
            trend_lgb=saved["trend_lgb"].astype(np.float64),
            rows=train.iloc[index].reset_index(drop=True),
        )


def _source_2021(project: Path, train: pd.DataFrame) -> Fold:
    path = project / "artifacts" / "followup" / "oof" / "wave0_incumbent_validate_2021.npz"
    with np.load(path) as saved:
        index = saved["valid_idx"].astype(np.int64)
        target = saved["target"].astype(np.float64)
        base = saved["incumbent"].astype(np.float64)
    zeros = np.zeros_like(base)
    return Fold(2021, index, target, base, zeros, zeros, zeros, train.iloc[index].reset_index(drop=True))


def load_v14_folds(project: Path, train: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    folds = {year: _fold_from_cache(project, train, year) for year in (2022, 2023, 2024)}
    source_2021 = _source_2021(project, train)
    source_bias, stable = fit_stable_component(train, source_2021, folds[2022])
    v13_2022 = predict_recipe(
        folds[2022], source_bias, stable, **V13_CORE, f_weight=V13_F_WEIGHT
    )
    output: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for year in (2022, 2023, 2024):
        rows, target, reconstructed_incumbent, v14 = _candidate_fold(project, train, year, 0.15)
        if year == 2022:
            # _candidate_fold deliberately uses a placeholder for unchanged
            # R_CORE in 2022.  Add its exact v14-v13 delta to the reconstructed
            # v13 analogue so every row is available to the correction screen.
            v14 = np.clip(v13_2022 + (v14 - reconstructed_incumbent), EPSILON, 1.0 - EPSILON)
        output[year] = (rows, target, v14)
    return output


def _domain_models(
    source_rows: pd.DataFrame,
    source_target: np.ndarray,
    source_prediction: np.ndarray,
    audit_rows: pd.DataFrame,
    audit_prediction: np.ndarray,
    method: str,
) -> np.ndarray:
    calibrated = np.asarray(source_prediction[:0], dtype=np.float64)
    output = np.zeros(len(audit_rows), dtype=np.float64)
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        source_mask = source_rows["domain3"].eq(domain).to_numpy()
        audit_mask = audit_rows["domain3"].eq(domain).to_numpy()
        p_fit = np.clip(source_prediction[source_mask], EPSILON, 1.0 - EPSILON)
        y_fit = source_target[source_mask]
        p_audit = np.clip(audit_prediction[audit_mask], EPSILON, 1.0 - EPSILON)
        if method == "affine":
            model = Ridge(alpha=100.0, fit_intercept=True)
            model.fit(p_fit.reshape(-1, 1), y_fit)
            calibrated = model.predict(p_audit.reshape(-1, 1))
        else:
            if method == "platt":
                x_fit = np.log(p_fit / (1.0 - p_fit)).reshape(-1, 1)
                x_audit = np.log(p_audit / (1.0 - p_audit)).reshape(-1, 1)
            elif method == "beta":
                x_fit = np.column_stack((np.log(p_fit), -np.log1p(-p_fit)))
                x_audit = np.column_stack((np.log(p_audit), -np.log1p(-p_audit)))
            else:
                raise ValueError(method)
            model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=500)
            model.fit(x_fit, y_fit)
            calibrated = model.predict_proba(x_audit)[:, 1]
        output[audit_mask] = np.clip(calibrated, EPSILON, 1.0 - EPSILON)
    return output


def _keys(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    pieces = [frame[column].astype("string").fillna("__MISSING__") for column in columns]
    key = pieces[0]
    for piece in pieces[1:]:
        key = key + "\x1f" + piece
    return key


def _smoothed_residual(
    source_rows: pd.DataFrame,
    source_target: np.ndarray,
    source_prediction: np.ndarray,
    audit_rows: pd.DataFrame,
    columns: tuple[str, ...],
    alpha: float,
    source_rcore_only: bool = False,
) -> np.ndarray:
    residual = source_target - source_prediction
    source_mask = (
        source_rows["domain3"].eq("R_CORE").to_numpy()
        if source_rcore_only
        else np.ones(len(source_rows), dtype=bool)
    )
    fit = pd.DataFrame(
        {
            "key": _keys(source_rows.loc[source_mask].reset_index(drop=True), columns),
            "residual": residual[source_mask],
        }
    )
    stats = fit.groupby("key", observed=True)["residual"].agg(["sum", "count"])
    correction = stats["sum"] / (stats["count"] + float(alpha))
    return _keys(audit_rows, columns).map(correction).fillna(0.0).to_numpy(np.float64)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(pd.read_csv(project / "data" / "train.csv", low_memory=False))
    folds = load_v14_folds(project, train)

    recipes: dict[str, dict[str, object]] = {}
    for method in ("affine", "platt", "beta"):
        for weight in (0.01, 0.025, 0.05, 0.075, 0.10):
            recipes[f"domain_{method}_w{weight:g}"] = {"kind": "calibration", "method": method, "weight": weight}

    groups = {
        "domain": (("domain3",), False, False),
        "domain_count": (("domain3", "balls_before", "strikes_before"), False, False),
        "domain_pitcher_team": (("domain3", "pitcher_team_id"), False, False),
        "domain_batter_team": (("domain3", "batter_team_id"), False, False),
        "domain_hands": (("domain3", "pitcher_hand", "batter_hand"), False, False),
        "rcore_pitcher": (("pitcher_id",), False, True),
        "rcore_batter": (("batter_id",), False, True),
        "rcore_pitcher_count": (("pitcher_id", "balls_before", "strikes_before"), False, True),
        "rcore_batter_count": (("batter_id", "balls_before", "strikes_before"), False, True),
        "rcore_pitcher_batter_hand": (("pitcher_id", "batter_hand"), False, True),
        "rcore_batter_pitcher_hand": (("batter_id", "pitcher_hand"), False, True),
        "corefit_pitcher_batter_hand": (("pitcher_id", "batter_hand"), True, True),
        "corefit_pitcher_count": (("pitcher_id", "balls_before", "strikes_before"), True, True),
    }
    alphas = (50.0, 200.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0)
    for group_name, (columns, source_rcore_only, apply_rcore_only) in groups.items():
        for alpha in alphas:
            for weight in (0.25, 0.5, 1.0):
                recipes[f"eb_{group_name}_a{alpha:g}_w{weight:g}"] = {
                    "kind": "eb", "group": group_name, "columns": columns,
                    "alpha": alpha, "weight": weight,
                    "rcore_only": apply_rcore_only,
                }
    for hand_alpha in (1600.0, 3200.0, 6400.0, 12800.0):
        for count_alpha in (1600.0, 3200.0, 6400.0, 12800.0):
            for hand_weight in (0.5, 1.0, 1.5):
                for count_weight in (0.25, 0.5, 1.0):
                    name = (
                        f"combo_hand_a{hand_alpha:g}_w{hand_weight:g}"
                        f"__count_a{count_alpha:g}_w{count_weight:g}"
                    )
                    recipes[name] = {
                        "kind": "combo",
                        "hand_alpha": hand_alpha,
                        "hand_weight": hand_weight,
                        "count_alpha": count_alpha,
                        "count_weight": count_weight,
                        "rcore_only": True,
                    }

    metric_rows: list[dict[str, object]] = []
    saved_predictions: dict[tuple[str, int], np.ndarray] = {}
    for source_year, audit_year in TRANSITIONS:
        source_rows, source_target, source_prediction = folds[source_year]
        audit_rows, audit_target, audit_prediction = folds[audit_year]
        calibration_cache = {
            method: _domain_models(
                source_rows,
                source_target,
                source_prediction,
                audit_rows,
                audit_prediction,
                method,
            )
            for method in ("affine", "platt", "beta")
        }
        residual_cache = {
            (group_name, alpha): _smoothed_residual(
                source_rows,
                source_target,
                source_prediction,
                audit_rows,
                columns,
                alpha,
                source_rcore_only,
            )
            for group_name, (columns, source_rcore_only, _) in groups.items()
            for alpha in alphas
        }
        for name, recipe in recipes.items():
            if recipe["kind"] == "calibration":
                raw = calibration_cache[str(recipe["method"])]
                weight = float(recipe["weight"])
                candidate = audit_prediction + weight * (raw - audit_prediction)
                changed = np.ones(len(audit_rows), dtype=bool)
            elif recipe["kind"] == "eb":
                correction = residual_cache[(str(recipe["group"]), float(recipe["alpha"]))]
                changed = audit_rows["domain3"].eq("R_CORE").to_numpy() if recipe["rcore_only"] else np.ones(len(audit_rows), dtype=bool)
                candidate = audit_prediction.copy()
                candidate[changed] += float(recipe["weight"]) * correction[changed]
            else:
                correction = (
                    float(recipe["hand_weight"])
                    * residual_cache[("rcore_pitcher_batter_hand", float(recipe["hand_alpha"]))]
                    + float(recipe["count_weight"])
                    * residual_cache[("rcore_pitcher_count", float(recipe["count_alpha"]))]
                )
                changed = audit_rows["domain3"].eq("R_CORE").to_numpy()
                candidate = audit_prediction.copy()
                candidate[changed] += correction[changed]
            candidate = np.clip(candidate, EPSILON, 1.0 - EPSILON)
            metric_rows.append({
                "candidate": name,
                "source_year": source_year,
                "audit_year": audit_year,
                "n_rows": len(audit_target),
                "n_changed": int(changed.sum()),
                "gain_vs_v14": _bss_gain(audit_target, candidate, audit_prediction),
                **{key: value for key, value in recipe.items() if key != "columns"},
            })
            saved_predictions[(name, audit_year)] = candidate

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust = (
        metrics.groupby("candidate", observed=True)["gain_vs_v14"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust["positive_folds"] = metrics.assign(positive=metrics["gain_vs_v14"].gt(0)).groupby("candidate", observed=True)["positive"].sum().reindex(robust["candidate"]).to_numpy()
    robust.to_csv(output_dir / "robust.csv", index=False)
    passing = robust.loc[robust["min_gain"].gt(0.0)]
    shortlist = passing.head(20)
    for name in shortlist["candidate"]:
        for year in (2023, 2024):
            rows, target, incumbent = folds[year]
            np.savez_compressed(
                output_dir / f"{name}_o{year}.npz",
                train_index=rows.index.to_numpy(np.int64), target=target,
                incumbent=incumbent, candidate=saved_predictions[(name, year)],
            )
    summary = {
        "protocol": "V16_STRICT_PREVIOUS_SEASON_RESIDUAL_CALIBRATION_V1",
        "transitions": [f"{source}->{audit}" for source, audit in TRANSITIONS],
        "n_recipes": len(recipes),
        "n_strict_improvements": int(len(passing)),
        "best": robust.head(20).to_dict(orient="records"),
        "selection_eligible": bool(len(passing)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v16_residual_calibration_20260815_01"))
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

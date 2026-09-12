"""Cache structurally diverse variants of the multi-year state residual model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.multi_year_state_model import _add_categories, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import V17_NAME
from src.archive.v16_residual_calibration_screen import load_v14_folds


RECIPES = {
    "global_h4_l15_b075": {"half_life": 4.0, "leaves": 15, "pitcher_weight": 0.75, "domain_models": False},
    "global_h05_l31_b075": {"half_life": 0.5, "leaves": 31, "pitcher_weight": 0.75, "domain_models": False},
    "global_h05_l15_b100": {"half_life": 0.5, "leaves": 15, "pitcher_weight": 1.00, "domain_models": False},
    "global_h05_l15_b050": {"half_life": 0.5, "leaves": 15, "pitcher_weight": 0.50, "domain_models": False},
    "domain_h05_l15_b075": {"half_life": 0.5, "leaves": 15, "pitcher_weight": 0.75, "domain_models": True},
}


def _variant_model(leaves: int, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=120 if leaves == 31 else 150,
        learning_rate=0.025,
        num_leaves=leaves,
        max_depth=5 if leaves == 31 else 4,
        min_child_samples=800,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=3.0,
        reg_lambda=20.0,
        max_bin=127,
        random_state=seed,
    )


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(pd.read_csv(project / "data" / "train.csv", low_memory=False))
    folds = load_v14_folds(project, train)
    numeric_state, _ = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [column for column in features if column.startswith("cat__")]
    pitcher_base = numeric_state["season__pitcher_rate_k80"].to_numpy(np.float64)
    batter_base = numeric_state["season__batter_rate_k80"].to_numpy(np.float64)
    summary_rows: list[dict[str, object]] = []
    for audit_year in (2022, 2023, 2024):
        fit_mask = train["season"].lt(audit_year).to_numpy()
        audit_mask = train["season"].eq(audit_year).to_numpy()
        fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
        target = train.loc[fit_mask, "control_success"].to_numpy(np.float64)
        audit = train.loc[audit_mask].reset_index(drop=True)
        predictions: dict[str, np.ndarray] = {}
        for recipe_index, (name, recipe) in enumerate(RECIPES.items()):
            print(f"[state-variant] year={audit_year} recipe={name}", flush=True)
            pitcher_weight = float(recipe["pitcher_weight"])
            baseline = pitcher_weight * pitcher_base + (1.0 - pitcher_weight) * batter_base
            half_life = float(recipe["half_life"])
            sample_weight = np.exp2(-(audit_year - 1.0 - fit_season) / half_life)
            sample_weight /= sample_weight.mean()
            raw = np.zeros(int(audit_mask.sum()), dtype=np.float64)
            domains = ("R_CORE", "R_ANCHOR", "F") if recipe["domain_models"] else ("ALL",)
            for domain_index, domain in enumerate(domains):
                if domain == "ALL":
                    local_fit = fit_mask
                    local_audit = audit_mask
                    audit_local = np.ones(len(audit), dtype=bool)
                else:
                    local_fit = fit_mask & train["domain3"].eq(domain).to_numpy()
                    local_audit = audit_mask & train["domain3"].eq(domain).to_numpy()
                    audit_local = audit["domain3"].eq(domain).to_numpy()
                model = _variant_model(
                    int(recipe["leaves"]),
                    seed=5100 + 100 * recipe_index + 10 * audit_year + domain_index,
                )
                model.fit(
                    features.loc[local_fit],
                    train.loc[local_fit, "control_success"].to_numpy(np.float64)
                    - baseline[local_fit],
                    sample_weight=sample_weight[train.loc[fit_mask, "domain3"].eq(domain).to_numpy()]
                    if domain != "ALL"
                    else sample_weight,
                    categorical_feature=categorical,
                )
                raw[audit_local] = np.clip(
                    baseline[local_audit] + model.predict(features.loc[local_audit]),
                    0.001,
                    0.999,
                )
                del model
                gc.collect()
            predictions[name] = raw
            summary_rows.append(
                {
                    "audit_year": audit_year,
                    "recipe": name,
                    "raw_mean": float(raw.mean()),
                    "raw_sd": float(raw.std()),
                    "target_mean": float(audit["control_success"].mean()),
                }
            )
        if audit_year == 2022:
            incumbent = folds[2022][2].astype(np.float64)
        else:
            with np.load(
                project / "artifacts" / "v16_multiseason_20260815_02" / f"{V17_NAME}_o{audit_year}.npz"
            ) as saved:
                incumbent = saved["candidate"].astype(np.float64)
        np.savez_compressed(
            output_dir / f"state_variants_o{audit_year}.npz",
            target=audit["control_success"].to_numpy(np.float64),
            incumbent=incumbent,
            domain3=audit["domain3"].astype(str).to_numpy(),
            game_month=audit["game_month"].to_numpy(np.int16),
            pitcher_id=audit["pitcher_id"].to_numpy(),
            **predictions,
        )
    summary = {"protocol": "MULTI_YEAR_STATE_VARIANTS_V1", "recipes": RECIPES, "folds": summary_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multi_year_state_variants_20260816_01"))
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

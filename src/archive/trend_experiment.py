"""Evaluate train-only season-trend calibration and promote the best OOF blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.archive.calibration import apply_logit_offset, choose_three_way_blend, fit_season_logit_offset
from src.archive.data import ID_COL, TARGET_COL, read_main
from src.archive.features import hierarchical_prior
from src.metrics import brier_score, brier_skill_score
from src.archive.train import (
    PRIMARY_SPLIT,
    _metric_row,
    log_result,
    make_official_rf,
    train_lgb_holdout,
    train_rf_holdout,
)
from src.archive.validation import season_holdout


def calibration_curve_frame(
    target: np.ndarray,
    raw: np.ndarray,
    final: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    # Bins are diagnostics on validation OOF only and are never inference features.
    bucket = pd.qcut(raw, q=bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bin": bucket, "target": target, "raw": raw, "final": final})
    return (
        frame.groupby("bin", observed=True)
        .agg(n=("target", "size"), raw_mean=("raw", "mean"), final_mean=("final", "mean"), actual_rate=("target", "mean"))
        .reset_index()
    )


def run(project: Path, config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiments_path = project / "reports" / "experiments.csv"
    train = read_main(project / "data" / "train.csv")
    features = [col for col in read_main(project / "data" / "test.csv", nrows=0).columns if col != ID_COL]
    train_idx, valid_idx = season_holdout(train, 2024)
    target = train.iloc[valid_idx][TARGET_COL].to_numpy()

    best_variant = next(
        item for item in config["lightgbm_variants"] if item["name"] == "lgb_engineered_l31"
    )
    print("Retraining selected LightGBM OOF for trend calibration...")
    lgb_result = train_lgb_holdout(
        train,
        train_idx,
        valid_idx,
        best_variant,
        int(config["max_boost_rounds"]),
        int(config["early_stopping_rounds"]),
        None,
    )
    print("Retraining official RF OOF for trend calibration...")
    rf_result = train_rf_holdout(train, train_idx, valid_idx, features)

    offsets: dict[int, tuple[float, float, dict[int, float]]] = {}
    candidates: list[dict] = []
    for window in (2, 3, 4, 5):
        offset, forecast_rate, season_rates = fit_season_logit_offset(
            train.iloc[train_idx]["season"].to_numpy(),
            train.iloc[train_idx][TARGET_COL].to_numpy(),
            2024,
            window,
        )
        offsets[window] = (offset, forecast_rate, season_rates)
        for model_name, raw in (
            ("lgb", lgb_result["prediction"]),
            ("rf", rf_result["prediction"]),
        ):
            prediction = apply_logit_offset(raw, offset)
            row = {
                "model": f"{model_name}_season_logit_w{window}",
                "family": model_name,
                "window": window,
                "offset": offset,
                "forecast_rate": forecast_rate,
                "prediction": prediction,
                "brier": brier_score(target, prediction),
                "skill": brier_skill_score(target, prediction),
            }
            candidates.append(row)
            log_result(
                experiments_path,
                row["model"],
                f"{model_name} + train-only season logit trend",
                {"window": window, "offset": offset, "forecast_rate": forecast_rate, "season_rates": season_rates},
                brier_score(target, raw),
                row["skill"],
                calibrated_brier=row["brier"],
                calibration="season_logit_offset",
                leakage_notes="offset uses only fold-train annual target rates; no validation/test batch statistic",
            )
            print(row["model"], row["brier"], "offset", offset, "forecast", forecast_rate)

    best_lgb = min((row for row in candidates if row["family"] == "lgb"), key=lambda row: row["brier"])
    best_rf = min((row for row in candidates if row["family"] == "rf"), key=lambda row: row["brier"])
    global_rate = float(train.iloc[train_idx][TARGET_COL].mean())
    prior = hierarchical_prior(train.iloc[valid_idx], global_rate, alpha=50.0)
    blend_weights, blend_brier = choose_three_way_blend(
        [best_lgb["prediction"], best_rf["prediction"], prior], target, step=0.05
    )
    blend = (
        blend_weights[0] * best_lgb["prediction"]
        + blend_weights[1] * best_rf["prediction"]
        + blend_weights[2] * prior
    )
    blend_skill = brier_skill_score(target, blend)
    log_result(
        experiments_path,
        "oof_tuned_trend_blend",
        "season-trend LGB + season-trend RF + hierarchical prior",
        {
            "weights": blend_weights,
            "lgb_window": best_lgb["window"],
            "rf_window": best_rf["window"],
        },
        min(best_lgb["brier"], best_rf["brier"]),
        blend_skill,
        calibrated_brier=blend_brier,
        calibration="OOF grid blend step=0.05",
        leakage_notes="weights selected on 2024 OOF; score is selection-biased, final uses only train artifacts",
    )
    print("trend blend", blend_weights, blend_brier, blend_skill)

    choices = [
        {**best_lgb, "weights": [1.0, 0.0, 0.0], "prediction": best_lgb["prediction"]},
        {**best_rf, "weights": [0.0, 1.0, 0.0], "prediction": best_rf["prediction"]},
        {
            "model": "oof_tuned_trend_blend",
            "brier": blend_brier,
            "skill": blend_skill,
            "weights": blend_weights,
            "prediction": blend,
        },
    ]
    winner = min(choices, key=lambda row: row["brier"])
    print("Production winner:", winner["model"], winner["brier"])

    # Final 2025 offsets use all official train labels and the window selected on
    # primary validation. No hidden-test values or predictions are involved.
    lgb_offset_2025, lgb_forecast_2025, full_rates = fit_season_logit_offset(
        train["season"].to_numpy(), train[TARGET_COL].to_numpy(), 2025, int(best_lgb["window"])
    )
    rf_offset_2025, rf_forecast_2025, _ = fit_season_logit_offset(
        train["season"].to_numpy(), train[TARGET_COL].to_numpy(), 2025, int(best_rf["window"])
    )
    weights = winner["weights"]
    model_dir = project / "model"
    ensemble = json.loads((model_dir / "ensemble.json").read_text(encoding="utf-8"))
    ensemble["calibration"] = {"method": "logit_offset", "offset": lgb_offset_2025}
    ensemble["rf_calibration"] = {"method": "logit_offset", "offset": rf_offset_2025}
    ensemble["weights"] = {
        "lightgbm": weights[0], "random_forest": weights[1], "hierarchical_prior": weights[2]
    }
    ensemble["season_trend"] = {
        "lgb_window": int(best_lgb["window"]),
        "rf_window": int(best_rf["window"]),
        "lgb_forecast_rate_2025": lgb_forecast_2025,
        "rf_forecast_rate_2025": rf_forecast_2025,
        "train_season_rates": full_rates,
    }
    (model_dir / "ensemble.json").write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if weights[1] > 0.0:
        print("Training final RF production member...")
        final_rf = make_official_rf(features)
        final_rf.fit(train[features], train[TARGET_COL])
        joblib.dump(final_rf, model_dir / "rf_model.joblib", compress=3)

    curve = calibration_curve_frame(target, lgb_result["prediction"], winner["prediction"])
    curve.to_csv(project / "reports" / "calibration_curve.csv", index=False, encoding="utf-8")
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "production_winner": winner["model"],
            "production_primary_brier": winner["brier"],
            "production_primary_skill": winner["skill"],
            "production_weights": weights,
            "lgb_trend_window": int(best_lgb["window"]),
            "rf_trend_window": int(best_rf["window"]),
            "trend_selection_note": "blend weights selected/evaluated on 2024 OOF; optimistic selection risk",
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "", "## Train-only season trend calibration", "",
        "Offsets are fitted only from fold-train annual target rates; validation/test batch means are never used.",
        "", "| candidate | Brier Score | local Brier Skill Score | forecast rate |", "| --- | ---: | ---: | ---: |",
    ]
    for row in candidates:
        summary_lines.append(
            f"| {row['model']} | {row['brier']:.9f} | {row['skill']:.3f} | {row['forecast_rate']:.6f} |"
        )
    summary_lines.extend(
        [
            f"| oof_tuned_trend_blend | {blend_brier:.9f} | {blend_skill:.3f} | n/a |",
            "",
            f"Production winner: **{winner['model']}**, weights `{weights}`.",
            "The blend score is selection-biased because its small weight grid was selected on the same 2024 OOF; this is explicitly retained as a validation risk.",
        ]
    )
    findings = project / "reports" / "initial_findings.md"
    with findings.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    args = parser.parse_args()
    project = args.project_dir.resolve()
    config = args.config if args.config.is_absolute() else project / args.config
    run(project, config)


if __name__ == "__main__":
    main()

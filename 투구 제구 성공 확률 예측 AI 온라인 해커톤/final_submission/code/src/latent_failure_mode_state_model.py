"""Full-history latent failure-mode mixture with forward-only inference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.failure_mode_privileged_distillation import (
    MODE_NAMES,
    reconstruct_failure_mode,
)
from src.multi_year_state_model import _add_categories, _model, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import V17_NAME, _diagnostics, bss
from src.archive.v16_residual_calibration_screen import load_v14_folds


def _mode_classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        verbosity=-1,
        n_jobs=6,
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=700,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0,
        reg_lambda=18.0,
        max_bin=127,
        random_state=seed,
    )


def _typed_features(features: pd.DataFrame, label: np.ndarray) -> pd.DataFrame:
    output = features.copy()
    values = np.full(len(label), "__MISSING__", dtype=object)
    valid = label >= 0
    names = np.asarray(MODE_NAMES, dtype=object)
    values[valid] = names[label[valid]]
    output["cat__latent_failure_mode"] = pd.Categorical(
        values, categories=["__MISSING__", *MODE_NAMES]
    )
    return output


def _predict_each_mode(
    model: lgb.LGBMRegressor, features: pd.DataFrame, categorical: list[str]
) -> np.ndarray:
    prediction = []
    for name in MODE_NAMES:
        typed = features.copy()
        typed["cat__latent_failure_mode"] = pd.Categorical(
            np.full(len(typed), name, dtype=object),
            categories=["__MISSING__", *MODE_NAMES],
        )
        prediction.append(model.predict(typed, categorical_feature=categorical))
        del typed
    return np.column_stack(prediction).astype(np.float64)


def _temperature(probability: np.ndarray, power: float) -> np.ndarray:
    output = np.power(np.clip(probability.astype(np.float64), 1e-8, 1.0), power)
    return output / output.sum(axis=1, keepdims=True)


def run(
    project: Path,
    output_dir: Path,
    outcome_half_life: float,
    audit_years: tuple[int, ...] = (2023, 2024),
) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    mode = reconstruct_failure_mode(train)
    print(f"[latent-mode] reconstructed coverage={np.mean(mode >= 0):.6f}", flush=True)
    numeric_state, baseline = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [column for column in features if column.startswith("cat__")]
    typed = _typed_features(features, mode)
    typed_categorical = [*categorical, "cat__latent_failure_mode"]
    v14_folds = load_v14_folds(project, train) if 2022 in audit_years else {}

    folds_out: list[dict[str, object]] = []
    for audit_year in audit_years:
        print(f"[latent-mode] audit_year={audit_year}", flush=True)
        fit_mask = train["season"].lt(audit_year).to_numpy() & (mode >= 0)
        audit_mask = train["season"].eq(audit_year).to_numpy()
        fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
        probability_by_name: dict[str, np.ndarray] = {}
        for half_life in (0.5, 2.0, 4.0):
            sample_weight = np.exp2(-(audit_year - 1.0 - fit_season) / half_life)
            sample_weight /= sample_weight.mean()
            classifier = _mode_classifier(
                seed=7100 + audit_year + int(round(10 * half_life))
            )
            classifier.fit(
                features.loc[fit_mask],
                mode[fit_mask].astype(np.int64),
                sample_weight=sample_weight,
                categorical_feature=categorical,
            )
            probability_by_name[f"mode_lgb_h{half_life:g}"] = classifier.predict_proba(
                features.loc[audit_mask]
            ).astype(np.float64)
            del classifier
            gc.collect()
        probability_by_name["mode_lgb_mean"] = np.mean(
            np.stack(list(probability_by_name.values())), axis=0
        )

        outcome_weight = np.exp2(
            -(audit_year - 1.0 - fit_season) / outcome_half_life
        )
        outcome_weight /= outcome_weight.mean()
        outcome = _model(leaves=15, seed=7200 + audit_year)
        outcome.fit(
            typed.loc[fit_mask],
            train.loc[fit_mask, "control_success"].to_numpy(np.float64)
            - baseline[fit_mask],
            sample_weight=outcome_weight,
            categorical_feature=typed_categorical,
        )
        residual_by_mode = _predict_each_mode(
            outcome, features.loc[audit_mask], typed_categorical
        )
        del outcome
        gc.collect()
        raw_by_mode = np.clip(
            baseline[audit_mask, None] + residual_by_mode, 0.001, 0.999
        )

        audit = train.loc[audit_mask].reset_index(drop=True)
        target = audit["control_success"].to_numpy(np.float64)
        audit_mode = mode[audit_mask]
        known = audit_mode >= 0
        if audit_year == 2022:
            fold_rows, fold_target, incumbent = v14_folds[2022]
            if not np.array_equal(target, fold_target.astype(np.float64)):
                raise ValueError("v14 fallback order mismatch for 2022")
            if not np.array_equal(
                audit["row_id"].to_numpy(), fold_rows["row_id"].to_numpy()
            ):
                raise ValueError("v14 fallback row order mismatch for 2022")
        else:
            with np.load(
                project
                / "artifacts"
                / "v16_multiseason_20260815_02"
                / f"{V17_NAME}_o{audit_year}.npz"
            ) as saved:
                incumbent = saved["candidate"].astype(np.float64)
                if not np.array_equal(target, saved["target"].astype(np.float64)):
                    raise ValueError(f"v17 order mismatch for {audit_year}")

        oracle = np.sum(
            probability_by_name["mode_lgb_mean"] * raw_by_mode, axis=1
        )
        oracle[known] = raw_by_mode[
            np.flatnonzero(known), audit_mode[known]
        ]
        metric_rows: list[dict[str, object]] = []
        raw_cache: dict[str, np.ndarray] = {"oracle": oracle}
        conditional_success = np.array(
            [
                np.average(
                    train.loc[fit_mask, "control_success"].to_numpy(np.float64)[
                        mode[fit_mask] == klass
                    ],
                    weights=outcome_weight[mode[fit_mask] == klass],
                )
                for klass in range(4)
            ],
            dtype=np.float64,
        )
        for probability_name, probability in probability_by_name.items():
            for power in (0.5, 1.0, 1.5, 2.0):
                transformed = _temperature(probability, power)
                name = f"{probability_name}_pow{power:g}"
                raw_cache[name] = np.sum(transformed * raw_by_mode, axis=1)
                raw_cache[f"conditional_{name}"] = transformed @ conditional_success
                accuracy = float(
                    np.mean(np.argmax(transformed[known], axis=1) == audit_mode[known])
                )
                logloss = float(
                    -np.mean(
                        np.log(
                            np.clip(
                                transformed[np.flatnonzero(known), audit_mode[known]],
                                1e-12,
                                1.0,
                            )
                        )
                    )
                )
                for raw_kind in (name, f"conditional_{name}"):
                    raw = raw_cache[raw_kind]
                    for weight in (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0):
                        candidate = np.clip(
                            incumbent + weight * (raw - incumbent), 0.001, 0.999
                        )
                        diagnostic = _diagnostics(
                            audit, target, incumbent, candidate
                        )
                        metric_rows.append(
                            {
                                "audit_year": audit_year,
                                "candidate": raw_kind,
                                "weight": weight,
                                "mode_accuracy": accuracy,
                                "mode_logloss": logloss,
                                "gain": diagnostic["gain"],
                                "month_positive_fraction": float(
                                    np.mean(
                                        [row["gain"] > 0 for row in diagnostic["months"]]
                                    )
                                ),
                                "worst_month_gain": float(
                                    min(row["gain"] for row in diagnostic["months"])
                                ),
                            }
                        )
        oracle_diagnostic = _diagnostics(audit, target, incumbent, oracle)
        metrics = pd.DataFrame(metric_rows).sort_values("gain", ascending=False)
        metrics.to_csv(output_dir / f"metrics_o{audit_year}.csv", index=False)
        names = list(raw_cache)
        np.savez_compressed(
            output_dir / f"latent_failure_mode_o{audit_year}.npz",
            target=target,
            incumbent=incumbent,
            domain3=audit["domain3"].astype(str).to_numpy(),
            game_month=audit["game_month"].to_numpy(np.int16),
            pitcher_id=audit["pitcher_id"].to_numpy(),
            names=np.asarray(names, dtype=object),
            raw=np.column_stack([raw_cache[name] for name in names]),
            mode_label=audit_mode,
            raw_by_mode=raw_by_mode,
        )
        fold_result = {
            "audit_year": audit_year,
            "mode_coverage": float(known.mean()),
            "conditional_success": {
                MODE_NAMES[index]: float(value)
                for index, value in enumerate(conditional_success)
            },
            "oracle_gain": oracle_diagnostic["gain"],
            "best": metrics.head(40).to_dict(orient="records"),
        }
        folds_out.append(fold_result)
        print(json.dumps(fold_result, ensure_ascii=False, indent=2), flush=True)

    combined = pd.concat(
        [pd.read_csv(output_dir / f"metrics_o{year}.csv") for year in audit_years],
        ignore_index=True,
    )
    robust = (
        combined.groupby(["candidate", "weight"], observed=True)["gain"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "LATENT_FAILURE_MODE_FULL_HISTORY_FORWARD_V1",
        "outcome_half_life": outcome_half_life,
        "audit_years": list(audit_years),
        "training_label_coverage": float(np.mean(mode >= 0)),
        "folds": folds_out,
        "robust": robust.head(60).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/latent_failure_mode_state_20260816_01"),
    )
    parser.add_argument("--outcome-half-life", type=float, default=0.5)
    parser.add_argument(
        "--audit-years",
        type=int,
        nargs="+",
        default=[2023, 2024],
        choices=[2022, 2023, 2024],
    )
    args = parser.parse_args()
    run(
        args.project,
        args.output_dir,
        args.outcome_half_life,
        tuple(args.audit_years),
    )


if __name__ == "__main__":
    main()

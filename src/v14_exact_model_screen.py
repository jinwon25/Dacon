"""Screen Brier-oriented exact-ASOF models under forward season transitions.

The incumbent exact model is a single LightGBM binary classifier.  This audit
adds two intentionally diverse candidates:

* squared-error LightGBM, aligned with the competition's Brier loss;
* a strongly regularised LightGBM residual model that edits the frozen base.

The source-season target is the only label used for fitting.  Audit-season
targets are read only after predictions have been produced and are used solely
for reporting.  The script never reads hidden evaluation data.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.recent_shared_exact_asof import _feature_pair
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v14_component_audit import bss, load_folds


TRANSITIONS = ((2021, 2022), (2022, 2023), (2023, 2024))


def _regressor(*, leaves: int, seed: int, residual: bool) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180 if leaves == 7 else 150,
        learning_rate=0.025,
        num_leaves=leaves,
        max_depth=3 if leaves == 7 else 4,
        min_child_samples=500 if residual else 350,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0 if residual else 1.0,
        reg_lambda=12.0 if residual else 8.0,
        max_bin=127,
        random_state=seed,
    )


def _classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=6,
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=7,
        min_child_samples=350,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=1.0,
        reg_lambda=6.0,
        max_bin=127,
        random_state=seed,
    )


def _domain_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "ALL": np.ones(len(rows), dtype=bool),
        "R_CORE": rows["domain3"].eq("R_CORE").to_numpy(),
        "R_ANCHOR": rows["domain3"].eq("R_ANCHOR").to_numpy(),
        "F": rows["domain3"].eq("F").to_numpy(),
    }


def _candidate_metrics(
    year: int,
    rows: pd.DataFrame,
    target: np.ndarray,
    base: np.ndarray,
    incumbent_exact: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for domain, mask in _domain_masks(rows).items():
        for name, prediction in predictions.items():
            output.append(
                {
                    "audit_year": year,
                    "domain": domain,
                    "candidate": name,
                    "blend_weight": 1.0,
                    "n_rows": int(mask.sum()),
                    "bss": bss(target[mask], prediction[mask]),
                    "gain_vs_base": bss(target[mask], prediction[mask])
                    - bss(target[mask], base[mask]),
                    "gain_vs_exact_lgb": bss(target[mask], prediction[mask])
                    - bss(target[mask], incumbent_exact[mask]),
                }
            )
        for name, prediction in predictions.items():
            if name in {"base", "exact_lgb"}:
                continue
            for weight in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
                blended = np.clip(
                    incumbent_exact + weight * (prediction - incumbent_exact),
                    1e-6,
                    1.0 - 1e-6,
                )
                output.append(
                    {
                        "audit_year": year,
                        "domain": domain,
                        "candidate": f"exact_lgb_plus_{name}",
                        "blend_weight": weight,
                        "n_rows": int(mask.sum()),
                        "bss": bss(target[mask], blended[mask]),
                        "gain_vs_base": bss(target[mask], blended[mask])
                        - bss(target[mask], base[mask]),
                        "gain_vs_exact_lgb": bss(target[mask], blended[mask])
                        - bss(target[mask], incumbent_exact[mask]),
                    }
                )
    return output


def run(project: Path, exact_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    exact_dir = exact_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds = load_folds(train, exact_dir)
    metric_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for source_year, audit_year in TRANSITIONS:
        source = train.loc[train["season"].eq(source_year)].reset_index(drop=True)
        audit = train.loc[train["season"].eq(audit_year)].reset_index(drop=True)
        fold = folds[audit_year]
        expected_index = np.flatnonzero(train["season"].eq(audit_year).to_numpy())
        if not np.array_equal(expected_index, fold.train_index):
            raise ValueError(f"fold order mismatch for {audit_year}")
        fit_x, audit_x = _feature_pair(
            train,
            source,
            audit,
            source_year,
            audit_year,
            include_categories=True,
        )
        target = source["control_success"].to_numpy(np.float64)
        predictions: dict[str, np.ndarray] = {
            "base": fold.base,
            "exact_lgb": fold.exact_lgb,
            "exact_ridge": fold.exact_ridge,
            "trend_lgb": fold.trend_lgb,
        }
        binary_predictions = [fold.exact_lgb]
        for seed in (202, 777):
            started = time.perf_counter()
            model = _classifier(seed)
            model.fit(fit_x, target)
            binary_predictions.append(
                model.predict_proba(audit_x)[:, 1].astype(np.float64)
            )
            timing_rows.append(
                {
                    "source_year": source_year,
                    "audit_year": audit_year,
                    "candidate": f"binary_seed{seed}",
                    "seconds": time.perf_counter() - started,
                }
            )
            del model
        predictions["binary_seed_ensemble"] = np.mean(
            np.column_stack(binary_predictions), axis=1
        )
        for leaves in (7, 15):
            started = time.perf_counter()
            model = _regressor(leaves=leaves, seed=918, residual=False)
            model.fit(fit_x, target)
            predictions[f"l2_leaves{leaves}"] = np.clip(
                model.predict(audit_x), 0.001, 0.999
            )
            timing_rows.append(
                {
                    "source_year": source_year,
                    "audit_year": audit_year,
                    "candidate": f"l2_leaves{leaves}",
                    "seconds": time.perf_counter() - started,
                }
            )
            del model

        # Residual models require the exact frozen base for the source season.
        # It is available for source years 2022 and 2023 in the handed-off OOF.
        if source_year in folds:
            source_fold = folds[source_year]
            source_index = np.flatnonzero(train["season"].eq(source_year).to_numpy())
            if not np.array_equal(source_index, source_fold.train_index):
                raise ValueError(f"source base order mismatch for {source_year}")
            fit_x_residual = fit_x.copy()
            audit_x_residual = audit_x.copy()
            fit_x_residual["frozen_base_probability"] = source_fold.base
            audit_x_residual["frozen_base_probability"] = fold.base
            for leaves in (7, 15):
                started = time.perf_counter()
                model = _regressor(leaves=leaves, seed=491, residual=True)
                model.fit(
                    fit_x_residual,
                    source_fold.target - source_fold.base,
                )
                correction = np.clip(model.predict(audit_x_residual), -0.08, 0.08)
                predictions[f"residual_l2_leaves{leaves}"] = np.clip(
                    fold.base + correction, 0.001, 0.999
                )
                timing_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "candidate": f"residual_l2_leaves{leaves}",
                        "seconds": time.perf_counter() - started,
                    }
                )
                del model
            del fit_x_residual, audit_x_residual

        np.savez_compressed(
            output_dir / f"exact_model_screen_o{audit_year}.npz",
            train_index=fold.train_index,
            target=fold.target,
            **predictions,
        )
        metric_rows.extend(
            _candidate_metrics(
                audit_year,
                fold.rows,
                fold.target,
                fold.base,
                fold.exact_lgb,
                predictions,
            )
        )
        del fit_x, audit_x, predictions
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output_dir / "timings.csv", index=False)
    core = metrics.loc[
        metrics["domain"].eq("R_CORE") & metrics["audit_year"].isin([2023, 2024])
    ]
    robust_core = (
        core.groupby(["candidate", "blend_weight"], observed=True)[
            "gain_vs_exact_lgb"
        ]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=[False, False])
    )
    robust_core.to_csv(output_dir / "robust_core_vs_exact_lgb.csv", index=False)
    f_post_break = metrics.loc[
        metrics["domain"].eq("F") & metrics["audit_year"].eq(2024)
    ].sort_values("gain_vs_exact_lgb", ascending=False)
    f_post_break.to_csv(output_dir / "f_2024_vs_exact_lgb.csv", index=False)
    summary = {
        "protocol": "V14_BRIER_EXACT_MODEL_SCREEN_V1",
        "transitions": [f"{source}->{audit}" for source, audit in TRANSITIONS],
        "robust_core_top": robust_core.head(12).to_dict(orient="records"),
        "post_break_f_top": f_post_break.head(12).to_dict(orient="records"),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.exact_dir, args.output_dir)


if __name__ == "__main__":
    main()

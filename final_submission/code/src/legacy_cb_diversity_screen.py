"""Audit variance-reduction and recency variants of the Public-positive CB axis."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.advanced_domain_residual_oof import _load_folds
from src.data import read_main
from src.metrics import brier_score


PUBLIC_WEIGHT = 0.5534087425546688


@dataclass(frozen=True)
class Variant:
    name: str
    model_seed: int
    sample_seed: int
    recency_half_life: float | None = None
    recent_seasons: int | None = None


VARIANTS = (
    Variant("baseline_s42", 42, 42),
    Variant("model_s202", 202, 42),
    Variant("sample_s202", 42, 202),
    Variant("recency_h2", 202, 42, recency_half_life=2.0),
)


def _load_legacy_builder(legacy_project: Path):
    source = legacy_project / "src" / "top1100_features.py"
    spec = importlib.util.spec_from_file_location("legacy_feature_builder", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_features


def _clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    categorical = list(
        frame.select_dtypes(include=["category", "object", "string"]).columns
    )
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("__MISSING__")
    return frame, categorical


def _fit_index(train: pd.DataFrame, year: int, variant: Variant) -> np.ndarray:
    seasons = train["season"].to_numpy()
    mask = seasons < year
    if variant.recent_seasons is not None:
        mask &= seasons >= year - variant.recent_seasons
    index = np.flatnonzero(mask)
    if len(index) > 1_200_000:
        rng = np.random.default_rng(variant.sample_seed)
        index = np.sort(rng.choice(index, size=1_200_000, replace=False))
    return index


def _fit_predict(
    train: pd.DataFrame,
    features: pd.DataFrame,
    fit_index: np.ndarray,
    audit_index: np.ndarray,
    audit_year: int,
    variant: Variant,
) -> tuple[CatBoostClassifier, np.ndarray, float]:
    fit, categorical = _clean(features.iloc[fit_index].reset_index(drop=True))
    audit, _ = _clean(features.iloc[audit_index].reset_index(drop=True))
    weights = None
    if variant.recency_half_life is not None:
        latest = audit_year - 1
        fit_seasons = train.iloc[fit_index]["season"].to_numpy()
        weights = np.power(
            0.5,
            np.maximum(latest - fit_seasons, 0) / variant.recency_half_life,
        ).astype(np.float32)
    model = CatBoostClassifier(
        iterations=350,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        l2_leaf_reg=30,
        random_strength=0.5,
        random_seed=variant.model_seed,
        thread_count=12,
        verbose=100,
        allow_writing_files=False,
        one_hot_max_size=32,
    )
    started = time.perf_counter()
    model.fit(
        fit,
        train.iloc[fit_index]["control_success"].to_numpy(np.int8),
        cat_features=categorical,
        sample_weight=weights,
    )
    prediction = np.asarray(model.predict_proba(audit)[:, 1], dtype=np.float64)
    return model, prediction, time.perf_counter() - started


def _residual_prediction(
    fold: pd.DataFrame,
    correction_dir: Path,
    year: int,
    eta: float,
) -> np.ndarray:
    mode = "r_only" if year == 2022 else "all"
    with np.load(
        correction_dir / f"correction_corrected_state_{mode}_o{year}.npz",
        allow_pickle=True,
    ) as bundle:
        row_id = bundle["row_id"].astype(str)
        correction = bundle["correction"].astype(np.float64)
    regular = fold["game_type"].eq("R").to_numpy()
    if not np.array_equal(row_id, fold.loc[regular, "row_id"].astype(str).to_numpy()):
        raise ValueError(f"residual row mismatch for {year}")
    prediction = fold["v2"].to_numpy(np.float64, copy=True)
    prediction[regular] = np.clip(
        prediction[regular] + eta * correction,
        1e-6,
        1.0 - 1e-6,
    )
    return prediction


def _advanced_prediction(
    fold: pd.DataFrame,
    advanced_dir: Path,
    year: int,
    eta: float,
) -> np.ndarray:
    with np.load(
        advanced_dir / f"correction_compact_l15_o{year}.npz", allow_pickle=True
    ) as bundle:
        row_id = bundle["row_id"].astype(str)
        correction = bundle["correction"].astype(np.float64)
    if not np.array_equal(row_id, fold["row_id"].astype(str).to_numpy()):
        raise ValueError(f"advanced row mismatch for {year}")
    prediction = fold["v2"].to_numpy(np.float64, copy=True)
    regular = fold["game_type"].eq("R").to_numpy()
    prediction[regular] = np.clip(
        prediction[regular] + eta * correction[regular],
        1e-6,
        1.0 - 1e-6,
    )
    return prediction


def _add_cb(
    base: np.ndarray,
    v2: np.ndarray,
    cb: np.ndarray,
    regular: np.ndarray,
    weight: float = PUBLIC_WEIGHT,
) -> np.ndarray:
    result = base.copy()
    result[regular] = np.clip(
        result[regular] + weight * (cb[regular] - v2[regular]),
        1e-6,
        1.0 - 1e-6,
    )
    return result


def run(
    data_project: Path,
    legacy_project: Path,
    corrected_cb_dir: Path,
    correction_dir: Path,
    advanced_dir: Path,
    out_dir: Path,
    years: tuple[int, ...],
    variants: tuple[Variant, ...] = VARIANTS,
) -> None:
    data_project = data_project.resolve()
    legacy_project = legacy_project.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project / "data" / "train.csv")
    folds = _load_folds(train, corrected_cb_dir.resolve())
    build_features = _load_legacy_builder(legacy_project)
    features = build_features(train, train, include_ids=True)

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[int, str], np.ndarray] = {}
    for year in years:
        audit_index = folds[year]["train_index"].to_numpy(np.int64)
        for variant in variants:
            fit_index = _fit_index(train, year, variant)
            model, prediction, seconds = _fit_predict(
                train, features, fit_index, audit_index, year, variant
            )
            predictions[(year, variant.name)] = prediction
            model.save_model(str(out_dir / f"legacy_{variant.name}_o{year}.cbm"))
            np.savez_compressed(
                out_dir / f"legacy_{variant.name}_o{year}.npz",
                row_id=folds[year]["row_id"].astype(str).to_numpy(),
                prediction=prediction,
            )
            rows.append(
                {
                    "season": year,
                    "variant": variant.name,
                    "fit_rows": len(fit_index),
                    "seconds": seconds,
                    "prediction_mean": float(prediction.mean()),
                    "prediction_std": float(prediction.std()),
                }
            )
            print(
                f"year={year} variant={variant.name} rows={len(fit_index)} "
                f"seconds={seconds:.1f}",
                flush=True,
            )
            del model
            gc.collect()

    fit_summary = pd.DataFrame(rows)
    fit_summary.to_csv(out_dir / "fit_summary.csv", index=False)
    metric_rows: list[dict[str, object]] = []
    for year in years:
        fold = folds[year]
        target = fold["target"].to_numpy(np.float64)
        v2 = fold["v2"].to_numpy(np.float64)
        regular = fold["game_type"].eq("R").to_numpy()
        residual = _residual_prediction(fold, correction_dir.resolve(), year, 1.0)
        advanced = _advanced_prediction(fold, advanced_dir.resolve(), year, 1.25)
        baseline_cb = predictions[(year, "baseline_s42")]
        current_v9 = _add_cb(residual, v2, baseline_cb, regular)
        advanced_v9 = _add_cb(advanced, v2, baseline_cb, regular)
        candidates = {
            "current_v9_rebuilt": current_v9,
            "advanced_v9": advanced_v9,
        }
        for variant in variants[1:]:
            cb = predictions[(year, variant.name)]
            candidates[f"replace_{variant.name}"] = _add_cb(
                advanced, v2, cb, regular
            )
            candidates[f"ensemble_{variant.name}"] = _add_cb(
                advanced, v2, 0.5 * (baseline_cb + cb), regular
            )
        if all(
            (year, name) in predictions
            for name in ("baseline_s42", "model_s202", "sample_s202")
        ):
            ensemble_all = np.mean(
                np.column_stack(
                    [
                        predictions[(year, name)]
                        for name in ("baseline_s42", "model_s202", "sample_s202")
                    ]
                ),
                axis=1,
            )
            candidates["ensemble_three_seed_sample"] = _add_cb(
                advanced, v2, ensemble_all, regular
            )
        baseline_brier = brier_score(target, current_v9)
        for name, prediction in candidates.items():
            metric_rows.append(
                {
                    "season": year,
                    "candidate": name,
                    "brier": brier_score(target, prediction),
                    "delta_vs_rebuilt_v9": brier_score(target, prediction)
                    - baseline_brier,
                    "delta_vs_v2": brier_score(target, prediction)
                    - brier_score(target, v2),
                }
            )
    metrics = pd.DataFrame(metric_rows).sort_values(["season", "brier"])
    metrics.to_csv(out_dir / "candidate_metrics.csv", index=False)
    manifest = {
        "years": years,
        "public_weight": PUBLIC_WEIGHT,
        "variants": [variant.__dict__ for variant in variants],
        "legacy_feature_source": str(legacy_project / "src" / "top1100_features.py"),
        "test_batch_aggregates_used": False,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-project", type=Path, default=Path("."))
    parser.add_argument("--legacy-project", type=Path, required=True)
    parser.add_argument("--corrected-cb-dir", type=Path, required=True)
    parser.add_argument("--correction-dir", type=Path, required=True)
    parser.add_argument("--advanced-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--years", default="2024")
    parser.add_argument(
        "--variants",
        default=",".join(variant.name for variant in VARIANTS),
    )
    args = parser.parse_args()
    years = tuple(int(value) for value in args.years.split(",") if value.strip())
    requested = {value.strip() for value in args.variants.split(",") if value.strip()}
    variants = tuple(variant for variant in VARIANTS if variant.name in requested)
    if not variants or variants[0].name != "baseline_s42":
        raise ValueError("variants must include baseline_s42 first")
    run(
        args.data_project,
        args.legacy_project,
        args.corrected_cb_dir,
        args.correction_dir,
        args.advanced_dir,
        args.out_dir,
        years,
        variants,
    )


if __name__ == "__main__":
    main()

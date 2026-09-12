"""Finalize the strict Component component for target season 2025.

Iteration counts are selected on 2024 after fitting only 2019--2023.  The
public final models were trained on the same official 2019--2024 rows with the
same fixed recipes, so their selected tree prefixes are copied rather than
needlessly refitted.  Calibration and conditional lookups use training rows
only; no evaluation file is read.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


PROTOCOL = "V180_STRICT_COMPONENT_2025_FINALIZE_V1"
TARGET = "control_success"
TARGET_YEAR = 2025
DROP_COLUMNS = ("row_id", "pitcher_id", "batter_id", TARGET, "game_month")
CAT_COLUMNS = (
    "top_bottom", "game_type", "base_state", "pitcher_team_id",
    "batter_team_id", "pitcher_hand", "batter_hand", "count_state",
)
MIN_ITERATIONS = 20
LGB_MAX_ITERATIONS = 700
CAT_MAX_ITERATIONS = 450
CAT_EVAL_PERIOD = 10
ITERATION_SCALE = 1.2
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.025, "num_leaves": 127,
    "min_data_in_leaf": 500, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
    "max_bin": 255, "verbosity": -1, "seed": 42,
}
CAT_PARAMS = {
    "loss_function": "Logloss", "learning_rate": 0.05, "depth": 8,
    "l2_leaf_reg": 10, "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0, "random_seed": 42, "verbose": False,
    "allow_writing_files": False, "used_ram_limit": "2500mb",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modules(component_root: Path) -> tuple[ModuleType, ModuleType, dict[str, str]]:
    common_path = component_root / "src" / "common_features.py"
    conditional_path = component_root / "src" / "conditional_features.py"
    return (
        _load_module(common_path, "v180_component_common"),
        _load_module(conditional_path, "v180_component_conditional"),
        {
            "common_features.py": _sha256(common_path),
            "conditional_features.py": _sha256(conditional_path),
        },
    )


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _find_delta(probability: np.ndarray, target_rate: float) -> float:
    logits = _logit(probability)
    lower, upper = -4.0, 4.0
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if float(_sigmoid(logits + midpoint).mean()) < float(target_rate):
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _monthly_rate(source: pd.DataFrame, target_year: int) -> float:
    if source.empty or int(source["season"].max()) >= int(target_year):
        raise ValueError("rate source must end before target year")
    monthly = (
        source.groupby(["season", "game_month"], observed=True)[TARGET]
        .agg(["mean", "size"])
        .reset_index()
    )
    monthly = monthly.loc[monthly["size"].gt(3000)].copy()
    monthly["time"] = monthly["season"] + (monthly["game_month"] - 3.0) / 8.0
    forecast = np.polyval(
        np.polyfit(monthly["time"].to_numpy(float), monthly["mean"].to_numpy(), 1),
        float(target_year) + 0.5,
    )
    return float(np.clip(forecast, 0.05, 0.95))


def _feature_frame(
    raw: pd.DataFrame, common: ModuleType, conditional: ModuleType
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    platoon, count = conditional.build_conditional(
        raw, range(int(raw["season"].min()) + 1, TARGET_YEAR + 1)
    )
    frame = conditional.join_conditional(raw.copy(), platoon, count)
    frame = common.add_features(frame)
    for column in frame.columns:
        if frame[column].dtype == "float32":
            frame[column] = frame[column].astype("float64")
    return frame, platoon, count


def _align_categories(
    train: pd.DataFrame, others: Iterable[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    train = train.copy()
    aligned = [value.copy() for value in others]
    cats = [column for column in CAT_COLUMNS if column in train.columns]
    for column in cats:
        train[column] = train[column].astype("category")
        categories = list(train[column].cat.categories)
        for value in aligned:
            value[column] = pd.Categorical(value[column], categories=categories)
    return train, aligned, cats


def _cat_frame(frame: pd.DataFrame, cats: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in cats:
        output[column] = output[column].astype("object").where(
            output[column].notna(), "__NA__"
        ).astype(str)
    return output


def _calibrated_brier(
    target: np.ndarray, probability: np.ndarray, rate: float
) -> float:
    calibrated = _sigmoid(_logit(probability) + _find_delta(probability, rate))
    return float(np.mean(np.square(calibrated - target)))


def _select_iterations(
    frame: pd.DataFrame, *, threads: int, checkpoint_dir: Path
) -> tuple[int, int, dict[str, float]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    complete_path = checkpoint_dir / "complete.json"
    if complete_path.is_file():
        saved = json.loads(complete_path.read_text(encoding="utf-8"))
        return (
            int(saved["lgb_iterations"]),
            int(saved["cat_iterations"]),
            dict(saved["selection"]),
        )
    season = frame["season"].to_numpy(np.int16)
    target = frame[TARGET].to_numpy(np.float64)
    features = frame.drop(
        columns=[column for column in DROP_COLUMNS if column in frame.columns]
    )
    fit = season < 2024
    validation = season == 2024
    x_train = features.loc[fit].copy()
    x_valid = features.loc[validation].copy()
    y_train = target[fit]
    y_valid = target[validation]
    x_train, [x_valid], cats = _align_categories(x_train, [x_valid])
    rate = _monthly_rate(
        frame.loc[fit, ["season", "game_month", TARGET]], 2024
    )

    def evaluate(prediction: np.ndarray, dataset):
        return (
            "source_rate_brier",
            _calibrated_brier(dataset.get_label(), prediction, rate),
            False,
        )

    lgb_path = checkpoint_dir / "lgb.json"
    if lgb_path.is_file():
        lgb_saved = json.loads(lgb_path.read_text(encoding="utf-8"))
        lgb_best = int(lgb_saved["best"])
        lgb_loss = float(lgb_saved["loss"])
    else:
        lgb_model = lgb.train(
            {**LGB_PARAMS, "num_threads": int(threads)},
            lgb.Dataset(x_train, y_train, categorical_feature=cats, free_raw_data=False),
            num_boost_round=LGB_MAX_ITERATIONS,
            valid_sets=[lgb.Dataset(x_valid, y_valid, categorical_feature=cats)],
            feval=evaluate,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        lgb_best = max(MIN_ITERATIONS, int(lgb_model.best_iteration))
        lgb_loss = _calibrated_brier(
            y_valid, lgb_model.predict(x_valid, num_iteration=lgb_best), rate
        )
        lgb_path.write_text(
            json.dumps({"best": lgb_best, "loss": lgb_loss}) + "\n",
            encoding="utf-8",
        )
        del lgb_model
        gc.collect()

    cat_path = checkpoint_dir / "cat.json"
    if cat_path.is_file():
        cat_saved = json.loads(cat_path.read_text(encoding="utf-8"))
        cat_best = int(cat_saved["best"])
        cat_loss = float(cat_saved["loss"])
    else:
        cat_model = CatBoostClassifier(
            iterations=CAT_MAX_ITERATIONS,
            thread_count=int(threads),
            **CAT_PARAMS,
        )
        train_pool = Pool(_cat_frame(x_train, cats), y_train, cat_features=cats)
        valid_pool = Pool(_cat_frame(x_valid, cats), cat_features=cats)
        del x_train, x_valid, features
        gc.collect()
        cat_model.fit(train_pool)
        cat_best = MIN_ITERATIONS
        cat_loss = float("inf")
        for index, prediction in enumerate(
            cat_model.staged_predict_proba(valid_pool, eval_period=CAT_EVAL_PERIOD),
            start=1,
        ):
            iteration = min(index * CAT_EVAL_PERIOD, CAT_MAX_ITERATIONS)
            loss = _calibrated_brier(y_valid, np.asarray(prediction)[:, 1], rate)
            if iteration >= MIN_ITERATIONS and loss < cat_loss:
                cat_best, cat_loss = iteration, loss
        cat_path.write_text(
            json.dumps({"best": cat_best, "loss": cat_loss}) + "\n",
            encoding="utf-8",
        )
    selection = {
            "inner_rate": float(rate),
            "lgb_best": int(lgb_best),
            "cat_best": int(cat_best),
            "lgb_calibrated_brier": float(lgb_loss),
            "cat_calibrated_brier": float(cat_loss),
    }
    result = {
        "lgb_iterations": max(MIN_ITERATIONS, int(round(lgb_best * ITERATION_SCALE))),
        "cat_iterations": max(MIN_ITERATIONS, int(round(cat_best * ITERATION_SCALE))),
        "selection": selection,
    }
    complete_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return int(result["lgb_iterations"]), int(result["cat_iterations"]), selection


def run(
    train_csv: Path,
    component_root: Path,
    public_model_dir: Path,
    output_dir: Path,
    *,
    threads: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    common, conditional, source_hashes = _modules(component_root)
    raw = pd.read_csv(train_csv, low_memory=False)
    frame, platoon, count = _feature_frame(raw, common, conditional)
    lgb_iterations, cat_iterations, selection = _select_iterations(
        frame, threads=threads, checkpoint_dir=output_dir / "selection_checkpoints"
    )

    public_config = json.loads(
        (public_model_dir / "config.json").read_text(encoding="utf-8")
    )
    feature_cols = list(public_config["feature_cols"])
    cat_cols = list(public_config["cat_cols"])
    local_features = frame.drop(
        columns=[column for column in DROP_COLUMNS if column in frame.columns]
    )
    if list(local_features.columns) != feature_cols:
        raise ValueError("public/local Component feature order mismatch")
    # The public CatBoost artifact has 319 trees while the source-only 2024
    # selection requires 324, and the public LightGBM text is not loadable by
    # the current native parser.  Refit the exact fixed recipes on all official
    # 2019--2024 rows at the already-selected iteration counts.
    local_lgb_path = output_dir / "lgbm_a_42.txt"
    local_cat_path = output_dir / "cat_bayes.cbm"
    for column in cat_cols:
        local_features[column] = pd.Categorical(
            local_features[column], categories=public_config["cat_maps"][column]
        )
    target = frame[TARGET].to_numpy(np.int8)
    reference_mask = frame["season"].eq(2024).to_numpy()
    reference = local_features.loc[reference_mask].copy()
    lgb_model = lgb.train(
        {**LGB_PARAMS, "num_threads": int(threads)},
        lgb.Dataset(local_features, target, categorical_feature=cat_cols),
        num_boost_round=lgb_iterations,
        callbacks=[lgb.log_evaluation(0)],
    )
    lgb_probability = lgb_model.predict(reference, num_iteration=lgb_iterations)
    lgb_model.save_model(str(local_lgb_path), num_iteration=lgb_iterations)
    del lgb_model
    gc.collect()

    cat_train = local_features.copy()
    for column in cat_cols:
        cat_train[column] = cat_train[column].astype(str)
    cat_reference = cat_train.loc[reference_mask].copy()
    cat_model = CatBoostClassifier(
        iterations=cat_iterations,
        thread_count=int(threads),
        **CAT_PARAMS,
    )
    cat_model.fit(Pool(cat_train, target, cat_features=cat_cols))
    cat_model.save_model(local_cat_path)
    cat_probability = cat_model.predict_proba(
        Pool(cat_reference, cat_features=cat_cols)
    )[:, 1]
    reference_raw = _sigmoid(
        0.5 * (_logit(lgb_probability) + _logit(cat_probability))
    )
    target_rate = _monthly_rate(raw[["season", "game_month", TARGET]], TARGET_YEAR)
    delta = _find_delta(reference_raw, target_rate)

    platoon25 = platoon.loc[platoon["target_season"].eq(TARGET_YEAR)].drop(
        columns="target_season"
    )
    count25 = count.loc[count["target_season"].eq(TARGET_YEAR)].drop(
        columns="target_season"
    )
    if platoon25.duplicated(["pitcher_id", "batter_hand"]).any():
        raise ValueError("duplicate platoon lookup")
    if count25.duplicated(["pitcher_id", "count_state"]).any():
        raise ValueError("duplicate count lookup")
    platoon25.to_csv(output_dir / "platoon_2025.csv", index=False)
    count25.to_csv(output_dir / "count_2025.csv", index=False)
    config = {
        "protocol": PROTOCOL,
        "target_year": TARGET_YEAR,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_maps": public_config["cat_maps"],
        "lgb_iterations": int(lgb_iterations),
        "cat_iterations": int(cat_iterations),
        "target_rate": float(target_rate),
        "delta": float(delta),
        "reference_raw_mean": float(reference_raw.mean()),
        "selection": selection,
        "external_source_sha256": source_hashes,
        "public_recipe_config_sha256": _sha256(public_model_dir / "config.json"),
        "refit_model_sha256": {
            "lgbm_a_42.txt": _sha256(local_lgb_path),
            "cat_bayes.cbm": _sha256(local_cat_path),
        },
        "official_train_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--public-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.component_root, args.public_model_dir,
        args.output_dir, threads=args.threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

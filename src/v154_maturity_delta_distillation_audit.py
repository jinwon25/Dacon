"""Distil the frozen v124-minus-intermediate maturity delta without outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V154_MATURITY_DELTA_DISTILLATION_AUDIT_V1"
TARGET_COL = "control_success"
DROP_COLUMNS = {"row_id", TARGET_COL, "season", "game_month"}
CATEGORICAL_COLUMNS = (
    "top_bottom", "game_type", "base_state", "balls_before", "strikes_before",
    "outs_before", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
)


def _feature_frame(
    frame: pd.DataFrame,
    columns: list[str],
    categories: dict[str, list[str]],
) -> pd.DataFrame:
    output = frame.loc[:, columns].copy()
    for column in CATEGORICAL_COLUMNS:
        if column in output:
            output[column] = pd.Categorical(
                output[column].astype("string").fillna("__MISSING__").astype(str),
                categories=categories[column],
            )
    for column in columns:
        if column not in CATEGORICAL_COLUMNS:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(np.float32)
    return output


def run(
    train_csv: Path,
    contract_path: Path,
    v104_path: Path,
    v141_path: Path,
    v138_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = _load_contract_axis(contract_path)
    with np.load(v104_path, allow_pickle=False) as saved:
        v104 = saved["full_2024"].astype(np.float64)
    with np.load(v141_path, allow_pickle=False) as saved:
        v124 = saved["v124_full_2024"].astype(np.float64)
        v142 = saved["full_2024"].astype(np.float64)
    with np.load(v138_path, allow_pickle=False) as saved:
        v138 = saved["full_2024"].astype(np.float64)
    if not (len(v104) == len(v124) == len(v142) == len(v138) == len(axis["raw_index"])):
        raise ValueError("distillation arrays are misaligned")

    header = pd.read_csv(train_csv, nrows=0).columns.tolist()
    columns = [column for column in header if column not in DROP_COLUMNS]
    source = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    frame = source.iloc[axis["raw_index"].astype(np.int64)].reset_index(drop=True)
    categories = {
        column: sorted(
            frame[column].astype("string").fillna("__MISSING__").astype(str).unique().tolist()
        )
        for column in CATEGORICAL_COLUMNS if column in columns
    }
    features = _feature_frame(frame, columns, categories)
    weight = float(config["bridge_weight"])
    intermediate = np.clip(v124 + weight * (v104 - v124), 0.001, 0.999)
    delta = v124 - intermediate
    maturity = np.isin(axis["game_month"], config["maturity_months"])
    if not maturity.any() or maturity.all():
        raise ValueError("maturity validation split is invalid")
    training = config["training"]
    train_set = lgb.Dataset(
        features.loc[~maturity], label=delta[~maturity],
        categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in columns],
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        features.loc[maturity], label=delta[maturity], reference=train_set,
        categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in columns],
        free_raw_data=False,
    )
    model = lgb.train(
        {
            "objective": "regression_l2", "metric": "l2",
            "learning_rate": float(training["learning_rate"]),
            "num_leaves": int(training["num_leaves"]),
            "min_data_in_leaf": int(training["min_data_in_leaf"]),
            "lambda_l2": float(training["lambda_l2"]),
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
            "seed": int(training["seed"]), "verbosity": -1, "num_threads": -1,
        },
        train_set,
        num_boost_round=int(training["num_boost_round"]),
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(int(training["early_stopping_rounds"]), verbose=False)],
    )
    predicted_delta = np.asarray(
        model.predict(features.loc[maturity], num_iteration=model.best_iteration),
        dtype=np.float64,
    )
    error = predicted_delta - delta[maturity]
    ss_total = float(np.sum((delta[maturity] - delta[maturity].mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / ss_total
    ungated = np.clip(v142 + weight * (v138 - v142), 0.001, 0.999)
    candidate = ungated.copy()
    candidate[maturity] = np.clip(intermediate[maturity] + predicted_delta, 0.001, 0.999)
    ungated_metrics = _axis_metrics({**axis, "parent": v124}, ungated)
    candidate_metrics = _axis_metrics({**axis, "parent": v124}, candidate)
    gate_increment = float(candidate_metrics["gain"] - ungated_metrics["gain"])
    public_before = float(config["public_before_gate"])
    estimates = {
        "conservative": public_before + 0.5 * gate_increment,
        "family_transfer_centre": public_before + float(config["family_transfer_rate"]) * gate_increment,
        "optimistic": public_before + gate_increment,
    }
    model_path = output_dir / "maturity_delta_lgb.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    schema = {
        "protocol": PROTOCOL,
        "feature_columns": columns,
        "categorical_columns": [c for c in CATEGORICAL_COLUMNS if c in columns],
        "categories": categories,
        "best_iteration": int(model.best_iteration),
        "maturity_months": config["maturity_months"],
        "bridge_weight": weight,
    }
    (output_dir / "maturity_delta_spec.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_package" if estimates["family_transfer_centre"] >= float(config["target_public"]) else "reject",
        "validation_rows": int(maturity.sum()),
        "training_rows": int((~maturity).sum()),
        "delta_validation": {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "maximum_absolute": float(np.max(np.abs(error))),
            "r2": r2,
            "correlation": float(np.corrcoef(predicted_delta, delta[maturity])[0, 1]),
        },
        "ungated_gain": float(ungated_metrics["gain"]),
        "candidate_metrics": candidate_metrics,
        "gate_increment": gate_increment,
        "exact_gate_increment_reference": 2.2985389201084683,
        "gate_increment_retention": gate_increment / 2.2985389201084683,
        "public_estimate": estimates,
        "model": {"path": str(model_path), "best_iteration": int(model.best_iteration)},
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--v141-path", type=Path, required=True)
    parser.add_argument("--v138-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_path, args.v104_path, args.v141_path,
        args.v138_path, args.config, args.output_dir)


if __name__ == "__main__":
    main()

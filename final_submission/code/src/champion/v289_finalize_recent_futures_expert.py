"""Fit the five-seed recent-F expert on 2024 official-train rows only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from src.archive.v287_recent_futures_direct_expert import (
    CAT_COLUMNS,
    PARAMS,
    TARGET,
    build_features,
)


PROTOCOL = "V289_FINALIZE_RECENT_FUTURES_EXPERT_V1"
SEEDS = (2871, 2873, 2875, 2877, 2879)
EXTRA_CAT_COLUMNS = ["count_code", "same_hand", "pressure_code"]


def run(train_csv: Path, v288_summary: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(v288_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V288_FUTURES_MULTISEED_FIXED_AUDIT_V1":
        raise ValueError("unexpected v288 summary protocol")
    if not audit.get("eligible_for_exploratory_packaging"):
        raise ValueError("v288 is not eligible for exploratory packaging")
    if float(audit.get("futures_weight")) != 0.10:
        raise ValueError("unexpected futures expert weight")

    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    fit_rows = train.loc[
        train["season"].eq(2024)
        & train["game_type"].astype(str).eq("F")
    ].reset_index(drop=True)
    features = build_features(fit_rows)
    cat_columns = [
        column
        for column in CAT_COLUMNS + EXTRA_CAT_COLUMNS
        if column in features.columns
    ]
    for seed in SEEDS:
        model = CatBoostClassifier(**PARAMS, random_seed=seed)
        model.fit(
            features,
            fit_rows[TARGET].to_numpy("int8"),
            cat_features=cat_columns,
        )
        model.save_model(output_dir / f"futures_expert_seed{seed}.cbm")

    (output_dir / "feature_columns.json").write_text(
        json.dumps(list(features.columns), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cat_columns.json").write_text(
        json.dumps(cat_columns, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "final_model_ready",
        "fit_season": 2024,
        "fit_game_type": "F",
        "fit_rows": len(fit_rows),
        "feature_count": len(features.columns),
        "seeds": list(SEEDS),
        "model_params": PARAMS,
        "futures_weight": 0.10,
        "v288_summary": str(v288_summary),
        "restrictions": {
            "official_train_only": True,
            "one_latest_prior_season_only": True,
            "test_csv_read": False,
            "public_score_used_for_training": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v288-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_csv, args.v288_summary, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

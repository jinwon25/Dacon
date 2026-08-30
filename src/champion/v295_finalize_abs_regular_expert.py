"""Fit the frozen five-seed 2024 post-ABS regular-season expert.

There is no honest labelled R-2025 validation axis.  This finalizer therefore
does not optimize a score, route, seed, or dose: it enforces the v294 contract,
fits the exact v287/v292 CatBoost recipe on official 2024 R rows, and records
the unavoidable deployment extrapolation explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
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


PROTOCOL = "V295_FINALIZE_ABS_REGULAR_EXPERT_V1"
SEEDS = (2871, 2873, 2875, 2877, 2879)
EXTRA_CAT_COLUMNS = ["count_code", "same_hand", "pressure_code"]
WEIGHT = 0.10
BLEND_MODE = "logit"


def run(train_csv: Path, v294_summary: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(v294_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V294_ABS_REGULAR_SANITY_AUDIT_V1":
        raise ValueError("unexpected v294 summary protocol")
    if audit.get("status") != "sanity_reproduced":
        raise ValueError("v294 F sanity was not reproduced")
    contract = audit["deployment_contract"]
    if float(contract["weight"]) != WEIGHT or contract["blend_mode"] != BLEND_MODE:
        raise ValueError("v294 deployment contract changed")
    if tuple(contract["seeds"]) != SEEDS:
        raise ValueError("v294 seed contract changed")

    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    fit_rows = train.loc[
        train["season"].eq(2024)
        & train["game_type"].astype(str).eq("R")
    ].reset_index(drop=True)
    features = build_features(fit_rows)
    cat_columns = [
        column
        for column in CAT_COLUMNS + EXTRA_CAT_COLUMNS
        if column in features.columns
    ]
    elapsed: dict[str, float] = {}
    for seed in SEEDS:
        started = time.perf_counter()
        print(f"[v295] fitting seed={seed} rows={len(fit_rows):,}", flush=True)
        model = CatBoostClassifier(**PARAMS, random_seed=seed)
        model.fit(
            features,
            fit_rows[TARGET].to_numpy("int8"),
            cat_features=cat_columns,
        )
        model.save_model(output_dir / f"abs_regular_expert_seed{seed}.cbm")
        elapsed[str(seed)] = float(time.perf_counter() - started)
        print(f"[v295] saved seed={seed} elapsed={elapsed[str(seed)]:.1f}s", flush=True)

    (output_dir / "feature_columns.json").write_text(
        json.dumps(list(features.columns), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cat_columns.json").write_text(
        json.dumps(cat_columns, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "final_model_ready",
        "fit_season": 2024,
        "fit_game_type": "R",
        "fit_rows": len(fit_rows),
        "feature_count": len(features.columns),
        "seeds": list(SEEDS),
        "model_params": PARAMS,
        "weight": WEIGHT,
        "blend_mode": BLEND_MODE,
        "fit_elapsed_seconds": elapsed,
        "v294_summary": str(v294_summary),
        "evidence_contract": {
            "kbo_league_abs_start": 2024,
            "futures_abs_start": 2020,
            "f_2023_break_is_abs_introduction": False,
            "honest_local_r2025_labels_available": False,
            "candidate_type": "high_risk_deployment_extrapolation",
        },
        "restrictions": {
            "official_train_only": True,
            "single_post_abs_regular_season_only": True,
            "test_csv_read": False,
            "public_score_used_for_training": False,
            "parameters_retuned": False,
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
    parser.add_argument("--v294-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v294_summary, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

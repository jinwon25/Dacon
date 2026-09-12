"""Audit a one-prior-season direct expert for the regular-season process.

This is a domain transfer of the recent-process idea, not a weight probe of
v290.  One CatBoost classifier is fitted on the immediately preceding regular
season for each origin and evaluated on the next season.  The blend dose is
fixed at 10% before opening any result.  Route selection uses 2022 and
late-2023 only; full-2024 remains locked confirmation.  No evaluation rows or
Public score enter fitting or selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.archive.v287_recent_futures_direct_expert import build_features


PROTOCOL = "V292_RECENT_REGULAR_DIRECT_EXPERT_V1"
TARGET = "control_success"
WEIGHT = 0.10
ANCHOR_TEAM = 13
CAT_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
    "count_code",
    "same_hand",
    "pressure_code",
]
PARAMS = {
    "iterations": 700,
    "depth": 7,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "l2_leaf_reg": 30.0,
    "random_strength": 0.35,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "one_hot_max_size": 16,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": 16,
}


def _fit_predict(
    fit_rows: pd.DataFrame, audit_rows: pd.DataFrame, seed: int
) -> tuple[np.ndarray, CatBoostClassifier]:
    fit_features = build_features(fit_rows)
    audit_features = build_features(audit_rows).reindex(columns=fit_features.columns)
    cats = [column for column in CAT_COLUMNS if column in fit_features.columns]
    model = CatBoostClassifier(**PARAMS, random_seed=int(seed))
    model.fit(
        fit_features,
        fit_rows[TARGET].to_numpy(np.int8),
        cat_features=cats,
    )
    return model.predict_proba(audit_features)[:, 1].astype(np.float64), model


def _routes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(ANCHOR_TEAM)
        | frame["batter_team_id"].eq(ANCHOR_TEAM)
    ).to_numpy()
    return {
        "R_ALL": regular,
        "R_CORE": regular & ~anchor,
        "R_ANCHOR": regular & anchor,
    }


def _blend(parent: np.ndarray, expert: np.ndarray, active: np.ndarray) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + WEIGHT * (expert[active] - output[active]),
        0.001,
        0.999,
    )
    return output


def run(
    train_csv: Path,
    parent_axes: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    regular = train["game_type"].astype(str).eq("R")
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    fits = {
        "full_2022": train.loc[train["season"].eq(2021) & regular].reset_index(drop=True),
        "late_2023": train.loc[train["season"].eq(2022) & regular].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2023) & regular].reset_index(drop=True),
    }
    with np.load(parent_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_{axis}"].astype(np.float64)
            for axis in frames
        }
    predictions: dict[str, np.ndarray] = {}
    for offset, axis in enumerate(frames):
        if len(parents[axis]) != len(frames[axis]):
            raise ValueError(f"parent alignment mismatch: {axis}")
        predictions[axis], model = _fit_predict(
            fits[axis], frames[axis], seed + offset
        )
        model.save_model(output_dir / f"recent_regular_{axis}.cbm")
        np.save(output_dir / f"expert_{axis}.npy", predictions[axis].astype(np.float32))

    metrics: dict[str, dict[str, Any]] = {}
    candidates: dict[tuple[str, str], np.ndarray] = {}
    source_rows: list[dict[str, Any]] = []
    for route in ("R_ALL", "R_CORE", "R_ANCHOR"):
        per_axis: dict[str, Any] = {}
        for axis, frame in frames.items():
            active = _routes(frame)[route]
            candidate = _blend(parents[axis], predictions[axis], active)
            candidates[(route, axis)] = candidate
            per_axis[axis] = paired_metrics(
                {
                    "target": frame[TARGET].to_numpy(np.float64),
                    "exact_mask": np.ones(len(frame), dtype=bool),
                    "game_month": frame["game_month"].to_numpy(),
                },
                parents[axis],
                candidate,
                active,
            )
        metrics[route] = per_axis
        source_gains = [
            per_axis["full_2022"]["gain"],
            per_axis["late_2023"]["gain"],
        ]
        source_rows.append(
            {
                "route": route,
                "source_min_gain": float(min(source_gains)),
                "source_mean_gain": float(np.mean(source_gains)),
                "source_gate_passed": bool(min(source_gains) > 0.0),
                "locked_gain": float(per_axis["full_2024"]["gain"]),
            }
        )

    eligible = [row for row in source_rows if row["source_gate_passed"]]
    selected = (
        max(eligible, key=lambda row: (row["source_min_gain"], row["source_mean_gain"]))
        if eligible
        else max(source_rows, key=lambda row: row["source_min_gain"])
    )
    selected_route = str(selected["route"])
    active_2024 = _routes(frames["full_2024"])[selected_route]
    robustness = _robustness(
        {
            "target": frames["full_2024"][TARGET].to_numpy(np.float64),
            "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
            "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        },
        parents["full_2024"],
        candidates[(selected_route, "full_2024")],
        active_2024,
        [candidates[(route, "full_2024")] for route in ("R_ALL", "R_CORE", "R_ANCHOR")],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    promote = bool(
        selected["source_gate_passed"]
        and selected["locked_gain"] > 0.0
        and robust_pass
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "numeric_promote" if promote else "rejected",
        "weight": WEIGHT,
        "seed": int(seed),
        "rows": {
            axis: {"fit_regular": len(fits[axis]), "audit_all": len(frames[axis])}
            for axis in frames
        },
        "model_params": PARAMS,
        "selected": selected,
        "all_routes": source_rows,
        "selected_metrics": metrics[selected_route],
        "locked_robustness": robustness,
        "numeric_promotion_gate_passed": promote,
        "restrictions": {
            "official_train_only": True,
            "one_prior_regular_season_only": True,
            "weight_fixed_before_results": True,
            "route_selected_on_full22_and_late23_only": True,
            "full_2024_used_for_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
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
    parser.add_argument("--parent-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2920)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.parent_axes, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()

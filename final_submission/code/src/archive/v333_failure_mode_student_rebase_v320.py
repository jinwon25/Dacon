"""Strict-forward four-mode failure student rebased above v320.

Training-only cumulative ASOF transitions reconstruct four mutually exclusive
failure modes.  A compact multiclass LightGBM predicts those modes from legal
pre-pitch fields.  Earlier-season conditional success rates turn the predicted
mode distribution into an alternative success probability.

The blend route and dose must improve full-2022 and late-2023 before full-2024
is opened.  This is an alternate-label model, not another direct-success tree.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.failure_mode_privileged_distillation import MODE_NAMES, reconstruct_failure_mode


PROTOCOL = "V333_FAILURE_MODE_STUDENT_REBASE_V320_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.04)
HALF_LIFE = 2.0

NUMERIC = (
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)
CATEGORICAL = (
    "game_dayofweek", "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "pitcher_id", "batter_id",
)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Legal row-local features with fixed baseball interactions."""

    output = pd.DataFrame(index=frame.index)
    for column in NUMERIC:
        output[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    for column in CATEGORICAL:
        output[f"cat__{column}"] = (
            frame[column].astype("string").fillna("__MISSING__").astype("category")
        )
    output["cat__count"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    ).astype("category")
    output["cat__same_hand"] = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).astype(str).astype("category")
    output["cat__pressure"] = np.select(
        [frame["balls_before"].eq(3), frame["strikes_before"].eq(2)],
        ["THREE_BALL", "TWO_STRIKE"], default="NORMAL",
    )
    output["cat__pressure"] = output["cat__pressure"].astype("category")
    cats = [column for column in output if column.startswith("cat__")]
    return output, cats


def mode_classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(MODE_NAMES),
        verbosity=-1,
        n_jobs=12,
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=1000,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=3.0,
        reg_lambda=20.0,
        max_bin=127,
        random_state=seed,
    )


def fit_predict_raw(
    train: pd.DataFrame,
    features: pd.DataFrame,
    categorical: list[str],
    mode: np.ndarray,
    audit_year: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit = train["season"].lt(audit_year).to_numpy() & (mode >= 0)
    audit = train["season"].eq(audit_year).to_numpy()
    fit_year = train.loc[fit, "season"].to_numpy(np.float64)
    sample_weight = np.exp2(-(audit_year - 1.0 - fit_year) / HALF_LIFE)
    sample_weight /= sample_weight.mean()
    model = mode_classifier(seed=33300 + audit_year)
    model.fit(
        features.loc[fit], mode[fit].astype(np.int8),
        sample_weight=sample_weight, categorical_feature=categorical,
    )
    probability = model.predict_proba(features.loc[audit]).astype(np.float64)
    del model
    gc.collect()
    fit_target = train.loc[fit, TARGET].to_numpy(np.float64)
    conditional = np.asarray(
        [
            np.average(
                fit_target[mode[fit] == klass],
                weights=sample_weight[mode[fit] == klass],
            )
            for klass in range(len(MODE_NAMES))
        ],
        dtype=np.float64,
    )
    raw = np.clip(probability @ conditional, 0.001, 0.999)
    audit_mode = mode[audit]
    known = audit_mode >= 0
    return raw, {
        "audit_year": audit_year,
        "fit_rows": int(fit.sum()),
        "audit_rows": int(audit.sum()),
        "mode_coverage": float(known.mean()),
        "mode_accuracy": float(
            np.mean(np.argmax(probability[known], axis=1) == audit_mode[known])
        ),
        "conditional_success": {
            name: float(value) for name, value in zip(MODE_NAMES, conditional)
        },
        "raw_mean": float(raw.mean()),
    }


def route_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    ).to_numpy()
    return {
        "ALL": np.ones(len(frame), dtype=bool),
        "R_CORE": regular & ~anchor,
        "R_ANCHOR": regular & anchor,
        "F": ~regular,
    }


def apply_blend(
    parent: np.ndarray, raw: np.ndarray, active: np.ndarray, weight: float
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        parent[active] + float(weight) * (raw[active] - parent[active]),
        0.001, 0.999,
    )
    return output


def run(
    train_csv: Path,
    v285_axes: Path,
    v318_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, low_memory=False)
    mode = reconstruct_failure_mode(train)
    features, categorical = build_features(train)
    raw_year: dict[int, np.ndarray] = {}
    classifier_metrics: dict[str, Any] = {}
    for year in (2022, 2023, 2024):
        cache = output_dir / f"raw_o{year}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                raw_year[year] = saved["raw"].astype(np.float64)
                classifier_metrics[str(year)] = json.loads(str(saved["metrics_json"].item()))
            print(f"[v333] reuse year={year}", flush=True)
        else:
            print(f"[v333] fit year={year}", flush=True)
            raw_year[year], metric = fit_predict_raw(
                train, features, categorical, mode, year
            )
            classifier_metrics[str(year)] = metric
            np.savez_compressed(
                cache,
                raw=raw_year[year],
                metrics_json=np.asarray(json.dumps(metric, ensure_ascii=False)),
            )
    del features
    gc.collect()

    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    late23 = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
    raw = {
        "full_2022": raw_year[2022],
        "late_2023": raw_year[2023][late23],
        "full_2024": raw_year[2024],
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parents = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v318_axes, allow_pickle=False) as saved:
        parents.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )
    axes = {
        origin: {
            "target": frames[origin][TARGET].to_numpy(np.float64),
            "game_month": frames[origin]["game_month"].to_numpy(np.int16),
            "pitcher_id": frames[origin]["pitcher_id"].to_numpy(),
            "batter_id": frames[origin]["batter_id"].to_numpy(),
            "exact_mask": np.ones(len(frames[origin]), dtype=bool),
        }
        for origin in ORIGINS
    }
    masks = {origin: route_masks(frames[origin]) for origin in ORIGINS}
    trials: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float, str], np.ndarray] = {}
    for route in ROUTES:
        for weight in WEIGHTS:
            metrics: dict[str, Any] = {}
            for origin in ORIGINS:
                candidate = apply_blend(
                    parents[origin], raw[origin], masks[origin][route], weight
                )
                predictions[(route, weight, origin)] = candidate
                metrics[origin] = paired_metrics(
                    axes[origin], parents[origin], candidate, masks[origin][route]
                )
            source = [metrics["full_2022"], metrics["late_2023"]]
            source_pass = bool(
                all(item["gain"] > 0.0 for item in source)
                and all(item["positive_month_fraction"] >= 0.50 for item in source)
                and all(item["worst_month_gain"] > -5.0 for item in source)
            )
            trials.append(
                {
                    "route": route,
                    "weight": weight,
                    "source_passed": source_pass,
                    "minimum_source_gain": min(item["gain"] for item in source),
                    "mean_source_gain": float(np.mean([item["gain"] for item in source])),
                    "metrics": metrics,
                }
            )
    selected = max(
        trials,
        key=lambda row: (
            row["source_passed"], row["minimum_source_gain"],
            row["mean_source_gain"], -row["weight"],
        ),
    )
    route = str(selected["route"])
    weight = float(selected["weight"])
    candidate24 = predictions[(route, weight, "full_2024")]
    robustness = _robustness(
        axes["full_2024"], parents["full_2024"], candidate24,
        masks["full_2024"][route],
        [parents["full_2024"], *[
            predictions[(candidate_route, candidate_weight, "full_2024")]
            for candidate_route in ROUTES for candidate_weight in WEIGHTS
        ]],
    )
    locked = selected["metrics"]["full_2024"]
    locked_pass = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{origin}": parents[origin] for origin in ORIGINS},
        **{
            f"candidate_{origin}": predictions[(route, weight, origin)]
            for origin in ORIGINS
        },
        **{f"raw_{origin}": raw[origin] for origin in ORIGINS},
        **{f"active_{origin}": masks[origin][route] for origin in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_candidate" if selected["source_passed"] and locked_pass and robust_pass else (
            "source_reject" if not selected["source_passed"] else "locked_reject"
        ),
        "classifier_metrics": classifier_metrics,
        "selected": selected,
        "all_trials": trials,
        "locked_robustness": robustness,
        "source_gate_passed": bool(selected["source_passed"]),
        "locked_gate_passed": locked_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(selected["source_passed"] and locked_pass and robust_pass),
        "restrictions": {
            "official_train_only": True,
            "failure_labels_reconstructed_only_inside_labelled_train": True,
            "current_pitch_failure_mode_not_used_at_inference": True,
            "route_and_weight_selected_on_2022_and_late_2023_only": True,
            "full_2024_locked_from_selection": True,
            "all_declared_candidates_in_reality_check": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
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
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v318-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v285_axes, args.v318_axes, args.output_dir
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

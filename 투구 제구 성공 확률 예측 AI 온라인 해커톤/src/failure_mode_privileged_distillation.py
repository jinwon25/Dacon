"""Distil current-pitch TrackMan information through interpretable failure modes.

The next row of each pitcher's cumulative ASOF counters reconstructs the
training pitch's reverse/middle/ball/strike flags.  Those labels are legal only
inside the labelled training range.  A privileged teacher sees current-pitch
TrackMan measurements, while all audit predictions are produced by students
that see main-table fields only.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.trackman_privileged_distillation import (
    CATEGORICAL_COLUMNS,
    PHYSICAL_COLUMNS,
    V17_NAME,
    _diagnostics,
    _prepare_frames,
    bss,
)


TRANSITIONS = ((2022, 2023), (2023, 2024))
RATE_COLUMNS = [
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
]
MODE_NAMES = ("bad_location", "ball_only", "strike_only", "other")


def reconstruct_failure_mode(main: pd.DataFrame) -> np.ndarray:
    """Return 0..3 failure modes, or -1 when the next ASOF row is unavailable."""
    n = main["asof_pitcher_n"].to_numpy(np.float64)
    next_n = main.groupby("pitcher_id", sort=False)["asof_pitcher_n"].shift(-1).to_numpy(np.float64)
    flags: list[np.ndarray] = []
    valid = np.isfinite(next_n) & np.isclose(next_n, n + 1.0)
    for column in RATE_COLUMNS:
        current_rate = main[column].fillna(0.0).to_numpy(np.float64)
        next_rate = (
            main.groupby("pitcher_id", sort=False)[column]
            .shift(-1)
            .to_numpy(np.float64)
        )
        increment = np.rint(next_n * next_rate) - np.rint(n * current_rate)
        flags.append(np.where(valid, increment, np.nan))
    reverse, middle, ball, strike = flags
    valid &= np.logical_and.reduce([np.isfinite(value) for value in flags])
    mode = np.full(len(main), -1, dtype=np.int8)
    bad = valid & ((reverse > 0.5) | (middle > 0.5))
    ball_only = valid & ~bad & (ball > 0.5)
    strike_only = valid & ~bad & ~ball_only & (strike > 0.5)
    other = valid & ~bad & ~ball_only & ~strike_only
    mode[bad] = 0
    mode[ball_only] = 1
    mode[strike_only] = 2
    mode[other] = 3
    return mode


def _teacher(*, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        verbosity=-1,
        n_jobs=6,
        n_estimators=140,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=500,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0,
        reg_lambda=12.0,
        max_bin=127,
        random_state=seed,
    )


def _student(*, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=700,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=3.0,
        reg_lambda=20.0,
        max_bin=127,
        random_state=seed,
    )


def _normalise(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, None)
    return probability / probability.sum(axis=1, keepdims=True)


def _crossfit_teachers(
    source_x: pd.DataFrame,
    mode: np.ndarray,
    groups: np.ndarray,
    safe_columns: list[str],
    full_columns: list[str],
    folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    safe_oof = np.zeros((len(mode), 4), dtype=np.float64)
    full_oof = np.zeros((len(mode), 4), dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit, valid) in enumerate(splitter.split(source_x, mode, groups), start=1):
        safe = _teacher(seed=2100 + fold)
        safe.fit(
            source_x.iloc[fit][safe_columns],
            mode[fit],
            categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in safe_columns],
        )
        safe_oof[valid] = safe.predict_proba(source_x.iloc[valid][safe_columns])
        full = _teacher(seed=2200 + fold)
        full.fit(
            source_x.iloc[fit][full_columns],
            mode[fit],
            categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in full_columns],
        )
        full_oof[valid] = full.predict_proba(source_x.iloc[valid][full_columns])
        del safe, full
        gc.collect()
    return _normalise(safe_oof), _normalise(full_oof)


def _fit_probability_students(
    source_x: pd.DataFrame,
    audit_x: pd.DataFrame,
    target_probability: np.ndarray,
    columns: list[str],
    seed: int,
) -> np.ndarray:
    prediction = np.zeros((len(audit_x), 4), dtype=np.float64)
    categories = [c for c in CATEGORICAL_COLUMNS if c in columns]
    for klass in range(4):
        model = _student(seed=seed + klass)
        model.fit(
            source_x[columns],
            target_probability[:, klass],
            categorical_feature=categories,
        )
        prediction[:, klass] = model.predict(audit_x[columns])
        del model
        gc.collect()
    return _normalise(prediction)


def _multiclass_brier(target: np.ndarray, probability: np.ndarray) -> float:
    onehot = np.eye(4, dtype=np.float64)[target]
    return float(np.mean(np.sum(np.square(probability - onehot), axis=1)))


def run(project: Path, alignment_dir: Path, output_dir: Path, folds: int) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = (project / alignment_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = np.load(alignment_dir / "pitch_alignment.npz")
    aligned_main_index = alignment["main_index"].astype(np.int64)
    aligned_trackman_index = alignment["trackman_index"].astype(np.int64)
    aligned_season = alignment["season"].astype(np.int16)
    aligned_game = alignment["main_game_id"].astype(np.int32)
    main = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    mode_all = reconstruct_failure_mode(main)
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=PHYSICAL_COLUMNS,
        low_memory=False,
    )
    safe_columns = [c for c in main.columns if c not in {"row_id", "control_success"}]
    physical_columns = [f"tm_{column}" for column in PHYSICAL_COLUMNS]
    full_columns = safe_columns + physical_columns
    metric_rows: list[dict[str, object]] = []
    teacher_rows: list[dict[str, object]] = []
    for source_year, audit_year in TRANSITIONS:
        source_mask = aligned_season == source_year
        source_index_all = aligned_main_index[source_mask]
        valid_mode = mode_all[source_index_all] >= 0
        source_index = source_index_all[valid_mode]
        source = main.iloc[source_index].reset_index(drop=True).copy()
        audit_all = main.loc[main["season"].eq(audit_year)].reset_index(drop=True).copy()
        source_physical = trackman.iloc[
            aligned_trackman_index[source_mask][valid_mode]
        ].reset_index(drop=True)
        for column in PHYSICAL_COLUMNS:
            source[f"tm_{column}"] = source_physical[column].to_numpy()
            audit_all[f"tm_{column}"] = np.nan
        source_x, _, audit_x = _prepare_frames(
            source, source.iloc[:0].copy(), audit_all, full_columns
        )
        source_mode = mode_all[source_index].astype(np.int64)
        source_target = source["control_success"].to_numpy(np.float64)
        source_groups = aligned_game[source_mask][valid_mode]
        safe_oof, full_oof = _crossfit_teachers(
            source_x,
            source_mode,
            source_groups,
            safe_columns,
            full_columns,
            folds,
        )
        conditional_success = np.array(
            [source_target[source_mode == klass].mean() for klass in range(4)],
            dtype=np.float64,
        )
        onehot = np.eye(4, dtype=np.float64)[source_mode]
        control_probability = _fit_probability_students(
            source_x, audit_x, onehot, safe_columns, seed=2300
        )
        control_success = control_probability @ conditional_success
        artifact = np.load(
            project
            / "artifacts"
            / "v16_multiseason_20260815_02"
            / f"{V17_NAME}_o{audit_year}.npz"
        )
        incumbent = artifact["candidate"].astype(np.float64)
        audit_target = artifact["target"].astype(np.float64)
        teacher_rows.append(
            {
                "source_year": source_year,
                "audit_year": audit_year,
                "source_rows": int(len(source_x)),
                "mode_counts": {
                    MODE_NAMES[klass]: int(np.sum(source_mode == klass))
                    for klass in range(4)
                },
                "conditional_success": {
                    MODE_NAMES[klass]: float(conditional_success[klass])
                    for klass in range(4)
                },
                "safe_multiclass_brier": _multiclass_brier(source_mode, safe_oof),
                "full_multiclass_brier": _multiclass_brier(source_mode, full_oof),
            }
        )
        for label_weight in (0.25, 0.50, 0.75):
            soft = (1.0 - label_weight) * onehot + label_weight * full_oof
            distilled_probability = _fit_probability_students(
                source_x,
                audit_x,
                soft,
                safe_columns,
                seed=2400 + int(100 * label_weight),
            )
            distilled_success = distilled_probability @ conditional_success
            correction = distilled_success - control_success
            for weight in (0.25, 0.50, 0.75, 1.0):
                candidate = np.clip(incumbent + weight * correction, 0.001, 0.999)
                diagnostic = _diagnostics(
                    audit_all, audit_target, incumbent, candidate
                )
                metric_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "candidate": f"mode_pfd_l{int(100 * label_weight):03d}",
                        "weight": weight,
                        "gain_vs_v17": diagnostic["gain"],
                        "month_positive_fraction": float(
                            np.mean([row["gain"] > 0 for row in diagnostic["months"]])
                        ),
                        "worst_month_gain": float(
                            min(row["gain"] for row in diagnostic["months"])
                        ),
                        "minimum_domain_gain": float(
                            min(row["gain"] for row in diagnostic["domains"])
                        ),
                        "correction_sd": float(np.std(correction)),
                    }
                )
        del source_x, audit_x, safe_oof, full_oof, control_probability
        gc.collect()
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust = (
        metrics.groupby(["candidate", "weight"], observed=True)["gain_vs_v17"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "FAILURE_MODE_PRIVILEGED_DISTILLATION_FORWARD_V1",
        "teacher": teacher_rows,
        "robust": robust.to_dict(orient="records"),
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
        "--alignment-dir",
        type=Path,
        default=Path("artifacts/trackman_privileged_20260816"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/failure_mode_pfd_20260816_01"),
    )
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.output_dir, args.folds)


if __name__ == "__main__":
    main()

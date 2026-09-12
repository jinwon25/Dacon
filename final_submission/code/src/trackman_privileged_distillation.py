"""Forward test of training-only TrackMan privileged-information distillation.

For each source season a pair of otherwise identical teachers is cross-fitted
by game.  The privileged teacher sees current-pitch TrackMan measurements; the
control teacher sees only inference-safe main-table fields.  Their OOF
prediction difference is then learned by a student that sees safe fields only.

Only the student's prediction is applied to the next season.  The current-pitch
TrackMan columns never enter the audit-season student prediction matrix.
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


TRANSITIONS = ((2022, 2023), (2023, 2024))
PHYSICAL_COLUMNS = [
    "pitch_type_group",
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
ID_COLUMNS = {
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
}
CATEGORICAL_COLUMNS = [
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "tm_pitch_type_group",
]
V17_NAME = "multi_pitcher_batter_hand_pressure_d1_a3200_w1.5"


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(target))
    return float(100000.0 * (1.0 - np.mean(np.square(prediction - target)) / (rate * (1.0 - rate))))


def _model(*, student: bool, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180 if student else 140,
        learning_rate=0.025,
        num_leaves=7 if student else 15,
        max_depth=3 if student else 4,
        min_child_samples=700 if student else 500,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=3.0 if student else 2.0,
        reg_lambda=20.0 if student else 12.0,
        max_bin=127,
        random_state=seed,
    )


def _prepare_frames(
    source: pd.DataFrame,
    audit_aligned: pd.DataFrame,
    audit_all: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sizes = (len(source), len(audit_aligned), len(audit_all))
    combined = pd.concat(
        [source[columns], audit_aligned[columns], audit_all[columns]],
        ignore_index=True,
    )
    for column in columns:
        if column in CATEGORICAL_COLUMNS:
            combined[column] = (
                combined[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype("category")
            )
        else:
            combined[column] = pd.to_numeric(
                combined[column], errors="coerce"
            ).astype(np.float32)
    first, second, third = sizes
    return (
        combined.iloc[:first].reset_index(drop=True),
        combined.iloc[first : first + second].reset_index(drop=True),
        combined.iloc[first + second : first + second + third].reset_index(drop=True),
    )


def _crossfit_teacher_delta(
    safe_x: pd.DataFrame,
    full_x: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    safe_columns: list[str],
    full_columns: list[str],
    folds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    safe_prediction = np.zeros(len(target), dtype=np.float64)
    full_prediction = np.zeros(len(target), dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    safe_categories = [c for c in CATEGORICAL_COLUMNS if c in safe_columns]
    full_categories = [c for c in CATEGORICAL_COLUMNS if c in full_columns]
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(safe_x, target, groups), start=1
    ):
        safe_model = _model(student=False, seed=8160 + fold)
        safe_model.fit(
            safe_x.iloc[fit_index][safe_columns],
            target[fit_index],
            categorical_feature=safe_categories,
        )
        safe_prediction[valid_index] = safe_model.predict(
            safe_x.iloc[valid_index][safe_columns]
        )
        full_model = _model(student=False, seed=9160 + fold)
        full_model.fit(
            full_x.iloc[fit_index][full_columns],
            target[fit_index],
            categorical_feature=full_categories,
        )
        full_prediction[valid_index] = full_model.predict(
            full_x.iloc[valid_index][full_columns]
        )
        del safe_model, full_model
        gc.collect()
    return safe_prediction, full_prediction, {
        "source_safe_bss": bss(target, np.clip(safe_prediction, 0.001, 0.999)),
        "source_full_bss": bss(target, np.clip(full_prediction, 0.001, 0.999)),
        "source_privileged_gain": bss(target, np.clip(full_prediction, 0.001, 0.999))
        - bss(target, np.clip(safe_prediction, 0.001, 0.999)),
        "teacher_delta_mean": float(np.mean(full_prediction - safe_prediction)),
        "teacher_delta_sd": float(np.std(full_prediction - safe_prediction)),
    }


def _oracle_audit_delta(
    source_safe: pd.DataFrame,
    source_full: pd.DataFrame,
    audit_safe: pd.DataFrame,
    audit_full: pd.DataFrame,
    target: np.ndarray,
    safe_columns: list[str],
    full_columns: list[str],
) -> np.ndarray:
    safe_model = _model(student=False, seed=1116)
    safe_model.fit(
        source_safe[safe_columns],
        target,
        categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in safe_columns],
    )
    full_model = _model(student=False, seed=1216)
    full_model.fit(
        source_full[full_columns],
        target,
        categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in full_columns],
    )
    delta = full_model.predict(audit_full[full_columns]) - safe_model.predict(
        audit_safe[safe_columns]
    )
    del safe_model, full_model
    gc.collect()
    return delta.astype(np.float64)


def _diagnostics(
    rows: pd.DataFrame,
    target: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    output: dict[str, object] = {
        "gain": bss(target, candidate) - bss(target, incumbent),
        "bss": bss(target, candidate),
        "incumbent_bss": bss(target, incumbent),
        "months": [],
        "domains": [],
    }
    for month, index in rows.groupby("game_month", observed=True).groups.items():
        idx = np.asarray(list(index), dtype=np.int64)
        output["months"].append(
            {
                "month": int(month),
                "n_rows": int(len(idx)),
                "gain": bss(target[idx], candidate[idx])
                - bss(target[idx], incumbent[idx]),
            }
        )
    for domain, index in rows.groupby("game_type", observed=True).groups.items():
        idx = np.asarray(list(index), dtype=np.int64)
        output["domains"].append(
            {
                "domain": str(domain),
                "n_rows": int(len(idx)),
                "gain": bss(target[idx], candidate[idx])
                - bss(target[idx], incumbent[idx]),
            }
        )
    return output


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
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=PHYSICAL_COLUMNS,
        low_memory=False,
    )
    safe_columns = [c for c in main.columns if c not in {"row_id", "control_success"}]
    physical_columns = [f"tm_{column}" for column in PHYSICAL_COLUMNS]
    full_columns = safe_columns + physical_columns
    metric_rows: list[dict[str, object]] = []
    fold_results: list[dict[str, object]] = []
    saved_predictions: dict[int, dict[str, np.ndarray]] = {}
    for source_year, audit_year in TRANSITIONS:
        source_mask = aligned_season == source_year
        audit_mask = aligned_season == audit_year
        source_index = aligned_main_index[source_mask]
        audit_aligned_index = aligned_main_index[audit_mask]
        source = main.iloc[source_index].reset_index(drop=True).copy()
        audit_aligned = main.iloc[audit_aligned_index].reset_index(drop=True).copy()
        audit_all = main.loc[main["season"].eq(audit_year)].reset_index(drop=True).copy()
        source_physical = trackman.iloc[aligned_trackman_index[source_mask]].reset_index(drop=True)
        audit_physical = trackman.iloc[aligned_trackman_index[audit_mask]].reset_index(drop=True)
        for column in PHYSICAL_COLUMNS:
            source[f"tm_{column}"] = source_physical[column].to_numpy()
            audit_aligned[f"tm_{column}"] = audit_physical[column].to_numpy()
            audit_all[f"tm_{column}"] = np.nan
        source_x, audit_aligned_x, audit_all_x = _prepare_frames(
            source, audit_aligned, audit_all, full_columns
        )
        source_target = source["control_success"].to_numpy(np.float64)
        safe_teacher_oof, full_teacher_oof, teacher_summary = _crossfit_teacher_delta(
            source_x,
            source_x,
            source_target,
            aligned_game[source_mask],
            safe_columns,
            full_columns,
            folds,
        )
        teacher_delta = full_teacher_oof - safe_teacher_oof
        oracle_delta = _oracle_audit_delta(
            source_x,
            source_x,
            audit_aligned_x,
            audit_aligned_x,
            source_target,
            safe_columns,
            full_columns,
        )
        artifact = np.load(
            project
            / "artifacts"
            / "v16_multiseason_20260815_02"
            / f"{V17_NAME}_o{audit_year}.npz"
        )
        incumbent = artifact["candidate"].astype(np.float64)
        audit_target = artifact["target"].astype(np.float64)
        if not np.array_equal(audit_target, audit_all["control_success"].to_numpy(np.float64)):
            raise ValueError(f"v17 target order mismatch for {audit_year}")
        audit_global = np.flatnonzero(main["season"].eq(audit_year).to_numpy())
        audit_aligned_local = np.searchsorted(audit_global, audit_aligned_index)
        if not np.array_equal(audit_global[audit_aligned_local], audit_aligned_index):
            raise ValueError("aligned audit index lookup failed")
        oracle_candidate = incumbent.copy()
        oracle_candidate[audit_aligned_local] = np.clip(
            oracle_candidate[audit_aligned_local] + 0.10 * oracle_delta,
            0.001,
            0.999,
        )
        teacher_summary["oracle_w010_gain_vs_v17"] = bss(audit_target, oracle_candidate) - bss(
            audit_target, incumbent
        )

        predictions: dict[str, np.ndarray] = {}
        for name, dropped_ids in (("student_with_ids", False), ("student_without_ids", True)):
            columns = [c for c in safe_columns if not (dropped_ids and c in ID_COLUMNS)]
            student = _model(student=True, seed=1316 if dropped_ids else 1416)
            student.fit(
                source_x[columns],
                teacher_delta,
                categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in columns],
            )
            prediction = student.predict(audit_all_x[columns]).astype(np.float64)
            predictions[name] = prediction
            del student
            gc.collect()
            for weight in (0.25, 0.50, 0.75, 1.00):
                candidate = np.clip(incumbent + weight * prediction, 0.001, 0.999)
                diagnostic = _diagnostics(audit_all, audit_target, incumbent, candidate)
                metric_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "candidate": name,
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
                        "prediction_sd": float(np.std(prediction)),
                        "teacher_delta_correlation": float(
                            np.corrcoef(
                                prediction[audit_aligned_local], oracle_delta
                            )[0, 1]
                        ),
                    }
                )
        # Generalized distillation: train an otherwise identical student on
        # a convex combination of the hard label and the privileged teacher's
        # game-group OOF probability.  The correction is measured against a
        # control student trained on the same aligned rows.
        pfd_columns = safe_columns
        pfd_categories = [c for c in CATEGORICAL_COLUMNS if c in pfd_columns]
        control_student = _model(student=True, seed=1516)
        control_student.fit(
            source_x[pfd_columns],
            source_target,
            categorical_feature=pfd_categories,
        )
        control_prediction = control_student.predict(
            audit_all_x[pfd_columns]
        ).astype(np.float64)
        del control_student
        for label_weight in (0.25, 0.50, 0.75):
            soft_target = (
                (1.0 - label_weight) * source_target
                + label_weight * np.clip(full_teacher_oof, 0.001, 0.999)
            )
            student = _model(student=True, seed=1600 + int(100 * label_weight))
            student.fit(
                source_x[pfd_columns],
                soft_target,
                categorical_feature=pfd_categories,
            )
            prediction = (
                student.predict(audit_all_x[pfd_columns]).astype(np.float64)
                - control_prediction
            )
            name = f"pfd_softlabel_l{int(100 * label_weight):03d}"
            predictions[name] = prediction
            del student
            gc.collect()
            for weight in (0.25, 0.50, 0.75, 1.00):
                candidate = np.clip(incumbent + weight * prediction, 0.001, 0.999)
                diagnostic = _diagnostics(audit_all, audit_target, incumbent, candidate)
                metric_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "candidate": name,
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
                        "prediction_sd": float(np.std(prediction)),
                        "teacher_delta_correlation": float(
                            np.corrcoef(
                                prediction[audit_aligned_local], oracle_delta
                            )[0, 1]
                        ),
                    }
                )
        del control_prediction
        gc.collect()
        saved_predictions[audit_year] = {
            "target": audit_target,
            "v17": incumbent,
            **predictions,
        }
        np.savez_compressed(
            output_dir / f"distillation_o{audit_year}.npz",
            **saved_predictions[audit_year],
        )
        fold_results.append(
            {
                "source_year": source_year,
                "audit_year": audit_year,
                "source_rows": int(len(source)),
                "audit_aligned_rows": int(len(audit_aligned)),
                "audit_all_rows": int(len(audit_all)),
                **teacher_summary,
            }
        )
        del source, audit_aligned, audit_all, source_x, audit_aligned_x, audit_all_x
        del source_physical, audit_physical, teacher_delta, oracle_delta
        del safe_teacher_oof, full_teacher_oof
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
    selected = robust.iloc[0].to_dict()
    result: dict[str, object] = {
        "protocol": "TRACKMAN_PRIVILEGED_GAME_CROSSFIT_DISTILLATION_V1",
        "alignment": json.loads((alignment_dir / "summary.json").read_text(encoding="utf-8")),
        "teacher_folds": int(folds),
        "teacher": fold_results,
        "n_candidates": int(len(robust)),
        "selected": selected,
        "metrics": metrics.to_dict(orient="records"),
        "near_1150_local_gate_passed": bool(
            selected["min_gain"] >= 40.0 and selected["mean_gain"] >= 50.0
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        "# TrackMan privileged-information distillation\n\n"
        f"- Alignment coverage: **{result['alignment']['row_coverage']:.2%}**.\n"
        f"- Cross-fitting: **{folds} game-group folds** in each source season.\n"
        f"- Best student/weight: `{selected['candidate']}` / **{selected['weight']:.2f}**.\n"
        f"- Forward min/mean gain vs v17: **{selected['min_gain']:+.4f} / {selected['mean_gain']:+.4f}**.\n"
        f"- Near-1150 point-estimate gate passed: **{result['near_1150_local_gate_passed']}**.\n"
        "- Current-pitch TrackMan is used only inside source-season teachers; audit predictions are student-only and row-local.\n",
        encoding="utf-8",
    )
    print(json.dumps({"teacher": fold_results, "robust": robust.to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)
    return result


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
        default=Path("artifacts/trackman_distillation_20260816_01"),
    )
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.output_dir, args.folds)


if __name__ == "__main__":
    main()

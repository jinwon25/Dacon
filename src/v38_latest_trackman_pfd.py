"""Latest-season nested TrackMan privileged-distillation pilot above v27.

Current-pitch TrackMan measurements are used only by cross-fitted teachers on
labelled source rows.  Students see official inference-safe columns only.
March-May 2024 fits the teacher/student pair, June-July selects a conservative
student/domain/weight recipe, and August-October is held out for one audit.
The selected student is refitted with three seeds; no audit TrackMan column is
used or populated.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    one_way_cluster_bootstrap,
)
from src.trackman_privileged_distillation import (
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    PHYSICAL_COLUMNS,
    _crossfit_teacher_delta,
    _model,
)
from src.core.axes import _load_axis
from src.core.axes import _early_to_late_2024
from src.core.diagnostics import diagnostics, v27_parent
from src.v37_latest_season_catboost_residual import _attach_v25, apply_correction


VARIANTS = ("with_ids", "without_ids")
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR")
WEIGHTS = (0.10, 0.20, 0.30, 0.40)
REFIT_SEEDS = (1816, 1916, 2016)


def _bootstrap_audit(
    frame: pd.DataFrame,
    target: np.ndarray,
    candidate: np.ndarray,
    parent: np.ndarray,
    *,
    n_resamples: int = 2000,
    block_size: int = 2000,
) -> dict[str, dict[str, float]]:
    return {
        "pitcher": one_way_cluster_bootstrap(
            target,
            candidate,
            parent,
            frame["pitcher_id"],
            n_resamples=n_resamples,
            seed=3816,
        ),
        "batter": one_way_cluster_bootstrap(
            target,
            candidate,
            parent,
            frame["batter_id"],
            n_resamples=n_resamples,
            seed=3916,
        ),
        "pitcher_x_batter": crossed_pigeonhole_bootstrap(
            target,
            candidate,
            parent,
            frame["pitcher_id"],
            frame["batter_id"],
            n_resamples=n_resamples,
            seed=4016,
        ),
        f"block_{block_size}": circular_block_bootstrap(
            target,
            candidate,
            parent,
            block_size=block_size,
            n_resamples=n_resamples,
            seed=4116,
        ),
    }


def student_columns(safe_columns: list[str], variant: str) -> list[str]:
    if variant == "with_ids":
        return list(safe_columns)
    if variant == "without_ids":
        return [column for column in safe_columns if column not in ID_COLUMNS]
    raise ValueError(f"unknown student variant: {variant}")


def _load_2024(path: Path) -> tuple[pd.DataFrame, int]:
    parts = []
    start: int | None = None
    offset = 0
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        mask = chunk["season"].eq(2024).to_numpy()
        if np.any(mask):
            indices = np.flatnonzero(mask)
            if start is None:
                start = offset + int(indices[0])
            parts.append(chunk.loc[mask].copy())
        offset += len(chunk)
    if start is None or not parts:
        raise ValueError("2024 is absent from train.csv")
    frame = pd.concat(parts, ignore_index=True)
    return frame, start


def _prepare_pair(
    source: pd.DataFrame,
    audit: pd.DataFrame,
    safe_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    physical = [f"tm_{column}" for column in PHYSICAL_COLUMNS]
    full_columns = [*safe_columns, *physical]
    audit_local = audit.copy()
    for column in physical:
        audit_local[column] = np.nan
    sizes = (len(source), len(audit_local))
    combined = pd.concat(
        [source[full_columns], audit_local[full_columns]], ignore_index=True
    )
    for column in full_columns:
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
    return (
        combined.iloc[: sizes[0]].reset_index(drop=True),
        combined.iloc[sizes[0] :].reset_index(drop=True),
        full_columns,
    )


def _teacher_delta(
    source_x: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    safe_columns: list[str],
    full_columns: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    safe_oof, full_oof, summary = _crossfit_teacher_delta(
        source_x,
        source_x,
        np.asarray(target, dtype=np.float64),
        np.asarray(groups),
        safe_columns,
        full_columns,
        folds=3,
    )
    return full_oof - safe_oof, summary


def _student_predict(
    source_x: pd.DataFrame,
    audit_x: pd.DataFrame,
    teacher_delta: np.ndarray,
    safe_columns: list[str],
    *,
    variant: str,
    seed: int,
    model_path: Path | None,
) -> np.ndarray:
    columns = student_columns(safe_columns, variant)
    categories = [column for column in CATEGORICAL_COLUMNS if column in columns]
    student = _model(student=True, seed=seed)
    student.fit(
        source_x[columns],
        np.asarray(teacher_delta, dtype=np.float64),
        categorical_feature=categories,
    )
    prediction = student.predict(audit_x[columns]).astype(np.float64)
    if model_path is not None:
        # LightGBM's Windows C API cannot write some non-ASCII paths.  Let
        # Python handle the path after serialising the booster in memory.
        model_path.write_text(student.booster_.model_to_string(), encoding="utf-8")
    del student
    gc.collect()
    return prediction


def _selection_rows(
    frame: pd.DataFrame,
    parent: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for variant, correction in predictions.items():
        for domain in DOMAINS:
            for weight in WEIGHTS:
                candidate, active = apply_correction(
                    frame, correction, eta=weight, domain=domain
                )
                result = diagnostics(frame, parent, candidate, active)
                applied_gain = (
                    min(result["domain_gains"].values())
                    if domain == "ALL"
                    else result["domain_gains"][domain]
                )
                rows.append(
                    {
                        "variant": variant,
                        "domain": domain,
                        "weight": weight,
                        "applied_domain_gain": applied_gain,
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"months", "domain_gains"}
                        },
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = metrics[
        ["gain", "worst_month_gain", "applied_domain_gain"]
    ].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
        & metrics["applied_domain_gain"].gt(0.0)
        & metrics["minimum_domain_gain"].gt(-5.0)
    )
    return metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)


def run(
    project: Path,
    alignment_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = (project / alignment_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw24_unprepared, year_start = _load_2024(project / "data" / "train.csv")
    raw24 = _add_domain_and_pressure(raw24_unprepared)
    del raw24_unprepared
    full_axis = _attach_v25(
        project,
        _load_axis(project, "y2023_to_y2024", raw24),
        "outer_full_2024",
    )
    _, replication = _early_to_late_2024(project, raw24)
    replication = _attach_v25(project, replication, "replication_late_2024")
    full_parent = v27_parent(full_axis)

    with np.load(alignment_dir / "pitch_alignment.npz") as alignment:
        year_mask = alignment["season"].astype(np.int16) == 2024
        main_local = alignment["main_index"].astype(np.int64)[year_mask] - year_start
        trackman_index = alignment["trackman_index"].astype(np.int64)[year_mask]
        game = alignment["main_game_id"].astype(np.int32)[year_mask]
    if np.any(main_local < 0) or np.any(main_local >= len(raw24)):
        raise ValueError("2024 alignment index is outside the local frame")
    aligned = raw24.iloc[main_local].reset_index(drop=True).copy()
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=PHYSICAL_COLUMNS,
        low_memory=False,
    )
    aligned_physical = trackman.iloc[trackman_index].reset_index(drop=True)
    del trackman
    for column in PHYSICAL_COLUMNS:
        aligned[f"tm_{column}"] = aligned_physical[column].to_numpy()
    del aligned_physical
    safe_columns = [
        column
        for column in raw24.columns
        if column not in {"row_id", "control_success", "domain3", "pressure"}
    ]

    aligned_fit_mask = aligned["game_month"].le(5).to_numpy()
    selection_mask = raw24["game_month"].isin((6, 7)).to_numpy()
    source = aligned.loc[aligned_fit_mask].reset_index(drop=True)
    selection_raw = raw24.loc[selection_mask].reset_index(drop=True)
    source_x, selection_x, full_columns = _prepare_pair(
        source, selection_raw, safe_columns
    )
    teacher_delta, selection_teacher = _teacher_delta(
        source_x,
        source["control_success"].to_numpy(np.float64),
        game[aligned_fit_mask],
        safe_columns,
        full_columns,
    )
    selection_predictions = {}
    for variant_index, variant in enumerate(VARIANTS):
        print(f"[v38] selection student={variant}", flush=True)
        selection_predictions[variant] = _student_predict(
            source_x,
            selection_x,
            teacher_delta,
            safe_columns,
            variant=variant,
            seed=1716 + 100 * variant_index,
            model_path=output_dir / f"selection_{variant}.txt",
        )
    selection = full_axis.loc[selection_mask].reset_index(drop=True)
    metrics = _selection_rows(
        selection, full_parent[selection_mask], selection_predictions
    )
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = metrics.loc[metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else metrics.iloc[0]
    recipe = {
        "variant": str(selected["variant"]),
        "domain": str(selected["domain"]),
        "weight": float(selected["weight"]),
    }
    del source_x, selection_x, source, selection_predictions, teacher_delta
    gc.collect()

    aligned_refit_mask = aligned["game_month"].le(7).to_numpy()
    audit_mask = raw24["game_month"].ge(8).to_numpy()
    source = aligned.loc[aligned_refit_mask].reset_index(drop=True)
    audit_raw = raw24.loc[audit_mask].reset_index(drop=True)
    if not np.array_equal(
        audit_raw["control_success"].to_numpy(np.float64),
        replication["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2024 target order mismatch")
    source_x, audit_x, full_columns = _prepare_pair(source, audit_raw, safe_columns)
    teacher_delta, refit_teacher = _teacher_delta(
        source_x,
        source["control_success"].to_numpy(np.float64),
        game[aligned_refit_mask],
        safe_columns,
        full_columns,
    )
    audit_parent = v27_parent(replication)
    seed_rows = []
    corrections = []
    for seed in REFIT_SEEDS:
        print(f"[v38] refit student seed={seed}", flush=True)
        correction = _student_predict(
            source_x,
            audit_x,
            teacher_delta,
            safe_columns,
            variant=recipe["variant"],
            seed=seed,
            model_path=output_dir / f"refit_{recipe['variant']}_s{seed}.txt",
        )
        corrections.append(correction)
        candidate, active = apply_correction(
            replication,
            correction,
            eta=recipe["weight"],
            domain=recipe["domain"],
        )
        result = diagnostics(replication, audit_parent, candidate, active)
        seed_rows.append(
            {
                "seed": seed,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"months", "domain_gains"}
                },
            }
        )
    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_audits.csv", index=False)
    mean_correction = np.mean(np.vstack(corrections), axis=0)
    candidate, active = apply_correction(
        replication,
        mean_correction,
        eta=recipe["weight"],
        domain=recipe["domain"],
    )
    audit_result = diagnostics(replication, audit_parent, candidate, active)
    audit_bootstrap = _bootstrap_audit(
        audit_raw,
        replication["target"].to_numpy(np.float64),
        candidate,
        audit_parent,
    )
    gates = {
        "selection_gate": bool(selected["passes_selection_gate"]),
        "selection_teacher_privileged_gain_positive": selection_teacher[
            "source_privileged_gain"
        ]
        > 0.0,
        "refit_teacher_privileged_gain_positive": refit_teacher[
            "source_privileged_gain"
        ]
        > 0.0,
        "audit_gain_at_least_3": audit_result["gain"] >= 3.0,
        "audit_all_months_positive": audit_result["positive_month_fraction"] == 1.0,
        "audit_worst_month_positive": audit_result["worst_month_gain"] > 0.0,
        "audit_minimum_domain_above_minus_5": audit_result[
            "minimum_domain_gain"
        ]
        > -5.0,
        "all_seed_gains_positive": all(row["gain"] > 0.0 for row in seed_rows),
        "all_bootstrap_p05_positive": all(
            result["p05"] > 0.0 for result in audit_bootstrap.values()
        ),
    }
    summary = {
        "protocol": "V38_LATEST_SEASON_NESTED_TRACKMAN_PFD_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "fit_axis": "aligned 2024 March-May",
        "selection_axis": "all 2024 June-July, student-only",
        "audit_axis": "all 2024 August-October, student-only",
        "aligned_2024_rows": int(len(aligned)),
        "selection_candidate_count": int(len(metrics)),
        "selection_gate_count": int(metrics["passes_selection_gate"].sum()),
        "chosen": recipe,
        "selection": {
            key: selected[key]
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "applied_domain_gain",
            )
        },
        "teacher": {
            "selection": selection_teacher,
            "refit": refit_teacher,
        },
        "seed_audits": seed_rows,
        "audit": audit_result,
        "audit_bootstrap": audit_bootstrap,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "current_pitch_trackman_in_audit_student": False,
        "latest_year_repeated_development_risk": True,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "audit_late_2024.npz",
        target=replication["target"].to_numpy(np.float64),
        v27=audit_parent,
        correction=mean_correction,
        candidate=candidate,
        active=active,
        domain3=replication["domain3"].astype(str).to_numpy(),
        game_month=replication["game_month"].to_numpy(np.int16),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
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
        default=Path("artifacts/v38_latest_trackman_pfd_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.output_dir)


if __name__ == "__main__":
    main()

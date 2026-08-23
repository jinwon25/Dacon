"""Multi-origin TrackMan distillation with inference-safe pitcher profiles.

The current-pitch TrackMan measurements remain privileged teacher inputs.
Unlike earlier PFD students, the profile student also sees a target-free
pitcher arsenal/release summary built strictly from seasons before the row's
forecast origin.  The recipe is selected on a 2022 -> late-2023 transition and
is opened on 2023 -> 2024 only if the profile student wins the selection gate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import (
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    PHYSICAL_COLUMNS,
    _model,
)
from src.core.axes import _load_axis
from src.core.axes import _early_to_late_2024
from src.core.diagnostics import diagnostics, v27_parent
from src.v37_latest_season_catboost_residual import _attach_v25, apply_correction
from src.v38_latest_trackman_pfd import (
    _bootstrap_audit,
    _prepare_pair,
    _selection_rows,
    _teacher_delta,
)


VARIANTS = ("official_without_ids", "profile_without_ids")
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR")
WEIGHTS = (0.10, 0.20, 0.30, 0.40)
AUDIT_SEEDS = (4421, 4521, 4621)


def _load_year(path: Path, year: int) -> tuple[pd.DataFrame, int]:
    """Load one contiguous season and its first global row offset."""
    pieces = []
    start: int | None = None
    offset = 0
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        mask = chunk["season"].eq(year).to_numpy()
        if np.any(mask):
            indices = np.flatnonzero(mask)
            if start is None:
                start = offset + int(indices[0])
            pieces.append(chunk.loc[mask].copy())
        offset += len(chunk)
    if start is None or not pieces:
        raise ValueError(f"season {year} is absent from {path}")
    return pd.concat(pieces, ignore_index=True), start


def profile_columns(profiles: pd.DataFrame) -> list[str]:
    """Return target-free numeric profile columns in stable file order."""
    excluded = {
        "season",
        "pitcher_id",
        "tm_link_confidence_code",
        "tm_link_assignment_rank",
    }
    return [column for column in profiles.columns if column not in excluded]


def attach_profiles(
    rows: pd.DataFrame,
    profiles: pd.DataFrame,
    year: int,
    columns: list[str],
) -> pd.DataFrame:
    """Map one prior-only origin profile to each row without row aggregation."""
    lookup = profiles.loc[
        profiles["season"].eq(year), ["pitcher_id", *columns]
    ]
    if lookup.duplicated("pitcher_id").any():
        raise ValueError(f"duplicate pitcher profiles for origin={year}")
    keys = rows[["pitcher_id"]].copy()
    keys["__row_order"] = np.arange(len(keys), dtype=np.int64)
    merged = keys.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
    merged = merged.sort_values("__row_order", kind="stable")
    if not np.array_equal(
        merged["__row_order"].to_numpy(), np.arange(len(rows), dtype=np.int64)
    ):
        raise ValueError("profile merge changed row order")
    output = merged[columns].apply(pd.to_numeric, errors="coerce")
    return output.astype(np.float32)


def student_columns(safe_columns: list[str], variant: str) -> list[str]:
    base = [column for column in safe_columns if column not in ID_COLUMNS]
    if variant not in VARIANTS:
        raise ValueError(f"unknown student variant: {variant}")
    return base


def _student_predict(
    source_x: pd.DataFrame,
    audit_x: pd.DataFrame,
    teacher_delta: np.ndarray,
    safe_columns: list[str],
    source_profile: pd.DataFrame,
    audit_profile: pd.DataFrame,
    *,
    variant: str,
    seed: int,
    model_path: Path | None = None,
) -> np.ndarray:
    columns = student_columns(safe_columns, variant)
    fit = source_x[columns].copy()
    audit = audit_x[columns].copy()
    if variant == "profile_without_ids":
        if list(source_profile.columns) != list(audit_profile.columns):
            raise ValueError("source/audit profile schema mismatch")
        for column in source_profile.columns:
            fit[column] = source_profile[column].to_numpy(np.float32)
            audit[column] = audit_profile[column].to_numpy(np.float32)
        columns = [*columns, *source_profile.columns.tolist()]
    categories = [column for column in CATEGORICAL_COLUMNS if column in columns]
    model = _model(student=True, seed=seed)
    model.fit(
        fit[columns],
        np.asarray(teacher_delta, dtype=np.float64),
        categorical_feature=categories,
    )
    prediction = model.predict(audit[columns]).astype(np.float64)
    if model_path is not None:
        model_path.write_text(model.booster_.model_to_string(), encoding="utf-8")
    del model, fit, audit
    gc.collect()
    return prediction


def _aligned_source(
    raw: pd.DataFrame,
    year_start: int,
    year: int,
    alignment: dict[str, np.ndarray],
    trackman: pd.DataFrame,
    *,
    month_max: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = alignment["season"] == year
    local_index = alignment["main_index"][mask] - int(year_start)
    trackman_index = alignment["trackman_index"][mask]
    groups = alignment["main_game_id"][mask]
    if np.any(local_index < 0) or np.any(local_index >= len(raw)):
        raise ValueError(f"alignment index outside season={year}")
    source = raw.iloc[local_index].reset_index(drop=True).copy()
    physical = trackman.iloc[trackman_index].reset_index(drop=True)
    for column in PHYSICAL_COLUMNS:
        source[f"tm_{column}"] = physical[column].to_numpy()
    if month_max is not None:
        keep = source["game_month"].le(month_max).to_numpy()
        source = source.loc[keep].reset_index(drop=True)
        groups = groups[keep]
    return source, groups


def _fit_students(
    source: pd.DataFrame,
    audit_raw: pd.DataFrame,
    groups: np.ndarray,
    profiles: pd.DataFrame,
    profile_names: list[str],
    source_year: int,
    audit_year: int,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    safe_columns = [
        column
        for column in audit_raw.columns
        if column not in {"row_id", "control_success", "domain3", "pressure"}
    ]
    source_x, audit_x, full_columns = _prepare_pair(
        source, audit_raw, safe_columns
    )
    teacher_delta, teacher = _teacher_delta(
        source_x,
        source["control_success"].to_numpy(np.float64),
        groups,
        safe_columns,
        full_columns,
    )
    source_profile = attach_profiles(
        source, profiles, source_year, profile_names
    )
    audit_profile = attach_profiles(
        audit_raw, profiles, audit_year, profile_names
    )
    predictions = {}
    for variant in variants:
        ensemble = []
        for seed in seeds:
            print(f"[v44] {prefix} variant={variant} seed={seed}", flush=True)
            ensemble.append(
                _student_predict(
                    source_x,
                    audit_x,
                    teacher_delta,
                    safe_columns,
                    source_profile,
                    audit_profile,
                    variant=variant,
                    seed=seed,
                    model_path=output_dir / f"{prefix}_{variant}_s{seed}.txt",
                )
            )
        predictions[variant] = np.mean(np.vstack(ensemble), axis=0)
    del source_x, audit_x, teacher_delta, source_profile, audit_profile
    gc.collect()
    return predictions, teacher


def _audit_result(
    frame: pd.DataFrame,
    correction: np.ndarray,
    recipe: dict[str, object],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    candidate, active = apply_correction(
        frame,
        correction,
        eta=float(recipe["weight"]),
        domain=str(recipe["domain"]),
    )
    return diagnostics(frame, parent, candidate, active), candidate, active


def run(
    project: Path,
    alignment_dir: Path,
    profiles_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = (project / alignment_dir).resolve()
    profiles_path = (project / profiles_path).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(alignment_dir / "pitch_alignment.npz") as saved:
        alignment = {
            "season": saved["season"].astype(np.int16),
            "main_index": saved["main_index"].astype(np.int64),
            "trackman_index": saved["trackman_index"].astype(np.int64),
            "main_game_id": saved["main_game_id"].astype(np.int32),
        }
    profiles = pd.read_csv(profiles_path, low_memory=False)
    names = profile_columns(profiles)
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=PHYSICAL_COLUMNS,
        low_memory=False,
    )

    raw22, start22 = _load_year(project / "data" / "train.csv", 2022)
    raw22 = _add_domain_and_pressure(raw22)
    source22, groups22 = _aligned_source(
        raw22, start22, 2022, alignment, trackman
    )
    raw23, start23 = _load_year(project / "data" / "train.csv", 2023)
    raw23 = _add_domain_and_pressure(raw23)
    selection_raw = raw23.loc[raw23["game_month"].ge(8)].reset_index(drop=True)
    selection = _attach_v25(
        project,
        _load_axis(project, "y2023_early_to_late", raw23),
        "selection_late_2023",
    )
    if not np.array_equal(
        selection_raw["control_success"].to_numpy(np.float64),
        selection["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 selection alignment failure")
    selection_predictions, teacher22 = _fit_students(
        source22,
        selection_raw,
        groups22,
        profiles,
        names,
        2022,
        2023,
        VARIANTS,
        (4401,),
        output_dir,
        "select22_to_late23",
    )
    selection_metrics = _selection_rows(
        selection, v27_parent(selection), selection_predictions
    )
    selection_metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = selection_metrics.loc[selection_metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else selection_metrics.iloc[0]
    recipe = {
        "variant": str(selected["variant"]),
        "domain": str(selected["domain"]),
        "weight": float(selected["weight"]),
    }
    selection_winner_is_profile = recipe["variant"] == "profile_without_ids"
    selection_gate = bool(selected["passes_selection_gate"])

    summary: dict[str, object] = {
        "protocol": "V44_MULTI_ORIGIN_PROFILE_AUGMENTED_PFD_ABOVE_V27_V1",
        "selection_axis": "aligned 2022 -> late 2023",
        "profile_contract": "origin S uses target-free TrackMan seasons < S",
        "profile_feature_count": len(names),
        "selection_candidate_count": int(len(selection_metrics)),
        "selection_gate_count": int(
            selection_metrics["passes_selection_gate"].sum()
        ),
        "chosen": recipe,
        "selection": {
            key: float(selected[key])
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "applied_domain_gain",
            )
        },
        "teacher_2022": teacher22,
        "selection_winner_is_profile": selection_winner_is_profile,
        "outer_audits_run": False,
        "eligible_for_packaging": False,
    }
    if not (selection_gate and selection_winner_is_profile):
        summary["decision"] = (
            "Stop before 2024: the profile student did not win the predeclared "
            "late-2023 selection gate."
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary

    raw24, start24 = _load_year(project / "data" / "train.csv", 2024)
    raw24 = _add_domain_and_pressure(raw24)
    full24 = _attach_v25(
        project,
        _load_axis(project, "y2023_to_y2024", raw24),
        "outer_full_2024",
    )
    source23, groups23 = _aligned_source(
        raw23, start23, 2023, alignment, trackman
    )
    full_predictions, teacher23 = _fit_students(
        source23,
        raw24,
        groups23,
        profiles,
        names,
        2023,
        2024,
        (recipe["variant"],),
        AUDIT_SEEDS,
        output_dir,
        "audit23_to_full24",
    )
    full_result, full_candidate, full_active = _audit_result(
        full24, full_predictions[recipe["variant"]], recipe
    )

    source24, groups24 = _aligned_source(
        raw24, start24, 2024, alignment, trackman, month_max=7
    )
    _, late24 = _early_to_late_2024(project, raw24)
    late24 = _attach_v25(project, late24, "replication_late_2024")
    late_raw = raw24.loc[raw24["game_month"].ge(8)].reset_index(drop=True)
    late_predictions, teacher24 = _fit_students(
        source24,
        late_raw,
        groups24,
        profiles,
        names,
        2024,
        2024,
        (recipe["variant"],),
        AUDIT_SEEDS,
        output_dir,
        "audit_early24_to_late24",
    )
    late_result, late_candidate, late_active = _audit_result(
        late24, late_predictions[recipe["variant"]], recipe
    )

    full_bootstrap = _bootstrap_audit(
        raw24,
        full24["target"].to_numpy(np.float64),
        full_candidate,
        v27_parent(full24),
        n_resamples=2000,
    )
    late_bootstrap = _bootstrap_audit(
        late_raw,
        late24["target"].to_numpy(np.float64),
        late_candidate,
        v27_parent(late24),
        n_resamples=2000,
    )
    gates = {
        "full_gain_at_least_3": full_result["gain"] >= 3.0,
        "late_gain_at_least_3": late_result["gain"] >= 3.0,
        "full_month_fraction_at_least_075": full_result[
            "positive_month_fraction"
        ]
        >= 0.75,
        "late_all_months_positive": late_result[
            "positive_month_fraction"
        ]
        == 1.0,
        "both_worst_months_positive": min(
            full_result["worst_month_gain"], late_result["worst_month_gain"]
        )
        > 0.0,
        "all_bootstrap_p05_positive": all(
            item["p05"] > 0.0
            for group in (full_bootstrap, late_bootstrap)
            for item in group.values()
        ),
    }
    summary.update(
        {
            "outer_audits_run": True,
            "teacher_2023": teacher23,
            "teacher_early_2024": teacher24,
            "full_2024": full_result,
            "late_2024": late_result,
            "bootstrap_full_2024": full_bootstrap,
            "bootstrap_late_2024": late_bootstrap,
            "gates": gates,
            "eligible_for_packaging": bool(all(gates.values())),
            "row_local_inference": True,
            "current_pitch_trackman_in_audit_student": False,
            "test_aggregate_used": False,
        }
    )
    np.savez_compressed(
        output_dir / "audit_predictions.npz",
        full_target=full24["target"].to_numpy(np.float64),
        full_parent=v27_parent(full24),
        full_candidate=full_candidate,
        full_active=full_active,
        late_target=late24["target"].to_numpy(np.float64),
        late_parent=v27_parent(late24),
        late_candidate=late_candidate,
        late_active=late_active,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
        "--profiles-path",
        type=Path,
        default=Path("artifacts/followup/trackman_pitcher_profiles.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v44_profile_augmented_pfd_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.profiles_path, args.output_dir)


if __name__ == "__main__":
    main()

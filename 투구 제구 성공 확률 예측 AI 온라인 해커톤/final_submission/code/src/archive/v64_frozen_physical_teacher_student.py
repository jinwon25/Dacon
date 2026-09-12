"""Close the untested v44 physical-teacher student route above Public 1158.

v44 froze ``official_without_ids / ALL / weight=0.4`` on a 2022 to
late-2023 transition, where every month and domain improved.  Its original
question was whether adding pitcher profiles helped, so the 2024 audit was
stopped when the official-only student won.  This closure experiment keeps
that exact recipe and opens the previously untouched axes without retuning.

TrackMan measurements are privileged *training teacher* inputs only.  The
student and every audit/test prediction use official row-local columns, omit
player IDs, and never aggregate evaluation rows.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import PHYSICAL_COLUMNS
from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.archive.v38_latest_trackman_pfd import _prepare_pair, _teacher_delta
from src.archive.v44_profile_augmented_pfd import (
    _aligned_source,
    _load_year,
    _student_predict,
)
from src.archive.v63_pitcher_balls_ahead_interaction import load_final_parent


FROZEN_VARIANT = "official_without_ids"
FROZEN_DOMAIN = "ALL"
FROZEN_WEIGHT = 0.40
PREAUDIT_SEED = (4421,)
AUDIT_SEEDS = (4421, 4521, 4621)


def apply_frozen_correction(
    parent: np.ndarray, correction: np.ndarray, weight: float = FROZEN_WEIGHT
) -> np.ndarray:
    """Apply the already-selected all-domain probability-space correction."""

    parent = np.asarray(parent, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if parent.ndim != 1 or parent.shape != correction.shape:
        raise ValueError("parent/correction must be aligned one-dimensional arrays")
    return np.clip(parent + float(weight) * correction, 0.001, 0.999)


def verify_frozen_selection(metrics_path: Path) -> dict[str, float | str]:
    """Read back the exact v44 row that fixed this closure recipe."""

    metrics = pd.read_csv(metrics_path)
    mask = (
        metrics["variant"].astype(str).eq(FROZEN_VARIANT)
        & metrics["domain"].astype(str).eq(FROZEN_DOMAIN)
        & np.isclose(pd.to_numeric(metrics["weight"]), FROZEN_WEIGHT)
    )
    selected = metrics.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"expected one frozen selection row, found {len(selected)}")
    row = selected.iloc[0]
    required_positive = (
        float(row["gain"]) > 0.0,
        float(row["positive_month_fraction"]) == 1.0,
        float(row["worst_month_gain"]) > 0.0,
        float(row["minimum_domain_gain"]) > 0.0,
    )
    if not all(required_positive):
        raise ValueError(f"frozen v44 selection evidence is not positive: {row.to_dict()}")
    return {
        "variant": FROZEN_VARIANT,
        "domain": FROZEN_DOMAIN,
        "weight": FROZEN_WEIGHT,
        "gain_late_2023": float(row["gain"]),
        "positive_month_fraction_late_2023": float(row["positive_month_fraction"]),
        "worst_month_gain_late_2023": float(row["worst_month_gain"]),
        "minimum_domain_gain_late_2023": float(row["minimum_domain_gain"]),
    }


def _fit_seed_predictions(
    source: pd.DataFrame,
    audit_raw: pd.DataFrame,
    groups: np.ndarray,
    seeds: tuple[int, ...],
    output_dir: Path,
    prefix: str,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Fit one teacher once and deterministic ID-free students by seed."""

    safe_columns = [
        column
        for column in audit_raw.columns
        if column not in {"row_id", "control_success", "domain3", "pressure"}
    ]
    source_x, audit_x, full_columns = _prepare_pair(source, audit_raw, safe_columns)
    teacher_delta, teacher = _teacher_delta(
        source_x,
        source["control_success"].to_numpy(np.float64),
        groups,
        safe_columns,
        full_columns,
    )
    empty_source_profile = pd.DataFrame(index=np.arange(len(source_x)))
    empty_audit_profile = pd.DataFrame(index=np.arange(len(audit_x)))
    predictions = []
    for seed in seeds:
        print(f"[v64] {prefix} student seed={seed}", flush=True)
        predictions.append(
            _student_predict(
                source_x,
                audit_x,
                teacher_delta,
                safe_columns,
                empty_source_profile,
                empty_audit_profile,
                variant=FROZEN_VARIANT,
                seed=seed,
                model_path=output_dir / f"{prefix}_{FROZEN_VARIANT}_s{seed}.txt",
            )
        )
    del source_x, audit_x, teacher_delta
    gc.collect()
    return predictions, teacher


def _seed_and_ensemble_audit(
    frame: pd.DataFrame,
    parent: np.ndarray,
    corrections: list[np.ndarray],
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    active = np.ones(len(frame), dtype=bool)
    seed_results = []
    for seed, correction in zip(AUDIT_SEEDS, corrections, strict=False):
        candidate = apply_frozen_correction(parent, correction)
        result = diagnostics(frame, parent, candidate, active)
        seed_results.append(
            {
                "seed": int(seed),
                "gain": float(result["gain"]),
                "positive_month_fraction": float(result["positive_month_fraction"]),
                "worst_month_gain": float(result["worst_month_gain"]),
                "minimum_domain_gain": float(result["minimum_domain_gain"]),
            }
        )
    mean_correction = np.mean(np.vstack(corrections), axis=0)
    ensemble = apply_frozen_correction(parent, mean_correction)
    return diagnostics(frame, parent, ensemble, active), seed_results, mean_correction


def _load_alignment(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {
            "season": saved["season"].astype(np.int16),
            "main_index": saved["main_index"].astype(np.int64),
            "trackman_index": saved["trackman_index"].astype(np.int64),
            "main_game_id": saved["main_game_id"].astype(np.int32),
        }


def _historical_2022_frame(project: Path, raw22: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    meta = _metadata(project, 2022)
    target = raw22["control_success"].to_numpy(np.float64)
    if not np.array_equal(meta["target"], target):
        raise ValueError("2022 historical parent target/order mismatch")
    frame = pd.DataFrame(
        {
            "target": meta["target"],
            "game_month": meta["month"],
            "domain3": meta["domain"].astype(str),
        }
    )
    return frame, meta["parent"]


def run(
    project: Path,
    alignment_dir: Path,
    final_parent_dir: Path,
    selection_metrics: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    final_parent_dir = final_parent_dir.resolve()
    selection_metrics = selection_metrics.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = verify_frozen_selection(selection_metrics)
    alignment = _load_alignment(alignment_dir / "pitch_alignment.npz")
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=PHYSICAL_COLUMNS,
        low_memory=False,
    )
    raw21, start21 = _load_year(project / "data" / "train.csv", 2021)
    raw21 = _add_domain_and_pressure(raw21)
    raw22, start22 = _load_year(project / "data" / "train.csv", 2022)
    raw22 = _add_domain_and_pressure(raw22)
    source21, groups21 = _aligned_source(
        raw21, start21, 2021, alignment, trackman
    )
    preaudit_predictions, teacher21 = _fit_seed_predictions(
        source21,
        raw22,
        groups21,
        PREAUDIT_SEED,
        output_dir,
        "preaudit21_to_full22",
    )
    frame22, parent22 = _historical_2022_frame(project, raw22)
    candidate22 = apply_frozen_correction(parent22, preaudit_predictions[0])
    preaudit22 = diagnostics(
        frame22, parent22, candidate22, np.ones(len(frame22), dtype=bool)
    )
    del source21, groups21, preaudit_predictions, raw21
    gc.collect()

    raw23, start23 = _load_year(project / "data" / "train.csv", 2023)
    raw23 = _add_domain_and_pressure(raw23)
    raw24, start24 = _load_year(project / "data" / "train.csv", 2024)
    raw24 = _add_domain_and_pressure(raw24)
    source23, groups23 = _aligned_source(
        raw23, start23, 2023, alignment, trackman
    )
    full_predictions, teacher23 = _fit_seed_predictions(
        source23,
        raw24,
        groups23,
        AUDIT_SEEDS,
        output_dir,
        "audit23_to_full24",
    )

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    del raw
    full_frame = axes["outer_full_2024"]
    full_parent = load_final_parent(final_parent_dir, "outer_full_2024", full_frame)
    full_result, full_seeds, full_correction = _seed_and_ensemble_audit(
        full_frame, full_parent, full_predictions
    )
    del source23, groups23, full_predictions
    gc.collect()

    source24, groups24 = _aligned_source(
        raw24, start24, 2024, alignment, trackman, month_max=7
    )
    late_raw = raw24.loc[raw24["game_month"].ge(8)].reset_index(drop=True)
    late_predictions, teacher24 = _fit_seed_predictions(
        source24,
        late_raw,
        groups24,
        AUDIT_SEEDS,
        output_dir,
        "audit_early24_to_late24",
    )
    late_frame = axes["replication_late_2024"]
    if not np.array_equal(
        late_raw["control_success"].to_numpy(np.float64),
        late_frame["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2024 target/order mismatch")
    late_parent = load_final_parent(
        final_parent_dir, "replication_late_2024", late_frame
    )
    late_result, late_seeds, late_correction = _seed_and_ensemble_audit(
        late_frame, late_parent, late_predictions
    )

    outer_domains = full_result["domain_gains"]
    gates = {
        "frozen_selection_all_months_domains_positive": True,
        "historical_2022_gain_positive": float(preaudit22["gain"]) > 0.0,
        "historical_2022_month_fraction_at_least_075": float(
            preaudit22["positive_month_fraction"]
        )
        >= 0.75,
        "outer_gain_at_least_5": float(full_result["gain"]) >= 5.0,
        "outer_month_fraction_at_least_075": float(
            full_result["positive_month_fraction"]
        )
        >= 0.75,
        "outer_worst_month_above_minus_5": float(full_result["worst_month_gain"]) > -5.0,
        "outer_all_domains_nonnegative": all(
            float(value) >= 0.0 for value in outer_domains.values()
        ),
        "replication_gain_positive": float(late_result["gain"]) > 0.0,
        "replication_all_months_positive": float(
            late_result["positive_month_fraction"]
        )
        == 1.0,
        "all_outer_seed_gains_positive": all(row["gain"] > 0.0 for row in full_seeds),
        "all_replication_seed_gains_positive": all(
            row["gain"] > 0.0 for row in late_seeds
        ),
    }
    summary = {
        "protocol": "V64_FROZEN_MULTI_ORIGIN_PHYSICAL_TEACHER_STUDENT_V1",
        "parent": "exact OOF analogue of standalone Public 1158.0745556751 on 2024 axes",
        "frozen_selection": selection,
        "historical_2022_parent_note": "historical incumbent used only as a robustness origin",
        "teacher_2021": teacher21,
        "teacher_2023": teacher23,
        "teacher_early_2024": teacher24,
        "historical_full_2022": preaudit22,
        "outer_full_2024": full_result,
        "replication_late_2024": late_result,
        "outer_seed_audits": full_seeds,
        "replication_seed_audits": late_seeds,
        "gates": {name: bool(value) for name, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "student_uses_player_ids": False,
        "current_pitch_trackman_at_inference": False,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "test_distribution_used": False,
        "audit_labels_used_for_recipe_or_weight": False,
    }
    np.savez_compressed(
        output_dir / "audit_predictions.npz",
        full_target=full_frame["target"].to_numpy(np.float64),
        full_parent=full_parent,
        full_correction=full_correction,
        full_candidate=apply_frozen_correction(full_parent, full_correction),
        late_target=late_frame["target"].to_numpy(np.float64),
        late_parent=late_parent,
        late_correction=late_correction,
        late_candidate=apply_frozen_correction(late_parent, late_correction),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--selection-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.project,
        args.alignment_dir,
        args.final_parent_dir,
        args.selection_metrics,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

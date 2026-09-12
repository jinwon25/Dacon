"""Train the frozen 2025 artifacts for the target-1160 v20 overlay."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.trackman_privileged_distillation import (
    CATEGORICAL_COLUMNS,
    PHYSICAL_COLUMNS,
    _crossfit_teacher_delta,
    _model,
)
from src.temporal_stable_conditional import _add_domain_and_pressure


SOURCE_YEAR = 2024
SEPARATOR = "\x1f"
OVERLAY_RECIPES = (
    {
        "name": "pitcher_hand_pressure",
        "columns": ("pitcher_id", "batter_hand", "pressure"),
        "domain": "ALL",
        "alpha": 50.0,
        "weight": 0.07,
    },
    {
        "name": "pitcher_hand_team",
        "columns": ("pitcher_team_id", "batter_hand"),
        "domain": "ALL",
        "alpha": 3200.0,
        "weight": 0.16,
    },
    {
        "name": "count_hands_anchor",
        "columns": (
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
        ),
        "domain": "R_ANCHOR",
        "alpha": 50.0,
        "weight": 0.18,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _save_booster(model: object, path: Path) -> None:
    path.write_text(model.booster_.model_to_string(), encoding="utf-8")


def _key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    pieces = [
        frame[column].astype("string").fillna("__MISSING__") for column in columns
    ]
    output = pieces[0]
    for piece in pieces[1:]:
        output = output + SEPARATOR + piece
    return output


def _fit_eb(train: pd.DataFrame, residual: np.ndarray) -> list[dict[str, object]]:
    recipes = []
    for recipe in OVERLAY_RECIPES:
        domain = str(recipe["domain"])
        mask = np.ones(len(train), dtype=bool)
        if domain != "ALL":
            mask &= train["domain3"].eq(domain).to_numpy()
        columns = tuple(str(value) for value in recipe["columns"])
        fit = pd.DataFrame(
            {
                "key": _key(train.loc[mask].reset_index(drop=True), columns),
                "residual": residual[mask],
            }
        )
        stats = fit.groupby("key", observed=True)["residual"].agg(["sum", "count"])
        effect = stats["sum"] / (stats["count"] + float(recipe["alpha"]))
        recipes.append(
            {
                **recipe,
                "columns": list(columns),
                "n_groups": int(len(effect)),
                "effects": {str(key): float(value) for key, value in effect.items()},
            }
        )
    return recipes


def _prepare_source(source: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = source.loc[:, columns].copy()
    for column in columns:
        if column in CATEGORICAL_COLUMNS:
            output[column] = (
                output[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype("category")
            )
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(
                np.float32
            )
    return output


def _read_season(path: Path, season: int) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    offset = 0
    raw_columns: list[str] | None = None
    for chunk in pd.read_csv(path, low_memory=False, chunksize=150_000):
        if raw_columns is None:
            raw_columns = list(chunk.columns)
        mask = chunk["season"].eq(season).to_numpy()
        if mask.any():
            selected = chunk.loc[mask].copy()
            selected["_global_index"] = offset + np.flatnonzero(mask)
            parts.append(selected)
        offset += len(chunk)
    if raw_columns is None or not parts:
        raise ValueError(f"season {season} is missing from {path}")
    return pd.concat(parts, ignore_index=True), raw_columns


def _read_indexed_rows(
    path: Path, indices: np.ndarray, columns: list[str]
) -> pd.DataFrame:
    order = np.argsort(indices, kind="stable")
    sorted_index = indices[order]
    parts = []
    offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        low_memory=False,
        chunksize=200_000,
    ):
        high = offset + len(chunk)
        left = int(np.searchsorted(sorted_index, offset, side="left"))
        right = int(np.searchsorted(sorted_index, high, side="left"))
        if right > left:
            local = sorted_index[left:right] - offset
            selected = chunk.iloc[local].copy()
            selected["_original_position"] = order[left:right]
            parts.append(selected)
        offset = high
    if not parts:
        raise ValueError(f"no requested rows found in {path}")
    output = pd.concat(parts, ignore_index=True).sort_values(
        "_original_position", kind="stable"
    )
    return output.drop(columns="_original_position").reset_index(drop=True)


def run(
    project: Path,
    alignment_dir: Path,
    joint_oof_path: Path,
    output_dir: Path,
    teacher_folds: int,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = (project / alignment_dir).resolve()
    joint_oof_path = (
        joint_oof_path.resolve()
        if joint_oof_path.is_absolute()
        else (project / joint_oof_path).resolve()
    )
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = project / "data" / "train.csv"
    source_all, raw_columns = _read_season(train_path, SOURCE_YEAR)
    source_all = _add_domain_and_pressure(source_all)
    with np.load(joint_oof_path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v19 = saved["candidate"].astype(np.float64)
    if not np.array_equal(
        target, source_all["control_success"].to_numpy(np.float64)
    ):
        raise ValueError("v19 OOF and 2024 training rows are not aligned")
    eb_recipes = _fit_eb(source_all, target - v19)

    alignment = np.load(alignment_dir / "pitch_alignment.npz")
    aligned_season = alignment["season"].astype(np.int16)
    source_mask = aligned_season == SOURCE_YEAR
    main_index = alignment["main_index"].astype(np.int64)[source_mask]
    trackman_index = alignment["trackman_index"].astype(np.int64)[source_mask]
    source_game = alignment["main_game_id"].astype(np.int32)[source_mask]
    source_position = pd.Index(source_all["_global_index"]).get_indexer(main_index)
    if (source_position < 0).any():
        raise ValueError("aligned 2024 main row is missing from the season slice")
    source = source_all.iloc[source_position].reset_index(drop=True).copy()
    physical = _read_indexed_rows(
        project / "data" / "trackman_history.csv",
        trackman_index,
        PHYSICAL_COLUMNS,
    )
    for column in PHYSICAL_COLUMNS:
        source[f"tm_{column}"] = physical[column].to_numpy()
    del physical
    safe_columns = [
        column
        for column in raw_columns
        if column not in {"row_id", "control_success"}
    ]
    physical_columns = [f"tm_{column}" for column in PHYSICAL_COLUMNS]
    full_columns = [*safe_columns, *physical_columns]
    source_x = _prepare_source(source, full_columns)
    source_target = source["control_success"].to_numpy(np.float64)
    _, full_teacher_oof, teacher_summary = _crossfit_teacher_delta(
        source_x,
        source_x,
        source_target,
        source_game,
        safe_columns,
        full_columns,
        teacher_folds,
    )
    categories = [column for column in CATEGORICAL_COLUMNS if column in safe_columns]
    control = _model(student=True, seed=1516)
    control.fit(
        source_x[safe_columns],
        source_target,
        categorical_feature=categories,
    )
    soft_target = 0.5 * source_target + 0.5 * np.clip(
        full_teacher_oof, 0.001, 0.999
    )
    soft = _model(student=True, seed=1650)
    soft.fit(
        source_x[safe_columns],
        soft_target,
        categorical_feature=categories,
    )
    control_name = "v20_pfd_control.txt"
    soft_name = "v20_pfd_soft_l050.txt"
    _save_booster(control, output_dir / control_name)
    _save_booster(soft, output_dir / soft_name)
    categorical_levels = {
        column: [str(value) for value in source_x[column].cat.categories]
        for column in categories
    }
    iterations = {
        "control": int(control.booster_.current_iteration()),
        "soft": int(soft.booster_.current_iteration()),
    }
    del control, soft, source_x
    gc.collect()

    spec = {
        "candidate": "v20_target1160_forward_overlay",
        "source_year": SOURCE_YEAR,
        "separator": SEPARATOR,
        "anchor_team_id": 13,
        "eb_recipes": eb_recipes,
        "extra_mode": {
            "name": "conditional_mode_lgb_h0.5_pow1.5",
            "weight": 0.045,
        },
        "pfd": {
            "label_weight": 0.5,
            "overlay_weight": 0.26,
            "control_model": control_name,
            "soft_model": soft_name,
            "feature_columns": safe_columns,
            "categorical_columns": categories,
            "categories": categorical_levels,
            "num_iterations": iterations,
            "training_rows": int(len(source)),
            "teacher": teacher_summary,
        },
        "local_evidence": {
            "y2023_to_y2024_gain": 14.251662840875076,
            "y2023_early_to_late_gain": 178.00793365243612,
            "y2024_early_to_late_gain": 20.652933302219225,
            "minimum_gain": 14.251662840875076,
            "public_center_estimate": 1159.75,
        },
        "row_local_inference": True,
        "test_aggregate_used": False,
        "current_pitch_trackman_used_at_inference": False,
    }
    spec_name = "v20_target1160_spec.json"
    (output_dir / spec_name).write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact_paths = [output_dir / spec_name, output_dir / control_name, output_dir / soft_name]
    manifest = {
        "protocol": "V20_TARGET1160_FINAL_TRAIN_2025_V1",
        "train_sha256": _sha256(train_path),
        "trackman_sha256": _sha256(project / "data" / "trackman_history.csv"),
        "joint_oof_sha256": _sha256(joint_oof_path),
        "pitch_alignment_sha256": _sha256(alignment_dir / "pitch_alignment.npz"),
        "source_year": SOURCE_YEAR,
        "source_rows": int(len(source_all)),
        "aligned_source_rows": int(len(source)),
        "teacher_folds": int(teacher_folds),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "reference_model_read_during_fit": False,
        "test_values_read_during_fit": False,
        "private_score_recomputed": False,
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifact_paths
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=Path("reproduction_inputs/v20_pfd"),
    )
    parser.add_argument(
        "--joint-oof",
        type=Path,
        default=Path("reproduction_inputs/v20_pfd/joint_candidate_o2024.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v20_target1160_final_20260816"),
    )
    parser.add_argument("--teacher-folds", type=int, default=3)
    args = parser.parse_args()
    target = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (args.project.resolve() / args.output_dir).resolve()
    )
    if target.exists():
        raise FileExistsError("Use a new output directory")
    run(
        args.project,
        args.alignment_dir,
        args.joint_oof,
        args.output_dir,
        args.teacher_folds,
    )


if __name__ == "__main__":
    main()

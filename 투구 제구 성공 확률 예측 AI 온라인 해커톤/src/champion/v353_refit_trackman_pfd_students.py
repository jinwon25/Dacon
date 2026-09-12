"""Refit the frozen v38 TrackMan-PFD student recipe on all aligned 2024 rows.

The model architecture, no-ID feature set, seeds, R_ANCHOR route, and 0.40
dose are inherited unchanged from v38/v352.  Current-pitch TrackMan fields are
used only inside game-group cross-fitted teachers on labelled 2024 rows.  The
saved students use official inference-safe row fields only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v38_latest_trackman_pfd import (
    REFIT_SEEDS,
    _load_2024,
    _prepare_pair,
    _teacher_delta,
    student_columns,
)
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.trackman_privileged_distillation import (
    CATEGORICAL_COLUMNS,
    PHYSICAL_COLUMNS,
    _model,
)


PROTOCOL = "V353_REFIT_TRACKMAN_PFD_STUDENTS_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(
    project: Path,
    alignment_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw24_unprepared, year_start = _load_2024(project / "data" / "train.csv")
    raw24 = _add_domain_and_pressure(raw24_unprepared)
    del raw24_unprepared
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
    empty_audit = raw24.iloc[:0].reset_index(drop=True)
    source_x, _empty_x, full_columns = _prepare_pair(
        aligned, empty_audit, safe_columns
    )
    teacher_delta, teacher = _teacher_delta(
        source_x,
        aligned["control_success"].to_numpy(np.float64),
        game,
        safe_columns,
        full_columns,
    )
    variant = "without_ids"
    columns = student_columns(safe_columns, variant)
    categories = [column for column in CATEGORICAL_COLUMNS if column in columns]
    model_rows = []
    train_predictions = []
    for seed in REFIT_SEEDS:
        print(f"[v353] fit full-2024 student seed={seed}", flush=True)
        student = _model(student=True, seed=seed)
        student.fit(
            source_x[columns],
            teacher_delta,
            categorical_feature=categories,
        )
        prediction = student.predict(source_x[columns]).astype(np.float64)
        train_predictions.append(prediction)
        path = output_dir / f"refit_without_ids_s{seed}.txt"
        # Binary write preserves LightGBM's tree_sizes offsets on Windows.
        path.write_bytes(student.booster_.model_to_string().encode("utf-8"))
        model_rows.append(
            {
                "seed": int(seed),
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "train_rmse_to_teacher_delta": float(
                    np.sqrt(np.mean(np.square(prediction - teacher_delta)))
                ),
                "train_correlation_to_teacher_delta": float(
                    np.corrcoef(prediction, teacher_delta)[0, 1]
                ),
            }
        )
        del student
        gc.collect()

    ensemble = np.mean(np.vstack(train_predictions), axis=0)
    summary = {
        "protocol": PROTOCOL,
        "status": "refit_complete",
        "recipe": {
            "variant": variant,
            "domain": "R_ANCHOR",
            "weight": 0.40,
            "seeds": list(REFIT_SEEDS),
            "architecture_or_feature_retuned": False,
        },
        "fit": {
            "season": 2024,
            "aligned_rows": int(len(aligned)),
            "games": int(np.unique(game).size),
            "safe_feature_count": int(len(columns)),
            "categorical_features": categories,
            "teacher": teacher,
            "teacher_delta_mean": float(np.mean(teacher_delta)),
            "teacher_delta_sd": float(np.std(teacher_delta)),
            "ensemble_train_rmse_to_teacher_delta": float(
                np.sqrt(np.mean(np.square(ensemble - teacher_delta)))
            ),
            "ensemble_train_correlation_to_teacher_delta": float(
                np.corrcoef(ensemble, teacher_delta)[0, 1]
            ),
        },
        "models": model_rows,
        "restrictions": {
            "official_train_and_trackman_only": True,
            "current_pitch_trackman_at_inference": False,
            "student_uses_player_ids": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_refit": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.output_dir)


if __name__ == "__main__":
    main()

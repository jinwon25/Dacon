"""Retrain the latent pitch-type student with aligned TrackMan labels.

V72 found only 93.3% agreement between training-only ASOF-delta labels and
aligned TrackMan pitch groups.  This experiment keeps the original latent
student architecture, half-life, and conservative 5% blend fixed while
changing only the source label to the target-free structural TrackMan match.

At inference the classifier sees one official row only.  Current-pitch
TrackMan labels and physical values are never used for audit prediction.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.latent_pitch_type_state_model import (
    TYPE_NAMES,
    _predict_each_type,
    _type_model,
    _typed_features,
)
from src.multi_year_state_model import _add_categories, _model, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v30_diverse_covariance_screen import _cached_v25_axes, diagnostics


TYPE_TO_LABEL = {name: index for index, name in enumerate(TYPE_NAMES)}
HALF_LIFE = 0.5
BLEND_WEIGHT = 0.05


def trackman_training_labels(
    row_count: int,
    main_index: np.ndarray,
    pitch_type_group: np.ndarray,
) -> np.ndarray:
    if len(main_index) != len(pitch_type_group):
        raise ValueError("alignment label length mismatch")
    if len(np.unique(main_index)) != len(main_index):
        raise ValueError("aligned main indices are not one-to-one")
    if np.any(main_index < 0) or np.any(main_index >= row_count):
        raise ValueError("aligned main index outside training rows")
    label = np.full(row_count, -1, dtype=np.int8)
    mapped = np.asarray(
        [TYPE_TO_LABEL.get(str(value), -1) for value in pitch_type_group],
        dtype=np.int8,
    )
    label[np.asarray(main_index, dtype=np.int64)] = mapped
    return label


def _fit_origin(
    train: pd.DataFrame,
    features: pd.DataFrame,
    typed_features: pd.DataFrame,
    categorical: list[str],
    typed_categorical: list[str],
    label: np.ndarray,
    baseline: np.ndarray,
    audit_year: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fit_mask = train["season"].lt(audit_year).to_numpy() & (label >= 0)
    audit_mask = train["season"].eq(audit_year).to_numpy()
    fit_season = train.loc[fit_mask, "season"].to_numpy(np.float64)
    sample_weight = np.exp2(-(audit_year - 1.0 - fit_season) / HALF_LIFE)
    sample_weight /= sample_weight.mean()

    classifier = _type_model(seed=7300 + audit_year)
    classifier.fit(
        features.loc[fit_mask],
        label[fit_mask].astype(np.int64),
        sample_weight=sample_weight,
        categorical_feature=categorical,
    )
    probability = classifier.predict_proba(features.loc[audit_mask]).astype(np.float64)
    del classifier
    gc.collect()

    outcome = _model(leaves=15, seed=7400 + audit_year)
    outcome.fit(
        typed_features.loc[fit_mask],
        train.loc[fit_mask, "control_success"].to_numpy(np.float64)
        - baseline[fit_mask],
        sample_weight=sample_weight,
        categorical_feature=typed_categorical,
    )
    raw_by_type = np.clip(
        baseline[audit_mask, None]
        + _predict_each_type(
            outcome, features.loc[audit_mask], typed_categorical
        ),
        0.001,
        0.999,
    )
    del outcome
    gc.collect()
    student_raw = np.sum(probability * raw_by_type, axis=1)
    audit_label = label[audit_mask]
    known = audit_label >= 0
    accuracy = float(
        np.mean(np.argmax(probability[known], axis=1) == audit_label[known])
    )
    logloss = float(
        -np.mean(
            np.log(
                np.clip(
                    probability[np.flatnonzero(known), audit_label[known]],
                    1e-12,
                    1.0,
                )
            )
        )
    )
    audit = {
        "audit_year": audit_year,
        "fit_rows": int(fit_mask.sum()),
        "audit_trackman_label_rows": int(known.sum()),
        "type_accuracy": accuracy,
        "type_logloss": logloss,
    }
    return student_raw, probability, audit


def _blend(
    frame: pd.DataFrame,
    parent: np.ndarray,
    raw: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    candidate = np.clip(
        parent + BLEND_WEIGHT * (np.asarray(raw, dtype=np.float64) - parent),
        0.001,
        0.999,
    )
    result = diagnostics(
        frame, parent, candidate, np.ones(len(frame), dtype=bool)
    )
    result.update(
        {
            "blend_weight": BLEND_WEIGHT,
            "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        }
    )
    return result, candidate


def run(
    project: Path,
    alignment_dir: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=["pitch_type_group"],
        low_memory=False,
    )
    with np.load(alignment_dir / "pitch_alignment.npz", allow_pickle=False) as saved:
        main_index = saved["main_index"].astype(np.int64)
        trackman_index = saved["trackman_index"].astype(np.int64)
    aligned_group = trackman.iloc[trackman_index]["pitch_type_group"].astype(str).to_numpy()
    label = trackman_training_labels(len(train), main_index, aligned_group)
    numeric_state, baseline = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [column for column in features if column.startswith("cat__")]
    typed_features = _typed_features(features, label)
    typed_categorical = [*categorical, "cat__latent_pitch_type"]

    raw_by_year: dict[int, np.ndarray] = {}
    probability_by_year: dict[int, np.ndarray] = {}
    classifier_audits = []
    for year in (2023, 2024):
        print(f"[v73] train TrackMan-label origin={year}", flush=True)
        raw_by_year[year], probability_by_year[year], audit = _fit_origin(
            train,
            features,
            typed_features,
            categorical,
            typed_categorical,
            label,
            baseline,
            year,
        )
        classifier_audits.append(audit)

    axes = _cached_v25_axes(project, train)
    late23 = axes["selection_late_2023"]
    full24 = axes["outer_full_2024"]
    replication24 = axes["replication_late_2024"]
    current23 = np.load(current_oof_dir / "selection_late_2023.npz")
    current24 = np.load(current_oof_dir / "outer_full_2024.npz")
    current_rep = np.load(current_oof_dir / "replication_late_2024.npz")
    month23 = train.loc[train["season"].eq(2023), "game_month"].to_numpy(np.int16)
    month24 = train.loc[train["season"].eq(2024), "game_month"].to_numpy(np.int16)
    late23_mask = month23 >= 8
    late24_mask = month24 >= 8
    axis_specs = {
        "selection_late_2023": (
            late23,
            current23["final_gate_parent"].astype(np.float64),
            raw_by_year[2023][late23_mask],
        ),
        "outer_full_2024": (
            full24,
            current24["final_gate_parent"].astype(np.float64),
            raw_by_year[2024],
        ),
        "replication_late_2024": (
            replication24,
            current_rep["final_gate_parent"].astype(np.float64),
            raw_by_year[2024][late24_mask],
        ),
    }
    audit_results: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, (frame, parent, raw_prediction) in axis_specs.items():
        result, candidate = _blend(frame, parent, raw_prediction)
        audit_results[name] = result
        predictions[name] = candidate
    np.savez_compressed(
        output_dir / "trackman_label_student.npz",
        raw_2023=raw_by_year[2023],
        raw_2024=raw_by_year[2024],
        probability_2023=probability_by_year[2023],
        probability_2024=probability_by_year[2024],
        **predictions,
    )
    gates = {
        "all_axis_gains_positive": all(
            float(result["gain"]) > 0 for result in audit_results.values()
        ),
        "all_month_fractions_at_least_075": all(
            float(result["positive_month_fraction"]) >= 0.75
            for result in audit_results.values()
        ),
        "all_minimum_domains_nonnegative": all(
            float(result["minimum_domain_gain"]) >= 0
            for result in audit_results.values()
        ),
    }
    result = {
        "protocol": "V73_TRACKMAN_LABEL_LATENT_PITCH_TYPE_STUDENT_ABOVE_1158_V1",
        "label_contract": "target-free structural alignment; source seasons < audit year",
        "half_life": HALF_LIFE,
        "blend_weight": BLEND_WEIGHT,
        "classifier_audits": classifier_audits,
        "audits": audit_results,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "decision": "promote" if all(gates.values()) else "reject",
        "row_local_inference": True,
        "current_pitch_trackman_used_at_inference": False,
        "other_test_rows_used": False,
        "test_distribution_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Joint latent pitch-type x failure-mode student above frozen v27.

Pitch type and failure mode are reconstructed only for labelled training rows
from their next cumulative ASOF snapshots.  A safe multiclass student predicts
the 3 x 4 joint state from the current row's inference-safe features.  Audit
probabilities are converted to control-success probabilities with conditional
rates estimated strictly from earlier seasons.

The exact model/temperature/domain/weight recipe must improve both 2022 and
late-2023 before full-2024 is opened.  Neither current-pitch labels nor any
other audit row are used for inference.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.failure_mode_privileged_distillation import (
    MODE_NAMES,
    reconstruct_failure_mode,
)
from src.latent_pitch_type_state_model import (
    TYPE_NAMES,
    reconstruct_current_pitch_type,
)
from src.multi_year_state_model import _add_categories
from src.recent_shared_exact_asof import NUMERIC_COLUMNS, _row_state
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v30_diverse_covariance_screen import (
    _cached_v25_axes,
    diagnostics,
    v27_parent,
)
from src.v35_three_stage_multibank import DOMAINS, _candidate, _metadata, grid_rows
from src.v36_two_origin_consensus import select_consensus


HALF_LIVES = (2.0,)
POWERS = (0.75, 1.0, 1.25)
N_TYPES = len(TYPE_NAMES)
N_MODES = len(MODE_NAMES)
N_CLASSES = N_TYPES * N_MODES


def joint_label(pitch_type: np.ndarray, failure_mode: np.ndarray) -> np.ndarray:
    pitch_type = np.asarray(pitch_type, dtype=np.int16)
    failure_mode = np.asarray(failure_mode, dtype=np.int16)
    if len(pitch_type) != len(failure_mode):
        raise ValueError("latent label arrays have different lengths")
    output = np.full(len(pitch_type), -1, dtype=np.int16)
    known = (pitch_type >= 0) & (failure_mode >= 0)
    output[known] = pitch_type[known] * N_MODES + failure_mode[known]
    return output


def temperature(probability: np.ndarray, power: float) -> np.ndarray:
    output = np.power(np.clip(np.asarray(probability, dtype=np.float64), 1e-9, 1.0), power)
    return output / output.sum(axis=1, keepdims=True)


def _classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        verbosity=-1,
        n_jobs=6,
        n_estimators=120,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=700,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0,
        reg_lambda=18.0,
        max_bin=127,
        random_state=seed,
    )


def safe_feature_frame(train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Use official row fields plus cheap row-local transforms only."""

    numeric = train[
        [column for column in NUMERIC_COLUMNS if column in train]
    ].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    numeric = pd.concat([numeric, _row_state(train)], axis=1).astype(np.float32)
    features = _add_categories(train, numeric)
    categorical = [column for column in features if column.startswith("cat__")]
    return features, categorical


def conditional_rates(
    target: np.ndarray,
    label: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shrunk success rates by mode and by joint state."""

    target = np.asarray(target, dtype=np.float64)
    label = np.asarray(label, dtype=np.int16)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if not (len(target) == len(label) == len(weight)):
        raise ValueError("conditional-rate arrays have different lengths")
    if np.any((label < 0) | (label >= N_CLASSES)):
        raise ValueError("conditional rates require known joint labels")
    mode = label % N_MODES
    global_rate = float(np.average(target, weights=weight))
    mode_rate = np.empty(N_MODES, dtype=np.float64)
    for index in range(N_MODES):
        mask = mode == index
        denominator = float(weight[mask].sum())
        numerator = float(np.sum(weight[mask] * target[mask]))
        mode_rate[index] = (numerator + 200.0 * global_rate) / (
            denominator + 200.0
        )
    joint_rate = np.empty(N_CLASSES, dtype=np.float64)
    for index in range(N_CLASSES):
        mask = label == index
        denominator = float(weight[mask].sum())
        numerator = float(np.sum(weight[mask] * target[mask]))
        prior = mode_rate[index % N_MODES]
        joint_rate[index] = (numerator + 200.0 * prior) / (
            denominator + 200.0
        )
    return mode_rate, joint_rate


def raw_predictions(
    probability: np.ndarray,
    mode_rate: np.ndarray,
    joint_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape[1] != N_CLASSES:
        raise ValueError("unexpected joint probability width")
    mode_probability = probability.reshape(-1, N_TYPES, N_MODES).sum(axis=1)
    return mode_probability @ mode_rate, probability @ joint_rate


def build_prediction_bank(
    train: pd.DataFrame,
    features: pd.DataFrame,
    categorical: list[str],
    label: np.ndarray,
    pitch_type: np.ndarray,
    failure_mode: np.ndarray,
    audit_year: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    fit = train["season"].lt(audit_year).to_numpy() & (label >= 0)
    audit = train["season"].eq(audit_year).to_numpy()
    fit_target = train.loc[fit, "control_success"].to_numpy(np.float64)
    fit_label = label[fit]
    fit_season = train.loc[fit, "season"].to_numpy(np.float64)
    audit_label = label[audit]
    audit_type = pitch_type[audit]
    audit_mode = failure_mode[audit]
    known = audit_label >= 0
    output: dict[str, np.ndarray] = {}
    metrics: list[dict[str, float]] = []
    for half_life in HALF_LIVES:
        weight = np.exp2(-(audit_year - 1.0 - fit_season) / half_life)
        weight /= weight.mean()
        model = _classifier(seed=41100 + 10 * audit_year + int(10 * half_life))
        model.fit(
            features.loc[fit],
            fit_label,
            sample_weight=weight,
            categorical_feature=categorical,
        )
        predicted = model.predict_proba(features.loc[audit]).astype(np.float64)
        probability = np.zeros((int(audit.sum()), N_CLASSES), dtype=np.float64)
        probability[:, model.classes_.astype(int)] = predicted
        del model, predicted
        gc.collect()
        probability = temperature(probability, 1.0)
        mode_rate, joint_rate = conditional_rates(fit_target, fit_label, weight)
        for power in POWERS:
            transformed = temperature(probability, power)
            mode_raw, joint_raw = raw_predictions(
                transformed, mode_rate, joint_rate
            )
            output[f"joint_student::mode_h{half_life:g}_pow{power:g}"] = np.clip(
                mode_raw, 0.001, 0.999
            )
            output[f"joint_student::joint_h{half_life:g}_pow{power:g}"] = np.clip(
                joint_raw, 0.001, 0.999
            )
        metrics.append(
            {
                "audit_year": float(audit_year),
                "half_life": float(half_life),
                "known_fraction": float(known.mean()),
                "joint_accuracy": float(
                    np.mean(np.argmax(probability[known], axis=1) == audit_label[known])
                ),
                "type_accuracy": float(
                    np.mean(
                        np.argmax(
                            probability[known].reshape(-1, N_TYPES, N_MODES).sum(axis=2),
                            axis=1,
                        )
                        == audit_type[known]
                    )
                ),
                "mode_accuracy": float(
                    np.mean(
                        np.argmax(
                            probability[known].reshape(-1, N_TYPES, N_MODES).sum(axis=1),
                            axis=1,
                        )
                        == audit_mode[known]
                    )
                ),
                "joint_logloss": float(
                    -np.mean(
                        np.log(
                            np.clip(
                                probability[np.flatnonzero(known), audit_label[known]],
                                1e-12,
                                1.0,
                            )
                        )
                    )
                ),
            }
        )
    return output, metrics


def _save_year_cache(
    path: Path,
    target: np.ndarray,
    bank: dict[str, np.ndarray],
    metrics: list[dict[str, float]],
) -> None:
    names = list(bank)
    np.savez_compressed(
        path,
        target=np.asarray(target, dtype=np.float64),
        names=np.asarray(names, dtype=object),
        raw=np.column_stack([bank[name] for name in names]),
        metrics_json=np.asarray(json.dumps(metrics, ensure_ascii=False)),
    )


def _load_year_cache(
    path: Path, target: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    with np.load(path, allow_pickle=True) as saved:
        if not np.array_equal(
            saved["target"].astype(np.float64), np.asarray(target, dtype=np.float64)
        ):
            raise ValueError(f"v41 cache target mismatch: {path}")
        names = [str(value) for value in saved["names"].tolist()]
        matrix = saved["raw"].astype(np.float64)
        metrics = json.loads(str(saved["metrics_json"].item()))
    return {name: matrix[:, index] for index, name in enumerate(names)}, metrics


def _selection_grid(
    frame: pd.DataFrame,
    parent: np.ndarray,
    bank: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal, raw in bank.items():
        for domain in DOMAINS:
            rows.extend(
                grid_rows(
                    frame,
                    parent,
                    parent,
                    raw,
                    signal=signal,
                    direction_mode="toward_parent",
                    route=domain,
                )
            )
    return pd.DataFrame(rows)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    pitch_type = reconstruct_current_pitch_type(train)
    failure_mode = reconstruct_failure_mode(train)
    label = joint_label(pitch_type, failure_mode)
    features, categorical = safe_feature_frame(train)

    banks: dict[int, dict[str, np.ndarray]] = {}
    classifier_metrics: list[dict[str, float]] = []
    for year in (2022, 2023, 2024):
        cache_path = output_dir / f"raw_o{year}.npz"
        year_target = train.loc[
            train["season"].eq(year), "control_success"
        ].to_numpy(np.float64)
        if cache_path.exists():
            print(f"[v41] reuse audit_year={year}", flush=True)
            banks[year], local_metrics = _load_year_cache(cache_path, year_target)
        else:
            print(f"[v41] joint student audit_year={year}", flush=True)
            banks[year], local_metrics = build_prediction_bank(
                train,
                features,
                categorical,
                label,
                pitch_type,
                failure_mode,
                year,
            )
            _save_year_cache(cache_path, year_target, banks[year], local_metrics)
        classifier_metrics.extend(local_metrics)
    pd.DataFrame(classifier_metrics).to_csv(
        output_dir / "classifier_metrics.csv", index=False
    )

    meta22 = _metadata(project, 2022)
    frame22 = pd.DataFrame(
        {
            "target": meta22["target"],
            "game_month": meta22["month"],
            "domain3": meta22["domain"],
        }
    )
    stage1 = _selection_grid(frame22, meta22["parent"], banks[2022])
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)

    axes = _cached_v25_axes(project, train)
    selection23 = axes["selection_late_2023"]
    late23 = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
    bank23 = {name: value[late23] for name, value in banks[2023].items()}
    stage2 = _selection_grid(selection23, v27_parent(selection23), bank23)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, selected = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(selected["signal"]),
        "direction": "toward_parent",
        "domain": str(selected["domain"]),
        "weight": float(selected["weight"]),
    }

    full24_month = train.loc[train["season"].eq(2024), "game_month"].to_numpy()
    late24 = full24_month >= 8
    results: dict[str, dict[str, object]] = {}
    for axis_name in ("outer_full_2024", "replication_late_2024"):
        frame = axes[axis_name]
        raw = banks[2024][recipe["signal"]]
        if axis_name == "replication_late_2024":
            raw = raw[late24]
        candidate, active = _candidate(
            frame, {recipe["signal"]: raw}, [recipe]
        )
        parent = v27_parent(frame)
        results[axis_name] = diagnostics(frame, parent, candidate, active)
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=parent,
            raw=raw,
            candidate=candidate,
            active=active,
            game_month=frame["game_month"].to_numpy(np.int16),
            domain3=frame["domain3"].astype(str).to_numpy(),
        )

    gates = {
        "consensus_gate": bool(selected["passes_consensus_gate"]),
        "outer_gain_at_least_5": results["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_nonnegative": results["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": results["replication_late_2024"]["gain"]
        > 0.0,
        "replication_month_fraction_at_least_two_thirds": results[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
        "replication_worst_month_above_minus_10": results[
            "replication_late_2024"
        ]["worst_month_gain"]
        > -10.0,
    }
    summary = {
        "protocol": "V41_JOINT_PITCH_TYPE_FAILURE_MODE_STUDENT_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "joint_label_coverage": float(np.mean(label >= 0)),
        "candidate_count": len(banks[2022]),
        "consensus_recipe_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "chosen": {
            **recipe,
            "gain_2022": float(selected["gain_2022"]),
            "gain_late_2023": float(selected["gain_2023"]),
            "worst_month_2022": float(selected["worst_month_gain_2022"]),
            "worst_month_late_2023": float(
                selected["worst_month_gain_2023"]
            ),
            "consensus_score": float(selected["consensus_score"]),
        },
        "classifier_metrics": classifier_metrics,
        "audits": results,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "current_pitch_type_or_failure_mode_used_at_inference": False,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v41_joint_pitch_failure_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

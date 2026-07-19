from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cross_group_trajectory_smoothing import (
    _current_cross_group_prediction,
    smooth_group_3,
)
from experiments.cross_group_transfer import (
    fit_predict_models,
    selected_prediction,
    transfer_features,
)
from experiments.oof_lineage_audit import extrapolate_submission
from src.feature_cache import load_or_build_features
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group
from train import make_catboost_model, select_feature_columns


TARGET = "kpx_group_3"
VALID_START = pd.Timestamp("2024-01-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")


def apply_calibration(
    prediction: np.ndarray,
    scale: float,
    offset: float,
    strength: float = 1.0,
) -> np.ndarray:
    effective_scale = 1.0 + strength * (scale - 1.0)
    effective_offset = strength * offset
    return np.asarray(prediction, dtype=float) * effective_scale + effective_offset


def _fit_cat_prediction(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    seed: int,
    iterations: int,
) -> np.ndarray:
    model = make_catboost_model(seed, iterations=iterations)
    model.fit(X_train, y_train)
    return model.predict(X_valid)


def _metric_delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": after.score - before.score,
        "one_minus_nmae": after.one_minus_nmae - before.one_minus_nmae,
        "ficr": after.ficr - before.ficr,
    }


def reconstruct(
    data_dir: Path,
    feature_cache_dir: Path,
    blend_report_path: Path,
    hybrid_report_path: Path,
    scada_aligned_cache_path: Path,
    weighted_cache_path: Path,
    pre_cross_test_path: Path,
    final_test_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    X = load_or_build_features(data_dir, "train", feature_cache_dir)
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)
    y = labels[TARGET]
    capacity = CAPACITY_KWH[TARGET]
    columns = select_feature_columns(X, TARGET, "base")
    train_all = (X.index < VALID_START) & y.notna()
    train_eligible = train_all & (y >= 0.10 * capacity)
    valid = (X.index >= VALID_START) & y.notna()
    valid_index = X.index[valid]
    truth = y.loc[valid].to_numpy(dtype=float)

    blend_report = json.loads(blend_report_path.read_text(encoding="utf-8"))
    blend_target = blend_report["targets"][TARGET]
    blend_predictions = []
    blend_weights = []
    blend_components = {}
    component_settings = (
        ("cat_base_all", train_all, 7_303),
        ("cat_base_eligible", train_eligible, 7_304),
    )
    for name, train_mask, seed in component_settings:
        settings = blend_target["candidate_metrics"][name]
        raw = _fit_cat_prediction(
            X.loc[train_mask, columns],
            y.loc[train_mask],
            X.loc[valid, columns],
            seed,
            int(settings["best_iteration"]),
        )
        prediction = np.clip(
            apply_calibration(raw, float(settings["scale"]), float(settings["offset"])),
            0.0,
            capacity,
        )
        weight = float(blend_target["selected_weights"][name])
        blend_predictions.append(prediction)
        blend_weights.append(weight)
        blend_components[name] = {
            "seed": seed,
            "iterations": int(settings["best_iteration"]),
            "weight": weight,
            "metric": evaluate_group(truth, prediction, capacity).to_dict(),
        }
    blend_oof = np.clip(
        np.column_stack(blend_predictions) @ np.asarray(blend_weights), 0.0, capacity
    )

    hybrid_report = json.loads(hybrid_report_path.read_text(encoding="utf-8"))
    hybrid_settings = hybrid_report["targets"][TARGET]
    hybrid_raw = _fit_cat_prediction(
        X.loc[train_eligible, columns],
        y.loc[train_eligible],
        X.loc[valid, columns],
        seed=3_029,
        iterations=int(hybrid_settings["best_iteration"]),
    )
    calibration_oof = np.clip(
        apply_calibration(
            hybrid_raw,
            float(hybrid_settings["scale"]),
            float(hybrid_settings["offset"]),
            strength=1.25,
        ),
        0.0,
        capacity,
    )
    over_oof = extrapolate_submission(calibration_oof, blend_oof, 1.15, capacity)

    scada = np.load(scada_aligned_cache_path, allow_pickle=True)
    scada_index = pd.DatetimeIndex(pd.to_datetime(scada["valid_index_ns"]))
    if not valid_index.equals(scada_index):
        raise ValueError("SCADA and CatBoost validation indices do not match")
    scada_oof = scada["valid_prediction"].astype(float)
    pre_cross_oof = np.clip(0.95 * over_oof + 0.05 * scada_oof, 0.0, capacity)

    weighted = np.load(weighted_cache_path, allow_pickle=True)
    weighted_index = pd.DatetimeIndex(pd.to_datetime(weighted[f"{TARGET}__valid_index_ns"]))
    if not valid_index.equals(weighted_index):
        raise ValueError("Weighted and exact-base validation indices do not match")
    group_1 = selected_prediction(weighted, "kpx_group_1")
    group_2 = selected_prediction(weighted, "kpx_group_2")
    group_1_ratio = group_1 / CAPACITY_KWH["kpx_group_1"]
    group_2_ratio = group_2 / CAPACITY_KWH["kpx_group_2"]
    transfer_train = labels[labels.index.year == 2023].dropna(
        subset=["kpx_group_1", "kpx_group_2", TARGET]
    )
    member = np.clip(
        fit_predict_models(
            transfer_features(
                transfer_train["kpx_group_1"].to_numpy(dtype=float)
                / CAPACITY_KWH["kpx_group_1"],
                transfer_train["kpx_group_2"].to_numpy(dtype=float)
                / CAPACITY_KWH["kpx_group_2"],
                transfer_train.index,
            ),
            transfer_train[TARGET].to_numpy(dtype=float),
            transfer_features(group_1_ratio, group_2_ratio, valid_index),
            seed=51_000,
            mode="base",
        ),
        0.0,
        capacity,
    )
    cross_oof = _current_cross_group_prediction(
        pre_cross_oof, member, group_1_ratio, group_2_ratio
    )
    current_oof, trajectory_mask = smooth_group_3(
        group_1,
        group_2,
        cross_oof,
        valid_index,
        alpha=0.05,
        max_delta_ratio=0.02,
    )

    periods = {
        "h1": valid_index < H2_START,
        "h2": valid_index >= H2_START,
        "full": np.ones(len(valid_index), dtype=bool),
    }
    metrics = {}
    for name, prediction in {
        "blend_v1": blend_oof,
        "cal125": calibration_oof,
        "over115": over_oof,
        "scada_aligned": scada_oof,
        "pre_cross_exact": pre_cross_oof,
        "cross25": cross_oof,
        "submission23_proxy": current_oof,
    }.items():
        metrics[name] = {
            period: evaluate_group(truth[mask], prediction[mask], capacity).to_dict()
            for period, mask in periods.items()
        }

    deltas = {
        "cross25_vs_pre_cross": {
            period: _metric_delta(
                evaluate_group(truth[mask], pre_cross_oof[mask], capacity),
                evaluate_group(truth[mask], cross_oof[mask], capacity),
            )
            for period, mask in periods.items()
        },
        "trajectory_vs_cross25": {
            period: _metric_delta(
                evaluate_group(truth[mask], cross_oof[mask], capacity),
                evaluate_group(truth[mask], current_oof[mask], capacity),
            )
            for period, mask in periods.items()
        },
    }

    pre_cross_test = pd.read_csv(pre_cross_test_path, encoding="utf-8-sig")
    final_test = pd.read_csv(final_test_path, encoding="utf-8-sig")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "exact_group3_oof.npz",
        valid_index_ns=valid_index.astype("int64").to_numpy(),
        valid_truth=truth.astype("float32"),
        blend_v1=blend_oof.astype("float32"),
        calibration125=calibration_oof.astype("float32"),
        over115=over_oof.astype("float32"),
        scada_aligned=scada_oof.astype("float32"),
        pre_cross=pre_cross_oof.astype("float32"),
        cross25=cross_oof.astype("float32"),
        submission23_proxy=current_oof.astype("float32"),
        test_pre_cross=pre_cross_test[TARGET].to_numpy(dtype="float32"),
        test_submission23=final_test[TARGET].to_numpy(dtype="float32"),
    )
    report: dict[str, object] = {
        "validation_rows": int(len(valid_index)),
        "blend_components": blend_components,
        "hybrid_component": {
            "seed": 3_029,
            "iterations": int(hybrid_settings["best_iteration"]),
            "calibration_strength": 1.25,
        },
        "trajectory_rows": int(trajectory_mask.sum()),
        "metrics": metrics,
        "deltas": deltas,
        "driver_note": (
            "Group-3 pre-cross OOF follows the exact test lineage. Cross-group drivers still "
            "use the retained weighted group-1/group-2 OOF forecasts."
        ),
    }
    (output_dir / "exact_group3_oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature-cache-dir", default="artifacts_feature_cache")
    parser.add_argument("--blend-report", default="artifacts_blend/blend_report.json")
    parser.add_argument("--hybrid-report", default="artifacts_hybrid/training_report.json")
    parser.add_argument(
        "--scada-aligned-cache",
        default="artifacts_oof_lineage/scada_group3_aligned_cache.npz",
    )
    parser.add_argument(
        "--weighted-cache", default="artifacts_weighted_metric/prediction_cache.npz"
    )
    parser.add_argument("--pre-cross-test", default="artifacts_cross_group/base_pre_cross.csv")
    parser.add_argument(
        "--final-test",
        default="submissions/archive/blend_best_crossg3_traj5_consensus.csv",
    )
    parser.add_argument("--output-dir", default="artifacts_oof_lineage")
    args = parser.parse_args()
    report = reconstruct(
        Path(args.data_dir),
        Path(args.feature_cache_dir),
        Path(args.blend_report),
        Path(args.hybrid_report),
        Path(args.scada_aligned_cache),
        Path(args.weighted_cache),
        Path(args.pre_cross_test),
        Path(args.final_test),
        Path(args.output_dir),
    )
    print(
        json.dumps(
            {
                "validation_rows": report["validation_rows"],
                "metrics": report["metrics"],
                "deltas": report["deltas"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

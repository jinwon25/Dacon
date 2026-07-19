from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.exact_group3_oof import apply_calibration
from experiments.oof_lineage_audit import fit_affine
from experiments.cross_group_transfer import selected_prediction
from src.feature_cache import load_or_build_features
from src.metrics import CAPACITY_KWH, evaluate_group
from train import make_catboost_model, make_model, select_feature_columns


VALID_START = pd.Timestamp("2024-01-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")


def recover_stack5(
    blend: np.ndarray,
    calibration: np.ndarray,
    stack15: np.ndarray,
    capacity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the historical over115, SCADA member, and stack5 vectors.

    The archived stack15 submission is enough to recover the missing SCADA
    member.  This is preferable to approximating the public lineage with a
    newly trained member.
    """
    blend = np.asarray(blend, dtype=float)
    calibration = np.asarray(calibration, dtype=float)
    stack15 = np.asarray(stack15, dtype=float)
    if blend.shape != calibration.shape or blend.shape != stack15.shape:
        raise ValueError("Historical submission vectors must have the same shape")
    over115 = np.clip(calibration + 1.15 * (blend - calibration), 0.0, capacity)
    scada = (stack15 - 0.85 * over115) / 0.15
    stack5 = np.clip(0.95 * over115 + 0.05 * scada, 0.0, capacity)
    return over115, scada, stack5


def apply_weighted_gate(
    base: np.ndarray,
    member: np.ndarray,
    capacity: float,
    alpha: float,
    max_disagreement: float = 0.04,
    min_base_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base, dtype=float)
    member = np.asarray(member, dtype=float)
    if base.shape != member.shape:
        raise ValueError("base and member must have the same shape")
    mask = (
        (np.abs(member - base) / capacity <= max_disagreement)
        & (base / capacity >= min_base_ratio)
        & (base / capacity <= 1.0)
    )
    output = base.copy()
    if alpha != 0.0:
        output[mask] = (1.0 - alpha) * base[mask] + alpha * member[mask]
    return np.clip(output, 0.0, capacity), mask


def _fit_fixed_prediction(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    seed: int,
    iterations: int,
) -> np.ndarray:
    if family == "lgbm":
        model = make_model(seed, n_estimators=max(100, iterations))
        model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
    elif family == "catboost":
        model = make_catboost_model(seed, iterations=max(100, iterations))
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported historical blend family: {family}")
    return np.asarray(model.predict(X_valid), dtype=float)


def _blend_oof(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    target_i: int,
    capacity: float,
    report: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    valid_time = X.index >= VALID_START
    valid = valid_time & y.notna()
    target_report = report["targets"][target]
    weights = target_report["selected_weights"]
    specs = {spec["name"]: (i, spec) for i, spec in enumerate(report["candidates"], start=1)}
    blended = np.zeros(int(valid.sum()), dtype=float)
    components: dict[str, object] = {}
    for name, weight in weights.items():
        spec_i, spec = specs[name]
        columns = select_feature_columns(X, target, spec["feature_set"])
        train = (~valid_time) & y.notna()
        if spec["train_variant"] == "eligible_only":
            train &= y >= 0.10 * capacity
        settings = target_report["candidate_metrics"][name]
        raw = _fit_fixed_prediction(
            spec["family"],
            X.loc[train, columns],
            y.loc[train],
            X.loc[valid, columns],
            seed=7_000 + 100 * target_i + spec_i,
            iterations=int(settings["best_iteration"]),
        )
        prediction = np.clip(
            apply_calibration(raw, float(settings["scale"]), float(settings["offset"])),
            0.0,
            capacity,
        )
        blended += float(weight) * prediction
        components[name] = {
            "family": spec["family"],
            "train_variant": spec["train_variant"],
            "iterations": int(settings["best_iteration"]),
            "weight": float(weight),
        }
    return np.clip(blended, 0.0, capacity), components


def _calibration_oof(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    target_i: int,
    capacity: float,
    report: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    settings = report["targets"][target]
    family = settings["selected_family"]
    variant = settings["selected_variant"]
    columns = select_feature_columns(X, target, report["feature_set"])
    valid_time = X.index >= VALID_START
    train = (~valid_time) & y.notna()
    if variant == "eligible_only":
        train &= y >= 0.10 * capacity
    valid = valid_time & y.notna()
    raw = _fit_fixed_prediction(
        family,
        X.loc[train, columns],
        y.loc[train],
        X.loc[valid, columns],
        seed=2_026 + target_i + (1_000 if family == "catboost" else 0),
        iterations=int(settings["best_iteration"]),
    )
    prediction = np.clip(
        apply_calibration(
            raw,
            float(settings["scale"]),
            float(settings["offset"]),
            strength=1.25,
        ),
        0.0,
        capacity,
    )
    return prediction, {
        "family": family,
        "train_variant": variant,
        "iterations": int(settings["best_iteration"]),
        "calibration_strength": 1.25,
    }


def _read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _period_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    index: pd.DatetimeIndex,
    capacity: float,
) -> dict[str, object]:
    periods = {
        "h1": index < H2_START,
        "h2": index >= H2_START,
        "full": np.ones(len(index), dtype=bool),
    }
    return {
        name: evaluate_group(truth[mask], prediction[mask], capacity).to_dict()
        for name, mask in periods.items()
    }


def reconstruct(
    data_dir: Path,
    feature_cache_dir: Path,
    blend_report_path: Path,
    hybrid_report_path: Path,
    scada_cache_path: Path,
    weighted_cache_path: Path,
    blend_test_path: Path,
    calibration_test_path: Path,
    stack15_test_path: Path,
    final_base_test_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    X = load_or_build_features(data_dir, "train", feature_cache_dir)
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)
    blend_report = json.loads(blend_report_path.read_text(encoding="utf-8"))
    hybrid_report = json.loads(hybrid_report_path.read_text(encoding="utf-8"))
    scada_cache = np.load(scada_cache_path, allow_pickle=True)
    weighted_cache = np.load(weighted_cache_path, allow_pickle=True)
    blend_test = _read_submission(blend_test_path)
    calibration_test = _read_submission(calibration_test_path)
    stack15_test = _read_submission(stack15_test_path)
    final_base_test = _read_submission(final_base_test_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}
    report: dict[str, object] = {"targets": {}}
    for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        valid = (X.index >= VALID_START) & y.notna()
        valid_index = X.index[valid]
        truth = y.loc[valid].to_numpy(dtype=float)
        blend_oof, blend_components = _blend_oof(
            X, y, target, target_i, capacity, blend_report
        )
        calibration_oof, hybrid_component = _calibration_oof(
            X, y, target, target_i, capacity, hybrid_report
        )
        over_oof = np.clip(
            calibration_oof + 1.15 * (blend_oof - calibration_oof), 0.0, capacity
        )

        historical_over, historical_scada, historical_stack5 = recover_stack5(
            blend_test[target].to_numpy(dtype=float),
            calibration_test[target].to_numpy(dtype=float),
            stack15_test[target].to_numpy(dtype=float),
            capacity,
        )
        rebuilt_test = scada_cache[f"{target}__test_matrix"].astype(float)[:, 0]
        slope, intercept = fit_affine(rebuilt_test, historical_scada)
        rebuilt_index = pd.DatetimeIndex(
            pd.to_datetime(scada_cache[f"{target}__valid_index_ns"])
        )
        if not valid_index.equals(rebuilt_index):
            raise ValueError(f"SCADA validation index mismatch for {target}")
        aligned_scada_oof = np.clip(
            slope * scada_cache[f"{target}__valid_matrix"].astype(float)[:, 0] + intercept,
            0.0,
            capacity,
        )
        stack5_oof = np.clip(0.95 * over_oof + 0.05 * aligned_scada_oof, 0.0, capacity)

        weighted_index = pd.DatetimeIndex(
            pd.to_datetime(weighted_cache[f"{target}__valid_index_ns"])
        )
        if not valid_index.equals(weighted_index):
            raise ValueError(f"Weighted validation index mismatch for {target}")
        weighted_oof = selected_prediction(weighted_cache, target)
        alpha = 0.02 if target in {"kpx_group_1", "kpx_group_2"} else 0.0
        exact_oof, gate = apply_weighted_gate(stack5_oof, weighted_oof, capacity, alpha)

        weighted_test = (
            weighted_cache[f"{target}__test_matrix"].astype(float)
            @ weighted_cache[f"{target}__selected_weights"].astype(float)
        )
        reconstructed_test, test_gate = apply_weighted_gate(
            historical_stack5, weighted_test, capacity, alpha
        )
        archived_test = final_base_test[target].to_numpy(dtype=float)
        test_error = np.abs(reconstructed_test - archived_test)

        cache[f"{target}__valid_index_ns"] = valid_index.astype("int64").to_numpy()
        cache[f"{target}__valid_truth"] = truth.astype("float32")
        cache[f"{target}__blend_v1"] = blend_oof.astype("float32")
        cache[f"{target}__calibration125"] = calibration_oof.astype("float32")
        cache[f"{target}__over115"] = over_oof.astype("float32")
        cache[f"{target}__scada_aligned"] = aligned_scada_oof.astype("float32")
        cache[f"{target}__stack5"] = stack5_oof.astype("float32")
        cache[f"{target}__weighted_member"] = weighted_oof.astype("float32")
        cache[f"{target}__exact_base"] = exact_oof.astype("float32")
        cache[f"{target}__test_exact_base"] = archived_test.astype("float32")
        report["targets"][target] = {
            "blend_components": blend_components,
            "hybrid_component": hybrid_component,
            "scada_alignment": {
                "slope": slope,
                "intercept": intercept,
                "test_correlation": float(np.corrcoef(rebuilt_test, historical_scada)[0, 1]),
            },
            "weighted_gate": {
                "alpha": alpha,
                "validation_rows": int(gate.sum()),
                "test_rows": int(test_gate.sum()),
            },
            "test_parity": {
                "mean_absolute_error_kwh": float(test_error.mean()),
                "max_absolute_error_kwh": float(test_error.max()),
            },
            "metrics": {
                "blend_v1": _period_metrics(truth, blend_oof, valid_index, capacity),
                "calibration125": _period_metrics(
                    truth, calibration_oof, valid_index, capacity
                ),
                "over115": _period_metrics(truth, over_oof, valid_index, capacity),
                "stack5": _period_metrics(truth, stack5_oof, valid_index, capacity),
                "exact_base": _period_metrics(truth, exact_oof, valid_index, capacity),
            },
        }

    np.savez_compressed(output_dir / "exact_driver_oof.npz", **cache)
    (output_dir / "exact_driver_oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature-cache-dir", default="artifacts_final/feature_cache")
    parser.add_argument(
        "--blend-report", default="artifacts_final/lineage_inputs/blend_report.json"
    )
    parser.add_argument(
        "--hybrid-report", default="artifacts_final/lineage_inputs/hybrid_training_report.json"
    )
    parser.add_argument(
        "--scada-cache",
        default="artifacts_final/lineage_inputs/scada_prediction_cache.npz",
    )
    parser.add_argument(
        "--weighted-cache",
        default="artifacts_final/lineage_inputs/weighted_prediction_cache.npz",
    )
    parser.add_argument("--blend-test", default="submissions/archive/blend_v1.csv")
    parser.add_argument(
        "--calibration-test",
        default="submissions/archive/hybrid_lgbm_cat_g3_cal125.csv",
    )
    parser.add_argument(
        "--stack15-test", default="submissions/archive/blend_over115_scada_stack15.csv"
    )
    parser.add_argument(
        "--final-base-test", default="artifacts_final/lineage_inputs/base_pre_cross.csv"
    )
    parser.add_argument("--output-dir", default="artifacts_final/lineage")
    args = parser.parse_args()
    report = reconstruct(
        Path(args.data_dir),
        Path(args.feature_cache_dir),
        Path(args.blend_report),
        Path(args.hybrid_report),
        Path(args.scada_cache),
        Path(args.weighted_cache),
        Path(args.blend_test),
        Path(args.calibration_test),
        Path(args.stack15_test),
        Path(args.final_base_test),
        Path(args.output_dir),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

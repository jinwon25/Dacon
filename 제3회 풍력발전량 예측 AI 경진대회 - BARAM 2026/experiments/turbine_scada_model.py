from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.blend_experiment import _search_weights
from src.features import TIME_COL, TURBINES_BY_GROUP, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition
from train import calibrate, make_model


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    nominal_only: bool
    generation_fraction: float


SPECS = (
    CandidateSpec("turbine_shared_all", False, 0.0),
    CandidateSpec("turbine_shared_nominal", True, 0.0),
    CandidateSpec("turbine_shared_nominal_generation35", True, 0.35),
)

TURBINE_CONFIG = {
    "kpx_group_1": {"prefix": "vestas", "ids": range(1, 7), "shift": 50, "rated": 3600.0, "maker": 0.0},
    "kpx_group_2": {"prefix": "vestas", "ids": range(7, 13), "shift": 50, "rated": 3600.0, "maker": 0.0},
    "kpx_group_3": {"prefix": "unison", "ids": range(1, 6), "shift": 60, "rated": 4200.0, "maker": 1.0},
}

WIND_TOKENS = (
    "__ws",
    "10_10u",
    "10_10v",
    "50_50MU",
    "50_50MV",
    "80_u",
    "80_v",
    "100_100u",
    "100_100v",
    "surface_0_gust",
)
SUPPORT_TOKENS = (
    "heightAboveGround_2_t",
    "heightAboveGround_2_2t",
    "heightAboveGround_2_r",
    "heightAboveGround_2_2r",
    "surface_0_sp",
    "meanSea_0_prmsl",
    "etc_0_blh",
)


def _feature_frame(X: pd.DataFrame, target: str) -> pd.DataFrame:
    calendar = {"hour", "month", "dayofweek", "lead_hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"}
    shared_cols = []
    own_cols = []
    for col in X.columns:
        if col in calendar:
            shared_cols.append(col)
        elif "__grid_" not in col and "__kpx_group_" not in col and (
            any(token in col for token in WIND_TOKENS + SUPPORT_TOKENS)
        ):
            shared_cols.append(col)
        elif f"__{target}__" in col and "hub_" not in col:
            own_cols.append(col)
    shared = X[shared_cols].copy()
    own = X[own_cols].copy()
    own.columns = [col.replace(f"__{target}__", "__own_group__") for col in own.columns]
    return shared.join(own).astype("float32")


def _hourly_turbine(
    df: pd.DataFrame,
    power_col: str,
    ws_col: str,
    shift_minutes: int,
) -> pd.DataFrame:
    power = df[power_col].where((df[power_col] >= 0) & (df[power_col] <= 10_000))
    ws = df[ws_col].where((df[ws_col] >= 0) & (df[ws_col] <= 60))
    timestamp = (df["kst_dtm"] + pd.Timedelta(minutes=shift_minutes)).dt.floor("h")
    tmp = pd.DataFrame({"timestamp": timestamp, "power": power, "ws": ws})
    hourly = tmp.groupby("timestamp").agg(power=("power", "sum"), power_count=("power", "count"), ws=("ws", "mean"), ws_count=("ws", "count"))
    hourly.loc[hourly["power_count"] != 6, "power"] = np.nan
    hourly.loc[hourly["ws_count"] < 4, "ws"] = np.nan
    return hourly[["power", "ws"]].astype("float32")


def build_turbine_targets(data_dir: Path) -> dict[str, list[pd.DataFrame]]:
    raw = {
        "vestas": pd.read_csv(data_dir / "train" / "scada_vestas_train.csv", encoding="utf-8-sig"),
        "unison": pd.read_csv(data_dir / "train" / "scada_unison_train.csv", encoding="utf-8-sig"),
    }
    for frame in raw.values():
        frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"])
    out: dict[str, list[pd.DataFrame]] = {}
    for target, config in TURBINE_CONFIG.items():
        frame = raw[config["prefix"]]
        out[target] = [
            _hourly_turbine(
                frame,
                f"{config['prefix']}_wtg{turbine_id:02d}_power_kw10m",
                f"{config['prefix']}_wtg{turbine_id:02d}_ws",
                int(config["shift"]),
            )
            for turbine_id in config["ids"]
        ]
    return out


def _turbine_frame(
    group_frame: pd.DataFrame,
    target: str,
    local_i: int,
) -> pd.DataFrame:
    config = TURBINE_CONFIG[target]
    lat, lon = TURBINES_BY_GROUP[target][local_i]
    out = group_frame.copy()
    out["target_group_id"] = np.float32(int(target.rsplit("_", 1)[1]))
    out["turbine_local_id"] = np.float32(local_i + 1)
    out["turbine_latitude"] = np.float32(lat)
    out["turbine_longitude"] = np.float32(lon)
    out["manufacturer_id"] = np.float32(config["maker"])
    out["rated_kw"] = np.float32(config["rated"])
    return out


def _stack(
    group_frames: dict[str, pd.DataFrame],
    turbine_targets: dict[str, list[pd.DataFrame]],
    time_mask: np.ndarray,
    spec: CandidateSpec,
) -> tuple[pd.DataFrame, pd.Series]:
    X_parts = []
    y_parts = []
    for target, frames in turbine_targets.items():
        rated = float(TURBINE_CONFIG[target]["rated"])
        for local_i, target_frame in enumerate(frames):
            aligned = target_frame.reindex(group_frames[target].index)
            mask = time_mask & aligned["power"].notna().to_numpy()
            if spec.nominal_only:
                curtailed = (aligned["ws"] >= 5.0) & (aligned["power"] < 0.05 * rated)
                mask &= ~curtailed.fillna(False).to_numpy()
            X_parts.append(_turbine_frame(group_frames[target].loc[mask], target, local_i))
            y_parts.append(pd.Series(aligned["power"].to_numpy()[mask] / rated))
    return pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True)


def _sample_weight(y: pd.Series, fraction: float) -> np.ndarray | None:
    if fraction <= 0:
        return None
    relative = y.to_numpy(dtype=float) / float(y.mean())
    return (1.0 - fraction) + fraction * relative


def _fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    spec: CandidateSpec,
    seed: int,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    model = make_model(seed, n_estimators=n_estimators or 1400)
    callbacks = [lgb.log_evaluation(0)]
    fit_args: dict[str, object] = {"sample_weight": _sample_weight(y_train, spec.generation_fraction)}
    if X_valid is not None and y_valid is not None:
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        fit_args.update({"eval_set": [(X_valid, y_valid)], "eval_metric": "l1"})
    model.fit(X_train, y_train, callbacks=callbacks, **fit_args)
    return model, int(model.best_iteration_ or model.n_estimators)


def _predict_group(
    model: object,
    group_frame: pd.DataFrame,
    target: str,
) -> np.ndarray:
    rated = float(TURBINE_CONFIG[target]["rated"])
    predictions = [
        np.clip(model.predict(_turbine_frame(group_frame, target, local_i)), 0, 1.05) * rated
        for local_i in range(len(TURBINES_BY_GROUP[target]))
    ]
    return np.sum(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_turbine")
    parser.add_argument("--output", default="artifacts_turbine/turbine_member.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--n-iter", type=int, default=15_000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print("Building lean NWP frames and turbine-level SCADA targets...", flush=True)
    X_raw = build_features(data_dir, "train")
    X_test_raw = build_features(data_dir, "test")
    group_frames = {target: _feature_frame(X_raw, target) for target in CAPACITY_KWH}
    test_frames = {target: _feature_frame(X_test_raw, target) for target in CAPACITY_KWH}
    turbine_targets = build_turbine_targets(data_dir)
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X_raw.index)
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    submission = sample.set_index(TIME_COL)
    valid_time = X_raw.index >= pd.Timestamp(args.valid_start)

    report: dict[str, object] = {"valid_start": args.valid_start, "specs": [s.__dict__ for s in SPECS], "targets": {target: {"candidates": {}} for target in CAPACITY_KWH}}
    cache: dict[str, np.ndarray] = {"candidate_names": np.asarray([s.name for s in SPECS]), "test_index_ns": X_test_raw.index.astype("int64").to_numpy()}
    valid_members: dict[str, list[np.ndarray]] = {target: [] for target in CAPACITY_KWH}
    test_members: dict[str, list[np.ndarray]] = {target: [] for target in CAPACITY_KWH}

    for spec_i, spec in enumerate(SPECS, start=1):
        print(f"Training {spec.name}...", flush=True)
        X_train, y_train = _stack(group_frames, turbine_targets, np.asarray(~valid_time), spec)
        X_valid_fit, y_valid_fit = _stack(group_frames, turbine_targets, np.asarray(valid_time), spec)
        X_full, y_full = _stack(group_frames, turbine_targets, np.ones(len(X_raw), dtype=bool), spec)
        seed = 45_000 + spec_i
        model, best_iteration = _fit(X_train, y_train, X_valid_fit, y_valid_fit, spec, seed)
        final_model, _ = _fit(X_full, y_full, None, None, spec, seed, n_estimators=max(100, best_iteration))

        for target, capacity in CAPACITY_KWH.items():
            y = labels[target]
            valid = valid_time & y.notna()
            raw_valid = np.clip(_predict_group(model, group_frames[target].loc[valid], target), 0, capacity)
            scale, offset, metric = calibrate(y.loc[valid].to_numpy(), raw_valid, capacity)
            pred_valid = np.clip(raw_valid * scale + offset, 0, capacity)
            pred_test = np.clip(_predict_group(final_model, test_frames[target], target) * scale + offset, 0, capacity)
            valid_members[target].append(pred_valid)
            test_members[target].append(pred_test)
            report["targets"][target]["candidates"][spec.name] = {"train_rows": len(X_train), "full_rows": len(X_full), "best_iteration": best_iteration, "scale": scale, "offset": offset, "metric": metric}
            print(target, spec.name, metric, flush=True)
        del X_train, y_train, X_valid_fit, y_valid_fit, X_full, y_full, model, final_model

    valid_truth: dict[str, np.ndarray] = {}
    valid_predictions: dict[str, np.ndarray] = {}
    for target_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        valid = valid_time & y.notna()
        truth = y.loc[valid].to_numpy()
        valid_matrix = np.column_stack(valid_members[target])
        test_matrix = np.column_stack(test_members[target])
        weights, metric = _search_weights(valid_matrix, truth, capacity, seed=46_000 + target_i, n_iter=args.n_iter)
        pred_valid = np.clip(valid_matrix @ weights, 0, capacity)
        pred_test = np.clip(test_matrix @ weights, 0, capacity)
        valid_truth[target] = truth
        valid_predictions[target] = pred_valid
        submission[target] = pred_test
        report["targets"][target]["blend_metric"] = metric
        report["targets"][target]["selected_weights"] = {spec.name: float(weight) for spec, weight in zip(SPECS, weights) if weight > 1e-8}
        cache[f"{target}__valid_index_ns"] = X_raw.index[valid].astype("int64").to_numpy()
        cache[f"{target}__valid_truth"] = truth.astype("float32")
        cache[f"{target}__valid_matrix"] = valid_matrix.astype("float32")
        cache[f"{target}__test_matrix"] = test_matrix.astype("float32")
        cache[f"{target}__selected_weights"] = weights.astype("float32")
        print(target, "BLEND", metric, report["targets"][target]["selected_weights"], flush=True)

    report["competition_metric"] = evaluate_competition(valid_truth, valid_predictions)
    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    (artifact_dir / "turbine_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(artifact_dir / "prediction_cache.npz", **cache)
    print(json.dumps(report["competition_metric"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved member to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

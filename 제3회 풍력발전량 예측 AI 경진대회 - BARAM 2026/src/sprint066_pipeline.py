"""Production training/inference for the frozen Sprint 0.66 policies."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH
from src.probabilistic import (
    DEFAULT_QUANTILE_LEVELS,
    enforce_noncrossing,
    metric_aware_bayes_action,
    shrink_action,
)
from src.scada import load_hourly_scada


TARGET_CONFIG = {
    "kpx_group_1": {"generation_lambda": 0.0, "scale": 1.07, "offset": 900.0},
    "kpx_group_2": {"generation_lambda": 0.0, "scale": 1.0, "offset": 0.0},
    "kpx_group_3": {"generation_lambda": 1.0, "scale": 1.0, "offset": 0.0},
}
SCADA_GROUP3_WEIGHT = 0.275
BAYES_GROUP2_ALPHA = 0.50
DIVERSE_GROUP3_WEIGHT = 0.20


def _base_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if "hub_" not in column and "__kpx_group_" not in column]


def _direct_model(seed: int, n_estimators: int, objective: str = "l1", alpha: float | None = None) -> lgb.LGBMRegressor:
    settings: dict[str, Any] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 48,
        "min_child_samples": 45,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if alpha is not None:
        settings["alpha"] = alpha
    return lgb.LGBMRegressor(**settings)


def _site_columns(frame: pd.DataFrame, target: str) -> list[str]:
    signals = ("__ws", "__hub_u", "__hub_v", "__hub_dir", "__u", "__v")
    own = [column for column in frame if target in column and any(signal in column for signal in signals)]
    time_columns = [
        column for column in ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead_hour")
        if column in frame
    ]
    if not own:
        raise ValueError(f"No site-aware NWP columns found for {target}")
    return sorted(set(own + time_columns))


def _site_wind_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="l1",
        n_estimators=250,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=45,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def train_sprint066(
    data_dir: str | Path,
    artifact_dir: str | Path,
    *,
    direct_estimators: int = 350,
    quantile_estimators: int = 300,
    seed: int = 2026,
    include_diverse_member: bool = False,
) -> dict[str, Any]:
    """Fit frozen full-history policies. Test data is never opened here."""

    started = time.perf_counter()
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    X = build_features(data_dir, "train")
    labels = pd.read_csv(
        data_dir / "train" / "train_labels.csv",
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm").reindex(X.index)
    columns = _base_columns(X)
    report: dict[str, Any] = {
        "pipeline": "sprint066",
        "selection_source": "locked time-ordered OOF; no test-distribution selection",
        "direct_estimators": direct_estimators,
        "quantile_estimators": quantile_estimators,
        "feature_count": len(columns),
        "targets": {},
    }

    for group_i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        mask = y.notna().to_numpy() & (y.to_numpy(dtype=float) >= 0.10 * capacity)
        settings = TARGET_CONFIG[target]
        model = _direct_model(seed + group_i, direct_estimators)
        sample_weight = None
        if settings["generation_lambda"] > 0:
            normalized = np.clip(y.loc[mask].to_numpy(dtype=float) / capacity, 0.0, None)
            lam = float(settings["generation_lambda"])
            sample_weight = (1.0 - lam) + lam * normalized
        model.fit(
            X.loc[mask, columns],
            y.loc[mask],
            sample_weight=sample_weight,
            callbacks=[lgb.log_evaluation(0)],
        )
        joblib.dump(model, artifact_dir / f"direct_{target}.joblib")
        report["targets"][target] = {
            "eligible_train_rows": int(mask.sum()),
            **settings,
        }

    # Bayes candidates in OOF used a separate 300-tree raw point model.
    target = "kpx_group_2"
    capacity = CAPACITY_KWH[target]
    y = labels[target]
    eligible = y.notna().to_numpy() & (y.to_numpy(dtype=float) >= 0.10 * capacity)
    bayes_raw = _direct_model(seed + 2_500, quantile_estimators)
    bayes_raw.fit(X.loc[eligible, columns], y.loc[eligible], callbacks=[lgb.log_evaluation(0)])
    joblib.dump(bayes_raw, artifact_dir / "bayes_raw_kpx_group_2.joblib")
    quantile_paths = []
    for quantile_i, level in enumerate(DEFAULT_QUANTILE_LEVELS, start=1):
        model = _direct_model(
            seed + 3_000 + quantile_i,
            quantile_estimators,
            objective="quantile",
            alpha=float(level),
        )
        model.fit(X.loc[eligible, columns], y.loc[eligible], callbacks=[lgb.log_evaluation(0)])
        path = artifact_dir / f"quantile_kpx_group_2_{quantile_i:02d}.joblib"
        joblib.dump(model, path)
        quantile_paths.append(path.name)
    report["bayes"] = {
        "target": target,
        "levels": DEFAULT_QUANTILE_LEVELS.tolist(),
        "alpha": BAYES_GROUP2_ALPHA,
        "mean_eligible_generation": float(y.loc[eligible].mean()),
        "raw_model": "bayes_raw_kpx_group_2.joblib",
        "quantile_models": quantile_paths,
    }

    # Training-only SCADA path: NWP -> site wind -> monotone power curve.
    target = "kpx_group_3"
    capacity = CAPACITY_KWH[target]
    scada = load_hourly_scada(data_dir)[target].reindex(X.index)
    clean = (
        scada["is_clean_for_curve"].eq(True)
        & scada["scada_power_kwh"].notna()
        & scada["scada_wind_speed"].notna()
    )
    if clean.sum() < 1_000:
        raise ValueError(f"Insufficient clean SCADA rows for production fit: {clean.sum()}")
    curve = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    curve.fit(
        scada.loc[clean, "scada_wind_speed"].to_numpy(),
        np.clip(scada.loc[clean, "scada_power_kwh"].to_numpy() / capacity, 0.0, 1.0),
    )
    site_columns = _site_columns(X, target)
    wind_model = _site_wind_model(seed + 4_003)
    wind_model.fit(
        X.loc[clean, site_columns],
        scada.loc[clean, "scada_wind_speed"],
        callbacks=[lgb.log_evaluation(0)],
    )
    joblib.dump(
        {"wind_model": wind_model, "curve": curve, "columns": site_columns},
        artifact_dir / "scada_physical_kpx_group_3.joblib",
    )
    report["scada_physical"] = {
        "target": target,
        "clean_train_rows": int(clean.sum()),
        "site_feature_count": len(site_columns),
        "blend_weight": SCADA_GROUP3_WEIGHT,
        "inference_scada_required": False,
    }
    if include_diverse_member:
        from src.spatiotemporal_member import train_spatiotemporal_member

        report["spatiotemporal_member"] = train_spatiotemporal_member(
            data_dir, artifact_dir / "spatiotemporal",
        )
    report["feature_columns"] = columns
    report["runtime_seconds"] = float(time.perf_counter() - started)
    (artifact_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    if len(frame) != 8_760:
        raise ValueError(f"Submission must contain 8,760 rows, got {len(frame)}")
    if frame.columns.tolist() != sample.columns.tolist():
        raise ValueError("Submission columns/order differ from sample_submission")
    for key in ("forecast_id", TIME_COL):
        if not frame[key].astype(str).equals(sample[key].astype(str)):
            raise ValueError(f"Submission {key} order differs from sample_submission")
    values = frame[list(CAPACITY_KWH)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Submission contains NaN or inf")
    for target, capacity in CAPACITY_KWH.items():
        if not frame[target].between(0.0, capacity, inclusive="both").all():
            raise ValueError(f"{target} prediction lies outside [0, capacity]")


def infer_sprint066(
    data_dir: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    diverse_member: str | Path | None = None,
) -> dict[str, Path]:
    data_dir, artifact_dir, output_dir = Path(data_dir), Path(artifact_dir), Path(output_dir)
    report = json.loads((artifact_dir / "training_report.json").read_text(encoding="utf-8"))
    if report.get("pipeline") != "sprint066":
        raise ValueError("Artifact report is not a sprint066 training run")
    X_all = build_features(data_dir, "test")
    columns = report["feature_columns"]
    X = X_all.reindex(columns=columns)
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample_times = pd.DatetimeIndex(pd.to_datetime(sample[TIME_COL]))
    if not sample_times.equals(X_all.index):
        raise ValueError("Test features do not exactly match sample_submission timestamps")

    safe = sample.copy()
    direct: dict[str, np.ndarray] = {}
    for target, capacity in CAPACITY_KWH.items():
        model = joblib.load(artifact_dir / f"direct_{target}.joblib")
        settings = report["targets"][target]
        raw = np.asarray(model.predict(X), dtype=float)
        direct[target] = np.clip(
            raw * float(settings["scale"]) + float(settings["offset"]), 0.0, capacity,
        )
        safe[target] = direct[target]

    physical = joblib.load(artifact_dir / "scada_physical_kpx_group_3.joblib")
    site_wind = np.clip(
        physical["wind_model"].predict(X_all[physical["columns"]]), 0.0, 60.0,
    )
    capacity = CAPACITY_KWH["kpx_group_3"]
    physical_prediction = np.clip(physical["curve"].predict(site_wind) * capacity, 0.0, capacity)
    safe["kpx_group_3"] = np.clip(
        (1.0 - SCADA_GROUP3_WEIGHT) * direct["kpx_group_3"]
        + SCADA_GROUP3_WEIGHT * physical_prediction,
        0.0,
        capacity,
    )

    bayes = safe.copy()
    bayes_settings = report["bayes"]
    raw_model = joblib.load(artifact_dir / bayes_settings["raw_model"])
    raw_point = np.clip(raw_model.predict(X), 0.0, CAPACITY_KWH["kpx_group_2"])
    quantiles = np.column_stack([
        np.clip(joblib.load(artifact_dir / path).predict(X), 0.0, CAPACITY_KWH["kpx_group_2"])
        for path in bayes_settings["quantile_models"]
    ])
    quantiles = enforce_noncrossing(quantiles)
    levels = np.asarray(bayes_settings["levels"], dtype=float)
    p50 = quantiles[:, int(np.argmin(np.abs(levels - 0.50)))]
    action = metric_aware_bayes_action(
        quantiles,
        capacity=CAPACITY_KWH["kpx_group_2"],
        mean_eligible_generation=float(bayes_settings["mean_eligible_generation"]),
        levels=levels,
        point_candidates=raw_point,
        reference=p50,
    ).action
    bayes["kpx_group_2"] = shrink_action(
        p50, action, float(bayes_settings["alpha"]), CAPACITY_KWH["kpx_group_2"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if diverse_member is None and (artifact_dir / "spatiotemporal" / "manifest.json").exists():
        from src.spatiotemporal_member import infer_spatiotemporal_member

        diverse_member = infer_spatiotemporal_member(
            data_dir,
            artifact_dir / "spatiotemporal",
            output_dir / "spatiotemporal_member.csv",
        )
    candidates = {"safe_cv_best": safe, "ficr_bayes": bayes}
    if diverse_member is not None:
        member = pd.read_csv(diverse_member, encoding="utf-8-sig")
        validate_submission(member, sample)
        diverse = safe.copy()
        diverse["kpx_group_3"] = np.clip(
            (1.0 - DIVERSE_GROUP3_WEIGHT) * safe["kpx_group_3"].to_numpy(dtype=float)
            + DIVERSE_GROUP3_WEIGHT * member["kpx_group_3"].to_numpy(dtype=float),
            0.0,
            CAPACITY_KWH["kpx_group_3"],
        )
        candidates["diverse_ensemble"] = diverse

    paths: dict[str, Path] = {}
    manifest = {"candidates": {}, "test_data_used_for_selection": False}
    for name, frame in candidates.items():
        validate_submission(frame, sample)
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        # Verify serialization/encoding and invariants from the saved bytes.
        with path.open("r", encoding="utf-8-sig") as handle:
            if handle.readline().strip().split(",") != sample.columns.tolist():
                raise ValueError(f"Serialized header is invalid: {path}")
        reloaded = pd.read_csv(path, encoding="utf-8-sig")
        validate_submission(reloaded, sample)
        paths[name] = path
        manifest["candidates"][name] = {
            "path": path.as_posix(),
            "rows": len(frame),
            "min": {target: float(frame[target].min()) for target in CAPACITY_KWH},
            "max": {target: float(frame[target].max()) for target in CAPACITY_KWH},
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return paths

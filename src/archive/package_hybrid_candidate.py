"""Train, package, and preserve the gated hybrid candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.archive.data import TARGET_COL, read_main
from src.archive.features import FeatureBuilder
from src.archive.hybrid_candidate import (
    CANDIDATE_NAME,
    ROLLING_METHODS,
    ROLLING_WEIGHTS,
    TRACKMAN_WEIGHT,
)
from src.package import verify_package
from src.archive.recency_training import season_decay_weights
from src.archive.rolling_drift import build_rolling_damped_ensemble
from src.archive.train import MemoryMonitor, _base_lgb_params, make_official_rf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_zip(package_dir: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def _validation_iterations(project_dir: Path) -> tuple[int, list[int]]:
    values: list[int] = []
    for year in (2021, 2022, 2023, 2024):
        path = (
            project_dir
            / "artifacts"
            / "followup"
            / "models"
            / f"lgb_trackman_pitcher_v1_validate_{year}.npz"
        )
        with np.load(path, allow_pickle=False) as saved:
            values.append(int(saved["best_iteration"][0]))
    return max(1, int(round(float(np.median(values))))), values


def run(project_dir: Path, output_name: str) -> dict[str, object]:
    project_dir = project_dir.resolve()
    gate_path = project_dir / "reports" / "hybrid_candidate_results.csv"
    if not gate_path.exists():
        raise FileNotFoundError("run hybrid_candidate.py before packaging")
    gate = pd.read_csv(gate_path)
    candidate_rows = gate.loc[gate["candidate"] == CANDIDATE_NAME]
    if len(candidate_rows) != 4 or not candidate_rows[
        "passes_statistical_gate"
    ].all():
        raise RuntimeError("hybrid candidate did not pass the fixed statistical gate")

    incumbent_zip = project_dir / "submit.zip"
    output = project_dir / output_name
    candidate_root = (
        project_dir / "artifacts" / "candidates" / CANDIDATE_NAME / output.stem
    )
    package_dir = candidate_root / "package"
    if package_dir.exists() or output.exists():
        raise FileExistsError(
            "candidate output already exists; use a new name to preserve artifacts"
        )
    (package_dir / "model").mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(incumbent_zip) as archive:
        for member in sorted(archive.namelist()):
            if member.startswith("model/") and not member.endswith("/"):
                destination = package_dir / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    shutil.copy2(project_dir / "script.py", package_dir / "script.py")
    shutil.copy2(project_dir / "requirements.txt", package_dir / "requirements.txt")

    train = read_main(project_dir / "data" / "train.csv")
    target = train[TARGET_COL].to_numpy(dtype=np.int8)
    incumbent = json.loads(
        (package_dir / "model" / "ensemble.json").read_text(encoding="utf-8")
    )
    official_features = list(incumbent["official_features"])
    weights = season_decay_weights(train["season"].to_numpy(), half_life=1.0)
    print("[Hybrid package] Training half-life-1 RandomForest...")
    started = time.perf_counter()
    with MemoryMonitor() as rf_memory:
        recency_rf = make_official_rf(official_features, random_state=42)
        recency_rf.fit(
            train[official_features],
            target,
            clf__sample_weight=weights,
        )
    rf_seconds = time.perf_counter() - started
    joblib.dump(
        recency_rf,
        package_dir / "model" / "rf_recency_h1.joblib",
        compress=3,
    )
    del recency_rf, weights
    gc.collect()

    profiles = pd.read_csv(
        project_dir / "artifacts" / "followup" / "trackman_pitcher_profiles.csv"
    )
    if profiles.duplicated(["season", "pitcher_id"]).any():
        raise RuntimeError("Trackman profile keys are not unique")
    config = json.loads(
        (project_dir / "configs" / "trackman_linkage.json").read_text(
            encoding="utf-8"
        )
    )
    rounds, validation_rounds = _validation_iterations(project_dir)
    print(f"[Hybrid package] Training Trackman LightGBM for {rounds} rounds...")
    builder = FeatureBuilder(
        feature_set="trackman_pitcher", trackman_context=profiles
    )
    started = time.perf_counter()
    with MemoryMonitor() as lgb_memory:
        x_train = builder.fit_transform(train, target)
        dataset = lgb.Dataset(
            x_train,
            label=target,
            categorical_feature=builder.categorical_columns,
            free_raw_data=True,
        )
        booster = lgb.train(
            _base_lgb_params(config["model"]),
            dataset,
            num_boost_round=rounds,
            callbacks=[lgb.log_evaluation(100)],
        )
    lgb_seconds = time.perf_counter() - started
    (package_dir / "model" / "trackman_lgb_model.txt").write_text(
        booster.model_to_string(), encoding="utf-8"
    )
    builder.save_spec(package_dir / "model" / "trackman_feature_spec.json")
    profiles.loc[profiles["season"] == 2025].to_csv(
        package_dir / "model" / "trackman_pitcher_profiles.csv",
        index=False,
        encoding="utf-8",
    )
    del booster, dataset, x_train
    gc.collect()

    season_rates = train.groupby("season", observed=True)[TARGET_COL].mean()
    _, drift_details = build_rolling_damped_ensemble(
        np.asarray([0.5], dtype=np.float64),
        season_rates,
        2025,
        ROLLING_METHODS,
        ROLLING_WEIGHTS,
    )
    hybrid = {
        "candidate": CANDIDATE_NAME,
        "recency_rf_model": "rf_recency_h1.joblib",
        "recency_half_life": 1.0,
        "recency_apply_game_type": "R",
        "trackman_weight": TRACKMAN_WEIGHT,
        "trackman_lgb_model": "trackman_lgb_model.txt",
        "trackman_feature_spec": "trackman_feature_spec.json",
        "trackman_num_iterations": rounds,
        "trackman_validation_iterations": validation_rounds,
        "trackman_drift_methods": [item["method"] for item in drift_details],
        "trackman_drift_offsets": [item["offset"] for item in drift_details],
        "trackman_drift_weights": [item["active_weight"] for item in drift_details],
        "forecast_season": 2025,
        "test_aggregate_used": False,
    }
    (package_dir / "model" / "hybrid.json").write_text(
        json.dumps(hybrid, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_zip(package_dir, output)
    verify_package(output)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate": CANDIDATE_NAME,
        "package_path": str(output),
        "package_size_bytes": output.stat().st_size,
        "package_sha256": _sha256(output),
        "incumbent_package_sha256": _sha256(incumbent_zip),
        "local_fold_deltas": {
            str(int(row.outer_validation_season)): float(row.delta_brier)
            for row in candidate_rows.itertuples(index=False)
        },
        "recency_weighted_delta": float(
            candidate_rows["recency_weighted_delta"].iloc[0]
        ),
        "worst_fold_delta": float(candidate_rows["worst_fold_delta"].iloc[0]),
        "combined_improvement_probability": float(
            candidate_rows["combined_improvement_probability"].iloc[0]
        ),
        "passes_statistical_gate": True,
        "rf_training_seconds": rf_seconds,
        "trackman_lgb_training_seconds": lgb_seconds,
        "rf_peak_memory_mb": rf_memory.peak_mb,
        "trackman_lgb_peak_memory_mb": lgb_memory.peak_mb,
        "hybrid": hybrid,
        "selection_caveat": (
            "Outer years are development data after repeated experiments; Public "
            "score remains a deployment probe rather than independent validation."
        ),
    }
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output", default="submit_candidate_hybrid_recency_trackman_v1.zip"
    )
    args = parser.parse_args()
    run(args.project_dir, args.output)


if __name__ == "__main__":
    main()

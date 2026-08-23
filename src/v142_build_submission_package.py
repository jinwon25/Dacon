"""Build and audit the deployable v141 v124/H1/C3 submission package."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.package import verify_package
from src.v84_build_public_probe import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
    run_package,
)
from src.v130_hoo_independent_oof_blend import post4
from src.v133_hoo_h1_c3_forward import CONTEXT_COLS, _pitcher_contrast


PROTOCOL = "V142_V141_SUBMISSION_PACKAGE_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"


def _tables(
    history: pd.DataFrame, residual: np.ndarray, config: dict[str, Any]
) -> dict[str, dict[int, float]]:
    pitcher = history["pitcher_id"].to_numpy(np.int64)
    return {
        "hand": _pitcher_contrast(
            pitcher,
            (history["pitcher_hand"].to_numpy(np.int8)
             == history["batter_hand"].to_numpy(np.int8)).astype(np.int8),
            residual,
            float(config["same_hand_shrink"]),
        ),
        "two": _pitcher_contrast(
            pitcher,
            (history["strikes_before"].to_numpy(np.int8) == 2).astype(np.int8),
            residual,
            float(config["two_strike_shrink"]),
        ),
        "runner": _pitcher_contrast(
            pitcher,
            (history["num_runners_on"].to_numpy(np.int8) > 0).astype(np.int8),
            residual,
            float(config["runner_on_shrink"]),
        ),
    }


def build_c3_bundle(
    train_csv: Path,
    oof_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    context = pd.read_csv(train_csv, usecols=list(CONTEXT_COLS), low_memory=False)
    season = context["season"].to_numpy(np.int16)
    frames = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in range(2019, 2025)
    }
    with np.load(oof_path, allow_pickle=False) as saved:
        h1 = {year: saved[f"h1_{year}"].astype(np.float64) for year in range(2020, 2025)}
    residual = {}
    for year in range(2020, 2025):
        correction = post4(
            context.loc[season < year].reset_index(drop=True), frames[year]
        )
        residual[year] = frames[year][TARGET_COL].to_numpy(np.float64) - (
            h1[year] + correction
        )
    available = list(range(2020, 2025))
    tables = {}
    sources = {}
    for label, width in config["windows"].items():
        selected = available if int(width) >= len(available) else available[-int(width):]
        history = pd.concat([frames[year] for year in selected], ignore_index=True)
        history_residual = np.concatenate([residual[year] for year in selected])
        tables[label] = _tables(history, history_residual, config)
        sources[label] = selected
    del context, frames, h1, residual
    gc.collect()
    return {
        "protocol": PROTOCOL,
        "window_order": list(config["windows"]),
        "window_sources": sources,
        "tables": tables,
        "contrast_scale": float(config["contrast_scale"]),
        "consensus": str(config["consensus"]),
        "external_2025_outcomes_used": False,
        "test_aggregate_used": False,
    }


def _extract_nested(package: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        _safe_extract(archive, destination)


def build_package(
    parent_zip: Path,
    h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    oof_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    _extract_nested(parent_zip, staging / "model" / "v124")
    _extract_nested(h1_zip, staging / "model" / "h1")
    shutil.copy2(runtime_script, staging / "script.py")

    h1_model_path = staging / "model" / "h1" / "model" / "rf.pkl"
    h1_bundle = joblib.load(h1_model_path)
    if len(h1_bundle.get("models", [])) != 3 or len(h1_bundle.get("features", [])) != 82:
        raise ValueError("unexpected H1 bundle structure")
    original_affine = {
        "alpha": float(h1_bundle.get("alpha", 1.0)),
        "center": float(h1_bundle.get("center", 0.5)),
    }
    h1_bundle["alpha"] = 1.0
    h1_bundle["center"] = 0.5
    h1_bundle["note"] = str(h1_bundle.get("note", "")) + " | v142 raw OOF-aligned affine=identity"
    joblib.dump(h1_bundle, h1_model_path, compress=3)

    c3_bundle = build_c3_bundle(train_csv, oof_path, config["c3"])
    joblib.dump(c3_bundle, staging / "model" / "c3_sign_all.joblib", compress=3)
    (staging / "requirements.txt").write_text(
        "lightgbm==4.6.0\ncatboost==1.2.8\njoblib==1.5.1\n"
        "numpy==2.2.6\npandas==2.2.3\nscikit-learn==1.6.1\n",
        encoding="utf-8",
    )
    package = output_dir / str(config["package_name"])
    if package.exists():
        _safe_remove_generated(package, output_dir)
    package_result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    verify_package(package)
    return package, {
        "package": package_result,
        "h1_original_affine": original_affine,
        "h1_packaged_affine": {"alpha": 1.0, "center": 0.5},
        "h1_models": 3,
        "h1_features": 82,
        "c3_window_sources": c3_bundle["window_sources"],
        "c3_table_pitchers": {
            window: {name: len(table) for name, table in parts.items()}
            for window, parts in c3_bundle["tables"].items()
        },
    }


def _audit_rows(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    mixed = pd.concat([source] * 6, ignore_index=True)
    mixed[ID_COL] = [f"v142_audit_{index:04d}" for index in range(len(mixed))]
    route = np.arange(len(mixed)) % 3
    mixed.loc[route == 0, "game_type"] = "R"
    mixed.loc[route == 0, "pitcher_team_id"] = 1
    mixed.loc[route == 0, "batter_team_id"] = 2
    mixed.loc[route == 1, "game_type"] = "R"
    mixed.loc[route == 1, "pitcher_team_id"] = 13
    mixed.loc[route == 2, "game_type"] = "F"
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET_COL: 0.0})
    return mixed, sample


def _aligned(frame: pd.DataFrame, output: pd.DataFrame) -> np.ndarray:
    return frame[ID_COL].map(output.set_index(ID_COL)[TARGET_COL]).to_numpy(np.float64)


def _load_packaged_runtime(package: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        _safe_extract(archive, destination)
    spec = importlib.util.spec_from_file_location("v142_packaged_runtime", destination / "script.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load packaged runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_package(
    package: Path,
    parent_zip: Path,
    data_dir: Path,
    train_csv: Path,
    output_dir: Path,
    timeout: int,
    runtime_limit: float,
    scale_proxy_rows: int,
) -> dict[str, Any]:
    mixed, sample = _audit_rows(data_dir)
    parent_output, parent_seconds, _ = run_package(parent_zip, mixed, sample, timeout=timeout)
    candidate_output, candidate_seconds, smoke_stdout = run_package(
        package, mixed, sample, timeout=timeout
    )
    parent = _aligned(mixed, parent_output)
    candidate = _aligned(mixed, candidate_output)
    runtime_dir = Path(tempfile.mkdtemp(prefix="v142_runtime_"))
    module = _load_packaged_runtime(package, runtime_dir)
    component_parent, h1, c3, active, composed = module.predict_components(mixed)
    formula_max_abs = float(np.max(np.abs(composed - candidate)))
    parent_component_max_abs = float(np.max(np.abs(component_parent - parent)))
    inactive_max_abs = float(np.max(np.abs(candidate[~active] - parent[~active])))
    if max(formula_max_abs, parent_component_max_abs, inactive_max_abs) > 1e-12:
        raise ValueError("v142 formula or inactive-route parity failure")
    if not np.any(np.abs(candidate[active] - parent[active]) > 0.0):
        raise ValueError("v142 active route is prediction-identical")
    c3_probe = pd.read_csv(
        train_csv,
        usecols=[
            "pitcher_id",
            "pitcher_hand",
            "batter_hand",
            "strikes_before",
            "num_runners_on",
        ],
        nrows=200000,
        low_memory=False,
    )
    c3_probe_values = module._predict_c3(c3_probe)
    c3_probe_nonzero = int(np.count_nonzero(np.abs(c3_probe_values) > 0.0))
    if c3_probe_nonzero == 0:
        raise ValueError("v142 C3 sign-consensus path is identically zero on train probe")
    _safe_remove_generated(runtime_dir, runtime_dir.parent)

    shuffled_test = mixed.sample(frac=1.0, random_state=142).reset_index(drop=True)
    shuffled_sample = pd.DataFrame({ID_COL: shuffled_test[ID_COL], TARGET_COL: 0.0})
    shuffled, _, _ = run_package(package, shuffled_test, shuffled_sample, timeout=timeout)
    base_map = candidate_output.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_max_abs = float(np.max(np.abs(base_map - shuffled_map)))

    pieces = []
    for indices in np.array_split(np.arange(len(mixed)), 2):
        local_test = mixed.iloc[indices].reset_index(drop=True)
        local_sample = pd.DataFrame({ID_COL: local_test[ID_COL], TARGET_COL: 0.0})
        output, _, _ = run_package(package, local_test, local_sample, timeout=timeout)
        pieces.append(output)
    partition_map = pd.concat(pieces).set_index(ID_COL)[TARGET_COL].sort_index()
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    if max(shuffled_max_abs, partition_max_abs) > 1e-12:
        raise ValueError("v142 row-order or partition parity failure")

    scale = mixed.iloc[np.arange(scale_proxy_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v142_scale_{index:06d}" for index in range(scale_proxy_rows)]
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(
        package, scale, scale_sample, timeout=timeout
    )
    values = scale_output[TARGET_COL].to_numpy(np.float64)
    if scale_seconds > runtime_limit:
        raise ValueError(f"v142 runtime exceeded: {scale_seconds:.3f} > {runtime_limit:.3f}")
    return {
        "smoke_rows": int(len(mixed)),
        "parent_smoke_seconds": parent_seconds,
        "candidate_smoke_seconds": candidate_seconds,
        "formula_max_abs": formula_max_abs,
        "parent_component_max_abs": parent_component_max_abs,
        "inactive_max_abs": inactive_max_abs,
        "active_fraction": float(active.mean()),
        "active_mean_abs_shift": float(np.mean(np.abs(candidate[active] - parent[active]))),
        "h1_mean": float(h1.mean()),
        "c3_mean_absolute": float(np.mean(np.abs(c3))),
        "c3_probe_rows": int(len(c3_probe)),
        "c3_probe_nonzero_rows": c3_probe_nonzero,
        "c3_probe_mean_absolute": float(np.mean(np.abs(c3_probe_values))),
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "scale_proxy_rows": int(scale_proxy_rows),
        "scale_proxy_seconds": scale_seconds,
        "runtime_limit_seconds": runtime_limit,
        "scale_min": float(values.min()),
        "scale_mean": float(values.mean()),
        "scale_max": float(values.max()),
        "smoke_stdout": smoke_stdout,
        "scale_stdout": scale_stdout,
    }


def run(
    parent_zip: Path,
    h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    oof_path: Path,
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if _sha256(parent_zip) != str(config["parent_v124_sha256"]).upper():
        raise ValueError("v124 parent SHA mismatch")
    if _sha256(h1_zip) != str(config["h1_package_sha256"]).upper():
        raise ValueError("H1 package SHA mismatch")
    package, build = build_package(
        parent_zip, h1_zip, runtime_script, train_csv, oof_path, config, output_dir
    )
    audit = audit_package(
        package,
        parent_zip,
        data_dir,
        train_csv,
        output_dir,
        timeout,
        float(config["runtime_limit_seconds"]),
        int(config["scale_proxy_rows"]),
    )
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_api_submission",
        "formula": "0.85*v124 + 0.15*H1_post4 + 0.5*sign_all_C3 on R_CORE",
        "package": {
            "path": str(package),
            "bytes": package.stat().st_size,
            "sha256": _sha256(package),
        },
        "build": build,
        "audit": audit,
        "eligible_for_api_submission": True,
        **config["restrictions"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--h1-zip", type=Path, required=True)
    parser.add_argument("--runtime-script", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    run(
        args.parent_zip,
        args.h1_zip,
        args.runtime_script,
        args.train_csv,
        args.oof_path,
        args.data_dir,
        args.config,
        args.output_dir,
        args.timeout,
    )


if __name__ == "__main__":
    main()

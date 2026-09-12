"""Build and audit the deployable v180 signed-stack package."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.champion.v130_catboost_independent_oof_blend import post4
from src.champion.v133_catboost_h1_c3_forward import CONTEXT_COLS
from src.champion.v142_build_submission_package import _tables
from src.core.packaging import _safe_extract, _safe_remove_generated, _sha256, _zip_directory, run_package
from src.package import verify_package


PROTOCOL = "V180_SIGNED_STACK_SUBMISSION_PACKAGE_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"
C3_CONFIG = {
    "windows": {"last1": 1, "last2": 2, "last3": 3, "expanding": 99},
    "same_hand_shrink": 1000.0,
    "two_strike_shrink": 1000.0,
    "runner_on_shrink": 2000.0,
    "contrast_scale": 0.5,
    "consensus": "0.85_sign_all_plus_0.15_mean_recent",
}


def _model_recipe(model: Any) -> dict[str, Any]:
    params = model.named_steps["clf"].get_params()
    return {
        "iterations": int(params["iterations"]),
        "learning_rate": float(params["learning_rate"]),
        "depth": int(params["depth"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "border_count": int(params["border_count"]),
        "seed": int(params["random_seed"]),
    }


def _extract(package: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        _safe_extract(archive, destination)


def _proxy_h1_model(package: Path, deployed_bundle: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    with zipfile.ZipFile(package) as archive:
        source = joblib.load(io.BytesIO(archive.read("model/rf.pkl")))
    if list(source.get("features", [])) != list(deployed_bundle.get("features", [])):
        raise ValueError("proxy/deployed H1 feature order mismatch")
    models = list(source.get("models", []))
    if len(models) != 3:
        raise ValueError("expected three public depth-6 H1 models")
    recipe = _model_recipe(models[0])
    expected = {
        "iterations": 1200, "learning_rate": 0.02, "depth": 6,
        "l2_leaf_reg": 100.0, "border_count": 32, "seed": 42,
    }
    if recipe != expected:
        raise ValueError(f"unexpected proxy H1 recipe: {recipe}")
    return models[0], {"recipe": recipe, "feature_count": len(source["features"])}


def _exact_c3_bundle(train_csv: Path, exact_oof: Path) -> dict[str, Any]:
    context = pd.read_csv(train_csv, usecols=list(CONTEXT_COLS), low_memory=False)
    season = context["season"].to_numpy(np.int16)
    frames = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in range(2019, 2025)
    }
    with np.load(exact_oof, allow_pickle=False) as saved:
        h1 = {
            year: saved[f"exact_h1_{year}"].astype(np.float64)
            for year in range(2020, 2025)
        }
    residual = {
        year: frames[year][TARGET_COL].to_numpy(np.float64)
        - (
            h1[year]
            + post4(context.loc[season < year].reset_index(drop=True), frames[year])
        )
        for year in range(2020, 2025)
    }
    available = list(range(2020, 2025))
    tables: dict[str, Any] = {}
    sources: dict[str, list[int]] = {}
    for label, width in C3_CONFIG["windows"].items():
        selected = available if int(width) >= len(available) else available[-int(width):]
        history = pd.concat([frames[year] for year in selected], ignore_index=True)
        history_residual = np.concatenate([residual[year] for year in selected])
        tables[label] = _tables(history, history_residual, C3_CONFIG)
        sources[label] = selected
    del context, frames, h1, residual
    gc.collect()
    return {
        "protocol": PROTOCOL,
        "basis": "exact depth-8 three-seed H1 strict-forward OOF residuals",
        "window_order": list(C3_CONFIG["windows"]),
        "window_sources": sources,
        "tables": tables,
        "contrast_scale": C3_CONFIG["contrast_scale"],
        "official_train_only": True,
        "test_aggregate_used": False,
    }


def build_package(
    champion_zip: Path,
    proxy_h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    exact_oof: Path,
    component_dir: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    _extract(champion_zip, staging)
    shutil.copy2(runtime_script, staging / "script.py")

    deployed_path = staging / "model" / "h1" / "model" / "rf.pkl"
    deployed = joblib.load(deployed_path)
    proxy_model, proxy_metadata = _proxy_h1_model(proxy_h1_zip, deployed)
    joblib.dump(proxy_model, staging / "model" / "proxy_h1_seed42.joblib", compress=3)
    exact_c3 = _exact_c3_bundle(train_csv, exact_oof)
    joblib.dump(exact_c3, staging / "model" / "c3_exact.joblib", compress=3)

    required_component = (
        "config.json", "lgbm_a_42.txt", "cat_bayes.cbm",
        "platoon_2025.csv", "count_2025.csv",
    )
    target_component = staging / "model" / "component"
    target_component.mkdir(parents=True)
    for name in required_component:
        source = component_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target_component / name)

    package = output_dir / "submit_v180_signed_stack.zip"
    if package.exists():
        _safe_remove_generated(package, output_dir)
    package_result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    verify_package(package)
    return package, {
        "package": package_result,
        "proxy_h1": proxy_metadata,
        "exact_c3_window_sources": exact_c3["window_sources"],
        "component_config_sha256": _sha256(component_dir / "config.json"),
    }


def _aligned(frame: pd.DataFrame, output: pd.DataFrame) -> np.ndarray:
    return frame[ID_COL].map(output.set_index(ID_COL)[TARGET_COL]).to_numpy(np.float64)


def _load_runtime(package: Path, destination: Path):
    _extract(package, destination)
    spec = importlib.util.spec_from_file_location("v180_packaged_runtime", destination / "script.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load v180 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_package(
    package: Path,
    champion_zip: Path,
    data_dir: Path,
    *,
    timeout: int,
) -> dict[str, Any]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    frame = pd.concat([source] * 6, ignore_index=True)
    frame[ID_COL] = [f"v180_audit_{index:04d}" for index in range(len(frame))]
    sample = pd.DataFrame({ID_COL: frame[ID_COL], TARGET_COL: 0.0})
    champion_output, champion_seconds, _ = run_package(
        champion_zip, frame, sample, timeout=timeout
    )
    candidate_output, candidate_seconds, stdout = run_package(
        package, frame, sample, timeout=timeout
    )
    champion = _aligned(frame, champion_output)
    candidate = _aligned(frame, candidate_output)

    runtime_dir = Path(tempfile.mkdtemp(prefix="v180_runtime_"))
    module = _load_runtime(package, runtime_dir)
    components = module.predict_components(frame)
    current_parity = float(np.max(np.abs(components["current_jy"] - champion)))
    formula_parity = float(np.max(np.abs(components["candidate"] - candidate)))
    if max(current_parity, formula_parity) > 1e-12:
        raise ValueError(
            f"runtime parity failure: current={current_parity} formula={formula_parity}"
        )
    if not np.any(np.abs(candidate - champion) > 0.0):
        raise ValueError("candidate is prediction-identical to champion")

    shuffled_frame = frame.sample(frac=1.0, random_state=180).reset_index(drop=True)
    shuffled_sample = pd.DataFrame({ID_COL: shuffled_frame[ID_COL], TARGET_COL: 0.0})
    shuffled, _, _ = run_package(package, shuffled_frame, shuffled_sample, timeout=timeout)
    base_map = candidate_output.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffle_max_abs = float(np.max(np.abs(base_map - shuffled_map)))

    pieces = []
    for indices in np.array_split(np.arange(len(frame)), 3):
        part = frame.iloc[indices].reset_index(drop=True)
        part_sample = pd.DataFrame({ID_COL: part[ID_COL], TARGET_COL: 0.0})
        output, _, _ = run_package(package, part, part_sample, timeout=timeout)
        pieces.append(output)
    partition_map = pd.concat(pieces).set_index(ID_COL)[TARGET_COL].sort_index()
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    singleton_outputs = []
    for index in range(len(frame)):
        part = frame.iloc[[index]].reset_index(drop=True)
        part_sample = pd.DataFrame({ID_COL: part[ID_COL], TARGET_COL: 0.0})
        output, _, _ = run_package(package, part, part_sample, timeout=timeout)
        singleton_outputs.append(output)
    singleton_map = pd.concat(singleton_outputs).set_index(ID_COL)[TARGET_COL].sort_index()
    singleton_max_abs = float(np.max(np.abs(base_map - singleton_map)))
    if max(shuffle_max_abs, partition_max_abs, singleton_max_abs) > 1e-12:
        raise ValueError("row-independence audit failed")
    _safe_remove_generated(runtime_dir, runtime_dir.parent)
    return {
        "rows": int(len(frame)),
        "champion_seconds": champion_seconds,
        "candidate_seconds": candidate_seconds,
        "current_champion_parity_max_abs": current_parity,
        "formula_parity_max_abs": formula_parity,
        "shuffle_max_abs": shuffle_max_abs,
        "partition_max_abs": partition_max_abs,
        "singleton_max_abs": singleton_max_abs,
        "mean_abs_shift": float(np.mean(np.abs(candidate - champion))),
        "min": float(candidate.min()),
        "mean": float(candidate.mean()),
        "max": float(candidate.max()),
        "stdout": stdout,
    }


def run(
    champion_zip: Path,
    proxy_h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    exact_oof: Path,
    component_dir: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    timeout: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    package, build = build_package(
        champion_zip, proxy_h1_zip, runtime_script, train_csv, exact_oof,
        component_dir, output_dir,
    )
    audit = audit_package(package, champion_zip, data_dir, timeout=timeout)
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_api_submission",
        "package": {
            "path": str(package), "bytes": package.stat().st_size,
            "sha256": _sha256(package),
        },
        "build": build,
        "audit": audit,
        "official_train_only": True,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "row_local_inference": True,
        "eligible_for_api_submission": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-zip", type=Path, required=True)
    parser.add_argument("--proxy-h1-zip", type=Path, required=True)
    parser.add_argument("--runtime-script", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--exact-oof", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = run(
        args.champion_zip, args.proxy_h1_zip, args.runtime_script,
        args.train_csv, args.exact_oof, args.component_dir, args.data_dir,
        args.output_dir, timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()

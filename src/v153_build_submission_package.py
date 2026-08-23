"""Build and audit the deployable v153 May-maturity bridge package."""

from __future__ import annotations

import argparse
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
from src.v124_public_quadratic_stack import build_package as build_intermediate
from src.v142_build_submission_package import build_c3_bundle
from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
    run_package,
)


PROTOCOL = "V153_MAY_MATURITY_BRIDGE_PACKAGE_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"
MATURITY_SPEC_PATHS = (
    "model/parent/parent/champion/v14_refinement_spec.json",
    "model/parent/parent/champion/v16_residual_spec.json",
    "model/parent/parent/champion/v20_target1160_spec.json",
    "model/parent/parent/champion/v21_context_state_eb_spec.json",
    "model/parent/parent/champion/v22_low_variance_spec.json",
    "model/parent/parent/champion/v25_postbreak_anchor_spec.json",
)


def _extract(package: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        _safe_extract(archive, destination)


def _load_runtime(package: Path, destination: Path):
    _extract(package, destination)
    spec = importlib.util.spec_from_file_location("v153_packaged_runtime", destination / "script.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load v153 packaged runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _member_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _shared_member_audit(intermediate_zip: Path, mature_zip: Path) -> dict[str, Any]:
    with zipfile.ZipFile(intermediate_zip) as left, zipfile.ZipFile(mature_zip) as right:
        left_hash = {name: _member_hash(left.read(name)) for name in left.namelist()}
        right_hash = {name: _member_hash(right.read(name)) for name in right.namelist()}
    if set(left_hash) != set(right_hash):
        raise ValueError("intermediate and mature member sets differ")
    changed = sorted(name for name in left_hash if left_hash[name] != right_hash[name])
    expected = sorted(MATURITY_SPEC_PATHS)
    if changed != expected:
        raise ValueError(f"unexpected intermediate/mature diff: {changed}")
    return {"member_count": len(left_hash), "changed_members": changed}


def build_package(
    parent_v104: Path,
    mature_v124: Path,
    h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    oof_path: Path,
    v124_config: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    base_config = dict(v124_config)
    base_config["package_name"] = "base_v153_intermediate.zip"
    base_config["selected_point"] = config["intermediate_point"]
    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    intermediate_zip, intermediate_build, intermediate_diff = build_intermediate(
        parent_v104, base_config, base_dir
    )
    shared_audit = _shared_member_audit(intermediate_zip, mature_v124)

    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    _extract(intermediate_zip, staging / "model" / "intermediate")
    mature_specs = staging / "model" / "mature_specs"
    mature_specs.mkdir(parents=True)
    with zipfile.ZipFile(mature_v124) as archive:
        for member in MATURITY_SPEC_PATHS:
            (mature_specs / Path(member).name).write_bytes(archive.read(member))
    _extract(h1_zip, staging / "model" / "h1")
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
    h1_bundle["note"] = str(h1_bundle.get("note", "")) + " | v153 raw OOF-aligned affine=identity"
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
        "intermediate": {
            "path": str(intermediate_zip),
            "sha256": _sha256(intermediate_zip),
            "point": config["intermediate_point"],
            "build": intermediate_build,
            "archive_diff_vs_v104": intermediate_diff,
        },
        "mature_v124_sha256": _sha256(mature_v124),
        "intermediate_vs_mature": shared_audit,
        "h1_original_affine": original_affine,
        "h1_packaged_affine": {"alpha": 1.0, "center": 0.5},
        "c3_window_sources": c3_bundle["window_sources"],
    }


def _audit_rows(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    mixed = pd.concat([source] * 6, ignore_index=True)
    mixed[ID_COL] = [f"v153_audit_{index:04d}" for index in range(len(mixed))]
    route = np.arange(len(mixed)) % 3
    mixed.loc[route == 0, "game_type"] = "R"
    mixed.loc[route == 0, "pitcher_team_id"] = 1
    mixed.loc[route == 0, "batter_team_id"] = 2
    mixed.loc[route == 1, "game_type"] = "R"
    mixed.loc[route == 1, "pitcher_team_id"] = 13
    mixed.loc[route == 2, "game_type"] = "F"
    mixed.loc[np.arange(len(mixed)) % 2 == 0, "game_month"] = 5
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET_COL: 0.0})
    return mixed, sample


def _aligned(frame: pd.DataFrame, output: pd.DataFrame) -> np.ndarray:
    return frame[ID_COL].map(output.set_index(ID_COL)[TARGET_COL]).to_numpy(np.float64)


def audit_package(
    package: Path,
    intermediate_zip: Path,
    mature_v124: Path,
    data_dir: Path,
    output_dir: Path,
    timeout: int,
    runtime_limit: float,
    scale_proxy_rows: int,
) -> dict[str, Any]:
    mixed, sample = _audit_rows(data_dir)
    intermediate_output, _, _ = run_package(intermediate_zip, mixed, sample, timeout=timeout)
    candidate_output, smoke_seconds, smoke_stdout = run_package(package, mixed, sample, timeout=timeout)
    runtime_dir = Path(tempfile.mkdtemp(prefix="v153_runtime_"))
    module = _load_runtime(package, runtime_dir)
    intermediate, mature, h1, c3, active, maturity, composed = module.predict_components(mixed)
    candidate = _aligned(mixed, candidate_output)
    parent = _aligned(mixed, intermediate_output)
    formula_max_abs = float(np.max(np.abs(composed - candidate)))
    intermediate_max_abs = float(np.max(np.abs(intermediate - parent)))
    mature_frame = mixed.loc[maturity].reset_index(drop=True)
    mature_sample = pd.DataFrame({ID_COL: mature_frame[ID_COL], TARGET_COL: 0.0})
    mature_output, _, _ = run_package(mature_v124, mature_frame, mature_sample, timeout=timeout)
    mature_max_abs = float(np.max(np.abs(mature[maturity] - _aligned(mature_frame, mature_output))))
    expected = intermediate.copy()
    expected[active] = np.clip(0.85 * intermediate[active] + 0.15 * h1[active] + 0.5 * c3[active], 0.001, 0.999)
    expected[maturity] = mature[maturity]
    composition_max_abs = float(np.max(np.abs(expected - candidate)))
    nonmay_inactive = ~maturity & ~active
    inactive_max_abs = float(np.max(np.abs(candidate[nonmay_inactive] - parent[nonmay_inactive])))
    if max(formula_max_abs, intermediate_max_abs, mature_max_abs, composition_max_abs, inactive_max_abs) > 1e-12:
        raise ValueError("v153 component or formula parity failure")
    if not np.any(np.abs(candidate[active & ~maturity] - parent[active & ~maturity]) > 0.0):
        raise ValueError("v153 bridge route is prediction-identical")
    if not np.any(np.abs(candidate[maturity] - parent[maturity]) > 0.0):
        raise ValueError("v153 May maturity gate is prediction-identical")
    _safe_remove_generated(runtime_dir, runtime_dir.parent)

    shuffled_test = mixed.sample(frac=1.0, random_state=153).reset_index(drop=True)
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
        raise ValueError("v153 row-order or partition parity failure")

    scale = mixed.iloc[np.arange(scale_proxy_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v153_scale_{index:06d}" for index in range(scale_proxy_rows)]
    scale["game_month"] = 3 + (np.arange(scale_proxy_rows) % 8)
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(package, scale, scale_sample, timeout=timeout)
    values = scale_output[TARGET_COL].to_numpy(np.float64)
    if scale_seconds > runtime_limit:
        raise ValueError(f"v153 runtime exceeded: {scale_seconds:.3f} > {runtime_limit:.3f}")
    return {
        "smoke_rows": int(len(mixed)),
        "smoke_seconds": smoke_seconds,
        "formula_max_abs": formula_max_abs,
        "intermediate_max_abs": intermediate_max_abs,
        "mature_may_max_abs": mature_max_abs,
        "composition_max_abs": composition_max_abs,
        "inactive_max_abs": inactive_max_abs,
        "active_fraction": float(active.mean()),
        "maturity_fraction": float(maturity.mean()),
        "active_nonmay_mean_abs_shift": float(np.mean(np.abs(candidate[active & ~maturity] - parent[active & ~maturity]))),
        "maturity_mean_abs_shift": float(np.mean(np.abs(candidate[maturity] - parent[maturity]))),
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "scale_proxy_rows": int(scale_proxy_rows),
        "scale_maturity_fraction": 0.125,
        "scale_proxy_seconds": scale_seconds,
        "runtime_limit_seconds": runtime_limit,
        "scale_min": float(values.min()),
        "scale_mean": float(values.mean()),
        "scale_max": float(values.max()),
        "smoke_stdout": smoke_stdout,
        "scale_stdout": scale_stdout,
    }


def run(
    parent_v104: Path,
    mature_v124: Path,
    h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    oof_path: Path,
    data_dir: Path,
    v124_config_path: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    for path, key in (
        (parent_v104, "parent_v104_sha256"),
        (mature_v124, "mature_v124_sha256"),
        (h1_zip, "h1_package_sha256"),
    ):
        if _sha256(path) != str(config[key]).upper():
            raise ValueError(f"SHA mismatch: {key}")
    output_dir.mkdir(parents=True, exist_ok=True)
    v124_config = json.loads(v124_config_path.read_text(encoding="utf-8"))
    package, build = build_package(
        parent_v104, mature_v124, h1_zip, runtime_script, train_csv,
        oof_path, v124_config, config, output_dir,
    )
    intermediate_zip = Path(build["intermediate"]["path"])
    audit = audit_package(
        package, intermediate_zip, mature_v124, data_dir, output_dir, timeout,
        float(config["runtime_limit_seconds"]), int(config["scale_proxy_rows"]),
    )
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_api_submission",
        "formula": (
            "May: mature v124; otherwise 0.85*intermediate(v124->v104,w=.85) + "
            "0.15*H1 + 0.5*(0.15*sign_all_C3 + 0.85*mean_recent_C3) on R_CORE"
        ),
        "package": {"path": str(package), "bytes": package.stat().st_size, "sha256": _sha256(package)},
        "public_estimate": config["public_estimate"],
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
    parser.add_argument("--parent-v104", type=Path, required=True)
    parser.add_argument("--mature-v124", type=Path, required=True)
    parser.add_argument("--h1-zip", type=Path, required=True)
    parser.add_argument("--runtime-script", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v124-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    run(
        args.parent_v104, args.mature_v124, args.h1_zip, args.runtime_script,
        args.train_csv, args.oof_path, args.data_dir, args.v124_config,
        args.config, args.output_dir, args.timeout,
    )


if __name__ == "__main__":
    main()

"""Build and audit the deployable v154 distilled May-maturity package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.package import verify_package
from src.v124_public_quadratic_stack import build_package as build_intermediate
from src.v142_build_submission_package import build_package as build_h1_c3
from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
    run_package,
)


PROTOCOL = "V154_MATURITY_DELTA_SUBMISSION_PACKAGE_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"


def _extract(package: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        _safe_extract(archive, destination)


def _load_runtime(package: Path, destination: Path):
    _extract(package, destination)
    spec = importlib.util.spec_from_file_location("v154_packaged_runtime", destination / "script.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load v154 packaged runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_package(
    parent_v104: Path,
    h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    oof_path: Path,
    maturity_model: Path,
    maturity_spec: Path,
    v124_config: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    base_config = dict(v124_config)
    base_config["package_name"] = "base_v154_intermediate.zip"
    base_config["selected_point"] = config["intermediate_point"]
    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    intermediate_zip, intermediate_build, intermediate_diff = build_intermediate(
        parent_v104, base_config, base_dir
    )
    stack_config = {"package_name": "base_v154_stack.zip", "c3": config["c3"]}
    stack_zip, stack_build = build_h1_c3(
        intermediate_zip, h1_zip, runtime_script, train_csv, oof_path,
        stack_config, output_dir,
    )
    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    _extract(stack_zip, staging)
    maturity_dir = staging / "model" / "maturity"
    maturity_dir.mkdir(parents=True)
    shutil.copy2(maturity_model, maturity_dir / "maturity_delta_lgb.txt")
    shutil.copy2(maturity_spec, maturity_dir / "maturity_delta_spec.json")
    package = output_dir / str(config["package_name"])
    if package.exists():
        _safe_remove_generated(package, output_dir)
    package_result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    verify_package(package)
    return package, {
        "package": package_result,
        "intermediate": {
            "path": str(intermediate_zip), "sha256": _sha256(intermediate_zip),
            "point": config["intermediate_point"], "build": intermediate_build,
            "archive_diff_vs_v104": intermediate_diff,
        },
        "stack": {"path": str(stack_zip), "sha256": _sha256(stack_zip), "build": stack_build},
        "maturity_model_sha256": _sha256(maturity_model),
        "maturity_spec_sha256": _sha256(maturity_spec),
    }


def _audit_rows(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    mixed = pd.concat([source] * 6, ignore_index=True)
    mixed[ID_COL] = [f"v154_audit_{index:04d}" for index in range(len(mixed))]
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
    data_dir: Path,
    timeout: int,
    runtime_limit: float,
    scale_proxy_rows: int,
    runtime_calibration: dict[str, float],
) -> dict[str, Any]:
    mixed, sample = _audit_rows(data_dir)
    parent_output, _, _ = run_package(intermediate_zip, mixed, sample, timeout=timeout)
    candidate_output, smoke_seconds, smoke_stdout = run_package(package, mixed, sample, timeout=timeout)
    runtime_dir = Path(tempfile.mkdtemp(prefix="v154_runtime_"))
    module = _load_runtime(package, runtime_dir)
    intermediate, delta, h1, c3, active, maturity, composed = module.predict_components(mixed)
    candidate = _aligned(mixed, candidate_output)
    parent = _aligned(mixed, parent_output)
    expected = intermediate.copy()
    expected[active] = np.clip(0.85 * intermediate[active] + 0.15 * h1[active] + 0.5 * c3[active], 0.001, 0.999)
    expected[maturity] = np.clip(intermediate[maturity] + delta[maturity], 0.001, 0.999)
    formula_max_abs = float(np.max(np.abs(composed - candidate)))
    composition_max_abs = float(np.max(np.abs(expected - candidate)))
    intermediate_max_abs = float(np.max(np.abs(intermediate - parent)))
    nonmay_inactive = ~maturity & ~active
    inactive_max_abs = float(np.max(np.abs(candidate[nonmay_inactive] - parent[nonmay_inactive])))
    if max(formula_max_abs, composition_max_abs, intermediate_max_abs, inactive_max_abs) > 1e-12:
        raise ValueError("v154 component or formula parity failure")
    if not np.any(np.abs(candidate[active & ~maturity] - parent[active & ~maturity]) > 0.0):
        raise ValueError("v154 bridge route is prediction-identical")
    if not np.any(np.abs(delta[maturity]) > 0.0):
        raise ValueError("v154 maturity delta is identically zero")
    _safe_remove_generated(runtime_dir, runtime_dir.parent)

    shuffled_test = mixed.sample(frac=1.0, random_state=154).reset_index(drop=True)
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
        raise ValueError("v154 row-order or partition parity failure")

    scale = mixed.iloc[np.arange(scale_proxy_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v154_scale_{index:06d}" for index in range(scale_proxy_rows)]
    scale["game_month"] = 3 + (np.arange(scale_proxy_rows) % 8)
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(package, scale, scale_sample, timeout=timeout)
    values = scale_output[TARGET_COL].to_numpy(np.float64)
    historical = float(runtime_calibration["historical_v148_seconds"])
    matched = float(runtime_calibration["matched_load_v148_seconds"])
    slowdown = matched / historical
    normalized_seconds = scale_seconds / slowdown
    calibrated_pass = bool(
        slowdown >= float(runtime_calibration["minimum_slowdown_factor"])
        and normalized_seconds <= runtime_limit
    )
    if scale_seconds > runtime_limit and not calibrated_pass:
        raise ValueError(
            f"v154 runtime exceeded: raw={scale_seconds:.3f}, "
            f"normalized={normalized_seconds:.3f}, limit={runtime_limit:.3f}"
        )
    return {
        "smoke_rows": int(len(mixed)), "smoke_seconds": smoke_seconds,
        "formula_max_abs": formula_max_abs, "composition_max_abs": composition_max_abs,
        "intermediate_max_abs": intermediate_max_abs, "inactive_max_abs": inactive_max_abs,
        "active_fraction": float(active.mean()), "maturity_fraction": float(maturity.mean()),
        "maturity_delta_mean_absolute": float(np.mean(np.abs(delta[maturity]))),
        "shuffled_max_abs": shuffled_max_abs, "partition_max_abs": partition_max_abs,
        "scale_proxy_rows": int(scale_proxy_rows), "scale_maturity_fraction": 0.125,
        "scale_proxy_seconds": scale_seconds, "runtime_limit_seconds": runtime_limit,
        "runtime_calibration": {
            "historical_v148_seconds": historical,
            "matched_load_v148_seconds": matched,
            "observed_slowdown_factor": slowdown,
            "normalized_v154_seconds": normalized_seconds,
            "normalized_headroom_seconds": runtime_limit - normalized_seconds,
            "calibrated_pass": calibrated_pass,
        },
        "scale_min": float(values.min()), "scale_mean": float(values.mean()),
        "scale_max": float(values.max()), "smoke_stdout": smoke_stdout,
        "scale_stdout": scale_stdout,
    }


def run(
    parent_v104: Path,
    h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    oof_path: Path,
    maturity_model: Path,
    maturity_spec: Path,
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
        (parent_v104, "parent_v104_sha256"), (h1_zip, "h1_package_sha256"),
        (maturity_model, "maturity_model_sha256"), (maturity_spec, "maturity_spec_sha256"),
    ):
        if _sha256(path) != str(config[key]).upper():
            raise ValueError(f"SHA mismatch: {key}")
    output_dir.mkdir(parents=True, exist_ok=True)
    v124_config = json.loads(v124_config_path.read_text(encoding="utf-8"))
    package, build = build_package(
        parent_v104, h1_zip, runtime_script, train_csv, oof_path,
        maturity_model, maturity_spec, v124_config, config, output_dir,
    )
    audit = audit_package(
        package, Path(build["intermediate"]["path"]), data_dir, timeout,
        float(config["runtime_limit_seconds"]), int(config["scale_proxy_rows"]),
        config["runtime_calibration"],
    )
    result = {
        "protocol": PROTOCOL, "status": "eligible_for_api_submission",
        "formula": (
            "May: intermediate(v124->v104,w=.85)+distilled(v124-intermediate); "
            "otherwise 0.85*intermediate + 0.15*H1 + "
            "0.5*(0.15*sign_all_C3+0.85*mean_recent_C3) on R_CORE"
        ),
        "package": {"path": str(package), "bytes": package.stat().st_size, "sha256": _sha256(package)},
        "public_estimate": config["public_estimate"], "build": build, "audit": audit,
        "eligible_for_api_submission": True, **config["restrictions"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-v104", type=Path, required=True)
    parser.add_argument("--h1-zip", type=Path, required=True)
    parser.add_argument("--runtime-script", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-path", type=Path, required=True)
    parser.add_argument("--maturity-model", type=Path, required=True)
    parser.add_argument("--maturity-spec", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v124-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    run(
        args.parent_v104, args.h1_zip, args.runtime_script, args.train_csv,
        args.oof_path, args.maturity_model, args.maturity_spec, args.data_dir,
        args.v124_config, args.config, args.output_dir, args.timeout,
    )


if __name__ == "__main__":
    main()

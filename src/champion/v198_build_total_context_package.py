"""Build and audit the exploratory v198 total context-stack package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from src.champion.v180_build_signed_stack_package import audit_package
from src.core.packaging import _safe_extract, _safe_remove_generated, _sha256, _zip_directory
from src.package import verify_package


PROTOCOL = "V198_TOTAL_CONTEXT_STACK_PACKAGE_V1"
PACKAGE_NAME = "submit_v198_total_context_stack.zip"
EXPECTED_V197_MODEL_SHA = "B3D57DB9D64F7EEAF78AC799BB2C51544774AA05383116468569DC83ABFC5DBE"
EXPECTED_V197_SNAPSHOT_SHA = "EC6A3F7A1166B85538112A0C5AABB491929F25E8643BB7248B377B25038E256F"

_OLD_PREDICT_COMPONENTS = '''def predict_components(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    values = _compose_components(frame)
    values["current_jy"] = _predict_current_jy(frame, values)
    values["candidate"] = np.clip(
        values["current_jy"] + STACK_SCALE * values["direction"], 0.001, 0.999
    )
    return values
'''

_NEW_PREDICT_COMPONENTS = '''def predict_components(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    values = _compose_components(frame)
    values["current_jy"] = _predict_current_jy(frame, values)
    candidate = np.clip(
        values["current_jy"] + STACK_SCALE * values["direction"], 0.001, 0.999
    )
    score = pd.to_numeric(
        frame["score_diff_pitcher_team"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    leverage = pd.to_numeric(frame["li"], errors="coerce").fillna(1.0).to_numpy(np.float64)
    gate = values["rcore"] & (np.abs(score) <= 1.0) & (leverage > 0.7) & (leverage <= 1.5)
    hierarchy = np.full(len(frame), np.nan, dtype=np.float64)
    if np.any(gate):
        module = _load_module(
            "v198_hier_context_runtime", MODEL_DIR / "hier_context" / "runtime.py"
        )
        local = module.predict(
            frame.loc[gate].reset_index(drop=True),
            MODEL_DIR / "h1",
            MODEL_DIR / "hier_context",
        )
        hierarchy[gate] = local
        candidate[gate] = np.clip(
            candidate[gate] + HIER_CONTEXT_WEIGHT * (local - candidate[gate]),
            0.001, 0.999,
        )
    values["hier_context"] = hierarchy
    values["hier_gate"] = gate
    values["candidate"] = candidate
    return values
'''

_REPLACEMENTS = {
    "Standalone runtime for the soft-LOO-selected v193 triyear signed stack.":
        "Standalone runtime for the exploratory v198 total context stack.",
    "V160_WEIGHT = 0.0": "V160_WEIGHT = 0.0\nHIER_CONTEXT_WEIGHT = 0.05",
    _OLD_PREDICT_COMPONENTS: _NEW_PREDICT_COMPONENTS,
    "v193 produced invalid probabilities": "v198 produced invalid probabilities",
    "candidate=v193_triyear_stack": "candidate=v198_total_context_stack",
}


def _patch_runtime(source: str) -> str:
    patched = source.replace("\r\n", "\n")
    for old, new in _REPLACEMENTS.items():
        count = patched.count(old)
        if count != 1:
            raise ValueError(f"expected one runtime token, found {count}: {old[:80]!r}")
        patched = patched.replace(old, new)
    return patched


def _contract(v196_path: Path, v197_path: Path) -> dict[str, Any]:
    v196 = json.loads(v196_path.read_text(encoding="utf-8"))
    v197 = json.loads(v197_path.read_text(encoding="utf-8"))
    if v196.get("protocol") != "V196_TOTAL_CONTEXT_STACK_AUDIT_V1":
        raise ValueError("unexpected v196 protocol")
    if v197.get("protocol") != "V197_FINAL_HIER_CONTEXT_MODEL_V1":
        raise ValueError("unexpected v197 protocol")
    if not v196.get("point_gate_passed"):
        raise ValueError("v196 point evidence did not pass")
    if v196.get("selected_gate_from_v195") != "score__li=close|li_mid":
        raise ValueError("unexpected v196 gate")
    if abs(float(v196.get("frozen_weight")) - 0.05) > 1e-15:
        raise ValueError("unexpected v196 weight")
    if v197.get("status") != "final_model_ready":
        raise ValueError("v197 model is not ready")
    if v197["model"]["sha256"] != EXPECTED_V197_MODEL_SHA:
        raise ValueError("v197 model hash mismatch")
    if v197["snapshot"]["sha256"] != EXPECTED_V197_SNAPSHOT_SHA:
        raise ValueError("v197 snapshot hash mismatch")
    return {
        "v196_status": v196["status"],
        "v196_point_gate_passed": v196["point_gate_passed"],
        "v196_robust_gate_passed": v196["robust_gate_passed"],
        "selected_gate": v196["selected_gate_from_v195"],
        "weight": v196["frozen_weight"],
        "total_metrics": v196["total_metrics"],
        "v197_status": v197["status"],
        "hierarchical_2024_parity_max_abs": v197["hierarchical_2024_parity_max_abs"],
    }


def build_package(
    source_package: Path,
    v196_summary: Path,
    v197_dir: Path,
    helper_script: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    contract = _contract(v196_summary, v197_dir / "summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(source_package) as archive:
        _safe_extract(archive, staging)
    runtime_path = staging / "script.py"
    source_runtime = runtime_path.read_text(encoding="utf-8")
    patched_runtime = _patch_runtime(source_runtime)
    runtime_path.write_text(patched_runtime, encoding="utf-8")

    target = staging / "model" / "hier_context"
    target.mkdir(parents=True)
    shutil.copy2(v197_dir / "hier_context.cbm", target / "hier_context.cbm")
    shutil.copy2(v197_dir / "snapshot.joblib", target / "snapshot.joblib")
    shutil.copy2(helper_script, target / "runtime.py")

    package = output_dir / PACKAGE_NAME
    if package.exists():
        _safe_remove_generated(package, output_dir)
    package_result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    verify_package(package)
    return package, {
        "package": package_result,
        "source_package": {
            "path": str(source_package), "bytes": source_package.stat().st_size,
            "sha256": _sha256(source_package),
        },
        "runtime_sha256": hashlib.sha256(patched_runtime.encode("utf-8")).hexdigest().upper(),
        "hier_context_model_sha256": _sha256(v197_dir / "hier_context.cbm"),
        "hier_context_snapshot_sha256": _sha256(v197_dir / "snapshot.joblib"),
        "contract": contract,
    }


def run(
    source_package: Path,
    champion_zip: Path,
    v196_summary: Path,
    v197_dir: Path,
    helper_script: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    timeout: int,
) -> dict[str, Any]:
    package, build = build_package(
        source_package, v196_summary, v197_dir, helper_script, output_dir
    )
    audit = audit_package(package, champion_zip, data_dir, timeout=timeout)
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_exploratory_user_submission",
        "package": {"path": str(package), "bytes": package.stat().st_size, "sha256": _sha256(package)},
        "build": build,
        "audit": audit,
        "official_train_only": True,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "row_local_inference": True,
        "public_score_used_for_selection": False,
        "champion_replaced": False,
        "caveat": "v196 point/cluster gates passed but 130-gate Reality Check p=0.1489 exceeded 0.10",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--champion-zip", type=Path, required=True)
    parser.add_argument("--v196-summary", type=Path, required=True)
    parser.add_argument("--v197-dir", type=Path, required=True)
    parser.add_argument("--helper-script", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = run(
        args.source_package, args.champion_zip, args.v196_summary,
        args.v197_dir, args.helper_script, args.data_dir, args.output_dir,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

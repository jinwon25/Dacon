"""Build and audit the deployable v193 soft-LOO triyear stack package.

The model assets and row-local inference graph are inherited byte-for-byte from
the audited v180 package.  Only the frozen signed-stack constants are replaced
with the official-train-only coefficients selected by v192.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from src.champion.v180_build_signed_stack_package import audit_package
from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
)
from src.package import verify_package


PROTOCOL = "V193_SOFT_LOO_TRIYEAR_STACK_PACKAGE_V1"
PACKAGE_NAME = "submit_v193_triyear_stack.zip"
NET_WEIGHTS = {
    "v114_independent_source_stability_mask_20260823_01": 0.015483964114884825,
    "v131_catboost_h1_independent_oof_20260823_01": -0.062056275020207705,
    "v135_c3_recent_window_20260823_01": 0.0724597608649307,
    "v160_original_h1_affine_audit_20260825_01": 0.0,
}

_RUNTIME_REPLACEMENTS = {
    "Standalone runtime for the source-selected v178 signed stack.":
        "Standalone runtime for the soft-LOO-selected v193 triyear signed stack.",
    "STACK_SCALE = 0.25": "STACK_SCALE = 1.0",
    "V114_WEIGHT = 0.06988829665167925":
        f"V114_WEIGHT = {NET_WEIGHTS['v114_independent_source_stability_mask_20260823_01']!r}",
    "V131_WEIGHT = -0.20028521988178444":
        f"V131_WEIGHT = {NET_WEIGHTS['v131_catboost_h1_independent_oof_20260823_01']!r}",
    "V135_WEIGHT = 0.18470959209793483":
        f"V135_WEIGHT = {NET_WEIGHTS['v135_c3_recent_window_20260823_01']!r}",
    "V160_WEIGHT = 0.04511689136862597":
        f"V160_WEIGHT = {NET_WEIGHTS['v160_original_h1_affine_audit_20260825_01']!r}",
    "v180 produced invalid probabilities": "v193 produced invalid probabilities",
    "candidate=v180_signed_stack": "candidate=v193_triyear_stack",
}


def _patch_runtime(source: str) -> str:
    """Apply an exact, fail-closed patch to the audited v180 runtime."""
    patched = source
    for old, new in _RUNTIME_REPLACEMENTS.items():
        count = patched.count(old)
        if count != 1:
            raise ValueError(f"expected one runtime token, found {count}: {old!r}")
        patched = patched.replace(old, new)
    return patched


def _verify_weight_contract(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != "V192_SOFT_LOO_TRIYEAR_STACK_V1":
        raise ValueError("unexpected v192 protocol")
    selected = summary.get("final_net_weights", {})
    for name, expected in NET_WEIGHTS.items():
        if name.startswith("v160_"):
            continue
        actual = float(selected.get(name, float("nan")))
        if abs(actual - expected) > 1e-15:
            raise ValueError(f"v192 weight mismatch for {name}: {actual} != {expected}")
    restrictions = summary.get("restrictions", {})
    required = {
        "official_train_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_components_only": True,
    }
    if any(restrictions.get(key) is not value for key, value in required.items()):
        raise ValueError("v192 restriction contract mismatch")
    return {
        "source_protocol": summary["protocol"],
        "selected_budget": float(summary["selected_budget_by_soft_loo"]),
        "final_net_weights": selected,
        "full_2022_gain": float(summary["final_metrics"]["full_2022"]["gain"]),
        "late_2023_gain": float(summary["final_metrics"]["late_2023"]["gain"]),
        "full_2024_gain": float(summary["final_metrics"]["full_2024"]["gain"]),
        "robust_gate_passed": bool(summary["robust_gate_passed"]),
    }


def build_package(
    source_package: Path,
    v192_summary: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Repackage v180 assets with the frozen v192 row-local stack formula."""
    contract = _verify_weight_contract(v192_summary)
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

    package = output_dir / PACKAGE_NAME
    if package.exists():
        _safe_remove_generated(package, output_dir)
    package_result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    verify_package(package)
    return package, {
        "package": package_result,
        "source_package": {
            "path": str(source_package),
            "bytes": source_package.stat().st_size,
            "sha256": _sha256(source_package),
        },
        "source_runtime_sha256": hashlib.sha256(
            source_runtime.encode("utf-8")
        ).hexdigest().upper(),
        "patched_runtime_sha256": hashlib.sha256(
            patched_runtime.encode("utf-8")
        ).hexdigest().upper(),
        "weight_contract": contract,
        "runtime_constants": {
            "stack_scale": 1.0,
            "v114_weight": NET_WEIGHTS["v114_independent_source_stability_mask_20260823_01"],
            "v131_weight": NET_WEIGHTS["v131_catboost_h1_independent_oof_20260823_01"],
            "v135_weight": NET_WEIGHTS["v135_c3_recent_window_20260823_01"],
            "v160_weight": NET_WEIGHTS["v160_original_h1_affine_audit_20260825_01"],
        },
    }


def run(
    source_package: Path,
    champion_zip: Path,
    v192_summary: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    timeout: int,
) -> dict[str, Any]:
    package, build = build_package(source_package, v192_summary, output_dir)
    audit = audit_package(package, champion_zip, data_dir, timeout=timeout)
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_user_submission",
        "package": {
            "path": str(package),
            "bytes": package.stat().st_size,
            "sha256": _sha256(package),
        },
        "build": build,
        "audit": audit,
        "official_train_only": True,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "row_local_inference": True,
        "public_score_used_for_selection": False,
        "eligible_for_user_submission": True,
        "champion_replaced": False,
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
    parser.add_argument("--v192-summary", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = run(
        args.source_package,
        args.champion_zip,
        args.v192_summary,
        args.data_dir,
        args.output_dir,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()

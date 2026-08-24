"""Build and audit the v160 fixed-H1-affine candidate from the v148 ZIP.

This intentionally performs a one-file mutation of the already submitted v148
package.  The H1 model bundle is the only changed ZIP member; the root runtime,
all fitted estimators, the v124 parent, and the C3 tables remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.champion.v142_build_submission_package import _audit_rows
from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    run_package,
)
from src.package import verify_package


PROTOCOL = "V167_H1_AFFINE_SUBMISSION_PACKAGE_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"


def _aligned(frame: pd.DataFrame, output: pd.DataFrame) -> np.ndarray:
    mapped = frame[ID_COL].map(output.set_index(ID_COL)[TARGET_COL])
    if mapped.isna().any():
        raise ValueError("package output row_id alignment failure")
    return mapped.to_numpy(np.float64)


def _member_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
        for name in names:
            hashes[name] = hashlib.sha256(archive.read(name)).hexdigest().upper()
    return hashes


def _deterministic_zip_directory(
    source: Path, output: Path, timestamp: list[int]
) -> dict[str, Any]:
    """Archive content with fixed metadata so identical inputs have one SHA-256."""
    date_time = tuple(int(value) for value in timestamp)
    if len(date_time) != 6:
        raise ValueError("deterministic ZIP timestamp must contain six integers")
    names: list[str] = []
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
            names.append(name)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"deterministic ZIP CRC failure: {bad}")
    return {
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "file_count": len(names),
        "files": names,
        "fixed_timestamp": list(date_time),
    }


def _load_runtime(package: Path, destination: Path, module_name: str):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        _safe_extract(archive, destination)
    spec = importlib.util.spec_from_file_location(module_name, destination / "script.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load packaged runtime: {package}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _affine(values: np.ndarray, alpha: float, center: float) -> np.ndarray:
    return np.clip(center + alpha * (values - center), 0.001, 0.999)


def _validate_evidence(summary_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = config["evidence"]
    if summary.get("protocol") != expected["protocol"]:
        raise ValueError("v160 evidence protocol mismatch")
    affine = summary.get("fixed_affine", {})
    candidate = config["candidate_affine"]
    if not np.isclose(float(affine.get("alpha")), float(candidate["alpha"])):
        raise ValueError("v160 evidence alpha mismatch")
    if not np.isclose(float(affine.get("center")), float(candidate["center"])):
        raise ValueError("v160 evidence center mismatch")
    gain = float(summary["locked_metrics"]["gain"])
    if not np.isclose(gain, float(expected["locked_2024_gain"]), atol=1e-12):
        raise ValueError("v160 locked gain mismatch")
    return {
        "path": str(summary_path),
        "protocol": summary["protocol"],
        "status": summary["status"],
        "locked_2024_gain": gain,
        "source_gate_passed": bool(summary["source_gate_passed"]),
        "locked_gate_passed": bool(summary["locked_gate_passed"]),
        "selection_note": expected["selection_note"],
    }


def build_package(
    parent_zip: Path,
    source_h1_zip: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    if _sha256(parent_zip) != str(config["parent_package_sha256"]).upper():
        raise ValueError("v148 parent SHA-256 mismatch")
    if parent_zip.stat().st_size != int(config["parent_package_bytes"]):
        raise ValueError("v148 parent byte-size mismatch")
    if _sha256(source_h1_zip) != str(config["source_h1_package_sha256"]).upper():
        raise ValueError("original H1 source SHA-256 mismatch")

    parent_hashes = _member_hashes(parent_zip)
    if len(parent_hashes) != int(config["expected_file_count"]):
        raise ValueError("unexpected v148 member count")
    roots = {name.split("/", 1)[0] for name in parent_hashes}
    if roots != {"model", "script.py", "requirements.txt"}:
        raise ValueError(f"unexpected v148 roots: {sorted(roots)}")

    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(parent_zip) as archive:
        _safe_extract(archive, staging)

    h1_member = str(config["h1_model_member"])
    h1_path = staging / Path(h1_member)
    parent_bundle = joblib.load(h1_path)
    if len(parent_bundle.get("models", [])) != 3 or len(parent_bundle.get("features", [])) != 82:
        raise ValueError("unexpected deployed H1 bundle structure")
    parent_affine = config["parent_affine"]
    actual_parent_affine = {
        "alpha": float(parent_bundle.get("alpha", 1.0)),
        "center": float(parent_bundle.get("center", 0.5)),
    }
    if not np.isclose(actual_parent_affine["alpha"], float(parent_affine["alpha"])):
        raise ValueError("unexpected deployed H1 alpha")
    if not np.isclose(actual_parent_affine["center"], float(parent_affine["center"])):
        raise ValueError("unexpected deployed H1 center")

    source_member = str(config["source_h1_model_member"])
    with zipfile.ZipFile(source_h1_zip) as source_archive:
        source_names = source_archive.namelist()
        if set(source_names) != {"model/rf.pkl", "script.py", "requirements.txt"}:
            raise ValueError("unexpected original H1 package layout")
        if hashlib.sha256(source_archive.read("script.py")).hexdigest().upper() != parent_hashes[
            "model/h1/script.py"
        ]:
            raise ValueError("original H1 runtime differs from deployed H1 runtime")
        if hashlib.sha256(source_archive.read("requirements.txt")).hexdigest().upper() != parent_hashes[
            "model/h1/requirements.txt"
        ]:
            raise ValueError("original H1 requirements differ from deployed H1 requirements")
        source_model_bytes = source_archive.read(source_member)
    source_bundle = joblib.load(io.BytesIO(source_model_bytes))
    candidate_affine = config["candidate_affine"]
    source_affine = {
        "alpha": float(source_bundle.get("alpha", 1.0)),
        "center": float(source_bundle.get("center", 0.5)),
    }
    if len(source_bundle.get("models", [])) != 3 or len(source_bundle.get("features", [])) != 82:
        raise ValueError("unexpected original H1 bundle structure")
    if not np.isclose(source_affine["alpha"], float(candidate_affine["alpha"])):
        raise ValueError("original H1 alpha mismatch")
    if not np.isclose(source_affine["center"], float(candidate_affine["center"])):
        raise ValueError("original H1 center mismatch")
    h1_path.write_bytes(source_model_bytes)

    package = output_dir / str(config["package_name"])
    if package.exists():
        _safe_remove_generated(package, output_dir)
    package_result = _deterministic_zip_directory(
        staging, package, config["deterministic_zip_timestamp"]
    )
    _safe_remove_generated(staging, output_dir)
    verify_package(package)

    candidate_hashes = _member_hashes(package)
    if set(candidate_hashes) != set(parent_hashes):
        raise ValueError("candidate ZIP member set differs from v148")
    changed = sorted(
        name for name in parent_hashes if parent_hashes[name] != candidate_hashes[name]
    )
    if changed != [h1_member]:
        raise ValueError(f"unexpected candidate member changes: {changed}")

    with tempfile.TemporaryDirectory(prefix="v167-bundle-check-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(package) as archive:
            _safe_extract(archive, stage)
        packaged = joblib.load(stage / Path(h1_member))
    packaged_affine = {
        "alpha": float(packaged["alpha"]),
        "center": float(packaged["center"]),
    }
    if not np.isclose(packaged_affine["alpha"], float(candidate_affine["alpha"])):
        raise ValueError("packaged H1 alpha mismatch")
    if not np.isclose(packaged_affine["center"], float(candidate_affine["center"])):
        raise ValueError("packaged H1 center mismatch")
    return package, {
        "package": package_result,
        "parent_sha256": _sha256(parent_zip),
        "source_h1_sha256": _sha256(source_h1_zip),
        "source_h1_model_sha256": hashlib.sha256(source_model_bytes).hexdigest().upper(),
        "parent_file_count": len(parent_hashes),
        "changed_members": changed,
        "h1_models": len(packaged.get("models", [])),
        "h1_features": len(packaged.get("features", [])),
        "parent_affine": actual_parent_affine,
        "candidate_affine": packaged_affine,
    }


def audit_package(
    package: Path,
    parent_zip: Path,
    data_dir: Path,
    config: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    mixed, sample = _audit_rows(data_dir)
    parent_output, parent_seconds, parent_stdout = run_package(
        parent_zip, mixed, sample, timeout=timeout
    )
    candidate_output, candidate_seconds, candidate_stdout = run_package(
        package, mixed, sample, timeout=timeout
    )
    parent_values = _aligned(mixed, parent_output)
    candidate_values = _aligned(mixed, candidate_output)

    with tempfile.TemporaryDirectory(prefix="v167-parent-runtime-") as parent_temp, tempfile.TemporaryDirectory(
        prefix="v167-candidate-runtime-"
    ) as candidate_temp:
        parent_module = _load_runtime(parent_zip, Path(parent_temp), "v167_parent_runtime")
        candidate_module = _load_runtime(
            package, Path(candidate_temp), "v167_candidate_runtime"
        )
        parent_component, raw_h1, parent_c3, parent_active, parent_composed = (
            parent_module.predict_components(mixed)
        )
        candidate_component, adjusted_h1, candidate_c3, active, composed = (
            candidate_module.predict_components(mixed)
        )

    candidate_affine = config["candidate_affine"]
    expected_h1 = _affine(
        np.asarray(raw_h1, dtype=np.float64),
        float(candidate_affine["alpha"]),
        float(candidate_affine["center"]),
    )
    expected = np.asarray(parent_component, dtype=np.float64).copy()
    expected[active] = np.clip(
        (1.0 - float(config["h1_weight"])) * np.asarray(parent_component)[active]
        + float(config["h1_weight"]) * expected_h1[active]
        + float(candidate_module.C3_WEIGHT) * np.asarray(parent_c3)[active],
        0.001,
        0.999,
    )
    parity = {
        "parent_runtime_output_max_abs": float(
            np.max(np.abs(np.asarray(parent_composed) - parent_values))
        ),
        "candidate_runtime_output_max_abs": float(
            np.max(np.abs(np.asarray(composed) - candidate_values))
        ),
        "parent_component_max_abs": float(
            np.max(np.abs(np.asarray(candidate_component) - np.asarray(parent_component)))
        ),
        "c3_component_max_abs": float(
            np.max(np.abs(np.asarray(candidate_c3) - np.asarray(parent_c3)))
        ),
        "h1_affine_max_abs": float(np.max(np.abs(np.asarray(adjusted_h1) - expected_h1))),
        "candidate_formula_max_abs": float(np.max(np.abs(candidate_values - expected))),
        "inactive_parent_max_abs": float(
            np.max(np.abs(candidate_values[~active] - parent_values[~active]))
        ),
    }
    if not np.array_equal(np.asarray(active), np.asarray(parent_active)):
        raise ValueError("v167 route mask differs from v148")
    if max(parity.values()) > 1e-12:
        raise ValueError(f"v167 formula/component parity failure: {parity}")
    active_shift = np.abs(candidate_values[active] - parent_values[active])
    if not np.any(active_shift > 0.0):
        raise ValueError("v167 active route is prediction-identical to v148")

    shuffled_test = mixed.sample(frac=1.0, random_state=167).reset_index(drop=True)
    shuffled_sample = pd.DataFrame({ID_COL: shuffled_test[ID_COL], TARGET_COL: 0.0})
    shuffled_output, _, _ = run_package(
        package, shuffled_test, shuffled_sample, timeout=timeout
    )
    base_map = candidate_output.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled_output.set_index(ID_COL)[TARGET_COL].sort_index()
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
        raise ValueError("v167 row-order or partition parity failure")

    scale_rows = int(config["scale_proxy_rows"])
    scale = mixed.iloc[np.arange(scale_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v167_scale_{index:06d}" for index in range(scale_rows)]
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(
        package, scale, scale_sample, timeout=timeout
    )
    scale_values = scale_output[TARGET_COL].to_numpy(np.float64)
    runtime_limit = float(config["runtime_limit_seconds"])
    if scale_seconds > runtime_limit:
        raise ValueError(
            f"v167 runtime exceeded: {scale_seconds:.3f} > {runtime_limit:.3f}"
        )
    if not np.isfinite(scale_values).all() or not np.all(
        (scale_values >= 0.0) & (scale_values <= 1.0)
    ):
        raise ValueError("v167 produced invalid scale-proxy probabilities")

    return {
        "smoke_rows": int(len(mixed)),
        "parent_smoke_seconds": parent_seconds,
        "candidate_smoke_seconds": candidate_seconds,
        "active_fraction": float(np.mean(active)),
        "active_mean_abs_shift": float(np.mean(active_shift)),
        "active_max_abs_shift": float(np.max(active_shift)),
        "parity": parity,
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "scale_proxy_rows": scale_rows,
        "scale_proxy_seconds": scale_seconds,
        "runtime_limit_seconds": runtime_limit,
        "scale_min": float(scale_values.min()),
        "scale_mean": float(scale_values.mean()),
        "scale_max": float(scale_values.max()),
        "parent_stdout": parent_stdout,
        "candidate_stdout": candidate_stdout,
        "scale_stdout": scale_stdout,
    }


def run(
    parent_zip: Path,
    source_h1_zip: Path,
    data_dir: Path,
    evidence_summary: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = _validate_evidence(evidence_summary, config)
    package, build = build_package(parent_zip, source_h1_zip, config, output_dir)
    audit = audit_package(package, parent_zip, data_dir, config, timeout)
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_user_authorized_exploratory_submission",
        "package": {
            "path": str(package),
            "bytes": package.stat().st_size,
            "sha256": _sha256(package),
        },
        "evidence": evidence,
        "build": build,
        "audit": audit,
        "eligible_for_api_submission": True,
        **config["restrictions"],
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--source-h1-zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--evidence-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    run(
        args.parent_zip,
        args.source_h1_zip,
        args.data_dir,
        args.evidence_summary,
        args.config,
        args.output_dir,
        args.timeout,
    )


if __name__ == "__main__":
    main()

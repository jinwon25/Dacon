"""Rebuild EXP-021 strict and package it with the standalone 1158 champion."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _safe_remove_generated(path: Path, output_dir: Path) -> None:
    resolved = path.resolve()
    root = output_dir.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"unsafe generated cleanup target: {resolved}")
    if resolved.exists():
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()


def _write_smoke_data(stage: Path, data_dir: Path, rows: int = 5) -> None:
    data = stage / "data"
    data.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(data_dir / "test.csv", nrows=rows, encoding="utf-8-sig")
    sample = pd.read_csv(
        data_dir / "sample_submission.csv", nrows=rows, encoding="utf-8-sig"
    )
    test.to_csv(data / "test.csv", index=False, encoding="utf-8")
    sample.to_csv(data / "sample_submission.csv", index=False, encoding="utf-8")


def _smoke_component(path: Path, data_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v82-strict-smoke-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(stage)
        _write_smoke_data(stage, data_dir)
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "script.py"],
            cwd=stage,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "strict component smoke failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        prediction = pd.read_csv(stage / "output" / "submission.csv")
        return {
            "runtime_seconds": time.perf_counter() - started,
            "rows": len(prediction),
            "stdout": result.stdout.strip(),
        }


def rebuild_strict(
    project: Path, external_root: Path, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    strict_zip = output_dir / "_strict_component.zip"
    report_path = (
        output_dir
        / "_external_report"
        / "artifacts"
        / "EXP-021"
        / "final_packages"
        / "validation_metrics.json"
    )
    if strict_zip.exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = report["results"]["strict"]["zip"]["sha256"].upper()
        if _sha256(strict_zip) != expected:
            raise ValueError("resumable strict component SHA-256 mismatch")
        return strict_zip, report

    experiments = external_root / "experiments"
    sys.path.insert(0, str(experiments))
    try:
        builder = importlib.import_module("build_exp021_final_candidates")
        strict_source = output_dir / "_strict_source"
        _safe_remove_generated(strict_source, output_dir)
        strict_source.mkdir(parents=True)
        builder.DATA_DIR = project / "data"
        builder.ROOT = output_dir / "_external_report"
        builder.PYTHON = Path(sys.executable)
        builder.VARIANTS = {
            "strict": {
                "directory": strict_source,
                "zip": strict_zip,
                "candidate": "strict_lowrank_s300_r6",
            }
        }
        original_smoke = builder.smoke_test
        builder.smoke_test = lambda path: _smoke_component(Path(path), project / "data")
        try:
            builder.main()
        finally:
            builder.smoke_test = original_smoke
    finally:
        sys.path.remove(str(experiments))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return strict_zip, report


def _zip_directory(source: Path, output: Path) -> dict[str, Any]:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    if bad is not None:
        raise ValueError(f"v82 ZIP CRC failure: {bad}")
    roots = {name.split("/", 1)[0] for name in names}
    if roots - {"script.py", "requirements.txt", "model"}:
        raise ValueError(f"unexpected v82 ZIP roots: {sorted(roots)}")
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "files": names,
    }


def build_combined_zip(
    champion_zip: Path,
    strict_zip: Path,
    wrapper_path: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    staging = output_dir / "_combined_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    champion_stage = output_dir / "_champion_extract"
    strict_stage = output_dir / "_strict_extract"
    _safe_remove_generated(champion_stage, output_dir)
    _safe_remove_generated(strict_stage, output_dir)
    champion_stage.mkdir()
    strict_stage.mkdir()
    with zipfile.ZipFile(champion_zip) as archive:
        archive.extractall(champion_stage)
    with zipfile.ZipFile(strict_zip) as archive:
        archive.extractall(strict_stage)
    (staging / "model" / "components").mkdir(parents=True)
    shutil.copytree(champion_stage / "model", staging / "model" / "champion")
    shutil.copytree(strict_stage / "model", staging / "model" / "strict")
    shutil.copyfile(
        champion_stage / "script.py",
        staging / "model" / "components" / "champion_script.py",
    )
    shutil.copyfile(
        strict_stage / "script.py",
        staging / "model" / "components" / "strict_script.py",
    )
    shutil.copyfile(wrapper_path, staging / "script.py")
    shutil.copyfile(champion_stage / "requirements.txt", staging / "requirements.txt")
    output = output_dir / "submit_v82_probe.zip"
    result = _zip_directory(staging, output)
    for generated in (staging, champion_stage, strict_stage):
        _safe_remove_generated(generated, output_dir)
    return output, result


def run_package(
    package: Path,
    data_dir: Path,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    timeout: int,
) -> tuple[pd.DataFrame, float, str]:
    with tempfile.TemporaryDirectory(prefix="v82-package-run-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(package) as archive:
            archive.extractall(stage)
        (stage / "data").mkdir()
        test.to_csv(stage / "data" / "test.csv", index=False, encoding="utf-8")
        sample.to_csv(
            stage / "data" / "sample_submission.csv", index=False, encoding="utf-8"
        )
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "script.py"],
            cwd=stage,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "v82 package execution failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        elapsed = time.perf_counter() - started
        return pd.read_csv(stage / "output" / "submission.csv"), elapsed, result.stdout.strip()


def audit_package(package: Path, data_dir: Path, timeout: int) -> dict[str, Any]:
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    if len(test) != len(sample) or len(test) == 0:
        raise ValueError("local official smoke data row mismatch")
    smoke_test = test.reset_index(drop=True)
    smoke_sample = sample.loc[
        sample["row_id"].isin(smoke_test["row_id"])
    ].reset_index(drop=True)
    baseline, smoke_seconds, stdout = run_package(
        package, data_dir, smoke_test, smoke_sample, timeout=timeout
    )
    shuffled_test = smoke_test.sample(frac=1.0, random_state=82).reset_index(drop=True)
    shuffled_sample = smoke_sample.sample(frac=1.0, random_state=83).reset_index(drop=True)
    shuffled, _, _ = run_package(
        package, data_dir, shuffled_test, shuffled_sample, timeout=timeout
    )
    base_map = baseline.set_index("row_id")["control_success"].sort_index()
    shuffled_map = shuffled.set_index("row_id")["control_success"].sort_index()
    shuffled_max_abs = float(np.max(np.abs(base_map - shuffled_map)))
    if shuffled_max_abs > 1e-12:
        raise ValueError(f"v82 shuffled row-independence failure: {shuffled_max_abs}")

    parts = []
    for indices in np.array_split(np.arange(len(smoke_test)), min(2, len(smoke_test))):
        local_test = smoke_test.iloc[indices].reset_index(drop=True)
        local_sample = smoke_sample.loc[
            smoke_sample["row_id"].isin(local_test["row_id"])
        ].reset_index(drop=True)
        local, _, _ = run_package(
            package, data_dir, local_test, local_sample, timeout=timeout
        )
        parts.append(local)
    partition_map = (
        pd.concat(parts, ignore_index=True)
        .set_index("row_id")["control_success"]
        .sort_index()
    )
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    if partition_max_abs > 1e-12:
        raise ValueError(f"v82 partition row-independence failure: {partition_max_abs}")

    scale_rows = 245789
    scale_test = test.iloc[
        np.arange(scale_rows, dtype=np.int64) % len(test)
    ].reset_index(drop=True)
    scale_test["row_id"] = [f"v82_scale_{index:06d}" for index in range(scale_rows)]
    scale_sample = pd.DataFrame(
        {"row_id": scale_test["row_id"], "control_success": 0.0}
    )
    full, full_seconds, full_stdout = run_package(
        package, data_dir, scale_test, scale_sample, timeout=timeout
    )
    values = full["control_success"].to_numpy(np.float64)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("v82 full inference produced invalid probabilities")
    return {
        "official_local_smoke_rows": len(smoke_test),
        "smoke_runtime_seconds": smoke_seconds,
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "scale_proxy_source": "repeated_valid_2025_smoke_rows_with_unique_row_ids",
        "scale_proxy_rows": len(full),
        "scale_proxy_runtime_seconds": full_seconds,
        "scale_proxy_prediction_mean": float(values.mean()),
        "scale_proxy_prediction_min": float(values.min()),
        "scale_proxy_prediction_max": float(values.max()),
        "smoke_stdout": stdout,
        "scale_proxy_stdout": full_stdout,
    }


def run(
    project: Path,
    external_root: Path,
    champion_zip: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    project = project.resolve()
    external_root = external_root.resolve()
    champion_zip = champion_zip.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    if config.get("protocol") != "V82_PUBLIC_PROBE_V57_STRICT_BLEND_V1":
        raise ValueError("unexpected v82 config protocol")
    strict_zip, strict_report = rebuild_strict(project, external_root, output_dir)
    package, package_result = build_combined_zip(
        champion_zip,
        strict_zip,
        Path(__file__).with_name("v82_probe_wrapper.py"),
        output_dir,
    )
    audit = audit_package(package, project / "data", timeout)
    strict_details = json.loads(
        json.dumps(strict_report["results"]["strict"], ensure_ascii=False)
    )
    strict_details["zip"]["path"] = "embedded:model/strict"
    strict_details["zip"]["intermediate_removed_after_audit"] = True
    result = {
        "protocol": config["protocol"],
        "config": config,
        "champion_zip_sha256": _sha256(champion_zip),
        "strict_component": strict_details,
        "package": package_result,
        "audit": audit,
        "target1170_gate_passed": False,
        "eligible_for_public_probe": True,
        "test_aggregate_used": False,
        "row_local_inference": True,
        "standalone_no_parent_zip_dependency": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    for generated in (
        output_dir / "_strict_source",
        output_dir / "_strict_component.zip",
        output_dir / "_external_report",
    ):
        _safe_remove_generated(generated, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--champion-zip", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    run(
        args.project,
        args.external_root,
        args.champion_zip,
        args.config,
        args.output_dir,
        args.timeout,
    )


if __name__ == "__main__":
    main()

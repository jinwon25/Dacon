"""Train, package, and audit the fixed v56 F-route probe above Public 1159."""

from __future__ import annotations

import argparse
import hashlib
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

from src.v84_fixed_v56_export import run as train_and_export
from src.v84_probe_wrapper import apply_fixed_v56


ID_COL = "row_id"
TARGET_COL = "control_success"


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


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in archive.infolist():
        target = (destination / info.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe ZIP member: {info.filename}")
    archive.extractall(destination)


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
        raise ValueError(f"v84 ZIP CRC failure: {bad}")
    roots = {name.split("/", 1)[0] for name in names}
    if roots - {"script.py", "requirements.txt", "model"}:
        raise ValueError(f"unexpected v84 ZIP roots: {sorted(roots)}")
    return {
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "file_count": len(names),
        "files": names,
    }


def build_combined_zip(
    parent_zip: Path,
    fm_export: Path,
    wrapper_path: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    staging = output_dir / "_combined_staging"
    parent_stage = output_dir / "_parent_extract"
    _safe_remove_generated(staging, output_dir)
    _safe_remove_generated(parent_stage, output_dir)
    staging.mkdir(parents=True)
    parent_stage.mkdir()
    with zipfile.ZipFile(parent_zip) as archive:
        _safe_extract(archive, parent_stage)
    (staging / "model" / "components").mkdir(parents=True)
    shutil.copytree(parent_stage / "model", staging / "model" / "parent")
    shutil.copytree(fm_export, staging / "model" / "v56_fm")
    shutil.copyfile(
        parent_stage / "script.py",
        staging / "model" / "components" / "parent_script.py",
    )
    shutil.copyfile(wrapper_path, staging / "script.py")
    shutil.copyfile(parent_stage / "requirements.txt", staging / "requirements.txt")
    output = output_dir / "submit_v84_probe.zip"
    result = _zip_directory(staging, output)
    _safe_remove_generated(staging, output_dir)
    _safe_remove_generated(parent_stage, output_dir)
    return output, result


def run_package(
    package: Path,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    timeout: int,
) -> tuple[pd.DataFrame, float, str]:
    with tempfile.TemporaryDirectory(prefix="v84-package-run-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(package) as archive:
            _safe_extract(archive, stage)
        (stage / "data").mkdir()
        test.to_csv(stage / "data" / "test.csv", index=False, encoding="utf-8")
        sample.to_csv(
            stage / "data" / "sample_submission.csv",
            index=False,
            encoding="utf-8",
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
                "v84 package execution failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        elapsed = time.perf_counter() - started
        return (
            pd.read_csv(stage / "output" / "submission.csv"),
            elapsed,
            result.stdout.strip(),
        )


def _mixed_row_local_smoke(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    if len(source) == 0:
        raise ValueError("local official smoke test is empty")
    mixed = pd.concat([source, source], ignore_index=True)
    mixed[ID_COL] = [f"v84_mixed_{index:03d}" for index in range(len(mixed))]
    mixed.loc[np.arange(len(mixed)) % 2 == 1, "game_type"] = "F"
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET_COL: 0.0})
    return mixed, sample


def audit_package(
    package: Path,
    parent_zip: Path,
    fm_export: Path,
    data_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    mixed, mixed_sample = _mixed_row_local_smoke(data_dir)
    parent_output, parent_seconds, _ = run_package(
        parent_zip, mixed, mixed_sample, timeout=timeout
    )
    candidate, smoke_seconds, stdout = run_package(
        package, mixed, mixed_sample, timeout=timeout
    )
    parent_map = parent_output.set_index(ID_COL)[TARGET_COL]
    parent = mixed[ID_COL].map(parent_map).to_numpy(np.float64)
    expected = apply_fixed_v56(parent, mixed, fm_export)
    candidate_map = candidate.set_index(ID_COL)[TARGET_COL]
    actual = mixed[ID_COL].map(candidate_map).to_numpy(np.float64)
    formula_max_abs = float(np.max(np.abs(expected - actual)))
    if formula_max_abs > 1e-12:
        raise ValueError(f"v84 packaged formula parity failure: {formula_max_abs}")
    active = mixed["game_type"].ne("R").to_numpy()
    inactive_max_abs = float(np.max(np.abs(actual[~active] - parent[~active])))
    active_shift_count = int(np.count_nonzero(np.abs(actual[active] - parent[active]) > 0.0))
    if inactive_max_abs > 1e-12:
        raise ValueError(f"v84 changed protected R rows: {inactive_max_abs}")
    if active_shift_count == 0:
        raise ValueError("v84 FM did not change any synthetic F smoke row")

    shuffled_test = mixed.sample(frac=1.0, random_state=84).reset_index(drop=True)
    shuffled_sample = mixed_sample.sample(frac=1.0, random_state=85).reset_index(drop=True)
    shuffled, _, _ = run_package(
        package, shuffled_test, shuffled_sample, timeout=timeout
    )
    base_map = candidate.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_max_abs = float(np.max(np.abs(base_map - shuffled_map)))
    if shuffled_max_abs > 1e-12:
        raise ValueError(f"v84 shuffled row-independence failure: {shuffled_max_abs}")

    parts = []
    for indices in np.array_split(np.arange(len(mixed)), min(2, len(mixed))):
        local_test = mixed.iloc[indices].reset_index(drop=True)
        local_sample = mixed_sample.loc[
            mixed_sample[ID_COL].isin(local_test[ID_COL])
        ].reset_index(drop=True)
        local, _, _ = run_package(
            package, local_test, local_sample, timeout=timeout
        )
        parts.append(local)
    partition_map = (
        pd.concat(parts, ignore_index=True)
        .set_index(ID_COL)[TARGET_COL]
        .sort_index()
    )
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    if partition_max_abs > 1e-12:
        raise ValueError(f"v84 partition row-independence failure: {partition_max_abs}")

    scale_rows = 245789
    scale_test = mixed.iloc[
        np.arange(scale_rows, dtype=np.int64) % len(mixed)
    ].reset_index(drop=True)
    scale_test[ID_COL] = [f"v84_scale_{index:06d}" for index in range(scale_rows)]
    scale_sample = pd.DataFrame({ID_COL: scale_test[ID_COL], TARGET_COL: 0.0})
    full, full_seconds, full_stdout = run_package(
        package, scale_test, scale_sample, timeout=timeout
    )
    values = full[TARGET_COL].to_numpy(np.float64)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("v84 full inference produced invalid probabilities")
    return {
        "mixed_smoke_rows": int(len(mixed)),
        "synthetic_f_rows": int(active.sum()),
        "parent_smoke_runtime_seconds": parent_seconds,
        "candidate_smoke_runtime_seconds": smoke_seconds,
        "formula_max_abs": formula_max_abs,
        "protected_r_max_abs": inactive_max_abs,
        "active_f_shift_count": active_shift_count,
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "scale_proxy_source": "repeated_valid_2025_smoke_rows_with_alternating_row_local_F_flag",
        "scale_proxy_rows": int(len(full)),
        "scale_proxy_runtime_seconds": full_seconds,
        "scale_proxy_prediction_mean": float(values.mean()),
        "scale_proxy_prediction_min": float(values.min()),
        "scale_proxy_prediction_max": float(values.max()),
        "smoke_stdout": stdout,
        "scale_proxy_stdout": full_stdout,
    }


def run(
    project: Path,
    parent_zip: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    project = project.resolve()
    parent_zip = parent_zip.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != "V84_FIXED_V56_F_ROUTE_PROBE_V1":
        raise ValueError("unexpected v84 config protocol")
    fixed = config["recipe"]
    if not (
        fixed["rank"] == 16
        and fixed["risk"] == "source_domain_equal"
        and fixed["route"] == "F"
        and float(fixed["eta"]) == 0.10
        and list(fixed["source_periods"]) == [2023, 2024]
    ):
        raise ValueError("v84 frozen recipe was changed")

    fm_export = output_dir / "_fm_export"
    _safe_remove_generated(fm_export, output_dir)
    print("[v84] fitting fixed full-2023 + late-2024 shared FM", flush=True)
    training = train_and_export(project, fm_export)
    package, package_result = build_combined_zip(
        parent_zip,
        fm_export,
        Path(__file__).with_name("v84_probe_wrapper.py"),
        output_dir,
    )
    audit = audit_package(
        package, parent_zip, fm_export, project / "data", timeout
    )
    result = {
        "protocol": config["protocol"],
        "config": config,
        "parent_zip_sha256": _sha256(parent_zip),
        "training": training,
        "package": package_result,
        "audit": audit,
        "target1170_gate_passed": False,
        "eligible_for_public_probe": True,
        "eligibility_basis": "positive conditional full-2024 and late-2024 headroom above exact v82 OOF parent; F route is disjoint from v82 R_CORE",
        "test_aggregate_used": False,
        "row_local_inference": True,
        "standalone_no_parent_zip_dependency": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    _safe_remove_generated(fm_export, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    run(
        args.project,
        args.parent_zip,
        args.config,
        args.output_dir,
        args.timeout,
    )


if __name__ == "__main__":
    main()

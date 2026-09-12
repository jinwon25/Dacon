"""Materialize and rebuild the 1158.0746 champion as one standalone release.

The historical package builders patch one submitted ZIP into the next.  That
is useful for lineage audits, but awkward for handoff.  This module performs a
one-time materialization of the final, already self-contained package and then
rebuilds it directly from a frozen payload directory.  Once materialized, no
v22/v25/v26 package is needed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.package import REQUIRED_ROOT, verify_package


PROTOCOL = "STANDALONE_CHAMPION_1158_V1"
PUBLIC_SCORE = 1158.0745556751
REQUIRED_SCRIPT_MARKERS = (
    "def apply_v25_postbreak_anchor_overlay(",
    "def apply_trackman_asof_gate_overlay(",
    "prediction = apply_v25_postbreak_anchor_overlay(prediction, frame)",
    "prediction = apply_trackman_asof_gate_overlay(",
)

# DACON notice 417123 explicitly forbids using another evaluation row or the
# evaluation-batch distribution.  These calls are high-confidence indicators
# of that pattern in an inference-only script.  Row-wise reductions such as
# ``frame[columns].mean(axis=1)`` are intentionally not banned here; the
# singleton/full-batch parity check below covers their semantics.
FORBIDDEN_TEST_BATCH_METHODS = frozenset(
    {
        "groupby",
        "rolling",
        "expanding",
        "ewm",
        "shift",
        "diff",
        "pct_change",
        "rank",
        "quantile",
        "value_counts",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "cumcount",
    }
)
FORBIDDEN_ORDER_FUNCTIONS = frozenset(
    {"argsort", "argpartition", "lexsort", "percentile", "nanpercentile"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe ZIP member: {member.filename}")
        resolved = (destination / member_path).resolve()
        if destination != resolved and destination not in resolved.parents:
            raise ValueError(f"ZIP member escapes destination: {member.filename}")
    archive.extractall(destination)


def _file_manifest(payload_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(payload_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(payload_dir.rglob("*"))
        if path.is_file()
    ]


def audit_row_independent_script(script: str) -> dict[str, object]:
    """Reject obvious evaluation-batch aggregation from an inference script."""

    tree = ast.parse(script)
    violations: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method in FORBIDDEN_TEST_BATCH_METHODS:
            violations.append({"line": node.lineno, "operation": method})
        elif method in FORBIDDEN_ORDER_FUNCTIONS:
            violations.append({"line": node.lineno, "operation": method})
    if violations:
        detail = ", ".join(
            f"line {item['line']}: {item['operation']}" for item in violations
        )
        raise ValueError(f"forbidden test-batch operations in script.py: {detail}")
    return {
        "notice": "https://dacon.io/competitions/official/236743/talkboard/417123",
        "forbidden_operation_count": 0,
        "status": "pass",
    }


def validate_payload(payload_dir: Path) -> dict[str, object]:
    payload_dir = payload_dir.resolve()
    if not payload_dir.is_dir():
        raise FileNotFoundError(payload_dir)
    roots = {path.name for path in payload_dir.iterdir()}
    if roots != REQUIRED_ROOT:
        raise ValueError(f"bad payload roots: {roots}, expected {REQUIRED_ROOT}")
    if not (payload_dir / "model").is_dir():
        raise ValueError("payload model root must be a directory")

    script = (payload_dir / "script.py").read_text(encoding="utf-8")
    missing_markers = [marker for marker in REQUIRED_SCRIPT_MARKERS if marker not in script]
    if missing_markers:
        raise ValueError(f"standalone script markers missing: {missing_markers}")
    if "from src" in script or "import src" in script:
        raise ValueError("standalone script must not import repository modules")
    row_independence_audit = audit_row_independent_script(script)

    hybrid_path = payload_dir / "model" / "hybrid.json"
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
    gate = hybrid.get("trackman_asof_gate")
    if not isinstance(gate, dict):
        raise ValueError("trackman_asof_gate metadata missing from hybrid.json")
    if float(gate.get("eta", -1.0)) != 0.03:
        raise ValueError("unexpected TrackMan-ASOF gate eta")
    if float(gate.get("public_score", -1.0)) != PUBLIC_SCORE:
        raise ValueError("unexpected champion public score metadata")

    files = _file_manifest(payload_dir)
    model_files = [item for item in files if str(item["path"]).startswith("model/")]
    return {
        "roots": sorted(roots),
        "file_count": len(files),
        "model_file_count": len(model_files),
        "payload_bytes": sum(int(item["size_bytes"]) for item in files),
        "row_independence_static_audit": row_independence_audit,
        "files": files,
    }


def materialize(
    source_package: Path,
    payload_dir: Path,
    *,
    expected_source_sha256: str | None = None,
) -> dict[str, object]:
    source_package = source_package.resolve()
    payload_dir = payload_dir.resolve()
    verify_package(source_package)
    source_sha256 = _sha256(source_package)
    if expected_source_sha256 and source_sha256 != expected_source_sha256.upper():
        raise ValueError("source package SHA-256 does not match the frozen value")
    if payload_dir.exists():
        raise FileExistsError(f"refusing to replace payload: {payload_dir}")

    payload_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="standalone_champion_", dir=payload_dir.parent))
    try:
        with zipfile.ZipFile(source_package) as archive:
            _safe_extract(archive, stage)
        validation = validate_payload(stage)
        stage.replace(payload_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    result = {
        "protocol": PROTOCOL,
        "public_score": PUBLIC_SCORE,
        "source_package": source_package.name,
        "source_sha256": source_sha256,
        "payload_dir": str(payload_dir),
        "historical_parent_package_required_after_materialization": False,
        **validation,
    }
    manifest_path = payload_dir.parent / "standalone_manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build(payload_dir: Path, output: Path) -> dict[str, object]:
    payload_dir = payload_dir.resolve()
    output = output.resolve()
    validation = validate_payload(payload_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(payload_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(payload_dir).as_posix())
    verify_package(output)
    return {
        "protocol": PROTOCOL,
        "public_score": PUBLIC_SCORE,
        "output": str(output),
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "historical_parent_package_required": False,
        **{key: validation[key] for key in ("file_count", "model_file_count")},
    }


def write_release_manifest(
    payload_dir: Path,
    result: dict[str, object],
) -> Path:
    """Persist the exact build and completed verification results."""

    payload_dir = payload_dir.resolve()
    validation = validate_payload(payload_dir)
    manifest = {
        "protocol": PROTOCOL,
        "public_score": PUBLIC_SCORE,
        "historical_parent_package_required": False,
        "payload": validation,
        "release": result,
    }
    manifest_path = payload_dir.parent / "standalone_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def smoke_test(package: Path, test_csv: Path) -> dict[str, object]:
    package = package.resolve()
    test_csv = test_csv.resolve()
    verify_package(package)
    expected = pd.read_csv(test_csv, usecols=["row_id"])
    with tempfile.TemporaryDirectory(prefix="standalone_champion_smoke_") as name:
        stage = Path(name)
        with zipfile.ZipFile(package) as archive:
            _safe_extract(archive, stage)
        (stage / "data").mkdir()
        shutil.copy2(test_csv, stage / "data" / "test.csv")
        completed = subprocess.run(
            [sys.executable, "script.py"],
            cwd=stage,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "standalone smoke test failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        prediction = pd.read_csv(stage / "output" / "submission.csv")
    if prediction["row_id"].tolist() != expected["row_id"].tolist():
        raise ValueError("standalone smoke test changed row order")
    probability = prediction["control_success"].to_numpy(np.float64)
    if not np.isfinite(probability).all():
        raise ValueError("standalone smoke test produced non-finite probabilities")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("standalone smoke test produced out-of-range probabilities")
    return {
        "rows": len(prediction),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "stdout": completed.stdout.strip(),
    }


def official_row_independence_test(
    package: Path,
    test_csv: Path,
    *,
    atol: float = 1e-12,
) -> dict[str, object]:
    """Apply the singleton-vs-full-batch test prescribed by DACON notice 417123."""

    package = package.resolve()
    test_csv = test_csv.resolve()
    verify_package(package)
    frame = pd.read_csv(test_csv, encoding="utf-8-sig", low_memory=False).reset_index(
        drop=True
    )
    if frame.empty:
        raise ValueError("row-independence test requires at least one row")
    if "row_id" not in frame.columns:
        raise ValueError("row-independence test requires row_id")
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("row-independence test requires unique, non-null row_id")

    with tempfile.TemporaryDirectory(prefix="standalone_champion_compliance_") as name:
        stage = Path(name)
        with zipfile.ZipFile(package) as archive:
            _safe_extract(archive, stage)
        spec = importlib.util.spec_from_file_location(
            "_standalone_champion_compliance", stage / "script.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not load standalone script.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict = getattr(module, "predict_dataframe", None)
        if not callable(predict):
            raise ValueError("standalone script.py has no callable predict_dataframe")

        def run(rows: pd.DataFrame) -> np.ndarray:
            # Each scenario represents a separately written/read test.csv.  A
            # fresh CSV has a 0-based RangeIndex, unlike an iloc slice whose
            # retained pandas index is not part of the competition inputs.
            inference_rows = rows.copy(deep=True).reset_index(drop=True)
            probability = np.asarray(predict(inference_rows), dtype=np.float64)
            if probability.shape != (len(rows),):
                raise ValueError("predict_dataframe returned an invalid shape")
            if not np.isfinite(probability).all():
                raise ValueError("row-independence test produced non-finite predictions")
            return probability

        full = run(frame)
        singleton = np.asarray(
            [run(frame.iloc[[index]])[0] for index in range(len(frame))],
            dtype=np.float64,
        )

        shuffled_frame = frame.sample(frac=1.0, random_state=417123)
        shuffled_prediction = run(shuffled_frame)
        shuffled_restored = np.empty_like(full)
        shuffled_restored[shuffled_frame.index.to_numpy()] = shuffled_prediction

        partitioned = np.empty_like(full)
        for indices in (np.arange(0, len(frame), 2), np.arange(1, len(frame), 2)):
            if len(indices):
                partitioned[indices] = run(frame.iloc[indices])

    singleton_max_abs_diff = float(np.max(np.abs(full - singleton)))
    shuffle_max_abs_diff = float(np.max(np.abs(full - shuffled_restored)))
    partition_max_abs_diff = float(np.max(np.abs(full - partitioned)))
    max_abs_diff = max(
        singleton_max_abs_diff,
        shuffle_max_abs_diff,
        partition_max_abs_diff,
    )
    if max_abs_diff > atol:
        raise ValueError(
            "DACON row-independence test failed: "
            f"max_abs_diff={max_abs_diff:.17g}, atol={atol:.17g}"
        )
    return {
        "notice": "https://dacon.io/competitions/official/236743/talkboard/417123",
        "rows": len(frame),
        "singleton_vs_full_max_abs_diff": singleton_max_abs_diff,
        "shuffled_vs_full_max_abs_diff": shuffle_max_abs_diff,
        "partitioned_vs_full_max_abs_diff": partition_max_abs_diff,
        "atol": atol,
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path)
    parser.add_argument(
        "--payload-dir",
        type=Path,
        default=Path("artifacts/standalone_champion_1158/payload"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/standalone_champion_1158/standalone_champion_1158.zip"),
    )
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--smoke-test-csv", type=Path)
    parser.add_argument("--compliance-test-csv", type=Path)
    args = parser.parse_args()

    payload_dir = args.payload_dir.resolve()
    result: dict[str, object] = {}
    if not payload_dir.exists():
        if args.source_package is None:
            raise FileNotFoundError(
                "payload does not exist; provide --source-package for one-time materialization"
            )
        result["materialize"] = materialize(
            args.source_package,
            payload_dir,
            expected_source_sha256=args.expected_source_sha256,
        )
    result["build"] = build(payload_dir, args.output)
    if args.smoke_test_csv is not None:
        result["smoke_test"] = smoke_test(args.output, args.smoke_test_csv)
    compliance_csv = args.compliance_test_csv or args.smoke_test_csv
    if compliance_csv is not None:
        result["official_row_independence"] = official_row_independence_test(
            args.output, compliance_csv
        )
    manifest_path = write_release_manifest(payload_dir, result)
    result["manifest"] = str(manifest_path)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

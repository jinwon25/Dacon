"""Audit a standalone DACON release ZIP without relying on repository models.

The audit extracts the archive into a temporary directory, runs only the
archive's own ``script.py`` and ``model/`` payload, and checks the official
row-independence constraint on a supplied valid test sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.package_standalone_champion import (
    _safe_extract,
    audit_row_independent_script,
    official_row_independence_test,
)
from src.package import verify_package


ID_COL = "row_id"
TARGET_COL = "control_success"
EXPECTED_ROOTS = {"model", "requirements.txt", "script.py"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def scale_proxy(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    """Repeat valid rows to benchmark official-scale inference safely."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    if frame.empty:
        raise ValueError("test sample must contain at least one row")

    positions = np.arange(rows) % len(frame)
    proxy = frame.iloc[positions].reset_index(drop=True).copy()
    proxy[ID_COL] = [f"standalone_scale_{index:06d}" for index in range(rows)]

    # Exercise both routing paths when the official schema exposes game_type.
    if "game_type" in proxy.columns:
        proxy.loc[proxy.index[1::2], "game_type"] = "F"
    return proxy


def run_archive(
    package: Path,
    test_frame: pd.DataFrame,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run the ZIP in isolation and validate output identity and probabilities."""
    with tempfile.TemporaryDirectory(prefix="standalone-release-audit-") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(package, "r") as archive:
            _safe_extract(archive, root)

        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        test_frame.to_csv(data_dir / "test.csv", index=False)
        pd.DataFrame(
            {ID_COL: test_frame[ID_COL], TARGET_COL: np.zeros(len(test_frame))}
        ).to_csv(data_dir / "sample_submission.csv", index=False)

        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "script.py"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        runtime = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                "standalone script failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        output_path = root / "output" / "submission.csv"
        if not output_path.exists():
            raise RuntimeError("script did not create output/submission.csv")
        output = pd.read_csv(output_path)
        if list(output.columns) != [ID_COL, TARGET_COL]:
            raise ValueError(f"unexpected output columns: {list(output.columns)}")
        if len(output) != len(test_frame):
            raise ValueError("output row count does not match test row count")
        if output[ID_COL].isna().any() or output[ID_COL].duplicated().any():
            raise ValueError("output row_id values must be non-null and unique")

        expected_ids = test_frame[ID_COL].astype(str).tolist()
        actual = output.assign(**{ID_COL: output[ID_COL].astype(str)}).set_index(ID_COL)
        actual = actual.reindex(expected_ids)
        if actual[TARGET_COL].isna().any():
            raise ValueError("output row_id values do not align with test.csv")
        predictions = actual[TARGET_COL].to_numpy(dtype=float)
        if not np.isfinite(predictions).all():
            raise ValueError("predictions contain non-finite values")
        if ((predictions < 0.0) | (predictions > 1.0)).any():
            raise ValueError("predictions must be within [0, 1]")

        return {
            "rows": int(len(predictions)),
            "runtime_seconds": float(runtime),
            "mean_prediction": float(predictions.mean()),
            "min_prediction": float(predictions.min()),
            "max_prediction": float(predictions.max()),
            "finite": True,
            "in_range": True,
        }


def audit_release(
    package: Path,
    test_csv: Path,
    *,
    sample_rows: int = 5,
    scale_rows: int = 0,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    package = package.resolve()
    test_csv = test_csv.resolve()
    verify_package(package)

    with zipfile.ZipFile(package, "r") as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
        roots = {name.split("/", 1)[0] for name in names if name and not name.endswith("/")}
        script_text = archive.read("script.py").decode("utf-8")
    if bad_member is not None:
        raise ValueError(f"CRC failure: {bad_member}")
    if roots != EXPECTED_ROOTS:
        raise ValueError(f"unexpected ZIP roots: {sorted(roots)}")
    if "from src" in script_text or "import src" in script_text:
        raise ValueError("standalone script imports repository src")

    static_audit = audit_row_independent_script(script_text)
    test_frame = pd.read_csv(test_csv)
    if ID_COL not in test_frame.columns:
        raise ValueError(f"test sample is missing {ID_COL}")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive")
    sample_frame = test_frame.head(sample_rows).copy()

    smoke = run_archive(
        package,
        sample_frame,
        timeout_seconds=timeout_seconds,
    )
    with tempfile.TemporaryDirectory(prefix="standalone-row-audit-") as temp_dir:
        row_audit_csv = Path(temp_dir) / "test.csv"
        sample_frame.to_csv(row_audit_csv, index=False)
        dynamic_audit = official_row_independence_test(package, row_audit_csv)

    result: dict[str, Any] = {
        "package": str(package),
        "sha256": sha256_file(package),
        "bytes": int(package.stat().st_size),
        "file_count": int(sum(not name.endswith("/") for name in names)),
        "roots": sorted(roots),
        "crc_ok": True,
        "static_row_independence": static_audit,
        "dynamic_row_independence": dynamic_audit,
        "sample_run": smoke,
    }
    if scale_rows:
        result["scale_run"] = run_archive(
            package,
            scale_proxy(test_frame, scale_rows),
            timeout_seconds=timeout_seconds,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--scale-rows", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_release(
        args.package,
        args.test_csv,
        sample_rows=args.sample_rows,
        scale_rows=args.scale_rows,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

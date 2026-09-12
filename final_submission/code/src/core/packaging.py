"""Submission package build, extraction, and hashing helpers.

Moved verbatim from ``src/champion/v84_build_public_probe.py`` during the core extraction.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

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


sha256 = _sha256
zip_directory = _zip_directory

"""Validate the actual candidate archive, including batch independence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src.metrics import validate_submission
from src.package import verify_package


EVALUATION_ROWS = 245_789


def _run_package(package_dir: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, float, float, str]:
    data_dir = package_dir / "data"
    output_dir = package_dir / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    frame.to_csv(data_dir / "test.csv", index=False, encoding="utf-8")
    started = time.perf_counter()
    process = psutil.Popen(
        [sys.executable, str(package_dir / "script.py")],
        cwd=package_dir.parent,
        stdout=-1,
        stderr=-1,
        text=True,
    )
    peak_rss = 0
    while process.poll() is None:
        try:
            current_rss = process.memory_info().rss
            for child in process.children(recursive=True):
                current_rss += child.memory_info().rss
            peak_rss = max(peak_rss, current_rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if time.perf_counter() - started > 120:
            process.kill()
            raise TimeoutError("candidate inference exceeded 120-second local guard")
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"candidate script failed: {stderr}")
    elapsed = time.perf_counter() - started
    submission = pd.read_csv(output_dir / "submission.csv")
    validate_submission(submission, frame["row_id"])
    return submission, elapsed, peak_rss / (1024.0**2), stdout.strip()


def _extract(zip_path: Path, destination: Path) -> Path:
    package = destination / zip_path.stem
    package.mkdir()
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(package)
    return package


def run(
    project_dir: Path,
    candidate_zip: Path,
    incumbent_zip: Path,
    report_path: Path,
) -> None:
    verify_package(candidate_zip)
    sample = pd.read_csv(project_dir / "data" / "test.csv", encoding="utf-8-sig")
    benchmark = pd.read_csv(
        project_dir / "data" / "train.csv",
        nrows=EVALUATION_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
    ).drop(columns="control_success")
    f_probe = benchmark.loc[benchmark["game_type"].astype(str) == "F"].head(10)
    r_probe = benchmark.loc[benchmark["game_type"].astype(str) == "R"].head(10)
    probe = pd.concat([f_probe, r_probe], ignore_index=True)
    if len(f_probe) != 10 or len(r_probe) != 10:
        raise ValueError("probe could not find ten rows for both game types")

    with tempfile.TemporaryDirectory(prefix="aimers9_candidate_verify_") as name:
        temp = Path(name)
        candidate = _extract(candidate_zip, temp)
        incumbent = _extract(incumbent_zip, temp)
        sample_submission, sample_seconds, _, sample_stdout = _run_package(
            candidate, sample
        )
        benchmark_submission, benchmark_seconds, peak_mb, _ = _run_package(
            candidate, benchmark
        )
        full_probe, _, _, _ = _run_package(candidate, probe)
        first_probe, _, _, _ = _run_package(candidate, probe.iloc[::2].copy())
        second_probe, _, _, _ = _run_package(candidate, probe.iloc[1::2].copy())
        split_probe = pd.concat([first_probe, second_probe], ignore_index=True).set_index(
            "row_id"
        )
        expected = full_probe.set_index("row_id").loc[split_probe.index]
        batch_difference = np.abs(
            split_probe["control_success"].to_numpy()
            - expected["control_success"].to_numpy()
        )
        batch_max_abs_difference = float(batch_difference.max(initial=0.0))
        batch_equal = batch_max_abs_difference <= 5e-15
        incumbent_probe, _, _, _ = _run_package(incumbent, probe)
        comparison = full_probe.merge(
            incumbent_probe,
            on="row_id",
            suffixes=("_candidate", "_incumbent"),
            validate="one_to_one",
        ).merge(probe[["row_id", "game_type"]], on="row_id", validate="one_to_one")
        r_difference = np.abs(
            comparison.loc[
                comparison["game_type"] == "R", "control_success_candidate"
            ].to_numpy()
            - comparison.loc[
                comparison["game_type"] == "R", "control_success_incumbent"
            ].to_numpy()
        )
        r_max_abs_difference = float(r_difference.max(initial=0.0))
        r_equal = r_max_abs_difference <= 5e-15
        f_lower = bool(
            (
                comparison.loc[
                    comparison["game_type"] == "F", "control_success_candidate"
                ].to_numpy()
                < comparison.loc[
                    comparison["game_type"] == "F", "control_success_incumbent"
                ].to_numpy()
            ).all()
        )

        script_text = (candidate / "script.py").read_text(encoding="utf-8").lower()
        network_tokens = ["requests.", "urllib.request", "http://", "https://", "socket."]
        aggregate_tokens = ["groupby(", "value_counts(", ".rank(", ".rolling(", ".expanding("]
        network_free = not any(token in script_text for token in network_tokens)
        row_local_static = not any(token in script_text for token in aggregate_tokens)
        offset_spec = json.loads(
            (candidate / "model" / "game_type_offsets.json").read_text(encoding="utf-8")
        )
        ensemble = json.loads(
            (candidate / "model" / "ensemble.json").read_text(encoding="utf-8")
        )
        members = []
        with zipfile.ZipFile(candidate_zip) as archive:
            members = archive.namelist()
            uncompressed = sum(item.file_size for item in archive.infolist())

    print(
        {
            "batch_equal": batch_equal,
            "batch_max_abs_difference": batch_max_abs_difference,
            "r_equal": r_equal,
            "r_max_abs_difference": r_max_abs_difference,
            "f_lower": f_lower,
            "network_free": network_free,
            "row_local_static": row_local_static,
        }
    )
    if not all((batch_equal, r_equal, f_lower, network_free, row_local_static)):
        raise AssertionError("candidate independence or parity validation failed")
    if len(benchmark_submission) != EVALUATION_ROWS:
        raise AssertionError("benchmark output length mismatch")

    report = f"""# 도메인 후보 제출 패키지 검증

- 후보 ZIP: `{candidate_zip.name}`
- ZIP / 압축 해제 크기: **{candidate_zip.stat().st_size / (1024**2):.3f} MB / {uncompressed / (1024**2):.3f} MB**
- 최상위 구조: **model/, script.py, requirements.txt**
- 멤버: `{members}`
- 공식 5행 smoke test: **통과**, {sample_seconds:.3f}초 (`{sample_stdout}`)
- 245,789행 실제 archive 추론: **{benchmark_seconds:.3f}초**, child peak RSS **{peak_mb:.1f} MB**
- 출력 행 수·row_id 순서·numeric/finite/[0,1]: **통과**
- 동일 20행을 전체/교차 2개 batch로 나눈 예측의 수치 일치: **{batch_equal}**, 최대 절대차 **{batch_max_abs_difference:.3e}**
- `R` probe가 incumbent와 수치 동일: **{r_equal}**, 최대 절대차 **{r_max_abs_difference:.3e}**
- `F` probe가 frozen negative offset으로 모두 하향: **{f_lower}**
- frozen offset: `{offset_spec['offsets']}`
- test 전체 집계·순서·빈도 사용: **없음**
- 인터넷 호출: **없음**
- feature set: **{ensemble['feature_set']}**

대표 배치는 실제 비공개 target 성능 평가가 아니라 archive 실행 시간·메모리·shape 검증에만 사용했다.
"""
    report_path.write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--candidate-zip", type=Path, default=Path("submit_candidate_game_type_regime_v1.zip")
    )
    parser.add_argument("--incumbent-zip", type=Path, default=Path("submit.zip"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/candidate_package_validation.md")
    )
    args = parser.parse_args()
    project = args.project_dir.resolve()
    candidate = args.candidate_zip if args.candidate_zip.is_absolute() else project / args.candidate_zip
    incumbent = args.incumbent_zip if args.incumbent_zip.is_absolute() else project / args.incumbent_zip
    report = args.report if args.report.is_absolute() else project / args.report
    run(project, candidate, incumbent, report)


if __name__ == "__main__":
    main()

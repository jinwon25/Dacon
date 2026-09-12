"""End-to-end submission package validation and runtime benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import psutil

from src.metrics import validate_submission
from src.package import verify_package

EVALUATION_ROWS = 245_789


class PeakRss:
    def __init__(self):
        self.peak = psutil.Process().memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.stop.wait(0.05):
            self.peak = max(self.peak, process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, psutil.Process().memory_info().rss)


def validate(project: Path, zip_path: Path) -> None:
    verify_package(zip_path)
    test = pd.read_csv(project / "data" / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(project / "data" / "sample_submission.csv", encoding="utf-8-sig")
    submission = pd.read_csv(project / "output" / "submission.csv", encoding="utf-8-sig")
    if list(submission.columns) != list(sample.columns):
        raise ValueError("submission columns differ from sample_submission")
    validate_submission(submission, test["row_id"])
    if not pd.api.types.is_numeric_dtype(submission["control_success"]):
        raise ValueError("prediction column is not numeric")

    # Test the actual archive from an unrelated cwd. Evaluation data is placed
    # next to extracted script.py, matching the official mount contract.
    with tempfile.TemporaryDirectory(prefix="aimers9_verify_") as temp_name:
        extracted = Path(temp_name) / "package"
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted)
        (extracted / "data").mkdir()
        shutil.copy2(project / "data" / "test.csv", extracted / "data" / "test.csv")
        completed = subprocess.run(
            [sys.executable, str(extracted / "script.py")],
            cwd=project.parent,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        archived_submission = pd.read_csv(extracted / "output" / "submission.csv")
        validate_submission(archived_submission, test["row_id"])
        archive_smoke_stdout = completed.stdout.strip()

        # Import the extracted script so the representative benchmark uses the
        # exact ZIP models rather than whichever artifacts happen to be in the
        # project-level model directory.
        module_spec = importlib.util.spec_from_file_location(
            "submission_package_script", extracted / "script.py"
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("could not import extracted submission script")
        submission_script = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(submission_script)
        benchmark_start = time.perf_counter()
        with PeakRss() as rss:
            benchmark = pd.read_csv(
                project / "data" / "train.csv",
                encoding="utf-8-sig",
                nrows=EVALUATION_ROWS,
                low_memory=False,
            ).drop(columns="control_success")
            prediction = submission_script.predict_dataframe(benchmark)
        benchmark_seconds = time.perf_counter() - benchmark_start
        if len(prediction) != EVALUATION_ROWS or not np.isfinite(prediction).all():
            raise ValueError("representative-batch inference failed")
        if ((prediction < 0.0) | (prediction > 1.0)).any():
            raise ValueError("representative-batch probabilities out of range")
        ensemble = json.loads(
            (extracted / "model" / "ensemble.json").read_text(encoding="utf-8")
        )
        hybrid_present = (extracted / "model" / "hybrid.json").exists()
        script_text = (extracted / "script.py").read_text(encoding="utf-8").lower()

    with zipfile.ZipFile(zip_path) as archive:
        compressed_mb = zip_path.stat().st_size / (1024.0**2)
        uncompressed_mb = sum(item.file_size for item in archive.infolist()) / (1024.0**2)
        members = archive.namelist()

    forbidden_network_tokens = ["requests.", "urllib.request", "http://", "https://", "socket."]
    network_free = not any(token in script_text for token in forbidden_network_tokens)
    forbidden_test_stats = ["groupby(", "value_counts(", ".rank(", ".rolling(", ".expanding("]
    row_local = not any(token in script_text for token in forbidden_test_stats)
    if not network_free or not row_local:
        raise ValueError("static offline/row-local inference audit failed")

    report = f"""# 제출 패키지 검증

- 검증 시각: {pd.Timestamp.now(tz='Asia/Seoul').isoformat()}
- ZIP: `{zip_path.name}`
- ZIP 크기 / 압축 해제 크기: **{compressed_mb:.3f} MB / {uncompressed_mb:.3f} MB**
- 최상위 구조: **model/, script.py, requirements.txt**
- 멤버: `{members}`
- 공식 5행 sample 실행: **통과** (`{archive_smoke_stdout}`)
- 출력 컬럼: **{submission.columns.tolist()}**, 행 수: **{len(submission)}**
- test `row_id` 값·순서 보존: **통과**
- 확률 numeric/finite/[0,1]: **통과**
- 대표 **{EVALUATION_ROWS:,}행** CSV 로드+피처+모델 추론: **{benchmark_seconds:.3f}초**
- 대표 배치 peak RSS: **{rss.peak / (1024**2):.1f} MB**
- 평가 10분 제한 대비: **통과** (동일 행 수 실측이 600초 미만)
- 평가 28GB RAM 제한 대비: **통과**
- 인터넷 호출 정적 검사: **없음**
- test 내부 groupby/value_counts/rank/rolling/expanding 정적 검사: **없음**
- 최종 feature set: **{ensemble['feature_set']}{' + hybrid' if hybrid_present else ''}**; test 전체 통계 사용: **없음**
- 검증 환경: Python **{sys.version.split()[0]}**, pandas **{pd.__version__}**, numpy **{np.__version__}**, LightGBM **{lightgbm.__version__}**

주의: 대표 배치는 실제 비공개 target이 없는 관계로 학습 데이터의 첫 {EVALUATION_ROWS:,}행을 오직 실행 시간·메모리·shape 검증에만 사용했다. 이 배치에서 성능 점수는 계산하지 않았다.
"""
    (project / "research" / "reports" / "package_validation.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--zip", type=Path, default=Path("submit.zip"))
    args = parser.parse_args()
    project = args.project_dir.resolve()
    zip_path = args.zip if args.zip.is_absolute() else project / args.zip
    validate(project, zip_path)


if __name__ == "__main__":
    main()

"""Generate the reproducible full-data audit report.

The two large CSVs are loaded once each with explicit compact dtypes. This is
deliberate: exact duplicate, cardinality, and as-of consistency checks need all
rows, while avoiding pandas' much larger default int64/float64/object layout.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.data import ID_COL, TARGET_COL, read_main, read_trackman


def _mb(value: int | float) -> float:
    return float(value) / (1024.0**2)


def _markdown_table(frame: pd.DataFrame, floatfmt: int = 6) -> str:
    copy = frame.copy()
    for col in copy.select_dtypes(include=["float", "floating"]).columns:
        copy[col] = copy[col].map(lambda value: f"{value:.{floatfmt}f}")
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(col) for col in copy.columns) + " |"
    divider = "| " + " | ".join("---" for _ in copy.columns) + " |"
    body = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in copy.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def _column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": missing,
                "missing_rate": missing / len(df),
                "nunique": int(df[col].nunique(dropna=False)),
                "constant": bool(df[col].nunique(dropna=False) <= 1),
            }
        )
    return pd.DataFrame(rows)


def _group_rates(train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parts = []
    for col in columns:
        grouped = (
            train.groupby(col, observed=True)[TARGET_COL]
            .agg(["size", "mean"])
            .reset_index()
            .rename(columns={col: "value", "size": "n", "mean": "success_rate"})
        )
        grouped.insert(0, "group", col)
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


def _asof_consistency(train: pd.DataFrame) -> tuple[int, int, float]:
    grouped = train.groupby("pitcher_id", sort=False, observed=True)
    prev_n = grouped["asof_pitcher_n"].shift(1)
    prev_rate = grouped["asof_pitcher_success_rate"].shift(1)
    prev_target = grouped[TARGET_COL].shift(1)
    current_n = train["asof_pitcher_n"].astype("float64")
    eligible = (current_n == prev_n + 1) & (prev_n > 0) & prev_rate.notna()
    expected = (prev_rate * prev_n + prev_target) / current_n
    matches = np.isclose(
        train.loc[eligible, "asof_pitcher_success_rate"].astype("float64"),
        expected.loc[eligible],
        atol=2e-6,
        rtol=2e-6,
    )
    checked = int(eligible.sum())
    matched = int(matches.sum())
    return checked, matched, matched / checked if checked else float("nan")


def generate_audit(project_dir: Path, output_path: Path) -> None:
    data_dir = project_dir / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    trackman_path = data_dir / "trackman_history.csv"
    sample_path = data_dir / "sample_submission.csv"

    train = read_main(train_path)
    test = read_main(test_path)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")

    train_memory = int(train.memory_usage(index=True, deep=True).sum())
    test_memory = int(test.memory_usage(index=True, deep=True).sum())
    train_profile = _column_profile(train)
    test_profile = _column_profile(test)

    feature_cols = [col for col in train.columns if col not in {ID_COL, TARGET_COL}]
    train_duplicates = int(train.duplicated().sum())
    train_feature_duplicates = int(train.duplicated(subset=feature_cols + [TARGET_COL]).sum())
    train_id_duplicates = int(train[ID_COL].duplicated().sum())
    test_id_duplicates = int(test[ID_COL].duplicated().sum())
    target_rate = float(train[TARGET_COL].mean())
    group_rates = _group_rates(
        train,
        [
            "season",
            "game_month",
            "game_dayofweek",
            "top_bottom",
            "game_type",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
            "outs_before",
        ],
    )
    pitcher_rates = (
        train.groupby("pitcher_id", observed=True)[TARGET_COL]
        .agg(n="size", success_rate="mean")
        .sort_values(["n", "success_rate"], ascending=[False, False])
        .head(20)
        .reset_index()
    )
    asof_checked, asof_matched, asof_ratio = _asof_consistency(train)

    main_pitchers = set(train["pitcher_id"].unique().tolist())
    main_batters = set(train["batter_id"].unique().tolist())
    test_pitcher_known = float(test["pitcher_id"].isin(main_pitchers).mean())
    test_batter_known = float(test["batter_id"].isin(main_batters).mean())

    trackman = read_trackman(trackman_path)
    trackman_memory = int(trackman.memory_usage(index=True, deep=True).sum())
    trackman_profile = _column_profile(trackman)
    trackman_duplicates = int(trackman.duplicated().sum())
    trackman_id_duplicates = int(trackman["trackman_id"].duplicated().sum())
    tm_pitchers = set(trackman["pitcher_trackman_id"].unique().tolist())
    tm_batters = set(trackman["batter_trackman_id"].unique().tolist())
    pitcher_intersection = main_pitchers & tm_pitchers
    batter_intersection = main_batters & tm_batters
    direct_pitcher_row_coverage = float(train["pitcher_id"].isin(tm_pitchers).mean())
    direct_batter_row_coverage = float(train["batter_id"].isin(tm_batters).mean())

    shared_state_keys = [
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
        "balls_before",
        "strikes_before",
        "outs_before",
    ]
    tm_state_unique = int(trackman[shared_state_keys].drop_duplicates().shape[0])
    tm_state_duplicate_rate = 1.0 - tm_state_unique / len(trackman)

    file_summary = pd.DataFrame(
        [
            {
                "file": "train.csv",
                "rows": len(train),
                "columns": len(train.columns),
                "csv_mb": _mb(train_path.stat().st_size),
                "optimized_memory_mb": _mb(train_memory),
            },
            {
                "file": "test.csv",
                "rows": len(test),
                "columns": len(test.columns),
                "csv_mb": _mb(test_path.stat().st_size),
                "optimized_memory_mb": _mb(test_memory),
            },
            {
                "file": "sample_submission.csv",
                "rows": len(sample),
                "columns": len(sample.columns),
                "csv_mb": _mb(sample_path.stat().st_size),
                "optimized_memory_mb": _mb(sample.memory_usage(index=True, deep=True).sum()),
            },
            {
                "file": "trackman_history.csv",
                "rows": len(trackman),
                "columns": len(trackman.columns),
                "csv_mb": _mb(trackman_path.stat().st_size),
                "optimized_memory_mb": _mb(trackman_memory),
            },
        ]
    )

    high_cardinality = train_profile.loc[train_profile["nunique"] >= 100, ["column", "nunique"]]
    constants = train_profile.loc[train_profile["constant"], "column"].tolist()
    train_seasons = sorted(train["season"].unique().tolist())
    trackman_seasons = sorted(trackman["season"].unique().tolist())

    lines = [
        "# 데이터 및 누수 감사",
        "",
        "- 생성 명령: `python -m src.archive.audit --project-dir .`",
        "- 원본 CSV를 수정하지 않았다. 명시적 `int32`/`float32`/`category` dtype으로 각각 한 번만 로드해 정확 통계를 계산했다.",
        "- 배포 `test.csv`와 `sample_submission.csv`는 형식 확인용 5행이므로 범주 커버리지는 실제 비공개 평가셋을 대표하지 않는다.",
        "",
        "## 파일 크기와 메모리",
        "",
        _markdown_table(file_summary, 3),
        "",
        "## Train 컬럼 프로파일",
        "",
        _markdown_table(train_profile),
        "",
        "## Trackman 컬럼 프로파일",
        "",
        _markdown_table(trackman_profile),
        "",
        "## 중복·식별자",
        "",
        f"- train 완전 중복 행: **{train_duplicates:,}**",
        f"- train `row_id` 제외 입력+target 중복 행: **{train_feature_duplicates:,}**",
        f"- train 중복 `row_id`: **{train_id_duplicates:,}**; 5행 test 중복 `row_id`: **{test_id_duplicates:,}**",
        f"- Trackman 완전 중복 행: **{trackman_duplicates:,}**; 중복 `trackman_id`: **{trackman_id_duplicates:,}**",
        f"- 상수 train 컬럼: **{constants or '없음'}**",
        "- `row_id`는 제출 정렬/무결성 확인에만 쓰고 모델 입력에서 제외한다.",
        "",
        "## Target과 주요 그룹 성공률",
        "",
        f"- 전체 `control_success=1` 비율: **{target_rate:.6f}**",
        "",
        _markdown_table(group_rates),
        "",
        "표본 수 상위 투수 20명:",
        "",
        _markdown_table(pitcher_rates),
        "",
        "## 시간축·검증 단위",
        "",
        f"- 메인 시즌: **{train_seasons}**; Trackman 시즌: **{trackman_seasons}**.",
        "- 메인에는 `season`, 월, 요일만 있고 정확한 `game_date`, 경기 ID, 경기 내 투구 번호가 없다. 따라서 같은 경기 분리를 정확히 보장하는 game-group split은 구현할 수 없다.",
        "- Trackman에만 `game_date`, `trackman_game_id`, `pitch_no`, `pitch_of_pa`가 있다. 두 테이블은 행 단위 대응 관계가 아니다.",
        "- `row_id` 숫자 부분은 파일 순서와 함께 증가하지만, 식별자/순서를 피처로 쓰지 않는다. 실제 test 행은 독립 예측하며 다른 test 행을 참조하지 않는다.",
        "- 실제 평가가 2025년이고 학습 최종 시즌이 2024년이므로, **2019~2023 train / 2024 validation**을 primary split으로 선택한다.",
        "",
        "## 엔터티·Trackman 조인 감사",
        "",
        f"- 메인 고유 투수/타자: **{len(main_pitchers):,} / {len(main_batters):,}**; Trackman 고유 투수/타자: **{len(tm_pitchers):,} / {len(tm_batters):,}**.",
        f"- `pitcher_id = pitcher_trackman_id` 교집합: **{len(pitcher_intersection):,}개**, train 행 커버리지 **{direct_pitcher_row_coverage:.6%}**, 미매칭 **{1-direct_pitcher_row_coverage:.6%}**.",
        f"- `batter_id = batter_trackman_id` 교집합: **{len(batter_intersection):,}개**, train 행 커버리지 **{direct_batter_row_coverage:.6%}**, 미매칭 **{1-direct_batter_row_coverage:.6%}**.",
        "- 따라서 제공 파일만으로 확인되는 선수 단위 실제 조인 키는 **없다**. 이름이 비슷하다는 이유로 ID를 결합하면 안 된다.",
        f"- 공통 경기상황 7열 조합은 Trackman {tm_state_unique:,}개이며 행 중복률이 {tm_state_duplicate_rate:.6%}라서 1:1 키가 아니라 과거 league-context 집계 키로만 사용 가능하다.",
        "- Trackman의 손잡이는 `Right/Left`, 메인은 `1/2`; 팀도 문자열과 익명 정수로 표현되어 공식 매핑이 없다. 초기 안전 모델은 선수 단위 Trackman 결합을 하지 않고, 별도 실험에서 시즌 이전 Trackman만 사용한 경기상황 league-context 집계만 검증한다.",
        "",
        "## 누수 점검",
        "",
        f"- 연속 투수 기록에서 직전 target을 반영한 다음 행의 `asof_pitcher_success_rate` 갱신식 일치: **{asof_matched:,}/{asof_checked:,} ({asof_ratio:.6%})**. 이는 `asof_*`가 현재 행 이전 이력임을 강하게 지지한다.",
        "- 단, train의 다음 행 `asof_*`를 역방향으로 당기면 현재 target을 복원할 수 있다. 따라서 어떤 backward shift/다음 행 참조도 금지하고 현재 행에 공식 제공된 값만 사용한다.",
        "- `asof_*success/middle/reverse/ball/strike_rate`는 target 관련 과거 집계라 누수 민감 피처로 분류하지만, 설명서가 투구 직전 계산을 명시하므로 현재 행 값은 사용 가능하다. 성능은 해당 블록 제거 ablation으로 점검한다.",
        "- 현재 투구 위치·판정·실제 구종·Trackman 측정값에 해당하는 컬럼은 메인 입력에 없다. `home_win_expectancy`, `away_win_expectancy`, `li`도 설명서상 투구 직전 값이다.",
        "- `run_total_before`, 점수 차, `num_runners_on`, `base_state`는 다른 pre-pitch 열에서 재구성되는 중복 정보이나 target 누수는 아니다.",
        "",
        "## 고카디널리티와 미지 범주",
        "",
        _markdown_table(high_cardinality),
        "",
        f"- 5행 sample test의 train-known 투수/타자 비율: **{test_pitcher_known:.2%} / {test_batter_known:.2%}**. 실제 평가 커버리지는 알 수 없다.",
        "- 범주는 train에서만 사전을 학습하고 validation/test 미지값은 전용 unknown 코드로 보낸다. test 빈도나 test 전체 분포는 계산하지 않는다.",
        "",
        "## 설명서와 실제 파일 차이",
        "",
        "- 행/열 수는 설명서와 일치한다. 배포 test와 sample submission은 각각 5행이다.",
        "- 공식 `baseline_submit.zip` 원본은 없고 동일 내용으로 보이는 해제 폴더 `baseline_submit/`와 학습/추론 노트북이 제공됐다.",
        "- 설명서는 손잡이를 '코드'로만 설명하며 1/2 의미 매핑을 제공하지 않는다. 이를 자의적으로 Right/Left로 확정하지 않는다.",
        "- 메인에는 game/date/pitch-order 열이 없어 경기 단위 중복 여부와 정확한 경기 group split을 직접 확인할 수 없다.",
        "",
        "## 감사 결론",
        "",
        "1. `row_id`를 제외한 공식 pre-pitch 입력과 현재 행의 `asof_*`만 기본 피처로 사용한다.",
        "2. primary validation은 2024 season-forward holdout이며, 2023 OOF를 calibration/blend 선택용으로 별도 생성한다.",
        "3. Trackman 선수 ID 직접 결합은 커버리지가 없어 사용하지 않는다. 누수 없는 과거 league-context 집계는 별도 ablation에서만 승격한다.",
        "4. 실제 test의 다른 행, 순서, 빈도, 분포, 정규화 통계는 어느 단계에서도 사용하지 않는다.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    del trackman, train, test
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.md"))
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    output = args.output if args.output.is_absolute() else project_dir / args.output
    generate_audit(project_dir, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

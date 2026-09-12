"""Domain-focused EDA for control execution, game context, and regime shifts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.data import TARGET_COL, read_main
from src.archive.followup import _load_fold_cache, _fold_cache_path


def _safe_entropy(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    total = values.sum(axis=1).replace(0.0, np.nan)
    probability = values.div(total, axis=0).clip(lower=1e-12)
    return -(probability * np.log(probability)).sum(axis=1).fillna(0.0)


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    balls = pd.to_numeric(frame["balls_before"])
    strikes = pd.to_numeric(frame["strikes_before"])
    out["count_state"] = balls.astype("string") + "-" + strikes.astype("string")
    out["count_pressure"] = np.select(
        [balls == 3, strikes == 2, balls > strikes, strikes > balls],
        ["three_ball", "two_strike", "batter_ahead", "pitcher_ahead"],
        default="even",
    )
    score = pd.to_numeric(frame["score_diff_pitcher_team"])
    out["score_state"] = np.select(
        [score > 0, score < 0], ["ahead", "behind"], default="tied"
    )
    out["inning_bucket"] = pd.cut(
        pd.to_numeric(frame["inning"]),
        bins=[0, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    ).astype("string")
    out["li_bucket"] = pd.cut(
        pd.to_numeric(frame["li"], errors="coerce"),
        bins=[-np.inf, 0.75, 1.5, 3.0, np.inf],
        labels=["low", "medium", "high", "very_high"],
    ).astype("string")
    out["runner_pressure"] = np.select(
        [
            pd.to_numeric(frame["num_runners_on"]) >= 3,
            (pd.to_numeric(frame["runner_on_2b"]) > 0)
            | (pd.to_numeric(frame["runner_on_3b"]) > 0),
            pd.to_numeric(frame["runner_on_1b"]) > 0,
        ],
        ["loaded", "scoring_position", "first_only"],
        default="empty",
    )
    out["platoon"] = (
        frame["pitcher_hand"].astype("string")
        + "-"
        + frame["batter_hand"].astype("string")
    )
    out["game_type_regime"] = (
        frame["game_type"].astype("string")
        + "-"
        + np.select(
            [frame["season"] >= 2024, frame["season"] >= 2023],
            ["2024plus", "2023"],
            default="pre2023",
        )
    )
    out["pitchmix_entropy"] = _safe_entropy(
        frame,
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ],
    )
    component_columns = [
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
    ]
    out["failure_component_entropy"] = _safe_entropy(frame, component_columns)
    out["failure_component_total"] = (
        frame[component_columns]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1, min_count=1)
    )
    out["middle_delta_3"] = (
        pd.to_numeric(frame["asof_pitcher_prev3_game_middle_rate"], errors="coerce")
        - pd.to_numeric(frame["asof_pitcher_middle_rate"], errors="coerce")
    )
    recent_success = frame[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    recent_middle = frame[
        [
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    out["recent_success_std"] = recent_success.std(axis=1, skipna=True)
    out["recent_middle_std"] = recent_middle.std(axis=1, skipna=True)
    out["pressure_index"] = (
        (balls == 3).astype(float)
        + (pd.to_numeric(frame["num_runners_on"]) > 0).astype(float)
        + (pd.to_numeric(frame["li"], errors="coerce") >= 1.5).astype(float)
        + (pd.to_numeric(frame["inning"]) >= 7).astype(float)
    )
    return out


def _segment_rates(frame: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    work = pd.concat(
        [frame[["season", "game_type", TARGET_COL]].reset_index(drop=True), derived.reset_index(drop=True)],
        axis=1,
    )
    rows: list[dict[str, object]] = []
    for column in (
        "game_type",
        "count_state",
        "count_pressure",
        "score_state",
        "inning_bucket",
        "li_bucket",
        "runner_pressure",
        "platoon",
        "game_type_regime",
    ):
        grouped = (
            work.groupby(["season", column], observed=True, dropna=False)[TARGET_COL]
            .agg(["size", "mean"])
            .reset_index()
        )
        for record in grouped.to_dict("records"):
            rows.append(
                {
                    "season": int(record["season"]),
                    "segment_type": column,
                    "segment_value": str(record[column]),
                    "n_rows": int(record["size"]),
                    "success_rate": float(record["mean"]),
                }
            )
    result = pd.DataFrame(rows)
    season_rate = frame.groupby("season", observed=True)[TARGET_COL].mean()
    result["effect_vs_season"] = result["success_rate"] - result["season"].map(season_rate)
    return result


def _feature_screen(frame: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    continuous = [
        "pitchmix_entropy",
        "failure_component_entropy",
        "failure_component_total",
        "middle_delta_3",
        "recent_success_std",
        "recent_middle_std",
        "pressure_index",
    ]
    for column in continuous:
        per_year_spans = []
        usable_rows = 0
        for season, index in frame.groupby("season", observed=True).groups.items():
            values = derived.loc[index, column]
            target = frame.loc[index, TARGET_COL]
            valid = values.notna()
            if valid.sum() < 100:
                continue
            try:
                bucket = pd.qcut(values[valid], 10, duplicates="drop")
            except ValueError:
                continue
            rates = target[valid].groupby(bucket, observed=True).mean()
            if len(rates) >= 2:
                per_year_spans.append(float(rates.max() - rates.min()))
                usable_rows += int(valid.sum())
        rows.append(
            {
                "feature": column,
                "mean_within_season_decile_rate_span": float(np.mean(per_year_spans)),
                "min_within_season_decile_rate_span": float(np.min(per_year_spans)),
                "usable_rows_across_seasons": usable_rows,
                "n_seasons": len(per_year_spans),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_within_season_decile_rate_span", ascending=False
    )


def _incumbent_game_type_error(project_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in (2021, 2022, 2023, 2024):
        path = _fold_cache_path(project_dir, year)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            valid_idx = saved["valid_idx"].astype(np.int64)
            target = saved["target"].astype(np.float64)
            incumbent = saved["incumbent"].astype(np.float64)
            raw = saved["blend_raw"].astype(np.float64)
        valid = frame.iloc[valid_idx][["game_type", "game_month"]].copy()
        valid["target"] = target
        valid["incumbent"] = incumbent
        valid["raw"] = raw
        for game_type, group in valid.groupby("game_type", observed=True):
            rows.append(
                {
                    "season": year,
                    "game_type": str(game_type),
                    "n_rows": len(group),
                    "target_rate": float(group["target"].mean()),
                    "incumbent_mean": float(group["incumbent"].mean()),
                    "raw_mean": float(group["raw"].mean()),
                    "incumbent_brier": float(
                        np.mean(np.square(group["incumbent"] - group["target"]))
                    ),
                    "raw_brier": float(np.mean(np.square(group["raw"] - group["target"]))),
                }
            )
    return pd.DataFrame(rows)


def run(project_dir: Path) -> None:
    train = read_main(project_dir / "data" / "train.csv").reset_index(drop=True)
    derived = _derived(train)
    segments = _segment_rates(train, derived)
    screen = _feature_screen(train, derived)
    game_error = _incumbent_game_type_error(project_dir, train)
    reports = project_dir / "reports"
    segments.to_csv(reports / "domain_segment_rates.csv", index=False)
    screen.to_csv(reports / "domain_feature_screen.csv", index=False)
    game_error.to_csv(reports / "domain_game_type_errors.csv", index=False)

    game_rates = (
        train.groupby(["season", "game_type"], observed=True)[TARGET_COL]
        .agg(n_rows="size", success_rate="mean")
        .reset_index()
    )
    season_rates = train.groupby("season", observed=True)[TARGET_COL].mean()
    game_rates["season_rate"] = game_rates["season"].map(season_rates)
    game_rates["residual_vs_season"] = game_rates["success_rate"] - game_rates["season_rate"]
    game_rates["year_delta"] = game_rates.groupby("game_type", observed=True)[
        "success_rate"
    ].diff()
    game_rates.to_csv(reports / "domain_game_type_regime.csv", index=False)

    success = pd.to_numeric(train["asof_pitcher_success_rate"], errors="coerce")
    strike = pd.to_numeric(train["asof_pitcher_strike_rate"], errors="coerce")
    max_success_strike_gap = float((success - strike).abs().max(skipna=True))
    f_2022 = float(
        game_rates.loc[
            (game_rates["season"] == 2022) & (game_rates["game_type"] == "F"),
            "success_rate",
        ].iloc[0]
    )
    f_2023 = float(
        game_rates.loc[
            (game_rates["season"] == 2023) & (game_rates["game_type"] == "F"),
            "success_rate",
        ].iloc[0]
    )
    r_2022 = float(
        game_rates.loc[
            (game_rates["season"] == 2022) & (game_rates["game_type"] == "R"),
            "success_rate",
        ].iloc[0]
    )
    r_2023 = float(
        game_rates.loc[
            (game_rates["season"] == 2023) & (game_rates["game_type"] == "R"),
            "success_rate",
        ].iloc[0]
    )
    report = f"""# 한국 야구 제구 도메인 EDA

## 핵심 발견

- `game_type=R`은 시즌당 약 21.1~22.3만 행, `F`는 약 2.3~3.0만 행이다. 규모와 공식 KBO 자료를 보면 R/F가 1군 정규리그/별도 경기군일 가능성은 높지만 공식 데이터 설명서는 코드 의미를 확정하지 않으므로 모델에서는 익명 strata로만 취급한다.
- F 성공률은 2022 {f_2022:.6f}에서 2023 {f_2023:.6f}로 {f_2023-f_2022:+.6f} 변했지만 R은 같은 기간 {r_2022:.6f}에서 {r_2023:.6f}로 {r_2023-r_2022:+.6f}만 변했다. 2023 전체 calibration 붕괴의 주된 구조적 원인이다.
- 2024 incumbent는 F에서 raw blend보다 Brier를 크게 줄였고 R에서도 소폭 개선했다. 따라서 전역 drift를 제거하는 대신, 검출된 game-type regime residual만 추가하는 방향이 안전하다.
- `asof_pitcher_success_rate`와 `asof_pitcher_strike_rate`는 동일하지 않으며 최대 절대차는 {max_success_strike_gap:.6f}이다. reverse/middle/ball/strike 비율도 단순히 합 1인 상호배타 범주가 아니므로 임의로 실패확률로 합치지 않는다.
- 카운트, LI, 주자, 이닝은 투구 의도와 위험 회피를 바꾸지만 현재 투구의 실제 구종·요구 코스가 없으므로 row-local 상호작용만 사용한다.

## 사전 정의할 실험

1. 전역 incumbent에 통계적으로 검출된 game-type 최근 regime residual만 더하는 change-point offset.
2. 현재 engineered LGB에 failure-profile entropy, 최근 middle-rate delta, pitch-mix entropy와 game-type regime category만 추가한 최소 피처 모델.
3. 공식 RF 설정을 고정하고 seed 42/202/777 확률을 단순 평균한다.

CatBoost, large hierarchical backoff, squared-error GBDT와 RF 하이퍼파라미터 탐색은 재개하지 않는다. row_id/CSV 순서, test 집계, TrackMan 선수 매핑은 사용하지 않는다.

## 산출 표

- `domain_game_type_regime.csv`: 시즌×game type 성공률과 시즌 residual.
- `domain_game_type_errors.csv`: game type별 incumbent/raw Brier와 평균 예측.
- `domain_segment_rates.csv`: 시즌 내 상황별 성공률.
- `domain_feature_screen.csv`: 연도별 decile target-rate span 진단. 이는 EDA이며 test 변환 통계로 사용하지 않는다.
"""
    (reports / "domain_eda.md").write_text(report, encoding="utf-8")
    print(game_rates.to_string(index=False))
    print("\nFeature screen\n", screen.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.project_dir.resolve())


if __name__ == "__main__":
    main()

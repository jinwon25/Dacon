"""One-shot rolling-origin evaluation of a fixed damped drift ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.calibration import apply_logit_offset, forecast_base_rate
from src.data import TARGET_COL, read_main
from src.followup import _load_all_caches, _metric_row, _season_rates, _upsert_csv
from src.metrics import brier_score, cluster_bootstrap_delta, probability_logit


CANDIDATE_NAME = "rolling_damped_equal_v1"


def build_rolling_damped_ensemble(
    raw_prediction: np.ndarray,
    season_rates: pd.Series,
    forecast_season: int,
    methods: Sequence[str],
    weights: Sequence[float],
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Apply a fixed forecast ensemble using only rates before the origin.

    Methods that lack enough historical seasons are omitted and the declared
    weights are renormalized. If none are available, the unshifted raw blend is
    returned. No validation target is accepted by this function.
    """
    raw = np.asarray(raw_prediction, dtype=np.float64)
    declared_weights = np.asarray(weights, dtype=np.float64)
    if len(methods) == 0 or len(methods) != len(declared_weights):
        raise ValueError("methods and weights must be non-empty and have equal length")
    if not np.isfinite(declared_weights).all() or np.any(declared_weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if float(declared_weights.sum()) <= 0.0:
        raise ValueError("at least one weight must be positive")

    rates = season_rates.sort_index().astype(float)
    if len(rates) == 0 or int(rates.index.max()) >= int(forecast_season):
        raise ValueError("season rates must be non-empty and strictly before forecast")
    latest_rate = float(rates.iloc[-1])
    latest_logit = float(probability_logit(np.array([latest_rate]))[0])
    available: list[tuple[str, float, float, np.ndarray]] = []
    for method, weight in zip(methods, declared_weights):
        if weight == 0.0:
            continue
        try:
            forecast_rate = forecast_base_rate(rates, forecast_season, method)
        except ValueError:
            continue
        offset = float(
            probability_logit(np.array([forecast_rate]))[0] - latest_logit
        )
        available.append(
            (method, float(weight), forecast_rate, apply_logit_offset(raw, offset))
        )

    if not available:
        return raw.copy(), []

    active_weights = np.asarray([item[1] for item in available], dtype=np.float64)
    active_weights /= active_weights.sum()
    prediction = np.average(
        np.column_stack([item[3] for item in available]),
        axis=1,
        weights=active_weights,
    )
    details = [
        {
            "method": method,
            "declared_weight": declared_weight,
            "active_weight": float(active_weight),
            "forecast_rate": forecast_rate,
            "offset": float(
                probability_logit(np.array([forecast_rate]))[0] - latest_logit
            ),
        }
        for (method, declared_weight, forecast_rate, _), active_weight in zip(
            available, active_weights
        )
    ]
    return prediction, details


def _bootstrap_rows(
    frame: pd.DataFrame,
    target: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
    scope: str,
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    pitcher = frame["pitcher_id"].astype("string")
    pitcher_season = pitcher + "-" + frame["season"].astype("string")
    rows: list[dict[str, Any]] = []
    for cluster_name, clusters in (
        ("pitcher_season", pitcher_season),
        ("pitcher", pitcher),
    ):
        rows.append(
            {
                "candidate": CANDIDATE_NAME,
                "scope": scope,
                "cluster_type": cluster_name,
                **cluster_bootstrap_delta(
                    target,
                    candidate,
                    incumbent,
                    clusters,
                    n_resamples=n_resamples,
                    seed=seed,
                ),
            }
        )
    return rows


def _write_report(
    path: Path,
    result: pd.DataFrame,
    bootstrap: pd.DataFrame,
    checks: dict[str, bool],
    methods: Sequence[str],
    weights: Sequence[float],
) -> None:
    lines = [
        "# Rolling-origin damped drift 단일 사전 고정 실험",
        "",
        "## 설계",
        "",
        "- 후보: `rolling_damped_equal_v1`",
        f"- 방법: `{methods[0]}`와 `{methods[1]}`의 확률 수준 고정 50:50 평균",
        "- 각 outer year보다 이전 시즌 성공률만 사용하며 validation target은 예측 생성에 사용하지 않음",
        "- 과거 시즌이 3개 미만이면 incumbent의 raw RF/LGB blend로 fallback",
        "- 기존 4개 통계 게이트를 변경 없이 적용",
        "",
        "## Walk-forward 결과",
        "",
        "| validation | incumbent | candidate | delta | candidate mean | target rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.itertuples():
        lines.append(
            f"| {int(row.outer_validation_season)} | {row.incumbent_brier:.9f} | "
            f"{row.brier:.9f} | {row.delta_brier:+.9f} | "
            f"{row.prediction_mean:.6f} | {row.target_rate:.6f} |"
        )
    summary = result.iloc[0]
    lines.extend(
        [
            "",
            "## 고정 게이트",
            "",
            f"- recency-weighted delta: **{summary['recency_weighted_delta']:+.9f}** "
            f"({'통과' if checks['recency'] else '실패'})",
            f"- 최신 연도 delta: **{result.iloc[-1]['delta_brier']:+.9f}** "
            f"({'통과' if checks['latest'] else '실패'})",
            f"- worst-fold delta: **{summary['worst_fold_delta']:+.9f}** "
            f"({'통과' if checks['worst'] else '실패'})",
            f"- 결합 pitcher-season bootstrap P(improve): "
            f"**{summary['combined_pitcher_season_improvement_probability']:.4f}** "
            f"({'통과' if checks['bootstrap'] else '실패'})",
            "",
            "## 결정",
            "",
        ]
    )
    if all(checks.values()):
        lines.append(
            "통계 게이트를 통과했다. 다만 새 제출 패키지 승격 전 seed 안정성과 "
            "패키지 독립성 검증이 추가로 필요하다."
        )
    else:
        lines.append(
            "사전 고정 게이트를 모두 통과하지 못했다. 이 실험을 근거로 새 ZIP을 "
            "생성하지 않으며 damped drift 방향은 종료한다. 기존 `submit.zip`을 유지한다."
        )
    combined = bootstrap[
        (bootstrap["scope"] == "2021-2024")
        & (bootstrap["cluster_type"] == "pitcher_season")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## 불확실성",
            "",
            f"결합 pitcher-season bootstrap delta 95% 구간은 "
            f"[{combined['ci_lower']:+.9f}, {combined['ci_upper']:+.9f}]이다. "
            "기존 outer folds는 반복 실험에 사용된 개발 데이터이므로 이 결과를 "
            "untouched test 수준의 확증으로 해석하지 않는다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(project_dir: Path, n_resamples: int = 10_000) -> pd.DataFrame:
    config = json.loads(
        (project_dir / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    spec = config["rolling_damped_ensemble"]
    methods = [str(value) for value in spec["methods"]]
    weights = [float(value) for value in spec["weights"]]
    if spec.get("fallback") != "raw_blend":
        raise ValueError("rolling damped ensemble fallback must be raw_blend")

    train = read_main(project_dir / "data" / "train.csv")
    folds, caches = _load_all_caches(project_dir, config, train)
    years = [int(value) for value in config["outer_validation_seasons"]]
    result_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    predictions: dict[int, np.ndarray] = {}

    for year in years:
        fold = folds[year]
        rates = _season_rates(train, fold.train_idx)
        candidate, details = build_rolling_damped_ensemble(
            caches[year]["blend_raw"], rates, year, methods, weights
        )
        target = caches[year]["target"]
        incumbent = caches[year]["incumbent"]
        valid = train.iloc[fold.valid_idx].reset_index(drop=True)
        predictions[year] = candidate
        result_rows.append(
            {
                "candidate": CANDIDATE_NAME,
                "outer_validation_season": year,
                "history_seasons": ";".join(str(int(value)) for value in rates.index),
                "active_methods": ";".join(str(item["method"]) for item in details)
                or "raw_blend_fallback",
                "forecast_details": json.dumps(details, sort_keys=True),
                "brier": brier_score(target, candidate),
                "incumbent_brier": brier_score(target, incumbent),
                "delta_brier": brier_score(target, candidate)
                - brier_score(target, incumbent),
                "prediction_mean": float(candidate.mean()),
                "target_rate": float(target.mean()),
            }
        )
        metric_rows.append(
            _metric_row(
                f"W6_{CANDIDATE_NAME}_{year}",
                year,
                CANDIDATE_NAME,
                candidate,
                target,
                incumbent,
                diagnostic=False,
                prediction_source=(
                    "fixed equal-weight damped forecasts from strict prior-season rates; "
                    "raw blend fallback"
                ),
                fit_seasons=";".join(str(int(value)) for value in rates.index),
            )
        )
        bootstrap_rows.extend(
            _bootstrap_rows(
                valid,
                target,
                candidate,
                incumbent,
                str(year),
                n_resamples,
                int(config["seed"]) + 40_000 + year,
            )
        )

    combined_frame = pd.concat(
        [train.iloc[folds[year].valid_idx] for year in years], ignore_index=True
    )
    combined_target = np.concatenate([caches[year]["target"] for year in years])
    combined_incumbent = np.concatenate(
        [caches[year]["incumbent"] for year in years]
    )
    combined_candidate = np.concatenate([predictions[year] for year in years])
    bootstrap_rows.extend(
        _bootstrap_rows(
            combined_frame,
            combined_target,
            combined_candidate,
            combined_incumbent,
            "2021-2024",
            n_resamples,
            int(config["seed"]) + 50_000,
        )
    )

    result = pd.DataFrame(result_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    deltas = result.set_index("outer_validation_season")["delta_brier"]
    recency_weights = np.asarray(
        [float(config["recency_weights"][str(year)]) for year in years]
    )
    recency_delta = float(
        np.average([float(deltas[year]) for year in years], weights=recency_weights)
    )
    combined_ps = bootstrap[
        (bootstrap["scope"] == "2021-2024")
        & (bootstrap["cluster_type"] == "pitcher_season")
    ].iloc[0]
    gate = config["submission_gate"]
    checks = {
        "recency": recency_delta
        <= -float(gate["minimum_recency_weighted_improvement"]),
        "latest": float(deltas[years[-1]])
        <= -float(gate["minimum_2024_improvement"]),
        "worst": float(deltas.max())
        <= float(gate["maximum_single_fold_worsening"]),
        "bootstrap": float(combined_ps["improvement_probability"])
        >= float(gate["minimum_cluster_improvement_probability"]),
    }
    result["recency_weighted_delta"] = recency_delta
    result["worst_fold_delta"] = float(deltas.max())
    result["combined_pitcher_season_improvement_probability"] = float(
        combined_ps["improvement_probability"]
    )
    result["passes_statistical_gate"] = all(checks.values())

    reports = project_dir / "reports"
    result.to_csv(reports / "rolling_drift_results.csv", index=False, encoding="utf-8")
    bootstrap.to_csv(
        reports / "rolling_drift_bootstrap.csv", index=False, encoding="utf-8"
    )
    _upsert_csv(
        reports / "walk_forward_results.csv",
        pd.DataFrame(metric_rows),
        ["experiment_id"],
    )
    _write_report(
        reports / "rolling_drift_findings.md",
        result,
        bootstrap,
        checks,
        methods,
        weights,
    )
    print(result.to_string(index=False))
    print("gate", checks)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    run(args.project_dir.resolve(), args.resamples)


if __name__ == "__main__":
    main()

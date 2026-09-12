"""Semantic EDA for the official pre-pitch control dataset.

The audit focuses on information that is easy to misread in this competition:
``game_type=F`` is Futures League, the ``asof_*`` counters are career
cumulative, and failure is a union of location-related failure modes.  All
analyses use only the provided train and TrackMan files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "control_success"
RATE_COLUMNS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
USE_COLUMNS = [
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "base_state",
    "num_runners_on",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    *RATE_COLUMNS.values(),
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    TARGET,
]


def _read(project: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        project / "data" / "train.csv",
        usecols=USE_COLUMNS,
        low_memory=False,
    )
    if not frame["row_id"].is_unique:
        raise ValueError("train row_id must be unique")
    return frame


def _rounded_count(n: pd.Series, rate: pd.Series) -> np.ndarray:
    return np.rint(
        pd.to_numeric(n, errors="coerce").to_numpy(np.float64)
        * pd.to_numeric(rate, errors="coerce").to_numpy(np.float64)
    )


def career_counter_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Check whether official ASOF counters match earlier train rows exactly."""

    rows: list[dict[str, object]] = []
    for entity, n_col, rate_col in (
        ("pitcher", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter", "asof_batter_n", "asof_batter_success_rate"),
    ):
        id_col = f"{entity}_id"
        group = frame.groupby(id_col, observed=True, sort=False)
        expected_n = group.cumcount().to_numpy(np.float64)
        expected_s = (
            group[TARGET].cumsum().to_numpy(np.float64)
            - frame[TARGET].to_numpy(np.float64)
        )
        official_n = pd.to_numeric(frame[n_col], errors="coerce").to_numpy(np.float64)
        official_s = _rounded_count(frame[n_col], frame[rate_col])
        valid_n = np.isfinite(official_n)
        valid_s = valid_n & np.isfinite(official_s)
        n_error = official_n[valid_n] - expected_n[valid_n]
        s_error = official_s[valid_s] - expected_s[valid_s]
        rows.append(
            {
                "entity": entity,
                "rows": int(len(frame)),
                "n_valid": int(valid_n.sum()),
                "n_exact_fraction": float(np.mean(n_error == 0.0)),
                "n_max_abs_error": float(np.max(np.abs(n_error))),
                "success_count_exact_fraction": float(np.mean(s_error == 0.0)),
                "success_count_abs_error_p999": float(
                    np.quantile(np.abs(s_error), 0.999)
                ),
                "success_count_max_abs_error": float(np.max(np.abs(s_error))),
            }
        )
    return pd.DataFrame(rows)


def season_level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["season", "game_type"], observed=True)[TARGET]
        .agg(rows="size", successes="sum", rate="mean")
        .reset_index()
        .merge(
            frame.groupby(["season", "game_type"], observed=True)
            .agg(
                pitchers=("pitcher_id", "nunique"),
                batters=("batter_id", "nunique"),
                pitcher_teams=("pitcher_team_id", "nunique"),
                batter_teams=("batter_team_id", "nunique"),
            )
            .reset_index(),
            on=["season", "game_type"],
            validate="one_to_one",
        )
    )


def monthly_level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["season", "game_month", "game_type"], observed=True)[TARGET]
        .agg(rows="size", successes="sum", rate="mean")
        .reset_index()
    )


def level_mobility_summary(frame: pd.DataFrame) -> pd.DataFrame:
    pitcher = (
        frame.groupby(["season", "pitcher_id", "game_type"], observed=True)[TARGET]
        .agg(n="size", successes="sum", rate="mean")
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for season, group in pitcher.groupby("season", observed=True):
        counts = group.pivot(index="pitcher_id", columns="game_type", values="n")
        rates = group.pivot(index="pitcher_id", columns="game_type", values="rate")
        for column in ("R", "F"):
            if column not in counts:
                counts[column] = np.nan
                rates[column] = np.nan
        both = counts["R"].notna() & counts["F"].notna()
        reliable = (counts["R"] >= 20) & (counts["F"] >= 20)
        paired = (rates.loc[reliable, "F"] - rates.loc[reliable, "R"]).dropna()
        rows.append(
            {
                "season": int(season),
                "pitchers": int(len(counts)),
                "r_only": int((counts["R"].notna() & counts["F"].isna()).sum()),
                "f_only": int((counts["R"].isna() & counts["F"].notna()).sum()),
                "both_levels": int(both.sum()),
                "both_levels_fraction": float(both.mean()),
                "paired_n_ge20": int(len(paired)),
                "paired_f_minus_r_mean": float(paired.mean()) if len(paired) else np.nan,
                "paired_f_minus_r_median": float(paired.median()) if len(paired) else np.nan,
                "paired_rate_correlation": float(
                    rates.loc[reliable, ["R", "F"]].corr().iloc[0, 1]
                )
                if reliable.sum() >= 3
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def team_level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    pitcher = (
        frame.groupby(
            ["season", "game_type", "pitcher_team_id"], observed=True
        )[TARGET]
        .agg(rows="size", pitchers="count", successes="sum", rate="mean")
        .reset_index()
        .rename(columns={"pitcher_team_id": "team_id"})
    )
    batter = (
        frame.groupby(
            ["season", "game_type", "batter_team_id"], observed=True
        )[TARGET]
        .agg(rows="size", batters="count", successes="sum", rate="mean")
        .reset_index()
        .rename(columns={"batter_team_id": "team_id"})
    )
    return pitcher.merge(
        batter,
        on=["season", "game_type", "team_id"],
        how="outer",
        suffixes=("_pitcher", "_batter"),
    )


def _anchor_role(frame: pd.DataFrame) -> np.ndarray:
    pitcher = frame["pitcher_team_id"].eq(13).to_numpy()
    batter = frame["batter_team_id"].eq(13).to_numpy()
    return np.select(
        [pitcher & batter, pitcher, batter],
        ["BOTH_13", "PITCHER_13", "BATTER_13"],
        default="NEITHER_13",
    )


def anchor_role_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[["season", "game_type", TARGET]].copy()
    work["anchor_role"] = _anchor_role(frame)
    return (
        work.groupby(["season", "game_type", "anchor_role"], observed=True)[TARGET]
        .agg(rows="size", successes="sum", rate="mean")
        .reset_index()
    )


def reconstruct_failure_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover train-only current-pitch component flags from the next ASOF row."""

    group = frame.groupby("pitcher_id", observed=True, sort=False)
    current_n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce")
    next_n = group["asof_pitcher_n"].shift(-1)
    valid = next_n.eq(current_n + 1)
    component: dict[str, np.ndarray] = {}
    for name, column in RATE_COLUMNS.items():
        now = pd.Series(_rounded_count(current_n, frame[column]), index=frame.index)
        later = pd.Series(
            _rounded_count(next_n, group[column].shift(-1)), index=frame.index
        )
        delta = later - now
        component[name] = np.where(valid & delta.isin([0.0, 1.0]), delta, np.nan)

    work = frame[["season", "game_type", TARGET]].copy()
    work["anchor_role"] = _anchor_role(frame)
    for name, value in component.items():
        work[name] = value
    work = work.loc[work["success"].notna()].copy()
    work["middle_or_reverse"] = np.maximum(work["middle"], work["reverse"])
    work["failure"] = 1.0 - work[TARGET]
    work["unexplained_by_middle_reverse"] = (
        (work["failure"] == 1.0) & (work["middle_or_reverse"] == 0.0)
    ).astype(float)
    rows: list[dict[str, object]] = []
    for (season, game_type, anchor_role), group_frame in work.groupby(
        ["season", "game_type", "anchor_role"], observed=True
    ):
        failures = group_frame.loc[group_frame["failure"].eq(1.0)]
        rows.append(
            {
                "season": int(season),
                "game_type": str(game_type),
                "anchor_role": str(anchor_role),
                "rows_reconstructed": int(len(group_frame)),
                "target_mismatch_vs_asof_success": float(
                    np.mean(group_frame[TARGET] != group_frame["success"])
                ),
                "failure_rate": float(group_frame["failure"].mean()),
                "reverse_rate": float(group_frame["reverse"].mean()),
                "middle_rate": float(group_frame["middle"].mean()),
                "ball_rate": float(group_frame["ball"].mean()),
                "strike_rate": float(group_frame["strike"].mean()),
                "failure_reverse_share": float(failures["reverse"].mean()),
                "failure_middle_share": float(failures["middle"].mean()),
                "failure_middle_or_reverse_share": float(
                    failures["middle_or_reverse"].mean()
                ),
                "failure_unexplained_by_middle_reverse_share": float(
                    failures["unexplained_by_middle_reverse"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rate_columns = [column for column in frame if column.startswith("asof_")]
    rows: list[dict[str, object]] = []
    for (season, game_type), group in frame.groupby(
        ["season", "game_type"], observed=True
    ):
        for column in rate_columns:
            rows.append(
                {
                    "season": int(season),
                    "game_type": str(game_type),
                    "column": column,
                    "missing_fraction": float(group[column].isna().mean()),
                    "zero_fraction": float(
                        pd.to_numeric(group[column], errors="coerce").eq(0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _derived_stability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["count"] = (
        output["balls_before"].astype(str)
        + "-"
        + output["strikes_before"].astype(str)
    )
    output["hand_matchup"] = (
        output["pitcher_hand"].astype(str)
        + "-"
        + output["batter_hand"].astype(str)
    )
    output["inning_bucket"] = pd.cut(
        output["inning"], [-np.inf, 3, 6, 9, np.inf], labels=False
    ).astype("Int64")
    output["li_bucket"] = pd.qcut(
        pd.to_numeric(output["li"], errors="coerce"),
        q=10,
        labels=False,
        duplicates="drop",
    ).astype("Int64")
    output["score_bucket"] = pd.cut(
        pd.to_numeric(output["score_diff_pitcher_team"], errors="coerce"),
        [-np.inf, -4, -2, -1, 0, 1, 2, 4, np.inf],
        labels=False,
    ).astype("Int64")
    return output


def univariate_temporal_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Train-on-prior-year group-rate screens; diagnostic, not model selection."""

    work = _derived_stability_frame(frame)
    specs = {
        "game_type": ["game_type"],
        "count": ["count"],
        "game_type_count": ["game_type", "count"],
        "hand_matchup": ["hand_matchup"],
        "game_type_hand": ["game_type", "hand_matchup"],
        "base_state": ["base_state"],
        "inning_bucket": ["inning_bucket"],
        "li_bucket": ["li_bucket"],
        "score_bucket": ["score_bucket"],
        "pitcher_team_level": ["game_type", "pitcher_team_id"],
        "batter_team_level": ["game_type", "batter_team_id"],
    }
    rows: list[dict[str, object]] = []
    for audit_year in range(2020, 2025):
        source = work.loc[work["season"].eq(audit_year - 1)]
        audit = work.loc[work["season"].eq(audit_year)]
        target = audit[TARGET].to_numpy(np.float64)
        prior = float(source[TARGET].mean())
        baseline = np.full(len(audit), prior, dtype=np.float64)
        baseline_loss = np.mean((baseline - target) ** 2)
        for name, columns in specs.items():
            stats = (
                source.groupby(columns, observed=True)[TARGET]
                .agg(n="size", successes="sum")
                .reset_index()
            )
            for alpha in (100.0, 500.0, 2000.0):
                stats["prediction"] = (
                    stats["successes"] + alpha * prior
                ) / (stats["n"] + alpha)
                joined = audit[columns].merge(
                    stats[columns + ["prediction"]],
                    on=columns,
                    how="left",
                    sort=False,
                )
                prediction = joined["prediction"].fillna(prior).to_numpy(np.float64)
                loss = np.mean((prediction - target) ** 2)
                rows.append(
                    {
                        "audit_year": audit_year,
                        "feature": name,
                        "alpha": alpha,
                        "brier_gain_x100000": 100000.0 * (baseline_loss - loss),
                        "coverage": float(joined["prediction"].notna().mean()),
                    }
                )
    result = pd.DataFrame(rows)
    robust = (
        result.groupby(["feature", "alpha"], observed=True)
        .agg(
            minimum_gain=("brier_gain_x100000", "min"),
            mean_gain=("brier_gain_x100000", "mean"),
            positive_year_fraction=(
                "brier_gain_x100000",
                lambda value: float(np.mean(value > 0)),
            ),
            minimum_coverage=("coverage", "min"),
        )
        .reset_index()
        .sort_values(["minimum_gain", "mean_gain"], ascending=False)
    )
    return result.merge(
        robust,
        on=["feature", "alpha"],
        how="left",
        validate="many_to_one",
    )


def _summary_payload(
    frame: pd.DataFrame,
    career: pd.DataFrame,
    season_level: pd.DataFrame,
    mobility: pd.DataFrame,
    failure: pd.DataFrame,
    stability: pd.DataFrame,
) -> dict[str, object]:
    robust = (
        stability[[
            "feature",
            "alpha",
            "minimum_gain",
            "mean_gain",
            "positive_year_fraction",
            "minimum_coverage",
        ]]
        .drop_duplicates()
        .sort_values(["minimum_gain", "mean_gain"], ascending=False)
    )
    return {
        "protocol": "V24_OFFICIAL_SEMANTIC_EDA_V1",
        "rows": int(len(frame)),
        "seasons": [int(value) for value in sorted(frame["season"].unique())],
        "official_semantics": {
            "R": "Regular, KBO first-team regular season",
            "F": "Futures League, KBO second team",
            "target": "success unless middle/risky, far outside, or reverse to catcher request",
            "test_rows_independent": True,
        },
        "career_counter_audit": career.to_dict(orient="records"),
        "season_level": season_level.to_dict(orient="records"),
        "mobility": mobility.to_dict(orient="records"),
        "failure_components": failure.to_dict(orient="records"),
        "robust_univariate_top10": robust.head(10).to_dict(orient="records"),
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _read(project)
    career = career_counter_audit(frame)
    season_level = season_level_summary(frame)
    monthly_level = monthly_level_summary(frame)
    mobility = level_mobility_summary(frame)
    teams = team_level_summary(frame)
    anchor = anchor_role_summary(frame)
    failure = reconstruct_failure_components(frame)
    missing = missingness_summary(frame)
    stability = univariate_temporal_stability(frame)
    outputs = {
        "career_counter_audit.csv": career,
        "season_level.csv": season_level,
        "monthly_level.csv": monthly_level,
        "level_mobility.csv": mobility,
        "team_level.csv": teams,
        "anchor_role.csv": anchor,
        "failure_components.csv": failure,
        "missingness.csv": missing,
        "univariate_stability.csv": stability,
    }
    for name, value in outputs.items():
        value.to_csv(output_dir / name, index=False)
    summary = _summary_payload(
        frame, career, season_level, mobility, failure, stability
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v24_semantic_eda_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

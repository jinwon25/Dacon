"""Forward screen for stable contextual residual structure above v17.

All bins are fixed before labels are inspected.  Residual tables are fitted on
the reconstructed v14 analogue because v17 itself was deployed as an additive
v14 residual correction.  Candidate corrections are then added to v17, so the
screen directly measures incremental value and exposes overlap with v17.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.archive.v16_residual_calibration_screen import EPSILON, _bss_gain, _keys, load_v14_folds


V17_NAME = "multi_pitcher_batter_hand_pressure_d1_a3200_w1.5"


def add_fixed_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_state"] = (
        out["balls_before"].astype("string")
        + "-"
        + out["strikes_before"].astype("string")
    )
    out["inning_group"] = pd.cut(
        pd.to_numeric(out["inning"], errors="coerce"),
        [-np.inf, 3, 6, 9, np.inf],
        labels=["early", "middle", "late", "extra"],
    ).astype("string")
    out["score_group"] = np.clip(
        pd.to_numeric(out["score_diff_pitcher_team"], errors="coerce").fillna(0),
        -4,
        4,
    ).round().astype(int).astype("string")
    out["li_group"] = pd.cut(
        pd.to_numeric(out["li"], errors="coerce"),
        [-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["vlow", "low", "medium", "high", "vhigh"],
    ).astype("string")
    out["home_we_group"] = np.floor(
        pd.to_numeric(out["home_win_expectancy"], errors="coerce").fillna(50.0)
        / 10.0
    ).clip(0, 9).astype(int).astype("string")
    for prefix, column in (
        ("pitcher_n", "asof_pitcher_n"),
        ("batter_n", "asof_batter_n"),
        ("mix_n", "asof_pitcher_pitchmix_n"),
    ):
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(lower=0.0)
        out[f"{prefix}_group"] = np.floor(np.log2(values + 1.0)).clip(0, 14).astype(int).astype("string")
    rate_columns = {
        "pitcher_success": "asof_pitcher_success_rate",
        "pitcher_reverse": "asof_pitcher_reverse_rate",
        "pitcher_middle": "asof_pitcher_middle_rate",
        "pitcher_ball": "asof_pitcher_ball_rate",
        "pitcher_strike": "asof_pitcher_strike_rate",
        "batter_success": "asof_batter_success_rate",
        "batter_middle": "asof_batter_middle_rate",
        "fastball": "asof_pitcher_fastball_rate",
        "breaking": "asof_pitcher_breaking_rate",
        "offspeed": "asof_pitcher_offspeed_rate",
    }
    for name, column in rate_columns.items():
        values = pd.to_numeric(out[column], errors="coerce").fillna(-1.0)
        out[f"{name}_group"] = np.where(
            values < 0,
            "missing",
            np.floor(values.clip(0.0, 0.999999) / 0.05).astype(int).astype(str),
        )
    recent = (
        0.50 * pd.to_numeric(out["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        + 0.30 * pd.to_numeric(out["asof_pitcher_prev3_game_success_rate"], errors="coerce")
        + 0.20 * pd.to_numeric(out["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    )
    career = pd.to_numeric(out["asof_pitcher_success_rate"], errors="coerce")
    delta = (recent - career).fillna(9.0)
    out["recent_delta_group"] = np.where(
        delta > 2.0,
        "missing",
        np.floor((delta.clip(-0.30, 0.299999) + 0.30) / 0.05).astype(int).astype(str),
    )
    return out


BASE_CONTEXTS = [
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "count_state",
    "outs_before",
    "base_state",
    "inning_group",
    "score_group",
    "li_group",
    "home_we_group",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_n_group",
    "batter_n_group",
    "mix_n_group",
    "pitcher_success_group",
    "pitcher_reverse_group",
    "pitcher_middle_group",
    "pitcher_ball_group",
    "pitcher_strike_group",
    "batter_success_group",
    "batter_middle_group",
    "fastball_group",
    "breaking_group",
    "offspeed_group",
    "recent_delta_group",
]


def context_groups() -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {
        f"domain_{column}": ("domain3", column) for column in BASE_CONTEXTS
    }
    groups.update(
        {
            "domain_count_hands": (
                "domain3",
                "count_state",
                "pitcher_hand",
                "batter_hand",
            ),
            "domain_count_base": ("domain3", "count_state", "base_state"),
            "domain_count_inning": ("domain3", "count_state", "inning_group"),
            "domain_count_outs": ("domain3", "count_state", "outs_before"),
            "domain_count_score": ("domain3", "count_state", "score_group"),
            "domain_count_li": ("domain3", "count_state", "li_group"),
            "domain_count_pitcher_rate": (
                "domain3",
                "count_state",
                "pitcher_success_group",
            ),
            "domain_count_middle_rate": (
                "domain3",
                "count_state",
                "pitcher_middle_group",
            ),
            "domain_count_ball_rate": (
                "domain3",
                "count_state",
                "pitcher_ball_group",
            ),
            "domain_count_strike_rate": (
                "domain3",
                "count_state",
                "pitcher_strike_group",
            ),
            "domain_pressure_hands": (
                "domain3",
                "pressure",
                "pitcher_hand",
                "batter_hand",
            ),
            "domain_inning_score": ("domain3", "inning_group", "score_group"),
            "domain_base_outs": ("domain3", "base_state", "outs_before"),
            "domain_teams": ("domain3", "pitcher_team_id", "batter_team_id"),
            "domain_pitcher_team_count": (
                "domain3",
                "pitcher_team_id",
                "count_state",
            ),
            "domain_batter_team_count": (
                "domain3",
                "batter_team_id",
                "count_state",
            ),
        }
    )
    return groups


def _history_stats(
    folds: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    audit_year: int,
    columns: tuple[str, ...],
    decay: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in sorted(year for year in folds if year < audit_year):
        rows, target, prediction = folds[year]
        weight = float(decay) ** float(audit_year - 1 - year)
        parts.append(
            pd.DataFrame(
                {
                    "key": _keys(rows, columns),
                    "weighted_residual": weight * (target - prediction),
                    "weight": weight,
                }
            )
        )
    return pd.concat(parts, ignore_index=True).groupby("key", observed=True).agg(
        weighted_residual=("weighted_residual", "sum"),
        effective_count=("weight", "sum"),
    )


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    folds_raw = load_v14_folds(project, train)
    folds = {
        year: (add_fixed_context(rows), target, prediction)
        for year, (rows, target, prediction) in folds_raw.items()
    }
    del folds_raw, train
    v17: dict[int, np.ndarray] = {}
    for year in (2023, 2024):
        with np.load(
            project
            / "artifacts"
            / "v16_multiseason_20260815_02"
            / f"{V17_NAME}_o{year}.npz"
        ) as saved:
            v17[year] = saved["candidate"].astype(np.float64)
    metric_rows: list[dict[str, object]] = []
    for audit_year in (2023, 2024):
        audit_rows, target, _ = folds[audit_year]
        incumbent = v17[audit_year]
        reference = float(target.mean() * (1.0 - target.mean()))
        incumbent_error = incumbent - target
        for group_name, columns in context_groups().items():
            for decay in (0.25, 0.5, 0.75, 1.0):
                stats = _history_stats(folds, audit_year, columns, decay)
                audit_keys = _keys(audit_rows, columns)
                for alpha in (100.0, 400.0, 800.0, 1600.0, 3200.0):
                    effect = stats["weighted_residual"] / (
                        stats["effective_count"] + alpha
                    )
                    mapped = audit_keys.map(effect)
                    correction = mapped.fillna(0.0).to_numpy(np.float64)
                    coverage = float(mapped.notna().mean())
                    for weight in (0.25, 0.5, 1.0, 1.5):
                        delta_loss = -2.0 * weight * incumbent_error * correction - (
                            weight * correction
                        ) ** 2
                        gain = 100_000.0 * float(delta_loss.mean()) / reference
                        metric_rows.append(
                            {
                                "candidate": (
                                    f"{group_name}_d{decay:g}_a{alpha:g}_w{weight:g}"
                                ),
                                "audit_year": audit_year,
                                "group": group_name,
                                "columns": "|".join(columns),
                                "decay": decay,
                                "alpha": alpha,
                                "weight": weight,
                                "coverage": coverage,
                                "gain_vs_v17": gain,
                            }
                        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust = (
        metrics.groupby("candidate", observed=True)["gain_vs_v17"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust["positive_folds"] = (
        metrics.assign(positive=metrics["gain_vs_v17"].gt(0.0))
        .groupby("candidate", observed=True)["positive"]
        .sum()
        .reindex(robust["candidate"])
        .to_numpy()
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    passing = robust.loc[robust["min_gain"].gt(0.0)]
    summary = {
        "protocol": "FIXED_CONTEXT_EB_FORWARD_V1",
        "audit_years": [2023, 2024],
        "n_groups": len(context_groups()),
        "n_recipes": int(metrics["candidate"].nunique()),
        "n_strict_improvements": int(len(passing)),
        "best": robust.head(40).to_dict(orient="records"),
    }
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
        default=Path("artifacts/context_residual_20260816_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

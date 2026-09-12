"""Audit and expose train-only season-state sufficient statistics from asof_* columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PITCHER_RATE_COLUMNS = {
    "success": "asof_pitcher_success_rate", "reverse": "asof_pitcher_reverse_rate", "middle": "asof_pitcher_middle_rate", "ball": "asof_pitcher_ball_rate", "strike": "asof_pitcher_strike_rate",
}
BATTER_RATE_COLUMNS = {"success": "asof_batter_success_rate", "middle": "asof_batter_middle_rate"}
PITCHMIX_RATE_COLUMNS = {"fastball": "asof_pitcher_fastball_rate", "breaking": "asof_pitcher_breaking_rate", "offspeed": "asof_pitcher_offspeed_rate"}


def _read(project: Path) -> pd.DataFrame:
    cols = ["row_id", "season", "pitcher_id", "batter_id", "control_success", "asof_pitcher_n", *PITCHER_RATE_COLUMNS.values(), "asof_batter_n", *BATTER_RATE_COLUMNS.values(), "asof_pitcher_pitchmix_n", *PITCHMIX_RATE_COLUMNS.values()]
    return pd.read_csv(project / "data/train.csv", usecols=cols, encoding="utf-8-sig", low_memory=False)


def _round_count(denom: pd.Series, rate: pd.Series) -> pd.Series:
    return np.rint(pd.to_numeric(denom, errors="coerce") * pd.to_numeric(rate, errors="coerce"))


def transition_audit(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # The next row in the CSV is not necessarily the next pitch for the same
    # pitcher. Grouping by pitcher while preserving row_id order recovers the
    # entity trajectory without using target for any feature.
    pitcher_group = frame.groupby("pitcher_id", observed=True, sort=False)
    rows = []
    current_success = _round_count(frame["asof_pitcher_n"], frame["asof_pitcher_success_rate"])
    next_n = pitcher_group["asof_pitcher_n"].shift(-1)
    next_rate = pitcher_group["asof_pitcher_success_rate"].shift(-1)
    n_delta = next_n - frame["asof_pitcher_n"]
    exact_error = next_n * next_rate - frame["asof_pitcher_n"] * frame["asof_pitcher_success_rate"] - frame["control_success"]
    mask = frame["asof_pitcher_n"].notna() & next_n.notna() & frame["asof_pitcher_success_rate"].notna() & next_rate.notna()
    rows.append({"check": "pitcher_next_n_delta_1", "n_checked": int(mask.sum()), "n_pass": int((n_delta[mask] == 1).sum()), "pass_rate": float((n_delta[mask] == 1).mean()), "max_abs_error": float((n_delta[mask] - 1).abs().max()) if mask.any() else np.nan})
    rows.append({"check": "next_success_rate_transition_within_display_precision", "n_checked": int(mask.sum()), "n_pass": int((exact_error[mask].abs() <= 0.01).sum()), "pass_rate": float((exact_error[mask].abs() <= 0.01).mean()), "max_abs_error": float(exact_error[mask].abs().max()) if mask.any() else np.nan})
    # Season baseline is the first pre-pitch snapshot in each pitcher-season.
    first = frame.groupby(["pitcher_id", "season"], observed=True).head(1).set_index(["pitcher_id", "season"])
    frame["pitcher_season_n"] = frame["asof_pitcher_n"] - frame.set_index(["pitcher_id", "season"]).index.map(first["asof_pitcher_n"]).to_numpy()
    frame["pitcher_season_success_count"] = current_success - frame.set_index(["pitcher_id", "season"]).index.map(_round_count(first["asof_pitcher_n"], first["asof_pitcher_success_rate"])).to_numpy()
    rows.append({"check": "pitcher_season_n_nonnegative", "n_checked": int(frame["pitcher_season_n"].notna().sum()), "n_pass": int((frame["pitcher_season_n"].dropna() >= 0).sum()), "pass_rate": float((frame["pitcher_season_n"].dropna() >= 0).mean()), "max_abs_error": float(frame["pitcher_season_n"].min()) if frame["pitcher_season_n"].notna().any() else np.nan})
    rows.append({"check": "pitcher_season_success_count_nonnegative", "n_checked": int(frame["pitcher_season_success_count"].notna().sum()), "n_pass": int((frame["pitcher_season_success_count"].dropna() >= 0).sum()), "pass_rate": float((frame["pitcher_season_success_count"].dropna() >= 0).mean()), "max_abs_error": float(frame["pitcher_season_success_count"].min()) if frame["pitcher_season_success_count"].notna().any() else np.nan})
    return pd.DataFrame(rows), frame


def integer_reconstruction(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("pitcher", "asof_pitcher_n", PITCHER_RATE_COLUMNS), ("batter", "asof_batter_n", BATTER_RATE_COLUMNS), ("pitchmix", "asof_pitcher_pitchmix_n", PITCHMIX_RATE_COLUMNS)]
    for entity, denom_col, rate_map in specs:
        denom = pd.to_numeric(frame[denom_col], errors="coerce")
        for name, rate_col in rate_map.items():
            rate = pd.to_numeric(frame[rate_col], errors="coerce")
            mask = denom.notna() & rate.notna() & (denom > 0)
            counts = _round_count(denom[mask], rate[mask])
            round_error = (counts - denom[mask] * rate[mask]).abs()
            rows.append({"entity": entity, "stat": name, "denominator": denom_col, "rate_column": rate_col, "n_checked": int(mask.sum()), "round_error_p999": float(round_error.quantile(0.999)) if mask.any() else np.nan, "round_error_max": float(round_error.max()) if mask.any() else np.nan, "count_min": float(counts.min()) if mask.any() else np.nan, "count_max": float(counts.max()) if mask.any() else np.nan, "negative_count_rows": int((counts < 0).sum()), "count_gt_denominator_rows": int((counts > denom[mask]).sum())})
    return pd.DataFrame(rows)


def run(project: Path) -> None:
    frame = _read(project)
    transition, enriched = transition_audit(frame)
    reconstruction = integer_reconstruction(frame)
    out_dir = project / "artifacts/top1100"; report_dir = project / "research/reports/top1100"; out_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    transition.to_csv(report_dir / "asof_transition_audit.csv", index=False)
    reconstruction.to_csv(report_dir / "asof_integer_reconstruction.csv", index=False)
    # Keep a compact train artifact for model prototyping; it contains no target.
    feature_cols = ["row_id", "season", "pitcher_id", "batter_id", "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n", "pitcher_season_n", "pitcher_season_success_count"]
    enriched[feature_cols].head(200_000).to_csv(out_dir / "asof_state_train_sample.csv", index=False)
    (report_dir / "season_state_feature_catalog.md").write_text("""# P0-A season-state feature catalog

The audit supports a cumulative, pre-pitch sufficient-statistic interpretation for pitcher state. Candidate features are computed from the current row and train-origin snapshots only:

- pitcher/batter season exposure and season success count;
- career-before-season, prior-season and current-season posterior rates;
- current-versus-career/prior logit deltas and posterior uncertainty;
- newcomer, returner, long-gap, team-switch and role-change flags;
- current-season pitch-mix composition and career drift;
- prev1/3/5 game disagreement against season/career posterior.

Rate displays are rounded before integer reconstruction. If a denominator/rate pair is not within its feasible range, it is excluded from count features and retained only as a missingness diagnostic. Fixed smoothing constants are not treated as validated hyperparameters; later state models must choose discount/process variance on earlier inner folds.
""", encoding="utf-8")
    print(transition.to_string(index=False)); print(reconstruction.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); run(args.project_dir.resolve())


if __name__ == "__main__": main()

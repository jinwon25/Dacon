"""Target-free structural alignment between main game blocks and Trackman games."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


MAIN_COLUMNS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before", "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand"]
TM_COLUMNS = ["season", "game_month", "game_dayofweek", "trackman_game_id", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand", "pitcher_team", "batter_team"]
FORBIDDEN = {"control_success", "target", "control_success_count", "y", "label"}


def _main_games(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(project / "data/train.csv", usecols=MAIN_COLUMNS, encoding="utf-8-sig", low_memory=False)
    # A 0-0, 0-out state recurs at every new plate appearance in the first
    # inning.  A real game boundary is the transition into inning 1/top from
    # a row outside that half-inning.
    first_top = frame["inning"].eq(1) & frame["top_bottom"].astype(str).eq("T")
    previous_first_top = frame["inning"].shift().eq(1) & frame["top_bottom"].shift().astype(str).eq("T")
    starts = first_top & ~previous_first_top
    starts |= frame["season"].ne(frame["season"].shift())
    frame["main_game_idx"] = starts.cumsum().astype(int)
    frame["token"] = frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str) + "-" + frame["outs_before"].astype(str) + "-" + frame["top_bottom"].astype(str)
    summary = frame.groupby(["season", "main_game_idx"], observed=True).agg(rows=("season", "size"), month=("game_month", "first"), dow=("game_dayofweek", "first"), max_inning=("inning", "max"), top_ratio=("top_bottom", lambda x: (x.astype(str) == "T").mean()), unique_pitchers=("pitcher_hand", "size"), hand_same=("batter_hand", lambda x: 0.0)).reset_index()
    # Compact token/hand distributions are target-free structural signatures.
    token = pd.crosstab([frame["season"], frame["main_game_idx"]], frame["token"]).reset_index()
    hand = frame.assign(hand_token=frame["pitcher_hand"].astype(str) + "_" + frame["batter_hand"].astype(str))
    hand = pd.crosstab([hand["season"], hand["main_game_idx"]], hand["hand_token"]).reset_index()
    summary = summary.merge(token, on=["season", "main_game_idx"], how="left").merge(hand, on=["season", "main_game_idx"], how="left")
    return frame, summary


def _trackman_games(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(project / "data/trackman_history.csv", usecols=TM_COLUMNS, encoding="utf-8-sig", low_memory=False)
    frame["trackman_game_id"] = frame["trackman_game_id"].astype(str)
    frame["token"] = frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str) + "-" + frame["outs_before"].astype(str) + "-" + frame["top_bottom"].astype(str)
    summary = frame.groupby(["season", "trackman_game_id"], observed=True).agg(rows=("season", "size"), month=("game_month", "first"), dow=("game_dayofweek", "first"), max_inning=("inning", "max"), top_ratio=("top_bottom", lambda x: (x.astype(str) == "Top").mean()), unique_pitchers=("pitcher_hand", "size"), hand_same=("batter_hand", lambda x: 0.0)).reset_index()
    token = pd.crosstab([frame["season"], frame["trackman_game_id"]], frame["token"]).reset_index()
    hand = frame.assign(hand_token=frame["pitcher_hand"].astype(str) + "_" + frame["batter_hand"].astype(str))
    hand = pd.crosstab([hand["season"], hand["trackman_game_id"]], hand["hand_token"]).reset_index()
    summary = summary.merge(token, on=["season", "trackman_game_id"], how="left").merge(hand, on=["season", "trackman_game_id"], how="left")
    # Both official tables use Monday=0.  For example, 2019-03-23 (Saturday)
    # is encoded as 5 in both files; no weekday offset is required.
    summary["dow"] = summary["dow"].astype(int)
    return frame, summary


def _numeric_signature(main: pd.DataFrame, trackman: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    keys = {"season", "month", "dow", "rows", "max_inning", "top_ratio", "unique_pitchers", "hand_same"}
    tokens = sorted((set(main.columns) | set(trackman.columns)) - {"season", "main_game_idx", "trackman_game_id", "month", "dow"} - keys)
    columns = ["rows", "max_inning", "top_ratio", "unique_pitchers", "hand_same", *tokens]
    for col in columns:
        if col not in main: main[col] = 0.0
        if col not in trackman: trackman[col] = 0.0
    left_frame = main[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    right_frame = trackman[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    # Counts of token/hand states describe a distribution, not game length.
    # Normalize them by rows and use log scale for the two size features.
    distribution_columns = [c for c in columns if c not in {"rows", "max_inning", "top_ratio", "unique_pitchers", "hand_same"}]
    for frame in (left_frame, right_frame):
        frame[distribution_columns] = frame[distribution_columns].div(frame["rows"].replace(0, 1), axis=0)
        frame["rows"] = np.log1p(frame["rows"])
        frame["unique_pitchers"] = np.log1p(frame["unique_pitchers"])
    left = left_frame.to_numpy(dtype=float)
    right = right_frame.to_numpy(dtype=float)
    scale = np.nanmedian(np.abs(np.concatenate([left, right], axis=0) - np.nanmedian(np.concatenate([left, right], axis=0), axis=0)), axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return pd.DataFrame(left / scale, columns=columns), pd.DataFrame(right / scale, columns=columns), columns


def _match_bucket(left: pd.DataFrame, right: pd.DataFrame, left_x: pd.DataFrame, right_x: pd.DataFrame, *, placebo_shift: int = 0) -> pd.DataFrame:
    rows = []
    for (season, month, dow), lgroup in left.groupby(["season", "month", "dow"], observed=True):
        candidates = right.loc[(right["season"] == season) & (right["month"] == month) & (right["dow"] == dow)]
        if candidates.empty: continue
        lx = left_x.loc[lgroup.index].to_numpy(); rx = right_x.loc[candidates.index].to_numpy()
        if placebo_shift:
            rx = np.roll(rx, placebo_shift % len(rx), axis=0)
        cost = cdist(lx, rx, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            ordered = np.sort(cost[i]); margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else np.nan
            rows.append({"season": int(season), "main_game_idx": left.loc[lgroup.index[i], "main_game_idx"], "trackman_game_id": candidates.iloc[j]["trackman_game_id"], "distance": float(cost[i, j]), "margin": margin, "candidate_count": len(candidates), "placebo_shift": placebo_shift})
    return pd.DataFrame(rows)


def run(project: Path, placebo_repeats: int = 20) -> None:
    main_rows, main_games = _main_games(project)
    tm_rows, tm_games = _trackman_games(project)
    # Explicit whitelist protects the aligner from accidental target joins.
    if FORBIDDEN.intersection(set(main_rows.columns) | set(tm_rows.columns)): raise AssertionError("target-derived column reached structural aligner")
    main_x, tm_x, _ = _numeric_signature(main_games.copy(), tm_games.copy())
    matches = _match_bucket(main_games, tm_games, main_x, tm_x)
    candidates = matches.sort_values(["season", "main_game_idx", "distance"]).copy()
    candidates["confidence"] = np.select([candidates["distance"].le(1.6) & candidates["margin"].ge(0.1), candidates["distance"].le(2.2)], ["high", "medium"], default="low")
    report_dir = project / "reports/top1100"; artifact_dir = project / "artifacts/top1100"; report_dir.mkdir(parents=True, exist_ok=True); artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(artifact_dir / "alignment_game_candidates.parquet", index=False)
    candidates.to_csv(report_dir / "alignment_game_candidates.csv", index=False)
    # Edge table keeps only the interpretable candidate fields; raw row-level
    # DTW/Needleman-Wunsch is intentionally not claimed without game dates in main.
    candidates[["season", "main_game_idx", "trackman_game_id", "distance", "margin", "confidence"]].to_parquet(artifact_dir / "alignment_edges.parquet", index=False)
    placebo_rows = []
    for shift in range(1, placebo_repeats + 1):
        # Permuting the month/day key destroys the structural calendar match.
        # A pure column cyclic shift would leave a full Hungarian objective
        # invariant and is therefore not a valid placebo for this assignment.
        rng = np.random.default_rng(10_000 + shift)
        placebo_tm = tm_games.copy()
        placebo_tm["month"] = rng.permutation(placebo_tm["month"].to_numpy())
        placebo_tm["dow"] = rng.permutation(placebo_tm["dow"].to_numpy())
        placebo = _match_bucket(main_games, placebo_tm, main_x, tm_x)
        placebo_rows.append({"shift": shift, "n_matches": len(placebo), "p05_distance": float(placebo["distance"].quantile(0.05)) if not placebo.empty else np.nan, "median_distance": float(placebo["distance"].median()) if not placebo.empty else np.nan, "high_rate": float((placebo["distance"].le(1.6) & placebo["margin"].ge(0.1)).mean()) if not placebo.empty else 0.0})
    placebo = pd.DataFrame(placebo_rows)
    real_summary = {"n_main_games": int(main_games.shape[0]), "n_trackman_games": int(tm_games.shape[0]), "n_matches": int(len(candidates)), "high_rate": float((candidates["confidence"] == "high").mean()) if not candidates.empty else 0.0, "median_distance": float(candidates["distance"].median()) if not candidates.empty else np.nan, "median_margin": float(candidates["margin"].median()) if not candidates.empty else np.nan}
    placebo["real_median_distance"] = real_summary["median_distance"]; placebo["real_high_rate"] = real_summary["high_rate"]; placebo["real_better_than_placebo"] = placebo["median_distance"] > real_summary["median_distance"]
    placebo.to_csv(report_dir / "alignment_placebo_fdr.csv", index=False)
    # Existing annual linkage is preserved as a fallback, annotated with the
    # structural game coverage rather than silently replacing it.
    old = pd.read_csv(project / "reports/trackman_linkage.csv")
    old["alignment_game_match_count"] = old["season"].map(candidates.groupby("season").size()).fillna(0).astype(int)
    old["mapping_method"] = "legacy_annual_fingerprint; structural_game_alignment_audit_only"
    old.to_csv(report_dir / "player_mapping_by_origin.csv", index=False)
    (report_dir / "alignment_audit.md").write_text("""# P0-B target-free structural alignment audit

The aligner uses only season/month/day-of-week, legal count/out/inning transitions, game length, half distribution and hand tokens. `control_success`, target encodings and target-derived statistics are prohibited by an explicit whitelist assertion. Main does not contain game_date or game_id, so this is a game-candidate structural alignment rather than a final pitch-level identity proof.

Real candidate summary:

```json
""" + json_dumps(real_summary) + """
```

Placebo cyclic shifts are written to `alignment_placebo_fdr.csv`. Existing annual player linkage is not promoted to a new physical profile solely from this audit; high-confidence structural coverage and permutation separation must be confirmed first. Medium/low/unmatched rows therefore remain on the V2 fallback.
""", encoding="utf-8")
    print(real_summary)


def json_dumps(value: object) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); parser.add_argument("--placebo-repeats", type=int, default=20); args = parser.parse_args(); run(args.project_dir.resolve(), args.placebo_repeats)


if __name__ == "__main__": main()

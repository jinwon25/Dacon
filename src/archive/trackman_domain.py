"""Leakage-safe Trackman domain profiles.

The competition does not expose a direct key between the pitch-by-pitch
Trackman log and the main table.  This module therefore only builds
pitcher-level historical summaries after the existing anonymous-ID linkage.
The summaries deliberately describe repeatability (release variation,
pitch-count/fatigue proxies, repertoire entropy and count selection), rather
than treating a raw historical mean as a current-pitch measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.archive.trackman_linkage import (
    PHYSICAL_MEASURES,
    PITCH_GROUPS,
    build_pitcher_profile_table,
)


def _safe_name(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def _robust_mad(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if not len(x):
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _slope(values: pd.Series, seasons: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    x = pd.to_numeric(seasons, errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.var(x[ok]) <= 0:
        return np.nan
    return float(np.cov(x[ok], y[ok], ddof=0)[0, 1] / np.var(x[ok]))


def _entropy(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    counts = frame[columns].fillna(0.0).astype(float)
    total = counts.sum(axis=1).replace(0.0, np.nan)
    p = counts.div(total, axis=0)
    return -(p.clip(lower=1e-12) * np.log(p.clip(lower=1e-12))).sum(axis=1)


def _base_profile(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one as-of history snapshot by Trackman pitcher ID."""
    pid = "pitcher_trackman_id"
    grouped = history.groupby(pid, observed=True)
    out = grouped.size().rename("tm_domain_pitch_n").to_frame()

    # Global robust release/flight distribution.  MAD and IQR are less
    # sensitive to occasional bad radar readings than a raw standard deviation.
    for measure in PHYSICAL_MEASURES:
        numeric = pd.to_numeric(history[measure], errors="coerce")
        stats = history.assign(__value=numeric).groupby(pid, observed=True)["__value"].agg(
            ["mean", "std", "median"]
        )
        stats.columns = [
            f"tm_domain_{measure}_{name}" for name in ("mean", "std", "median")
        ]
        out = out.join(stats)
        mad = history.assign(__value=numeric).groupby(pid, observed=True)["__value"].apply(_robust_mad)
        out[f"tm_domain_{measure}_mad"] = mad
        out[f"tm_domain_{measure}_cv"] = out[f"tm_domain_{measure}_std"] / (
            out[f"tm_domain_{measure}_mean"].abs() + 1e-3
        )

    # Repertoire and pitch-group-specific distributions.
    counts = history.groupby([pid, "pitch_type_group"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[*PITCH_GROUPS, "other"], fill_value=0)
    counts.columns = [f"tm_domain_{_safe_name(c)}_n" for c in counts.columns]
    out = out.join(counts)
    total = counts.sum(axis=1).replace(0, np.nan)
    for group in [*PITCH_GROUPS, "other"]:
        ncol = f"tm_domain_{group}_n"
        out[f"tm_domain_{group}_rate"] = counts[ncol] / total
    out["tm_domain_repertoire_entropy"] = _entropy(
        out, [f"tm_domain_{group}_n" for group in [*PITCH_GROUPS, "other"]]
    )
    out["tm_domain_effective_repertoire"] = np.exp(out["tm_domain_repertoire_entropy"])
    out["tm_domain_fastball_to_offspeed"] = out["tm_domain_fastball_n"] / (
        out["tm_domain_offspeed_n"] + 20.0
    )

    for group in PITCH_GROUPS:
        subset = history[history["pitch_type_group"].astype("string") == group]
        if subset.empty:
            continue
        g = subset.groupby(pid, observed=True)
        for measure in PHYSICAL_MEASURES:
            stats = g[measure].agg(["mean", "std"])
            stats.columns = [
                f"tm_domain_{group}_{measure}_{name}" for name in ("mean", "std")
            ]
            out = out.join(stats)

    # Temporal stability across seasons.  We retain a slope and a recent-vs-
    # prior shift, both computed only from seasons before the forecast origin.
    annual = history.groupby([pid, "season"], observed=True)[PHYSICAL_MEASURES].mean().reset_index()
    latest_season = int(history["season"].max())
    recent = annual[annual["season"] == latest_season].set_index(pid)
    for measure in PHYSICAL_MEASURES:
        slope = annual.groupby(pid, observed=True).apply(
            lambda frame: _slope(frame[measure], frame["season"]), include_groups=False
        )
        out[f"tm_domain_{measure}_season_slope"] = slope
        out[f"tm_domain_{measure}_latest"] = recent[measure]
        out[f"tm_domain_{measure}_latest_minus_mean"] = (
            recent[measure] - out[f"tm_domain_{measure}_mean"]
        )

    # Game workload and within-game fatigue proxies.
    game_keys = [pid, "trackman_game_id"]
    game = history.groupby(game_keys, observed=True)
    game_pitch_n = game.size().rename("__game_pitch_n").reset_index()
    game_summary = game_pitch_n.groupby(pid, observed=True)["__game_pitch_n"].agg(
        ["mean", "std", "max"]
    )
    game_summary.columns = [
        "tm_domain_game_pitch_n_mean",
        "tm_domain_game_pitch_n_std",
        "tm_domain_game_pitch_n_max",
    ]
    out = out.join(game_summary)
    out["tm_domain_game_n"] = game_pitch_n.groupby(pid, observed=True).size()
    for measure in ("rel_speed", "spin_rate", "extension", "rel_height", "rel_side"):
        game_measure = game[measure].mean().reset_index()
        variability = game_measure.groupby(pid, observed=True)[measure].agg(["mean", "std"])
        variability.columns = [
            f"tm_domain_game_{measure}_mean",
            f"tm_domain_game_{measure}_std",
        ]
        out = out.join(variability)

    pitch_no = pd.to_numeric(history["pitch_no"], errors="coerce")
    out["tm_domain_pitch_no_mean"] = history.assign(__pitch_no=pitch_no).groupby(pid, observed=True)["__pitch_no"].mean()
    out["tm_domain_pitch_no_std"] = history.assign(__pitch_no=pitch_no).groupby(pid, observed=True)["__pitch_no"].std()
    out["tm_domain_pitch_no_max"] = history.assign(__pitch_no=pitch_no).groupby(pid, observed=True)["__pitch_no"].max()
    out["tm_domain_late_pitch_share"] = history.assign(__pitch_no=pitch_no).groupby(pid, observed=True)["__pitch_no"].apply(
        lambda x: float((x >= 50).mean())
    )
    pa_pitch = pd.to_numeric(history["pitch_of_pa"], errors="coerce")
    out["tm_domain_pa_pitch_mean"] = history.assign(__pa_pitch=pa_pitch).groupby(pid, observed=True)["__pa_pitch"].mean()
    out["tm_domain_pa_pitch_std"] = history.assign(__pa_pitch=pa_pitch).groupby(pid, observed=True)["__pa_pitch"].std()
    out["tm_domain_pa_3plus_share"] = history.assign(__pa_pitch=pa_pitch).groupby(pid, observed=True)["__pa_pitch"].apply(
        lambda x: float((x >= 3).mean())
    )

    # Opposite/same-handed batter repertoire.  This is often more relevant to
    # command than a league-wide pitch mix, while remaining target-free.
    for batter_hand in sorted(history["batter_hand"].dropna().astype("string").unique()):
        subset = history[history["batter_hand"].astype("string") == batter_hand]
        bh_counts = subset.groupby([pid, "pitch_type_group"], observed=True).size().unstack(fill_value=0)
        bh_counts = bh_counts.reindex(columns=PITCH_GROUPS, fill_value=0)
        denom = bh_counts.sum(axis=1).replace(0, np.nan)
        for group in PITCH_GROUPS:
            out[f"tm_domain_bh{batter_hand}_{group}_rate"] = bh_counts[group] / denom

    # Count-conditioned selection rates.  Counts are low-cardinality and give
    # a compact proxy for pitch-calling consistency without needing location.
    count_frame = history.assign(__count=history["balls_before"].astype("string") + "_" + history["strikes_before"].astype("string"))
    cc = count_frame.groupby([pid, "__count", "pitch_type_group"], observed=True).size().unstack(fill_value=0)
    cc = cc.reindex(columns=PITCH_GROUPS, fill_value=0).reset_index()
    for count_name, count_rows in cc.groupby("__count", observed=True):
        count_rows = count_rows.set_index(pid)
        denom = count_rows[PITCH_GROUPS].sum(axis=1).replace(0, np.nan)
        for group in PITCH_GROUPS:
            out[f"tm_domain_count_{count_name}_{group}_rate"] = count_rows[group] / denom

    return out.reset_index()


def build_domain_profile_table(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    origins: Iterable[int],
    spec: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build enriched, snapshot-specific pitcher profiles and linkage rows."""
    # Reuse the tested linkage implementation for the anonymous ID problem.
    base_profiles, linkage = build_pitcher_profile_table(main, trackman, origins, spec)
    profile_pieces: list[pd.DataFrame] = []
    for origin in sorted({int(value) for value in origins}):
        mapping = linkage[linkage["season"] == origin].copy()
        if mapping.empty:
            continue
        history = trackman[trackman["season"] < origin]
        if history.empty:
            continue
        enriched = _base_profile(history)
        merged = mapping.merge(enriched, on="pitcher_trackman_id", how="left", validate="many_to_one")
        feature_columns = [
            c for c in merged.columns if c.startswith("tm_domain_")
        ]
        merged.loc[merged["tm_linked"] == 0, feature_columns] = np.nan
        keep = ["season", "pitcher_id", *feature_columns]
        profile_pieces.append(merged[keep])
    enriched_profiles = pd.concat(profile_pieces, ignore_index=True) if profile_pieces else pd.DataFrame()
    if not enriched_profiles.empty and enriched_profiles.duplicated(["season", "pitcher_id"]).any():
        raise RuntimeError("domain profile table contains duplicate keys")
    if not base_profiles.empty:
        # Keep linkage diagnostics and the established physical/repertoire
        # features too.  The new names are prefixed ``tm_domain_`` so there is
        # no column collision with the tested profile table.
        base_keep = [c for c in base_profiles.columns if c not in {"season", "pitcher_id"}]
        enriched_profiles = enriched_profiles.merge(
            base_profiles[["season", "pitcher_id", *base_keep]],
            on=["season", "pitcher_id"],
            how="left",
            validate="one_to_one",
        )
    return enriched_profiles, linkage


def main() -> None:
    import argparse
    import json

    from src.archive.data import read_main, read_trackman

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/followup/trackman_domain_profiles.csv"))
    parser.add_argument("--origins", nargs="+", type=int, default=list(range(2020, 2026)))
    args = parser.parse_args()
    project = args.project_dir.resolve()
    config = json.loads((project / "research/configs/trackman_linkage.json").read_text(encoding="utf-8"))
    main_df = read_main(project / "data/train.csv")
    tm_df = read_trackman(project / "data/trackman_history.csv")
    profiles, linkage = build_domain_profile_table(main_df, tm_df, args.origins, config["linkage"])
    output = args.output if args.output.is_absolute() else project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output, index=False, encoding="utf-8")
    linkage.to_csv(project / "research/reports/trackman_domain_linkage.csv", index=False, encoding="utf-8")
    print(f"wrote {output} shape={profiles.shape}")


if __name__ == "__main__":
    main()

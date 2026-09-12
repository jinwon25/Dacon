"""Leakage-safe longitudinal linkage to official Trackman pitcher profiles."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.archive.data import TARGET_COL, read_main, read_trackman
from src.archive.followup import _load_all_caches, _upsert_csv
from src.metrics import brier_score
from src.archive.train import train_lgb_holdout
from src.archive.validation import walk_forward_splits


PITCH_GROUPS = ["fastball", "breaking", "offspeed"]
PHYSICAL_MEASURES = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
MAIN_LINKAGE_COLUMNS = [
    "season",
    "pitcher_id",
    "pitcher_hand",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
TRACKMAN_LINKAGE_COLUMNS = [
    "season",
    "pitcher_trackman_id",
    "pitcher_hand",
    "pitch_type_group",
]


def main_annual_fingerprints(main: pd.DataFrame) -> pd.DataFrame:
    """Recover annual pitch-group trajectories from pre-pitch cumulative fields."""
    missing = set(MAIN_LINKAGE_COLUMNS) - set(main.columns)
    if missing:
        raise ValueError(f"main linkage columns missing: {sorted(missing)}")
    indices = main.groupby(["pitcher_id", "season"], observed=True)[
        "asof_pitcher_pitchmix_n"
    ].idxmax()
    annual = main.loc[indices, MAIN_LINKAGE_COLUMNS].copy()
    annual = annual.sort_values(["pitcher_id", "season"]).reset_index(drop=True)
    annual["n_cumulative"] = annual["asof_pitcher_pitchmix_n"].astype(float)
    for group in PITCH_GROUPS:
        annual[f"{group}_cumulative"] = (
            annual["n_cumulative"]
            * annual[f"asof_pitcher_{group}_rate"].astype(float)
        )
    grouped = annual.groupby("pitcher_id", observed=True)
    annual["n"] = grouped["n_cumulative"].diff().fillna(annual["n_cumulative"])
    for group in PITCH_GROUPS:
        cumulative = f"{group}_cumulative"
        annual[group] = grouped[cumulative].diff().fillna(annual[cumulative])
    count_columns = ["n", *PITCH_GROUPS]
    annual[count_columns] = annual[count_columns].clip(lower=0.0)
    denominator = annual[PITCH_GROUPS].sum(axis=1).replace(0.0, np.nan)
    for group in PITCH_GROUPS:
        annual[f"{group}_rate"] = annual[group] / denominator
    return annual[
        [
            "pitcher_id",
            "season",
            "pitcher_hand",
            "n",
            *PITCH_GROUPS,
            *[f"{group}_rate" for group in PITCH_GROUPS],
        ]
    ]


def trackman_annual_fingerprints(trackman: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build annual Trackman pitch-group trajectories and stable hand labels."""
    missing = set(TRACKMAN_LINKAGE_COLUMNS) - set(trackman.columns)
    if missing:
        raise ValueError(f"Trackman linkage columns missing: {sorted(missing)}")
    annual = (
        trackman.groupby(
            ["pitcher_trackman_id", "season", "pitch_type_group"], observed=True
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for group in [*PITCH_GROUPS, "other"]:
        if group not in annual:
            annual[group] = 0
    annual["n"] = annual[[*PITCH_GROUPS, "other"]].sum(axis=1).astype(float)
    denominator = annual[PITCH_GROUPS].sum(axis=1).replace(0.0, np.nan)
    for group in PITCH_GROUPS:
        annual[f"{group}_rate"] = annual[group] / denominator
    annual_hands = (
        trackman.groupby(
            ["pitcher_trackman_id", "season"], observed=True
        )["pitcher_hand"]
        .agg(lambda values: str(values.mode().iloc[0]))
        .rename("pitcher_hand")
        .reset_index()
    )
    annual = annual.merge(
        annual_hands,
        on=["pitcher_trackman_id", "season"],
        how="left",
        validate="one_to_one",
    )
    hands = (
        trackman.groupby("pitcher_trackman_id", observed=True)["pitcher_hand"]
        .agg(lambda values: str(values.mode().iloc[0]))
    )
    return annual, hands


def _trajectory_tensor(
    frame: pd.DataFrame,
    id_column: str,
    identifiers: list[int],
    years: list[int],
) -> np.ndarray:
    tensor = np.full((len(identifiers), len(years), 4), np.nan, dtype=np.float64)
    id_position = {value: index for index, value in enumerate(identifiers)}
    year_position = {value: index for index, value in enumerate(years)}
    columns = ["n", *[f"{group}_rate" for group in PITCH_GROUPS]]
    for row in frame[[id_column, "season", *columns]].itertuples(index=False):
        identifier = int(getattr(row, id_column))
        season = int(row.season)
        values = np.asarray(
            [float(getattr(row, column)) for column in columns], dtype=np.float64
        )
        # A zero-delta cumulative snapshot carries no annual pitch-mix signal.
        if (
            identifier in id_position
            and season in year_position
            and values[0] > 0
            and np.isfinite(values).all()
        ):
            tensor[id_position[identifier], year_position[season], :] = values
    return tensor


def _linkage_cost(
    main_tensor: np.ndarray,
    trackman_tensor: np.ndarray,
    spec: dict[str, Any],
) -> np.ndarray:
    active_main = np.isfinite(main_tensor[:, :, 0])
    active_trackman = np.isfinite(trackman_tensor[:, :, 0])
    total = np.zeros((len(main_tensor), len(trackman_tensor)), dtype=np.float64)
    common = np.zeros_like(total)
    union = np.zeros_like(total)
    for year_position in range(main_tensor.shape[1]):
        main_active = active_main[:, year_position, None]
        trackman_active = active_trackman[None, :, year_position]
        both = main_active & trackman_active
        rate_distance = np.sqrt(
            np.square(
                main_tensor[:, None, year_position, 1:]
                - trackman_tensor[None, :, year_position, 1:]
            ).sum(axis=2)
        )
        count_distance = np.abs(
            np.log1p(main_tensor[:, None, year_position, 0])
            - np.log1p(trackman_tensor[None, :, year_position, 0])
        )
        distance = rate_distance + float(spec["log_count_weight"]) * count_distance
        total += np.where(both, distance, 0.0)
        common += both
        union += main_active | trackman_active
    trajectory = np.divide(
        total,
        common,
        out=np.ones_like(total),
        where=common > 0,
    )
    activity = float(spec["activity_penalty"]) * (
        1.0 - common / np.maximum(union, 1.0)
    )

    main_counts = np.nansum(
        main_tensor[:, :, 0, None] * main_tensor[:, :, 1:], axis=1
    )
    trackman_counts = np.nansum(
        trackman_tensor[:, :, 0, None] * trackman_tensor[:, :, 1:], axis=1
    )
    main_total = main_counts.sum(axis=1)
    trackman_total = trackman_counts.sum(axis=1)
    main_rates = main_counts / np.maximum(main_total[:, None], 1.0)
    trackman_rates = trackman_counts / np.maximum(trackman_total[:, None], 1.0)
    cumulative = np.sqrt(
        np.square(main_rates[:, None, :] - trackman_rates[None, :, :]).sum(axis=2)
    ) + float(spec["log_count_weight"]) * np.abs(
        np.log1p(main_total[:, None]) - np.log1p(trackman_total[None, :])
    )
    return (
        float(spec["annual_weight"]) * trajectory
        + float(spec["cumulative_weight"]) * cumulative
        + activity
    )


def link_pitchers(
    main_annual: pd.DataFrame,
    trackman_annual: pd.DataFrame,
    trackman_hands: pd.Series,
    forecast_season: int,
    spec: dict[str, Any],
) -> pd.DataFrame:
    """One-to-one longitudinal linkage using only seasons before the forecast."""
    years = sorted(
        int(value)
        for value in main_annual.loc[
            main_annual["season"] < forecast_season, "season"
        ].unique()
    )
    if not years:
        return pd.DataFrame(
            columns=[
                "season",
                "pitcher_id",
                "pitcher_trackman_id",
                "tm_link_distance",
                "tm_link_margin",
                "tm_link_assignment_rank",
                "tm_link_confidence",
                "tm_linked",
            ]
        )
    main_history = main_annual[main_annual["season"] < forecast_season]
    trackman_history = trackman_annual[
        trackman_annual["season"] < forecast_season
    ]
    # IDs occasionally carry inconsistent hand labels across seasons. Resolve each
    # pitcher to exactly one modal hand using only history available at the origin;
    # otherwise the same anonymous pitcher can enter both assignment pools.
    main_hands = main_history.groupby("pitcher_id", observed=True)["pitcher_hand"].agg(
        lambda values: int(values.mode().iloc[0])
    )
    if "pitcher_hand" in trackman_history:
        prior_trackman_hands = trackman_history.groupby(
            "pitcher_trackman_id", observed=True
        )["pitcher_hand"].agg(lambda values: str(values.mode().iloc[0]))
    else:
        # Backward-compatible fallback for callers with precomputed annual tables.
        prior_trackman_hands = trackman_hands
    rows: list[dict[str, Any]] = []
    for main_hand, trackman_hand in ((1, "Left"), (2, "Right")):
        main_ids = sorted(
            int(value)
            for value in main_hands[main_hands == main_hand].index
        )
        trackman_ids = sorted(
            int(value)
            for value in prior_trackman_hands[
                prior_trackman_hands == trackman_hand
            ].index
        )
        if not main_ids or not trackman_ids:
            continue
        main_tensor = _trajectory_tensor(
            main_history[main_history["pitcher_id"].isin(main_ids)],
            "pitcher_id",
            main_ids,
            years,
        )
        trackman_tensor = _trajectory_tensor(
            trackman_history[
                trackman_history["pitcher_trackman_id"].isin(trackman_ids)
            ],
            "pitcher_trackman_id",
            trackman_ids,
            years,
        )
        cost = _linkage_cost(main_tensor, trackman_tensor, spec)
        main_positions, trackman_positions = linear_sum_assignment(cost)
        nearest = np.sort(cost, axis=1)[:, :2]
        for main_position, trackman_position in zip(
            main_positions, trackman_positions
        ):
            distance = float(cost[main_position, trackman_position])
            assignment_rank = int(
                np.count_nonzero(cost[main_position] < distance - 1e-12) + 1
            )
            linked = (
                distance <= float(spec["maximum_distance"])
                and assignment_rank <= int(spec["maximum_assignment_rank"])
            )
            if not linked:
                confidence = "unmatched"
            elif distance <= float(spec["high_confidence_distance"]):
                confidence = "high"
            elif distance <= float(spec["medium_confidence_distance"]):
                confidence = "medium"
            else:
                confidence = "low"
            rows.append(
                {
                    "season": forecast_season,
                    "pitcher_id": main_ids[main_position],
                    "pitcher_trackman_id": trackman_ids[trackman_position],
                    "tm_link_distance": distance,
                    "tm_link_margin": float(
                        nearest[main_position, 1] - nearest[main_position, 0]
                    ),
                    "tm_link_assignment_rank": assignment_rank,
                    "tm_link_confidence": confidence,
                    "tm_linked": int(linked),
                }
            )
    result = pd.DataFrame(rows)
    if result["pitcher_id"].duplicated().any():
        raise RuntimeError("linkage produced duplicate pitcher_id rows")
    return result


def _physical_profiles(history: pd.DataFrame) -> pd.DataFrame:
    grouped = history.groupby("pitcher_trackman_id", observed=True)
    profile = grouped.size().rename("tm_pitcher_n").to_frame()
    overall = grouped[PHYSICAL_MEASURES].agg(["mean", "std"])
    overall.columns = [f"tm_{measure}_{stat}" for measure, stat in overall.columns]
    profile = profile.join(overall)
    for pitch_group in PITCH_GROUPS:
        subset = history[history["pitch_type_group"].astype("string") == pitch_group]
        group_stats = subset.groupby("pitcher_trackman_id", observed=True)[
            PHYSICAL_MEASURES
        ].mean()
        group_stats.columns = [
            f"tm_{pitch_group}_{measure}_mean" for measure in PHYSICAL_MEASURES
        ]
        profile = profile.join(group_stats)
    pitch_counts = (
        history.groupby(
            ["pitcher_trackman_id", "pitch_type_group"], observed=True
        )
        .size()
        .unstack(fill_value=0)
    )
    total = pitch_counts.sum(axis=1).replace(0, np.nan)
    for pitch_group in [*PITCH_GROUPS, "other"]:
        values = pitch_counts[pitch_group] if pitch_group in pitch_counts else 0
        profile[f"tm_{pitch_group}_rate"] = values / total
    profile["tm_tagged_repertoire"] = grouped["tagged_pitch_type"].nunique()
    profile["tm_auto_repertoire"] = grouped["auto_pitch_type"].nunique()
    latest_season = int(history["season"].max())
    latest = history[history["season"] == latest_season].groupby(
        "pitcher_trackman_id", observed=True
    )[PHYSICAL_MEASURES].mean()
    latest.columns = [f"tm_latest_{measure}_mean" for measure in PHYSICAL_MEASURES]
    profile = profile.join(latest)
    for measure in PHYSICAL_MEASURES:
        profile[f"tm_latest_{measure}_delta"] = (
            profile[f"tm_latest_{measure}_mean"] - profile[f"tm_{measure}_mean"]
        )
    return profile.reset_index()


def build_pitcher_profile_table(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    origins: Iterable[int],
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build snapshot-specific profiles and linkage diagnostics."""
    main_annual = main_annual_fingerprints(main)
    trackman_annual, trackman_hands = trackman_annual_fingerprints(trackman)
    profile_pieces: list[pd.DataFrame] = []
    linkage_pieces: list[pd.DataFrame] = []
    for origin in sorted(set(int(value) for value in origins)):
        mapping = link_pitchers(
            main_annual, trackman_annual, trackman_hands, origin, spec
        )
        linkage_pieces.append(mapping)
        history = trackman[trackman["season"] < origin]
        if history.empty or mapping.empty:
            continue
        profiles = _physical_profiles(history)
        merged = mapping.merge(
            profiles, on="pitcher_trackman_id", how="left", validate="many_to_one"
        )
        physical_columns = [
            column
            for column in merged.columns
            if column.startswith("tm_")
            and column
            not in {
                "tm_link_distance",
                "tm_link_margin",
                "tm_link_assignment_rank",
                "tm_link_confidence",
                "tm_linked",
            }
        ]
        merged.loc[merged["tm_linked"] == 0, physical_columns] = np.nan
        merged["tm_link_confidence_code"] = merged["tm_link_confidence"].map(
            {"unmatched": 0, "low": 1, "medium": 2, "high": 3}
        )
        keep = [
            "season",
            "pitcher_id",
            "tm_link_distance",
            "tm_link_margin",
            "tm_link_assignment_rank",
            "tm_linked",
            "tm_link_confidence_code",
            *physical_columns,
        ]
        profile_pieces.append(merged[keep])
    profiles = pd.concat(profile_pieces, ignore_index=True) if profile_pieces else pd.DataFrame()
    nonempty_linkage = [piece for piece in linkage_pieces if not piece.empty]
    linkage = (
        pd.concat(nonempty_linkage, ignore_index=True)
        if nonempty_linkage
        else pd.DataFrame()
    )
    if not profiles.empty and profiles.duplicated(["season", "pitcher_id"]).any():
        raise RuntimeError("profile table contains duplicate season/pitcher_id keys")
    return profiles, linkage


def linkage_stability(linkage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seasons = sorted(int(value) for value in linkage["season"].unique())
    for first, second in zip(seasons, seasons[1:]):
        left = linkage[
            (linkage["season"] == first) & (linkage["tm_linked"] == 1)
        ]
        right = linkage[linkage["season"] == second]
        joined = left.merge(right, on="pitcher_id", suffixes=("_first", "_second"))
        for confidence in ("high", "medium", "low", "all"):
            subset = (
                joined
                if confidence == "all"
                else joined[joined["tm_link_confidence_first"] == confidence]
            )
            rows.append(
                {
                    "first_origin": first,
                    "second_origin": second,
                    "confidence": confidence,
                    "n_links": len(subset),
                    "stable_rate": float(
                        (
                            subset["pitcher_trackman_id_first"]
                            == subset["pitcher_trackman_id_second"]
                        ).mean()
                    )
                    if len(subset)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(
    project_dir: Path,
    config_path: Path,
    validation_seasons: list[int] | None = None,
    build_only: bool = False,
) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    years = validation_seasons or [
        int(value) for value in config["validation_seasons"]
    ]
    print("[Trackman linkage] Loading official data...")
    train = read_main(project_dir / "data" / "train.csv")
    trackman = read_trackman(project_dir / "data" / "trackman_history.csv")
    origins = range(int(train["season"].min()), int(train["season"].max()) + 2)
    profiles, linkage = build_pitcher_profile_table(
        train, trackman, origins, config["linkage"]
    )
    artifacts = project_dir / "artifacts" / "followup"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports = project_dir / "research" / "reports"
    profiles.to_csv(
        artifacts / "trackman_pitcher_profiles.csv", index=False, encoding="utf-8"
    )
    linkage.to_csv(reports / "trackman_linkage.csv", index=False, encoding="utf-8")
    stability = linkage_stability(linkage)
    stability.to_csv(
        reports / "trackman_linkage_stability.csv", index=False, encoding="utf-8"
    )
    print(stability.to_string(index=False))
    del trackman
    gc.collect()
    if build_only:
        return pd.DataFrame()

    folds = {
        fold.validation_season: fold
        for fold in walk_forward_splits(train, validation_seasons=tuple(years))
    }
    followup_config = json.loads(
        (project_dir / "research" / "configs" / "followup.json").read_text(encoding="utf-8")
    )
    _, caches = _load_all_caches(project_dir, followup_config, train)
    rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_dir = artifacts / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        print(f"[Trackman linkage] Training validation {year}...")
        fold = folds[year]
        result = train_lgb_holdout(
            train,
            fold.train_idx,
            fold.valid_idx,
            config["model"],
            int(config["max_boost_rounds"]),
            int(config["early_stopping_rounds"]),
            profiles,
        )
        prediction = np.asarray(result["prediction"], dtype=np.float64)
        target = caches[year]["target"]
        incumbent = caches[year]["incumbent"]
        np.savez_compressed(
            model_dir / f"lgb_trackman_pitcher_v1_validate_{year}.npz",
            prediction=prediction,
            target=target,
            valid_idx=fold.valid_idx,
            best_iteration=np.array([result["best_iteration"]]),
        )
        rows.append(
            {
                "candidate": "lgb_trackman_pitcher_v1",
                "outer_validation_season": year,
                "n_rows": len(target),
                "brier": brier_score(target, prediction),
                "incumbent_brier": brier_score(target, incumbent),
                "delta_brier": brier_score(target, prediction)
                - brier_score(target, incumbent),
                "best_iteration": result["best_iteration"],
                "training_seconds": result["fit_seconds"],
                "build_seconds": result["build_seconds"],
                "peak_memory_mb": result["peak_memory_mb"],
            }
        )
        gains = result["booster"].feature_importance(importance_type="gain")
        for feature, gain in zip(result["feature_names"], gains):
            importance_rows.append(
                {
                    "outer_validation_season": year,
                    "feature": feature,
                    "gain": float(gain),
                }
            )
        del result
        gc.collect()
    frame = pd.DataFrame(rows)
    _upsert_csv(
        reports / "trackman_pitcher_results.csv",
        frame,
        ["candidate", "outer_validation_season"],
    )
    _upsert_csv(
        reports / "trackman_pitcher_importance.csv",
        pd.DataFrame(importance_rows),
        ["outer_validation_season", "feature"],
    )
    print(frame.to_string(index=False))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--config", type=Path, default=Path("research/configs/trackman_linkage.json")
    )
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    config_path = args.config if args.config.is_absolute() else project_dir / args.config
    run(project_dir, config_path, args.years, args.build_only)


if __name__ == "__main__":
    main()

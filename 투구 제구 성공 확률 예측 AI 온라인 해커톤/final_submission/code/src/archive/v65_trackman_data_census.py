"""Comprehensive, target-safe census of the official TrackMan pathways.

The hidden test does not expose current-pitch TrackMan measurements or a
shared TrackMan identifier.  This audit therefore separates three questions:

1. how much of labelled train is structurally aligned one-to-one;
2. how accurately a strictly pre-origin alignment-derived player map can
   transfer historical TrackMan profiles to one current official row; and
3. whether physical measurements have within-pitcher/pitch-type signal and
   persist from one season to the next.

The script never reads ``test.csv`` and never uses an audit season to build
its player map.  Outputs are small aggregate reports, not model artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PHYSICAL_COLUMNS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
MAIN_COLUMNS = (
    "season",
    "game_month",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "control_success",
)
TRACKMAN_COLUMNS = (
    "season",
    "trackman_game_id",
    "pitcher_trackman_id",
    "batter_trackman_id",
    "pitch_type_group",
    *PHYSICAL_COLUMNS,
)
MAP_THRESHOLDS = (5, 20, 100)
MIN_PURITY = 0.99


def derive_temporal_entity_map(
    pairs: pd.DataFrame,
    origin: int,
    *,
    main_column: str = "pitcher_id",
    trackman_column: str = "pitcher_trackman_id",
    minimum_support: int = 20,
    minimum_purity: float = MIN_PURITY,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Derive a majority identity map using aligned rows before ``origin``."""

    source = pairs.loc[pairs["season"].lt(int(origin))]
    counts = (
        source.groupby([main_column, trackman_column], observed=True)
        .size()
        .rename("support")
        .reset_index()
    )
    if counts.empty:
        return pd.DataFrame(), {
            "origin": int(origin),
            "source_pairs": 0,
            "observed_entities": 0,
            "accepted_entities": 0,
            "weighted_majority_purity": float("nan"),
        }
    totals = counts.groupby(main_column, observed=True)["support"].sum().rename(
        "total_support"
    )
    best = (
        counts.sort_values([main_column, "support", trackman_column], kind="stable")
        .groupby(main_column, observed=True)
        .tail(1)
        .set_index(main_column)
        .join(totals)
        .reset_index()
    )
    best["purity"] = best["support"] / best["total_support"]
    best["accepted"] = best["total_support"].ge(int(minimum_support)) & best[
        "purity"
    ].ge(float(minimum_purity))
    accepted = best.loc[best["accepted"]].copy()
    audit = {
        "origin": int(origin),
        "source_pairs": int(len(source)),
        "observed_entities": int(len(best)),
        "accepted_entities": int(len(accepted)),
        "minimum_support": int(minimum_support),
        "minimum_purity": float(minimum_purity),
        "weighted_majority_purity": float(best["support"].sum() / best["total_support"].sum()),
        "accepted_weighted_purity": float(
            accepted["support"].sum() / accepted["total_support"].sum()
        )
        if len(accepted)
        else float("nan"),
        "minimum_accepted_purity": float(accepted["purity"].min())
        if len(accepted)
        else float("nan"),
    }
    return accepted, audit


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(right, errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0.0 or np.std(y[valid]) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def within_group_correlation(
    frame: pd.DataFrame, feature: str, target: str, groups: list[str]
) -> float:
    """Correlation after removing season/player/pitch-type group means."""

    local = frame[[*groups, feature, target]].dropna(subset=[feature, target]).copy()
    if len(local) < 3:
        return float("nan")
    grouped = local.groupby(groups, observed=True)
    feature_residual = local[feature] - grouped[feature].transform("mean")
    target_residual = local[target] - grouped[target].transform("mean")
    return safe_correlation(feature_residual, target_residual)


def _load_alignment_pairs(
    alignment_path: Path, main: pd.DataFrame, trackman: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    with np.load(alignment_path, allow_pickle=False) as saved:
        main_index = saved["main_index"].astype(np.int64)
        trackman_index = saved["trackman_index"].astype(np.int64)
        season = saved["season"].astype(np.int16)
        game_dice = saved["game_dice"].astype(np.float64)
        game_margin = saved["game_margin"].astype(np.float64)
    if len(main_index) != len(trackman_index):
        raise ValueError("alignment index length mismatch")
    if len(np.unique(main_index)) != len(main_index) or len(np.unique(trackman_index)) != len(
        trackman_index
    ):
        raise ValueError("alignment is not one-to-one")
    left = main.iloc[main_index].reset_index(drop=True)
    right = trackman.iloc[trackman_index].reset_index(drop=True)
    if not np.array_equal(left["season"].to_numpy(np.int16), season):
        raise ValueError("alignment/main season mismatch")
    if not np.array_equal(right["season"].to_numpy(np.int16), season):
        raise ValueError("alignment/TrackMan season mismatch")
    pairs = pd.DataFrame(
        {
            "season": season,
            "pitcher_id": left["pitcher_id"].to_numpy(),
            "batter_id": left["batter_id"].to_numpy(),
            "pitcher_trackman_id": right["pitcher_trackman_id"].to_numpy(),
            "batter_trackman_id": right["batter_trackman_id"].to_numpy(),
            "game_type": left["game_type"].astype(str).to_numpy(),
            "balls_before": left["balls_before"].to_numpy(),
            "strikes_before": left["strikes_before"].to_numpy(),
            "batter_hand": left["batter_hand"].to_numpy(),
            "control_success": left["control_success"].to_numpy(np.float64),
            "pitch_type_group": right["pitch_type_group"].astype(str).to_numpy(),
        }
    )
    for column in PHYSICAL_COLUMNS:
        pairs[column] = pd.to_numeric(right[column], errors="coerce").to_numpy(
            np.float64
        )
    audit = {
        "aligned_rows": int(len(pairs)),
        "main_index_unique": True,
        "trackman_index_unique": True,
        "game_dice_min": float(np.min(game_dice)),
        "game_dice_p50": float(np.median(game_dice)),
        "game_dice_p05": float(np.quantile(game_dice, 0.05)),
        "game_margin_min": float(np.min(game_margin)),
        "game_margin_p50": float(np.median(game_margin)),
        "game_margin_p05": float(np.quantile(game_margin, 0.05)),
    }
    return pairs, audit


def _season_domain_table(main: pd.DataFrame) -> pd.DataFrame:
    return (
        main.groupby(["season", "game_type"], observed=True)["control_success"]
        .agg(rows="size", target_rate="mean", pitchers=lambda _: np.nan)
        .drop(columns="pitchers")
        .reset_index()
        .merge(
            main.groupby(["season", "game_type"], observed=True)["pitcher_id"]
            .nunique()
            .rename("pitchers")
            .reset_index(),
            on=["season", "game_type"],
            validate="one_to_one",
        )
    )


def _trackman_season_pitch_table(trackman: pd.DataFrame) -> pd.DataFrame:
    output = (
        trackman.groupby(["season", "pitch_type_group"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = output.groupby("season", observed=True)["rows"].transform("sum")
    output["row_fraction"] = output["rows"] / totals
    return output


def _mapping_audit(
    pairs: pd.DataFrame,
    main: pd.DataFrame,
    hungarian: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for origin in range(2021, 2026):
        query_year = origin if origin <= int(main["season"].max()) else int(main["season"].max())
        query = main.loc[main["season"].eq(query_year)]
        for minimum_support in MAP_THRESHOLDS:
            mapping, audit = derive_temporal_entity_map(
                pairs, origin, minimum_support=minimum_support
            )
            accepted_ids = set(mapping["pitcher_id"].tolist())
            query_covered = query["pitcher_id"].isin(accepted_ids)
            item: dict[str, object] = {
                **audit,
                "query_basis": "same_origin" if origin <= 2024 else "2024_proxy",
                "query_rows": int(len(query)),
                "query_row_coverage": float(query_covered.mean()),
                "query_pitcher_coverage": float(
                    query.loc[query_covered, "pitcher_id"].nunique()
                    / max(1, query["pitcher_id"].nunique())
                ),
                "hungarian_linked_entities": np.nan,
                "direct_hungarian_agreement": np.nan,
            }
            if hungarian is not None and minimum_support == 20:
                legacy = hungarian.loc[
                    hungarian["season"].eq(origin) & hungarian["tm_linked"].eq(1)
                ]
                joined = mapping.merge(
                    legacy[["pitcher_id", "pitcher_trackman_id"]],
                    on="pitcher_id",
                    suffixes=("_direct", "_hungarian"),
                )
                item["hungarian_linked_entities"] = int(len(legacy))
                item["direct_hungarian_overlap"] = int(len(joined))
                item["direct_hungarian_agreement"] = float(
                    (
                        joined["pitcher_trackman_id_direct"]
                        == joined["pitcher_trackman_id_hungarian"]
                    ).mean()
                ) if len(joined) else np.nan
            rows.append(item)
    return pd.DataFrame(rows)


def _physical_signal_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["season", "pitcher_id", "pitch_type_group"]
    for season, local in pairs.groupby("season", observed=True):
        for feature in PHYSICAL_COLUMNS:
            rows.append(
                {
                    "season": int(season),
                    "feature": feature,
                    "rows": int(local[[feature, "control_success"]].dropna().shape[0]),
                    "raw_target_correlation": safe_correlation(
                        local[feature], local["control_success"]
                    ),
                    "within_pitcher_pitchtype_correlation": within_group_correlation(
                        local, feature, "control_success", groups
                    ),
                }
            )
    return pd.DataFrame(rows)


def _profile_stability_table(pairs: pd.DataFrame) -> pd.DataFrame:
    annual = (
        pairs.groupby(["season", "pitcher_id", "pitch_type_group"], observed=True)[
            list(PHYSICAL_COLUMNS)
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    seasons = sorted(int(value) for value in annual["season"].unique())
    for first, second in zip(seasons, seasons[1:]):
        left = annual.loc[annual["season"].eq(first)].drop(columns="season")
        right = annual.loc[annual["season"].eq(second)].drop(columns="season")
        joined = left.merge(
            right,
            on=["pitcher_id", "pitch_type_group"],
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )
        for feature in PHYSICAL_COLUMNS:
            rows.append(
                {
                    "first_season": first,
                    "second_season": second,
                    "feature": feature,
                    "pitcher_pitchtype_pairs": int(len(joined)),
                    "correlation": safe_correlation(
                        joined[f"{feature}_first"], joined[f"{feature}_second"]
                    ),
                    "mean_abs_change": float(
                        np.nanmean(
                            np.abs(
                                joined[f"{feature}_second"].to_numpy(np.float64)
                                - joined[f"{feature}_first"].to_numpy(np.float64)
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pitch_type_target_table(pairs: pd.DataFrame) -> pd.DataFrame:
    return (
        pairs.groupby(["season", "pitch_type_group"], observed=True)["control_success"]
        .agg(rows="size", target_rate="mean")
        .reset_index()
    )


def run(
    project: Path,
    alignment_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main = pd.read_csv(
        project / "data" / "train.csv", usecols=list(MAIN_COLUMNS), low_memory=False
    )
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=list(TRACKMAN_COLUMNS),
        low_memory=False,
    )
    pairs, alignment_audit = _load_alignment_pairs(
        alignment_dir / "pitch_alignment.npz", main, trackman
    )
    hungarian_path = project / "reports" / "trackman_linkage.csv"
    hungarian = pd.read_csv(hungarian_path) if hungarian_path.exists() else None

    season_domain = _season_domain_table(main)
    season_pitch = _trackman_season_pitch_table(trackman)
    mapping = _mapping_audit(pairs, main, hungarian)
    physical = _physical_signal_table(pairs)
    stability = _profile_stability_table(pairs)
    pitch_target = _pitch_type_target_table(pairs)

    season_domain.to_csv(output_dir / "train_season_domain.csv", index=False)
    season_pitch.to_csv(output_dir / "trackman_season_pitch_type.csv", index=False)
    mapping.to_csv(output_dir / "alignment_mapping_by_origin.csv", index=False)
    physical.to_csv(output_dir / "physical_target_signal.csv", index=False)
    stability.to_csv(output_dir / "physical_profile_stability.csv", index=False)
    pitch_target.to_csv(output_dir / "pitch_type_target_rate.csv", index=False)

    primary_mapping = mapping.loc[mapping["minimum_support"].eq(20)]
    result: dict[str, object] = {
        "protocol": "V65_OFFICIAL_TRACKMAN_DATA_CENSUS_V1",
        "test_csv_read": False,
        "data": {
            "train_rows": int(len(main)),
            "train_columns_total": 49,
            "train_seasons": [int(value) for value in sorted(main["season"].unique())],
            "trackman_rows": int(len(trackman)),
            "trackman_columns_total": 30,
            "trackman_seasons": [
                int(value) for value in sorted(trackman["season"].unique())
            ],
            "trackman_games": int(trackman["trackman_game_id"].nunique()),
            "trackman_pitchers": int(trackman["pitcher_trackman_id"].nunique()),
            "trackman_has_plate_location": False,
            "trackman_has_intended_target": False,
            "test_has_current_pitch_physics": False,
            "test_has_trackman_id": False,
        },
        "alignment": {
            **alignment_audit,
            "row_coverage": float(len(pairs) / len(main)),
            "target_columns_used_for_alignment": False,
        },
        "direct_mapping_support20": primary_mapping.to_dict(orient="records"),
        "physical_signal": {
            "maximum_raw_abs_correlation": float(
                physical["raw_target_correlation"].abs().max()
            ),
            "maximum_within_abs_correlation": float(
                physical["within_pitcher_pitchtype_correlation"].abs().max()
            ),
            "median_profile_yoy_correlation": float(
                stability["correlation"].median()
            ),
            "minimum_profile_yoy_correlation": float(
                stability["correlation"].min()
            ),
        },
        "pathway_conclusions": [
            "Current-pitch physics and plate/intended location are unavailable at inference.",
            "Alignment-derived pre-origin pitcher maps can replace low-coverage pitch-mix Hungarian linkage.",
            "Current-pitch privileged distillation is an upper-bound path, not a deployable direct feature path.",
            "Deployable candidates must be frozen historical profiles or row-local mixtures predicted from official columns.",
        ],
        "row_local_deployment_possible": True,
        "other_test_rows_required": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.output_dir)


if __name__ == "__main__":
    main()

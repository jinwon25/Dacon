from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from experiments.gefs_spread_residual_screen import (
    CAPACITY,
    H2_START,
    Q1_END,
    TARGET,
    _apply_policy,
    _calendar_features,
    _compare,
    _fit_members,
    _select_policy,
)


def build_disagreement_features(cfsv2_path: Path, gfs_path: Path) -> pd.DataFrame:
    cfs = pd.read_csv(cfsv2_path, encoding="utf-8-sig")
    gfs_columns = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
        "heightAboveGround_100_100u",
        "heightAboveGround_100_100v",
    ]
    gfs = pd.read_csv(gfs_path, encoding="utf-8-sig", usecols=gfs_columns)
    keys = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
    ]
    for frame in (cfs, gfs):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
        frame["data_available_kst_dtm"] = pd.to_datetime(
            frame["data_available_kst_dtm"]
        )
    if cfs.duplicated(keys).any() or gfs.duplicated(keys).any():
        raise ValueError("CFSv2/GFS source rows must be unique on the causal join keys")
    merged = cfs.merge(gfs, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(cfs):
        raise ValueError("CFSv2 and supplied GFS rows are not exactly aligned")
    merged = merged.rename(
        columns={
            "heightAboveGround_10_10u": "gfs_u10",
            "heightAboveGround_10_10v": "gfs_v10",
            "heightAboveGround_100_100u": "gfs_u100",
            "heightAboveGround_100_100v": "gfs_v100",
        }
    )
    merged["gfs_speed10"] = np.hypot(merged["gfs_u10"], merged["gfs_v10"])
    merged["gfs_speed100"] = np.hypot(merged["gfs_u100"], merged["gfs_v100"])
    merged["delta_u10"] = merged["cfsv2_u10"] - merged["gfs_u10"]
    merged["delta_v10"] = merged["cfsv2_v10"] - merged["gfs_v10"]
    merged["vector_disagreement"] = np.hypot(
        merged["delta_u10"], merged["delta_v10"]
    )
    merged["speed_disagreement"] = (
        merged["cfsv2_speed10"] - merged["gfs_speed10"]
    )
    denominator = np.maximum(
        merged["cfsv2_speed10"] * merged["gfs_speed10"], 1e-6
    )
    merged["direction_cosine"] = (
        merged["cfsv2_u10"] * merged["gfs_u10"]
        + merged["cfsv2_v10"] * merged["gfs_v10"]
    ) / denominator
    merged["direction_cosine"] = merged["direction_cosine"].clip(-1.0, 1.0)
    merged["cfsv2_vs_gfs100_speed_ratio"] = merged["cfsv2_speed10"] / np.maximum(
        merged["gfs_speed100"], 0.25
    )
    raw = [
        "cfsv2_u10",
        "cfsv2_v10",
        "cfsv2_speed10",
        "delta_u10",
        "delta_v10",
        "vector_disagreement",
        "speed_disagreement",
        "direction_cosine",
        "cfsv2_vs_gfs100_speed_ratio",
    ]
    aggregate = merged.groupby("forecast_kst_dtm")[raw].agg(
        ["mean", "min", "max", "std"]
    )
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    center = (
        merged.loc[merged["grid_id"] == 5, ["forecast_kst_dtm"] + raw]
        .drop_duplicates("forecast_kst_dtm")
        .set_index("forecast_kst_dtm")
        .add_suffix("_center")
    )
    result = aggregate.join(center, how="inner").sort_index()
    issue = (
        merged[["forecast_kst_dtm", "data_available_kst_dtm"]]
        .drop_duplicates("forecast_kst_dtm")
        .set_index("forecast_kst_dtm")
    )
    result.attrs["issue_times"] = issue.reindex(result.index)[
        "data_available_kst_dtm"
    ].to_numpy()
    if result.empty or result.isna().any().any():
        raise ValueError("hourly CFSv2/GFS disagreement features are incomplete")
    return result


def _selection_family(
    name: str,
    features: pd.DataFrame,
    truth: np.ndarray,
    base: np.ndarray,
    train: np.ndarray,
    selection: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    seed_predictions, fits = _fit_members(
        features, truth - base, train, selection, seeds
    )
    prediction = seed_predictions.mean(axis=0)
    policy, leaderboard = _select_policy(truth, base, prediction, selection)
    if policy is None:
        return {
            "name": name,
            "feature_count": int(features.shape[1]),
            "fits": fits,
            "status": "rejected_no_all_component_q2_policy",
            "policy": None,
            "leaderboard": leaderboard,
            "selection": None,
            "seed_selection": [],
        }
    candidate, gate = _apply_policy(base, prediction, policy)
    selected = _compare(truth, base, candidate, selection)
    seed_rows = []
    for seed, seed_prediction in zip(seeds, seed_predictions):
        seed_candidate, seed_gate = _apply_policy(base, seed_prediction, policy)
        seed_rows.append(
            {
                "seed": seed,
                "changed_ratio": float((seed_gate & selection).sum() / selection.sum()),
                "delta": _compare(truth, base, seed_candidate, selection)["delta"],
            }
        )
    return {
        "name": name,
        "feature_count": int(features.shape[1]),
        "fits": fits,
        "status": "passed_q2",
        "policy": asdict(policy),
        "leaderboard": leaderboard,
        "selection": selected,
        "selection_changed_ratio": float((gate & selection).sum() / selection.sum()),
        "seed_selection": seed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfsv2-features",
        default="artifacts_final/external_weather/noaa_cfsv2_h1_2024/features.csv",
    )
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/noaa_cfsv2_h1_2024/selection_screen.json",
    )
    parser.add_argument("--seeds", default="29401,29402,29403")
    parser.add_argument("--minimum-incremental-score", type=float, default=0.0001)
    args = parser.parse_args()

    external = build_disagreement_features(
        Path(args.cfsv2_features), Path(args.gfs)
    )
    driver = np.load(args.driver_cache)
    meta = np.load(args.meta_cache)
    index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not index.equals(driver_index):
        raise ValueError("driver and meta OOF indexes differ")
    common = index.intersection(external.index)
    common = common[(common >= pd.Timestamp("2024-01-01")) & (common < H2_START)]
    positions = index.get_indexer(common)
    truth = driver[f"{TARGET}__valid_truth"].astype(float)[positions]
    base = meta["valid_candidate"].astype(float)[positions]
    external = external.reindex(common)
    train = np.asarray(common < Q1_END)
    selection = np.asarray(common >= Q1_END)
    if train.sum() < 2_000 or selection.sum() < 2_000:
        raise ValueError("CFSv2 H1 screen requires complete Q1 and Q2 coverage")
    calendar = _calendar_features(common, base)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    control = _selection_family(
        "calendar_base_control", calendar, truth, base, train, selection, seeds
    )
    cfs = _selection_family(
        "calendar_base_plus_cfsv2_gfs_disagreement",
        calendar.join(external),
        truth,
        base,
        train,
        selection,
        seeds,
    )
    incremental = None
    if cfs["selection"] is not None and control["selection"] is not None:
        incremental = {
            key: float(
                cfs["selection"]["delta"][key]
                - control["selection"]["delta"][key]
            )
            for key in ("score", "one_minus_nmae", "ficr")
        }
    all_seed_components_positive = bool(cfs["seed_selection"]) and all(
        min(item["delta"].values()) > 0.0 for item in cfs["seed_selection"]
    )
    expand_h2 = bool(
        cfs["selection"] is not None
        and incremental is not None
        and incremental["score"] >= args.minimum_incremental_score
        and incremental["ficr"] >= 0.0
        and all_seed_components_positive
    )
    eligible = truth >= 0.10 * CAPACITY
    abs_error = np.abs(truth - base) / CAPACITY
    correlations = {
        column: {
            "q1_abs_error_spearman": float(
                pd.Series(external.loc[train, column].to_numpy()[eligible[train]]).corr(
                    pd.Series(abs_error[train][eligible[train]]), method="spearman"
                )
            ),
            "q2_abs_error_spearman": float(
                pd.Series(external.loc[selection, column].to_numpy()[eligible[selection]]).corr(
                    pd.Series(abs_error[selection][eligible[selection]]), method="spearman"
                )
            ),
        }
        for column in external.columns
    }
    report = {
        "family": "cfsv2_gfs_disagreement_h1_selection_screen",
        "split": {
            "train": "2024 Q1",
            "selection": "2024 Q2 through 2024-07-01 00:00",
            "locked_h2": "unopened",
            "common_rows": int(len(common)),
        },
        "control": control,
        "with_cfsv2": cfs,
        "incremental_q2_vs_control": incremental,
        "correlations": correlations,
        "decision": {
            "expand_h2_collection": expand_h2,
            "minimum_incremental_score": args.minimum_incremental_score,
            "all_seed_components_positive": all_seed_components_positive,
            "reason": (
                "CFSv2 adds stable Q2 score/FICR beyond the identical control"
                if expand_h2
                else "CFSv2 failed the preregistered incremental Q2 gate"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

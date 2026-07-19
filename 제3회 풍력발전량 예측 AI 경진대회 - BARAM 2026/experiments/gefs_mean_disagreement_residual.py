from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gefs_spread_residual_screen import (
    END,
    H2_START,
    Q1_END,
    TARGET,
    _calendar_features,
    _screen_family,
)


def build_disagreement_features(mean_path: Path, gfs_path: Path) -> pd.DataFrame:
    mean = pd.read_csv(mean_path, encoding="utf-8-sig")
    mean["forecast_kst_dtm"] = pd.to_datetime(mean["forecast_kst_dtm"])
    gfs = pd.read_csv(
        gfs_path,
        encoding="utf-8-sig",
        usecols=[
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "grid_id",
            "latitude",
            "longitude",
            "heightAboveGround_10_10u",
            "heightAboveGround_10_10v",
            "heightAboveGround_80_u",
            "heightAboveGround_80_v",
            "heightAboveGround_100_100u",
            "heightAboveGround_100_100v",
        ],
    )
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    keys = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
    ]
    merged = mean.merge(gfs, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(mean):
        raise ValueError("GEFS mean and supplied GFS rows are not exactly aligned")
    merged = merged.rename(
        columns={
            "heightAboveGround_10_10u": "gfs_u10",
            "heightAboveGround_10_10v": "gfs_v10",
            "heightAboveGround_80_u": "gfs_u80",
            "heightAboveGround_80_v": "gfs_v80",
            "heightAboveGround_100_100u": "gfs_u100",
            "heightAboveGround_100_100v": "gfs_v100",
        }
    )
    merged["delta_u10"] = merged["gefs_u10_mean"] - merged["gfs_u10"]
    merged["delta_v10"] = merged["gefs_v10_mean"] - merged["gfs_v10"]
    merged["vector_disagreement"] = np.hypot(
        merged["delta_u10"], merged["delta_v10"]
    )
    merged["gefs_speed10"] = np.hypot(
        merged["gefs_u10_mean"], merged["gefs_v10_mean"]
    )
    merged["gfs_speed10"] = np.hypot(merged["gfs_u10"], merged["gfs_v10"])
    merged["gfs_speed80"] = np.hypot(merged["gfs_u80"], merged["gfs_v80"])
    merged["gfs_speed100"] = np.hypot(merged["gfs_u100"], merged["gfs_v100"])
    merged["speed_disagreement"] = merged["gefs_speed10"] - merged["gfs_speed10"]
    denominator = np.maximum(merged["gefs_speed10"] * merged["gfs_speed10"], 1e-6)
    merged["direction_cosine"] = (
        merged["gefs_u10_mean"] * merged["gfs_u10"]
        + merged["gefs_v10_mean"] * merged["gfs_v10"]
    ) / denominator
    merged["direction_cosine"] = merged["direction_cosine"].clip(-1.0, 1.0)
    merged["gefs_vs_gfs100_speed_ratio"] = merged["gefs_speed10"] / np.maximum(
        merged["gfs_speed100"], 0.25
    )

    raw_features = [
        "gefs_u10_mean",
        "gefs_v10_mean",
        "gefs_speed10",
        "gfs_speed10",
        "gfs_speed80",
        "gfs_speed100",
        "delta_u10",
        "delta_v10",
        "vector_disagreement",
        "speed_disagreement",
        "direction_cosine",
        "gefs_vs_gfs100_speed_ratio",
    ]
    aggregate = merged.groupby("forecast_kst_dtm")[raw_features].agg(
        ["mean", "max", "min", "std"]
    )
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    center = (
        merged.loc[merged["grid_id"] == 5, ["forecast_kst_dtm"] + raw_features]
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
    issue["data_available_kst_dtm"] = pd.to_datetime(issue["data_available_kst_dtm"])
    result.attrs["issue_times"] = issue.reindex(result.index)[
        "data_available_kst_dtm"
    ].to_numpy()
    if result.isna().any().any():
        raise ValueError("hourly GEFS/GFS disagreement features contain missing values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mean-features",
        default="artifacts_final/external_weather/noaa_gefs_mean_f33_2024/features.csv",
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
        default="artifacts_final/external_weather/noaa_gefs_mean_f33_2024/hourly_residual.json",
    )
    parser.add_argument("--seeds", default="29201,29202,29203")
    args = parser.parse_args()

    external = build_disagreement_features(Path(args.mean_features), Path(args.gfs))
    driver = np.load(args.driver_cache)
    meta = np.load(args.meta_cache)
    index = pd.DatetimeIndex(pd.to_datetime(meta["valid_index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not index.equals(driver_index):
        raise ValueError("driver and meta OOF indexes differ")
    common = index.intersection(external.index)
    common = common[(common >= pd.Timestamp("2024-01-01")) & (common < END)]
    positions = index.get_indexer(common)
    truth = driver[f"{TARGET}__valid_truth"].astype(float)[positions]
    base = meta["valid_candidate"].astype(float)[positions]
    external = external.reindex(common)
    calendar = _calendar_features(common, base)
    with_external = calendar.join(external)
    train = np.asarray(common < Q1_END)
    selection = np.asarray((common >= Q1_END) & (common < H2_START))
    locked = np.asarray(common >= H2_START)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())

    control = _screen_family(
        "calendar_base_control",
        calendar,
        truth,
        base,
        train,
        selection,
        locked,
        seeds,
    )
    gefs = _screen_family(
        "calendar_base_plus_gefs_mean_gfs_disagreement",
        with_external,
        truth,
        base,
        train,
        selection,
        locked,
        seeds,
    )
    locked_available = gefs["locked_h2"] is not None
    incremental = None
    if locked_available and control["locked_h2"] is not None:
        incremental = {
            key: float(
                gefs["locked_h2"]["delta"][key]
                - control["locked_h2"]["delta"][key]
            )
            for key in ("score", "one_minus_nmae", "ficr")
        }
    decision = {
        "selection_status": gefs["selection_status"],
        "locked_h2_opened": locked_available,
        "locked_components_positive": bool(locked_available)
        and all(
            gefs["locked_h2"]["delta"][key] > 0.0
            for key in ("score", "one_minus_nmae", "ficr")
        ),
        "incremental_score_vs_control_positive": bool(incremental)
        and incremental["score"] > 0.0,
        "all_seed_components_positive": bool(gefs["seed_locked_h2"])
        and all(
            min(item["delta"].values()) > 0.0 for item in gefs["seed_locked_h2"]
        ),
        "all_locked_month_ficr_nonnegative": bool(
            gefs["monthly_locked_h2_delta"]
        )
        and all(
            value["ficr"] >= 0.0
            for value in gefs["monthly_locked_h2_delta"].values()
        ),
    }
    report = {
        "family": "gefs_mean_gfs_disagreement_hourly_residual",
        "split": {
            "train": "2024 Q1",
            "selection": "2024 Q2 through 2024-07-01 00:00",
            "locked_h2": "2024-07-01 01:00 through 2024-12-31 23:00",
            "common_rows": int(len(common)),
        },
        "control": control,
        "with_gefs_disagreement": gefs,
        "incremental_locked_h2_vs_control": incremental,
        "decision": decision,
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

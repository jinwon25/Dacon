from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_service.compliance import validate_external_data_manifest
from experiments.gefs_spread_residual_screen import (
    END,
    H2_START,
    Q1_END,
    TARGET,
    _calendar_features,
    _screen_family,
)


KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]


def build_kma_gfs_disagreement_features(
    kma_path: Path, gfs_path: Path
) -> pd.DataFrame:
    """Build hourly features that isolate information added by KMA UM.

    Supplied GFS values are used only as a reference for cross-model differences;
    the retained columns are KMA values or KMA-minus-GFS disagreement features.
    """
    kma = pd.read_csv(kma_path, encoding="utf-8-sig")
    required_kma = set(
        KEYS
        + [
            "point_id",
            "kma_um_u10",
            "kma_um_v10",
            "kma_um_speed10",
            "initialization_utc",
            "public_availability_utc",
        ]
    )
    missing = required_kma.difference(kma.columns)
    if missing:
        raise ValueError(f"KMA features are missing columns: {sorted(missing)}")
    if kma.duplicated(KEYS + ["point_id"]).any():
        raise ValueError("KMA features contain duplicate forecast/issue/point rows")
    for column in KEYS:
        kma[column] = pd.to_datetime(kma[column])

    gfs_columns = KEYS + [
        "grid_id",
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
        "heightAboveGround_100_100u",
        "heightAboveGround_100_100v",
    ]
    gfs = pd.read_csv(gfs_path, encoding="utf-8-sig", usecols=gfs_columns)
    if gfs.duplicated(KEYS + ["grid_id"]).any():
        raise ValueError("GFS features contain duplicate forecast/issue/grid rows")
    for column in KEYS:
        gfs[column] = pd.to_datetime(gfs[column])
    gfs["gfs_u10"] = gfs["heightAboveGround_10_10u"]
    gfs["gfs_v10"] = gfs["heightAboveGround_10_10v"]
    gfs["gfs_speed10"] = np.hypot(gfs["gfs_u10"], gfs["gfs_v10"])
    gfs["gfs_speed100"] = np.hypot(
        gfs["heightAboveGround_100_100u"],
        gfs["heightAboveGround_100_100v"],
    )

    kma_hourly = kma.groupby(KEYS)[
        ["kma_um_u10", "kma_um_v10", "kma_um_speed10"]
    ].agg(["mean", "min", "max", "std"])
    kma_hourly.columns = [
        f"{column}_{stat}" for column, stat in kma_hourly.columns
    ]
    # A one-point pilot has undefined spatial std; zero is the exact no-spread value.
    kma_hourly = kma_hourly.fillna(0.0)

    center = (
        gfs.loc[
            gfs["grid_id"] == 5,
            KEYS + ["gfs_u10", "gfs_v10", "gfs_speed10", "gfs_speed100"],
        ]
        .set_index(KEYS)
        .rename(columns=lambda value: f"{value}_center")
    )
    if center.index.duplicated().any():
        raise ValueError("GFS center grid is not unique by forecast and issue")
    merged = kma_hourly.join(center, how="inner", validate="one_to_one")
    if len(merged) != len(kma_hourly):
        raise ValueError("KMA and supplied GFS hours/issues are not exactly aligned")

    merged["kma_gfs_delta_u10"] = (
        merged["kma_um_u10_mean"] - merged["gfs_u10_center"]
    )
    merged["kma_gfs_delta_v10"] = (
        merged["kma_um_v10_mean"] - merged["gfs_v10_center"]
    )
    merged["kma_gfs_vector_disagreement"] = np.hypot(
        merged["kma_gfs_delta_u10"], merged["kma_gfs_delta_v10"]
    )
    merged["kma_gfs_speed_disagreement"] = (
        merged["kma_um_speed10_mean"] - merged["gfs_speed10_center"]
    )
    denominator = np.maximum(
        merged["kma_um_speed10_mean"] * merged["gfs_speed10_center"], 1e-6
    )
    merged["kma_gfs_direction_cosine"] = (
        merged["kma_um_u10_mean"] * merged["gfs_u10_center"]
        + merged["kma_um_v10_mean"] * merged["gfs_v10_center"]
    ) / denominator
    merged["kma_gfs_direction_cosine"] = merged[
        "kma_gfs_direction_cosine"
    ].clip(-1.0, 1.0)
    merged["kma_vs_gfs100_speed_ratio"] = merged[
        "kma_um_speed10_mean"
    ] / np.maximum(merged["gfs_speed100_center"], 0.25)

    retained = [
        column
        for column in merged.columns
        if column.startswith("kma_um_") or column.startswith("kma_gfs_")
    ] + ["kma_vs_gfs100_speed_ratio"]
    result = merged[retained].reset_index("data_available_kst_dtm")
    result.attrs["issue_times"] = result.pop(
        "data_available_kst_dtm"
    ).to_numpy()
    result.index.name = "forecast_kst_dtm"
    result = result.sort_index()
    if result.empty or result.isna().any().any():
        raise ValueError("KMA/GFS disagreement features are empty or incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/kma_um_global_2024/manifest.json",
    )
    parser.add_argument(
        "--kma-features",
        default="artifacts_final/external_weather/kma_um_global_2024/features.csv",
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
        default="artifacts_final/external_weather/kma_um_global_2024/disagreement_screen.json",
    )
    parser.add_argument("--seeds", default="29301,29302,29303")
    args = parser.parse_args()

    validate_external_data_manifest(Path(args.manifest), Path.cwd().resolve())
    external = build_kma_gfs_disagreement_features(
        Path(args.kma_features), Path(args.gfs)
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
    common = common[(common >= pd.Timestamp("2024-01-01")) & (common < END)]
    if len(common) < 8_000:
        raise ValueError("KMA screen requires near-complete 2024 hourly coverage")
    positions = index.get_indexer(common)
    truth = driver[f"{TARGET}__valid_truth"].astype(float)[positions]
    base = meta["valid_candidate"].astype(float)[positions]
    external = external.reindex(common)
    calendar = _calendar_features(common, base)
    with_kma = calendar.join(external)
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
    kma = _screen_family(
        "calendar_base_plus_kma_um_gfs_disagreement",
        with_kma,
        truth,
        base,
        train,
        selection,
        locked,
        seeds,
    )
    incremental = None
    if kma["locked_h2"] is not None and control["locked_h2"] is not None:
        incremental = {
            key: float(
                kma["locked_h2"]["delta"][key]
                - control["locked_h2"]["delta"][key]
            )
            for key in ("score", "one_minus_nmae", "ficr")
        }
    report = {
        "family": "kma_um_gfs_disagreement_group3_residual",
        "source_manifest": args.manifest,
        "split": {
            "train": "2024 Q1",
            "selection": "2024 Q2 through 2024-07-01 00:00",
            "locked_h2": "2024-07-01 01:00 through 2024-12-31 23:00",
            "common_rows": int(len(common)),
        },
        "control": control,
        "with_kma": kma,
        "incremental_locked_h2_vs_control": incremental,
        "decision": {
            "selection_status": kma["selection_status"],
            "locked_h2_opened": kma["locked_h2"] is not None,
            "all_locked_components_positive": bool(kma["locked_h2"])
            and min(kma["locked_h2"]["delta"].values()) > 0.0,
            "incremental_all_components_positive": bool(incremental)
            and min(incremental.values()) > 0.0,
            "all_seed_components_positive": bool(kma["seed_locked_h2"])
            and all(
                min(item["delta"].values()) > 0.0
                for item in kma["seed_locked_h2"]
            ),
            "all_locked_month_ficr_nonnegative": bool(
                kma["monthly_locked_h2_delta"]
            )
            and all(
                value["ficr"] >= 0.0
                for value in kma["monthly_locked_h2_delta"].values()
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

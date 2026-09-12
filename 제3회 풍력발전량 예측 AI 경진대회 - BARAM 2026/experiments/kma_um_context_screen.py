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


def build_context_screen_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(KEYS).difference(frame.columns)
    if missing:
        raise ValueError(f"KMA context features are missing keys: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError("KMA context features contain duplicate forecast/issue rows")
    for key in KEYS:
        frame[key] = pd.to_datetime(frame[key])
    columns = [
        column for column in frame.columns if column.startswith("kma_um_ctx_")
    ]
    if not columns:
        raise ValueError("KMA context feature file has no model features")
    result = frame.set_index(KEYS)[columns]
    numeric = result.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("KMA context features contain missing or non-finite values")
    # Constant columns cannot transfer across time and only add tree split noise.
    numeric = numeric.loc[:, numeric.nunique(dropna=False) > 1]
    if numeric.empty:
        raise ValueError("KMA context features are all constant")
    result = numeric.reset_index("data_available_kst_dtm")
    result.attrs["issue_times"] = result.pop("data_available_kst_dtm").to_numpy()
    result.index.name = "forecast_kst_dtm"
    return result.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/kma_um_context_2024/manifest.json",
    )
    parser.add_argument(
        "--features",
        default="artifacts_final/external_weather/kma_um_context_2024/features.csv",
    )
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--meta-cache", default="artifacts_final/meta_gate/meta_gate_cache.npz"
    )
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/kma_um_context_2024/context_screen.json",
    )
    parser.add_argument("--seeds", default="29501,29502,29503")
    parser.add_argument("--minimum-common-rows", type=int, default=8_000)
    parser.add_argument("--minimum-locked-score-gain", type=float, default=0.001)
    parser.add_argument("--minimum-incremental-score-gain", type=float, default=0.0005)
    parser.add_argument(
        "--preliminary-through",
        help="Mark an incomplete H2 screen as diagnostic-only through this timestamp.",
    )
    args = parser.parse_args()

    validate_external_data_manifest(Path(args.manifest), Path.cwd().resolve())
    external = build_context_screen_features(Path(args.features))
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
    if len(common) < args.minimum_common_rows:
        raise ValueError("KMA context screen does not meet minimum hourly coverage")
    positions = index.get_indexer(common)
    truth = driver[f"{TARGET}__valid_truth"].astype(float)[positions]
    base = meta["valid_candidate"].astype(float)[positions]
    external = external.reindex(common)
    if external.isna().any().any() or not np.isfinite(truth).all():
        raise ValueError("KMA context screen inputs are incomplete")
    calendar = _calendar_features(common, base)
    with_context = calendar.join(external)
    train = np.asarray(common < Q1_END)
    selection = np.asarray((common >= Q1_END) & (common < H2_START))
    locked = np.asarray(common >= H2_START)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    control = _screen_family(
        "calendar_base_control", calendar, truth, base, train, selection, locked, seeds
    )
    context = _screen_family(
        "calendar_base_plus_kma_um_run_profile_context",
        with_context,
        truth,
        base,
        train,
        selection,
        locked,
        seeds,
    )
    incremental = None
    if context["locked_h2"] is not None and control["locked_h2"] is not None:
        incremental = {
            key: float(
                context["locked_h2"]["delta"][key]
                - control["locked_h2"]["delta"][key]
            )
            for key in ("score", "one_minus_nmae", "ficr")
        }
    locked_positive = bool(context["locked_h2"]) and min(
        context["locked_h2"]["delta"].values()
    ) > 0.0
    incremental_positive = bool(incremental) and min(incremental.values()) > 0.0
    seeds_positive = bool(context["seed_locked_h2"]) and all(
        min(item["delta"].values()) > 0.0
        for item in context["seed_locked_h2"]
    )
    months_positive = bool(context["monthly_locked_h2_delta"]) and all(
        value["ficr"] >= 0.0
        for value in context["monthly_locked_h2_delta"].values()
    )
    score_large_enough = bool(context["locked_h2"]) and (
        context["locked_h2"]["delta"]["score"]
        >= args.minimum_locked_score_gain
    )
    incremental_large_enough = bool(incremental) and (
        incremental["score"] >= args.minimum_incremental_score_gain
    )
    submission_eligible = (
        not bool(args.preliminary_through)
        and context["selection_status"] == "passed"
        and locked_positive
        and incremental_positive
        and seeds_positive
        and months_positive
        and score_large_enough
        and incremental_large_enough
    )
    report = {
        "family": "kma_um_run_profile_context_group3_residual",
        "evaluation_status": (
            "preliminary_diagnostic" if args.preliminary_through else "locked"
        ),
        "source_manifest": args.manifest,
        "split": {
            "train": "2024 Q1",
            "selection": "2024 Q2 through 2024-07-01 00:00",
            "locked_h2": (
                f"preliminary 2024-07-01 01:00 through {args.preliminary_through}"
                if args.preliminary_through
                else "2024-07-01 01:00 through 2024-12-31 23:00"
            ),
            "common_rows": int(len(common)),
        },
        "promotion_thresholds": {
            "minimum_locked_score_gain": args.minimum_locked_score_gain,
            "minimum_incremental_score_gain": args.minimum_incremental_score_gain,
        },
        "control": control,
        "with_context": context,
        "incremental_locked_h2_vs_control": incremental,
        "decision": {
            "submission_eligible": submission_eligible,
            "selection_status": context["selection_status"],
            "locked_h2_opened": context["locked_h2"] is not None,
            "all_locked_components_positive": locked_positive,
            "incremental_all_components_positive": incremental_positive,
            "all_seed_components_positive": seeds_positive,
            "all_locked_month_ficr_nonnegative": months_positive,
            "minimum_locked_score_gain_passed": score_large_enough,
            "minimum_incremental_score_gain_passed": incremental_large_enough,
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

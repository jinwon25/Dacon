"""Audit whether ASOF counters reset by season and what frozen v242 anchors store."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
LOOKUP = ROOT / "JY_fallback_XGB_active50_w030" / "fallback_lookups.joblib"


def summarize_entity(frame: pd.DataFrame, entity: str, ncol: str) -> pd.DataFrame:
    values = frame[[entity, "season", ncol]].copy()
    grouped = values.groupby([entity, "season"], sort=False)[ncol]
    summary = grouped.agg(["min", "max", "first", "last", "size"]).reset_index()
    summary["first_minus_min"] = summary["first"] - summary["min"]
    summary["last_minus_max"] = summary["last"] - summary["max"]
    summary["starts_zero"] = summary["min"].eq(0)
    summary["monotone_end"] = summary["last"].eq(summary["max"])
    return summary


def main() -> None:
    train = pd.read_csv(TRAIN, encoding="utf-8-sig", low_memory=False)
    lookup = joblib.load(LOOKUP)
    print(f"rows={len(train):,} seasons={sorted(train.season.unique())}")
    for entity, ncol, prefix in (
        ("pitcher_id", "asof_pitcher_n", "p_succ"),
        ("batter_id", "asof_batter_n", "b_succ"),
    ):
        summary = summarize_entity(train, entity, ncol)
        print(f"\n[{entity}]")
        print(
            summary.groupby("season", sort=True).agg(
                groups=(entity, "size"),
                min_median=("min", "median"),
                min_p95=("min", lambda x: float(np.quantile(x, 0.95))),
                max_median=("max", "median"),
                first_eq_min=("first_minus_min", lambda x: float((x == 0).mean())),
                last_eq_max=("last_minus_max", lambda x: float((x == 0).mean())),
                starts_zero=("starts_zero", "mean"),
            ).to_string()
        )
        latest = summary.loc[summary.season.eq(int(train.season.max()))].set_index(entity)
        anchors = lookup["anchors"][prefix]
        common = latest.index.intersection(pd.Index(anchors.keys()))
        stored = pd.Series({key: anchors[int(key)][0] for key in common})
        compare = pd.DataFrame(
            {
                "stored": stored,
                "season_min": latest.loc[common, "min"],
                "season_max": latest.loc[common, "max"],
            }
        )
        print(
            "lookup latest-season common={} stored==min={:.6f} stored==max={:.6f} "
            "stored_zero={:.6f}".format(
                len(compare),
                float(np.isclose(compare.stored, compare.season_min).mean()),
                float(np.isclose(compare.stored, compare.season_max).mean()),
                float(np.isclose(compare.stored, 0.0).mean()),
            )
        )
        print(compare.describe(percentiles=[0.05, 0.5, 0.95]).to_string())


if __name__ == "__main__":
    main()

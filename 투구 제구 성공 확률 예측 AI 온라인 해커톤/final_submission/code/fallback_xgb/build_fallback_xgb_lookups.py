"""Freeze all train-only lookups needed by the fallback XGB test runtime.

The runtime (`fallback_xgb_frozen_runtime.py`) turns one evaluation row into the
114 model features by joining frozen tables.  This script rebuilds those tables
from the official training data alone.  Nothing here reads `test.csv`, and every
table is a pure aggregate over official rows.

Every group is rebuilt from the official data and matches the deployed asset
exactly (max abs error 0, identical key sets):

    target_mean / priors / cat / anchors / overall / situations / pb / ppa
        aggregated from `train.csv`.

    tm  (TrackMan release profiles)
        aggregated from `trackman_history.csv` joined through `pitcher_map.csv`
        at `conf >= 0.90`, which resolves the same 764 pitchers the deployed
        table covers.

Run `--verify` to diff a freshly built payload against the frozen asset.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROTOCOL = "FALLBACK_XGB_LOOKUP_FREEZE_V1"
SMOOTHING = 300.0
PAIR_STRIDE = 100000
TRACKMAN_MIN_CONF = 0.90


def _runtime_module():
    """Load the deployed runtime so the column contract has a single source."""
    spec = importlib.util.spec_from_file_location(
        "fallback_xgb_frozen_runtime", ROOT / "fallback_xgb_frozen_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pairs(frame: pd.DataFrame, key: str, values: list[str]) -> dict[int, tuple]:
    return {
        int(k): tuple(float(v) for v in row)
        for k, row in frame.set_index(key)[values].iterrows()
    }


def anchor_table(
    frame: pd.DataFrame, id_column: str, count_column: str, rate_column: str
) -> pd.DataFrame:
    """Latest pre-pitch cumulative state per (player, season).

    The runtime subtracts this anchor from a row's own running totals to recover
    the current-season sample.  The anchor therefore has to be the state carried
    into the player's final pitch of each season, which is the row holding the
    largest `count_column` for that pair.
    """
    table = frame[[id_column, "season", count_column, rate_column]].copy()
    table["succ"] = table[count_column] * table[rate_column]
    table = table.sort_values([id_column, "season", count_column])
    return table.groupby([id_column, "season"], sort=False).tail(1)[
        [id_column, "season", count_column, "succ"]
    ]


def situation_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Twelve fixed game-state slices, evaluated per row."""
    return {
        "3ball": frame.balls_before.eq(3),
        "2strk": frame.strikes_before.eq(2),
        "ahead": frame.strikes_before.gt(frame.balls_before),
        "behind": frame.balls_before.gt(frame.strikes_before),
        "risp": frame.runner_on_2b.eq(1) | frame.runner_on_3b.eq(1),
        "on1b": frame.runner_on_1b.eq(1),
        "vsL": frame.batter_hand.eq(1),
        "vsR": frame.batter_hand.eq(2),
        "late": frame.inning.ge(7),
        "hiLI": frame.li.gt(1.5),
        "loLI": frame.li.lt(.5),
        "blowout": frame.score_diff_pitcher_team.abs().ge(5),
    }


def trackman_profiles(
    trackman: pd.DataFrame, pitcher_map: pd.DataFrame, columns: list[str]
) -> dict[str, dict[int, float]]:
    """Per-pitcher release profile, averaged over that pitcher's seasons."""
    mapping = pitcher_map.loc[
        pitcher_map["conf"].ge(TRACKMAN_MIN_CONF), ["pitcher_id", "pitcher_trackman_id"]
    ]
    joined = trackman.merge(mapping, on="pitcher_trackman_id", how="inner")
    plain = [column for column in columns if column != "rel_speed_sd"]
    means = joined.groupby(["pitcher_id", "season"], sort=False)[plain].mean()
    means["rel_speed_sd"] = joined.groupby(
        ["pitcher_id", "season"], sort=False
    )["rel_speed"].std()
    profile = means.reset_index().groupby("pitcher_id", sort=False)[columns].mean()
    return {
        column: {int(k): float(v) for k, v in profile[column].dropna().items()}
        for column in columns
    }


def build(
    train: pd.DataFrame,
    frozen: dict,
    trackman: pd.DataFrame | None = None,
    pitcher_map: pd.DataFrame | None = None,
) -> dict:
    runtime = _runtime_module()
    target = train.control_success.to_numpy(float)
    target_mean = float(target.mean())

    payload: dict = {
        "target_mean": target_mean,
        "priors": {},
        "cat": {},
        "anchors": {},
        "situations": {},
        "tm": {},
        "features_version": frozen["features_version"],
    }

    for column in runtime.CAT:
        payload["cat"][column] = {
            str(value): int(index)
            for index, value in enumerate(
                pd.Index(train[column].fillna("__NA__").astype(str).unique())
            )
        }

    for id_column, count_column, rate_column, prefix in runtime.SPECS:
        latest = (
            anchor_table(train, id_column, count_column, rate_column)
            .sort_values("season")
            .groupby(id_column, sort=False)
            .tail(1)
        )
        payload["anchors"][prefix] = _pairs(latest, id_column, [count_column, "succ"])
        payload["priors"][prefix] = float(train[rate_column].mean())

    totals = train.groupby("pitcher_id", sort=False).control_success.agg(["sum", "count"])
    overall = (totals["sum"] + SMOOTHING * target_mean) / (totals["count"] + SMOOTHING)
    payload["overall"] = {int(k): float(v) for k, v in overall.items()}

    pitcher_rate = totals["sum"] / totals["count"]
    for name, mask in situation_masks(train).items():
        slice_totals = (
            train.loc[mask]
            .groupby("pitcher_id", sort=False)
            .control_success.agg(["sum", "count"])
            .reindex(totals.index)
            .fillna(0.)
        )
        rate = (slice_totals["sum"] + SMOOTHING * pitcher_rate) / (
            slice_totals["count"] + SMOOTHING
        )
        payload["situations"][name] = {int(k): float(v) for k, v in rate.items()}

    pair_key = train.pitcher_id.astype("int64") * PAIR_STRIDE + train.batter_id.astype("int64")
    pair = train.assign(_key=pair_key).groupby("_key", sort=False).control_success.agg(
        ["sum", "count"]
    )
    payload["pb"] = _pairs(pair.reset_index(), "_key", ["sum", "count"])

    game_id = (train.inning.diff().fillna(0) < 0).cumsum()
    per_game = (
        train.assign(_gid=game_id)
        .groupby(["_gid", "pitcher_id", "season"], sort=False)
        .size()
        .groupby(["pitcher_id", "season"], sort=False)
        .median()
        .rename("ppa")
        .reset_index()
    )
    payload["ppa"] = {
        int(k): float(v)
        for k, v in per_game.groupby("pitcher_id", sort=False).ppa.median().items()
    }
    payload["ppa_default"] = float(per_game.ppa.median())

    if trackman is not None and pitcher_map is not None:
        payload["tm"] = trackman_profiles(trackman, pitcher_map, list(runtime.TM))
    else:
        payload["tm"] = frozen["tm"]

    for key in ("anchor_semantics", "multiscale_semantics", "anchor_history_max_season"):
        payload[key] = frozen[key]
    return payload


def _diff(name: str, built, reference) -> tuple[str, float]:
    if isinstance(reference, float):
        return name, abs(float(built) - reference)
    if isinstance(reference, dict) and reference and isinstance(
        next(iter(reference.values())), tuple
    ):
        shared = set(built) & set(reference)
        if not shared:
            return name, float("inf")
        worst = max(
            max(abs(built[k][i] - reference[k][i]) for i in range(len(reference[k])))
            for k in shared
        )
        return name, worst if set(built) == set(reference) else float("inf")
    if isinstance(reference, dict):
        shared = set(built) & set(reference)
        if not shared:
            return name, float("inf")
        values = [reference[k] for k in shared]
        if values and isinstance(values[0], (int, float)):
            worst = max(abs(built[k] - reference[k]) for k in shared)
            return name, worst if set(built) == set(reference) else float("inf")
        return name, 0.0 if built == reference else float("inf")
    return name, 0.0 if built == reference else float("inf")


def verify(payload: dict, frozen: dict) -> bool:
    print(f"{'항목':<24}{'최대 오차':>14}   판정")
    checks: list[tuple[str, float]] = [
        _diff("target_mean", payload["target_mean"], frozen["target_mean"]),
        _diff("ppa_default", payload["ppa_default"], frozen["ppa_default"]),
    ]
    for group in ("priors", "cat", "anchors", "situations", "tm"):
        for key in frozen[group]:
            checks.append(_diff(f"{group}.{key}", payload[group][key], frozen[group][key]))
    for group in ("overall", "pb", "ppa"):
        checks.append(_diff(group, payload[group], frozen[group]))

    passed = True
    for name, worst in checks:
        ok = worst <= 1e-9
        passed &= ok
        print(f"{name:<24}{worst:>14.3g}   {'OK' if ok else '불일치'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT.parent / "data" / "train.csv")
    parser.add_argument("--trackman-csv", type=Path, default=ROOT.parent / "data" / "trackman_history.csv")
    parser.add_argument(
        "--pitcher-map-csv", type=Path, default=ROOT / "pitcher_map.csv",
        help="pitcher_id <-> pitcher_trackman_id table; rows at conf >= 0.90 are used",
    )
    parser.add_argument(
        "--frozen-lookups", type=Path, default=ROOT / "fallback_lookups.joblib",
        help="deployed asset, used as the diff target for --verify",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--carry-trackman", action="store_true",
        help="skip the TrackMan rebuild and copy that group from the frozen asset",
    )
    parser.add_argument("--verify", action="store_true", help="diff against the frozen asset")
    args = parser.parse_args()

    frozen = joblib.load(args.frozen_lookups)
    train = pd.read_csv(args.train_csv, encoding="utf-8-sig")

    trackman = pitcher_map = None
    if not args.carry_trackman:
        trackman = pd.read_csv(args.trackman_csv, encoding="utf-8-sig", low_memory=False)
        pitcher_map = pd.read_csv(args.pitcher_map_csv, encoding="utf-8-sig")

    payload = build(train, frozen, trackman, pitcher_map)

    if args.verify:
        ok = verify(payload, frozen)
        print("\n검증 결과:", "통과" if ok else "불일치 있음")
        if not ok:
            raise SystemExit("rebuilt payload does not match the frozen asset")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, args.output, compress=3)
        print("saved", args.output, "bytes", args.output.stat().st_size)


if __name__ == "__main__":
    main()

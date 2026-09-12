"""Pre-registered strict-forward conditional direct-model screen.

The model never receives raw player IDs.  Player identity is used only to
look up three empirical-Bayes summaries made from strictly earlier seasons:
pitcher command, pitcher x batter hand, and pitcher x count.  Audit/test rows
never contribute to a lookup table and are predicted independently.

The blend route and weight are selected on common-parent 2020--2021 axes.
That recipe is then frozen for all later common and exact-v84 diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from src.core.contract import _diagnostics, _load_contract_axis


PROTOCOL = "V97_CONDITIONAL_DIRECT_FORWARD_V1"
ANCHOR_TEAM = 13
EPS = 1e-6

CATEGORICAL = (
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "platoon",
    "team_matchup",
)

EXCLUDED = {
    "row_id",
    "control_success",
    "season",
    "game_month",
    "pitcher_id",
    "batter_id",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _domain(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype("string").eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(ANCHOR_TEAM)
        | frame["batter_team_id"].eq(ANCHOR_TEAM)
    ).to_numpy()
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def _count_state(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["balls_before"].astype("Int64").astype("string")
        + "-"
        + frame["strikes_before"].astype("Int64").astype("string")
    )


def _aggregate(history: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        history.groupby(keys, observed=True, sort=False)["control_success"]
        .agg(success="sum", n="size")
        .reset_index()
    )


def _bank(
    history: pd.DataFrame, strengths: dict[str, float]
) -> dict[str, Any]:
    if history.empty:
        return {"global": 0.5, "pitcher": None, "hand": None, "count": None}
    work = history[["pitcher_id", "batter_hand", "control_success"]].copy()
    work["count_state"] = _count_state(history).to_numpy()
    global_rate = float(work["control_success"].mean())
    pitcher = _aggregate(work, ["pitcher_id"])
    alpha_p = float(strengths["pitcher"])
    pitcher["rate"] = (
        pitcher["success"] + alpha_p * global_rate
    ) / (pitcher["n"] + alpha_p)

    def child(keys: list[str], strength: float) -> pd.DataFrame:
        table = _aggregate(work, keys)
        table = table.merge(
            pitcher[["pitcher_id", "rate"]].rename(columns={"rate": "parent"}),
            on="pitcher_id",
            how="left",
            validate="many_to_one",
        )
        table["parent"] = table["parent"].fillna(global_rate)
        table["rate"] = (
            table["success"] + float(strength) * table["parent"]
        ) / (table["n"] + float(strength))
        table["dev"] = table["rate"] - table["parent"]
        table["reliability"] = table["n"] / (table["n"] + float(strength))
        return table

    hand = child(
        ["pitcher_id", "batter_hand"], strengths["pitcher_batter_hand"]
    )
    count = child(
        ["pitcher_id", "count_state"], strengths["pitcher_count"]
    )
    return {"global": global_rate, "pitcher": pitcher, "hand": hand, "count": count}


def _lookup(
    rows: pd.DataFrame,
    table: pd.DataFrame | None,
    keys: list[str],
    values: list[str],
) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame({name: np.zeros(len(rows), dtype=np.float64) for name in values})
    left = rows[keys].reset_index(drop=True)
    merged = left.merge(
        table[keys + values], on=keys, how="left", sort=False, validate="many_to_one"
    )
    return merged[values]


def _feature_frame(rows: pd.DataFrame, bank: dict[str, Any]) -> pd.DataFrame:
    out = rows.drop(columns=[column for column in EXCLUDED if column in rows], errors="ignore").copy()
    out["count_state"] = _count_state(rows).astype("string")
    out["platoon"] = (
        rows["pitcher_hand"].astype("string") + "-" + rows["batter_hand"].astype("string")
    )
    out["team_matchup"] = (
        rows["pitcher_team_id"].astype("string")
        + "-"
        + rows["batter_team_id"].astype("string")
    )
    out["same_hand"] = rows["pitcher_hand"].eq(rows["batter_hand"]).astype(np.int8)

    global_rate = float(bank["global"])
    p_n = pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0.0).to_numpy()
    p_rate = pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce").to_numpy()
    b_n = pd.to_numeric(rows["asof_batter_n"], errors="coerce").fillna(0.0).to_numpy()
    b_rate = pd.to_numeric(rows["asof_batter_success_rate"], errors="coerce").to_numpy()
    p_rate = np.where(np.isfinite(p_rate), p_rate, global_rate)
    b_rate = np.where(np.isfinite(b_rate), b_rate, global_rate)
    out["shrunk_pitcher_rate"] = (p_n * p_rate + 300.0 * global_rate) / (p_n + 300.0)
    out["shrunk_batter_rate"] = (b_n * b_rate + 300.0 * global_rate) / (b_n + 300.0)
    for window in (1, 3, 5):
        recent = pd.to_numeric(
            rows[f"asof_pitcher_prev{window}_game_success_rate"], errors="coerce"
        ).to_numpy()
        out[f"form_diff_{window}"] = np.where(np.isfinite(recent), recent - p_rate, 0.0)
    out["ball_minus_strike_rate"] = (
        pd.to_numeric(rows["asof_pitcher_ball_rate"], errors="coerce")
        - pd.to_numeric(rows["asof_pitcher_strike_rate"], errors="coerce")
    )
    mix = rows[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64)
    out["pitchmix_entropy"] = -np.sum(
        np.where(mix > 0.0, mix * np.log(np.clip(mix, EPS, 1.0)), 0.0), axis=1
    )

    row_keys = rows[["pitcher_id", "batter_hand"]].copy()
    row_keys["count_state"] = _count_state(rows).to_numpy()
    pitcher = _lookup(row_keys, bank["pitcher"], ["pitcher_id"], ["rate", "n"])
    hand = _lookup(
        row_keys, bank["hand"], ["pitcher_id", "batter_hand"],
        ["rate", "dev", "n", "reliability"],
    )
    count = _lookup(
        row_keys, bank["count"], ["pitcher_id", "count_state"],
        ["rate", "dev", "n", "reliability"],
    )
    out["prior_pitcher_rate"] = pitcher["rate"].fillna(global_rate).to_numpy()
    out["prior_pitcher_n"] = pitcher["n"].fillna(0.0).to_numpy()
    for prefix, block in (("prior_hand", hand), ("prior_count", count)):
        out[f"{prefix}_rate"] = block["rate"].fillna(global_rate).to_numpy()
        out[f"{prefix}_dev"] = block["dev"].fillna(0.0).to_numpy()
        out[f"{prefix}_n"] = block["n"].fillna(0.0).to_numpy()
        out[f"{prefix}_reliability"] = block["reliability"].fillna(0.0).to_numpy()

    for column in CATEGORICAL:
        out[column] = out[column].astype("string").fillna("__MISSING__")
    if "game_month" in out or "pitcher_id" in out or "batter_id" in out:
        raise AssertionError("prohibited feature survived construction")
    return out


def _align_categories(
    fit: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = fit.copy()
    audit = audit.copy()
    for column in CATEGORICAL:
        values = pd.Index(fit[column].astype(str).unique())
        fit[column] = pd.Categorical(fit[column].astype(str), categories=values)
        audit[column] = pd.Categorical(audit[column].astype(str), categories=values)
    ordered = sorted(fit.columns)
    return fit[ordered], audit[ordered]


def _fit_predict(
    raw: pd.DataFrame,
    feature_by_season: dict[int, pd.DataFrame],
    audit_year: int,
    config: dict[str, Any],
) -> np.ndarray:
    fit_years = [year for year in sorted(feature_by_season) if year < audit_year]
    fit = pd.concat([feature_by_season[year] for year in fit_years], ignore_index=True)
    audit = feature_by_season[audit_year]
    fit, audit = _align_categories(fit, audit)
    fit_mask = raw["season"].lt(audit_year).to_numpy()
    target = raw.loc[fit_mask, "control_success"].to_numpy(np.int8)
    seasons = raw.loc[fit_mask, "season"].to_numpy(np.int16)
    weights = np.exp2(
        -((audit_year - 1) - seasons) / float(config["recency_half_life"])
    )
    model = lgb.LGBMClassifier(**config["model"])
    model.fit(fit, target, sample_weight=weights, categorical_feature=list(CATEGORICAL))
    prediction = model.predict_proba(audit)[:, 1].astype(np.float64)
    return np.clip(prediction, 0.001, 0.999)


def _eta(
    axes: list[dict[str, np.ndarray]], route: tuple[str, ...], cap: float
) -> float:
    numerator = 0.0
    denominator = 0.0
    for axis in axes:
        active = np.isin(axis["domain3"].astype(str), route)
        direction = axis["direct"][active] - axis["parent"][active]
        numerator += float(np.dot(axis["target"][active] - axis["parent"][active], direction))
        denominator += float(np.dot(direction, direction))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, cap))


def run(train_csv: Path, contract_dir: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    years = [int(year) for year in config["audit_years"]]
    feature_years = list(range(int(raw["season"].min()), max(years) + 1))
    feature_by_season: dict[int, pd.DataFrame] = {}
    for year in feature_years:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        history = raw.loc[raw["season"].lt(year)]
        bank = _bank(history, config["conditional_strength"])
        feature_by_season[year] = _feature_frame(rows, bank)
        print(f"[v97 features] year={year} rows={len(rows)} columns={len(feature_by_season[year].columns)}", flush=True)

    predictions: dict[int, np.ndarray] = {}
    common_axes: dict[int, dict[str, np.ndarray]] = {}
    for year in years:
        print(f"[v97 model] audit_year={year}", flush=True)
        direct = _fit_predict(raw, feature_by_season, year, config)
        predictions[year] = direct
        axis = _load_contract_axis(contract_dir / f"common_full_{year}.npz")
        expected_index = raw.index[raw["season"].eq(year)].to_numpy(np.int64)
        if not np.array_equal(axis["raw_index"].astype(np.int64), expected_index):
            raise ValueError(f"contract row order mismatch for {year}")
        axis["direct"] = direct
        common_axes[year] = axis
        np.savez_compressed(
            output_dir / f"direct_full_{year}.npz",
            raw_index=axis["raw_index"], target=axis["target"], direct=direct,
            domain3=axis["domain3"], game_month=axis["game_month"],
            pitcher_id=axis["pitcher_id"], batter_id=axis["batter_id"],
        )

    selection_axes = [common_axes[int(year)] for year in config["selection_years"]]
    route_rows = []
    for name, domains in config["routes"].items():
        route = tuple(str(value) for value in domains)
        eta = _eta(selection_axes, route, float(config["eta_cap"]))
        diagnostics = [_diagnostics(axis, route, eta) for axis in selection_axes]
        route_rows.append(
            {
                "route": name,
                "eta": eta,
                "selection_min_gain": min(item["gain"] for item in diagnostics),
                "selection_mean_gain": float(np.mean([item["gain"] for item in diagnostics])),
                "selection_min_month_fraction": min(item["positive_month_fraction"] for item in diagnostics),
                "selection_worst_month_gain": min(item["worst_month_gain"] for item in diagnostics),
            }
        )
    ledger = pd.DataFrame(route_rows).sort_values(
        ["selection_min_gain", "selection_mean_gain"], ascending=False
    ).reset_index(drop=True)
    ledger.to_csv(output_dir / "selection_ledger.csv", index=False)
    selected_name = str(ledger.iloc[0]["route"])
    selected_eta = float(ledger.iloc[0]["eta"])
    selected_route = tuple(str(value) for value in config["routes"][selected_name])

    common_results = {
        str(year): _diagnostics(common_axes[year], selected_route, selected_eta)
        for year in years
    }
    exact_results: dict[str, Any] = {}
    for year in (2022, 2024):
        exact = _load_contract_axis(contract_dir / f"v84_full_{year}.npz")
        direct = predictions[year]
        if len(direct) != len(exact["target"]):
            raise ValueError(f"exact axis length mismatch for {year}")
        exact_axis = {**exact, "direct": direct}
        mask = exact["exact_mask"].astype(bool)
        exact_masked = {
            name: np.asarray(value)[mask]
            for name, value in exact_axis.items()
            if np.asarray(value).shape == mask.shape
        }
        exact_results[str(year)] = _diagnostics(exact_masked, selected_route, selected_eta)

    result = {
        "protocol": PROTOCOL,
        "train_csv": str(train_csv.resolve()),
        "train_sha256": _sha256(train_csv),
        "config_sha256": _sha256(config_path),
        "feature_count": int(len(next(iter(feature_by_season.values())).columns)),
        "categorical_features": list(CATEGORICAL),
        "selected_on_common_years_only": list(config["selection_years"]),
        "selected_recipe": {"route": selected_name, "domains": list(selected_route), "eta": selected_eta},
        "route_trial_count": int(len(ledger)),
        "common_diagnostics": common_results,
        "exact_v84_diagnostics": exact_results,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
        "eligible_for_packaging": False,
        "packaging_reason": "screen only; bootstrap, Reality Check, final refit and standalone parity are pending",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()

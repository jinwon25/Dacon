"""Strict-forward player transition residuals above the v320 portfolio.

This experiment covers a feature family named in the early season-state audit
but not previously rebased above the modern champion: newcomer, returner,
same-team and team-switch state, optionally crossed with prior-season workload.

For prediction season Y every state uses only rows from seasons <Y plus fields
on the current row.  Corrections are learned from forward OOF residuals in
strictly earlier seasons and are centred within source season/game type.  The
recipe is selected on full-2022 and late-2023; full-2024 is opened once after
selection.  No evaluation-row order or aggregate is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V324_PLAYER_TRANSITION_RESIDUAL_V1"
TARGET = "control_success"
SOURCE_YEARS = (2020, 2021, 2022, 2023)
ALPHAS = (250.0, 1000.0, 4000.0)
WEIGHTS = (0.25, 0.50, 1.00)
ROUTES = ("ALL", "R", "F")
LOW_VOLUME_MAX = 100
MEDIUM_VOLUME_MAX = 800


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rate = float(target.mean())
    return float(
        100000.0
        * (1.0 - np.mean(np.square(target - prediction)) / (rate * (1.0 - rate)))
    )


def _dominant_team(history: pd.DataFrame, id_column: str, team_column: str) -> pd.Series:
    counts = history.groupby([id_column, team_column], observed=True, sort=False).size()
    if counts.empty:
        return pd.Series(dtype="object")
    return counts.reset_index(name="n").sort_values(
        [id_column, "n", team_column], ascending=[True, False, True]
    ).drop_duplicates(id_column).set_index(id_column)[team_column]


def entity_transition_state(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    year: int,
    id_column: str,
    team_column: str,
) -> pd.DataFrame:
    """Return row-local transition status and frozen prior-season workload."""

    history = train.loc[train["season"].lt(year), ["season", id_column, team_column]]
    previous = history.loc[history["season"].eq(year - 1)]
    last_year = history.groupby(id_column, observed=True)["season"].max()
    previous_volume = previous.groupby(id_column, observed=True).size()
    previous_team = _dominant_team(previous, id_column, team_column)

    identifiers = rows[id_column]
    seen_year = identifiers.map(last_year)
    volume = identifiers.map(previous_volume).fillna(0).astype(np.int64)
    old_team = identifiers.map(previous_team)
    current_team = rows[team_column]
    status = np.select(
        [
            seen_year.isna(),
            seen_year.lt(year - 1),
            old_team.notna() & old_team.eq(current_team),
        ],
        ["NEW", "RETURN", "SAME"],
        default="SWITCH",
    )
    volume_bucket = np.select(
        [volume.eq(0), volume.le(LOW_VOLUME_MAX), volume.le(MEDIUM_VOLUME_MAX)],
        ["NONE", "LOW", "MEDIUM"],
        default="HIGH",
    )
    return pd.DataFrame(
        {
            "status": pd.Series(status, index=rows.index, dtype="string"),
            "volume_bucket": pd.Series(volume_bucket, index=rows.index, dtype="string"),
            "previous_volume": volume.to_numpy(np.int64),
        },
        index=rows.index,
    )


def transition_features(train: pd.DataFrame, rows: pd.DataFrame, year: int) -> pd.DataFrame:
    pitcher = entity_transition_state(
        train, rows, year, "pitcher_id", "pitcher_team_id"
    )
    batter = entity_transition_state(
        train, rows, year, "batter_id", "batter_team_id"
    )
    count = (
        pd.to_numeric(rows["balls_before"], errors="coerce").fillna(-1).astype(int).astype(str)
        + "-"
        + pd.to_numeric(rows["strikes_before"], errors="coerce").fillna(-1).astype(int).astype(str)
    )
    output = pd.DataFrame(index=rows.index)
    output["pitcher_status"] = pitcher["status"]
    output["batter_status"] = batter["status"]
    output["crossed_status"] = pitcher["status"] + "|" + batter["status"]
    output["pitcher_exposure"] = pitcher["status"] + "|" + pitcher["volume_bucket"]
    output["batter_exposure"] = batter["status"] + "|" + batter["volume_bucket"]
    output["crossed_exposure"] = output["pitcher_exposure"] + "|" + output["batter_exposure"]
    output["pitcher_status_count"] = pitcher["status"] + "|" + count.astype("string")
    output["batter_status_count"] = batter["status"] + "|" + count.astype("string")
    return output


def load_oof(oof_dir: Path, year: int, expected: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    with np.load(oof_dir / f"wave0_incumbent_validate_{year}.npz", allow_pickle=False) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
    if not np.array_equal(target, expected[TARGET].to_numpy(np.float64)):
        raise ValueError(f"wave0 target/order mismatch for {year}")
    return target, parent


def build_signal_bank(
    train: pd.DataFrame,
    query_rows: pd.DataFrame,
    query_year: int,
    oof_dir: Path,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pool stable transition-class OOF contrasts from all prior source years."""

    query_features = transition_features(train, query_rows, query_year)
    source_frames: list[pd.DataFrame] = []
    used_years = [year for year in SOURCE_YEARS if year < query_year]
    for year in used_years:
        source_rows = train.loc[train["season"].eq(year)].reset_index(drop=True)
        target, parent = load_oof(oof_dir, year, source_rows)
        source_features = transition_features(train, source_rows, year)
        residual = target - parent
        centering = pd.DataFrame(
            {"season": year, "game_type": source_rows["game_type"].astype(str), "residual": residual}
        ).groupby(["season", "game_type"], observed=True)["residual"].transform("mean")
        source = source_features.copy()
        source["game_type"] = source_rows["game_type"].astype(str).to_numpy()
        source["residual"] = residual - centering.to_numpy(np.float64)
        source_frames.append(source)
    pooled = pd.concat(source_frames, ignore_index=True)

    bank: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {"source_years": used_years, "signals": {}}
    for feature_name in query_features.columns:
        keys = [feature_name, "game_type"]
        stats = pooled.groupby(keys, observed=True)["residual"].agg(n="size", s="sum")
        stats["correction"] = stats["s"] / (stats["n"] + float(alpha))
        query_index = pd.MultiIndex.from_arrays(
            [query_features[feature_name].astype(str), query_rows["game_type"].astype(str)],
            names=keys,
        )
        correction = stats["correction"].reindex(query_index).fillna(0.0).to_numpy(np.float64)
        bank[feature_name] = correction
        diagnostics["signals"][feature_name] = {
            "groups": int(len(stats)),
            "coverage": float(stats["correction"].reindex(query_index).notna().mean()),
            "rms": float(np.sqrt(np.mean(np.square(correction)))),
        }
    bank["additive_status"] = 0.75 * bank["pitcher_status"] + 0.25 * bank["batter_status"]
    bank["additive_exposure"] = 0.75 * bank["pitcher_exposure"] + 0.25 * bank["batter_exposure"]
    bank["additive_status_count"] = (
        0.75 * bank["pitcher_status_count"] + 0.25 * bank["batter_status_count"]
    )
    return bank, diagnostics


def _route_mask(frame: pd.DataFrame, route: str) -> np.ndarray:
    if route == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["game_type"].astype(str).eq(route).to_numpy()


def _candidate(
    parent: np.ndarray, signal: np.ndarray, active: np.ndarray, weight: float
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(weight) * np.asarray(signal, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output


def axis_metrics(
    frame: pd.DataFrame, parent: np.ndarray, candidate: np.ndarray, active: np.ndarray
) -> dict[str, Any]:
    axes = {
        "target": frame[TARGET].to_numpy(np.float64),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame), dtype=bool),
    }
    return paired_metrics(axes, parent, candidate, active)


def run(
    train_csv: Path,
    oof_dir: Path,
    v285_axes: Path,
    v318_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parents = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v318_axes, allow_pickle=False) as saved:
        parents.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )
        v320_direction = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )
    for name, frame in frames.items():
        if len(frame) != len(parents[name]):
            raise ValueError(f"parent alignment mismatch: {name}")

    banks: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    bank_diagnostics: dict[str, Any] = {}
    for name, frame in frames.items():
        year = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}[name]
        banks[name] = {}
        bank_diagnostics[name] = {}
        for alpha in ALPHAS:
            banks[name][alpha], bank_diagnostics[name][str(int(alpha))] = build_signal_bank(
                train, frame, year, oof_dir, alpha
            )

    grid_rows: list[dict[str, Any]] = []
    signal_names = sorted(banks["full_2022"][ALPHAS[0]])
    for alpha in ALPHAS:
        for signal_name in signal_names:
            for route in ROUTES:
                for weight in WEIGHTS:
                    record: dict[str, Any] = {
                        "alpha": alpha, "signal": signal_name, "route": route, "weight": weight
                    }
                    for name in ("full_2022", "late_2023"):
                        active = _route_mask(frames[name], route)
                        candidate = _candidate(
                            parents[name], banks[name][alpha][signal_name], active, weight
                        )
                        result = axis_metrics(frames[name], parents[name], candidate, active)
                        record[f"{name}_gain"] = result["gain"]
                        record[f"{name}_positive_month_fraction"] = result["positive_month_fraction"]
                        record[f"{name}_worst_month_gain"] = result["worst_month_gain"]
                    record["minimum_gain"] = min(
                        record["full_2022_gain"], record["late_2023_gain"]
                    )
                    record["source_pass"] = bool(
                        record["full_2022_gain"] > 0.0
                        and record["late_2023_gain"] > 0.0
                        and record["full_2022_positive_month_fraction"] >= 4.0 / 7.0
                        and record["late_2023_positive_month_fraction"] >= 0.5
                    )
                    grid_rows.append(record)
    grid = pd.DataFrame(grid_rows).sort_values(
        ["source_pass", "minimum_gain", "full_2022_gain", "late_2023_gain"],
        ascending=False,
    ).reset_index(drop=True)
    grid.to_csv(output_dir / "source_grid.csv", index=False)
    chosen = grid.iloc[0]
    recipe = {
        "alpha": float(chosen["alpha"]),
        "signal": str(chosen["signal"]),
        "route": str(chosen["route"]),
        "weight": float(chosen["weight"]),
    }

    metrics: dict[str, Any] = {}
    candidates: dict[str, np.ndarray] = {}
    actives: dict[str, np.ndarray] = {}
    directions: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        signal = banks[name][recipe["alpha"]][recipe["signal"]]
        active = _route_mask(frame, recipe["route"])
        candidate = _candidate(parents[name], signal, active, recipe["weight"])
        candidates[name], actives[name] = candidate, active
        directions[name] = candidate - parents[name]
        metrics[name] = axis_metrics(frame, parents[name], candidate, active)

    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    family24 = [parents["full_2024"]]
    for weight in WEIGHTS:
        family24.append(
            _candidate(
                parents["full_2024"],
                banks["full_2024"][recipe["alpha"]][recipe["signal"]],
                actives["full_2024"],
                weight,
            )
        )
    robustness = _robustness(
        axes24,
        parents["full_2024"],
        candidates["full_2024"],
        actives["full_2024"],
        family24,
    )
    selected_direction = directions["full_2024"]
    nonzero = (np.abs(selected_direction) > 0.0) | (np.abs(v320_direction) > 0.0)
    direction_correlation = (
        float(np.corrcoef(selected_direction[nonzero], v320_direction[nonzero])[0, 1])
        if nonzero.sum() > 2
        and np.std(selected_direction[nonzero]) > 0.0
        and np.std(v320_direction[nonzero]) > 0.0
        else 0.0
    )
    source_pass = bool(chosen["source_pass"])
    locked_pass = bool(
        metrics["full_2024"]["gain"] >= 3.0
        and metrics["full_2024"]["positive_month_fraction"] >= 0.625
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
        **{f"direction_{name}": directions[name] for name in frames},
        **{f"active_{name}": actives[name] for name in frames},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "parent": {"full_2022": "v285", "late_2023": "v320-equivalent", "full_2024": "v320"},
        "selected_recipe": recipe,
        "source_grid_size": int(len(grid)),
        "source_passing_recipes": int(grid["source_pass"].sum()),
        "metrics": metrics,
        "locked_robustness": robustness,
        "locked_direction_correlation_with_v320_increment": direction_correlation,
        "bank_diagnostics": bank_diagnostics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "restrictions": {
            "official_train_only": True,
            "strictly_prior_season_state": True,
            "source_residuals_forward_oof": True,
            "full_2024_used_for_recipe_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v318-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.oof_dir, args.v285_axes, args.v318_axes, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

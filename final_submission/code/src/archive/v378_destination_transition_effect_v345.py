"""Strict-forward destination-team transition effect above v345.

The deployed v345 transition lookup models pitcher transition status crossed
with count and game type, but deliberately pools over destination teams.  This
experiment tests a separate baseball mechanism: a pitcher joining a new club
may inherit a persistent coaching/catching environment.  For prediction year
Y, destination effects are estimated only from seasons before Y.  Within each
source season, game type, transition status and count, the target is centred
before destination-team pooling, so the signal cannot act as a global or
count-level calibration shift.

Recipe selection uses full-2022 and late-2023.  Full-2024 is opened once for
the selected source recipe.  Evaluation rows are mapped independently from
their current team and a frozen prior-season pitcher history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v324_player_transition_residual import entity_transition_state
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics


PROTOCOL = "V378_DESTINATION_TRANSITION_EFFECT_V345_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
SOURCE_START_YEAR = 2020
ANCHOR_TEAM = 13
WEIGHT = 0.25
ALPHAS = (250.0, 1000.0, 4000.0)
SCHEMAS = ("destination_status", "destination_status_count")
SCOPES = {
    "SWITCH": frozenset(("SWITCH",)),
    "MOBILE": frozenset(("SWITCH", "RETURN")),
    "NONSAME": frozenset(("SWITCH", "RETURN", "NEW")),
}


def _count_key(rows: pd.DataFrame) -> pd.Series:
    balls = pd.to_numeric(rows["balls_before"], errors="raise").astype(int).astype(str)
    strikes = pd.to_numeric(rows["strikes_before"], errors="raise").astype(int).astype(str)
    return (balls + "-" + strikes).astype("string")


def _transition_frame(train: pd.DataFrame, rows: pd.DataFrame, year: int) -> pd.DataFrame:
    state = entity_transition_state(
        train, rows, year, "pitcher_id", "pitcher_team_id"
    )
    return pd.DataFrame(
        {
            "status": state["status"].astype("string"),
            "count": _count_key(rows),
            "game_type": rows["game_type"].astype("string"),
            "destination": pd.to_numeric(
                rows["pitcher_team_id"], errors="raise"
            ).astype("int64"),
        },
        index=rows.index,
    )


def build_destination_signal(
    train: pd.DataFrame,
    query_rows: pd.DataFrame,
    query_year: int,
    schema: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema: {schema}")
    source_parts: list[pd.DataFrame] = []
    source_years = tuple(range(SOURCE_START_YEAR, query_year))
    for year in source_years:
        rows = train.loc[train["season"].eq(year)].reset_index(drop=True)
        features = _transition_frame(train, rows, year)
        work = features.copy()
        work["target"] = rows[TARGET].to_numpy(np.float64)
        centre_keys = ["game_type", "status", "count"]
        centre = work.groupby(centre_keys, observed=True)["target"].transform("mean")
        work["relative"] = work["target"] - centre
        source_parts.append(work)
    pooled = pd.concat(source_parts, ignore_index=True)

    key_columns = ["destination", "status", "game_type"]
    if schema == "destination_status_count":
        key_columns.insert(2, "count")
    stats = pooled.groupby(key_columns, observed=True)["relative"].agg(n="size", s="sum")
    stats["correction"] = stats["s"] / (stats["n"] + float(alpha))

    query = _transition_frame(train, query_rows, query_year)
    query_index = pd.MultiIndex.from_frame(query[key_columns])
    correction = stats["correction"].reindex(query_index)
    mapped = correction.notna().to_numpy()
    signal = correction.fillna(0.0).to_numpy(np.float64)
    return signal, query["status"].astype(str).to_numpy(), {
        "source_years": list(source_years),
        "groups": int(len(stats)),
        "mapped_fraction": float(mapped.mean()),
        "signal_rms": float(np.sqrt(np.mean(np.square(signal)))),
        "signal_max_abs": float(np.max(np.abs(signal))),
    }


def _parents(v338_axes: Path, v345_axes: Path) -> dict[str, np.ndarray]:
    with np.load(v338_axes, allow_pickle=False) as saved:
        output = {
            "full_2022": saved["candidate_full_2022"].astype(np.float64),
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        output["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
    return output


def _apply(
    frame: pd.DataFrame,
    parent: np.ndarray,
    signal: np.ndarray,
    status: np.ndarray,
    scope: frozenset[str],
) -> tuple[np.ndarray, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].astype("int64").eq(ANCHOR_TEAM).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(ANCHOR_TEAM).to_numpy()
    )
    active = regular & ~anchor & np.isin(status, tuple(scope)) & (signal != 0.0)
    candidate = np.asarray(parent, dtype=np.float64).copy()
    candidate[active] = np.clip(
        candidate[active] + WEIGHT * signal[active], 0.001, 0.999
    )
    return candidate, active


def run(
    train_csv: Path,
    v338_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before",
        TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    years = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    parents = _parents(v338_axes, v345_axes)
    for origin in ORIGINS:
        if len(frames[origin]) != len(parents[origin]):
            raise ValueError(f"parent row mismatch: {origin}")

    source_trials: list[dict[str, Any]] = []
    source_cache: dict[tuple[str, float, str], tuple[np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    for schema in SCHEMAS:
        for alpha in ALPHAS:
            metrics: dict[str, Any] = {}
            for origin in ("full_2022", "late_2023"):
                signal, status, diag = build_destination_signal(
                    train, frames[origin], years[origin], schema, alpha
                )
                source_cache[(schema, alpha, origin)] = (signal, status)
                diagnostics[f"{schema}|a{alpha:g}|{origin}"] = diag
            for scope_name, scope in SCOPES.items():
                for origin in ("full_2022", "late_2023"):
                    signal, status = source_cache[(schema, alpha, origin)]
                    candidate, active = _apply(
                        frames[origin], parents[origin], signal, status, scope
                    )
                    metrics[origin] = axis_metrics(
                        frames[origin], parents[origin], candidate, active
                    )
                gains = [metrics[origin]["gain"] for origin in ("full_2022", "late_2023")]
                eligible = bool(
                    min(gains) > 0.0
                    and all(metrics[o]["positive_month_fraction"] >= 0.5 for o in metrics)
                )
                source_trials.append(
                    {
                        "schema": schema,
                        "alpha": alpha,
                        "scope": scope_name,
                        "eligible": eligible,
                        "minimum_source_gain": float(min(gains)),
                        "mean_source_gain": float(np.mean(gains)),
                        "metrics": metrics,
                    }
                )
    eligible = [trial for trial in source_trials if trial["eligible"]]
    if not eligible:
        selected = max(
            source_trials,
            key=lambda trial: (trial["minimum_source_gain"], trial["mean_source_gain"]),
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_contract": {
                "origins": ["full_2022", "late_2023"],
                "schemas": list(SCHEMAS),
                "alphas": list(ALPHAS),
                "weight": WEIGHT,
                "scopes": {key: sorted(value) for key, value in SCOPES.items()},
            },
            "selected_diagnostic": selected,
            "source_trials": source_trials,
            "locked_origin_opened": False,
            "restrictions": {
                "official_train_only": True,
                "strictly_prior_season_destination_tables": True,
                "source_only_recipe_selection": True,
                "test_csv_read": False,
                "test_aggregate_used": False,
                "public_score_used_for_selection": False,
                "row_local_inference": True,
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    selected = max(
        eligible,
        key=lambda trial: (
            trial["minimum_source_gain"],
            trial["mean_source_gain"],
            -trial["alpha"],
            trial["schema"] == "destination_status",
        ),
    )
    schema = str(selected["schema"])
    alpha = float(selected["alpha"])
    scope_name = str(selected["scope"])
    signal24, status24, diag24 = build_destination_signal(
        train, frames["full_2024"], 2024, schema, alpha
    )
    candidate24, active24 = _apply(
        frames["full_2024"], parents["full_2024"], signal24, status24,
        SCOPES[scope_name],
    )
    locked = axis_metrics(
        frames["full_2024"], parents["full_2024"], candidate24, active24
    )
    locked["full_row_rms_shift"] = float(
        np.sqrt(np.mean(np.square(candidate24 - parents["full_2024"])))
    )
    frame24 = frames["full_2024"]
    axes24 = {
        "target": frame24[TARGET].to_numpy(np.float64),
        "game_month": frame24["game_month"].to_numpy(np.int16),
        "pitcher_id": frame24["pitcher_id"].to_numpy(),
        "batter_id": frame24["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frame24), dtype=bool),
    }
    robustness = _robustness(
        axes24,
        parents["full_2024"],
        candidate24,
        active24,
        [parents["full_2024"], candidate24],
    )
    gate = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and robustness["pitcher"]["prob_positive"] >= 0.90
        and robustness["chronological_block"]["prob_positive"] >= 0.90
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=candidate24,
        active_full_2024=active24,
        signal_full_2024=signal24,
        status_full_2024=status24,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if gate else "locked_reject",
        "source_contract": {
            "origins": ["full_2022", "late_2023"],
            "schemas": list(SCHEMAS),
            "alphas": list(ALPHAS),
            "weight": WEIGHT,
            "scopes": {key: sorted(value) for key, value in SCOPES.items()},
        },
        "selected": selected,
        "locked_full_2024": locked,
        "locked_robustness": robustness,
        "diagnostics": {**diagnostics, "selected_locked": diag24},
        "candidate_gate_passed": gate,
        "selection_warning": (
            "2022/late-2023 use the nearest available v345 analogue (v335 plus "
            "the frozen player-transition component); full-2024 uses exact v345."
        ),
        "restrictions": {
            "official_train_only": True,
            "strictly_prior_season_destination_tables": True,
            "source_only_recipe_selection": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
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
    parser.add_argument("--v338-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.train_csv, args.v338_axes, args.v345_axes, args.output_dir),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()

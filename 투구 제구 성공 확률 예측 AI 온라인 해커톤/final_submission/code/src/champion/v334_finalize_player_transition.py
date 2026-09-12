"""Finalize the 2025 player-transition residual lookup from official OOF rows.

The deployed signal is deliberately small and row local.  It identifies a
pitcher as NEW, RETURN, SAME or SWITCH using only official seasons before 2025,
crosses that state with the pre-pitch count, and maps a pooled forward-OOF
residual correction.  Source residuals are centred by season/game type before
pooling so the lookup cannot act as a hidden global calibration shift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v324_player_transition_residual import entity_transition_state


PROTOCOL = "V334_FINALIZE_PLAYER_TRANSITION_V1"
TARGET = "control_success"
SOURCE_YEARS = (2020, 2021, 2022, 2023, 2024)
QUERY_YEAR = 2025
ALPHA = 1000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _count_key(rows: pd.DataFrame) -> pd.Series:
    balls = pd.to_numeric(rows["balls_before"], errors="raise").astype(int).astype(str)
    strikes = pd.to_numeric(rows["strikes_before"], errors="raise").astype(int).astype(str)
    return (balls + "-" + strikes).astype("string")


def _load_oof(oof_dir: Path, year: int, expected: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    path = oof_dir / f"wave0_incumbent_validate_{year}.npz"
    with np.load(path, allow_pickle=False) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
    if len(target) != len(expected):
        raise ValueError(f"wave0 row count mismatch for {year}")
    if not np.array_equal(target, expected[TARGET].to_numpy(np.float64)):
        raise ValueError(f"wave0 target/order mismatch for {year}")
    return target, parent


def build_lookup(train: pd.DataFrame, oof_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_parts: list[pd.DataFrame] = []
    source_rows_by_year: dict[str, int] = {}
    for year in SOURCE_YEARS:
        rows = train.loc[train["season"].eq(year)].reset_index(drop=True)
        target, parent = _load_oof(oof_dir, year, rows)
        state = entity_transition_state(
            train, rows, year, "pitcher_id", "pitcher_team_id"
        )["status"].astype(str)
        residual = target - parent
        centering = pd.DataFrame(
            {"game_type": rows["game_type"].astype(str), "residual": residual}
        ).groupby("game_type", observed=True)["residual"].transform("mean")
        source_parts.append(
            pd.DataFrame(
                {
                    "key": state + "|" + _count_key(rows).astype(str)
                    + "|" + rows["game_type"].astype(str),
                    "residual": residual - centering.to_numpy(np.float64),
                }
            )
        )
        source_rows_by_year[str(year)] = int(len(rows))
    pooled = pd.concat(source_parts, ignore_index=True)
    stats = pooled.groupby("key", observed=True)["residual"].agg(n="size", s="sum")
    stats["correction"] = stats["s"] / (stats["n"] + ALPHA)
    stats = stats.sort_index()

    history = train.loc[train["season"].lt(QUERY_YEAR)]
    last_year = history.groupby("pitcher_id", observed=True)["season"].max().sort_index()
    previous = history.loc[history["season"].eq(QUERY_YEAR - 1)]
    counts = previous.groupby(
        ["pitcher_id", "pitcher_team_id"], observed=True, sort=False
    ).size().reset_index(name="n")
    dominant = (
        counts.sort_values(
            ["pitcher_id", "n", "pitcher_team_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("pitcher_id")
        .set_index("pitcher_id")["pitcher_team_id"]
    )
    pitcher_ids = last_year.index.to_numpy(np.int64)
    previous_team = dominant.reindex(last_year.index).fillna(-1).to_numpy(np.int64)
    payload = {
        "pitcher_ids": pitcher_ids,
        "last_year": last_year.to_numpy(np.int16),
        "previous_team": previous_team,
        "correction_keys": stats.index.to_numpy(dtype=str),
        "corrections": stats["correction"].to_numpy(np.float64),
        "group_n": stats["n"].to_numpy(np.int64),
        "source_years": np.asarray(SOURCE_YEARS, dtype=np.int16),
        "query_year": np.asarray([QUERY_YEAR], dtype=np.int16),
        "alpha": np.asarray([ALPHA], dtype=np.float64),
    }
    diagnostics: dict[str, Any] = {
        "source_rows_by_year": source_rows_by_year,
        "pooled_rows": int(len(pooled)),
        "groups": int(len(stats)),
        "known_pitchers": int(len(pitcher_ids)),
        "previous_season_pitchers": int((previous_team >= 0).sum()),
        "correction_rms_unweighted": float(
            np.sqrt(np.mean(np.square(stats["correction"].to_numpy(np.float64))))
        ),
        "correction_max_abs": float(stats["correction"].abs().max()),
        "residual_center_check": {
            key: float(value)
            for key, value in pooled.assign(
                source_year=np.repeat(
                    np.asarray(SOURCE_YEARS),
                    [source_rows_by_year[str(year)] for year in SOURCE_YEARS],
                )
            ).groupby("source_year", observed=True)["residual"].mean().items()
        },
    }
    return payload, diagnostics


def run(train_csv: Path, oof_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_type", "pitcher_id", "pitcher_team_id",
        "balls_before", "strikes_before", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    payload, diagnostics = build_lookup(train, oof_dir)
    lookup_path = output_dir / "player_transition_lookup.npz"
    np.savez_compressed(lookup_path, **payload)
    summary = {
        "protocol": PROTOCOL,
        "status": "finalized_for_aggressive_backup",
        "lookup_path": str(lookup_path),
        "lookup_sha256": sha256(lookup_path),
        "lookup_bytes": lookup_path.stat().st_size,
        "recipe": {
            "signal": "pitcher_status_count",
            "alpha": ALPHA,
            "deployed_weight": 0.25,
            "route": "R_CORE",
            "source_years": list(SOURCE_YEARS),
            "query_year": QUERY_YEAR,
        },
        "diagnostics": diagnostics,
        "restrictions": {
            "official_train_only": True,
            "source_predictions_forward_oof": True,
            "current_row_fields_only_at_inference": True,
            "test_csv_read": False,
            "test_distribution_or_order_used": False,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.oof_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

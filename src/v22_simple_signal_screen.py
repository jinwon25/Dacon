"""Audit simple row-local probability signals against reconstructed v21 OOF.

This is a diagnostic screen, not a deployment selector.  It asks whether the
champion has left obvious shrinkage, recent-form, or ASOF-posterior information
unused.  Each candidate is a small convex move from v21 toward a deterministic
single-row probability and is scored on all three champion audit axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v20_residual_overlay_screen import _bss


WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)
AXES = (
    "y2023_to_y2024",
    "y2023_early_to_late",
    "y2024_early_to_late",
)
RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
)


def _load_seasons(project: Path) -> dict[int, pd.DataFrame]:
    columns = [
        "season",
        "game_month",
        "asof_pitcher_n",
        "asof_batter_n",
        *RATE_COLUMNS,
    ]
    pieces: dict[int, list[pd.DataFrame]] = {2023: [], 2024: []}
    for chunk in pd.read_csv(
        project / "data" / "train.csv",
        usecols=columns,
        chunksize=200_000,
        low_memory=False,
    ):
        for year in pieces:
            selected = chunk.loc[chunk["season"].eq(year)]
            if len(selected):
                pieces[year].append(selected.copy())
    return {
        year: pd.concat(rows, ignore_index=True) for year, rows in pieces.items()
    }


def _axis_frame(seasons: dict[int, pd.DataFrame], name: str) -> pd.DataFrame:
    if name == "y2023_to_y2024":
        return seasons[2024].reset_index(drop=True)
    if name == "y2023_early_to_late":
        return seasons[2023].loc[seasons[2023]["game_month"].ge(8)].reset_index(drop=True)
    if name == "y2024_early_to_late":
        return seasons[2024].loc[seasons[2024]["game_month"].ge(8)].reset_index(drop=True)
    raise ValueError(name)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.5) -> np.ndarray:
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(default)
        .to_numpy(np.float64)
    )


def _signals(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    p = _numeric(frame, "asof_pitcher_success_rate")
    b = _numeric(frame, "asof_batter_success_rate")
    p_n = _numeric(frame, "asof_pitcher_n", 0.0)
    b_n = _numeric(frame, "asof_batter_n", 0.0)
    r1 = _numeric(frame, "asof_pitcher_prev1_game_success_rate")
    r3 = _numeric(frame, "asof_pitcher_prev3_game_success_rate")
    r5 = _numeric(frame, "asof_pitcher_prev5_game_success_rate")
    reverse = _numeric(frame, "asof_pitcher_reverse_rate", 0.0)
    middle = _numeric(frame, "asof_pitcher_middle_rate", 0.0)
    ball = _numeric(frame, "asof_pitcher_ball_rate", 0.0)
    strike = _numeric(frame, "asof_pitcher_strike_rate", 0.0)
    recent_middle_columns = []
    for column in (
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ):
        value = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        recent_middle_columns.append(np.where(np.isfinite(value), value, middle))
    recent_middle = np.column_stack(recent_middle_columns)
    output = {
        "constant_045": np.full(len(frame), 0.45),
        "constant_0475": np.full(len(frame), 0.475),
        "constant_050": np.full(len(frame), 0.50),
        "constant_0525": np.full(len(frame), 0.525),
        "constant_055": np.full(len(frame), 0.55),
        "pitcher_career": p,
        "batter_career": b,
        "pitcher75_batter25": 0.75 * p + 0.25 * b,
        "pitcher90_batter10": 0.90 * p + 0.10 * b,
        "recent1": r1,
        "recent3": r3,
        "recent5": r5,
        "recent_532": 0.50 * r1 + 0.30 * r3 + 0.20 * r5,
        "recent_235": 0.20 * r1 + 0.30 * r3 + 0.50 * r5,
        "career75_recent25": 0.75 * p + 0.25 * (0.2 * r1 + 0.3 * r3 + 0.5 * r5),
        "one_minus_reverse_middle": 1.0 - reverse - middle,
        "one_minus_middle": 1.0 - middle,
        "one_minus_reverse": 1.0 - reverse,
        "one_minus_ball": 1.0 - ball,
        "strike": strike,
        "one_minus_recent_middle": 1.0 - np.nanmean(recent_middle, axis=1),
    }
    for alpha in (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0):
        pitcher = (p_n * p + alpha * 0.5) / (p_n + alpha)
        batter = (b_n * b + alpha * 0.5) / (b_n + alpha)
        output[f"pitcher_k{int(alpha)}"] = pitcher
        output[f"pitcher75_batter25_k{int(alpha)}"] = 0.75 * pitcher + 0.25 * batter
    return {name: np.clip(value, 0.001, 0.999) for name, value in output.items()}


def run(project: Path, champion_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    champion_dir = (project / champion_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seasons = _load_seasons(project)
    rows = []
    for axis in AXES:
        frame = _axis_frame(seasons, axis)
        with np.load(champion_dir / f"{axis}.npz", allow_pickle=True) as saved:
            target = saved["target"].astype(np.float64)
            v21 = saved["v21"].astype(np.float64)
        if len(frame) != len(target):
            raise ValueError(f"row count mismatch for {axis}")
        for name, signal in _signals(frame).items():
            for weight in WEIGHTS:
                candidate = np.clip(v21 + weight * (signal - v21), 0.001, 0.999)
                rows.append(
                    {
                        "axis": axis,
                        "signal": name,
                        "weight": weight,
                        "gain": _bss(target, candidate) - _bss(target, v21),
                        "mean_shift": float(np.mean(candidate - v21)),
                        "mean_abs_shift": float(np.mean(np.abs(candidate - v21))),
                    }
                )
    metrics = pd.DataFrame(rows)
    robust = (
        metrics.groupby(["signal", "weight"], as_index=False)
        .agg(
            min_gain=("gain", "min"),
            mean_gain=("gain", "mean"),
            max_gain=("gain", "max"),
        )
        .sort_values(["min_gain", "mean_gain"], ascending=False)
        .reset_index(drop=True)
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "V22_SIMPLE_ROW_LOCAL_SIGNAL_SCREEN_V1",
        "axes": list(AXES),
        "candidate_count": int(len(robust)),
        "strictly_positive": int((robust["min_gain"] > 0.0).sum()),
        "best": robust.head(30).to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--champion-dir",
        type=Path,
        default=Path("artifacts/champion_oof_20260817_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v22_simple_signal_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.champion_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Gate the v321 Beta-Binomial R complement by within-season maturity.

The dose is frozen at the source-selected v321 R weight (0.10).  Only a small,
predeclared library of sample-support and calendar-maturity gates is compared.
Gate selection uses full-2022 and late-2023 only; full-2024 remains locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v321_strict_beta_binomial_complement import (
    TARGET,
    bss,
    current_season_state,
    terminal_snapshot,
)


PROTOCOL = "V322_BETA_MATURITY_GATE_V1"
FROZEN_DOSE = 0.10


def maturity_gates(frame: pd.DataFrame, train: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    history = train.loc[train["season"].lt(year)].reset_index(drop=True)
    pitcher_snapshot = terminal_snapshot(
        history, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
    )
    pitcher_n, _ = current_season_state(
        frame,
        pitcher_snapshot,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    return {
        "all": np.ones(len(frame), dtype=np.float64),
        "linear_n200": np.minimum(1.0, pitcher_n / 200.0),
        "linear_n500": np.minimum(1.0, pitcher_n / 500.0),
        "pitcher_n_ge100": (pitcher_n >= 100.0).astype(np.float64),
        "pitcher_n_ge300": (pitcher_n >= 300.0).astype(np.float64),
        "pitcher_n_ge600": (pitcher_n >= 600.0).astype(np.float64),
        "month_ge7": frame["game_month"].ge(7).to_numpy(np.float64),
        "month_ge8": frame["game_month"].ge(8).to_numpy(np.float64),
        "month_ge9": frame["game_month"].ge(9).to_numpy(np.float64),
    }


def axis_metrics(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    target = frame[TARGET].to_numpy(np.float64)
    months = []
    for month in sorted(frame["game_month"].unique()):
        selected = frame["game_month"].eq(month).to_numpy()
        months.append(
            {
                "month": int(month),
                "rows": int(selected.sum()),
                "active_rows": int(np.sum(active[selected] > 0.0)),
                "gain": bss(target[selected], candidate[selected])
                - bss(target[selected], baseline[selected]),
            }
        )
    active_months = [row for row in months if row["active_rows"] > 0]
    shift = candidate - baseline
    return {
        "gain": bss(target, candidate) - bss(target, baseline),
        "active_rows": int(np.sum(active > 0.0)),
        "active_fraction": float(np.mean(active > 0.0)),
        "active_month_positive_fraction": float(
            np.mean([row["gain"] > 0.0 for row in active_months])
        ),
        "worst_active_month_gain": float(min(row["gain"] for row in active_months)),
        "rms_shift": float(np.sqrt(np.mean(shift**2))),
        "months": months,
    }


def run(train_csv: Path, v321_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    years = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    with np.load(v321_axes, allow_pickle=False) as saved:
        baselines = {
            name: saved[f"baseline_{name}"].astype(np.float64) for name in frames
        }
        beta = {name: saved[f"beta_{name}"].astype(np.float64) for name in frames}
    gates = {
        name: maturity_gates(frame, train, years[name]) for name, frame in frames.items()
    }
    gate_names = tuple(gates["full_2022"])
    source_rows = []
    for gate_name in gate_names:
        gains = []
        for name in ("full_2022", "late_2023"):
            regular = frames[name]["game_type"].astype(str).eq("R").to_numpy(np.float64)
            active = regular * gates[name][gate_name]
            candidate = np.clip(
                baselines[name] + FROZEN_DOSE * active * (beta[name] - baselines[name]),
                0.001,
                0.999,
            )
            gains.append(
                bss(frames[name][TARGET].to_numpy(np.float64), candidate)
                - bss(frames[name][TARGET].to_numpy(np.float64), baselines[name])
            )
        source_rows.append(
            {
                "gate": gate_name,
                "full_2022_gain": gains[0],
                "late_2023_gain": gains[1],
                "source_min_gain": float(min(gains)),
                "source_mean_gain": float(np.mean(gains)),
            }
        )
    selected = max(
        source_rows,
        key=lambda row: (row["source_min_gain"], row["source_mean_gain"]),
    )
    selected_gate = str(selected["gate"])

    candidates: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, frame in frames.items():
        regular = frame["game_type"].astype(str).eq("R").to_numpy(np.float64)
        active = regular * gates[name][selected_gate]
        candidates[name] = np.clip(
            baselines[name] + FROZEN_DOSE * active * (beta[name] - baselines[name]),
            0.001,
            0.999,
        )
        metrics[name] = axis_metrics(frame, baselines[name], candidates[name], active)

    source_pass = bool(
        metrics["full_2022"]["gain"] > 0.0 and metrics["late_2023"]["gain"] > 0.0
    )
    locked_pass = bool(
        metrics["full_2024"]["gain"] > 0.0
        and metrics["full_2024"]["active_month_positive_fraction"] >= 0.5
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if source_pass and locked_pass else "screen_reject",
        "frozen_dose": FROZEN_DOSE,
        "selected": selected,
        "source_candidates": source_rows,
        "metrics": metrics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_portfolio": bool(source_pass and locked_pass),
        "restrictions": {
            "official_train_only": True,
            "gate_selected_on_full_2022_and_late_2023_only": True,
            "full_2024_locked_confirmation": True,
            "v321_dose_frozen": True,
            "r_only": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{
            f"baseline_{name}": baselines[name] for name in frames
        },
        **{
            f"candidate_{name}": candidates[name] for name in frames
        },
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v321-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v321_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

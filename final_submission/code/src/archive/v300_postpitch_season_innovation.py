"""Audit post-pitch season innovations above the frozen v290-equivalent axes.

v286/v290 fixed the fallback model's career-count anchor, but deliberately
left the final prior-season pitch in the new-season delta and only changed
rows routed through the fallback model.  For success rates the official train
target makes the true post-pitch terminal state observable.  This experiment
uses that state to construct row-local current-season pitcher and batter
innovations, then applies them to every R_CORE row.  Candidate selection uses
only full-2022 and late-2023; full-2024 is locked confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "V300_POSTPITCH_SEASON_INNOVATION_V1"
TARGET = "control_success"
SHRINKAGES = (50.0, 100.0, 200.0, 400.0)
DOSES = (0.05, 0.10, 0.20, 0.30)


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rate = float(target.mean())
    return float(100000.0 * (1.0 - np.mean((target - prediction) ** 2) / (rate * (1.0 - rate))))


def terminal_snapshot(
    history: pd.DataFrame,
    id_column: str,
    n_column: str,
    rate_column: str,
) -> dict[int, tuple[float, float]]:
    """Return the exact observable post-pitch terminal success state."""
    work = history[[id_column, n_column, rate_column, TARGET]].copy()
    work["_n"] = pd.to_numeric(work[n_column], errors="coerce").fillna(0.0)
    work["_events"] = (
        work["_n"]
        * pd.to_numeric(work[rate_column], errors="coerce").fillna(0.0)
        + pd.to_numeric(work[TARGET], errors="raise")
    )
    work["_n_after"] = work["_n"] + 1.0
    latest = work.loc[work.groupby(id_column, sort=False)["_n"].idxmax()]
    return {
        int(row[0]): (float(row[1]), float(row[2]))
        for row in latest[[id_column, "_n_after", "_events"]].itertuples(index=False, name=None)
    }


def season_innovation(
    frame: pd.DataFrame,
    snapshot: dict[int, tuple[float, float]],
    id_column: str,
    n_column: str,
    rate_column: str,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = pd.to_numeric(frame[id_column], errors="raise").to_numpy(np.int64)
    opening_n = np.fromiter(
        (snapshot.get(int(value), (0.0, 0.0))[0] for value in ids),
        dtype=np.float64,
        count=len(frame),
    )
    opening_events = np.fromiter(
        (snapshot.get(int(value), (0.0, 0.0))[1] for value in ids),
        dtype=np.float64,
        count=len(frame),
    )
    career_n = pd.to_numeric(frame[n_column], errors="coerce").fillna(0.0).to_numpy(np.float64)
    career_rate = pd.to_numeric(frame[rate_column], errors="coerce").fillna(0.5).to_numpy(np.float64)
    career_events = career_n * career_rate
    season_n = np.maximum(career_n - opening_n, 0.0)
    season_events = np.clip(career_events - opening_events, 0.0, season_n)
    smoothed = (season_events + shrinkage * career_rate) / (season_n + shrinkage)
    return smoothed - career_rate, season_n, opening_n


def axis_metrics(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    target = frame[TARGET].to_numpy(np.float64)
    gains = []
    for month in sorted(frame["game_month"].unique()):
        selected = frame["game_month"].eq(month).to_numpy()
        gains.append({
            "month": int(month),
            "rows": int(selected.sum()),
            "gain": bss(target[selected], candidate[selected]) - bss(target[selected], baseline[selected]),
        })
    shift = candidate - baseline
    return {
        "gain": bss(target, candidate) - bss(target, baseline),
        "positive_month_fraction": float(np.mean([row["gain"] > 0.0 for row in gains])),
        "worst_month_gain": float(min(row["gain"] for row in gains)),
        "mean_abs_shift": float(np.mean(np.abs(shift))),
        "rms_shift": float(np.sqrt(np.mean(shift ** 2))),
        "changed_rows": int(np.sum(np.abs(shift) > 0.0)),
        "months": gains,
    }


def run(train_csv: Path, v285_axes: Path, v288_axes: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    with np.load(v285_axes, allow_pickle=False) as saved:
        baseline_2022 = saved["candidate_full_2022"].astype(np.float64)
    with np.load(v288_axes, allow_pickle=False) as saved:
        baseline_late_2023 = saved["candidate_late_2023"].astype(np.float64)
        baseline_2024 = saved["candidate_full_2024"].astype(np.float64)

    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[train["season"].eq(2023) & train["game_month"].ge(8)].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    baselines = {
        "full_2022": baseline_2022,
        "late_2023": baseline_late_2023,
        "full_2024": baseline_2024,
    }
    years = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    for name, frame in frames.items():
        if len(frame) != len(baselines[name]):
            raise ValueError(f"axis alignment mismatch: {name}")

    signals: dict[tuple[str, float], np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for name, frame in frames.items():
        year = years[name]
        history = train.loc[train["season"].lt(year)].reset_index(drop=True)
        pitcher_snapshot = terminal_snapshot(
            history, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"
        )
        batter_snapshot = terminal_snapshot(
            history, "batter_id", "asof_batter_n", "asof_batter_success_rate"
        )
        first = frame.sort_values("asof_pitcher_n").groupby("pitcher_id", sort=False).head(1)
        first_ids = first["pitcher_id"].astype(int).to_numpy()
        first_n = first["asof_pitcher_n"].to_numpy(np.float64)
        opening_n = np.asarray([pitcher_snapshot.get(int(value), (0.0, 0.0))[0] for value in first_ids])
        returning = np.asarray([int(value) in pitcher_snapshot for value in first_ids])
        diagnostics[name] = {
            "returning_pitchers": int(returning.sum()),
            "opening_n_exact_fraction": float(np.mean(np.isclose(first_n[returning], opening_n[returning], atol=1e-6))) if returning.any() else None,
            "opening_n_mean_abs_error": float(np.mean(np.abs(first_n[returning] - opening_n[returning]))) if returning.any() else None,
        }
        for shrinkage in SHRINKAGES:
            pitcher_delta, pitcher_n, _ = season_innovation(
                frame, pitcher_snapshot, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", shrinkage
            )
            batter_delta, batter_n, _ = season_innovation(
                frame, batter_snapshot, "batter_id", "asof_batter_n", "asof_batter_success_rate", shrinkage
            )
            # Pitcher command is primary; batter expectation supplies a smaller
            # but separately observed current-season environment adjustment.
            reliability = np.minimum(1.0, (pitcher_n + 0.25 * batter_n) / 150.0)
            signals[(name, shrinkage)] = reliability * (0.75 * pitcher_delta + 0.25 * batter_delta)

    rows = []
    candidates: dict[tuple[str, float, float], np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for shrinkage in SHRINKAGES:
        for dose in DOSES:
            key = f"k{shrinkage:g}_d{dose:g}"
            per_axis = {}
            for name, frame in frames.items():
                regular = frame["game_type"].astype(str).eq("R").to_numpy()
                rcore = regular & ~(
                    frame["pitcher_team_id"].eq(13).to_numpy()
                    | frame["batter_team_id"].eq(13).to_numpy()
                )
                candidate = baselines[name].copy()
                candidate[rcore] = np.clip(
                    candidate[rcore] + dose * signals[(name, shrinkage)][rcore], 0.001, 0.999
                )
                candidates[(name, shrinkage, dose)] = candidate
                per_axis[name] = axis_metrics(frame, baselines[name], candidate)
            source_gains = [per_axis["full_2022"]["gain"], per_axis["late_2023"]["gain"]]
            rows.append({
                "key": key,
                "shrinkage": shrinkage,
                "dose": dose,
                "source_min_gain": float(min(source_gains)),
                "source_mean_gain": float(np.mean(source_gains)),
                "locked_gain": per_axis["full_2024"]["gain"],
            })
            metrics[key] = per_axis

    # Maximin selection is fully source-only.  The locked axis cannot change it.
    selected = max(rows, key=lambda row: (row["source_min_gain"], row["source_mean_gain"], -row["dose"]))
    selected_metrics = metrics[str(selected["key"])]
    source_pass = bool(selected["source_min_gain"] > 0.0)
    locked_pass = bool(
        selected_metrics["full_2024"]["gain"] > 0.0
        and selected_metrics["full_2024"]["positive_month_fraction"] >= 0.625
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if source_pass and locked_pass else "screen_reject",
        "selected": selected,
        "selected_metrics": selected_metrics,
        "all_candidates": rows,
        "opening_anchor_diagnostics": diagnostics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_packaging": bool(source_pass and locked_pass and selected["locked_gain"] >= 2.0),
        "restrictions": {
            "official_train_only": True,
            "post_pitch_anchor_uses_only_prior_season_target": True,
            "source_only_selection": True,
            "full_2024_locked_confirmation": True,
            "r_core_only": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        baseline_full_2022=baselines["full_2022"],
        candidate_full_2022=candidates[("full_2022", selected["shrinkage"], selected["dose"])],
        baseline_late_2023=baselines["late_2023"],
        candidate_late_2023=candidates[("late_2023", selected["shrinkage"], selected["dose"])],
        baseline_full_2024=baselines["full_2024"],
        candidate_full_2024=candidates[("full_2024", selected["shrinkage"], selected["dose"])],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v288-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.v285_axes, args.v288_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

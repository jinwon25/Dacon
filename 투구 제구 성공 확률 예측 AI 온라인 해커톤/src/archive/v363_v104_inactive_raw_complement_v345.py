"""Audit the frozen v103 raw correction on v104-inactive R_CORE rows.

v104 retained the v103 FM/conditional correction only when at least two of
three pre-frozen safety lists agreed.  This audit asks whether a conservative
fraction of that already-frozen direction transports on the disjoint inactive
rows above v345.  Region and dose are selected on full 2022 and late 2023;
full 2024 is opened once only after the recipe passes both sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness


PROTOCOL = "V363_V104_INACTIVE_RAW_COMPLEMENT_V345_V1"
TARGET = "control_success"
SAFE = {
    "count": ("0-0", "1-1", "1-2", "2-0", "3-1", "3-2"),
    "history": ("1000+", "200-999", "30-199"),
    "platoon": ("1-1", "2-1", "2-2"),
}
REGIONS = ("ONE_VOTE", "ZERO_VOTES", "INACTIVE_ALL")
DOSES = (0.25, 0.50, 1.00)


def safe_votes(frame: pd.DataFrame) -> np.ndarray:
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int8)
    count = np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str))
    history = pd.cut(
        pd.to_numeric(frame["asof_pitcher_n"], errors="raise"),
        bins=[-np.inf, 29, 199, 999, np.inf],
        labels=["0-29", "30-199", "200-999", "1000+"],
    ).astype(str).to_numpy()
    platoon = np.char.add(
        np.char.add(frame["pitcher_hand"].astype(str).to_numpy(), "-"),
        frame["batter_hand"].astype(str).to_numpy(),
    )
    return (
        np.isin(count, SAFE["count"]).astype(np.int8)
        + np.isin(history, SAFE["history"]).astype(np.int8)
        + np.isin(platoon, SAFE["platoon"]).astype(np.int8)
    )


def rcore_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return regular & ~anchor


def region_mask(frame: pd.DataFrame, region: str) -> np.ndarray:
    votes = safe_votes(frame)
    rcore = rcore_mask(frame)
    if region == "ONE_VOTE":
        return rcore & (votes == 1)
    if region == "ZERO_VOTES":
        return rcore & (votes == 0)
    if region == "INACTIVE_ALL":
        return rcore & (votes < 2)
    raise ValueError(f"unknown region: {region}")


def make_candidate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    direction: np.ndarray,
    region: str,
    dose: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = region_mask(frame, region) & np.not_equal(direction, 0.0)
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(dose) * np.asarray(direction)[active],
        0.001,
        0.999,
    )
    return output, active


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(candidate) - np.asarray(parent)))))


def axis_metrics(
    frame: pd.DataFrame,
    parent: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    """Paired Brier diagnostics with a full-axis denominator for sparse cells."""
    target = frame[TARGET].to_numpy(np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    rate = float(target.mean())
    denominator = rate * (1.0 - rate)
    if denominator <= 0.0:
        raise ValueError("full-axis Brier denominator is undefined")

    def gain(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        baseline_mse = float(np.mean(np.square(target[mask] - parent[mask])))
        candidate_mse = float(np.mean(np.square(target[mask] - candidate[mask])))
        return 100000.0 * (baseline_mse - candidate_mse) / denominator

    months = []
    for month in sorted(frame.loc[active, "game_month"].unique().tolist()):
        mask = active & frame["game_month"].eq(month).to_numpy()
        months.append({"month": int(month), "n_rows": int(mask.sum()), "gain": gain(mask)})
    month_gains = np.asarray([item["gain"] for item in months], dtype=np.float64)
    return {
        "gain": gain(np.ones(len(frame), dtype=bool)),
        "candidate_brier": float(np.mean(np.square(target - candidate))),
        "baseline_brier": float(np.mean(np.square(target - parent))),
        "active_gain": gain(active),
        "active_rows": int(active.sum()),
        "mean_abs_shift_active": (
            float(np.mean(np.abs(candidate[active] - parent[active])))
            if np.any(active) else 0.0
        ),
        "positive_month_fraction": (
            float(np.mean(month_gains > 0.0)) if len(month_gains) else 0.0
        ),
        "worst_month_gain": float(month_gains.min()) if len(month_gains) else 0.0,
        "months": months,
    }


def load_raw_directions(
    v84_dir: Path, v103_axes: Path
) -> dict[str, np.ndarray]:
    names = ("full_2022", "late_2023", "full_2024")
    raw: dict[str, np.ndarray] = {}
    with np.load(v103_axes, allow_pickle=False) as saved:
        candidates = {
            name: saved[f"candidate_{name}"].astype(np.float64) for name in names
        }
    for name in names:
        with np.load(v84_dir / f"v84_{name}.npz", allow_pickle=False) as saved:
            parent = saved["parent"].astype(np.float64)
        if len(parent) != len(candidates[name]):
            raise ValueError(f"v84/v103 alignment mismatch: {name}")
        raw[name] = candidates[name] - parent
    return raw


def run(
    train_csv: Path,
    v84_dir: Path,
    v103_axes: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", "balls_before", "strikes_before",
        "pitcher_hand", "batter_hand", "asof_pitcher_n", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    directions = load_raw_directions(v84_dir, v103_axes)
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in ("full_2022", "late_2023")
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = (
            saved["candidate_full_2024"].astype(np.float64)
            - saved["parent_full_2024"].astype(np.float64)
        )
    for name in frames:
        if not (len(frames[name]) == len(parents[name]) == len(directions[name])):
            raise ValueError(f"axis alignment mismatch: {name}")

    rows: list[dict[str, Any]] = []
    payload: dict[tuple[str, float], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for region in REGIONS:
        for dose in DOSES:
            key = (region, float(dose))
            payload[key] = {}
            record: dict[str, Any] = {"region": region, "dose": float(dose)}
            for axis in ("full_2022", "late_2023"):
                candidate, active = make_candidate(
                    frames[axis], parents[axis], directions[axis], region, dose
                )
                payload[key][axis] = (candidate, active)
                metrics = axis_metrics(frames[axis], parents[axis], candidate, active)
                record[f"{axis}_gain"] = metrics["gain"]
                record[f"{axis}_positive_month_fraction"] = metrics[
                    "positive_month_fraction"
                ]
                record[f"{axis}_worst_month_gain"] = metrics["worst_month_gain"]
                record[f"{axis}_active_rows"] = metrics["active_rows"]
                record[f"{axis}_rms"] = full_row_rms(parents[axis], candidate)
            record["minimum_source_gain"] = min(
                record["full_2022_gain"], record["late_2023_gain"]
            )
            record["source_pass"] = bool(
                record["full_2022_gain"] > 0.0
                and record["late_2023_gain"] > 0.0
                and record["full_2022_positive_month_fraction"] >= 4.0 / 7.0
                and record["late_2023_positive_month_fraction"] >= 2.0 / 3.0
                and record["full_2022_worst_month_gain"] > -10.0
                and record["late_2023_worst_month_gain"] > -10.0
            )
            rows.append(record)
    grid = pd.DataFrame(rows).sort_values(
        ["source_pass", "minimum_source_gain", "full_2022_gain", "late_2023_gain"],
        ascending=False,
    ).reset_index(drop=True)
    grid.to_csv(output_dir / "source_grid.csv", index=False, encoding="utf-8-sig")
    passing = grid.loc[grid["source_pass"]]
    restrictions = {
        "official_train_only": True,
        "v103_raw_direction_frozen": True,
        "v104_safe_lists_frozen": True,
        "region_and_dose_selected_on_full2022_and_late2023_only": True,
        "full2024_opened_once_after_source_gate": True,
        "v345_f_anchor_and_v104_active_rows_preserved_exactly": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_grid": rows,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return summary

    chosen = passing.iloc[0]
    recipe = {"region": str(chosen["region"]), "dose": float(chosen["dose"])}
    locked_candidate, locked_active = make_candidate(
        frames["full_2024"], parents["full_2024"], directions["full_2024"],
        recipe["region"], recipe["dose"],
    )
    locked = axis_metrics(
        frames["full_2024"], parents["full_2024"], locked_candidate, locked_active
    )
    locked["full_row_rms_shift"] = full_row_rms(parents["full_2024"], locked_candidate)
    protected = ~locked_active
    if not np.array_equal(locked_candidate[protected], parents["full_2024"][protected]):
        raise ValueError("protected row parity failure")
    increment = locked_candidate - parents["full_2024"]
    nonzero = (np.abs(increment) > 1e-15) | (np.abs(v345_increment) > 1e-15)
    correlation = (
        float(np.corrcoef(increment[nonzero], v345_increment[nonzero])[0, 1])
        if int(nonzero.sum()) > 2
        and float(np.std(increment[nonzero])) > 0.0
        and float(np.std(v345_increment[nonzero])) > 0.0
        else 0.0
    )
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], locked_candidate, locked_active,
        [parents["full_2024"], locked_candidate],
    )
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    key = (recipe["region"], recipe["dose"])
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2022=parents["full_2022"],
        candidate_full_2022=payload[key]["full_2022"][0],
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=payload[key]["late_2023"][0],
        parent_full_2024=parents["full_2024"],
        candidate_full_2024=locked_candidate,
        active_full_2024=locked_active,
        raw_direction_full_2024=directions["full_2024"],
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "selected_recipe": recipe,
        "source_selected": chosen.to_dict(),
        "source_passing_recipes": int(len(passing)),
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": passed,
        "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v84-dir", type=Path, required=True)
    parser.add_argument("--v103-axes", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.v84_dir, args.v103_axes, args.v335_axes,
        args.v345_axes, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

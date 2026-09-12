"""Blend v214 joint and v220 prior-role H1 using source years only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import affine, metrics
from src.archive.v205_h1_workload_strict_forward_audit import (
    apply_component_delta,
    strict_axis,
)
from src.champion.v130_hoo_independent_oof_blend import post4


PROTOCOL = "V221_JOINT_ROLE_H1_SOURCE_BLEND_V1"
YEARS = (2022, 2023, 2024)
ROLE_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
SCREEN_SCALE = 0.10


def select_role_weight(results: dict[str, dict[str, Any]]) -> float | None:
    eligible: list[tuple[float, float, float]] = []
    for key, yearly in results.items():
        source = [yearly["2022"], yearly["2023"]]
        if all(
            item["gain"] > 0.0
            and item["positive_month_fraction"] >= 0.70
            and item["worst_month_gain"] > -5.0
            for item in source
        ):
            gains = [item["gain"] for item in source]
            eligible.append((min(gains), float(np.mean(gains)), float(key)))
    return None if not eligible else max(eligible)[2]


def restrictions() -> dict[str, bool]:
    return {
        "fixed_seed42_screen": True,
        "fixed_screen_scale": True,
        "source_only_weight_selection": True,
        "locked_2024_not_used_for_selection": True,
        "no_new_model_fit": True,
        "test_csv_read": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    joint_checkpoint_dir: Path,
    role_checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "season", "game_month", "pitcher_id", "batter_id", "batter_hand",
        "balls_before", "strikes_before", "num_runners_on", "control_success",
    ]
    train = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    season = train["season"].to_numpy(np.int16)
    target = train["control_success"].to_numpy(np.float64)
    prepared: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        audit = season == year
        frame = train.loc[audit].reset_index(drop=True)
        correction = post4(train.loc[season < year].reset_index(drop=True), frame)
        baseline = affine(np.load(
            baseline_checkpoint_dir / f"h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64) + correction)
        joint = affine(np.load(
            joint_checkpoint_dir / f"joint_h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64) + correction)
        role = affine(np.load(
            role_checkpoint_dir / f"role_joint_h1_year{year}_seed42.npy",
            allow_pickle=False,
        ).astype(np.float64) + correction)
        prepared[year] = {
            "axis": strict_axis(frame, target[audit], baseline),
            "baseline": baseline,
            "joint": joint,
            "role": role,
        }

    details: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[int, np.ndarray]] = {}
    for role_weight in ROLE_WEIGHTS:
        key = f"{role_weight:.2f}"
        details[key], predictions[key] = {}, {}
        for year in YEARS:
            values = prepared[year]
            h1 = (
                (1.0 - role_weight) * values["joint"]
                + role_weight * values["role"]
            )
            candidate = apply_component_delta(
                values["baseline"], h1, SCREEN_SCALE
            )
            predictions[key][year] = candidate
            details[key][str(year)] = metrics(
                values["axis"], values["baseline"], candidate
            )
    selected = select_role_weight(details)
    selected_key = None if selected is None else f"{selected:.2f}"
    locked = None if selected_key is None else details[selected_key]["2024"]
    joint_locked = details["0.00"]["2024"]
    eligible = bool(
        locked is not None
        and locked["gain"] > joint_locked["gain"]
        and locked["positive_month_fraction"] >= 0.875
        and locked["worst_month_gain"] > 0.0
    )
    if selected_key is not None:
        np.savez_compressed(
            output_dir / "selected_blend_axes.npz",
            **{
                f"baseline_{year}": prepared[year]["baseline"]
                for year in YEARS
            },
            **{
                f"candidate_{year}": predictions[selected_key][year]
                for year in YEARS
            },
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirm_multiseed" if eligible else "screen_reject",
        "role_weight_results": details,
        "selected_role_weight_from_sources": selected,
        "locked_2024": locked,
        "joint_only_locked_2024": joint_locked,
        "locked_incremental_gain": None if locked is None else float(
            locked["gain"] - joint_locked["gain"]
        ),
        "eligible_for_multiseed_confirmation": eligible,
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--joint-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--role-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.baseline_checkpoint_dir,
        args.joint_checkpoint_dir, args.role_checkpoint_dir, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

"""Screen fixed row-local workload corrections above the exact JY parent.

v199 established that the common denominators of the official previous-game
success and middle rates recover a real recent pitch-count lower bound.  This
experiment turns that signal into a deliberately small, target-free direction
library.  Recipes and doses are selected on full-2022 plus late-2023; full-2024
is opened only for the frozen selection and robustness audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v199_recent_workload_reconstruction import (
    HORIZONS,
    MAX_DENOMINATOR,
    infer_denominators,
    rate_columns,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V200_RECENT_WORKLOAD_DIRECTION_SCREEN_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
SHRINKAGE = {1: 20.0, 3: 60.0, 5: 100.0}
WEIGHTS = (0.0025, 0.005, 0.01, 0.025)
WORKLOAD_COLUMNS = (
    "season",
    "game_month",
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
)


def workload_directions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Create fixed probability-scale directions from one row at a time."""

    career_success = (
        pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce")
        .fillna(0.5)
        .to_numpy(np.float64)
    )
    career_middle = (
        pd.to_numeric(frame["asof_pitcher_middle_rate"], errors="coerce")
        .fillna(0.2)
        .to_numpy(np.float64)
    )
    success_corrections: dict[int, np.ndarray] = {}
    middle_corrections: dict[int, np.ndarray] = {}
    success_rates: dict[int, np.ndarray] = {}
    middle_rates: dict[int, np.ndarray] = {}
    denominator: dict[int, np.ndarray] = {}
    valid: dict[int, np.ndarray] = {}

    for horizon in HORIZONS:
        success_column, middle_column = rate_columns(horizon)
        raw_success = pd.to_numeric(frame[success_column], errors="coerce").to_numpy(
            np.float64
        )
        raw_middle = pd.to_numeric(frame[middle_column], errors="coerce").to_numpy(
            np.float64
        )
        inferred = infer_denominators(
            pd.Series(raw_success),
            pd.Series(raw_middle),
            max_denominator=MAX_DENOMINATOR[horizon],
        )
        n = pd.to_numeric(
            inferred["minimum_denominator"], errors="coerce"
        ).to_numpy(np.float64)
        n = np.nan_to_num(n, nan=0.0)
        row_valid = (
            inferred["rounded_rational_fit"].fillna(False).to_numpy(bool)
            & np.isfinite(raw_success)
            & np.isfinite(raw_middle)
            & (n > 0.0)
        )
        success = np.where(row_valid, raw_success, career_success)
        middle = np.where(row_valid, raw_middle, career_middle)
        strength = float(SHRINKAGE[horizon])
        shrink_fraction = np.divide(
            strength,
            n + strength,
            out=np.zeros(len(frame), dtype=np.float64),
            where=row_valid,
        )
        success_rates[horizon] = success
        middle_rates[horizon] = middle
        denominator[horizon] = n
        valid[horizon] = row_valid
        # EB minus the raw recent rate: small workloads move farther back to
        # the official career prior, while large workloads remain nearly raw.
        success_corrections[horizon] = shrink_fraction * (career_success - success)
        middle_corrections[horizon] = shrink_fraction * (career_middle - middle)

    fixed_weights = {1: 0.5, 3: 0.3, 5: 0.2}
    success_consensus = sum(
        fixed_weights[horizon] * success_corrections[horizon]
        for horizon in HORIZONS
    )
    middle_consensus = sum(
        fixed_weights[horizon] * middle_corrections[horizon]
        for horizon in HORIZONS
    )

    # Compare a denominator-aware recent average with the fixed 0.5/0.3/0.2
    # average used by the existing row-state recipe.  Nested windows are
    # downweighted rather than treated as independent samples.
    decay = {1: 1.0, 3: 0.5, 5: 0.25}
    effective = {
        horizon: decay[horizon] * denominator[horizon] * valid[horizon]
        for horizon in HORIZONS
    }
    total = sum(effective.values())
    denominator_success = np.divide(
        sum(effective[h] * success_rates[h] for h in HORIZONS),
        total,
        out=sum(fixed_weights[h] * success_rates[h] for h in HORIZONS),
        where=total > 0.0,
    )
    denominator_middle = np.divide(
        sum(effective[h] * middle_rates[h] for h in HORIZONS),
        total,
        out=sum(fixed_weights[h] * middle_rates[h] for h in HORIZONS),
        where=total > 0.0,
    )
    fixed_success = sum(fixed_weights[h] * success_rates[h] for h in HORIZONS)
    fixed_middle = sum(fixed_weights[h] * middle_rates[h] for h in HORIZONS)

    output = {
        "success_prev1_shrink": success_corrections[1],
        "success_prev3_shrink": success_corrections[3],
        "success_prev5_shrink": success_corrections[5],
        "success_shrink_consensus": success_consensus,
        "middle_shrink_consensus": middle_consensus,
        "success_denominator_reweight": denominator_success - fixed_success,
        "middle_denominator_reweight": denominator_middle - fixed_middle,
    }
    return {
        name: np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        for name, values in output.items()
    }


def apply_direction(
    parent: np.ndarray,
    direction: np.ndarray,
    exact: np.ndarray,
    domain3: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.asarray(exact, dtype=bool)
        & np.asarray(domain3).astype(str).__eq__("R_CORE")
        & np.not_equal(np.asarray(direction, dtype=np.float64), 0.0)
    )
    candidate = np.asarray(parent, dtype=np.float64).copy()
    candidate[active] = np.clip(
        candidate[active] + float(weight) * np.asarray(direction)[active],
        0.001,
        0.999,
    )
    return candidate, active


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.5
        and result["worst_month_gain"] > -2.0
        and result["minimum_domain_gain"] >= 0.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "fixed_target_free_direction_library": True,
        "source_selection_before_locked_2024": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, raw_frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )

    workload = pd.read_csv(train_csv, usecols=list(WORKLOAD_COLUMNS), low_memory=False)
    season = workload["season"].to_numpy(np.int16)
    by_year = {
        year: workload.loc[season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = by_year[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": by_year[2022],
        "late_2023": by_year[2023].loc[late23].reset_index(drop=True),
        "full_2024": by_year[2024],
    }
    directions = {axis: workload_directions(frames[axis]) for axis in AXES}
    direction_names = tuple(directions["full_2022"])
    if any(tuple(directions[axis]) != direction_names for axis in AXES):
        raise ValueError("direction library mismatch")
    for axis in AXES:
        if len(frames[axis]) != len(parents[axis]):
            raise ValueError(f"axis alignment mismatch: {axis}")

    rows: list[dict[str, Any]] = []
    detail: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    active_masks: dict[str, dict[str, np.ndarray]] = {}
    for name in direction_names:
        for weight in WEIGHTS:
            key = f"{name}__w={weight:g}"
            detail[key] = {}
            candidates[key] = {}
            active_masks[key] = {}
            row: dict[str, Any] = {"direction": name, "weight": weight}
            for axis in AXES:
                candidate, active = apply_direction(
                    parents[axis], directions[axis][name],
                    axes[axis]["exact_mask"], axes[axis]["domain3"], weight,
                )
                candidates[key][axis] = candidate
                active_masks[key][axis] = active
                score = metrics(axes[axis], parents[axis], candidate)
                detail[key][axis] = score
                row[f"{axis}_gain"] = score["gain"]
                row[f"{axis}_month_fraction"] = score["positive_month_fraction"]
                row[f"{axis}_worst_month"] = score["worst_month_gain"]
            row["source_gate_passed"] = all(
                source_gate(detail[key][axis]) for axis in SOURCE_AXES
            )
            row["source_min_gain"] = min(
                detail[key][axis]["gain"] for axis in SOURCE_AXES
            )
            row["source_mean_gain"] = float(
                np.mean([detail[key][axis]["gain"] for axis in SOURCE_AXES])
            )
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain"],
        ascending=[False, False, False],
        kind="stable",
    )
    table.to_csv(output_dir / "source_direction_screen.csv", index=False, encoding="utf-8-sig")
    passing = table.loc[table["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "candidate_count": int(len(table)),
            "source_top": table.head(10).to_dict(orient="records"),
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected = passing.iloc[0]
        selected_key = f"{selected['direction']}__w={float(selected['weight']):g}"
        selected_candidate = candidates[selected_key]["full_2024"]
        selected_active = active_masks[selected_key]["full_2024"]
        locked_family = [
            candidates[f"{row.direction}__w={float(row.weight):g}"]["full_2024"]
            for row in passing.itertuples(index=False)
        ]
        locked_family.append(parents["full_2024"].copy())
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], selected_candidate,
            selected_active, locked_family,
        )
        locked = detail[selected_key]["full_2024"]
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -2.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"],
            candidate=selected_candidate,
            direction=directions["full_2024"][str(selected["direction"])],
            active=selected_active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "candidate_count": int(len(table)),
            "source_passing_count": int(len(passing)),
            "selected": {
                "direction": str(selected["direction"]),
                "weight": float(selected["weight"]),
                "source": {
                    axis: detail[selected_key][axis] for axis in SOURCE_AXES
                },
            },
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
            "source_top": table.head(10).to_dict(orient="records"),
            "parity": parity,
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
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

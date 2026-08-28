"""Gate the proven H1 affine by reconstructed starter/reliever workload.

The fixed H1 affine was strongly positive on Public and on 2022/2024 OOF but
negative in late-2023.  v199 supplies a new row-local role proxy.  This audit
tests a small baseball-defined gate library with one fixed half-dose rollback
or amplification of the already deployed affine contribution.  Selection is
restricted to full-2022 and late-2023 before locked full-2024 evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import (
    AFFINE_ALPHA,
    AFFINE_CENTER,
    H1_ACTIVE_WEIGHT,
    H1_BASE_WEIGHT,
    _load_year_context,
    affine,
    metrics,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v199_recent_workload_reconstruction import (
    MAX_DENOMINATOR,
    infer_denominators,
    rate_columns,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V202_WORKLOAD_GATED_H1_AFFINE_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
DOSE = 0.5
WORKLOAD_COLUMNS = (
    "season",
    "game_month",
    "num_runners_on",
    "li",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev5_game_middle_rate",
)


def inferred_workloads(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[int, np.ndarray] = {}
    valid_all = np.ones(len(frame), dtype=bool)
    for horizon in (1, 5):
        success, middle = rate_columns(horizon)
        inferred = infer_denominators(
            frame[success], frame[middle],
            max_denominator=MAX_DENOMINATOR[horizon],
        )
        values[horizon] = pd.to_numeric(
            inferred["minimum_denominator"], errors="coerce"
        ).fillna(0).to_numpy(np.float64)
        valid_all &= inferred["rounded_rational_fit"].fillna(False).to_numpy(bool)
        valid_all &= values[horizon] > 0.0
    return values[1], values[5], valid_all


def role_gates(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fixed KBO-style appearance workload regimes; no labels are consulted."""

    n1, n5, valid = inferred_workloads(frame)
    return {
        "starter_prev1": valid & (n1 >= 40.0),
        "short_prev1": valid & (n1 <= 30.0),
        "heavy_prev5": valid & (n5 >= 200.0),
        "light_prev5": valid & (n5 <= 120.0),
        "starter_or_heavy": valid & ((n1 >= 40.0) | (n5 >= 200.0)),
        "short_and_light": valid & (n1 <= 30.0) & (n5 <= 120.0),
    }


def h1_affine_top_direction(
    identity_h1: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    identity = np.asarray(identity_h1, dtype=np.float64)
    adjusted = np.clip(
        AFFINE_CENTER + AFFINE_ALPHA * (identity - AFFINE_CENTER), 0.001, 0.999
    )
    champion_gate = (
        pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(frame["li"], errors="coerce").fillna(1.0).ge(1.5)
    ).to_numpy()
    weight = np.where(champion_gate, H1_ACTIVE_WEIGHT, H1_BASE_WEIGHT)
    return weight * (adjusted - identity)


def apply_gated_affine(
    parent: np.ndarray,
    top_direction: np.ndarray,
    gate: np.ndarray,
    exact: np.ndarray,
    domain3: np.ndarray,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.asarray(exact, dtype=bool)
        & np.asarray(domain3).astype(str).__eq__("R_CORE")
        & np.asarray(gate, dtype=bool)
    )
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active] + float(sign) * DOSE * np.asarray(top_direction)[active],
        0.001,
        0.999,
    )
    return output, active


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.5
        and result["worst_month_gain"] > -3.0
        and result["minimum_domain_gain"] >= 0.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "fixed_baseball_role_gates": True,
        "fixed_affine_and_half_dose": True,
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
    with np.load(h1_path, allow_pickle=False) as saved:
        raw_h1 = {
            year: saved[f"exact_h1_{year}"].astype(np.float64)
            for year in (2022, 2023, 2024)
        }
    identity_by_year = {
        year: raw_h1[year] + correction[year] for year in (2022, 2023, 2024)
    }
    identities = {
        "full_2022": identity_by_year[2022],
        "late_2023": identity_by_year[2023][late23],
        "full_2024": identity_by_year[2024],
    }
    gates = {axis: role_gates(frames[axis]) for axis in AXES}
    top_direction = {
        axis: h1_affine_top_direction(identities[axis], frames[axis])
        for axis in AXES
    }

    rows: list[dict[str, Any]] = []
    detail: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    active_masks: dict[str, dict[str, np.ndarray]] = {}
    for gate_name in gates["full_2022"]:
        for action, sign in (("rollback", -1.0), ("amplify", 1.0)):
            key = f"{gate_name}__{action}"
            detail[key], candidates[key], active_masks[key] = {}, {}, {}
            row: dict[str, Any] = {"gate": gate_name, "action": action}
            for axis in AXES:
                candidate, active = apply_gated_affine(
                    parents[axis], top_direction[axis], gates[axis][gate_name],
                    axes[axis]["exact_mask"], axes[axis]["domain3"], sign,
                )
                candidates[key][axis], active_masks[key][axis] = candidate, active
                score = metrics(axes[axis], parents[axis], candidate)
                detail[key][axis] = score
                row[f"{axis}_gain"] = score["gain"]
                row[f"{axis}_month_fraction"] = score["positive_month_fraction"]
                row[f"{axis}_worst_month"] = score["worst_month_gain"]
                row[f"{axis}_active_fraction"] = float(active.mean())
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
        ascending=[False, False, False], kind="stable",
    )
    table.to_csv(output_dir / "source_role_gate_screen.csv", index=False, encoding="utf-8-sig")
    passing = table.loc[table["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "candidate_count": int(len(table)),
            "screen": table.to_dict(orient="records"),
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected = passing.iloc[0]
        key = f"{selected['gate']}__{selected['action']}"
        candidate = candidates[key]["full_2024"]
        active = active_masks[key]["full_2024"]
        family = [
            candidates[f"{row.gate}__{row.action}"]["full_2024"]
            for row in passing.itertuples(index=False)
        ]
        family.append(parents["full_2024"].copy())
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate, active, family
        )
        locked = detail[key]["full_2024"]
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -3.0
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
            parent=parents["full_2024"], candidate=candidate,
            top_direction=top_direction["full_2024"], active=active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "candidate_count": int(len(table)),
            "source_passing_count": int(len(passing)),
            "selected": {
                "gate": str(selected["gate"]),
                "action": str(selected["action"]),
                "dose": DOSE,
                "source": {axis: detail[key][axis] for axis in SOURCE_AXES},
            },
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
            "screen": table.to_dict(orient="records"),
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

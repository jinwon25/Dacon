"""Route the frozen TrackMan-ASOF anchor prior to command-pressure counts.

The Public-improving 0.03 TrackMan-ASOF gate shrinks every R_ANCHOR row toward
an official as-of player prior, with only a 1.25/0.75 pressure multiplier.
Command is most directly exposed in three-ball and two-strike counts.  This
bounded audit reconstructs the frozen prior from the archived OOF gate and
tests whether suppressing low-pressure shrinkage improves temporal stability.

Policy selection uses late-2023 only.  Full/late-2024 are outer diagnostics;
the single source axis means passing point estimates still cannot alone
authorize packaging under evaluation v3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.core.oof_bank import (
    _assert_target,
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_parent_axis,
)
from src.archive.v85_lowrank_policy_replacement import V82_NAME, exact_v84_parent


PROTOCOL = "V90_TRACKMAN_ASOF_HIGH_PRESSURE_R_ANCHOR_GATE_V1"
POLICIES = (
    "high_only_x100",
    "high_only_x125",
    "high_only_x150",
    "high_plus_low_half_x100",
)


def pressure_mask(frame: pd.DataFrame) -> np.ndarray:
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1)
    return balls.eq(3).to_numpy() | strikes.eq(2).to_numpy()


def alternative_gate(
    current_gate: np.ndarray, high_pressure: np.ndarray, policy: str
) -> np.ndarray:
    if policy not in POLICIES:
        raise ValueError(f"unknown TrackMan pressure policy: {policy}")
    gate = np.asarray(current_gate, dtype=np.float64)
    high = np.asarray(high_pressure, dtype=bool)
    if gate.shape != high.shape or gate.ndim != 1:
        raise ValueError("TrackMan gate arrays are not aligned")
    if policy == "high_plus_low_half_x100":
        multiplier = np.where(high, 1.0, 0.5)
    else:
        scale = {
            "high_only_x100": 1.0,
            "high_only_x125": 1.25,
            "high_only_x150": 1.50,
        }[policy]
        multiplier = np.where(high, scale, 0.0)
    return np.clip(gate * multiplier, 0.0, 0.05)


def reconstruct_prior(
    pre_gate_parent: np.ndarray,
    current_parent: np.ndarray,
    current_gate: np.ndarray,
) -> np.ndarray:
    base = np.asarray(pre_gate_parent, dtype=np.float64)
    current = np.asarray(current_parent, dtype=np.float64)
    gate = np.asarray(current_gate, dtype=np.float64)
    if not (base.shape == current.shape == gate.shape):
        raise ValueError("TrackMan prior arrays are not aligned")
    prior = base.copy()
    active = gate > 0.0
    prior[active] = (current[active] - base[active] * (1.0 - gate[active])) / gate[active]
    reconstructed = base * (1.0 - gate) + prior * gate
    if float(np.max(np.abs(reconstructed - current))) > 2e-12:
        raise ValueError("frozen TrackMan prior reconstruction failed")
    return np.clip(prior, 0.0, 1.0)


def apply_gate_policy(
    pre_gate_parent: np.ndarray,
    current_parent: np.ndarray,
    current_gate: np.ndarray,
    high_pressure: np.ndarray,
    domain3: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = reconstruct_prior(pre_gate_parent, current_parent, current_gate)
    gate = alternative_gate(current_gate, high_pressure, policy)
    active = np.asarray(domain3).astype(str) == "R_ANCHOR"
    gate = np.where(active, gate, 0.0)
    candidate = np.asarray(current_parent, dtype=np.float64).copy()
    candidate[active] = (
        np.asarray(pre_gate_parent, dtype=np.float64)[active] * (1.0 - gate[active])
        + prior[active] * gate[active]
    )
    return np.clip(candidate, 0.001, 0.999), active, gate


def _load_gate_axis(
    directory: Path,
    filename: str,
    raw_rows: pd.DataFrame,
) -> dict[str, np.ndarray]:
    # The trusted local OOF archive stores ``domain3`` as an object-string
    # array; no executable objects are present.
    with np.load(directory / filename, allow_pickle=True) as saved:
        output = {name: saved[name] for name in saved.files}
    target = raw_rows["control_success"].to_numpy(np.float64)
    month = raw_rows["game_month"].to_numpy(np.int16)
    if not np.array_equal(output["target"].astype(np.float64), target):
        raise ValueError(f"TrackMan gate target/order mismatch: {filename}")
    if not np.array_equal(output["game_month"].astype(np.int16), month):
        raise ValueError(f"TrackMan gate month/order mismatch: {filename}")
    return {name: np.asarray(value) for name, value in output.items()}


def _metric(axis: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": axis["target"].astype(np.float64),
            "game_month": axis["game_month"].astype(np.int16),
            "domain3": axis["domain3"].astype(str),
        }
    )


def _compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "minimum_domain_gain": float(result["minimum_domain_gain"]),
        "applied_domain_gain": float(result["domain_gains"]["R_ANCHOR"]),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def run(
    project: Path,
    final_parent_dir: Path,
    gate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    final_parent_dir = final_parent_dir.resolve()
    gate_dir = gate_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    rows23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    rows24 = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    late23 = rows23.loc[rows23["game_month"].ge(8)].reset_index(drop=True)
    late24 = rows24.loc[rows24["game_month"].ge(8)].reset_index(drop=True)
    axes = {
        "selection_late_2023": (
            _load_gate_axis(gate_dir, "selection_late_2023.npz", late23),
            late23,
        ),
        "outer_full_2024": (
            _load_gate_axis(gate_dir, "outer_full_2024.npz", rows24),
            rows24,
        ),
        "replication_late_2024": (
            _load_gate_axis(gate_dir, "replication_late_2024.npz", late24),
            late24,
        ),
    }

    source_axis, source_rows = axes["selection_late_2023"]
    source_metric = _metric(source_axis)
    source_records = []
    source_details: dict[str, Any] = {}
    for policy in POLICIES:
        candidate, active, gate = apply_gate_policy(
            source_axis["eta15_parent"],
            source_axis["final_gate_parent"],
            source_axis["gate"],
            pressure_mask(source_rows),
            source_axis["domain3"],
            policy,
        )
        result = diagnostics(
            source_metric, source_axis["final_gate_parent"], candidate, active
        )
        result["gate_mean_active"] = float(gate[gate > 0.0].mean()) if np.any(gate > 0.0) else 0.0
        result["gate_active_rows"] = int(np.sum(gate > 0.0))
        source_details[policy] = result
        source_records.append({"policy": policy, **_compact(result)})
    source_table = pd.DataFrame(source_records)
    source_table["source_gate_passed"] = (
        source_table["gain"].gt(0.0)
        & source_table["positive_month_fraction"].ge(2.0 / 3.0)
        & source_table["worst_month_gain"].gt(-5.0)
        & source_table["minimum_domain_gain"].ge(0.0)
        & source_table["applied_domain_gain"].gt(0.0)
    )
    source_table["robust_score"] = source_table[
        ["gain", "worst_month_gain", "applied_domain_gain"]
    ].min(axis=1)
    source_table = source_table.sort_values(
        ["source_gate_passed", "robust_score", "gain"], ascending=False
    ).reset_index(drop=True)
    selected = str(
        (
            source_table.loc[source_table["source_gate_passed"]]
            if source_table["source_gate_passed"].any()
            else source_table
        ).iloc[0]["policy"]
    )
    source_passed = bool(
        source_table.set_index("policy").loc[selected, "source_gate_passed"]
    )
    source_table.to_csv(output_dir / "source_policy_metrics.csv", index=False)

    parent_axis = _load_parent_axis(final_parent_dir, "full_2024")
    common = _load_common_candidates(project, parent_axis, "full_2024")
    registry = _load_full24_diagnostic_candidates(project, parent_axis, common)
    v56_path = (
        project
        / "artifacts"
        / "v56_shared_horizon_fm_20260817_01"
        / "outer_full_2024.npz"
    )
    with np.load(v56_path, allow_pickle=True) as saved:
        _assert_target(parent_axis["target"], saved, "v56/full_2024")
        v56_candidate = saved["candidate"].astype(np.float64)
    v84, composition = exact_v84_parent(parent_axis, registry[V82_NAME], v56_candidate)

    full_axis, full_rows = axes["outer_full_2024"]
    anchor = full_axis["domain3"].astype(str) == "R_ANCHOR"
    parent_parity = float(
        np.max(np.abs(v84[anchor] - full_axis["final_gate_parent"][anchor]))
    )
    if parent_parity > 1e-12:
        raise ValueError(f"v84 R_ANCHOR parent parity failed: {parent_parity}")
    replacement, active24, alt_gate = apply_gate_policy(
        full_axis["eta15_parent"],
        full_axis["final_gate_parent"],
        full_axis["gate"],
        pressure_mask(full_rows),
        full_axis["domain3"],
        selected,
    )
    candidate24 = v84.copy()
    candidate24[active24] = replacement[active24]
    full_audit = diagnostics(_metric(full_axis), v84, candidate24, active24)
    late_mask = full_rows["game_month"].ge(8).to_numpy()
    late_axis, _ = axes["replication_late_2024"]
    if not np.array_equal(full_axis["target"][late_mask], late_axis["target"]):
        raise ValueError("full/late-2024 TrackMan target mismatch")
    late_audit = diagnostics(
        _metric(late_axis),
        v84[late_mask],
        candidate24[late_mask],
        active24[late_mask],
    )
    point_gates = {
        "single_pre2024_source_recipe_passed": source_passed,
        "full_gain_positive": float(full_audit["gain"]) > 0.0,
        "late_gain_positive": float(late_audit["gain"]) > 0.0,
        "full_month_fraction_at_least_075": float(full_audit["positive_month_fraction"]) >= 0.75,
        "late_month_fraction_at_least_075": float(late_audit["positive_month_fraction"]) >= 0.75,
        "both_worst_months_above_minus_5": min(
            float(full_audit["worst_month_gain"]), float(late_audit["worst_month_gain"])
        ) > -5.0,
        "both_minimum_domains_nonnegative": min(
            float(full_audit["minimum_domain_gain"]), float(late_audit["minimum_domain_gain"])
        ) >= 0.0,
    }
    np.savez_compressed(
        output_dir / "outer_full_2024.npz",
        target=parent_axis["target"],
        v84=v84,
        candidate=candidate24,
        active=active24,
        alternative_gate=alt_gate,
        current_gate=full_axis["gate"],
        domain3=full_axis["domain3"].astype(str),
        game_month=full_axis["game_month"].astype(np.int16),
    )
    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue / Public 1161.2020600422",
        "policies": list(POLICIES),
        "selected_policy": selected,
        "selection": "late-2023 only; single source axis",
        "source_gate_passed": source_passed,
        "source_policy_metrics": source_table.to_dict(orient="records"),
        "source_details": source_details,
        "composition_audit": composition,
        "v84_r_anchor_parent_parity_max_abs": parent_parity,
        "audits": {
            "outer_full_2024": full_audit,
            "replication_late_2024": late_audit,
        },
        "point_gates": {name: bool(value) for name, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_block": "evaluation v3 requires two pre-audit source axes; TrackMan linkage archive begins at origin 2023",
        "same_family_2024_labels_used_for_policy_selection": False,
        "public_score_used_for_policy_selection": False,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.final_parent_dir, args.gate_dir, args.output_dir)


if __name__ == "__main__":
    main()

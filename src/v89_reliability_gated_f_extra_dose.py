"""Reliability-gated extra dose for the successful v84 F shared FM.

v84 applies the frozen v56 shared FM at logit eta 0.10 to non-regular (F)
rows.  A globally larger eta is unsafe on 2022, while late-2023 has substantial
headroom.  This bounded experiment preserves eta 0.10 for every row and adds a
small dose only in rows whose official as-of player sample sizes support it::

    logit(p) = logit(p_v82) + (0.10 + delta * reliability) * correction

The reliability policy and delta are selected on 2022 and late-2023 rolling
origins.  Full/late-2024 are opened only after freezing that recipe.  Test-row
aggregates and order are never used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.v53_factorization_offset import TARGET, _expit, _logit, apply_offset, prepare_fields
from src.v56_shared_horizon_fm import fit_shared_offset
from src.core.oof_bank import (
    _assert_target,
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_parent_axis,
)
from src.v85_lowrank_policy_replacement import V82_NAME, exact_v84_parent
from src.v86_reliability_gated_r_fm import reliability_gate


PROTOCOL = "V89_RELIABILITY_GATED_EXTRA_F_SHARED_FM_DOSE_V1"
BASE_ETA = 0.10
RISK = "source_domain_equal"
ROUTE = "F"
RECIPES = (
    ("uniform", 0.025),
    ("pitcher_eb200", 0.025),
    ("pair_geomean_eb200", 0.025),
    ("pitcher_eb200", 0.050),
    ("pair_geomean_eb200", 0.050),
)


def extra_dose_gate(frame: pd.DataFrame, policy: str) -> np.ndarray:
    if policy == "uniform":
        return np.ones(len(frame), dtype=np.float64)
    return reliability_gate(frame, policy)


def apply_extra_dose(
    frame: pd.DataFrame,
    base: np.ndarray,
    correction: np.ndarray,
    policy: str,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add a row-local dose above an already applied eta-0.10 base."""

    prepared = prepare_fields(frame).reset_index(drop=True)
    base = np.asarray(base, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if not (len(prepared) == len(base) == len(correction)):
        raise ValueError("extra-dose arrays are not aligned")
    if float(delta) <= 0.0:
        raise ValueError("extra dose must be positive")
    gate = extra_dose_gate(prepared, policy)
    active = prepared["domain3"].astype(str).eq(ROUTE).to_numpy()
    candidate = base.copy()
    candidate[active] = _expit(
        _logit(base[active]) + float(delta) * gate[active] * correction[active]
    )
    return np.clip(candidate, 0.001, 0.999), active, gate


def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_fields(frame)
    return pd.DataFrame(
        {
            "target": prepared[TARGET].to_numpy(np.float64),
            "game_month": prepared["game_month"].to_numpy(np.int16),
            "domain3": prepared["domain3"].astype(str).to_numpy(),
        }
    )


def _compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "minimum_domain_gain": float(result["minimum_domain_gain"]),
        "applied_domain_gain": float(result["domain_gains"][ROUTE]),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def evaluate_recipes(
    frames: dict[str, pd.DataFrame],
    bases: dict[str, np.ndarray],
    corrections: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for policy, delta in RECIPES:
        name = f"{policy}_d{delta:.3f}"
        details[name] = {}
        for axis in ("full_2022", "late_2023"):
            candidate, active, gate = apply_extra_dose(
                frames[axis], bases[axis], corrections[axis], policy, delta
            )
            result = diagnostics(
                _metric_frame(frames[axis]), bases[axis], candidate, active
            )
            result["extra_gate_mean_active"] = float(gate[active].mean())
            details[name][axis] = result
            rows.append(
                {
                    "recipe": name,
                    "policy": policy,
                    "delta": float(delta),
                    "axis": axis,
                    **_compact(result),
                    "extra_gate_mean_active": result["extra_gate_mean_active"],
                }
            )
    return pd.DataFrame(rows), details


def select_recipe(table: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    records = []
    for policy, delta in RECIPES:
        name = f"{policy}_d{delta:.3f}"
        local = table.loc[table["recipe"].eq(name)].set_index("axis")
        passed = bool(
            set(local.index) == {"full_2022", "late_2023"}
            and (local["gain"] > 0.0).all()
            and (local["positive_month_fraction"] >= 0.75).all()
            and (local["worst_month_gain"] > -5.0).all()
            and (local["minimum_domain_gain"] >= 0.0).all()
            and (local["applied_domain_gain"] > 0.0).all()
        )
        robust = float(
            min(
                local["gain"].min(),
                local["worst_month_gain"].min(),
                local["applied_domain_gain"].min(),
            )
        )
        records.append(
            {
                "recipe": name,
                "policy": policy,
                "delta": float(delta),
                "source_gate_passed": passed,
                "robust_score": robust,
                "minimum_gain": float(local["gain"].min()),
                "minimum_month_fraction": float(
                    local["positive_month_fraction"].min()
                ),
                "minimum_worst_month_gain": float(local["worst_month_gain"].min()),
                "minimum_applied_domain_gain": float(
                    local["applied_domain_gain"].min()
                ),
            }
        )
    ranking = pd.DataFrame(records).sort_values(
        ["source_gate_passed", "robust_score", "minimum_gain"], ascending=False
    )
    passing = ranking.loc[ranking["source_gate_passed"]]
    selected = (passing if len(passing) else ranking).iloc[0]
    return {
        "recipe": str(selected["recipe"]),
        "policy": str(selected["policy"]),
        "delta": float(selected["delta"]),
        "source_gate_passed": bool(selected["source_gate_passed"]),
    }, ranking.reset_index(drop=True)


def _wave_parent(project: Path, year: int, target: np.ndarray) -> np.ndarray:
    path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["target"].astype(float), target):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        return saved["incumbent"].astype(np.float64)


def run(
    project: Path,
    final_parent_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    final_parent_dir = final_parent_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    rows = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2020, 2021, 2022)
    }
    axes = _cached_v25_axes(project, raw)
    frame23 = axes["selection_late_2023"].reset_index(drop=True)
    frame24 = axes["outer_full_2024"].reset_index(drop=True)
    parent20 = _wave_parent(
        project, 2020, rows[2020][TARGET].to_numpy(np.float64)
    )
    parent21 = _wave_parent(
        project, 2021, rows[2021][TARGET].to_numpy(np.float64)
    )
    meta22 = _metadata(project, 2022)
    print("[v89] shared source-domain-equal 2020+2021 -> 2022", flush=True)
    correction22, fit22 = fit_shared_offset(
        [rows[2020], rows[2021]],
        [parent20, parent21],
        [2020, 2021],
        rows[2022],
        risk=RISK,
    )
    print("[v89] shared source-domain-equal 2021+2022 -> late-2023", flush=True)
    correction23, fit23 = fit_shared_offset(
        [rows[2021], rows[2022]],
        [parent21, meta22["parent"]],
        [2021, 2022],
        frame23,
        risk=RISK,
    )
    parent23 = v27_parent(frame23)
    base22, _ = apply_offset(
        prepare_fields(rows[2022]),
        meta22["parent"],
        correction22,
        ROUTE,
        BASE_ETA,
    )
    base23, _ = apply_offset(
        prepare_fields(frame23), parent23, correction23, ROUTE, BASE_ETA
    )
    source_table, source_details = evaluate_recipes(
        {"full_2022": rows[2022], "late_2023": frame23},
        {"full_2022": base22, "late_2023": base23},
        {"full_2022": correction22, "late_2023": correction23},
    )
    selected, ranking = select_recipe(source_table)
    source_table.to_csv(output_dir / "source_recipe_metrics.csv", index=False)
    ranking.to_csv(output_dir / "source_recipe_ranking.csv", index=False)
    np.savez_compressed(
        output_dir / "source_corrections.npz",
        correction22=correction22,
        correction23=correction23,
    )

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
        correction24 = saved["correction"].astype(np.float64)
        v56_candidate = saved["candidate"].astype(np.float64)
    v84, composition = exact_v84_parent(
        parent_axis, registry[V82_NAME], v56_candidate
    )
    if not np.array_equal(parent_axis["target"], frame24[TARGET].to_numpy(float)):
        raise ValueError("v84/full-2024 target mismatch")
    candidate24, active24, extra_gate = apply_extra_dose(
        frame24,
        v84,
        correction24,
        selected["policy"],
        selected["delta"],
    )
    metric24 = _metric_frame(frame24)
    full_audit = diagnostics(metric24, v84, candidate24, active24)
    late_mask = metric24["game_month"].ge(8).to_numpy()
    candidate_late, active_late, _ = apply_extra_dose(
        frame24.loc[late_mask].reset_index(drop=True),
        v84[late_mask],
        correction24[late_mask],
        selected["policy"],
        selected["delta"],
    )
    late_audit = diagnostics(
        metric24.loc[late_mask].reset_index(drop=True),
        v84[late_mask],
        candidate_late,
        active_late,
    )
    point_gates = {
        "pre2024_source_recipe_passed": selected["source_gate_passed"],
        "full_gain_positive": float(full_audit["gain"]) > 0.0,
        "late_gain_positive": float(late_audit["gain"]) > 0.0,
        "full_month_fraction_at_least_075": float(
            full_audit["positive_month_fraction"]
        )
        >= 0.75,
        "late_month_fraction_at_least_075": float(
            late_audit["positive_month_fraction"]
        )
        >= 0.75,
        "both_worst_months_above_minus_5": min(
            float(full_audit["worst_month_gain"]),
            float(late_audit["worst_month_gain"]),
        )
        > -5.0,
        "both_minimum_domains_nonnegative": min(
            float(full_audit["minimum_domain_gain"]),
            float(late_audit["minimum_domain_gain"]),
        )
        >= 0.0,
    }
    np.savez_compressed(
        output_dir / "outer_full_2024.npz",
        target=parent_axis["target"],
        v84=v84,
        correction=correction24,
        extra_gate=extra_gate,
        candidate=candidate24,
        active=active24,
        domain3=metric24["domain3"].to_numpy(str),
        game_month=metric24["game_month"].to_numpy(np.int16),
    )
    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue / Public 1161.2020600422",
        "configuration": {
            "base_eta": BASE_ETA,
            "risk": RISK,
            "route": ROUTE,
            "recipes": [
                {"policy": policy, "delta": delta} for policy, delta in RECIPES
            ],
            "selected": selected,
        },
        "selection": "exact policy/delta on full-2022 and late-2023 only",
        "source_recipe_ranking": ranking.to_dict(orient="records"),
        "source_details": source_details,
        "fit_audits": {"to_2022": fit22, "to_late2023": fit23},
        "composition_audit": composition,
        "extra_gate_2024_f": {
            "mean": float(extra_gate[active24].mean()),
            "p10": float(np.quantile(extra_gate[active24], 0.10)),
            "p90": float(np.quantile(extra_gate[active24], 0.90)),
        },
        "audits": {
            "outer_full_2024": full_audit,
            "replication_late_2024": late_audit,
        },
        "point_gates": {name: bool(value) for name, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_note": (
            "point gates authorize dependence-aware bootstrap only; packaging "
            "also requires all v3 uncertainty gates"
        ),
        "same_family_2024_labels_used_for_recipe_selection": False,
        "public_score_used_for_policy_or_delta": False,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.final_parent_dir, args.output_dir)


if __name__ == "__main__":
    main()

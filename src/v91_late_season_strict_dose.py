"""Late-season extra dose for the Public-positive strict R_CORE blend.

The v82 strict blend uses probability weight 0.10 on every R_CORE row.  Its
archived unconstrained optimum is 0.087 on full-2024 but 0.161 on late-2024.
This bounded experiment keeps the champion unchanged before August and tests
four total weights on row-local August-or-later R_CORE rows.  Weight selection
uses full-2022 and late-2023 only; 2024 is an outer audit above exact v84.
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
from src.v57_public_strict_blend import _load_strict, blend_candidate
from src.core.oof_bank import (
    _assert_target,
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_parent_axis,
)
from src.v85_lowrank_policy_replacement import V82_NAME, exact_v84_parent


PROTOCOL = "V91_LATE_SEASON_EXTRA_STRICT_R_CORE_DOSE_V1"
BASE_WEIGHT = 0.10
TOTAL_WEIGHTS = (0.125, 0.150, 0.160, 0.200)


def apply_late_strict_weight(
    frame: pd.DataFrame,
    current: np.ndarray,
    pre_strict_parent: np.ndarray,
    strict: np.ndarray,
    total_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only late R_CORE rows with a larger probability blend."""

    current = np.asarray(current, dtype=np.float64)
    parent = np.asarray(pre_strict_parent, dtype=np.float64)
    challenger = np.asarray(strict, dtype=np.float64)
    if not (len(frame) == len(current) == len(parent) == len(challenger)):
        raise ValueError("late strict arrays are not aligned")
    if not BASE_WEIGHT < float(total_weight) <= 1.0:
        raise ValueError("total strict weight must exceed the 0.10 parent")
    domain = frame["domain3"].astype(str).to_numpy()
    month = pd.to_numeric(frame["game_month"], errors="coerce").fillna(-1).to_numpy()
    active = (domain == "R_CORE") & (month >= 8)
    candidate = current.copy()
    candidate[active] = np.clip(
        parent[active]
        + float(total_weight) * (challenger[active] - parent[active]),
        0.001,
        0.999,
    )
    return candidate, active


def _frame(target: np.ndarray, month: np.ndarray, domain: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "game_month": np.asarray(month, dtype=np.int16),
            "domain3": np.asarray(domain).astype(str),
        }
    )


def _compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "minimum_domain_gain": float(result["minimum_domain_gain"]),
        "applied_domain_gain": float(result["domain_gains"]["R_CORE"]),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def run(
    project: Path,
    external_root: Path,
    final_parent_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    external_root = external_root.resolve()
    final_parent_dir = final_parent_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    strict, strict_provenance = _load_strict(external_root, raw)
    axes = _cached_v25_axes(project, raw)
    rows22 = raw.loc[raw["season"].eq(2022)].reset_index(drop=True)
    rows23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    late23_mask = rows23["game_month"].ge(8).to_numpy()
    frame23 = axes["selection_late_2023"].reset_index(drop=True)
    meta22 = _metadata(project, 2022)
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])
    parent23 = v27_parent(frame23)
    base22, _ = blend_candidate(
        frame22,
        meta22["parent"],
        strict[2022],
        mode="probability",
        route="R_CORE",
        weight=BASE_WEIGHT,
    )
    base23, _ = blend_candidate(
        frame23,
        parent23,
        strict[2023][late23_mask],
        mode="probability",
        route="R_CORE",
        weight=BASE_WEIGHT,
    )
    source_inputs = {
        "full_2022": (
            frame22,
            base22,
            meta22["parent"],
            strict[2022],
        ),
        "late_2023": (
            frame23,
            base23,
            parent23,
            strict[2023][late23_mask],
        ),
    }
    records = []
    source_details: dict[str, Any] = {}
    for weight in TOTAL_WEIGHTS:
        name = f"w{weight:.3f}"
        source_details[name] = {}
        for axis, (frame, current, parent, challenger) in source_inputs.items():
            candidate, active = apply_late_strict_weight(
                frame, current, parent, challenger, weight
            )
            result = diagnostics(frame, current, candidate, active)
            source_details[name][axis] = result
            records.append({"weight": weight, "axis": axis, **_compact(result)})
    source_table = pd.DataFrame(records)
    ranking_rows = []
    for weight in TOTAL_WEIGHTS:
        local = source_table.loc[source_table["weight"].eq(weight)].set_index("axis")
        passed = bool(
            (local["gain"] > 0.0).all()
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
        ranking_rows.append(
            {
                "weight": weight,
                "source_gate_passed": passed,
                "robust_score": robust,
                "minimum_gain": float(local["gain"].min()),
                "minimum_month_fraction": float(local["positive_month_fraction"].min()),
                "minimum_worst_month_gain": float(local["worst_month_gain"].min()),
                "minimum_applied_domain_gain": float(local["applied_domain_gain"].min()),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_gate_passed", "robust_score", "minimum_gain"], ascending=False
    ).reset_index(drop=True)
    passing = ranking.loc[ranking["source_gate_passed"]]
    selected_weight = float((passing if len(passing) else ranking).iloc[0]["weight"])
    source_passed = bool(
        ranking.set_index("weight").loc[selected_weight, "source_gate_passed"]
    )
    source_table.to_csv(output_dir / "source_weight_metrics.csv", index=False)
    ranking.to_csv(output_dir / "source_weight_ranking.csv", index=False)

    parent_axis = _load_parent_axis(final_parent_dir, "full_2024")
    common = _load_common_candidates(project, parent_axis, "full_2024")
    registry = _load_full24_diagnostic_candidates(project, parent_axis, common)
    v82 = registry[V82_NAME]
    v56_path = (
        project
        / "artifacts"
        / "v56_shared_horizon_fm_20260817_01"
        / "outer_full_2024.npz"
    )
    with np.load(v56_path, allow_pickle=True) as saved:
        _assert_target(parent_axis["target"], saved, "v56/full_2024")
        v56_candidate = saved["candidate"].astype(np.float64)
    v84, composition = exact_v84_parent(parent_axis, v82, v56_candidate)
    frame24 = _frame(parent_axis["target"], parent_axis["month"], parent_axis["domain3"])
    base_v27 = parent_axis["parent"]
    candidate24, active24 = apply_late_strict_weight(
        frame24, v84, base_v27, strict[2024], selected_weight
    )
    full_audit = diagnostics(frame24, v84, candidate24, active24)
    late_mask = frame24["game_month"].ge(8).to_numpy()
    late_audit = diagnostics(
        frame24.loc[late_mask].reset_index(drop=True),
        v84[late_mask],
        candidate24[late_mask],
        active24[late_mask],
    )
    point_gates = {
        "pre2024_source_recipe_passed": source_passed,
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
        v27=base_v27,
        strict=strict[2024],
        candidate=candidate24,
        active=active24,
        domain3=parent_axis["domain3"].astype(str),
        game_month=parent_axis["month"].astype(np.int16),
    )
    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue / Public 1161.2020600422",
        "base_weight": BASE_WEIGHT,
        "total_weights": list(TOTAL_WEIGHTS),
        "selected_weight": selected_weight,
        "route": "R_CORE and game_month >= 8",
        "selection": "exact total weight on full-2022 and late-2023 only",
        "source_gate_passed": source_passed,
        "source_weight_ranking": ranking.to_dict(orient="records"),
        "source_details": source_details,
        "strict_provenance": strict_provenance,
        "composition_audit": composition,
        "audits": {
            "outer_full_2024": full_audit,
            "replication_late_2024": late_audit,
        },
        "point_gates": {name: bool(value) for name, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_note": "point gates authorize dependence-aware bootstrap only",
        "same_family_2024_labels_used_for_weight_selection": False,
        "public_score_used_for_weight_selection": False,
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
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.external_root, args.final_parent_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Reliability-gated R_CORE FM correction above the exact v84 champion.

The v53 interaction FM produced useful R_CORE headroom but its month-to-month
variance was too large.  This experiment tests one baseball-motivated cause:
player interaction embeddings are least trustworthy when the current row says
that either participant has little prior official history.  A frozen
empirical-Bayes reliability therefore damps the already frozen rank-16 FM
correction using only current-row ``asof_*_n`` fields::

    r(n) = n / (n + 200)
    correction_gated = correction * gate(r_pitcher, r_batter)

The route (R_CORE), FM rank (16), damping (0.10), prior strength (200), and
three small gate family are fixed before the 2024 audit.  Gate selection uses
2022 and late-2023 rolling origins only.  The 2024 comparison is against the
exact OOF analogue of the Public-1161 v84 parent.  No test rows are read.
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
from src.champion.v53_factorization_offset import (
    TARGET,
    _expit,
    _frame,
    _logit,
    fit_predict_offset,
    prepare_fields,
)
from src.archive.v57_public_strict_blend import _load_strict, blend_candidate
from src.core.oof_bank import (
    _assert_target,
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_parent_axis,
)
from src.archive.v85_lowrank_policy_replacement import V82_NAME, exact_v84_parent


PROTOCOL = "V86_RELIABILITY_GATED_R_CORE_FM_ABOVE_V84_V1"
RANK = 16
ROUTE = "R_CORE"
ETA = 0.10
PRIOR_STRENGTH = 200.0
GATE_POLICIES = (
    "none",
    "pitcher_eb200",
    "pair_geomean_eb200",
    "pair_min_eb200",
)


def reliability_gate(
    frame: pd.DataFrame,
    policy: str,
    *,
    prior_strength: float = PRIOR_STRENGTH,
) -> np.ndarray:
    """Return a bounded row-local reliability multiplier."""

    if policy not in GATE_POLICIES:
        raise ValueError(f"unknown reliability policy: {policy}")
    if float(prior_strength) <= 0.0:
        raise ValueError("prior_strength must be positive")
    if policy == "none":
        return np.ones(len(frame), dtype=np.float64)
    missing = sorted({"asof_pitcher_n", "asof_batter_n"} - set(frame.columns))
    if missing:
        raise ValueError(f"missing reliability columns: {missing}")
    pitcher_n = np.nan_to_num(
        pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    batter_n = np.nan_to_num(
        pd.to_numeric(frame["asof_batter_n"], errors="coerce").to_numpy(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    pitcher_n = np.maximum(pitcher_n, 0.0)
    batter_n = np.maximum(batter_n, 0.0)
    pitcher = pitcher_n / (pitcher_n + float(prior_strength))
    batter = batter_n / (batter_n + float(prior_strength))
    if policy == "pitcher_eb200":
        gate = pitcher
    elif policy == "pair_geomean_eb200":
        gate = np.sqrt(pitcher * batter)
    else:
        gate = np.minimum(pitcher, batter)
    return np.clip(gate, 0.0, 1.0).astype(np.float64)


def apply_reliability_offset(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
    policy: str,
    *,
    eta: float = ETA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen FM only on R_CORE after row-local EB damping."""

    prepared = prepare_fields(frame).reset_index(drop=True)
    parent = np.asarray(parent, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if not (len(prepared) == len(parent) == len(correction)):
        raise ValueError("reliability candidate length mismatch")
    if not (np.isfinite(parent).all() and np.isfinite(correction).all()):
        raise ValueError("reliability candidate contains non-finite values")
    gate = reliability_gate(prepared, policy)
    active = prepared["domain3"].astype(str).eq(ROUTE).to_numpy()
    candidate = parent.copy()
    candidate[active] = _expit(
        _logit(parent[active]) + float(eta) * correction[active] * gate[active]
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


def source_policy_table(
    frames: dict[str, pd.DataFrame],
    parents: dict[str, np.ndarray],
    corrections: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    """Score the fixed gate family on both pre-audit temporal origins."""

    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in GATE_POLICIES:
        details[policy] = {}
        for axis in ("full_2022", "late_2023"):
            metric_frame = _metric_frame(frames[axis])
            candidate, active, gate = apply_reliability_offset(
                frames[axis], parents[axis], corrections[axis], policy
            )
            result = diagnostics(metric_frame, parents[axis], candidate, active)
            result["gate_mean"] = float(gate[active].mean())
            result["gate_p10"] = float(np.quantile(gate[active], 0.10))
            result["gate_p90"] = float(np.quantile(gate[active], 0.90))
            details[policy][axis] = result
            rows.append(
                {
                    "policy": policy,
                    "axis": axis,
                    **_compact(result),
                    "gate_mean": result["gate_mean"],
                    "gate_p10": result["gate_p10"],
                    "gate_p90": result["gate_p90"],
                }
            )
    return pd.DataFrame(rows), details


def select_source_policy(table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select only among policies passing both frozen source-axis gates."""

    records = []
    for policy in GATE_POLICIES:
        local = table.loc[table["policy"].eq(policy)].set_index("axis")
        if set(local.index) != {"full_2022", "late_2023"}:
            raise ValueError(f"incomplete source policy rows: {policy}")
        source_pass = bool(
            (local["gain"] > 0.0).all()
            and (local["positive_month_fraction"] >= 0.75).all()
            and (local["worst_month_gain"] > -5.0).all()
            and (local["minimum_domain_gain"] >= 0.0).all()
            and (local["applied_domain_gain"] > 0.0).all()
        )
        robust_score = float(
            min(
                local["gain"].min(),
                local["worst_month_gain"].min(),
                local["applied_domain_gain"].min(),
            )
        )
        records.append(
            {
                "policy": policy,
                "source_gate_passed": source_pass,
                "robust_score": robust_score,
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
        ["source_gate_passed", "robust_score", "minimum_gain"],
        ascending=False,
    )
    passing = ranking.loc[ranking["source_gate_passed"]]
    chosen = str((passing if len(passing) else ranking).iloc[0]["policy"])
    return chosen, ranking.reset_index(drop=True)


def _strict_parent(
    frame: pd.DataFrame, parent: np.ndarray, strict: np.ndarray
) -> np.ndarray:
    candidate, _ = blend_candidate(
        _metric_frame(frame),
        parent,
        strict,
        mode="probability",
        route=ROUTE,
        weight=0.10,
    )
    return candidate


def run(
    project: Path,
    component_root: Path,
    final_parent_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    component_root = component_root.resolve()
    final_parent_dir = final_parent_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    rows22 = raw.loc[raw["season"].eq(2022)].reset_index(drop=True)
    rows23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    rows21 = raw.loc[raw["season"].eq(2021)].reset_index(drop=True)
    axes = _cached_v25_axes(project, raw)
    frame23 = axes["selection_late_2023"].reset_index(drop=True)
    frame24 = axes["outer_full_2024"].reset_index(drop=True)

    strict, strict_provenance = _load_strict(component_root, raw)
    target21_path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / "wave0_incumbent_validate_2021.npz"
    )
    with np.load(target21_path, allow_pickle=False) as saved:
        if not np.array_equal(
            saved["target"].astype(np.float64), rows21[TARGET].to_numpy(np.float64)
        ):
            raise ValueError("2021 source target/order mismatch")
        parent21 = saved["incumbent"].astype(np.float64)
    meta22 = _metadata(project, 2022)
    if not np.array_equal(meta22["target"], rows22[TARGET].to_numpy(np.float64)):
        raise ValueError("2022 target/order mismatch")
    late23_mask = rows23["game_month"].ge(8).to_numpy()
    if not np.array_equal(
        frame23[TARGET].to_numpy(np.float64),
        rows23.loc[late23_mask, TARGET].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 target/order mismatch")

    print("[v86] fit rank16 2021 -> 2022", flush=True)
    correction22, fit22 = fit_predict_offset(
        rows21, parent21, rows22, rank=RANK
    )
    print("[v86] fit rank16 2022 -> late-2023", flush=True)
    correction23, fit23 = fit_predict_offset(
        rows22, meta22["parent"], frame23, rank=RANK
    )
    source_frames = {"full_2022": rows22, "late_2023": frame23}
    source_parents = {
        "full_2022": _strict_parent(rows22, meta22["parent"], strict[2022]),
        "late_2023": _strict_parent(
            frame23, v27_parent(frame23), strict[2023][late23_mask]
        ),
    }
    source_corrections = {
        "full_2022": correction22,
        "late_2023": correction23,
    }
    source_table, source_details = source_policy_table(
        source_frames, source_parents, source_corrections
    )
    selected_policy, source_ranking = select_source_policy(source_table)
    source_passed = bool(
        source_ranking.set_index("policy").loc[selected_policy, "source_gate_passed"]
    )
    source_table.to_csv(output_dir / "source_policy_metrics.csv", index=False)
    source_ranking.to_csv(output_dir / "source_policy_ranking.csv", index=False)

    # Compose the exact v84 parent: v82 on R domains, v56 on F.
    parent_axis = _load_parent_axis(final_parent_dir, "full_2024")
    common = _load_common_candidates(project, parent_axis, "full_2024")
    registry = _load_full24_diagnostic_candidates(project, parent_axis, common)
    if V82_NAME not in registry:
        raise ValueError("v82 exact OOF candidate is missing")
    v56_path = (
        project
        / "artifacts"
        / "v56_shared_horizon_fm_20260817_01"
        / "outer_full_2024.npz"
    )
    with np.load(v56_path, allow_pickle=True) as saved:
        _assert_target(parent_axis["target"], saved, "v56/full_2024")
        v56_candidate = saved["candidate"].astype(np.float64)
    v84_full, composition = exact_v84_parent(
        parent_axis, registry[V82_NAME], v56_candidate
    )
    if not np.array_equal(
        frame24[TARGET].to_numpy(np.float64), parent_axis["target"]
    ):
        raise ValueError("v84/full-2024 target mismatch")

    v53_path = (
        project
        / "artifacts"
        / "v53_factorization_offset_20260817_01"
        / "outer_full_2024.npz"
    )
    with np.load(v53_path, allow_pickle=False) as saved:
        if not np.array_equal(
            saved["target"].astype(np.float64), parent_axis["target"]
        ):
            raise ValueError("v53/full-2024 target mismatch")
        correction24 = saved["correction"].astype(np.float64)

    metric24 = _metric_frame(frame24)
    candidate24, active24, gate24 = apply_reliability_offset(
        frame24, v84_full, correction24, selected_policy
    )
    full_audit = diagnostics(metric24, v84_full, candidate24, active24)
    late_mask = metric24["game_month"].ge(8).to_numpy()
    late_frame = frame24.loc[late_mask].reset_index(drop=True)
    late_metric = metric24.loc[late_mask].reset_index(drop=True)
    candidate_late, active_late, gate_late = apply_reliability_offset(
        late_frame,
        v84_full[late_mask],
        correction24[late_mask],
        selected_policy,
    )
    late_audit = diagnostics(
        late_metric, v84_full[late_mask], candidate_late, active_late
    )
    point_gates = {
        "pre2024_source_recipe_passed": source_passed,
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
        v84=v84_full,
        correction=correction24,
        gate=gate24,
        candidate=candidate24,
        active=active24,
        domain3=metric24["domain3"].to_numpy(str),
        game_month=metric24["game_month"].to_numpy(np.int16),
    )
    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue / Public 1161.2020600422",
        "configuration": {
            "fm_rank": RANK,
            "route": ROUTE,
            "eta": ETA,
            "prior_strength": PRIOR_STRENGTH,
            "gate_policies": list(GATE_POLICIES),
            "selected_policy": selected_policy,
        },
        "selection": "fixed gate policy on full-2022 and late-2023 only",
        "source_gate_passed": source_passed,
        "source_policy_ranking": source_ranking.to_dict(orient="records"),
        "source_details": source_details,
        "fit_audits": {"2021_to_2022": fit22, "2022_to_late2023": fit23},
        "strict_provenance": strict_provenance,
        "composition_audit": composition,
        "audits": {
            "outer_full_2024": full_audit,
            "replication_late_2024": late_audit,
        },
        "gate_distribution_2024_r_core": {
            "mean": float(gate24[active24].mean()),
            "p10": float(np.quantile(gate24[active24], 0.10)),
            "p90": float(np.quantile(gate24[active24], 0.90)),
            "late_mean": float(gate_late[active_late].mean()),
        },
        "point_gates": {name: bool(value) for name, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_note": (
            "point gates authorize dependence-aware bootstrap only; packaging "
            "also requires all v3 uncertainty gates"
        ),
        "same_family_2024_labels_used_for_gate_selection": False,
        "family_audit_note": (
            "FM family has prior 2024 exposure; the reliability mechanism is "
            "new but remains development-contaminated until bootstrap passes"
        ),
        "public_score_used_for_policy_or_weight": False,
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
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.component_root, args.final_parent_dir, args.output_dir)


if __name__ == "__main__":
    main()

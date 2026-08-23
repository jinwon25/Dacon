"""Cross-season consensus FM correction above the exact v84 champion.

Player matchup effects that reverse between source seasons are unlikely to be
stable command signal.  This experiment fits the frozen v53 rank-16 FM on two
source periods independently and combines their row-local corrections only by
a small, preregistered stability family.  Selection uses full-2022 and
late-2023 only; full/late-2024 remain outer audits against exact v84.

No prediction-row aggregate, ordering feature, or test data is used.
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
from src.champion.v53_factorization_offset import TARGET, fit_predict_offset
from src.archive.v57_public_strict_blend import _load_strict
from src.core.oof_bank import (
    _assert_target,
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_parent_axis,
)
from src.archive.v85_lowrank_policy_replacement import V82_NAME, exact_v84_parent
from src.archive.v86_reliability_gated_r_fm import (
    ETA,
    RANK,
    ROUTE,
    _metric_frame,
    _strict_parent,
    apply_reliability_offset,
)


PROTOCOL = "V87_CROSS_SEASON_CONSENSUS_R_CORE_FM_ABOVE_V84_V1"
POLICIES = (
    "source_mean",
    "sign_mean",
    "sign_min",
    "disagreement_quarter",
)


def combine_corrections(
    older: np.ndarray, recent: np.ndarray, policy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two independent source corrections without audit statistics."""

    if policy not in POLICIES:
        raise ValueError(f"unknown consensus policy: {policy}")
    older = np.asarray(older, dtype=np.float64)
    recent = np.asarray(recent, dtype=np.float64)
    if older.shape != recent.shape or older.ndim != 1:
        raise ValueError("source correction shape mismatch")
    if not (np.isfinite(older).all() and np.isfinite(recent).all()):
        raise ValueError("source correction contains non-finite values")
    agree = (np.signbit(older) == np.signbit(recent)) & (older != 0.0) & (recent != 0.0)
    mean = 0.5 * (older + recent)
    if policy == "source_mean":
        output = mean
    elif policy == "sign_mean":
        output = np.where(agree, mean, 0.0)
    elif policy == "sign_min":
        output = np.where(
            agree,
            np.sign(older) * np.minimum(np.abs(older), np.abs(recent)),
            0.0,
        )
    else:
        output = np.where(agree, mean, 0.25 * mean)
    return np.clip(output, -0.25, 0.25), agree


def _compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "minimum_domain_gain": float(result["minimum_domain_gain"]),
        "applied_domain_gain": float(result["domain_gains"][ROUTE]),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def evaluate_policy_family(
    frames: dict[str, pd.DataFrame],
    parents: dict[str, np.ndarray],
    correction_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in POLICIES:
        details[policy] = {}
        for axis in ("full_2022", "late_2023"):
            correction, agree = combine_corrections(*correction_pairs[axis], policy)
            candidate, active, _ = apply_reliability_offset(
                frames[axis], parents[axis], correction, "none", eta=ETA
            )
            result = diagnostics(
                _metric_frame(frames[axis]), parents[axis], candidate, active
            )
            result["sign_agreement_fraction_active"] = float(agree[active].mean())
            details[policy][axis] = result
            rows.append(
                {
                    "policy": policy,
                    "axis": axis,
                    **_compact(result),
                    "sign_agreement_fraction_active": result[
                        "sign_agreement_fraction_active"
                    ],
                }
            )
    return pd.DataFrame(rows), details


def select_policy(table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    records = []
    for policy in POLICIES:
        local = table.loc[table["policy"].eq(policy)].set_index("axis")
        if set(local.index) != {"full_2022", "late_2023"}:
            raise ValueError(f"incomplete policy table: {policy}")
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
        records.append(
            {
                "policy": policy,
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
    chosen = str((passing if len(passing) else ranking).iloc[0]["policy"])
    return chosen, ranking.reset_index(drop=True)


def _load_wave_parent(project: Path, year: int, target: np.ndarray) -> np.ndarray:
    path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["target"].astype(np.float64), target):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        return saved["incumbent"].astype(np.float64)


def _fit(
    source: pd.DataFrame,
    source_parent: np.ndarray,
    audit: pd.DataFrame,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    print(f"[v87] fit {label}", flush=True)
    return fit_predict_offset(source, source_parent, audit, rank=RANK)


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
    rows = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2020, 2021, 2022, 2023)
    }
    axes = _cached_v25_axes(project, raw)
    frame23 = axes["selection_late_2023"].reset_index(drop=True)
    frame24 = axes["outer_full_2024"].reset_index(drop=True)
    late23_mask = rows[2023]["game_month"].ge(8).to_numpy()
    if not np.array_equal(
        frame23[TARGET].to_numpy(np.float64),
        rows[2023].loc[late23_mask, TARGET].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 target/order mismatch")

    parent20 = _load_wave_parent(
        project, 2020, rows[2020][TARGET].to_numpy(np.float64)
    )
    parent21 = _load_wave_parent(
        project, 2021, rows[2021][TARGET].to_numpy(np.float64)
    )
    meta22 = _metadata(project, 2022)
    if not np.array_equal(
        meta22["target"], rows[2022][TARGET].to_numpy(np.float64)
    ):
        raise ValueError("2022 metadata target/order mismatch")
    parent23 = v27_parent(frame23)

    fit_audits: dict[str, Any] = {}
    correction22_old, fit_audits["2020_to_2022"] = _fit(
        rows[2020], parent20, rows[2022], "2020 -> 2022"
    )
    correction22_recent, fit_audits["2021_to_2022"] = _fit(
        rows[2021], parent21, rows[2022], "2021 -> 2022"
    )
    correction23_old, fit_audits["2021_to_late2023"] = _fit(
        rows[2021], parent21, frame23, "2021 -> late-2023"
    )
    correction23_recent, fit_audits["2022_to_late2023"] = _fit(
        rows[2022], meta22["parent"], frame23, "2022 -> late-2023"
    )

    strict, strict_provenance = _load_strict(external_root, raw)
    source_frames = {"full_2022": rows[2022], "late_2023": frame23}
    source_parents = {
        "full_2022": _strict_parent(
            rows[2022], meta22["parent"], strict[2022]
        ),
        "late_2023": _strict_parent(
            frame23, parent23, strict[2023][late23_mask]
        ),
    }
    pairs = {
        "full_2022": (correction22_old, correction22_recent),
        "late_2023": (correction23_old, correction23_recent),
    }
    source_table, source_details = evaluate_policy_family(
        source_frames, source_parents, pairs
    )
    selected, ranking = select_policy(source_table)
    source_passed = bool(
        ranking.set_index("policy").loc[selected, "source_gate_passed"]
    )
    source_table.to_csv(output_dir / "source_policy_metrics.csv", index=False)
    ranking.to_csv(output_dir / "source_policy_ranking.csv", index=False)
    np.savez_compressed(
        output_dir / "source_corrections.npz",
        correction22_old=correction22_old,
        correction22_recent=correction22_recent,
        correction23_old=correction23_old,
        correction23_recent=correction23_recent,
    )

    print("[v87] source recipe frozen; fit 2022 -> 2024", flush=True)
    correction24_old, fit_audits["2022_to_2024"] = fit_predict_offset(
        rows[2022], meta22["parent"], frame24, rank=RANK
    )
    v53_path = (
        project
        / "artifacts"
        / "v53_factorization_offset_20260817_01"
        / "outer_full_2024.npz"
    )
    with np.load(v53_path, allow_pickle=False) as saved:
        if not np.array_equal(
            saved["target"].astype(np.float64), frame24[TARGET].to_numpy(np.float64)
        ):
            raise ValueError("v53/full-2024 target mismatch")
        correction24_recent = saved["correction"].astype(np.float64)

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
    v84, composition = exact_v84_parent(
        parent_axis, registry[V82_NAME], v56_candidate
    )
    if not np.array_equal(
        parent_axis["target"], frame24[TARGET].to_numpy(np.float64)
    ):
        raise ValueError("v84/full-2024 target mismatch")

    correction24, agreement24 = combine_corrections(
        correction24_old, correction24_recent, selected
    )
    candidate24, active24, _ = apply_reliability_offset(
        frame24, v84, correction24, "none", eta=ETA
    )
    metric24 = _metric_frame(frame24)
    full_audit = diagnostics(metric24, v84, candidate24, active24)
    late_mask = metric24["game_month"].ge(8).to_numpy()
    candidate_late, active_late, _ = apply_reliability_offset(
        frame24.loc[late_mask].reset_index(drop=True),
        v84[late_mask],
        correction24[late_mask],
        "none",
        eta=ETA,
    )
    late_audit = diagnostics(
        metric24.loc[late_mask].reset_index(drop=True),
        v84[late_mask],
        candidate_late,
        active_late,
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
        v84=v84,
        correction_old=correction24_old,
        correction_recent=correction24_recent,
        correction=correction24,
        sign_agreement=agreement24,
        candidate=candidate24,
        active=active24,
        domain3=metric24["domain3"].to_numpy(str),
        game_month=metric24["game_month"].to_numpy(np.int16),
    )
    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue / Public 1161.2020600422",
        "configuration": {
            "rank": RANK,
            "route": ROUTE,
            "eta": ETA,
            "policies": list(POLICIES),
            "selected_policy": selected,
        },
        "selection": "exact policy on full-2022 and late-2023 only",
        "source_gate_passed": source_passed,
        "source_policy_ranking": ranking.to_dict(orient="records"),
        "source_details": source_details,
        "fit_audits": fit_audits,
        "strict_provenance": strict_provenance,
        "composition_audit": composition,
        "audit_sign_agreement_fraction_r_core": float(
            agreement24[active24].mean()
        ),
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
        "same_family_2024_labels_used_for_policy_selection": False,
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
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.external_root, args.final_parent_dir, args.output_dir)


if __name__ == "__main__":
    main()

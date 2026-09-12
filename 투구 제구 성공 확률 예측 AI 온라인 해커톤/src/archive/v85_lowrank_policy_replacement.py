"""Audit recency policies as exact replacements for v84's strict component.

The Public-1161 parent already contains the EXP-021 equal-source low-rank
challenger at probability weight 0.10 on R_CORE.  EXP-037 published a fixed
family of alternative source-combination policies for that *same* low-rank
model.  This module does not stack another challenger on top.  It replaces the
embedded equal-source direction row by row::

    p_candidate = p_v84 + 0.10 * (p_policy - p_equal)  [R_CORE only]

The policy predictions are strict rolling OOF predictions.  Test rows are not
read, grouped, sorted, or otherwise used.  Because all policies are
structurally identical in 2022 (only one source OOF season exists), policy
choice has only one informative pre-2024 season.  The result is therefore a
diagnostic headroom audit and cannot by itself authorize packaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.archive.v57_public_strict_blend import blend_candidate
from src.core.oof_bank import (
    _assert_target,
    _load_common_candidates,
    _load_full24_diagnostic_candidates,
    _load_parent_axis,
)


PROTOCOL = "V85_EXP037_LOWRANK_POLICY_REPLACEMENT_ABOVE_V84_V1"
V82_NAME = "v57_independent_blend_shift"
STRICT_WEIGHT = 0.10
STRICT_ROUTE = "R_CORE"
POLICIES = (
    "equal",
    "last",
    "recency2",
    "recency4",
    "median",
    "trend025",
    "direction_shrink",
    "sign_consensus",
    "adaptive_recency2",
    "last_guarded",
)


def exact_v84_parent(
    axis: dict[str, np.ndarray],
    v82: np.ndarray,
    v56_candidate: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compose the exact OOF analogue of v84 from its disjoint R/F parts."""

    old_parent = np.asarray(axis["parent"], dtype=np.float64)
    current = np.asarray(v82, dtype=np.float64)
    v56 = np.asarray(v56_candidate, dtype=np.float64)
    domain = np.asarray(axis["domain3"]).astype(str)
    if not (old_parent.shape == current.shape == v56.shape == domain.shape):
        raise ValueError("v84 parent components are not aligned")
    futures = domain == "F"
    protected_diff = float(
        np.max(np.abs(current[futures] - old_parent[futures]))
    ) if futures.any() else 0.0
    if protected_diff > 1e-12:
        raise ValueError(
            "v82 unexpectedly changed F, so exact v84 composition is invalid: "
            f"{protected_diff}"
        )
    output = current.copy()
    output[futures] = v56[futures]
    return np.clip(output, 0.001, 0.999), {
        "f_rows": int(futures.sum()),
        "v82_f_parent_max_abs": protected_diff,
        "v84_f_mean_abs_shift": float(
            np.mean(np.abs(output[futures] - current[futures]))
        ) if futures.any() else 0.0,
    }


def replace_strict_policy(
    frame: pd.DataFrame,
    parent_with_equal: np.ndarray,
    equal_prediction: np.ndarray,
    policy_prediction: np.ndarray,
    *,
    weight: float = STRICT_WEIGHT,
    route: str = STRICT_ROUTE,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the equal-source raw challenger without double-counting it."""

    parent = np.asarray(parent_with_equal, dtype=np.float64)
    equal = np.asarray(equal_prediction, dtype=np.float64)
    policy = np.asarray(policy_prediction, dtype=np.float64)
    if not (len(frame) == len(parent) == len(equal) == len(policy)):
        raise ValueError("policy replacement arrays are not aligned")
    if not all(np.isfinite(value).all() for value in (parent, equal, policy)):
        raise ValueError("policy replacement contains non-finite values")
    domain = frame["domain3"].astype(str).to_numpy()
    active = domain == str(route)
    output = parent.copy()
    output[active] = np.clip(
        parent[active] + float(weight) * (policy[active] - equal[active]),
        0.001,
        0.999,
    )
    return output, active


def _policy_path(policy_dir: Path, policy: str, year: int) -> Path:
    return policy_dir / f"predictions_{policy}_{year}.npy"


def _target_path(external_root: Path, year: int) -> Path:
    return (
        external_root
        / "artifacts"
        / "EXP-020"
        / "low_rank_pitcher_context_eb"
        / f"targets_{year}.npy"
    )


def load_policy_oof(
    external_root: Path, policy_dir: Path, raw: pd.DataFrame
) -> dict[str, dict[int, np.ndarray]]:
    """Load strict rolling OOF policies with exact target/order checks."""

    output: dict[str, dict[int, np.ndarray]] = {name: {} for name in POLICIES}
    for year in (2022, 2023, 2024):
        expected = raw.loc[raw["season"].eq(year), "control_success"].to_numpy(
            np.float64
        )
        target = np.load(_target_path(external_root, year), allow_pickle=False).astype(
            np.float64
        )
        if not np.array_equal(target, expected):
            raise ValueError(f"EXP-037 target/order mismatch for {year}")
        for policy in POLICIES:
            value = np.load(
                _policy_path(policy_dir, policy, year), allow_pickle=False
            ).astype(np.float64)
            if value.shape != expected.shape or not np.isfinite(value).all():
                raise ValueError(f"invalid EXP-037 policy OOF: {policy}/{year}")
            if float(value.min()) < 0.0 or float(value.max()) > 1.0:
                raise ValueError(f"out-of-range EXP-037 policy OOF: {policy}/{year}")
            output[policy][year] = value
    return output


def _frame(axis: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": axis["target"],
            "game_month": axis["month"],
            "domain3": axis["domain3"],
        }
    )


def _compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "minimum_domain_gain": float(result["minimum_domain_gain"]),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def run(
    project: Path,
    external_root: Path,
    policy_dir: Path,
    final_parent_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    external_root = external_root.resolve()
    policy_dir = policy_dir.resolve()
    final_parent_dir = final_parent_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    policies = load_policy_oof(external_root, policy_dir, raw)
    axes = {
        name: _load_parent_axis(final_parent_dir, name)
        for name in ("full_2024", "late_2024")
    }
    common = _load_common_candidates(project, axes["full_2024"], "full_2024")
    full_registry = _load_full24_diagnostic_candidates(
        project, axes["full_2024"], common
    )
    if V82_NAME not in full_registry:
        raise ValueError("v82 exact OOF candidate is missing")

    v56_path = (
        project
        / "artifacts"
        / "v56_shared_horizon_fm_20260817_01"
        / "outer_full_2024.npz"
    )
    with np.load(v56_path, allow_pickle=True) as saved:
        _assert_target(axes["full_2024"]["target"], saved, "v56/full_2024")
        v56_candidate = saved["candidate"].astype(np.float64)
    v84_full, composition = exact_v84_parent(
        axes["full_2024"], full_registry[V82_NAME], v56_candidate
    )
    late_mask = axes["full_2024"]["month"] >= 8
    if not np.array_equal(
        axes["full_2024"]["target"][late_mask], axes["late_2024"]["target"]
    ):
        raise ValueError("full/late 2024 target mismatch")
    v84_late = v84_full[late_mask]

    full_frame = _frame(axes["full_2024"])
    late_frame = _frame(axes["late_2024"])
    equal24 = policies["equal"][2024]
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for policy in POLICIES:
        candidate_full, active_full = replace_strict_policy(
            full_frame, v84_full, equal24, policies[policy][2024]
        )
        candidate_late, active_late = replace_strict_policy(
            late_frame,
            v84_late,
            equal24[late_mask],
            policies[policy][2024][late_mask],
        )
        full_result = diagnostics(
            full_frame, v84_full, candidate_full, active_full
        )
        late_result = diagnostics(
            late_frame, v84_late, candidate_late, active_late
        )
        details[policy] = {
            "full_2024": full_result,
            "late_2024": late_result,
        }
        for axis_name, result in (
            ("full_2024", full_result),
            ("late_2024", late_result),
        ):
            rows.append({"policy": policy, "axis": axis_name, **_compact(result)})

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "policy_replacement_metrics.csv", index=False)
    audit_rank = (
        table.pivot(index="policy", columns="axis", values="gain")
        .assign(min_gain=lambda item: item.min(axis=1))
        .sort_values(["min_gain", "full_2024"], ascending=False)
    )
    best_policy = str(audit_rank.index[0])
    best = details[best_policy]
    point_gates = {
        "full_gain_positive": float(best["full_2024"]["gain"]) > 0.0,
        "late_gain_positive": float(best["late_2024"]["gain"]) > 0.0,
        "full_month_fraction_at_least_075": float(
            best["full_2024"]["positive_month_fraction"]
        ) >= 0.75,
        "late_month_fraction_at_least_075": float(
            best["late_2024"]["positive_month_fraction"]
        ) >= 0.75,
        "both_worst_months_above_minus_5": min(
            float(best["full_2024"]["worst_month_gain"]),
            float(best["late_2024"]["worst_month_gain"]),
        ) > -5.0,
        "both_minimum_domains_nonnegative": min(
            float(best["full_2024"]["minimum_domain_gain"]),
            float(best["late_2024"]["minimum_domain_gain"]),
        ) >= 0.0,
    }
    result = {
        "protocol": PROTOCOL,
        "parent": "exact v84 OOF analogue / Public 1161.2020600422",
        "replacement": {
            "embedded_policy": "equal",
            "route": STRICT_ROUTE,
            "weight": STRICT_WEIGHT,
            "formula": "p_v84 + 0.10 * (p_policy - p_equal) on R_CORE",
        },
        "composition_audit": composition,
        "policies": list(POLICIES),
        "audit_best_policy": best_policy,
        "audit_best": best,
        "point_gates": {name: bool(value) for name, value in point_gates.items()},
        "point_gates_passed": bool(all(point_gates.values())),
        "eligible_for_packaging": False,
        "promotion_block": (
            "policy directions are identical in 2022, leaving only one "
            "informative pre-2024 policy-selection season; v3 requires two"
        ),
        "same_family_2024_labels_used_for_audit_ranking": True,
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
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.project,
        args.external_root,
        args.policy_dir,
        args.final_parent_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

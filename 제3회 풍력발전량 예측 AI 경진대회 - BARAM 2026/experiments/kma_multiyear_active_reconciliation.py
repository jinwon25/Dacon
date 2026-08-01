"""Reconcile a multiyear KMA ablation against the frozen active OOF surface.

This is a diagnostic-only experiment.  It compares otherwise identical
2023-only and 2022+2023 pooled KMA models, then replaces targets on the exact
active validation baseline.  The script never creates a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.incumbent_residual_noncrossing import (
    COMPONENTS,
    complementary_subset_stress_all,
    interval_month,
    issue_block_bootstrap_all,
    metric_delta,
    movement_summary,
)
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
DEFAULT_PRIMARY_CACHE = (
    "artifacts_final/lineage/"
    "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
)
DEFAULT_RESIDUAL_CACHE = (
    "artifacts_final/lineage/kma_jma_msm_stencil_production_20260726.npz"
)
DEFAULT_GROUP3_CACHE = (
    "artifacts_final/external_weather/"
    "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_pooled_cache(path: Path) -> dict[str, Any]:
    """Load and validate a pooled KMA validation cache."""
    with np.load(path, allow_pickle=False) as cache:
        indexes: dict[str, pd.DatetimeIndex] = {}
        issues: dict[str, pd.DatetimeIndex] = {}
        payload: dict[str, dict[str, np.ndarray]] = {}
        for target in TARGETS:
            prefix = f"{target}__"
            required = (
                "index_ns",
                "issue_ns",
                "truth",
                "reference",
                "expert",
                "candidate",
            )
            missing = [name for name in required if prefix + name not in cache]
            if missing:
                raise ValueError(f"{path} is missing {target} keys: {missing}")
            indexes[target] = pd.DatetimeIndex(
                pd.to_datetime(cache[prefix + "index_ns"])
            )
            issues[target] = pd.DatetimeIndex(
                pd.to_datetime(cache[prefix + "issue_ns"])
            )
            payload[target] = {
                name: np.asarray(cache[prefix + name], dtype=float)
                for name in ("truth", "reference", "expert", "candidate")
            }

    index = indexes[TARGETS[0]]
    issue = issues[TARGETS[0]]
    for target in TARGETS[1:]:
        if not indexes[target].equals(index):
            raise ValueError(f"pooled cache indexes differ for {target}")
        if not issues[target].equals(issue):
            raise ValueError(f"pooled cache issue cycles differ for {target}")
    if any(
        len(payload[target][name]) != len(index)
        for target in TARGETS
        for name in payload[target]
    ):
        raise ValueError("pooled cache arrays do not match the validation index")
    return {"index": index, "issues": issue, "targets": payload}


def compose_replacement(
    active: dict[str, np.ndarray],
    cache: dict[str, Any],
    replace_targets: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Replace specified active target surfaces with locked cache candidates."""
    unknown = sorted(set(replace_targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown replacement targets: {unknown}")
    output = {
        target: np.asarray(active[target], dtype=float).copy()
        for target in TARGETS
    }
    for target in replace_targets:
        output[target] = np.asarray(
            cache["targets"][target]["candidate"], dtype=float
        ).copy()
    return output


def period_rows(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    months = interval_month(index)
    return {
        "q1": months <= 3,
        "q2": (months >= 4) & (months <= 6),
        "h2": months >= 7,
        "full": np.ones(len(index), dtype=bool),
    }


def _subset_pass(result: dict[str, Any]) -> bool:
    return bool(
        all(
            result[split][component]["q05"] >= 0.0
            for split in ("public", "private")
            for component in COMPONENTS
        )
    )


def _bootstrap_pass(result: dict[str, Any]) -> bool:
    return bool(
        all(
            result["summary"][component]["q05"] >= 0.0
            for component in COMPONENTS
        )
    )


def evaluate_replacement(
    truth: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    issues: pd.Series,
    *,
    run_stress: bool,
    family_guard_pass: bool,
    subset_repetitions: int,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rows = period_rows(index)
    months = interval_month(index)
    periods = {
        name: metric_delta(truth, active, candidate, mask)
        for name, mask in rows.items()
    }
    monthly = {
        str(month): metric_delta(
            truth, active, candidate, months == month
        )
        for month in sorted(set(months))
    }
    result: dict[str, Any] = {
        "period_deltas": periods,
        "monthly_deltas": monthly,
        "movement": movement_summary(active, candidate),
        "pre_stress_gates": {
            "q2_components_positive": all(
                periods["q2"][component] > 0.0
                for component in COMPONENTS
            ),
            "h2_components_positive": all(
                periods["h2"][component] > 0.0
                for component in COMPONENTS
            ),
            "full_components_positive": all(
                periods["full"][component] > 0.0
                for component in COMPONENTS
            ),
            "all_month_scores_nonnegative": all(
                record["score"] >= 0.0 for record in monthly.values()
            ),
        },
        "stress_executed": bool(run_stress),
        "historical_public_failure_family_guard_pass": bool(
            family_guard_pass
        ),
    }
    if not run_stress:
        result["promotion_eligible"] = False
        result["decision"] = "diagnostic_only_not_stress_tested"
        return result

    full_rows = rows["full"]
    iid = complementary_subset_stress_all(
        truth,
        active,
        candidate,
        index,
        full_rows,
        repetitions=subset_repetitions,
        seed=seed,
        stratify_month=False,
    )
    stratified = complementary_subset_stress_all(
        truth,
        active,
        candidate,
        index,
        full_rows,
        repetitions=subset_repetitions,
        seed=seed + 1,
        stratify_month=True,
    )
    bootstrap = issue_block_bootstrap_all(
        truth,
        active,
        candidate,
        index,
        issues,
        rows["h2"],
        repetitions=bootstrap_repetitions,
        seed=seed + 2,
    )
    result["iid_complementary_40_60"] = iid
    result["month_stratified_complementary_40_60"] = stratified
    result["h2_issue_block_bootstrap"] = bootstrap
    result["stress_gates"] = {
        "iid_all_q05_nonnegative": _subset_pass(iid),
        "month_stratified_all_q05_nonnegative": _subset_pass(stratified),
        "h2_issue_bootstrap_all_q05_nonnegative": _bootstrap_pass(bootstrap),
    }
    statistical_eligible = bool(
        all(result["pre_stress_gates"].values())
        and all(result["stress_gates"].values())
    )
    eligible = bool(statistical_eligible and family_guard_pass)
    result["statistical_gate_eligible"] = statistical_eligible
    result["promotion_eligible"] = eligible
    if eligible:
        result["decision"] = "eligible"
    elif statistical_eligible and not family_guard_pass:
        result["decision"] = "rejected_historical_public_failure_guard"
    else:
        result["decision"] = "rejected_fail_closed"
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    multiyear = load_pooled_cache(_rooted(args.multiyear_cache))
    control = load_pooled_cache(_rooted(args.control_cache))
    multiyear_report = _load_json(_rooted(args.multiyear_report))
    control_report = _load_json(_rooted(args.control_report))

    baselines, truths, index, issues = load_frozen_validation_baselines(
        _rooted(args.primary_cache),
        _rooted(args.residual_cache),
        _rooted(args.group3_cache),
    )
    active = {
        target: baselines[target].to_numpy(dtype=float)
        for target in TARGETS
    }
    truth = {
        target: truths[target].to_numpy(dtype=float)
        for target in TARGETS
    }

    for name, cache in (("multiyear", multiyear), ("control", control)):
        if not cache["index"].equals(index):
            raise ValueError(f"{name} index does not match the active surface")
        for target in TARGETS:
            if not np.allclose(
                cache["targets"][target]["truth"],
                truth[target],
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{name} truth does not match active truth for {target}"
                )

    reference_difference = {
        target: float(
            np.max(
                np.abs(
                    multiyear["targets"][target]["reference"]
                    - control["targets"][target]["reference"]
                )
            )
        )
        for target in TARGETS
    }
    selected_ablation = {
        name: metric_delta(
            truth,
            {
                target: control["targets"][target]["candidate"]
                for target in TARGETS
            },
            {
                target: multiyear["targets"][target]["candidate"]
                for target in TARGETS
            },
            mask,
        )
        for name, mask in period_rows(index).items()
    }

    result: dict[str, Any] = {
        "family": "kma_multiyear_active_reconciliation",
        "method": (
            "2022+2023 versus 2023-only ablation followed by exact frozen-"
            "active target replacement; no submission writer"
        ),
        "contract": {
            "selection": "locked by each upstream Q1-only experiment",
            "confirmation": (
                "Q2, H2, months, full-year complementary 40/60, "
                "H2 issue bootstrap"
            ),
            "subset_repetitions": args.subset_repetitions,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "public_score_used_for_selection": False,
            "submission_written": False,
            "historical_public_failure_guard": {
                "family": "group3_external_weather_replacement",
                "blocked_targets": ["kpx_group_3"],
                "reason": (
                    "multiple locally positive group-3 weather replacements "
                    "reversed on the public leaderboard"
                ),
            },
        },
        "sources": {
            "multiyear_cache": args.multiyear_cache,
            "control_cache": args.control_cache,
            "multiyear_report": args.multiyear_report,
            "control_report": args.control_report,
            "primary_active_cache": args.primary_cache,
            "residual_active_cache": args.residual_cache,
            "group3_active_cache": args.group3_cache,
        },
        "alignment": {
            "timestamps": len(index),
            "truth_matches_active": True,
            "maximum_reference_difference_multiyear_vs_control": reference_difference,
        },
        "upstream_expected_macro_delta": {
            "multiyear": multiyear_report[
                "expected_macro_score_delta_if_2024_transfers"
            ],
            "control": control_report[
                "expected_macro_score_delta_if_2024_transfers"
            ],
            "multiyear_minus_control": (
                multiyear_report[
                    "expected_macro_score_delta_if_2024_transfers"
                ]
                - control_report[
                    "expected_macro_score_delta_if_2024_transfers"
                ]
            ),
        },
        "selected_surface_multiyear_minus_control": selected_ablation,
        "active_replacements": {},
    }

    for family, cache, report in (
        ("multiyear", multiyear, multiyear_report),
        ("control", control, control_report),
    ):
        deployment = tuple(report["deployment_targets"])
        variants = {
            target: (target,) for target in TARGETS
        }
        variants["deployment"] = deployment
        family_result: dict[str, Any] = {
            "locked_upstream_deployment_targets": list(deployment),
            "variants": {},
        }
        for variant, replace_targets in variants.items():
            candidate = compose_replacement(active, cache, replace_targets)
            run_stress = variant == "deployment" or (
                family == "multiyear" and variant == "kpx_group_3"
            )
            family_result["variants"][variant] = {
                "replace_targets": list(replace_targets),
                **evaluate_replacement(
                    truth,
                    active,
                    candidate,
                    index,
                    issues,
                    run_stress=run_stress,
                    family_guard_pass=(
                        "kpx_group_3" not in replace_targets
                    ),
                    subset_repetitions=args.subset_repetitions,
                    bootstrap_repetitions=args.bootstrap_repetitions,
                    seed=args.seed,
                ),
            }
        result["active_replacements"][family] = family_result

    eligible = [
        f"{family}:{variant}"
        for family, family_result in result["active_replacements"].items()
        for variant, record in family_result["variants"].items()
        if record["promotion_eligible"]
    ]
    result["eligible_replacements"] = eligible
    result["verdict"] = (
        "eligible_replacement_found" if eligible else "no_replacement_passed"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiyear-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_umrg_full2022_pooled_all3_20260802.npz"
        ),
    )
    parser.add_argument(
        "--control-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_umrg_2023only_pooled_all3_control_20260802.npz"
        ),
    )
    parser.add_argument(
        "--multiyear-report",
        default=(
            "artifacts_final/diagnostics/"
            "kma_umrg_full2022_pooled_all3_20260802.json"
        ),
    )
    parser.add_argument(
        "--control-report",
        default=(
            "artifacts_final/diagnostics/"
            "kma_umrg_2023only_pooled_all3_control_20260802.json"
        ),
    )
    parser.add_argument("--primary-cache", default=DEFAULT_PRIMARY_CACHE)
    parser.add_argument("--residual-cache", default=DEFAULT_RESIDUAL_CACHE)
    parser.add_argument("--group3-cache", default=DEFAULT_GROUP3_CACHE)
    parser.add_argument("--subset-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output-report",
        default=(
            "artifacts_final/diagnostics/"
            "kma_multiyear_active_reconciliation_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    output_path = _rooted(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "upstream_expected_macro_delta": report[
            "upstream_expected_macro_delta"
        ],
        "eligible_replacements": report["eligible_replacements"],
        "verdict": report["verdict"],
        "output_report": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Compose a multiyear-gated February-only group-2 dose candidate.

The scored group-2 weight 0.1825 remains the year-round anchor.  The extra
February is the final narrow calendar hypothesis after broader Q1 and
January--February routes fail their gates. A year-round extra dose regresses
2024 H2. Public metrics calibrate only the expected scale after the policy is
selected by local multiyear gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.blocked_rolling_validation import assign_issue_blocks
from experiments.compose_public_positive_multifactor_candidate import (
    OBSERVED_G1_WEIGHT,
    expand_observed_factor,
)
from experiments.compose_residual_stack_candidate import (
    apply_capped_residual_stack,
)
from experiments.incumbent_residual_noncrossing import (
    COMPONENTS,
    complementary_subset_stress_all,
    interval_month,
    issue_block_bootstrap_all,
    metric_delta,
    movement_summary,
)
from experiments.kma_year_forward_quantile_blend import apply_bounded_blend
from experiments.mechanism_diversity_blend_audit import period_rows
from experiments.multimodel_expanding_quantile_blend import (
    load_frozen_validation_baselines,
)
from src.metrics import CAPACITY_KWH, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
ID_COLUMNS = ("forecast_id", "forecast_kst_dtm")
TARGET = "kpx_group_2"
BASE_PRODUCTION_WEIGHT = 0.05
ANCHOR_WEIGHT = 0.1825
Q1_WEIGHT = 0.235
ACTIVE_MONTHS = (2,)
PUBLIC_LOWER = {
    "score": 0.6470679857,
    "one_minus_nmae": 0.8759467997,
    "ficr": 0.4181891717,
}
PUBLIC_ANCHOR = {
    "score": 0.6474704399,
    "one_minus_nmae": 0.8761125755,
    "ficr": 0.4188283043,
}


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def q1_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Return January--March rows."""
    return np.asarray(index.month <= 3, dtype=bool)


def active_month_mask(
    index: pd.DatetimeIndex,
    months: tuple[int, ...] = ACTIVE_MONTHS,
) -> np.ndarray:
    """Return rows belonging to the predeclared active calendar months."""
    return np.asarray(index.month.isin(months), dtype=bool)


def apply_month_route(
    anchor: np.ndarray,
    q1_treatment: np.ndarray,
    index: pd.DatetimeIndex,
    months: tuple[int, ...] = ACTIVE_MONTHS,
) -> np.ndarray:
    """Replace only active-month rows while preserving all others exactly."""
    anchor = np.asarray(anchor, dtype=float)
    treatment = np.asarray(q1_treatment, dtype=float)
    if anchor.shape != treatment.shape or len(anchor) != len(index):
        raise ValueError("anchor, treatment, and index must align")
    output = anchor.copy()
    rows = active_month_mask(index, months)
    output[rows] = treatment[rows]
    return output


def _single_group_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float]:
    base = evaluate_group(
        truth[rows], reference[rows], CAPACITY_KWH[TARGET]
    )
    new = evaluate_group(
        truth[rows], candidate[rows], CAPACITY_KWH[TARGET]
    )
    return {
        "score": float(new.score - base.score),
        "one_minus_nmae": float(new.one_minus_nmae - base.one_minus_nmae),
        "ficr": float(new.ficr - base.ficr),
    }


def _group_issue_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    issues: pd.DatetimeIndex,
    rows: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    issue_values = np.asarray(issues)
    unique_issues = np.unique(issue_values[rows])
    positions = {
        issue: np.flatnonzero(rows & (issue_values == issue))
        for issue in unique_issues
    }
    rng = np.random.default_rng(seed)
    values = {
        component: np.empty(repetitions, dtype=float)
        for component in COMPONENTS
    }
    for repetition in range(repetitions):
        sampled = np.concatenate(
            [
                positions[issue]
                for issue in rng.choice(
                    unique_issues, size=len(unique_issues), replace=True
                )
            ]
        )
        delta = _single_group_delta(
            truth, reference, candidate, sampled
        )
        for component in COMPONENTS:
            values[component][repetition] = delta[component]
    return {
        "repetitions": int(repetitions),
        "issues": int(len(unique_issues)),
        "summary": {
            component: {
                "q05": float(np.quantile(component_values, 0.05)),
                "median": float(np.quantile(component_values, 0.50)),
                "q95": float(np.quantile(component_values, 0.95)),
                "positive_fraction": float(
                    np.mean(component_values > 0.0)
                ),
            }
            for component, component_values in values.items()
        },
    }


def _subset_min_q05(result: dict[str, Any], component: str) -> float:
    return float(
        min(
            result[split][component]["q05"]
            for split in ("public", "private")
        )
    )


def _load_prior_year(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cache:
        return {
            "index": pd.DatetimeIndex(pd.to_datetime(cache["index_ns"])),
            "issues": pd.DatetimeIndex(pd.to_datetime(cache["issue_ns"])),
            "truth": cache["truth"].astype(float),
            "l1": cache["l1"].astype(float),
            "quantile": cache["quantile"].astype(float),
        }


def _period_and_month_deltas_group(
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    rows = period_rows(index)
    months = interval_month(index)
    return {
        "periods": {
            name: _single_group_delta(truth, anchor, candidate, mask)
            for name, mask in rows.items()
        },
        "months": {
            str(month): _single_group_delta(
                truth, anchor, candidate, months == month
            )
            for month in range(1, 13)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "anchor_submission": _rooted(args.anchor_submission),
        "production_incumbent": _rooted(args.production_incumbent),
        "group2_control": _rooted(args.group2_control),
        "primary_cache": _rooted(args.primary_cache),
        "residual_cache": _rooted(args.residual_cache),
        "group3_cache": _rooted(args.group3_cache),
        "prior_year_cache": _rooted(args.prior_year_cache),
    }
    anchor_submission = pd.read_csv(
        paths["anchor_submission"], encoding="utf-8-sig"
    )
    production_incumbent = pd.read_csv(
        paths["production_incumbent"], encoding="utf-8-sig"
    )
    group2_control = pd.read_csv(
        paths["group2_control"], encoding="utf-8-sig"
    )
    for name, frame in (
        ("production_incumbent", production_incumbent),
        ("group2_control", group2_control),
    ):
        if not anchor_submission[list(ID_COLUMNS)].equals(frame[list(ID_COLUMNS)]):
            raise ValueError(f"{name} identifiers differ from anchor")

    baselines, truth_series, index_2024, issues_2024 = (
        load_frozen_validation_baselines(
            paths["primary_cache"],
            paths["residual_cache"],
            paths["group3_cache"],
        )
    )
    active = {
        target: baselines[target].to_numpy(dtype=float) for target in TARGETS
    }
    truth_2024 = {
        target: truth_series[target].to_numpy(dtype=float)
        for target in TARGETS
    }
    with np.load(paths["primary_cache"], allow_pickle=False) as primary, np.load(
        paths["residual_cache"], allow_pickle=False
    ) as residual:
        anchor_2024 = {
            target: active[target].copy() for target in TARGETS
        }
        anchor_2024["kpx_group_1"] = apply_capped_residual_stack(
            primary["kpx_group_1__reference"],
            primary["kpx_group_1__candidate"],
            residual["kpx_group_1__candidate"],
            residual_weight=OBSERVED_G1_WEIGHT,
            capacity=CAPACITY_KWH["kpx_group_1"],
            movement_cap_ratio=0.05,
        )
        lower_2024 = {
            target: values.copy() for target, values in anchor_2024.items()
        }
        lower_2024[TARGET] = apply_bounded_blend(
            primary[f"{TARGET}__reference"],
            primary[f"{TARGET}__expert"],
            weight=0.095,
            capacity=CAPACITY_KWH[TARGET],
        )
        anchor_2024[TARGET] = apply_bounded_blend(
            primary[f"{TARGET}__reference"],
            primary[f"{TARGET}__expert"],
            weight=ANCHOR_WEIGHT,
            capacity=CAPACITY_KWH[TARGET],
        )
        full_dose_2024 = apply_bounded_blend(
            primary[f"{TARGET}__reference"],
            primary[f"{TARGET}__expert"],
            weight=Q1_WEIGHT,
            capacity=CAPACITY_KWH[TARGET],
        )
    candidate_2024 = {
        target: values.copy() for target, values in anchor_2024.items()
    }
    candidate_2024[TARGET] = apply_month_route(
        anchor_2024[TARGET], full_dose_2024, index_2024
    )
    periods_2024 = period_rows(index_2024)
    months_2024_values = interval_month(index_2024)
    validation_2024 = {
        "periods": {
            name: metric_delta(
                truth_2024, anchor_2024, candidate_2024, rows
            )
            for name, rows in periods_2024.items()
        },
        "months": {
            str(month): metric_delta(
                truth_2024,
                anchor_2024,
                candidate_2024,
                months_2024_values == month,
            )
            for month in range(1, 13)
        },
    }
    full_rows_2024 = periods_2024["full"]
    iid = complementary_subset_stress_all(
        truth_2024,
        anchor_2024,
        candidate_2024,
        index_2024,
        full_rows_2024,
        repetitions=args.subset_repetitions,
        seed=args.seed,
        stratify_month=False,
    )
    stratified = complementary_subset_stress_all(
        truth_2024,
        anchor_2024,
        candidate_2024,
        index_2024,
        full_rows_2024,
        repetitions=args.subset_repetitions,
        seed=args.seed + 1,
        stratify_month=True,
    )
    q1_bootstrap_2024 = issue_block_bootstrap_all(
        truth_2024,
        anchor_2024,
        candidate_2024,
        index_2024,
        issues_2024,
        active_month_mask(index_2024),
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 2,
    )

    prior = _load_prior_year(paths["prior_year_cache"])
    prior_anchor = np.clip(
        prior["l1"] + ANCHOR_WEIGHT * (prior["quantile"] - prior["l1"]),
        0.0,
        CAPACITY_KWH[TARGET],
    )
    prior_full_dose = np.clip(
        prior["l1"] + Q1_WEIGHT * (prior["quantile"] - prior["l1"]),
        0.0,
        CAPACITY_KWH[TARGET],
    )
    prior_candidate = apply_month_route(
        prior_anchor, prior_full_dose, prior["index"]
    )
    validation_2023 = _period_and_month_deltas_group(
        prior["truth"],
        prior_anchor,
        prior_candidate,
        prior["index"],
    )
    seasons_2023, _ = assign_issue_blocks(
        prior["index"], prior["issues"]
    )
    validation_2023["issue_seasons"] = {
        str(season): _single_group_delta(
            prior["truth"],
            prior_anchor,
            prior_candidate,
            seasons_2023 == season,
        )
        for season in dict.fromkeys(seasons_2023)
    }
    q1_bootstrap_2023 = _group_issue_bootstrap(
        prior["truth"],
        prior_anchor,
        prior_candidate,
        prior["issues"],
        active_month_mask(prior["index"]),
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 3,
    )

    local_calibration = metric_delta(
        truth_2024, lower_2024, anchor_2024, full_rows_2024
    )
    public_calibration = {
        component: float(PUBLIC_ANCHOR[component] - PUBLIC_LOWER[component])
        for component in COMPONENTS
    }
    score_transfer_ratio = (
        public_calibration["score"] / local_calibration["score"]
    )
    projected_public_score = float(
        PUBLIC_ANCHOR["score"]
        + validation_2024["periods"]["full"]["score"]
        * score_transfer_ratio
    )

    production_full_dose = expand_observed_factor(
        group2_control[TARGET].to_numpy(dtype=float),
        production_incumbent[TARGET].to_numpy(dtype=float),
        base_weight=BASE_PRODUCTION_WEIGHT,
        expanded_weight=Q1_WEIGHT,
        capacity=CAPACITY_KWH[TARGET],
        maximum_factor_ratio=args.maximum_factor_ratio,
    )
    production_anchor_rebuilt = expand_observed_factor(
        group2_control[TARGET].to_numpy(dtype=float),
        production_incumbent[TARGET].to_numpy(dtype=float),
        base_weight=BASE_PRODUCTION_WEIGHT,
        expanded_weight=ANCHOR_WEIGHT,
        capacity=CAPACITY_KWH[TARGET],
        maximum_factor_ratio=args.maximum_factor_ratio,
    )
    anchor_error = float(
        np.max(
            np.abs(
                production_anchor_rebuilt
                - anchor_submission[TARGET].to_numpy(dtype=float)
            )
        )
    )
    if anchor_error > 1e-8:
        raise ValueError("provided anchor does not match rebuilt 0.1825 factor")
    output = anchor_submission.copy()
    production_index = pd.DatetimeIndex(
        pd.to_datetime(output["forecast_kst_dtm"])
    )
    output[TARGET] = apply_month_route(
        output[TARGET].to_numpy(dtype=float),
        production_full_dose,
        production_index,
    )
    output_path = _rooted(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(ROOT)).audit(output_path)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")

    months_active = ACTIVE_MONTHS
    gates = {
        "2023_all_active_month_scores_positive": all(
            validation_2023["months"][str(month)]["score"] > 0.0
            for month in months_active
        ),
        "2023_all_issue_season_scores_nonnegative": all(
            record["score"] >= 0.0
            for record in validation_2023["issue_seasons"].values()
        ),
        "2023_q1_bootstrap_score_q05_positive": bool(
            q1_bootstrap_2023["summary"]["score"]["q05"] > 0.0
        ),
        "2023_q1_bootstrap_ficr_q05_positive": bool(
            q1_bootstrap_2023["summary"]["ficr"]["q05"] > 0.0
        ),
        "2024_all_active_month_scores_positive": all(
            validation_2024["months"][str(month)]["score"] > 0.0
            for month in months_active
        ),
        "2024_iid_score_q05_positive": bool(
            _subset_min_q05(iid, "score") > 0.0
        ),
        "2024_stratified_score_q05_positive": bool(
            _subset_min_q05(stratified, "score") > 0.0
        ),
        "2024_q1_bootstrap_score_q05_positive": bool(
            q1_bootstrap_2024["summary"]["score"]["q05"] > 0.0
        ),
        "2024_q1_bootstrap_ficr_q05_positive": bool(
            q1_bootstrap_2024["summary"]["ficr"]["q05"] > 0.0
        ),
    }
    qualified = bool(all(gates.values()))
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    retained = bool(qualified or args.write_rejected_candidate)
    if not retained:
        output_path.unlink()
    report = {
        "schema_version": "multiyear_g2_q1_dose_candidate.v1",
        "contract": {
            "anchor_submission_id": 1508386,
            "anchor_group2_weight": ANCHOR_WEIGHT,
            "q1_group2_weight": Q1_WEIGHT,
            "active_months": ACTIVE_MONTHS,
            "q2_h2_group2_weight": ANCHOR_WEIGHT,
            "group1_frozen": True,
            "group3_frozen": True,
            "public_score_selects_policy": False,
            "policy_selected_on_multiyear_oof": True,
        },
        "validation_2023": validation_2023,
        "validation_2024": validation_2024,
        "stress": {
            "2023_q1_issue_block_bootstrap": q1_bootstrap_2023,
            "2024_iid_complementary_40_60": iid,
            "2024_month_stratified_complementary_40_60": stratified,
            "2024_q1_issue_block_bootstrap": q1_bootstrap_2024,
        },
        "public_scale_projection": {
            "calibration_submissions": [1508365, 1508386],
            "local_calibration_delta": local_calibration,
            "observed_public_calibration_delta": public_calibration,
            "direct_score_transfer_ratio": float(score_transfer_ratio),
            "projected_public_score": projected_public_score,
            "warning": (
                "Scale-only estimate after multiyear local policy selection; "
                "it is not an observed score."
            ),
        },
        "production": {
            "anchor_rebuild_max_abs_error": anchor_error,
            "active_rows": int(active_month_mask(production_index).sum()),
            "movement_vs_anchor": movement_summary(
                {
                    target: anchor_submission[target].to_numpy(dtype=float)
                    for target in TARGETS
                },
                {
                    target: output[target].to_numpy(dtype=float)
                    for target in TARGETS
                },
            ),
        },
        "promotion_gates": gates,
        "qualified": qualified,
        "decision": (
            "candidate_created" if qualified else "rejected_fail_closed"
        ),
        "candidate": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": output_sha256,
            "rows": int(len(output)),
            "retained": retained,
            "candidate_validator": audit.to_dict(),
        },
    }
    report_path = _rooted(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor-submission",
        default=(
            "artifacts_final/candidates/"
            "public_positive_g1w1375_g2w1825_g3frozen_20260802.csv"
        ),
    )
    parser.add_argument(
        "--production-incumbent",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument(
        "--group2-control",
        default=(
            "artifacts_final/candidates/"
            "kma_group2_overlay_alpha2375_20260725.csv"
        ),
    )
    parser.add_argument(
        "--primary-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_pooled_all3_g1g2_nearstable_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--residual-cache",
        default=(
            "artifacts_final/lineage/"
            "kma_jma_msm_stencil_production_20260726.npz"
        ),
    )
    parser.add_argument(
        "--group3-cache",
        default=(
            "artifacts_final/external_weather/"
            "kma_um_regional_context_2024/power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--prior-year-cache",
        default="artifacts_final/lineage/prior_year_nested_group2_oof.npz",
    )
    parser.add_argument("--maximum-factor-ratio", type=float, default=0.10)
    parser.add_argument("--write-rejected-candidate", action="store_true")
    parser.add_argument("--subset-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        default=(
            "artifacts_final/candidates/"
            "public_positive_g1w1375_g2febw2350_else1825_g3frozen_20260802.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "artifacts_final/diagnostics/"
            "multiyear_g2_q1_dose_candidate_20260802.json"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "qualified": report["qualified"],
                "promotion_gates": report["promotion_gates"],
                "validation_2023_full": report["validation_2023"][
                    "periods"
                ]["full"],
                "validation_2024_full": report["validation_2024"][
                    "periods"
                ]["full"],
                "projected_public_score": report["public_scale_projection"][
                    "projected_public_score"
                ],
                "candidate": report["candidate"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

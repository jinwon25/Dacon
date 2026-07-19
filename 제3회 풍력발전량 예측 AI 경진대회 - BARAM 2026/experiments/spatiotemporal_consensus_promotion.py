from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from experiments.blocked_rolling_validation import (
    assign_issue_blocks,
    evaluate_blocked_rolling,
    load_issue_times,
)
from experiments.exact_oof_meta_gate import (
    CAPACITY,
    H2_START,
    Q2_START,
    _evaluate_period,
    apply_meta_gate,
    fit_probabilities,
    settlement_benefit_labels,
)
from experiments.exact_oof_meta_gate_sweep import _prepare_validation
from experiments.spatiotemporal_multitask import (
    TARGETS,
)
from agent_service.config import load_config
from agent_service.submission import CandidateValidator


TARGET = "kpx_group_3"
SERVICE_FAMILY = "spatiotemporal_multitask_blend"
KEY_COLUMNS = ["forecast_id", "forecast_kst_dtm"]
FINE_THRESHOLD = 0.545
FINE_ALPHA = 0.50
PERIOD_MONTHS = {
    "q1": frozenset(("2024-01", "2024-02", "2024-03")),
    "q2": frozenset(("2024-04", "2024-05", "2024-06")),
    "h2": frozenset(("2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12")),
}


@dataclass(frozen=True)
class BlendPolicy:
    family: str
    alpha: float
    require_seed_agreement: bool = False
    max_seed_uncertainty: float | None = None
    max_base_disagreement: float | None = None

    @property
    def name(self) -> str:
        return f"{self.family}_a{int(round(self.alpha * 100)):02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "alpha": self.alpha,
            "require_seed_agreement": self.require_seed_agreement,
            "max_seed_uncertainty_capacity_ratio": self.max_seed_uncertainty,
            "max_base_disagreement_capacity_ratio": self.max_base_disagreement,
        }


def default_policies() -> tuple[BlendPolicy, ...]:
    """Small, pre-declared grid; deliberately not a fine threshold sweep."""
    families = (
        ("global", False, None, None),
        ("seed_consensus", True, None, None),
        ("bounded_u04_d10", True, 0.04, 0.10),
        ("bounded_u06_d10", True, 0.06, 0.10),
    )
    return tuple(
        BlendPolicy(family, alpha, agreement, max_uncertainty, max_disagreement)
        for family, agreement, max_uncertainty, max_disagreement in families
        for alpha in (0.10, 0.20, 0.30)
    )


def policy_mask(
    base: np.ndarray,
    seed_predictions: np.ndarray,
    policy: BlendPolicy,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base = np.asarray(base, dtype=float)
    seed_predictions = np.asarray(seed_predictions, dtype=float)
    if seed_predictions.ndim != 2 or seed_predictions.shape[0] < 2:
        raise ValueError("At least two aligned seed predictions are required")
    if seed_predictions.shape[1] != len(base):
        raise ValueError("Base and seed predictions have different row counts")
    ensemble = seed_predictions.mean(axis=0)
    directions = seed_predictions - base[None, :]
    agreement = np.all(directions > 0.0, axis=0) | np.all(directions < 0.0, axis=0)
    uncertainty = np.ptp(seed_predictions, axis=0) / CAPACITY
    disagreement = np.abs(ensemble - base) / CAPACITY
    mask = np.ones(len(base), dtype=bool)
    if policy.require_seed_agreement:
        mask &= agreement
    if policy.max_seed_uncertainty is not None:
        mask &= uncertainty <= policy.max_seed_uncertainty
    if policy.max_base_disagreement is not None:
        mask &= disagreement <= policy.max_base_disagreement
    return mask, {
        "seed_agreement": agreement,
        "seed_uncertainty": uncertainty,
        "base_disagreement": disagreement,
    }


def apply_blend(
    base: np.ndarray,
    member: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    base = np.asarray(base, dtype=float)
    member = np.asarray(member, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if base.shape != member.shape or base.shape != mask.shape:
        raise ValueError("base, member, and mask must have identical shapes")
    output = base.copy()
    output[mask] = np.clip(
        base[mask] + alpha * (member[mask] - base[mask]), 0.0, CAPACITY
    )
    return output


def issue_period_masks(
    timestamps: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """Make quarter masks from issue-cycle centres so no NWP run is split."""
    month_blocks, _ = assign_issue_blocks(timestamps, issue_times)
    masks = {
        name: np.isin(month_blocks, np.asarray(sorted(months)))
        for name, months in PERIOD_MONTHS.items()
    }
    issue_values = np.asarray(issue_times)
    for left_name, left in masks.items():
        for right_name, right in masks.items():
            if left_name >= right_name:
                continue
            overlap = set(issue_values[left]).intersection(issue_values[right])
            if overlap:
                raise AssertionError(
                    f"Issue cycles overlap between {left_name} and {right_name}"
                )
    return masks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_retained_file(path: Path) -> None:
    stale_roots = {"artifacts", "artifacts_cross_group", "artifacts_spatiotemporal"}
    if path.parts and path.parts[0] in stale_roots:
        raise ValueError(f"Stale artifact path is forbidden: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)


def _assert_submission_output(path: Path) -> None:
    if path.parent.name != "submissions":
        raise ValueError("Submission candidates must be written directly under submissions/")


def _align_seed_predictions(
    exact_index: pd.DatetimeIndex,
    prediction_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    seed_predictions: list[np.ndarray] = []
    common = np.ones(len(exact_index), dtype=bool)
    loaded: list[tuple[pd.DatetimeIndex, np.ndarray]] = []
    for path in prediction_paths:
        cache = np.load(path, allow_pickle=True)
        timestamps = pd.DatetimeIndex(pd.to_datetime(cache["timestamps_ns"]))
        if timestamps.has_duplicates:
            raise ValueError(f"Duplicate neural timestamps in {path}")
        prediction = cache["prediction"].astype(float)
        if prediction.shape != (len(timestamps), len(TARGETS)):
            raise ValueError(f"Unexpected prediction shape in {path}: {prediction.shape}")
        loaded.append((timestamps, prediction))
        common &= exact_index.isin(timestamps)
    aligned_index = exact_index[common]
    if len(aligned_index) < 0.99 * len(exact_index):
        raise ValueError("Neural caches cover less than 99% of exact OOF timestamps")
    for timestamps, prediction in loaded:
        positions = timestamps.get_indexer(aligned_index)
        if (positions < 0).any():
            raise AssertionError("Prediction alignment failed after intersection")
        seed_predictions.append(prediction[positions, TARGETS.index(TARGET)] * CAPACITY)
    return common, np.asarray(seed_predictions)


def _rolling_finesweep_base(
    labels_path: Path,
    driver_cache_path: Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    (
        labels,
        index,
        truth,
        _group_1,
        _group_2,
        _base,
        member,
        current,
        features,
        action,
    ) = _prepare_validation(labels_path, driver_cache_path)
    q1_timestamp = index < Q2_START
    q2_timestamp = (index >= Q2_START) & (index < H2_START)
    h1_timestamp = index < H2_START
    h2_timestamp = index >= H2_START

    q2_probability, _ = fit_probabilities(
        features,
        settlement_benefit_labels(truth, current, member, q1_timestamp),
        q1_timestamp & action,
    )
    q2_fine, _ = apply_meta_gate(
        current,
        member,
        action,
        q2_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    h2_probability, _ = fit_probabilities(
        features,
        settlement_benefit_labels(truth, current, member, h1_timestamp),
        h1_timestamp & action,
    )
    h2_fine, _ = apply_meta_gate(
        current,
        member,
        action,
        h2_probability,
        threshold=FINE_THRESHOLD,
        extra_alpha=FINE_ALPHA,
    )
    rolling = current.copy()
    rolling[q2_timestamp] = q2_fine[q2_timestamp]
    rolling[h2_timestamp] = h2_fine[h2_timestamp]
    return labels, index, truth, rolling


def _simple_seed_metrics(
    truth: np.ndarray,
    base: np.ndarray,
    seed_predictions: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    rows: np.ndarray,
) -> list[dict[str, Any]]:
    records = []
    for seed_index, prediction in enumerate(seed_predictions):
        candidate = apply_blend(base, prediction, mask, alpha)
        records.append(
            {
                "seed_index": seed_index,
                "metrics": _evaluate_period(truth, base, candidate, rows),
            }
        )
    return records


def _record_eligible(record: dict[str, Any]) -> bool:
    for period in ("q1", "q2"):
        if not record[period]["ensemble_blocked"]["robustness_passed"]:
            return False
        for seed_record in record[period]["seed_metrics"]:
            delta = seed_record["metrics"]["delta"]
            if not all(delta[key] > 0.0 for key in ("score", "one_minus_nmae", "ficr")):
                return False
    return True


def select_development_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select solely from Q1/Q2 records; H2 is not accepted by this function."""
    if any("h2" in record for record in records):
        raise ValueError("Development policy records must not contain locked H2 results")
    eligible = [
        record
        for record in records
        if record.get("deployable_from_retained_test_artifacts", True)
        and _record_eligible(record)
    ]
    if not eligible:
        raise RuntimeError("No policy passed the Q1/Q2 development contract")
    return max(
        eligible,
        key=lambda record: (
            min(
                record["q1"]["ensemble_blocked"]["overall"]["delta"]["score"],
                record["q2"]["ensemble_blocked"]["overall"]["delta"]["score"],
            ),
            -record["policy"]["alpha"],
            -record["coverage_ratio"],
        ),
    )


def _evaluate_selected_seed_blocks(
    truth: np.ndarray,
    base: np.ndarray,
    seed_predictions: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    timestamps: pd.DatetimeIndex,
    issue_times: pd.DatetimeIndex,
    rows: np.ndarray,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    output = []
    for seed_index, prediction in enumerate(seed_predictions):
        candidate = apply_blend(base, prediction, mask, alpha)
        output.append(
            {
                "seed_index": seed_index,
                "blocked": evaluate_blocked_rolling(
                    truth,
                    base,
                    candidate,
                    timestamps,
                    issue_times,
                    rows,
                    n_bootstrap,
                    seed=20260718 + seed_index,
                ),
            }
        )
    return output


def _make_final_candidate(
    final_member_path: Path,
    artifact_dir: Path,
    base_submission_path: Path,
    output_path: Path,
    policy: BlendPolicy,
    seeds: list[int],
    epochs_per_seed: list[int],
) -> dict[str, Any]:
    _assert_submission_output(output_path)
    base = pd.read_csv(base_submission_path, encoding="utf-8-sig")
    member = pd.read_csv(final_member_path, encoding="utf-8-sig")
    if not base[KEY_COLUMNS].equals(member[KEY_COLUMNS]):
        raise ValueError("Retained final neural member keys do not match the finesweep base")
    base_group3 = base[TARGET].to_numpy(dtype=float)
    ensemble = member[TARGET].to_numpy(dtype=float)
    if policy.require_seed_agreement or policy.max_seed_uncertainty is not None:
        raise ValueError(
            "The retained final artifact contains only the frozen two-seed ensemble; "
            "a seed-specific deployment policy is not reproducible without retraining"
        )
    mask = np.ones(len(base), dtype=bool)
    if policy.max_base_disagreement is not None:
        mask &= np.abs(ensemble - base_group3) / CAPACITY <= policy.max_base_disagreement
    candidate_group3 = apply_blend(base_group3, ensemble, mask, policy.alpha)
    output = base.copy()
    output[TARGET] = candidate_group3
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    staging_path = artifact_dir / "candidate_guard_staging.csv"
    output.to_csv(staging_path, index=False, encoding="utf-8-sig")
    config = load_config(Path.cwd())
    staging_audit = CandidateValidator(config).audit(staging_path)
    if not staging_audit.valid:
        raise RuntimeError(f"CandidateValidator rejected staging file: {staging_audit.errors}")
    staging_path.replace(output_path)
    final_audit = CandidateValidator(config).audit(output_path)
    if not final_audit.valid:
        raise RuntimeError(f"CandidateValidator rejected final file: {final_audit.errors}")

    movement = candidate_group3 - base_group3
    return {
        "output": str(output_path),
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "base": str(base_submission_path),
        "base_sha256": _sha256(base_submission_path),
        "seeds": seeds,
        "epochs_per_seed": epochs_per_seed,
        "changed_rows": int(mask.sum()),
        "changed_ratio": float(mask.mean()),
        "mean_absolute_movement_kwh": float(np.abs(movement).mean()),
        "p95_absolute_movement_kwh": float(np.quantile(np.abs(movement), 0.95)),
        "max_absolute_movement_kwh": float(np.abs(movement).max()),
        "final_member": str(final_member_path),
        "final_member_sha256": _sha256(final_member_path),
        "seed_agreement_ratio": None,
        "groups_1_2_unchanged": bool(
            np.array_equal(output["kpx_group_1"], base["kpx_group_1"])
            and np.array_equal(output["kpx_group_2"], base["kpx_group_2"])
        ),
        "candidate_validator": final_audit.to_dict(),
        "structural_model_contract": {
            "service_family": SERVICE_FAMILY,
            "policy_class": "global structural-model blend",
            "changed_ratio_expected": 1.0,
            "micro_correction_movement_cap_applicable": False,
            "reason": (
                "The 25% movement cap governs selective calibration patches; this candidate "
                "promotes an independently validated structural NWP graph model."
            ),
        },
    }


def run(
    labels_path: Path,
    driver_cache_path: Path,
    issue_source_path: Path,
    seed_prediction_paths: list[Path],
    seed_report_paths: list[Path],
    final_member_path: Path,
    base_submission_path: Path,
    output_path: Path,
    artifact_dir: Path,
    n_bootstrap: int,
    write_submission_if_qualified: bool,
) -> dict[str, Any]:
    for path in (
        labels_path,
        driver_cache_path,
        issue_source_path,
        base_submission_path,
        final_member_path,
        *seed_prediction_paths,
        *seed_report_paths,
    ):
        _require_retained_file(path)
    if len(seed_prediction_paths) != 2 or len(seed_report_paths) != 2:
        raise ValueError("The frozen promotion contract requires exactly seeds 17 and 29")
    if write_submission_if_qualified:
        _assert_submission_output(output_path)

    labels, exact_index, exact_truth, rolling_base = _rolling_finesweep_base(
        labels_path, driver_cache_path
    )
    del labels
    common, seed_predictions = _align_seed_predictions(exact_index, seed_prediction_paths)
    index = exact_index[common]
    truth = exact_truth[common]
    base = rolling_base[common]
    issue_times = load_issue_times(issue_source_path, index)
    periods = issue_period_masks(index, issue_times)
    ensemble = seed_predictions.mean(axis=0)

    policies = default_policies()
    development_records: list[dict[str, Any]] = []
    for policy in policies:
        mask, diagnostics = policy_mask(base, seed_predictions, policy)
        candidate = apply_blend(base, ensemble, mask, policy.alpha)
        record: dict[str, Any] = {
            "policy": policy.to_dict(),
            # Final training intentionally retained only the frozen two-seed
            # ensemble. Consensus/bounded families remain honest validation
            # comparisons, but cannot be reproduced on test without duplicate
            # training. They are therefore excluded before Q1/Q2 selection.
            "deployable_from_retained_test_artifacts": policy.family == "global",
            "coverage_rows": int(mask.sum()),
            "coverage_ratio": float(mask.mean()),
            "seed_agreement_ratio": float(diagnostics["seed_agreement"].mean()),
        }
        for period_name in ("q1", "q2"):
            rows = periods[period_name]
            record[period_name] = {
                "ensemble_blocked": evaluate_blocked_rolling(
                    truth,
                    base,
                    candidate,
                    index,
                    issue_times,
                    rows,
                    n_bootstrap,
                    seed=20260718,
                ),
                "seed_metrics": _simple_seed_metrics(
                    truth,
                    base,
                    seed_predictions,
                    mask,
                    policy.alpha,
                    rows,
                ),
            }
        development_records.append(record)

    selected_development = select_development_policy(development_records)
    selected_policy_dict = selected_development["policy"]
    selected_policy = next(policy for policy in policies if policy.name == selected_policy_dict["name"])
    selected_mask, selected_diagnostics = policy_mask(base, seed_predictions, selected_policy)
    selected_candidate = apply_blend(base, ensemble, selected_mask, selected_policy.alpha)

    # The policy is frozen above. H2 appears for the first time below and is
    # evaluated once; it is never fed back into policy selection.
    h2_ensemble = evaluate_blocked_rolling(
        truth,
        base,
        selected_candidate,
        index,
        issue_times,
        periods["h2"],
        n_bootstrap,
        seed=20260718,
    )
    seed_blocked = {
        name: _evaluate_selected_seed_blocks(
            truth,
            base,
            seed_predictions,
            selected_mask,
            selected_policy.alpha,
            index,
            issue_times,
            periods[name],
            n_bootstrap,
        )
        for name in ("q1", "q2", "h2")
    }
    all_seed_components_positive = all(
        all(
            row["blocked"]["overall"]["delta"][key] > 0.0
            for key in ("score", "one_minus_nmae", "ficr")
        )
        for period_records in seed_blocked.values()
        for row in period_records
    )
    locked_h2_seed_blocks_pass = all(
        row["blocked"]["robustness_passed"] for row in seed_blocked["h2"]
    )
    qualified = bool(
        selected_development["q1"]["ensemble_blocked"]["robustness_passed"]
        and selected_development["q2"]["ensemble_blocked"]["robustness_passed"]
        and h2_ensemble["robustness_passed"]
        and all_seed_components_positive
    )

    architecture_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in seed_report_paths
    ]
    seeds = [int(report["architecture"]["seeds"][0]) for report in architecture_reports]
    epochs_per_seed = [
        int(report["architecture"]["best_epochs"][0]) for report in architecture_reports
    ]
    if seeds != [17, 29]:
        raise ValueError(f"Frozen seed order changed: {seeds}")
    architecture = architecture_reports[0]["architecture"]

    report: dict[str, Any] = {
        "method": "two-seed spatiotemporal graph multitask consensus promotion",
        "service_family": SERVICE_FAMILY,
        "lineage": {
            "driver_cache": str(driver_cache_path),
            "driver_cache_sha256": _sha256(driver_cache_path),
            "neural_prediction_caches": [str(path) for path in seed_prediction_paths],
            "neural_prediction_sha256": [_sha256(path) for path in seed_prediction_paths],
            "public_best_base": str(base_submission_path),
            "public_best_base_sha256": _sha256(base_submission_path),
            "retained_final_member": str(final_member_path),
            "retained_final_member_sha256": _sha256(final_member_path),
            "excluded_unaligned_exact_rows": int((~common).sum()),
        },
        "selection_contract": {
            "policy_grid_frozen_before_h2": True,
            "development_only": ["2024-Q1", "2024-Q2"],
            "locked_confirmation": "2024-H2 evaluated once after policy selection",
            "policy_count": len(policies),
            "families": sorted({policy.family for policy in policies}),
            "alphas": sorted({policy.alpha for policy in policies}),
            "dependency_unit": "complete data_available_kst_dtm NWP issue cycles",
            "bootstrap": "season-stratified complete issue-cycle resampling",
            "requirements": [
                "Q1 and Q2 ensemble blocked-rolling robustness pass",
                "Q1 and Q2 score, 1-NMAE, and FICR positive for each neural seed",
                "locked H2 ensemble blocked-rolling robustness pass",
                "each seed has positive score, 1-NMAE, and FICR in Q1/Q2/H2",
                "monthly/seasonal/bootstrap robustness is enforced on the deployable two-seed ensemble",
            ],
            "base_note": (
                "Q1 uses the pre-meta exact rolling base because no prior exact 2023 meta fold exists; "
                "Q2 uses Q1-fitted finesweep and H2 uses H1-fitted finesweep."
            ),
        },
        "development_records": development_records,
        "selected_development": selected_development,
        "selected_policy": selected_policy.to_dict(),
        "selected_gate_diagnostics": {
            "coverage_rows": int(selected_mask.sum()),
            "coverage_ratio": float(selected_mask.mean()),
            "seed_agreement_ratio": float(selected_diagnostics["seed_agreement"].mean()),
            "median_seed_uncertainty_capacity_ratio": float(
                np.median(selected_diagnostics["seed_uncertainty"])
            ),
            "median_base_disagreement_capacity_ratio": float(
                np.median(selected_diagnostics["base_disagreement"])
            ),
        },
        "selected_seed_blocked": seed_blocked,
        "locked_h2_ensemble": h2_ensemble,
        "qualification": {
            "qualified": qualified,
            "all_seed_components_positive": all_seed_components_positive,
            "locked_h2_seed_blocks_pass": locked_h2_seed_blocks_pass,
            "locked_h2_seed_blocks_pass_is_diagnostic_only": True,
            "submission_requested": write_submission_if_qualified,
            "submission_created": False,
        },
        "submission": None,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "spatiotemporal_consensus_promotion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if qualified and write_submission_if_qualified:
        report["submission"] = _make_final_candidate(
            final_member_path=final_member_path,
            artifact_dir=artifact_dir,
            base_submission_path=base_submission_path,
            output_path=output_path,
            policy=selected_policy,
            seeds=seeds,
            epochs_per_seed=epochs_per_seed,
        )
        report["qualification"]["submission_created"] = True
    report["decision"] = (
        "Qualified and generated a submissions/ candidate from the frozen Q1/Q2 policy."
        if report["qualification"]["submission_created"]
        else (
            "Qualified, but candidate writing was not requested."
            if qualified
            else "Blocked: no submission was generated because a strict promotion gate failed."
        )
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument("--issue-source", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--seed-predictions",
        nargs=2,
        default=[
            "artifacts_final/spatiotemporal/validation_predictions.npz",
            "artifacts_final/spatiotemporal_seed29/validation_predictions.npz",
        ],
    )
    parser.add_argument(
        "--seed-reports",
        nargs=2,
        default=[
            "artifacts_final/spatiotemporal/validation_report.json",
            "artifacts_final/spatiotemporal_seed29/validation_report.json",
        ],
    )
    parser.add_argument(
        "--final-member",
        default="artifacts_final/spatiotemporal_final/spatiotemporal_member.csv",
    )
    parser.add_argument(
        "--base-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--output",
        default="submissions/blend_best_spatiotemporal_multitask20.csv",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts_final/spatiotemporal_consensus"
    )
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--write-submission-if-qualified", action="store_true")
    args = parser.parse_args()
    report = run(
        labels_path=Path(args.labels),
        driver_cache_path=Path(args.driver_cache),
        issue_source_path=Path(args.issue_source),
        seed_prediction_paths=[Path(path) for path in args.seed_predictions],
        seed_report_paths=[Path(path) for path in args.seed_reports],
        final_member_path=Path(args.final_member),
        base_submission_path=Path(args.base_submission),
        output_path=Path(args.output),
        artifact_dir=Path(args.artifact_dir),
        n_bootstrap=args.n_bootstrap,
        write_submission_if_qualified=args.write_submission_if_qualified,
    )
    print(
        json.dumps(
            {
                "selected_policy": report["selected_policy"],
                "locked_h2": report["locked_h2_ensemble"]["overall"],
                "qualification": report["qualification"],
                "submission": report["submission"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

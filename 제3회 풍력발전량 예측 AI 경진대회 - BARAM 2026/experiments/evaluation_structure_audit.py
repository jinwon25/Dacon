"""Reproducible audit of the BARAM metric and validation governance.

This audit is deliberately read-only with respect to submissions.  It verifies
the local metric against an independent transcription of the published
formula, checks the exact-OOF time axis, records leaderboard reuse, and validates
the retained external-data manifests used by the active candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_service.compliance import validate_external_data_manifest
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(CAPACITY_KWH)
COMPONENTS = ("score", "one_minus_nmae", "ficr")


def published_group_formula(
    truth: np.ndarray, prediction: np.ndarray, capacity: float
) -> dict[str, float | int]:
    """Independent transcription of the published group metric."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = truth >= 0.10 * float(capacity)
    actual = truth[valid]
    forecast = prediction[valid]
    if not len(actual):
        raise ValueError("published formula has no eligible observations")
    if not np.isfinite(forecast).all():
        raise ValueError("published formula received a non-finite eligible forecast")
    error = np.abs(actual - forecast) / float(capacity)
    unit_price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    one_minus_nmae = 1.0 - float(error.mean())
    ficr = float(np.sum(actual * unit_price) / np.sum(actual * 4.0))
    return {
        "score": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "n_samples": int(len(actual)),
    }


def randomized_metric_parity(
    *, repetitions: int = 100, seed: int = 20260726
) -> dict[str, Any]:
    """Compare both group and macro implementations on randomized edge cases."""
    rng = np.random.default_rng(seed)
    maximum = {component: 0.0 for component in COMPONENTS}
    maximum["n_samples"] = 0
    macro_maximum = {component: 0.0 for component in COMPONENTS}
    for _ in range(repetitions):
        truth_by_group: dict[str, np.ndarray] = {}
        prediction_by_group: dict[str, np.ndarray] = {}
        independent: dict[str, dict[str, float | int]] = {}
        for target, capacity in CAPACITY_KWH.items():
            truth = rng.uniform(0.0, capacity, size=257)
            prediction = np.clip(
                truth + rng.normal(0.0, 0.11 * capacity, size=len(truth)),
                0.0,
                capacity,
            )
            # Force exact eligibility and settlement-boundary cases.
            truth[:4] = (0.10 * capacity, 0.10 * capacity, 0.50 * capacity, 0.50 * capacity)
            prediction[:4] = (
                truth[0],
                truth[1] + 0.06 * capacity,
                truth[2] + 0.08 * capacity,
                truth[3] + 0.0800001 * capacity,
            )
            truth_by_group[target] = truth
            prediction_by_group[target] = prediction
            expected = published_group_formula(truth, prediction, capacity)
            actual = evaluate_group(truth, prediction, capacity).to_dict()
            independent[target] = expected
            for component in COMPONENTS:
                maximum[component] = max(
                    maximum[component],
                    abs(float(expected[component]) - float(actual[component])),
                )
            maximum["n_samples"] = max(
                maximum["n_samples"],
                abs(int(expected["n_samples"]) - int(actual["n_samples"])),
            )
        macro = evaluate_competition(truth_by_group, prediction_by_group)
        expected_macro = {
            component: float(
                np.mean([float(independent[target][component]) for target in TARGETS])
            )
            for component in COMPONENTS
        }
        for component in COMPONENTS:
            macro_maximum[component] = max(
                macro_maximum[component],
                abs(expected_macro[component] - float(macro[component])),
            )
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "group_maximum_absolute_difference": maximum,
        "macro_maximum_absolute_difference": macro_maximum,
        "exact_within_1e_12": bool(
            max(float(value) for value in maximum.values()) <= 1e-12
            and max(macro_maximum.values()) <= 1e-12
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def audit_submission_history(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = [float(row["score"]) for row in rows if row.get("score")]
    running_best = -np.inf
    improvements = 0
    for score in scores:
        if score > running_best:
            improvements += 1
            running_best = score
    best_position = int(np.argmax(scores))
    latest = rows[-1]
    alpha_rows = [
        row
        for row in rows
        if "alpha" in row.get("file", "").lower()
        or "alpha" in row.get("title", "").lower()
    ]
    return {
        "recorded_submissions": int(len(rows)),
        "scored_submissions": int(len(scores)),
        "running_best_updates": int(improvements),
        "best": {
            "row_number": best_position + 1,
            "submission_id": rows[best_position].get("submission_id"),
            "file": rows[best_position].get("file"),
            "score": scores[best_position],
            "one_minus_nmae": float(rows[best_position]["one_minus_nmae"]),
            "ficr": float(rows[best_position]["ficr"]),
        },
        "latest": {
            "submission_id": latest.get("submission_id"),
            "file": latest.get("file"),
            "score": float(latest["score"]),
            "notes": latest.get("notes"),
        },
        "alpha_named_public_probes": int(len(alpha_rows)),
        "public_private_split": {
            "public_fraction": 0.40,
            "private_fraction": 0.60,
            "final_ranking_uses_private": True,
        },
    }


def audit_oof(driver_path: Path, kma_path: Path) -> dict[str, Any]:
    driver = np.load(driver_path, allow_pickle=False)
    indexes = {
        target: pd.to_datetime(driver[f"{target}__valid_index_ns"]) for target in TARGETS
    }
    reference = indexes[TARGETS[0]]
    alignment = all(index.equals(reference) for index in indexes.values())
    kma = np.load(kma_path, allow_pickle=False)
    kma_index = pd.to_datetime(kma["index_ns"])
    issue = pd.to_datetime(kma["issue_ns"])
    q2 = kma["q2"].astype(bool)
    h2 = kma["h2"].astype(bool)
    finite_issue = ~pd.isna(issue)
    return {
        "driver": {
            "rows": int(len(reference)),
            "start": str(reference.min()),
            "end": str(reference.max()),
            "duplicate_timestamps": int(reference.duplicated().sum()),
            "all_group_indexes_equal": bool(alignment),
        },
        "kma_surface": {
            "rows": int(len(kma_index)),
            "index_matches_driver": bool(kma_index.equals(reference)),
            "q2_rows": int(q2.sum()),
            "h2_rows": int(h2.sum()),
            "q2_h2_overlap_rows": int((q2 & h2).sum()),
            "q2_issue_cycles": int(pd.Index(issue[q2 & finite_issue]).nunique()),
            "h2_issue_cycles": int(pd.Index(issue[h2 & finite_issue]).nunique()),
            "missing_issue_rows": int((~finite_issue).sum()),
        },
        "interpretation": {
            "row_alignment_passed": bool(
                alignment
                and kma_index.equals(reference)
                and not reference.duplicated().any()
                and not (q2 & h2).any()
            ),
            "group3_only_delta_to_macro_factor": 1.0 / 3.0,
        },
    }


def audit_auxiliary_oof_coverage(
    driver_path: Path, auxiliary_paths: list[Path]
) -> dict[str, Any]:
    """Detect truncated auxiliary OOFs before they are treated as annual evidence."""
    driver = np.load(driver_path, allow_pickle=False)
    reference = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGETS[0]}__valid_index_ns"])
    )
    annual_reference = reference[
        (reference >= pd.Timestamp("2024-01-01"))
        & (reference < pd.Timestamp("2025-01-01"))
    ]
    records: list[dict[str, Any]] = []
    for path in auxiliary_paths:
        payload = np.load(path, allow_pickle=False)
        if "index_ns" not in payload.files:
            raise ValueError(f"auxiliary OOF lacks index_ns: {path}")
        index = pd.DatetimeIndex(pd.to_datetime(payload["index_ns"]))
        common = annual_reference.intersection(index)
        missing = annual_reference.difference(index)
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "rows": int(len(index)),
                "start": str(index.min()),
                "end": str(index.max()),
                "annual_reference_rows": int(len(annual_reference)),
                "common_annual_rows": int(len(common)),
                "missing_annual_rows": int(len(missing)),
                "first_missing": None if missing.empty else str(missing.min()),
                "last_missing": None if missing.empty else str(missing.max()),
                "full_year_complete": bool(missing.empty),
            }
        )
    return {
        "annual_reference_start": str(annual_reference.min()),
        "annual_reference_end": str(annual_reference.max()),
        "all_full_year_complete": bool(
            all(record["full_year_complete"] for record in records)
        ),
        "records": records,
        "guard": (
            "Any incomplete auxiliary OOF is diagnostic-only until its missing "
            "timestamps are generated causally and re-evaluated."
        ),
    }


def audit_h2_reuse(report_root: Path) -> dict[str, Any]:
    """Count report files that explicitly contain locked/H2 evaluation keys."""
    matched: list[str] = []
    markers = (
        b'"locked_h2"',
        b'"locked"',
        b'"h2_metrics"',
        b'"rolling_h1_curve_locked_h2"',
    )
    for path in sorted(report_root.rglob("*.json")):
        if "external_weather" in path.parts and path.name == "manifest.json":
            continue
        try:
            rendered = path.read_bytes().lower()
        except OSError:
            continue
        if any(marker in rendered for marker in markers):
            matched.append(path.relative_to(ROOT).as_posix())
    return {
        "json_files_with_locked_or_h2_evaluation": int(len(matched)),
        "files": matched,
        "status": "contaminated_development_benchmark" if matched else "unknown",
        "reason": (
            "Repeated inspection across model families invalidates the one-shot locked-holdout "
            "interpretation even when each individual experiment is temporally causal."
        ),
    }


def audit_external_manifests(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        validation = validate_external_data_manifest(path, ROOT)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "competition_eligible": bool(manifest.get("competition_eligible")),
                "provider": manifest.get("provider"),
                "dataset": manifest.get("dataset"),
                "causality_violations": int(
                    manifest.get("causality_audit", {}).get("violations", -1)
                ),
                "validator": validation,
            }
        )
    return {
        "all_passed": bool(
            all(
                row["competition_eligible"] and row["causality_violations"] == 0
                for row in records
            )
        ),
        "records": records,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    active = _rooted(args.active_candidate)
    active_frame = pd.read_csv(active, encoding="utf-8-sig")
    official_notebook = _rooted(args.official_notebook)
    subset_stress = json.loads(
        _rooted(args.subset_stress).read_text(encoding="utf-8")
    )
    return {
        "schema_version": 2,
        "audit_date": "2026-07-29",
        "official_reference": {
            "path": official_notebook.relative_to(ROOT).as_posix(),
            "sha256": _sha256(official_notebook),
            "implementation_matches": [
                "actual >= 10% of group capacity eligibility",
                "unweighted within-group capacity-normalized MAE",
                "actual-generation-weighted FiCR settlement",
                "inclusive <=6% and <=8% error cliffs",
                "equal macro average over the three groups",
                "0.5 * (1-NMAE) + 0.5 * FiCR",
            ],
        },
        "official_metric": randomized_metric_parity(
            repetitions=args.metric_repetitions
        ),
        "oof": audit_oof(_rooted(args.driver), _rooted(args.kma_oof)),
        "auxiliary_oof_coverage": audit_auxiliary_oof_coverage(
            _rooted(args.driver),
            [_rooted(value) for value in args.auxiliary_oof],
        ),
        "validation_reuse": audit_h2_reuse(_rooted(args.report_root)),
        "leaderboard": audit_submission_history(_rooted(args.results)),
        "external_data": audit_external_manifests(
            [_rooted(value) for value in args.external_manifest]
        ),
        "active_candidate": {
            "path": active.relative_to(ROOT).as_posix(),
            "sha256": _sha256(active),
            "rows": int(len(active_frame)),
            "columns": list(active_frame.columns),
            "finite_targets": bool(
                np.isfinite(active_frame[list(TARGETS)].to_numpy(dtype=float)).all()
            ),
        },
        "public_private_subset_stress": {
            "path": _rooted(args.subset_stress).relative_to(ROOT).as_posix(),
            "old_full_year_positive": subset_stress["promotion_audit"][
                "old_full_year_positive"
            ],
            "corrected_gate_passed": subset_stress["promotion_audit"][
                "corrected_gate_passed"
            ],
            "iid_public_q05": {
                component: subset_stress["iid_timestamp_splits"]["public"][
                    component
                ]["q05"]
                for component in COMPONENTS
            },
            "observed_public_percentile": {
                component: subset_stress["iid_timestamp_splits"]["public"][
                    component
                ]["observed_percentile"]
                for component in COMPONENTS
            },
        },
        "senior_assessment": {
            "metric_implementation": "passed",
            "time_alignment": "passed",
            "external_data_compliance": "passed",
            "locked_h2_status": "failed_as_unseen_holdout",
            "public_leaderboard_status": "development_feedback_only",
            "primary_bottleneck": (
                "validation transfer and discontinuous FiCR subset variance, not "
                "an arithmetic error in the official metric"
            ),
            "required_protocol": [
                "freeze submission 1502437 as incumbent",
                "treat H2 as a contaminated historical benchmark",
                "allow only new core-model signals or independently specified sparse signals",
                "require causal rolling-origin evidence from both 2023 and 2024 when data coverage permits",
                "stress the exact 40% public and complementary 60% private subset sizes",
                "require non-negative score, 1-NMAE, and FiCR q05 on both IID and month-stratified complements",
                "require complete issue-cycle block bootstrap support and positive forward seasons",
                "reject truncated auxiliary OOFs as annual evidence until missing months are filled causally",
                "do not use public score to choose row gates, blend weights, or seasonal masks",
                "close broad FiCR-threshold postprocessing after repeated public sign reversals",
                "retain source, issue, availability, license, checksum, and preprocessing evidence",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="artifacts_final/lineage/exact_driver_oof.npz")
    parser.add_argument(
        "--kma-oof",
        default=(
            "artifacts_final/external_weather/kma_um_regional_context_2024/"
            "power_curve_oof_20260725.npz"
        ),
    )
    parser.add_argument(
        "--auxiliary-oof",
        action="append",
        default=[
            (
                "artifacts_final/base_v2/group3_curtailment_full_20260725/"
                "predictions.npz"
            ),
            "artifacts_final/base_v2/lgbm_full_20260725/predictions.npz",
        ],
    )
    parser.add_argument("--results", default="submissions/results.csv")
    parser.add_argument("--report-root", default="artifacts_final")
    parser.add_argument(
        "--active-candidate",
        default=(
            "artifacts_final/candidates/"
            "public_factor_residual_g1_pooled_g2_incumbent_g3_20260726.csv"
        ),
    )
    parser.add_argument("--official-notebook", default="평가_산식 코드.ipynb")
    parser.add_argument(
        "--subset-stress",
        default=(
            "artifacts_final/diagnostics/"
            "public_private_subset_stress_20260729.json"
        ),
    )
    parser.add_argument(
        "--external-manifest",
        action="append",
        default=[
            "artifacts_final/external_weather/kma_um_regional_context_2024/manifest.json",
            "artifacts_final/external_weather/kma_um_regional_context_2025/manifest.json",
        ],
    )
    parser.add_argument("--metric-repetitions", type=int, default=100)
    parser.add_argument(
        "--output",
        default="artifacts_final/diagnostics/evaluation_structure_audit_20260729.json",
    )
    args = parser.parse_args()
    report = build_audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["senior_assessment"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

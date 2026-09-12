"""Calendar-stable single-IVB TrackMan residual candidate above public 1158.

V69 found that a source-only single-feature model transferred positively in
both forward transitions and all three anonymous domains, but reversed at the
small season-boundary months.  This experiment declares one coarse baseball
calendar rule: fit and apply only April through September.  It does not tune a
weight per month and remains row-local because the current row's month is an
official feature.

The 2024 audits have been reused during development, so even a pass here is a
research candidate rather than automatic packaging authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes
from src.archive.v68_trackman_profile_ridge import (
    FEATURE_SETS,
    apply_pitcher_correction,
    fit_source_only_ridge,
    profile_training_table,
)


ACTIVE_MONTHS = (4, 5, 6, 7, 8, 9)
FEATURE_SET = "single_ivb"
DOMAIN = "ALL"


def calendar_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame["game_month"].isin(ACTIVE_MONTHS).to_numpy()


def calendar_transition(
    source_frame: pd.DataFrame,
    source_parent: np.ndarray,
    source_profile: pd.DataFrame,
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    audit_profile: pd.DataFrame,
) -> tuple[dict[str, object], np.ndarray]:
    source_mask = calendar_mask(source_frame)
    source = profile_training_table(
        source_frame.loc[source_mask].reset_index(drop=True),
        np.asarray(source_parent, dtype=np.float64)[source_mask],
        source_profile,
        domain=DOMAIN,
    )
    correction, fit_audit = fit_source_only_ridge(
        source, audit_profile, FEATURE_SETS[FEATURE_SET]
    )
    candidate, profile_active, row_correction = apply_pitcher_correction(
        audit_frame,
        audit_parent,
        audit_profile,
        correction,
        domain=DOMAIN,
    )
    active = profile_active & calendar_mask(audit_frame)
    candidate[~active] = np.asarray(audit_parent, dtype=np.float64)[~active]
    result = diagnostics(audit_frame, audit_parent, candidate, active)
    result.update(
        {
            **fit_audit,
            "source_rows_before_calendar_gate": int(len(source_frame)),
            "source_rows_after_calendar_gate": int(source_mask.sum()),
            "active_rows": int(active.sum()),
            "active_fraction": float(active.mean()),
            "correction_sd": float(np.std(row_correction[active]))
            if active.any()
            else 0.0,
            "correction_max_abs": float(np.max(np.abs(row_correction[active])))
            if active.any()
            else 0.0,
        }
    )
    return result, candidate


def run(
    project: Path,
    profiles_path: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    profiles_path = profiles_path.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    profiles = pd.read_csv(profiles_path, low_memory=False)
    profile23 = profiles.loc[profiles["origin"].eq(2023)].drop(columns="origin")
    profile24 = profiles.loc[profiles["origin"].eq(2024)].drop(columns="origin")
    axes = _cached_v25_axes(project, raw)
    late23 = axes["selection_late_2023"]
    full24 = axes["outer_full_2024"]
    replication24 = axes["replication_late_2024"]
    cache23 = np.load(current_oof_dir / "selection_late_2023.npz")
    cache24 = np.load(current_oof_dir / "outer_full_2024.npz")
    cache_rep = np.load(current_oof_dir / "replication_late_2024.npz")
    parent23 = cache23["final_gate_parent"].astype(np.float64)
    parent24 = cache24["final_gate_parent"].astype(np.float64)
    parent_rep = cache_rep["final_gate_parent"].astype(np.float64)
    early_mask = full24["game_month"].le(7).to_numpy()
    early24 = full24.loc[early_mask].reset_index(drop=True)
    parent_early24 = parent24[early_mask]

    full_result, full_candidate = calendar_transition(
        late23,
        parent23,
        profile23,
        full24,
        parent24,
        profile24,
    )
    replication_result, replication_candidate = calendar_transition(
        early24,
        parent_early24,
        profile24,
        replication24,
        parent_rep,
        profile24,
    )
    metrics = pd.DataFrame(
        [
            {"transition": "late23_to_full24", **full_result},
            {"transition": "early24_to_late24", **replication_result},
        ]
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "predictions.npz",
        full24=full_candidate,
        replication24=replication_candidate,
    )
    gates = {
        "both_transition_gains_positive": bool((metrics["gain"] > 0).all()),
        "both_month_fractions_one": bool(
            (metrics["positive_month_fraction"] == 1.0).all()
        ),
        "both_minimum_domains_nonnegative": bool(
            (metrics["minimum_domain_gain"] >= 0).all()
        ),
        "both_source_oof_eta_positive": bool(
            (metrics["source_oof_eta"] > 0).all()
        ),
    }
    result = {
        "protocol": "V70_SINGLE_IVB_APRIL_SEPTEMBER_GATE_ABOVE_1158_V1",
        "recipe": {
            "feature_set": FEATURE_SET,
            "feature": FEATURE_SETS[FEATURE_SET][0],
            "domain": DOMAIN,
            "active_months": list(ACTIVE_MONTHS),
            "ridge_and_eta": "source-only OOF",
        },
        "audits": {
            "late23_to_full24": full_result,
            "early24_to_late24": replication_result,
        },
        "gates": gates,
        "passes_local_gates": bool(all(gates.values())),
        "eligible_for_packaging": False,
        "decision": "hold_for_independent_validation_or_predeclared_public_probe",
        "reason_not_packaged": "calendar gate was formulated after inspecting reused 2024 audits",
        "row_local_inference": True,
        "other_test_rows_used": False,
        "test_distribution_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.profiles, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""Chronologically nested, identity-shrunk calibration screen for v27.

Calibration coefficients are fitted on late-2023 v27 OOF predictions.  Model,
ridge strength, deployment domain, and blend weight are selected on early-2024
only.  The selected transform is frozen before it is evaluated on late-2024
predictions produced by the corresponding early-2024 fit.

The fitted target is ``y - p`` and every coefficient, including the intercept,
is penalized toward zero.  Zero therefore means the frozen v27 identity map.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.v30_diverse_covariance_screen import (
    _cached_v25_axes,
    diagnostics,
    v27_parent,
)


FAMILIES = ("global", "slope", "beta", "domain", "domain_slope")
ALPHAS = (100.0, 500.0, 2_000.0, 10_000.0, 50_000.0)
ETAS = (0.25, 0.50, 0.75, 1.00)
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR")
EPSILON = 1e-4


@dataclass(frozen=True)
class CalibrationState:
    family: str
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]


def _raw_features(
    prediction: np.ndarray, domain: np.ndarray, family: str
) -> np.ndarray:
    probability = np.clip(
        np.asarray(prediction, dtype=np.float64), EPSILON, 1.0 - EPSILON
    )
    domain = np.asarray(domain, dtype=str)
    log_p = np.log(probability)
    log_one_minus_p = np.log1p(-probability)
    anchor = (domain == "R_ANCHOR").astype(np.float64)
    futures = (domain == "F").astype(np.float64)
    centered_probability = probability - 0.5
    if family == "global":
        return np.empty((len(probability), 0), dtype=np.float64)
    if family == "slope":
        return centered_probability[:, None]
    if family == "beta":
        return np.column_stack([log_p, log_one_minus_p])
    if family == "domain":
        return np.column_stack([anchor, futures])
    if family == "domain_slope":
        return np.column_stack(
            [
                centered_probability,
                anchor,
                futures,
                centered_probability * anchor,
                centered_probability * futures,
            ]
        )
    raise ValueError(f"unknown calibration family: {family}")


def fit_calibrator(
    target: np.ndarray,
    prediction: np.ndarray,
    domain: np.ndarray,
    *,
    family: str,
    alpha: float,
) -> CalibrationState:
    """Fit a ridge residual map whose exact prior is the identity transform."""
    raw = _raw_features(prediction, domain, family)
    if raw.shape[1]:
        means = raw.mean(axis=0)
        scales = raw.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        standardized = (raw - means) / scales
    else:
        means = np.empty(0, dtype=np.float64)
        scales = np.empty(0, dtype=np.float64)
        standardized = raw
    design = np.column_stack([np.ones(len(prediction)), standardized])
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    gram = design.T @ design
    penalty = float(alpha) * np.eye(design.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(gram + penalty, design.T @ residual)
    return CalibrationState(
        family=family,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
    )


def calibration_correction(
    prediction: np.ndarray, domain: np.ndarray, state: CalibrationState
) -> np.ndarray:
    raw = _raw_features(prediction, domain, state.family)
    if raw.shape[1]:
        standardized = (
            raw - np.asarray(state.means, dtype=np.float64)
        ) / np.asarray(state.scales, dtype=np.float64)
    else:
        standardized = raw
    design = np.column_stack([np.ones(len(prediction)), standardized])
    return design @ np.asarray(state.coefficients, dtype=np.float64)


def apply_calibrator(
    frame: pd.DataFrame,
    state: CalibrationState,
    *,
    eta: float,
    apply_domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    domain = frame["domain3"].astype(str).to_numpy()
    correction = calibration_correction(parent, domain, state)
    mask = (
        np.ones(len(frame), dtype=bool)
        if apply_domain == "ALL"
        else domain == apply_domain
    )
    candidate = parent.copy()
    candidate[mask] = np.clip(
        parent[mask] + float(eta) * correction[mask], 0.001, 0.999
    )
    return candidate, mask


def _state_dict(state: CalibrationState) -> dict[str, object]:
    return {
        "family": state.family,
        "means": list(state.means),
        "scales": list(state.scales),
        "coefficients": list(state.coefficients),
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    source = axes["selection_late_2023"]
    full_2024 = axes["outer_full_2024"]
    selection = full_2024.loc[full_2024["game_month"].le(7)].reset_index(drop=True)
    audit = axes["replication_late_2024"]
    expected = full_2024.loc[full_2024["game_month"].ge(8), "target"].to_numpy(
        np.float64
    )
    if not np.array_equal(expected, audit["target"].to_numpy(np.float64)):
        raise ValueError("late-2024 target alignment failed")

    source_target = source["target"].to_numpy(np.float64)
    source_parent = v27_parent(source)
    source_domain = source["domain3"].astype(str).to_numpy()
    selection_parent = v27_parent(selection)
    states: dict[tuple[str, float], CalibrationState] = {}
    rows: list[dict[str, object]] = []
    for family, alpha in itertools.product(FAMILIES, ALPHAS):
        state = fit_calibrator(
            source_target,
            source_parent,
            source_domain,
            family=family,
            alpha=alpha,
        )
        states[(family, alpha)] = state
        for apply_domain, eta in itertools.product(DOMAINS, ETAS):
            candidate, mask = apply_calibrator(
                selection,
                state,
                eta=eta,
                apply_domain=apply_domain,
            )
            result = diagnostics(selection, selection_parent, candidate, mask)
            rows.append(
                {
                    "family": family,
                    "alpha": alpha,
                    "apply_domain": apply_domain,
                    "eta": eta,
                    **{
                        name: value
                        for name, value in result.items()
                        if name not in {"months", "domain_gains"}
                    },
                }
            )
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = metrics[["gain", "worst_month_gain"]].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
        & metrics["minimum_domain_gain"].gt(-5.0)
    )
    metrics = metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = metrics.loc[metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else metrics.iloc[0]
    chosen = {
        "family": str(selected["family"]),
        "alpha": float(selected["alpha"]),
        "apply_domain": str(selected["apply_domain"]),
        "eta": float(selected["eta"]),
    }
    state = states[(chosen["family"], chosen["alpha"])]
    selection_candidate, selection_mask = apply_calibrator(
        selection,
        state,
        eta=chosen["eta"],
        apply_domain=chosen["apply_domain"],
    )
    audit_candidate, audit_mask = apply_calibrator(
        audit,
        state,
        eta=chosen["eta"],
        apply_domain=chosen["apply_domain"],
    )
    results = {
        "selection_early_2024": diagnostics(
            selection, selection_parent, selection_candidate, selection_mask
        ),
        "audit_late_2024": diagnostics(
            audit, v27_parent(audit), audit_candidate, audit_mask
        ),
    }
    gates = {
        "selection_gate": bool(selected["passes_selection_gate"]),
        "audit_gain_at_least_1": results["audit_late_2024"]["gain"] >= 1.0,
        "audit_all_months_positive": results["audit_late_2024"][
            "positive_month_fraction"
        ]
        == 1.0,
        "audit_worst_month_positive": results["audit_late_2024"][
            "worst_month_gain"
        ]
        > 0.0,
        "audit_minimum_domain_above_minus_5": results["audit_late_2024"][
            "minimum_domain_gain"
        ]
        > -5.0,
    }
    summary = {
        "protocol": "V34_TEMPORAL_V27_IDENTITY_SHRUNK_CALIBRATION_V1",
        "parent": "submit_v27.zip",
        "fit_axis": "late-2023 OOF",
        "selection_axis": "early-2024 OOF",
        "independent_audit": "late-2024 OOF with early-2024 direct-model fit",
        "candidate_count": int(len(metrics)),
        "selection_gate_count": int(metrics["passes_selection_gate"].sum()),
        "chosen": chosen,
        "fitted_state": _state_dict(state),
        "results": results,
        "gates": gates,
        "eligible": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "audit_late_2024.npz",
        target=audit["target"].to_numpy(np.float64),
        v27=v27_parent(audit),
        candidate=audit_candidate,
        domain3=audit["domain3"].astype(str).to_numpy(),
        game_month=audit["game_month"].to_numpy(np.int16),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v34_temporal_v27_calibration_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

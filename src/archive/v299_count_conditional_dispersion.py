"""Audit a prior-period, count-conditional dispersion correction for v290.

The incumbent is globally calibrated, but its within-count resolution can still
be too wide or too narrow.  This audit learns only the *difference* between each
count's residual slope and the pooled within-count residual slope.  Group means
are therefore protected by construction and no evaluation-batch aggregate is
needed at inference time.

The dose is selected on two source transfers (early-to-late 2022 and full-2022
to late-2023).  Late-2023 to full-2024 is opened once after the dose is frozen.
Only official training rows and row-local fields are used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.contract import _load_contract_axis


PROTOCOL = "V299_COUNT_CONDITIONAL_DISPERSION_V1"
TARGET = "control_success"
PRIOR_EQUIVALENT_ROWS = 20_000.0
CORRECTION_CAP = 0.025


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(target.mean() * (1.0 - target.mean()))
    return float(
        100000.0
        * (1.0 - np.mean(np.square(target - prediction)) / denominator)
    )


def count_labels(frame: pd.DataFrame) -> np.ndarray:
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int8)
    return np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str))


@dataclass(frozen=True)
class CountDispersionMap:
    centers: dict[str, float]
    coefficients: dict[str, float]
    pooled_coefficient: float


def fit_count_dispersion(
    labels: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    prior_equivalent_rows: float = PRIOR_EQUIVALENT_ROWS,
) -> CountDispersionMap:
    """Fit shrunk count-specific residual slopes around fixed source centers."""

    labels = np.asarray(labels).astype(str)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (len(labels) == len(prediction) == len(target) == len(fit_mask)):
        raise ValueError("count-dispersion inputs must have identical lengths")
    if not fit_mask.any():
        raise ValueError("fit mask is empty")

    centers: dict[str, float] = {}
    raw_numerators: dict[str, float] = {}
    raw_denominators: dict[str, float] = {}
    group_rows: dict[str, int] = {}
    pooled_num = 0.0
    pooled_den = 0.0
    for level in sorted(np.unique(labels[fit_mask])):
        mask = fit_mask & (labels == level)
        center = float(np.mean(prediction[mask]))
        direction = prediction[mask] - center
        numerator = float(np.dot(target[mask] - prediction[mask], direction))
        denominator = float(np.dot(direction, direction))
        centers[level] = center
        raw_numerators[level] = numerator
        raw_denominators[level] = denominator
        group_rows[level] = int(mask.sum())
        pooled_num += numerator
        pooled_den += denominator

    pooled = pooled_num / pooled_den if pooled_den > 0.0 else 0.0
    mean_square_direction = pooled_den / float(fit_mask.sum())
    ridge = float(prior_equivalent_rows) * mean_square_direction
    coefficients: dict[str, float] = {}
    for level in centers:
        denominator = raw_denominators[level]
        raw = raw_numerators[level] / denominator if denominator > 0.0 else pooled
        shrink = denominator / (denominator + ridge) if denominator + ridge > 0.0 else 0.0
        coefficients[level] = float(shrink * (raw - pooled))
    return CountDispersionMap(
        centers=centers,
        coefficients=coefficients,
        pooled_coefficient=float(pooled),
    )


def predict_correction(
    model: CountDispersionMap,
    labels: np.ndarray,
    prediction: np.ndarray,
    active: np.ndarray,
    cap: float = CORRECTION_CAP,
) -> np.ndarray:
    """Apply saved source constants row by row; unseen counts remain unchanged."""

    labels = np.asarray(labels).astype(str)
    prediction = np.asarray(prediction, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    correction = np.zeros(len(prediction), dtype=np.float64)
    for level, coefficient in model.coefficients.items():
        mask = active & (labels == level)
        correction[mask] = coefficient * (prediction[mask] - model.centers[level])
    return np.clip(correction, -float(cap), float(cap))


def paired_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    months = []
    for month in sorted(pd.unique(frame["game_month"])):
        mask = frame["game_month"].eq(month).to_numpy() & active
        if mask.any():
            months.append(
                {
                    "month": int(month),
                    "rows": int(mask.sum()),
                    "gain": bss(target[mask], candidate[mask])
                    - bss(target[mask], parent[mask]),
                }
            )
    return {
        "gain": bss(target, candidate) - bss(target, parent),
        "active_gain": (
            bss(target[active], candidate[active]) - bss(target[active], parent[active])
            if active.any()
            else 0.0
        ),
        "active_rows": int(active.sum()),
        "rms_shift_active": float(
            np.sqrt(np.mean(np.square(candidate[active] - parent[active])))
        ),
        "positive_month_fraction": float(
            np.mean([item["gain"] > 0.0 for item in months])
        ),
        "worst_month_gain": float(min(item["gain"] for item in months)),
        "months": months,
    }


def _domain_active(axis: dict[str, np.ndarray]) -> np.ndarray:
    return axis["exact_mask"].astype(bool) & (
        axis["domain3"].astype(str) == "R_CORE"
    )


def _compose(
    parent: np.ndarray, correction: np.ndarray, active: np.ndarray, dose: float
) -> np.ndarray:
    result = np.asarray(parent, dtype=np.float64).copy()
    result[active] = np.clip(
        result[active] + float(dose) * correction[active], 0.001, 0.999
    )
    return result


def run(
    train_csv: Path,
    contract_dir: Path,
    v285_axes: Path,
    v290_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[
            "season",
            "game_month",
            "balls_before",
            "strikes_before",
            TARGET,
        ],
        low_memory=False,
    )
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in frames
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parent = {
            "full_2022": saved["candidate_full_2022"].astype(np.float64)
        }
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )
    labels = {name: count_labels(frame) for name, frame in frames.items()}
    target = {
        name: frame[TARGET].to_numpy(np.float64) for name, frame in frames.items()
    }
    active = {name: _domain_active(axes[name]) for name in frames}
    for name in frames:
        if len(parent[name]) != len(frames[name]):
            raise ValueError(f"parent length mismatch: {name}")
        if not np.array_equal(target[name], axes[name]["target"].astype(np.float64)):
            raise ValueError(f"target order mismatch: {name}")

    frame22 = frames["full_2022"]
    early22 = frame22["game_month"].le(7).to_numpy() & active["full_2022"]
    late22 = frame22["game_month"].ge(8).to_numpy() & active["full_2022"]
    model_early22 = fit_count_dispersion(
        labels["full_2022"], parent["full_2022"], target["full_2022"], early22
    )
    correction_late22 = predict_correction(
        model_early22,
        labels["full_2022"],
        parent["full_2022"],
        late22,
    )
    model22 = fit_count_dispersion(
        labels["full_2022"],
        parent["full_2022"],
        target["full_2022"],
        active["full_2022"],
    )
    correction23 = predict_correction(
        model22, labels["late_2023"], parent["late_2023"], active["late_2023"]
    )

    residual22 = target["full_2022"][late22] - parent["full_2022"][late22]
    residual23 = (
        target["late_2023"][active["late_2023"]]
        - parent["late_2023"][active["late_2023"]]
    )
    direction22 = correction_late22[late22]
    direction23 = correction23[active["late_2023"]]
    numerator = float(
        np.dot(residual22, direction22) + np.dot(residual23, direction23)
    )
    denominator = float(
        np.dot(direction22, direction22) + np.dot(direction23, direction23)
    )
    dose = float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0

    candidate_late22 = _compose(
        parent["full_2022"], correction_late22, late22, dose
    )[late22]
    candidate23 = _compose(
        parent["late_2023"], correction23, active["late_2023"], dose
    )
    source_metrics = {
        "early_2022_to_late_2022": paired_metrics(
            frame22.loc[late22].reset_index(drop=True),
            target["full_2022"][late22],
            parent["full_2022"][late22],
            candidate_late22,
            np.ones(int(late22.sum()), dtype=bool),
        ),
        "full_2022_to_late_2023": paired_metrics(
            frames["late_2023"],
            target["late_2023"],
            parent["late_2023"],
            candidate23,
            active["late_2023"],
        ),
    }

    model23 = fit_count_dispersion(
        labels["late_2023"],
        parent["late_2023"],
        target["late_2023"],
        active["late_2023"],
    )
    correction24 = predict_correction(
        model23, labels["full_2024"], parent["full_2024"], active["full_2024"]
    )
    candidate24 = _compose(
        parent["full_2024"], correction24, active["full_2024"], dose
    )
    locked_metrics = paired_metrics(
        frames["full_2024"],
        target["full_2024"],
        parent["full_2024"],
        candidate24,
        active["full_2024"],
    )
    source_pass = bool(
        dose > 0.0
        and all(item["gain"] > 0.0 for item in source_metrics.values())
        and all(
            item["positive_month_fraction"] >= 2.0 / 3.0
            for item in source_metrics.values()
        )
    )
    locked_pass = bool(
        locked_metrics["gain"] > 0.0
        and locked_metrics["positive_month_fraction"] >= 0.75
        and locked_metrics["worst_month_gain"] > -5.0
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if source_pass and locked_pass else "reject",
        "prior_equivalent_rows": PRIOR_EQUIVALENT_ROWS,
        "correction_cap": CORRECTION_CAP,
        "source_selected_dose": dose,
        "source_metrics": source_metrics,
        "locked_full_2024": locked_metrics,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "learned_maps": {
            "early_2022": {
                "pooled_coefficient": model_early22.pooled_coefficient,
                "coefficients": model_early22.coefficients,
            },
            "full_2022": {
                "pooled_coefficient": model22.pooled_coefficient,
                "coefficients": model22.coefficients,
            },
            "late_2023": {
                "pooled_coefficient": model23.pooled_coefficient,
                "coefficients": model23.coefficients,
            },
        },
        "restrictions": {
            "official_train_only": True,
            "test_csv_read": False,
            "evaluation_batch_aggregate_used": False,
            "row_id_or_csv_order_used": False,
            "full_2024_used_for_selection": False,
            "public_score_used_for_selection": False,
            "global_level_protected": True,
        },
    }
    np.savez_compressed(
        output_dir / "audit_axes.npz",
        candidate_late_2022=candidate_late22,
        candidate_late_2023=candidate23,
        candidate_full_2024=candidate24,
        correction_full_2024=correction24,
        active_full_2024=active["full_2024"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.contract_dir,
        args.v285_axes,
        args.v290_axes,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

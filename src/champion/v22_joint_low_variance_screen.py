"""Joint low-variance residual screen above the reconstructed v21 champion.

The screen deliberately uses only six small, row-local correction directions:

* one calibration direction for each of the three official pitch domains,
* a pitcher/batter ASOF prior,
* a privileged-feature-distillation residual, and
* an alternative legal failure-mode prediction.

All source models are season-forward OOF artifacts.  The grid is small enough
to audit every combination by month and domain, which makes instability visible
instead of selecting solely on one aggregate score.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss
from src.champion.v22_domain_calibration_screen import AXES, DOMAINS
from src.champion.v22_simple_signal_screen import _axis_frame, _load_seasons, _signals


SIGNAL_NAMES = (
    "cal_r_core_to_044",
    "cal_r_anchor_to_048",
    "cal_f_to_052",
    "pitcher75_batter25",
    "pfd_student_with_ids",
    "failure_mode_h05_pow1",
)
GRIDS = (
    (0.0, 0.01, 0.02, 0.035, 0.05),
    (0.0, 0.01, 0.02, 0.035, 0.05),
    (0.0, 0.01, 0.02, 0.035, 0.05, 0.075),
    (0.0, 0.01, 0.02, 0.035, 0.05),
    (0.0, 0.05, 0.10, 0.15, 0.20),
    (0.0, 0.005, 0.01, 0.02, 0.035),
)
MODE_NAME = "conditional_mode_lgb_h0.5_pow1"


def _year_and_mask(axis: str, month: np.ndarray) -> tuple[int, np.ndarray]:
    if axis == "y2023_early_to_late":
        return 2023, month >= 8
    if axis == "y2023_to_y2024":
        return 2024, np.ones(len(month), dtype=bool)
    if axis == "y2024_early_to_late":
        return 2024, month >= 8
    raise ValueError(axis)


def _full_artifacts(project: Path, year: int) -> dict[str, np.ndarray]:
    mode_path = (
        project
        / "artifacts"
        / "latent_failure_mode_state_20260816_01"
        / f"latent_failure_mode_o{year}.npz"
    )
    with np.load(mode_path, allow_pickle=True) as saved:
        names = [str(value) for value in saved["names"].tolist()]
        mode = saved["raw"][:, names.index(MODE_NAME)].astype(np.float64)
        target = saved["target"].astype(np.float64)
        month = saved["game_month"].astype(np.int16)
    pfd_path = (
        project
        / "artifacts"
        / "trackman_distillation_20260816_02"
        / f"distillation_o{year}.npz"
    )
    with np.load(pfd_path, allow_pickle=True) as saved:
        if not np.array_equal(target, saved["target"].astype(np.float64)):
            raise ValueError(f"PFD target order mismatch for {year}")
        pfd = saved["student_with_ids"].astype(np.float64)
    return {"target": target, "month": month, "mode": mode, "pfd": pfd}


def _quadratic(
    target: np.ndarray, prediction: np.ndarray, signals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear and quadratic BSS-gain terms for correction S @ w."""
    scale = 100000.0 / (target.mean() * (1.0 - target.mean()))
    residual = target - prediction
    linear = scale * 2.0 * np.mean(residual[:, None] * signals, axis=0)
    quadratic = scale * (signals.T @ signals) / len(target)
    return linear, quadratic


def _gains(
    weights: np.ndarray, linear: np.ndarray, quadratic: np.ndarray
) -> np.ndarray:
    return weights @ linear - np.einsum(
        "ij,jk,ik->i", weights, quadratic, weights, optimize=True
    )


def _candidate(prediction: np.ndarray, signals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.clip(prediction + signals @ weights, 0.001, 0.999)


def _diagnostics(
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    month: np.ndarray,
    domain: np.ndarray,
) -> dict[str, object]:
    month_rows = []
    for value in sorted(np.unique(month)):
        mask = month == value
        month_rows.append(
            {
                "month": int(value),
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    domain_rows = []
    for value in DOMAINS:
        mask = domain == value
        domain_rows.append(
            {
                "domain": value,
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    loss_improvement = np.square(target - parent) - np.square(target - candidate)
    scale = 100000.0 / (target.mean() * (1.0 - target.mean()))
    standard_error = scale * float(np.std(loss_improvement, ddof=1)) / np.sqrt(len(target))
    gain = _bss(target, candidate) - _bss(target, parent)
    return {
        "gain": gain,
        "row_bootstrap_se": standard_error,
        "row_normal_ci95": [gain - 1.96 * standard_error, gain + 1.96 * standard_error],
        "positive_month_fraction": float(np.mean([row["gain"] > 0 for row in month_rows])),
        "worst_month_gain": float(min(row["gain"] for row in month_rows)),
        "minimum_domain_gain": float(min(row["gain"] for row in domain_rows)),
        "months": month_rows,
        "domains": domain_rows,
    }


def run(project: Path, champion_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    champion_dir = (project / champion_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seasons = _load_seasons(project)
    raw_artifacts = {year: _full_artifacts(project, year) for year in (2023, 2024)}

    weights = np.asarray(list(itertools.product(*GRIDS)), dtype=np.float64)
    folds: dict[str, dict[str, np.ndarray]] = {}
    aggregate_gains: dict[str, np.ndarray] = {}
    group_gains: dict[str, dict[str, np.ndarray]] = {}
    for axis in AXES:
        with np.load(champion_dir / f"{axis}.npz", allow_pickle=True) as saved:
            target = saved["target"].astype(np.float64)
            v21 = saved["v21"].astype(np.float64)
            domain = saved["domain3"].astype(str)
            month = saved["game_month"].astype(np.int16)
        year, mask = _year_and_mask(axis, raw_artifacts[2023 if axis == "y2023_early_to_late" else 2024]["month"])
        artifact = raw_artifacts[year]
        if not np.array_equal(target, artifact["target"][mask]):
            raise ValueError(f"artifact target order mismatch for {axis}")
        if not np.array_equal(month, artifact["month"][mask]):
            raise ValueError(f"artifact month order mismatch for {axis}")
        frame = _axis_frame(seasons, axis)
        prior = _signals(frame)["pitcher75_batter25"]
        if len(prior) != len(target):
            raise ValueError(f"ASOF row mismatch for {axis}")
        signal = np.column_stack(
            [
                np.where(domain == "R_CORE", 0.44 - v21, 0.0),
                np.where(domain == "R_ANCHOR", 0.48 - v21, 0.0),
                np.where(domain == "F", 0.52 - v21, 0.0),
                prior - v21,
                artifact["pfd"][mask],
                artifact["mode"][mask] - v21,
            ]
        )
        linear, quadratic = _quadratic(target, v21, signal)
        aggregate_gains[axis] = _gains(weights, linear, quadratic)
        fold_groups: dict[str, np.ndarray] = {}
        for value in sorted(np.unique(month)):
            selected = month == value
            terms = _quadratic(target[selected], v21[selected], signal[selected])
            fold_groups[f"month_{int(value):02d}"] = _gains(weights, *terms)
        for value in DOMAINS:
            selected = domain == value
            terms = _quadratic(target[selected], v21[selected], signal[selected])
            fold_groups[f"domain_{value}"] = _gains(weights, *terms)
        group_gains[axis] = fold_groups
        folds[axis] = {
            "target": target,
            "v21": v21,
            "domain": domain,
            "month": month,
            "signals": signal,
        }

    table = pd.DataFrame(weights, columns=[f"weight_{name}" for name in SIGNAL_NAMES])
    for axis in AXES:
        table[axis] = aggregate_gains[axis]
        months = np.column_stack(
            [value for name, value in group_gains[axis].items() if name.startswith("month_")]
        )
        domains = np.column_stack(
            [value for name, value in group_gains[axis].items() if name.startswith("domain_")]
        )
        table[f"{axis}__positive_month_fraction"] = np.mean(months > 0.0, axis=1)
        table[f"{axis}__worst_month_gain"] = np.min(months, axis=1)
        table[f"{axis}__minimum_domain_gain"] = np.min(domains, axis=1)
    table["min_gain"] = table[list(AXES)].min(axis=1)
    table["mean_gain"] = table[list(AXES)].mean(axis=1)
    stable = (
        (table["min_gain"] > 0.0)
        & (table["y2023_to_y2024__positive_month_fraction"] >= 0.75)
        & (table["y2024_early_to_late__positive_month_fraction"] >= 1.0)
        & (table["y2023_to_y2024__minimum_domain_gain"] > 0.0)
        & (table["y2024_early_to_late__minimum_domain_gain"] > 0.0)
    )
    table["passes_stability_gate"] = stable
    table = table.sort_values(
        ["passes_stability_gate", "min_gain", "mean_gain"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(output_dir / "metrics.csv", index=False)

    chosen = table.loc[table["passes_stability_gate"]].head(1)
    if chosen.empty:
        chosen = table.head(1)
    selected = chosen.iloc[0]
    selected_weights = selected[[f"weight_{name}" for name in SIGNAL_NAMES]].to_numpy(np.float64)
    diagnostics = {}
    for axis, fold in folds.items():
        candidate = _candidate(fold["v21"], fold["signals"], selected_weights)
        diagnostics[axis] = _diagnostics(
            fold["target"], fold["v21"], candidate, fold["month"], fold["domain"]
        )

    deployment_profile_weights = {
        "conservative": np.asarray([0.020, 0.010, 0.010, 0.035, 0.0, 0.0]),
        "balanced": np.asarray([0.035, 0.020, 0.020, 0.050, 0.0, 0.0]),
    }
    deployment_profiles = {}
    for name, profile_weights in deployment_profile_weights.items():
        profile_diagnostics = {}
        for axis, fold in folds.items():
            candidate = _candidate(fold["v21"], fold["signals"], profile_weights)
            profile_diagnostics[axis] = _diagnostics(
                fold["target"], fold["v21"], candidate, fold["month"], fold["domain"]
            )
        deployment_profiles[name] = {
            "weights": {
                signal_name: float(profile_weights[index])
                for index, signal_name in enumerate(SIGNAL_NAMES)
            },
            "diagnostics": profile_diagnostics,
        }

    scale_sweep = []
    for scale in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75):
        scaled_weights = scale * deployment_profile_weights["balanced"]
        gains = {}
        for axis, fold in folds.items():
            candidate = _candidate(fold["v21"], fold["signals"], scaled_weights)
            gains[axis] = _bss(fold["target"], candidate) - _bss(
                fold["target"], fold["v21"]
            )
        scale_sweep.append(
            {
                "scale": scale,
                **gains,
                "min_gain": min(gains.values()),
                "mean_gain": float(np.mean(list(gains.values()))),
            }
        )

    # A genuine nested stress test: select only on 2023 late and report, without
    # refitting, how that selected recipe transfers to both 2024 audits.
    nested_index = int(np.argmax(aggregate_gains["y2023_early_to_late"]))
    nested = {
        "selection_axis": "y2023_early_to_late",
        "weights": {
            name: float(weights[nested_index, index])
            for index, name in enumerate(SIGNAL_NAMES)
        },
        "gains": {axis: float(values[nested_index]) for axis, values in aggregate_gains.items()},
    }
    summary = {
        "protocol": "V22_JOINT_LOW_VARIANCE_SCREEN_V1",
        "candidate_count": int(len(table)),
        "strictly_positive": int((table["min_gain"] > 0.0).sum()),
        "stability_gate_passed": int(table["passes_stability_gate"].sum()),
        "selected": {
            key: (bool(value) if isinstance(value, (np.bool_, bool)) else float(value))
            for key, value in selected.items()
        },
        "selected_diagnostics": diagnostics,
        "deployment_profiles": deployment_profiles,
        "balanced_scale_sweep": scale_sweep,
        "nested_2023_selection_stress_test": nested,
        "top": table.head(30).to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--champion-dir", type=Path, default=Path("artifacts/champion_oof_20260817_01")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v22_joint_low_variance_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.champion_dir, args.output_dir)


if __name__ == "__main__":
    main()

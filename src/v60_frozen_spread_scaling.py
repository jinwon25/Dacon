"""Frozen train-centred probability spread scaling above eta=0.15.

The transform uses a scalar centre computed only from the labelled source
rows used before each forecast origin.  No centre, mean, or spread is computed
from the audit/test batch.  A recipe is selected on late 2023 and then applied
unchanged to full and late 2024 with their corresponding source-train centre.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v30_diverse_covariance_screen import _cached_v25_axes, diagnostics
from src.v58_eta15_rebase_audit import CURRENT_ETA, eta_parent


ALPHAS = (0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15)
MODES = ("linear", "logit")
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
EPSILON = 1e-6


def spread_scale(
    probability: np.ndarray,
    *,
    center: float,
    alpha: float,
    mode: str,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if not 0.0 < center < 1.0:
        raise ValueError("spread centre must lie in (0, 1)")
    if alpha <= 0.0:
        raise ValueError("spread alpha must be positive")
    if mode == "linear":
        output = center + float(alpha) * (probability - center)
    elif mode == "logit":
        clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
        center_logit = np.log(center / (1.0 - center))
        logits = np.log(clipped / (1.0 - clipped))
        scaled = center_logit + float(alpha) * (logits - center_logit)
        output = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))
    else:
        raise ValueError(f"unknown spread mode: {mode}")
    return np.clip(output, 0.001, 0.999)


def apply_route(
    frame: pd.DataFrame,
    parent: np.ndarray,
    transformed: np.ndarray,
    route: str,
) -> tuple[np.ndarray, np.ndarray]:
    if route == "ALL":
        active = np.ones(len(frame), dtype=bool)
    elif route in {"R_CORE", "R_ANCHOR", "F"}:
        active = frame["domain3"].astype(str).eq(route).to_numpy()
    else:
        raise ValueError(f"unknown route: {route}")
    candidate = np.asarray(parent, dtype=np.float64).copy()
    candidate[active] = np.asarray(transformed, dtype=np.float64)[active]
    return candidate, active


def _screen(frame: pd.DataFrame, center: float) -> pd.DataFrame:
    parent = eta_parent(frame, CURRENT_ETA)
    rows = []
    for mode in MODES:
        for alpha in ALPHAS:
            transformed = spread_scale(
                parent, center=center, alpha=alpha, mode=mode
            )
            for route in ROUTES:
                candidate, active = apply_route(
                    frame, parent, transformed, route
                )
                result = diagnostics(frame, parent, candidate, active)
                applied_gain = (
                    min(result["domain_gains"].values())
                    if route == "ALL"
                    else result["domain_gains"][route]
                )
                rows.append(
                    {
                        "mode": mode,
                        "alpha": alpha,
                        "route": route,
                        "source_center": center,
                        "applied_domain_gain": applied_gain,
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"months", "domain_gains"}
                        },
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = metrics[
        ["gain", "worst_month_gain", "applied_domain_gain"]
    ].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["alpha"].ne(1.0)
        & metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
        & metrics["applied_domain_gain"].gt(0.0)
        & metrics["minimum_domain_gain"].ge(0.0)
    )
    return metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def _audit(
    frame: pd.DataFrame,
    center: float,
    recipe: dict[str, object],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    parent = eta_parent(frame, CURRENT_ETA)
    transformed = spread_scale(
        parent,
        center=center,
        alpha=float(recipe["alpha"]),
        mode=str(recipe["mode"]),
    )
    candidate, active = apply_route(
        frame, parent, transformed, str(recipe["route"])
    )
    return diagnostics(frame, parent, candidate, active), candidate, active


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    centers = {
        "selection_late_2023": float(
            raw.loc[
                raw["season"].eq(2023) & raw["game_month"].le(7),
                "control_success",
            ].mean()
        ),
        "outer_full_2024": float(
            raw.loc[raw["season"].eq(2023), "control_success"].mean()
        ),
        "replication_late_2024": float(
            raw.loc[
                raw["season"].eq(2024) & raw["game_month"].le(7),
                "control_success",
            ].mean()
        ),
    }
    del raw

    selection_metrics = _screen(
        axes["selection_late_2023"], centers["selection_late_2023"]
    )
    selection_metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = selection_metrics.loc[selection_metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else selection_metrics.iloc[0]
    recipe = {
        "mode": str(selected["mode"]),
        "alpha": float(selected["alpha"]),
        "route": str(selected["route"]),
    }

    audits = {}
    for axis_name in ("outer_full_2024", "replication_late_2024"):
        result, candidate, active = _audit(
            axes[axis_name], centers[axis_name], recipe
        )
        audits[axis_name] = result
        frame = axes[axis_name]
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            eta15=eta_parent(frame, CURRENT_ETA),
            source_center=centers[axis_name],
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "selection_gate": bool(selected["passes_selection_gate"]),
        "outer_gain_at_least_3": audits["outer_full_2024"]["gain"] >= 3.0,
        "outer_month_fraction_at_least_075": audits["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_5": audits["outer_full_2024"][
            "worst_month_gain"
        ]
        > -5.0,
        "outer_minimum_domain_nonnegative": audits["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": audits["replication_late_2024"]["gain"]
        > 0.0,
        "replication_all_months_positive": audits["replication_late_2024"][
            "positive_month_fraction"
        ]
        == 1.0,
    }
    summary = {
        "protocol": "V60_TRAIN_CENTERED_SPREAD_SCALING_ABOVE_ETA15_V1",
        "parent": "eta=0.15 R_ANCHOR parent; final 0819 gate excluded locally",
        "candidate_count": int(len(selection_metrics)),
        "selection_gate_count": int(
            selection_metrics["passes_selection_gate"].sum()
        ),
        "source_centers": centers,
        "chosen": recipe,
        "selection": {
            key: float(selected[key])
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "applied_domain_gain",
            )
        },
        "audits": audits,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "test_distribution_used": False,
        "center_source": "labelled source-train rows only",
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v60_frozen_spread_scaling_20260822_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

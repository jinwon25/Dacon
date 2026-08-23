"""Temporal convex stacking of independently generated OOF families.

Earlier screens evaluated individual OOF predictions and small routed pairs.
This experiment asks a different question: can a low-degree convex meta-model
combine four distinct prediction generators without using the audit year?

The chronology is fixed:

* fit simplex weights on full 2022 OOF predictions;
* choose representation, shrinkage, route, and blend strength on late 2023;
* refit the same simplex recipe on full 2023 OOF predictions;
* open full 2024 once, with late 2024 as a secondary stability audit.

All base columns are season-forward OOF predictions.  The fitted stack is a
row-local convex combination and never reads another audit or test row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _load_bank, _metadata, signal_family


MODES = ("family", "all")
LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2)
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
ETAS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)


def bank_matrix(
    bank: dict[str, np.ndarray],
    names: list[str],
    mode: str,
) -> tuple[np.ndarray, list[str]]:
    """Return a deterministic direct-prediction matrix."""

    missing = sorted(set(names) - set(bank))
    if missing:
        raise ValueError(f"bank is missing signals: {missing}")
    raw = np.column_stack(
        [np.asarray(bank[name], dtype=np.float64) for name in names]
    )
    if mode == "all":
        return raw, list(names)
    if mode != "family":
        raise ValueError(f"unknown stack mode: {mode}")
    families = sorted({signal_family(name) for name in names})
    values = []
    for family in families:
        selected = [signal_family(name) == family for name in names]
        values.append(raw[:, selected].mean(axis=1))
    return np.column_stack(values), families


def fit_simplex(
    features: np.ndarray,
    target: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Fit a non-negative, sum-to-one ridge stack from sufficient statistics."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or x.shape[1] == 0:
        raise ValueError("invalid feature/target shape for simplex stack")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("simplex stack input contains non-finite values")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    width = x.shape[1]
    prior = np.full(width, 1.0 / width, dtype=np.float64)
    gram = x.T @ x / len(y)
    cross = x.T @ y / len(y)

    def objective(weight: np.ndarray) -> float:
        delta = weight - prior
        return float(
            weight @ gram @ weight
            - 2.0 * cross @ weight
            + float(regularization) * (delta @ delta)
        )

    def gradient(weight: np.ndarray) -> np.ndarray:
        return 2.0 * (
            gram @ weight
            - cross
            + float(regularization) * (weight - prior)
        )

    result = minimize(
        objective,
        prior,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * width,
        constraints={
            "type": "eq",
            "fun": lambda weight: float(weight.sum() - 1.0),
            "jac": lambda weight: np.ones_like(weight),
        },
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"simplex optimization failed: {result.message}")
    weight = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weight /= weight.sum()
    return weight


def _route_mask(frame: pd.DataFrame, route: str) -> np.ndarray:
    if route == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["domain3"].astype(str).eq(route).to_numpy()


def compose(
    frame: pd.DataFrame,
    direct: np.ndarray,
    route: str,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    raw = np.asarray(direct, dtype=np.float64)
    if len(raw) != len(frame):
        raise ValueError("direct prediction and frame lengths differ")
    active = _route_mask(frame, route)
    candidate = parent.copy()
    candidate[active] = np.clip(
        parent[active] + float(eta) * (raw[active] - parent[active]),
        0.001,
        0.999,
    )
    return candidate, active


def applied_domain_gain(result: dict[str, object], route: str) -> float:
    values = result["domain_gains"]
    if not isinstance(values, dict):
        raise TypeError("diagnostics domain_gains must be a mapping")
    if route == "ALL":
        return float(min(values.values()))
    return float(values[route])


def selection_grid(
    frame: pd.DataFrame,
    direct: np.ndarray,
    *,
    mode: str,
    regularization: float,
) -> list[dict[str, object]]:
    parent = v27_parent(frame)
    rows = []
    for route in ROUTES:
        for eta in ETAS:
            candidate, active = compose(frame, direct, route, eta)
            result = diagnostics(frame, parent, candidate, active)
            applied = applied_domain_gain(result, route)
            selection_score = min(
                float(result["gain"]),
                float(result["worst_month_gain"]),
                applied,
            )
            rows.append(
                {
                    "mode": mode,
                    "regularization": float(regularization),
                    "route": route,
                    "eta": float(eta),
                    "gain": float(result["gain"]),
                    "positive_month_fraction": float(
                        result["positive_month_fraction"]
                    ),
                    "worst_month_gain": float(result["worst_month_gain"]),
                    "minimum_domain_gain": float(result["minimum_domain_gain"]),
                    "applied_domain_gain": applied,
                    "mean_abs_shift": float(result["mean_abs_shift"]),
                    "selection_score": selection_score,
                }
            )
    return rows


def _audit(
    frame: pd.DataFrame,
    direct: np.ndarray,
    route: str,
    eta: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    candidate, active = compose(frame, direct, route, eta)
    result = diagnostics(frame, parent, candidate, active)
    result["applied_domain_gain"] = applied_domain_gain(result, route)
    return result, candidate, active


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(_load_bank(project, 2022))
    banks = {
        year: _load_bank(project, year, set(names))
        for year in (2022, 2023, 2024)
    }
    metadata = {year: _metadata(project, year) for year in (2022, 2023, 2024)}
    matrices = {
        (year, mode): bank_matrix(banks[year], names, mode)
        for year in (2022, 2023, 2024)
        for mode in MODES
    }

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    del raw, banks
    late23 = metadata[2023]["month"] >= 8
    late24 = metadata[2024]["month"] >= 8
    selection_frame = axes["selection_late_2023"]
    if not np.array_equal(
        metadata[2023]["target"][late23],
        selection_frame["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 stack/frame target alignment failed")

    rows: list[dict[str, object]] = []
    selection_weights: dict[tuple[str, float], tuple[np.ndarray, list[str]]] = {}
    for mode in MODES:
        x22, columns = matrices[(2022, mode)]
        x23, columns23 = matrices[(2023, mode)]
        if columns != columns23:
            raise ValueError(f"stack columns differ for mode={mode}")
        for regularization in LAMBDAS:
            weight = fit_simplex(
                x22,
                metadata[2022]["target"],
                regularization,
            )
            selection_weights[(mode, regularization)] = (weight, columns)
            direct23 = x23 @ weight
            rows.extend(
                selection_grid(
                    selection_frame,
                    direct23[late23],
                    mode=mode,
                    regularization=regularization,
                )
            )

    selection = pd.DataFrame(rows)
    selection["passes_selection_gate"] = (
        selection["gain"].gt(0.0)
        & selection["positive_month_fraction"].eq(1.0)
        & selection["worst_month_gain"].gt(0.0)
        & selection["applied_domain_gain"].gt(0.0)
        & selection["minimum_domain_gain"].ge(-2.0)
    )
    selection = selection.sort_values(
        ["passes_selection_gate", "selection_score", "gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    selection.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = selection.loc[selection["passes_selection_gate"]]
    chosen = passing.iloc[0] if len(passing) else selection.iloc[0]
    recipe = {
        "mode": str(chosen["mode"]),
        "regularization": float(chosen["regularization"]),
        "route": str(chosen["route"]),
        "eta": float(chosen["eta"]),
    }

    x23, weight_columns = matrices[(2023, recipe["mode"])]
    x24, columns24 = matrices[(2024, recipe["mode"])]
    if weight_columns != columns24:
        raise ValueError("2023/2024 refit columns differ")
    refit_weight = fit_simplex(
        x23,
        metadata[2023]["target"],
        recipe["regularization"],
    )
    direct24 = x24 @ refit_weight
    original_weight, original_columns = selection_weights[
        (recipe["mode"], recipe["regularization"])
    ]
    if original_columns != weight_columns:
        raise ValueError("selection/refit weight columns differ")
    selection_result, selection_candidate, selection_active = _audit(
        selection_frame,
        (x23 @ original_weight)[late23],
        recipe["route"],
        recipe["eta"],
    )
    np.savez_compressed(
        output_dir / "selection_late_2023.npz",
        target=selection_frame["target"].to_numpy(np.float64),
        v27=v27_parent(selection_frame),
        direct=(x23 @ original_weight)[late23],
        candidate=selection_candidate,
        active=selection_active,
        domain3=selection_frame["domain3"].astype(str).to_numpy(),
        game_month=selection_frame["game_month"].to_numpy(np.int16),
    )
    pd.DataFrame(
        {
            "feature": weight_columns,
            "fit_2022_weight": original_weight,
            "refit_2023_weight": refit_weight,
        }
    ).to_csv(output_dir / "chosen_weights.csv", index=False)

    audits: dict[str, dict[str, object]] = {}
    for axis, keep in (
        ("outer_full_2024", np.ones(len(direct24), dtype=bool)),
        ("replication_late_2024", late24),
    ):
        frame = axes[axis]
        result, candidate, active = _audit(
            frame,
            direct24[keep],
            recipe["route"],
            recipe["eta"],
        )
        audits[axis] = result
        np.savez_compressed(
            output_dir / f"{axis}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=v27_parent(frame),
            direct=direct24[keep],
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    outer = audits["outer_full_2024"]
    replication = audits["replication_late_2024"]
    gates = {
        "selection_gate": bool(chosen["passes_selection_gate"]),
        "outer_gain_at_least_5": outer["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": outer[
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": outer["worst_month_gain"] > -10.0,
        "outer_applied_domain_positive": outer["applied_domain_gain"] > 0.0,
        "replication_gain_positive": replication["gain"] > 0.0,
        "replication_month_fraction_at_least_two_thirds": replication[
            "positive_month_fraction"
        ]
        >= 2.0 / 3.0,
        "replication_worst_month_above_minus_10": replication[
            "worst_month_gain"
        ]
        > -10.0,
    }
    summary = {
        "protocol": "V49_TEMPORAL_CONVEX_OOF_STACK_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "base_signal_count": len(names),
        "families": sorted({signal_family(name) for name in names}),
        "selection_fit": "full-2022 season-forward OOF",
        "selection_audit": "late-2023 v27 analogue",
        "outer_refit": "full-2023 season-forward OOF",
        "outer_audit": "full-2024; late-2024 secondary",
        "selection_candidate_count": int(len(selection)),
        "selection_gate_count": int(selection["passes_selection_gate"].sum()),
        "chosen": {
            **recipe,
            "gain": float(chosen["gain"]),
            "worst_month_gain": float(chosen["worst_month_gain"]),
            "applied_domain_gain": float(chosen["applied_domain_gain"]),
            "selection_score": float(chosen["selection_score"]),
        },
        "selection_diagnostics": selection_result,
        "weight_l1_shift_2022_to_2023": float(
            np.sum(np.abs(refit_weight - original_weight))
        ),
        "weight_max_shift_2022_to_2023": float(
            np.max(np.abs(refit_weight - original_weight))
        ),
        "audits": audits,
        "gates": {name: bool(value) for name, value in gates.items()},
        "numerically_eligible": bool(all(gates.values())),
        "eligible_for_packaging": False,
        "promotion_block": (
            "Numerical gates fail; the constituent OOF family has also been "
            "audited repeatedly on 2024 in earlier experiments."
        ),
        "family_reused_audit_risk": True,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_recipe_selection": False,
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
        default=Path("artifacts/v49_temporal_convex_stack_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()

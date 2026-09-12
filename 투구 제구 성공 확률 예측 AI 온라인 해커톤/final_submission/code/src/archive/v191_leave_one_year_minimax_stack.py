"""Fit a tri-year L1-bounded stack with leave-one-year-out validation.

Official-data candidate directions are measured relative to the common v158
base and rebased above the exact row-region parent.  An L1 budget is selected only when
weights fitted on two origins improve the third held-out origin for all three
rotations.  The final weights then use 2022, late-2023 and 2024 as development
origins; no test rows or Public scores are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v178_row_region_signed_stack_rebase import load_weights
from src.core.contract import _load_contract_axis


PROTOCOL = "V191_LEAVE_ONE_YEAR_MINIMAX_STACK_V1"
AXES = ("full_2022", "late_2023", "full_2024")
L1_BUDGETS = (0.025, 0.05, 0.10, 0.15)
MONTH_FLOOR = -0.5
MAX_SIGNED_WEIGHT = 0.10
RIDGE = 0.005


def quadratic_gain_form(
    target: np.ndarray,
    base: np.ndarray,
    directions: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    local_target = np.asarray(target, dtype=np.float64)[mask]
    local_base = np.asarray(base, dtype=np.float64)[mask]
    local_direction = np.asarray(directions, dtype=np.float64)[mask]
    rate = float(np.mean(local_target))
    reference = max(rate * (1.0 - rate), 1e-12)
    scale = 100_000.0 / (reference * len(local_target))
    residual = local_target - local_base
    linear = scale * (2.0 * local_direction.T @ residual)
    quadratic = scale * (local_direction.T @ local_direction)
    return linear, quadratic


def form_gain(weight: np.ndarray, form: tuple[np.ndarray, np.ndarray]) -> float:
    linear, quadratic = form
    return float(linear @ weight - weight @ quadratic @ weight)


def solve_minimax(
    forms: dict[str, tuple[np.ndarray, np.ndarray]],
    origin_names: list[str],
    month_names: list[str],
    budget: float,
    seed: int,
) -> dict[str, Any]:
    n_direction = len(forms[origin_names[0]][0])

    def objective(value: np.ndarray) -> float:
        return float(-value[-1] + RIDGE * np.square(value[:-1]).sum())

    constraints: list[dict[str, Any]] = [
        {"type": "ineq", "fun": lambda value: float(budget) - float(value[:-1].sum())}
    ]
    for name in origin_names:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda value, key=name: form_gain(value[:-1], forms[key]) - value[-1],
            }
        )
    for name in month_names:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda value, key=name: form_gain(value[:-1], forms[key]) - MONTH_FLOOR,
            }
        )

    # A tangent linear program supplies a non-zero improving start at the
    # otherwise sticky all-zero boundary.
    c = np.r_[np.zeros(n_direction), -1.0]
    a_rows = []
    b_rows = []
    for name in origin_names:
        a_rows.append(np.r_[-forms[name][0], 1.0])
        b_rows.append(0.0)
    for name in month_names:
        a_rows.append(np.r_[-forms[name][0], 0.0])
        b_rows.append(0.0)
    a_rows.append(np.r_[np.ones(n_direction), 0.0])
    b_rows.append(float(budget))
    tangent = linprog(
        c,
        A_ub=np.vstack(a_rows),
        b_ub=np.asarray(b_rows),
        bounds=[(0.0, MAX_SIGNED_WEIGHT)] * n_direction + [(-100.0, 100.0)],
        method="highs",
    )
    starts = [np.zeros(n_direction, dtype=np.float64)]
    if tangent.success and tangent.x[-1] > 0.0:
        starts.extend(tangent.x[:-1] * scale for scale in (0.1, 0.5, 1.0))
    rng = np.random.default_rng(seed)
    for _ in range(3):
        random = rng.uniform(0.0, 1.0, size=n_direction)
        random *= min(float(budget) * 0.5, 0.05) / random.sum()
        starts.append(random)

    best: dict[str, Any] | None = None
    bounds = [(0.0, MAX_SIGNED_WEIGHT)] * n_direction + [(-100.0, 100.0)]
    for start in starts:
        initial_t = min(form_gain(start, forms[name]) for name in origin_names)
        fitted = minimize(
            objective,
            np.r_[start, initial_t],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )
        violation = max(
            [0.0] + [-float(constraint["fun"](fitted.x)) for constraint in constraints]
        )
        record = {
            "success": bool(fitted.success),
            "message": str(fitted.message),
            "objective": float(fitted.fun),
            "constraint_violation": float(violation),
            "iterations": int(fitted.nit),
            "minimum_origin_gain": float(fitted.x[-1]),
            "weight": fitted.x[:-1].astype(np.float64),
            "tangent_success": bool(tangent.success),
            "tangent_t": float(tangent.x[-1]) if tangent.success else float("nan"),
        }
        if best is None or (record["constraint_violation"], record["objective"]) < (
            best["constraint_violation"], best["objective"]
        ):
            best = record
    if best is None:
        raise RuntimeError("minimax solver produced no result")
    return best


def stack_prediction(
    parent: np.ndarray,
    signed_directions: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(parent, dtype=np.float64)
        + np.asarray(signed_directions, dtype=np.float64) @ np.asarray(weight),
        0.001,
        0.999,
    )


def held_pass(result: dict[str, Any]) -> bool:
    minimum_month_fraction = 2.0 / 3.0 if len(result["months"]) <= 3 else 0.625
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= minimum_month_fraction
        and result["worst_month_gain"] > -2.0
        and result["minimum_domain_gain"] >= 0.0
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, post4 = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, post4, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    frozen_weights = load_weights(v165_summary)
    frozen_summary = json.loads(v165_summary.read_text(encoding="utf-8"))
    families = [str(name) for name in frozen_summary["families"]]
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {name: saved[name].astype(np.float64) for name in AXES}
    raw_directions: dict[str, np.ndarray] = {}
    for axis_name in AXES:
        columns = []
        for family in families:
            with np.load(library_root / family / "selected_axes.npz", allow_pickle=False) as saved:
                columns.append(saved[axis_name].astype(np.float64) - v158_base[axis_name])
        raw = np.column_stack(columns)
        raw_directions[axis_name] = np.column_stack([raw, -raw])
    signed_names = [f"+{name}" for name in families] + [f"-{name}" for name in families]

    forms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    months_by_axis: dict[str, list[str]] = {}
    for axis_name in AXES:
        axis = axes[axis_name]
        exact = np.asarray(axis["exact_mask"], dtype=bool)
        forms[axis_name] = quadratic_gain_form(
            axis["target"], parents[axis_name], raw_directions[axis_name], exact
        )
        months_by_axis[axis_name] = []
        for month in sorted(np.unique(axis["game_month"]).tolist()):
            key = f"{axis_name}|month={int(month)}"
            mask = exact & np.asarray(axis["game_month"]).__eq__(month)
            forms[key] = quadratic_gain_form(
                axis["target"], parents[axis_name], raw_directions[axis_name], mask
            )
            months_by_axis[axis_name].append(key)

    loo_rows: list[dict[str, Any]] = []
    loo_detail: dict[str, Any] = {}
    loo_weights: dict[tuple[float, str], np.ndarray] = {}
    for budget in L1_BUDGETS:
        for held in AXES:
            training = [name for name in AXES if name != held]
            training_months = [
                key for name in training for key in months_by_axis[name]
            ]
            solved = solve_minimax(
                forms, training, training_months, budget,
                seed=19100 + int(budget * 1000) + AXES.index(held),
            )
            weight = solved.pop("weight")
            loo_weights[(budget, held)] = weight
            candidate = stack_prediction(
                parents[held], raw_directions[held], weight
            )
            result = metrics(axes[held], parents[held], candidate)
            key = f"b{budget:g}__held_{held}"
            loo_detail[key] = {"solver": solved, "metrics": result}
            loo_rows.append(
                {
                    "budget": budget,
                    "held_axis": held,
                    "held_passed": held_pass(result),
                    "gain": result["gain"],
                    "positive_month_fraction": result["positive_month_fraction"],
                    "worst_month_gain": result["worst_month_gain"],
                    "l1_weight": float(weight.sum()),
                    "nonzero_net_weights": int(
                        np.sum(np.abs(weight[:len(families)] - weight[len(families):]) > 1e-8)
                    ),
                }
            )
    loo = pd.DataFrame(loo_rows)
    budget_rows = []
    for budget in L1_BUDGETS:
        part = loo.loc[loo["budget"].eq(budget)]
        budget_rows.append(
            {
                "budget": budget,
                "all_held_passed": bool(part["held_passed"].all()),
                "minimum_held_gain": float(part["gain"].min()),
                "mean_held_gain": float(part["gain"].mean()),
                "worst_held_month_gain": float(part["worst_month_gain"].min()),
            }
        )
    budget_ranking = pd.DataFrame(budget_rows).sort_values(
        ["all_held_passed", "minimum_held_gain", "mean_held_gain"],
        ascending=[False, False, False], kind="stable",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loo.to_csv(output_dir / "leave_one_year_results.csv", index=False, encoding="utf-8-sig")
    budget_ranking.to_csv(output_dir / "budget_ranking.csv", index=False, encoding="utf-8-sig")
    passing = budget_ranking.loc[budget_ranking["all_held_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "loo_reject",
            "families": families,
            "budget_ranking": budget_ranking.to_dict(orient="records"),
            "leave_one_year_detail": loo_detail,
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_budget = float(passing.iloc[0]["budget"])
        all_months = [key for name in AXES for key in months_by_axis[name]]
        final_solutions: dict[float, tuple[dict[str, Any], np.ndarray]] = {}
        final_candidates: dict[float, np.ndarray] = {}
        for budget in L1_BUDGETS:
            solved = solve_minimax(
                forms, list(AXES), all_months, budget,
                seed=19200 + int(budget * 1000),
            )
            weight = solved.pop("weight")
            final_solutions[budget] = (solved, weight)
            final_candidates[budget] = stack_prediction(
                parents["full_2024"], raw_directions["full_2024"], weight
            )
        selected_solver, selected_weight = final_solutions[selected_budget]
        candidates = {
            name: stack_prediction(parents[name], raw_directions[name], selected_weight)
            for name in AXES
        }
        final_metrics = {
            name: metrics(axes[name], parents[name], candidates[name]) for name in AXES
        }
        exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidates["full_2024"],
            exact24, [final_candidates[budget] for budget in L1_BUDGETS],
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        final_point_pass = all(held_pass(final_metrics[name]) for name in AXES)
        net = selected_weight[:len(families)] - selected_weight[len(families):]
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            full_2022=candidates["full_2022"],
            late_2023=candidates["late_2023"],
            full_2024=candidates["full_2024"],
            parent_full_2024=parents["full_2024"],
            signed_weight=selected_weight,
            signed_names=np.asarray(signed_names),
            net_weight=net,
            family_names=np.asarray(families),
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if final_point_pass and robust_pass else (
                "point_pass_robust_reject" if final_point_pass else "final_point_reject"
            ),
            "families": families,
            "selected_l1_budget_by_loo": selected_budget,
            "budget_ranking": budget_ranking.to_dict(orient="records"),
            "leave_one_year_detail": loo_detail,
            "final_solver": selected_solver,
            "final_l1_weight": float(selected_weight.sum()),
            "final_net_weights": {
                name: float(value) for name, value in zip(families, net) if abs(value) > 1e-8
            },
            "final_metrics": final_metrics,
            "robustness_2024": robust,
            "final_point_gate_passed": final_point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(final_point_pass and robust_pass),
            "parity": parity,
            "restrictions": restrictions(),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "full_2024_is_development_origin": True,
        "l1_budget_selected_by_leave_one_year_out": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_components_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v158-path", type=Path, required=True)
    parser.add_argument("--v165-summary", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

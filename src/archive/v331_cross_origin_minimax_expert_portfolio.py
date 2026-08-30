"""Cross-origin minimax portfolio of frozen baseball experts above v320.

The eight inputs are independently developed, fixed-dose prediction directions.
Only regular-season rows are eligible because the v320 Futures route is already
protected by its direct and low-rank components.  A total dose budget is chosen
by leave-one-origin-out validation over full-2022, late-2023 and full-2024.

Unlike a cell router, this model uses one small non-negative portfolio for every
regular-season row.  It therefore has only eight global degrees of freedom and
cannot infer anything from evaluation-row order, frequency, or other rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import (
    EXPERT_ORDER,
    TARGET,
    _candidate_minus_parent,
    axis_metrics,
)


PROTOCOL = "V331_CROSS_ORIGIN_MINIMAX_EXPERT_PORTFOLIO_V1"
ORIGINS = ("full_2022", "late_2023", "full_2024")
BUDGETS = (0.25, 0.50, 1.00, 1.50)
MAX_EXPERT_DOSE = 1.0
MONTH_FLOOR = -0.50
RIDGE = 0.0025


def quadratic_gain_form(
    target: np.ndarray,
    parent: np.ndarray,
    directions: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact BSS-gain quadratic for ``parent + D @ weight``."""

    target = np.asarray(target, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    local_target = target[mask]
    local_parent = parent[mask]
    local_direction = directions[mask]
    rate = float(local_target.mean())
    scale = 100000.0 / (max(rate * (1.0 - rate), 1e-12) * len(local_target))
    residual = local_target - local_parent
    linear = scale * (2.0 * local_direction.T @ residual)
    quadratic = scale * (local_direction.T @ local_direction)
    return linear, quadratic


def form_gain(weight: np.ndarray, form: tuple[np.ndarray, np.ndarray]) -> float:
    linear, quadratic = form
    return float(linear @ weight - weight @ quadratic @ weight)


def solve_minimax(
    forms: dict[str, tuple[np.ndarray, np.ndarray]],
    origins: list[str],
    safety_groups: list[str],
    budget: float,
    seed: int,
) -> dict[str, Any]:
    """Maximize the worst origin gain with weak month/domain loss floors."""

    n_expert = len(forms[origins[0]][0])

    def objective(value: np.ndarray) -> float:
        return float(-value[-1] + RIDGE * np.square(value[:-1]).sum())

    constraints: list[dict[str, Any]] = [
        {
            "type": "ineq",
            "fun": lambda value: float(budget) - float(value[:-1].sum()),
        }
    ]
    for origin in origins:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda value, key=origin: (
                    form_gain(value[:-1], forms[key]) - value[-1]
                ),
            }
        )
    for group in safety_groups:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda value, key=group: (
                    form_gain(value[:-1], forms[key]) - MONTH_FLOOR
                ),
            }
        )

    # A linearized solution avoids the stationary all-zero SLSQP start.
    c = np.r_[np.zeros(n_expert), -1.0]
    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    for origin in origins:
        a_rows.append(np.r_[-forms[origin][0], 1.0])
        b_rows.append(0.0)
    for group in safety_groups:
        a_rows.append(np.r_[-forms[group][0], 0.0])
        b_rows.append(-MONTH_FLOOR)
    a_rows.append(np.r_[np.ones(n_expert), 0.0])
    b_rows.append(float(budget))
    tangent = linprog(
        c,
        A_ub=np.vstack(a_rows),
        b_ub=np.asarray(b_rows),
        bounds=[(0.0, MAX_EXPERT_DOSE)] * n_expert + [(-100.0, 100.0)],
        method="highs",
    )
    starts = [np.zeros(n_expert, dtype=np.float64)]
    if tangent.success and tangent.x[-1] > 0.0:
        starts.extend(tangent.x[:-1] * scale for scale in (0.25, 0.50, 1.0))
    rng = np.random.default_rng(seed)
    for _ in range(4):
        random = rng.uniform(size=n_expert)
        random *= min(float(budget) * 0.5, 0.5) / random.sum()
        starts.append(random)

    bounds = [(0.0, MAX_EXPERT_DOSE)] * n_expert + [(-100.0, 100.0)]
    best: dict[str, Any] | None = None
    for start in starts:
        initial_t = min(form_gain(start, forms[name]) for name in origins)
        fitted = minimize(
            objective,
            np.r_[start, initial_t],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-11, "disp": False},
        )
        violation = max(
            [0.0] + [-float(constraint["fun"](fitted.x)) for constraint in constraints]
        )
        result = {
            "success": bool(fitted.success),
            "message": str(fitted.message),
            "constraint_violation": float(violation),
            "minimum_origin_gain": float(fitted.x[-1]),
            "weight": fitted.x[:-1].astype(np.float64),
        }
        if best is None or (result["constraint_violation"], -result["minimum_origin_gain"]) < (
            best["constraint_violation"], -best["minimum_origin_gain"]
        ):
            best = result
    if best is None:
        raise RuntimeError("minimax solver produced no result")
    return best


def apply_portfolio(
    parent: np.ndarray,
    directions: np.ndarray,
    weight: np.ndarray,
    regular: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    output = parent.copy()
    active = np.asarray(regular, dtype=bool) & np.any(np.abs(directions) > 0.0, axis=1)
    output[active] = np.clip(
        parent[active] + directions[active] @ np.asarray(weight, dtype=np.float64),
        0.001,
        0.999,
    )
    return output, active


def _held_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["gain"] > 0.0
        and metrics["positive_month_fraction"] >= 0.50
        and metrics["worst_month_gain"] > -2.0
    )


def run(
    train_csv: Path,
    v285_axes: Path,
    v318_axes: Path,
    expert_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "season", "game_month", "game_type", "pitcher_id", "batter_id",
        "pitcher_team_id", "batter_team_id", TARGET,
    ]
    train = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parents = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v318_axes, allow_pickle=False) as saved:
        parents.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )

    direction_by_origin = {
        origin: np.column_stack(
            [_candidate_minus_parent(expert_paths[name], origin) for name in EXPERT_ORDER]
        )
        for origin in ORIGINS
    }
    routes = {
        origin: np.select(
            [
                ~frames[origin]["game_type"].astype(str).eq("R").to_numpy(),
                (
                    frames[origin]["pitcher_team_id"].eq(13)
                    | frames[origin]["batter_team_id"].eq(13)
                ).to_numpy(),
            ],
            ["F", "R_ANCHOR"],
            default="R_CORE",
        )
        for origin in ORIGINS
    }
    regular = {
        origin: frames[origin]["game_type"].astype(str).eq("R").to_numpy()
        for origin in ORIGINS
    }
    # The expert library is allowed only on R; F remains bit-identical to v320.
    for origin in ORIGINS:
        direction_by_origin[origin][~regular[origin], :] = 0.0

    forms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    safety_by_origin: dict[str, list[str]] = {}
    for origin in ORIGINS:
        target = frames[origin][TARGET].to_numpy(np.float64)
        directions = direction_by_origin[origin]
        forms[origin] = quadratic_gain_form(target, parents[origin], directions, regular[origin])
        safety_by_origin[origin] = []
        months = frames[origin]["game_month"].to_numpy(np.int16)
        origin_routes = routes[origin]
        for month in sorted(np.unique(months[regular[origin]]).tolist()):
            key = f"{origin}|month={int(month)}"
            mask = regular[origin] & (months == month)
            forms[key] = quadratic_gain_form(target, parents[origin], directions, mask)
            safety_by_origin[origin].append(key)
        for route in ("R_CORE", "R_ANCHOR"):
            key = f"{origin}|route={route}"
            mask = regular[origin] & (origin_routes == route)
            forms[key] = quadratic_gain_form(target, parents[origin], directions, mask)
            safety_by_origin[origin].append(key)

    loo_rows: list[dict[str, Any]] = []
    loo_detail: dict[str, Any] = {}
    for budget in BUDGETS:
        for held in ORIGINS:
            fitted_origins = [origin for origin in ORIGINS if origin != held]
            fitted_safety = [
                key for origin in fitted_origins for key in safety_by_origin[origin]
            ]
            solved = solve_minimax(
                forms, fitted_origins, fitted_safety, budget,
                seed=33100 + int(100 * budget) + ORIGINS.index(held),
            )
            weight = solved.pop("weight")
            candidate, active = apply_portfolio(
                parents[held], direction_by_origin[held], weight, regular[held]
            )
            metrics = axis_metrics(frames[held], parents[held], candidate, active)
            # Explicit route gains are held-out checks, not fit constraints.
            target = frames[held][TARGET].to_numpy(np.float64)
            held_routes = routes[held]
            route_gains = {}
            for route in ("R_CORE", "R_ANCHOR"):
                mask = regular[held] & (held_routes == route)
                route_gains[route] = form_gain(weight, quadratic_gain_form(
                    target, parents[held], direction_by_origin[held], mask
                ))
            passed = bool(_held_pass(metrics) and min(route_gains.values()) >= 0.0)
            key = f"b{budget:g}__held_{held}"
            loo_detail[key] = {
                "solver": solved,
                "weights": dict(zip(EXPERT_ORDER, map(float, weight))),
                "metrics": metrics,
                "route_gains": route_gains,
                "passed": passed,
            }
            loo_rows.append(
                {
                    "budget": budget,
                    "held_origin": held,
                    "passed": passed,
                    "gain": metrics["gain"],
                    "positive_month_fraction": metrics["positive_month_fraction"],
                    "worst_month_gain": metrics["worst_month_gain"],
                    "minimum_route_gain": min(route_gains.values()),
                    "dose": float(weight.sum()),
                }
            )
    loo = pd.DataFrame(loo_rows)
    budget_rows = []
    for budget in BUDGETS:
        part = loo.loc[loo["budget"].eq(budget)]
        budget_rows.append(
            {
                "budget": budget,
                "all_held_passed": bool(part["passed"].all()),
                "minimum_held_gain": float(part["gain"].min()),
                "mean_held_gain": float(part["gain"].mean()),
                "worst_held_month_gain": float(part["worst_month_gain"].min()),
                "minimum_held_route_gain": float(part["minimum_route_gain"].min()),
            }
        )
    ranking = pd.DataFrame(budget_rows).sort_values(
        ["all_held_passed", "minimum_held_gain", "mean_held_gain"],
        ascending=[False, False, False], kind="stable",
    )
    selected_budget = float(ranking.iloc[0]["budget"])
    all_safety = [key for origin in ORIGINS for key in safety_by_origin[origin]]
    final = solve_minimax(
        forms, list(ORIGINS), all_safety, selected_budget,
        seed=33200 + int(100 * selected_budget),
    )
    weight = final.pop("weight")
    candidates: dict[str, np.ndarray] = {}
    actives: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        candidates[origin], actives[origin] = apply_portfolio(
            parents[origin], direction_by_origin[origin], weight, regular[origin]
        )
        metrics[origin] = axis_metrics(
            frames[origin], parents[origin], candidates[origin], actives[origin]
        )

    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], candidates["full_2024"],
        actives["full_2024"], [parents["full_2024"], candidates["full_2024"]],
    )
    loo_pass = bool(ranking.iloc[0]["all_held_passed"])
    final_point_pass = bool(
        min(metrics[origin]["gain"] for origin in ORIGINS) > 0.0
        and all(metrics[origin]["positive_month_fraction"] >= 0.50 for origin in ORIGINS)
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] <= 0.10
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{origin}": parents[origin] for origin in ORIGINS},
        **{f"candidate_{origin}": candidates[origin] for origin in ORIGINS},
        **{f"active_{origin}": actives[origin] for origin in ORIGINS},
        expert_names=np.asarray(EXPERT_ORDER),
        expert_weights=weight,
    )
    loo.to_csv(output_dir / "leave_one_origin_results.csv", index=False)
    ranking.to_csv(output_dir / "budget_ranking.csv", index=False)
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_candidate" if loo_pass and final_point_pass and robust_pass else (
            "loo_reject" if not loo_pass else "locked_reject"
        ),
        "selected_budget": selected_budget,
        "budget_ranking": ranking.to_dict(orient="records"),
        "leave_one_origin_detail": loo_detail,
        "final_solver": final,
        "final_weights": dict(zip(EXPERT_ORDER, map(float, weight))),
        "final_metrics": metrics,
        "full_2024_robustness": robustness,
        "loo_gate_passed": loo_pass,
        "final_point_gate_passed": final_point_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(loo_pass and final_point_pass and robust_pass),
        "restrictions": {
            "official_train_only": True,
            "v320_f_route_bit_identical": True,
            "budget_selected_by_leave_one_origin_out": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v318-axes", type=Path, required=True)
    for name in EXPERT_ORDER:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expert_paths = {name: getattr(args, name) for name in EXPERT_ORDER}
    print(json.dumps(run(
        args.train_csv, args.v285_axes, args.v318_axes, expert_paths, args.output_dir
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

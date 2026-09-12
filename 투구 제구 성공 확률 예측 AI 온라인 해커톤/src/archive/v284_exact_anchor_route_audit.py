"""Source-selected routing of the exact-anchor XGB as an independent expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.archive.v248_fixed_route_dual_tree_fallback import _load_axes


PROTOCOL = "V284_EXACT_ANCHOR_ROUTE_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
YEAR_BY_AXIS = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
WEIGHTS = (0.05, 0.10, 0.15, 0.25, 0.40)


def route_library(frame, baseline: np.ndarray, exact: np.ndarray) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    core = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    pressure = pressure_gate(frame)
    same = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).to_numpy()
    two = frame["strikes_before"].eq(2).to_numpy()
    runner = frame["num_runners_on"].gt(0).to_numpy()
    balls_ahead = frame["balls_before"].gt(frame["strikes_before"]).to_numpy()
    delta = np.asarray(exact, np.float64) - np.asarray(baseline, np.float64)
    finite = np.isfinite(exact)
    routes = {
        "core_all": core,
        "pressure": core & pressure,
        "nonpressure": core & ~pressure,
        "same_hand": core & same,
        "opposite_hand": core & ~same,
        "pressure_same": core & pressure & same,
        "pressure_opposite": core & pressure & ~same,
        "nonpressure_same": core & ~pressure & same,
        "nonpressure_opposite": core & ~pressure & ~same,
        "two_strike": core & two,
        "not_two_strike": core & ~two,
        "runner_on": core & runner,
        "bases_empty": core & ~runner,
        "balls_ahead": core & balls_ahead,
        "not_balls_ahead": core & ~balls_ahead,
        "base_lt046": core & (baseline < 0.46),
        "base_046_050": core & (baseline >= 0.46) & (baseline < 0.50),
        "base_050_054": core & (baseline >= 0.50) & (baseline < 0.54),
        "base_ge054": core & (baseline >= 0.54),
        "exact_up": core & (delta > 0.0),
        "exact_down": core & (delta <= 0.0),
        "agreement_001": core & (np.abs(delta) <= 0.01),
        "agreement_002": core & (np.abs(delta) <= 0.02),
        "disagreement_002": core & (np.abs(delta) > 0.02),
        "nonpressure_same_agree002": (
            core & ~pressure & same & (np.abs(delta) <= 0.02)
        ),
        "pressure_agree002": core & pressure & (np.abs(delta) <= 0.02),
    }
    return {name: mask & finite for name, mask in routes.items()}


def apply(
    baseline: np.ndarray,
    exact: np.ndarray,
    route: np.ndarray,
    weight: float,
) -> np.ndarray:
    candidate = np.asarray(baseline, np.float64).copy()
    candidate[route] = np.clip(
        baseline[route] + float(weight) * (exact[route] - baseline[route]),
        0.001,
        0.999,
    )
    return candidate


def run(
    train_csv: Path,
    exact_oof_dir: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, _correction = _load_year_context(train_csv)
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022].reset_index(drop=True),
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024].reset_index(drop=True),
    }
    axes = _load_axes(contract_dir, bridge_oof)
    with np.load(v244_axes, allow_pickle=False) as saved:
        baseline = {
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(
                np.float64
            )
            for axis in AXES
        }
    exact_full = {
        year: align_regular_prediction(
            frames[year],
            exact_oof_dir / f"training_parity_exact_xgb_{year}.npy",
        )
        for year in (2022, 2023, 2024)
    }
    exact = {
        "full_2022": exact_full[2022],
        "late_2023": exact_full[2023][late23],
        "full_2024": exact_full[2024],
    }
    routes = {
        axis: route_library(axis_frames[axis], baseline[axis], exact[axis])
        for axis in AXES
    }
    route_names = tuple(routes["full_2022"])
    trials: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for route_name in route_names:
        for weight in WEIGHTS:
            key = f"{route_name}__w{weight:g}"
            per_axis = {}
            for axis in AXES:
                candidate = apply(
                    baseline[axis], exact[axis], routes[axis][route_name], weight
                )
                predictions[(key, axis)] = candidate
                per_axis[axis] = paired_metrics(
                    axes[axis], baseline[axis], candidate, routes[axis][route_name]
                )
            source_pass = all(
                per_axis[axis]["gain"] > 0.0
                and per_axis[axis]["positive_month_fraction"] >= 0.5
                and per_axis[axis]["worst_month_gain"] > -10.0
                for axis in SOURCE_AXES
            )
            trials.append(
                {
                    "key": key,
                    "route": route_name,
                    "weight": float(weight),
                    "source_gate_passed": bool(source_pass),
                    "source_min_gain": float(
                        min(per_axis[a]["gain"] for a in SOURCE_AXES)
                    ),
                    "source_mean_gain": float(
                        np.mean([per_axis[a]["gain"] for a in SOURCE_AXES])
                    ),
                    "source_worst_month": float(
                        min(per_axis[a]["worst_month_gain"] for a in SOURCE_AXES)
                    ),
                    "locked_gain": float(per_axis["full_2024"]["gain"]),
                    "locked_positive_month_fraction": float(
                        per_axis["full_2024"]["positive_month_fraction"]
                    ),
                    "locked_worst_month": float(
                        per_axis["full_2024"]["worst_month_gain"]
                    ),
                }
            )
            details[key] = per_axis
    ranking = pd.DataFrame(trials)
    eligible = ranking.loc[ranking["source_gate_passed"]]
    selection = eligible if len(eligible) else ranking
    selected = selection.sort_values(
        ["source_min_gain", "source_mean_gain", "source_worst_month"],
        ascending=False,
        kind="stable",
    ).iloc[0].to_dict()
    selected_key = str(selected["key"])
    route_name = str(selected["route"])
    locked_family = [
        predictions[(f"{route_name}__w{weight:g}", "full_2024")]
        for weight in WEIGHTS
    ]
    robustness = _robustness(
        axes["full_2024"],
        baseline["full_2024"],
        predictions[(selected_key, "full_2024")],
        routes["full_2024"][route_name],
        locked_family,
    )
    locked = details[selected_key]["full_2024"]
    locked_pass = bool(
        locked["gain"] >= 4.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -10.0
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    promote = bool(selected["source_gate_passed"] and locked_pass and robust_pass)
    ranking.sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain"],
        ascending=False,
        kind="stable",
    ).to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"baseline_{axis}": baseline[axis] for axis in AXES},
        **{f"exact_{axis}": exact[axis] for axis in AXES},
        **{f"route_{axis}": routes[axis][route_name] for axis in AXES},
        **{
            f"candidate_{axis}": predictions[(selected_key, axis)]
            for axis in AXES
        },
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "promote_to_release_build" if promote else "reject",
        "selected": selected,
        "selected_metrics": details[selected_key],
        "locked_robustness": robustness,
        "locked_point_passed": locked_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": promote,
        "route_count": len(route_names),
        "trial_count": len(trials),
        "restrictions": {
            "official_train_only": True,
            "exact_anchor_model_strict_forward": True,
            "route_and_weight_selected_on_2022_and_late2023_only": True,
            "full_2024_used_for_selection": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
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
    parser.add_argument("--exact-oof-dir", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.exact_oof_dir,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": result["selected"],
                "metrics": result["selected_metrics"],
                "robustness": result["locked_robustness"],
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()

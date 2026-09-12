"""Select a baseball-context route for the frozen v188 pressure correction.

The v188 profile/ridge/eta recipe remains frozen.  Route selection uses only
early-2022 to late-2022 and full-2022 to late-2023 corrections; 2024 is opened
after a route passes both source checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v178_row_region_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
)
from src.archive.v187_pitcher_pressure_profile_residual import (
    V178_SCALE,
    apply_correction,
    attach_profile,
    condition_frame,
    fit_ridge,
    predict_ridge,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V189_PRESSURE_PROFILE_ROUTE_SCREEN_V1"
Route = Callable[[pd.DataFrame], np.ndarray]


def route_library() -> dict[str, Route]:
    def conditions(frame: pd.DataFrame) -> pd.DataFrame:
        return condition_frame(frame.reset_index(drop=True)).astype(bool)

    return {
        "all": lambda frame: np.ones(len(frame), dtype=bool),
        "high_li": lambda frame: conditions(frame)["high_li"].to_numpy(),
        "extreme_li": lambda frame: conditions(frame)["extreme_li"].to_numpy(),
        "traffic": lambda frame: conditions(frame)["traffic"].to_numpy(),
        "risp": lambda frame: conditions(frame)["risp"].to_numpy(),
        "late": lambda frame: conditions(frame)["late"].to_numpy(),
        "close": lambda frame: conditions(frame)["close"].to_numpy(),
        "behind": lambda frame: conditions(frame)["behind"].to_numpy(),
        "three_ball": lambda frame: conditions(frame)["three_ball"].to_numpy(),
        "two_strike": lambda frame: conditions(frame)["two_strike"].to_numpy(),
        "full_count": lambda frame: conditions(frame)["full_count"].to_numpy(),
        "compound_pressure": lambda frame: conditions(frame)["compound_pressure"].to_numpy(),
        "high_li_or_traffic": lambda frame: (
            conditions(frame)["high_li"] | conditions(frame)["traffic"]
        ).to_numpy(),
        "traffic_or_pressure_count": lambda frame: (
            conditions(frame)["traffic"]
            | conditions(frame)["three_ball"]
            | conditions(frame)["two_strike"]
        ).to_numpy(),
        "high_li_or_pressure_count": lambda frame: (
            conditions(frame)["high_li"]
            | conditions(frame)["three_ball"]
            | conditions(frame)["two_strike"]
        ).to_numpy(),
        "high_li_and_traffic": lambda frame: (
            conditions(frame)["high_li"] & conditions(frame)["traffic"]
        ).to_numpy(),
        "close_pressure": lambda frame: (
            conditions(frame)["close"]
            & (
                conditions(frame)["traffic"]
                | conditions(frame)["three_ball"]
                | conditions(frame)["two_strike"]
            )
        ).to_numpy(),
    }


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in axis.items()}


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= (2.0 / 3.0)
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
    v188_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frozen = json.loads(v188_summary.read_text(encoding="utf-8"))
    if frozen.get("protocol") != "V188_PRESSURE_PROFILE_SOFT_TRANSITION_V1":
        raise ValueError("expected frozen v188 summary")
    selected = frozen["selected_on_soft_sources_only"]
    profile_alpha = float(selected["profile_alpha"])
    ridge_alpha = float(selected["ridge_alpha"])
    eta = float(selected["eta"])

    full_train = pd.read_csv(train_csv, low_memory=False)
    _context, raw_frames, post4 = _load_year_context(train_csv)
    frames = {
        "full_2022": full_train.loc[full_train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": full_train.loc[
            full_train["season"].eq(2023) & full_train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": full_train.loc[full_train["season"].eq(2024)].reset_index(drop=True),
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, post4, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    weights = load_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in ("full_2022", "late_2023", "full_2024")
        }
    direction = {
        name: load_direction(name, v158_base[name], weights, library_root)
        for name in v158_base
    }
    base = {
        name: apply_direction(parents[name], direction[name], V178_SCALE)
        for name in parents
    }
    features = {
        "full_2022": attach_profile(
            frames["full_2022"], full_train.loc[full_train["season"].lt(2022)], profile_alpha
        ),
        "late_2023": attach_profile(
            frames["late_2023"], full_train.loc[full_train["season"].lt(2023)], profile_alpha
        ),
        "full_2024": attach_profile(
            frames["full_2024"], full_train.loc[full_train["season"].lt(2024)], profile_alpha
        ),
    }
    exact22 = np.asarray(axes["full_2022"]["exact_mask"], dtype=bool)
    exact23 = np.asarray(axes["late_2023"]["exact_mask"], dtype=bool)
    early22 = frames["full_2022"]["game_month"].le(7).to_numpy()
    late22 = ~early22
    residual22 = np.asarray(axes["full_2022"]["target"]) - base["full_2022"]
    x22 = features["full_2022"].to_numpy(np.float64)
    spec_early = fit_ridge(
        x22[exact22 & early22], residual22[exact22 & early22], ridge_alpha
    )
    correction22 = predict_ridge(spec_early, x22[late22])
    spec_full = fit_ridge(x22[exact22], residual22[exact22], ridge_alpha)
    correction23 = predict_ridge(
        spec_full, features["late_2023"].to_numpy(np.float64)
    )
    axis_late22 = _slice_axis(axes["full_2022"], late22)

    rows: list[dict[str, Any]] = []
    source_details: dict[str, Any] = {}
    for route_name, route in route_library().items():
        active22 = exact22[late22] & route(frames["full_2022"].loc[late22].reset_index(drop=True))
        active23 = exact23 & route(frames["late_2023"])
        candidate22 = apply_correction(
            base["full_2022"][late22], correction22, active22, eta
        )
        candidate23 = apply_correction(base["late_2023"], correction23, active23, eta)
        detail = {
            "early22_to_late22": metrics(
                axis_late22, base["full_2022"][late22], candidate22
            ),
            "full22_to_late23": metrics(
                axes["late_2023"], base["late_2023"], candidate23
            ),
        }
        passed = all(source_gate(item) for item in detail.values())
        source_details[route_name] = detail
        rows.append(
            {
                "route": route_name,
                "source_gate_passed": passed,
                "minimum_gain": min(item["gain"] for item in detail.values()),
                "mean_gain": float(np.mean([item["gain"] for item in detail.values()])),
                "minimum_positive_month_fraction": min(
                    item["positive_month_fraction"] for item in detail.values()
                ),
                "worst_month_gain": min(item["worst_month_gain"] for item in detail.values()),
                "minimum_active_fraction": min(item["active_fraction"] for item in detail.values()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=[False, False, False, False], kind="stable",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_route_ranking.csv", index=False, encoding="utf-8-sig")
    passing = ranking.loc[ranking["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "frozen_v188_recipe": selected,
            "source_top": ranking.head(10).to_dict(orient="records"),
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_route = str(passing.iloc[0]["route"])
        exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        residual23 = np.asarray(axes["late_2023"]["target"])[exact23] - base["late_2023"][exact23]
        fit_x = np.vstack(
            [x22[exact22], features["late_2023"].to_numpy(np.float64)[exact23]]
        )
        fit_y = np.concatenate([residual22[exact22], residual23])
        fit_weight = np.concatenate(
            [np.full(int(exact22.sum()), 0.55), np.ones(len(residual23))]
        )
        final_spec = fit_ridge(fit_x, fit_y, ridge_alpha, fit_weight)
        correction24 = predict_ridge(
            final_spec, features["full_2024"].to_numpy(np.float64)
        )
        active24 = exact24 & route_library()[selected_route](frames["full_2024"])
        candidate24 = apply_correction(base["full_2024"], correction24, active24, eta)
        locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
        incremental = metrics(axes["full_2024"], base["full_2024"], candidate24)
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate24,
            active24, [candidate24, base["full_2024"]],
        )
        point_pass = bool(
            locked["gain"] > 0.0 and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -5.0 and locked["minimum_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_model.npz",
            mean=final_spec.mean, scale=final_spec.scale,
            coefficient=final_spec.coefficient, profile_alpha=profile_alpha,
            ridge_alpha=ridge_alpha, eta=eta, route=selected_route,
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"], base=base["full_2024"],
            candidate=candidate24, correction=correction24, active=active24,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "frozen_v188_recipe": selected,
            "selected_route_on_sources_only": selected_route,
            "source": source_details[selected_route],
            "locked_2024": locked,
            "incremental_over_v178": incremental,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
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
        "frozen_v188_profile_ridge_eta": True,
        "route_selected_on_sources_only": True,
        "official_train_labels_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_prediction_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
        "locked_2024_development_contaminated": True,
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
    parser.add_argument("--v188-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.v188_summary, args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

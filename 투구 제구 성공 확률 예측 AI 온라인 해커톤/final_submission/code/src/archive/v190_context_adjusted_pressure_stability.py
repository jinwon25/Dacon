"""Context-adjusted, season-stable pitcher pressure residual.

Observable count/base/hand/inning/leverage context is removed from historical
labels with leave-one-out smoothing.  Pitcher pressure effects are then shrunk
and multiplied by their across-season sign consistency.  Query-season labels
never enter the profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    fit_ridge,
    predict_ridge,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V190_CONTEXT_ADJUSTED_PRESSURE_STABILITY_V1"
CONTEXT_ALPHA = 400.0
PROFILE_ALPHAS = (100.0, 300.0, 600.0)
RIDGE_ALPHAS = (1000.0, 10000.0, 30000.0)
ETA_GRID = (0.05, 0.1, 0.2, 0.4)
PRESSURE_NAMES = (
    "high_li", "extreme_li", "traffic", "risp", "two_out_risp",
    "three_ball", "full_count", "late_high_li", "close_high_li",
    "compound_crisis",
)


def context_and_pressure(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.reset_index(drop=True)
    balls = pd.to_numeric(data["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(data["strikes_before"], errors="coerce").fillna(0).astype(int)
    outs = pd.to_numeric(data["outs_before"], errors="coerce").fillna(0).astype(int)
    inning = pd.to_numeric(data["inning"], errors="coerce").fillna(0.0)
    leverage = pd.to_numeric(data["li"], errors="coerce").fillna(0.0)
    score = pd.to_numeric(data["score_diff_pitcher_team"], errors="coerce").fillna(0.0)
    runners = pd.to_numeric(data["num_runners_on"], errors="coerce").fillna(0).astype(int)
    risp = data["runner_on_2b"].fillna(0).astype(bool) | data["runner_on_3b"].fillna(0).astype(bool)
    context = pd.DataFrame(
        {
            "count": balls.astype(str) + "-" + strikes.astype(str),
            "outs": outs.astype(str),
            "base": data["base_state"].astype("string").fillna("NA").astype(str),
            "hand": data["pitcher_hand"].astype(str) + "-" + data["batter_hand"].astype(str),
            "inning_band": pd.cut(
                inning, [-np.inf, 3, 6, 9, np.inf], labels=False
            ).fillna(-1).astype(int).astype(str),
            "leverage_band": pd.cut(
                leverage, [-np.inf, 0.75, 1.5, 3, np.inf], labels=False
            ).fillna(-1).astype(int).astype(str),
            "home_side": data["top_bottom"].astype(str),
            "score_band": pd.cut(
                score, [-np.inf, -3, -1, 1, 3, np.inf], labels=False
            ).fillna(-1).astype(int).astype(str),
        }
    )
    pressure = pd.DataFrame(
        {
            "high_li": leverage >= 1.5,
            "extreme_li": leverage >= 3.0,
            "traffic": runners > 0,
            "risp": risp,
            "two_out_risp": (outs == 2) & risp,
            "three_ball": balls == 3,
            "full_count": (balls == 3) & (strikes == 2),
            "late_high_li": (inning >= 7) & (leverage >= 1.5),
            "close_high_li": (score.abs() <= 1) & (leverage >= 1.5),
            "compound_crisis": (leverage >= 1.5) & risp & (balls >= 2),
        }
    ).astype(np.int8)
    return context, pressure


def context_adjusted_residual(
    history: pd.DataFrame,
    alpha: float = CONTEXT_ALPHA,
) -> np.ndarray:
    data = history.reset_index(drop=True)
    target = data["control_success"].to_numpy(np.float64)
    prior = float(np.mean(target))
    context, _pressure = context_and_pressure(data)
    levels = (
        ("count", "outs", "base", "hand", "inning_band", "leverage_band", "home_side"),
        ("count", "outs", "base", "hand"),
        ("count", "outs", "base"),
        ("count", "hand"),
    )
    estimates = []
    reliability = []
    for columns in levels:
        work = context.loc[:, list(columns)].copy()
        work["target"] = target
        grouped = work.groupby(list(columns), dropna=False, observed=True)["target"]
        group_sum = grouped.transform("sum").to_numpy(np.float64)
        group_count = grouped.transform("count").to_numpy(np.float64)
        prior_count = np.maximum(group_count - 1.0, 0.0)
        estimates.append((group_sum - target + float(alpha) * prior) / (prior_count + float(alpha)))
        reliability.append(prior_count / (prior_count + float(alpha)))
    estimate_matrix = np.column_stack(estimates)
    weight_matrix = 0.25 + np.column_stack(reliability)
    expected = np.sum(estimate_matrix * weight_matrix, axis=1) / np.sum(weight_matrix, axis=1)
    return target - expected


def stable_pressure_features(
    query: pd.DataFrame,
    history: pd.DataFrame,
    profile_alpha: float,
) -> pd.DataFrame:
    data = history.reset_index(drop=True)
    residual = context_adjusted_residual(data)
    _context, pressure = context_and_pressure(data)
    _query_context, query_pressure = context_and_pressure(query)
    pitcher = data["pitcher_id"].astype(str)
    query_pitcher = query["pitcher_id"].astype(str)
    global_residual = float(np.mean(residual))
    overall = pd.DataFrame({"pitcher": pitcher, "residual": residual}).groupby(
        "pitcher", sort=False
    )["residual"].agg(["sum", "count"])
    overall_rate = (overall["sum"] + profile_alpha * global_residual) / (
        overall["count"] + profile_alpha
    )
    output = pd.DataFrame(index=query.reset_index(drop=True).index)
    for name in PRESSURE_NAMES:
        active_history = pressure[name].to_numpy(bool)
        grouped = pd.DataFrame(
            {
                "pitcher": pitcher[active_history],
                "residual": residual[active_history],
            }
        ).groupby("pitcher", sort=False)["residual"].agg(["sum", "count"])
        sums = grouped["sum"].reindex(overall.index).fillna(0.0)
        counts = grouped["count"].reindex(overall.index).fillna(0.0)
        rate = (sums + profile_alpha * overall_rate) / (counts + profile_alpha)
        effect = rate - overall_rate
        reliability = counts / (counts + profile_alpha)

        season_rows = []
        for (_season, _pitcher), index in data.groupby(["season", "pitcher_id"], sort=False).groups.items():
            positions = np.asarray(list(index), dtype=np.int64)
            selected = positions[pressure.loc[positions, name].to_numpy(bool)]
            if len(selected):
                season_rows.append(
                    {
                        "pitcher": str(_pitcher),
                        "season": int(_season),
                        "effect": float(np.mean(residual[selected])),
                    }
                )
        if season_rows:
            season_effect = pd.DataFrame(season_rows)
            stability = season_effect.groupby("pitcher")["effect"].apply(
                lambda values: float(abs(np.sign(values).mean()))
            )
        else:
            stability = pd.Series(dtype=float)
        stable = effect * stability.reindex(overall.index).fillna(0.0)

        mapped_effect = query_pitcher.map(effect).fillna(0.0).to_numpy(np.float64)
        mapped_stable = query_pitcher.map(stable).fillna(0.0).to_numpy(np.float64)
        mapped_reliability = query_pitcher.map(reliability).fillna(0.0).to_numpy(np.float64)
        active_query = query_pressure[name].to_numpy(np.float64)
        output[f"{name}_effect"] = mapped_effect * active_query
        output[f"{name}_stable"] = mapped_stable * active_query
        output[f"{name}_reliability"] = mapped_reliability * active_query
    return output.astype(np.float32)


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
    output_dir: Path,
) -> dict[str, Any]:
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
    directions = {
        name: load_direction(name, v158_base[name], weights, library_root)
        for name in v158_base
    }
    base = {
        name: apply_direction(parents[name], directions[name], V178_SCALE)
        for name in parents
    }
    exact22 = np.asarray(axes["full_2022"]["exact_mask"], dtype=bool)
    exact23 = np.asarray(axes["late_2023"]["exact_mask"], dtype=bool)
    early22 = frames["full_2022"]["game_month"].le(7).to_numpy()
    late22 = ~early22
    axis_late22 = _slice_axis(axes["full_2022"], late22)
    history = {
        2022: full_train.loc[full_train["season"].lt(2022)],
        2023: full_train.loc[full_train["season"].lt(2023)],
        2024: full_train.loc[full_train["season"].lt(2024)],
    }
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    feature_cache: dict[float, dict[str, pd.DataFrame]] = {}
    for profile_alpha in PROFILE_ALPHAS:
        features = {
            "full_2022": stable_pressure_features(
                frames["full_2022"], history[2022], profile_alpha
            ),
            "late_2023": stable_pressure_features(
                frames["late_2023"], history[2023], profile_alpha
            ),
            "full_2024": stable_pressure_features(
                frames["full_2024"], history[2024], profile_alpha
            ),
        }
        feature_cache[profile_alpha] = features
        x22 = features["full_2022"].to_numpy(np.float64)
        x23 = features["late_2023"].to_numpy(np.float64)
        residual22 = np.asarray(axes["full_2022"]["target"]) - base["full_2022"]
        for ridge_alpha in RIDGE_ALPHAS:
            early_spec = fit_ridge(
                x22[exact22 & early22], residual22[exact22 & early22], ridge_alpha
            )
            correction22 = predict_ridge(early_spec, x22[late22])
            full_spec = fit_ridge(x22[exact22], residual22[exact22], ridge_alpha)
            correction23 = predict_ridge(full_spec, x23)
            for eta in ETA_GRID:
                key = f"p{profile_alpha:g}__r{ridge_alpha:g}__e{eta:g}"
                candidate22 = apply_correction(
                    base["full_2022"][late22], correction22, exact22[late22], eta
                )
                candidate23 = apply_correction(base["late_2023"], correction23, exact23, eta)
                detail = {
                    "early22_to_late22": metrics(
                        axis_late22, base["full_2022"][late22], candidate22
                    ),
                    "full22_to_late23": metrics(
                        axes["late_2023"], base["late_2023"], candidate23
                    ),
                }
                passed = all(source_gate(item) for item in detail.values())
                details[key] = detail
                rows.append(
                    {
                        "key": key, "profile_alpha": profile_alpha,
                        "ridge_alpha": ridge_alpha, "eta": eta,
                        "source_gate_passed": passed,
                        "minimum_gain": min(item["gain"] for item in detail.values()),
                        "mean_gain": float(np.mean([item["gain"] for item in detail.values()])),
                        "minimum_positive_month_fraction": min(
                            item["positive_month_fraction"] for item in detail.values()
                        ),
                        "worst_month_gain": min(item["worst_month_gain"] for item in detail.values()),
                    }
                )
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=[False, False, False, False], kind="stable",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    passing = ranking.loc[ranking["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source_top": ranking.head(10).to_dict(orient="records"),
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected = passing.iloc[0]
        profile_alpha = float(selected["profile_alpha"])
        ridge_alpha = float(selected["ridge_alpha"])
        eta = float(selected["eta"])
        features = feature_cache[profile_alpha]
        x22 = features["full_2022"].to_numpy(np.float64)
        x23 = features["late_2023"].to_numpy(np.float64)
        x24 = features["full_2024"].to_numpy(np.float64)
        residual22 = np.asarray(axes["full_2022"]["target"])[exact22] - base["full_2022"][exact22]
        residual23 = np.asarray(axes["late_2023"]["target"])[exact23] - base["late_2023"][exact23]
        fit_x = np.vstack([x22[exact22], x23[exact23]])
        fit_y = np.concatenate([residual22, residual23])
        fit_weight = np.concatenate(
            [np.full(len(residual22), 0.55), np.ones(len(residual23))]
        )
        final_spec = fit_ridge(fit_x, fit_y, ridge_alpha, fit_weight)
        correction24 = predict_ridge(final_spec, x24)
        exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        candidate24 = apply_correction(base["full_2024"], correction24, exact24, eta)
        locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
        incremental = metrics(axes["full_2024"], base["full_2024"], candidate24)
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], candidate24,
            exact24, [candidate24, base["full_2024"]],
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
            ridge_alpha=ridge_alpha, eta=eta,
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"], base=base["full_2024"],
            candidate=candidate24, correction=correction24, active=exact24,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "selected_on_sources_only": selected.to_dict(),
            "source": details[str(selected["key"])],
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
        "official_train_labels_only": True,
        "profiles_use_complete_prior_seasons_only": True,
        "leave_one_out_context_expectation": True,
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

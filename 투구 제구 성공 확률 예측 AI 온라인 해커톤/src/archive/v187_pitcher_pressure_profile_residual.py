"""Fit shrunk pitcher pressure-response profiles above JY plus v178.

Profiles use complete seasons strictly before the query season.  Every feature
is a within-pitcher context effect and is active only when that context occurs,
so the correction cannot learn a free league-level or calendar intercept.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v178_jy_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V187_PITCHER_PRESSURE_PROFILE_RESIDUAL_V1"
V178_SCALE = 0.25
PROFILE_ALPHAS = (100.0, 300.0, 500.0, 1000.0)
RIDGE_ALPHAS = (300.0, 1000.0, 3000.0, 10000.0, 30000.0)
ETA_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
CORRECTION_CAP = 0.02
CONDITIONS = (
    "high_li", "extreme_li", "traffic", "risp", "late", "close", "behind",
    "three_ball", "two_strike", "full_count", "compound_pressure",
)


@dataclass(frozen=True)
class RidgeSpec:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray


def condition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    li = pd.to_numeric(frame["li"], errors="coerce").fillna(0.0)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(0.0)
    score = pd.to_numeric(frame["score_diff_pitcher_team"], errors="coerce").fillna(0.0)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0.0)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0.0)
    runners = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0.0)
    risp = frame["runner_on_2b"].fillna(0).astype(bool) | frame["runner_on_3b"].fillna(0).astype(bool)
    return pd.DataFrame(
        {
            "high_li": li >= 1.5,
            "extreme_li": li >= 3.0,
            "traffic": runners > 0,
            "risp": risp,
            "late": inning >= 7,
            "close": score.abs() <= 1,
            "behind": score < 0,
            "three_ball": balls == 3,
            "two_strike": strikes == 2,
            "full_count": (balls == 3) & (strikes == 2),
            "compound_pressure": (li >= 1.5) & (risp | (balls == 3)) & (score.abs() <= 2),
        },
        index=frame.index,
    ).astype(np.int8)


def build_profile(history: pd.DataFrame, alpha: float) -> pd.DataFrame:
    history = history.reset_index(drop=True)
    target = history["control_success"].astype(float)
    pitcher = history["pitcher_id"].astype(str)
    conditions = condition_frame(history)
    global_rate = float(target.mean())
    overall = pd.DataFrame({"pitcher": pitcher, "target": target}).groupby(
        "pitcher", sort=False
    )["target"].agg(["sum", "count"])
    overall_rate = (overall["sum"] + float(alpha) * global_rate) / (
        overall["count"] + float(alpha)
    )
    profile = pd.DataFrame(index=overall.index)
    for name in CONDITIONS:
        active = conditions[name].astype(bool).to_numpy()
        grouped = pd.DataFrame(
            {"pitcher": pitcher[active], "target": target[active]}
        ).groupby("pitcher", sort=False)["target"].agg(["sum", "count"])
        sums = grouped["sum"].reindex(profile.index).fillna(0.0)
        counts = grouped["count"].reindex(profile.index).fillna(0.0)
        context_rate = (sums + float(alpha) * overall_rate) / (counts + float(alpha))
        profile[f"{name}_effect"] = context_rate - overall_rate
        profile[f"{name}_reliability"] = counts / (counts + float(alpha))
    return profile


def attach_profile(
    query: pd.DataFrame,
    history: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    profile = build_profile(history, alpha)
    pitcher = query["pitcher_id"].astype(str)
    conditions = condition_frame(query.reset_index(drop=True))
    joined = profile.reindex(pd.Index(pitcher)).reset_index(drop=True).fillna(0.0)
    output = pd.DataFrame(index=query.reset_index(drop=True).index)
    for name in CONDITIONS:
        active = conditions[name].to_numpy(np.float64)
        effect = joined[f"{name}_effect"].to_numpy(np.float64)
        reliability = joined[f"{name}_reliability"].to_numpy(np.float64)
        output[f"{name}_active_effect"] = effect * active
        output[f"{name}_active_reliability"] = reliability * active
    return output.astype(np.float32)


def fit_ridge(
    features: np.ndarray,
    residual: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> RidgeSpec:
    x = np.nan_to_num(np.asarray(features, dtype=np.float64))
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    model.fit(z, np.asarray(residual, dtype=np.float64), sample_weight=sample_weight)
    return RidgeSpec(mean=mean, scale=scale, coefficient=model.coef_.astype(np.float64))


def predict_ridge(spec: RidgeSpec, features: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(features, dtype=np.float64))
    correction = ((x - spec.mean) / spec.scale) @ spec.coefficient
    return np.clip(correction, -CORRECTION_CAP, CORRECTION_CAP)


def apply_correction(
    base: np.ndarray,
    correction: np.ndarray,
    active: np.ndarray,
    eta: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    output[active] = np.clip(
        output[active] + float(eta) * np.asarray(correction)[active], 0.001, 0.999
    )
    return output


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in axis.items()}


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= (2.0 / 3.0)
        and result["worst_month_gain"] > -5.0
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
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
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
    parents, parity = exact_jy_parents(
        axes, raw_frames, post4, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    frozen_weights = load_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in ("full_2022", "late_2023", "full_2024")
        }
    directions = {
        name: load_direction(name, v158_base[name], frozen_weights, library_root)
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

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    feature_cache: dict[float, dict[str, pd.DataFrame]] = {}
    for profile_alpha in PROFILE_ALPHAS:
        profile_features = {
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
        feature_cache[profile_alpha] = profile_features
        x22 = profile_features["full_2022"].to_numpy(np.float64)
        x23 = profile_features["late_2023"].to_numpy(np.float64)
        residual22 = np.asarray(axes["full_2022"]["target"]) - base["full_2022"]
        for ridge_alpha in RIDGE_ALPHAS:
            spec_early = fit_ridge(
                x22[exact22 & early22], residual22[exact22 & early22], ridge_alpha
            )
            correction22 = predict_ridge(spec_early, x22[late22])
            spec_full = fit_ridge(x22[exact22], residual22[exact22], ridge_alpha)
            correction23 = predict_ridge(spec_full, x23)
            for eta in ETA_GRID:
                key = f"p{profile_alpha:g}__r{ridge_alpha:g}__e{eta:g}"
                candidate22 = apply_correction(
                    base["full_2022"][late22], correction22, exact22[late22], eta
                )
                candidate23 = apply_correction(base["late_2023"], correction23, exact23, eta)
                result = {
                    "early22_to_late22": metrics(
                        axis_late22, base["full_2022"][late22], candidate22
                    ),
                    "full22_to_late23": metrics(
                        axes["late_2023"], base["late_2023"], candidate23
                    ),
                }
                passed = all(source_gate(item) for item in result.values())
                details[key] = result
                rows.append(
                    {
                        "key": key,
                        "profile_alpha": profile_alpha,
                        "ridge_alpha": ridge_alpha,
                        "eta": eta,
                        "source_gate_passed": passed,
                        "minimum_gain": min(item["gain"] for item in result.values()),
                        "mean_gain": float(np.mean([item["gain"] for item in result.values()])),
                        "minimum_positive_month_fraction": min(
                            item["positive_month_fraction"] for item in result.values()
                        ),
                        "worst_month_gain": min(item["worst_month_gain"] for item in result.values()),
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
            coefficient=final_spec.coefficient,
            profile_alpha=profile_alpha, ridge_alpha=ridge_alpha, eta=eta,
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
        "free_intercept_or_calendar_correction": False,
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

"""Audit a public-recipe current-season decomposition above exact v104.

The source recipe is the independently published D/X/H1 family from
``hoo743-ui/LG_Aimers09``.  Only the formula is reused: all prior tables are
rebuilt from the official train file and every query feature depends on that
row plus a frozen pre-season train-only bank.  Candidate family, route, ridge
strength and dose are selected on two source transfers.  Full 2024 is opened
once after that selection is frozen.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.recent_shared_exact_asof import _current_season_state, _make_season_bank
from src.v103_fixed_union_robust import _axis_metrics
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V127_HOO_CURRENT_STATE_REBASE_V1"
RATE_SPEC = (
    ("asof_pitcher_success_rate", "asof_pitcher_n", "pitcher_id", "succ", "pitch"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n", "pitcher_id", "mid", "pitch"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n", "pitcher_id", "ball", "pitch"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n", "pitcher_id", "rev", "pitch"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n", "pitcher_id", "str", "pitch"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", "pitcher_id", "fb", "mix"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", "pitcher_id", "bb", "mix"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", "pitcher_id", "os", "mix"),
    ("asof_batter_success_rate", "asof_batter_n", "batter_id", "bsucc", "bat"),
    ("asof_batter_middle_rate", "asof_batter_n", "batter_id", "bmid", "bat"),
)
KIND_SPEC = {
    "pitch": ("asof_pitcher_n", "pitcher_id"),
    "mix": ("asof_pitcher_pitchmix_n", "pitcher_id"),
    "bat": ("asof_batter_n", "batter_id"),
}
D_RATE_NAMES = tuple(item[3] for item in RATE_SPEC)
D_COLUMNS = tuple([f"cur_{name}" for name in D_RATE_NAMES] + [
    "cur_logn_pitch", "cur_logn_mix", "cur_logn_bat",
])
MISSING_COLUMNS = (
    "cur_mid", "cur_ball", "cur_rev", "cur_str", "cur_fb", "cur_bb", "cur_os",
    "cur_bmid", "cur_logn_mix",
)
DELTA_COLUMNS = (
    "delta_succ", "delta_mid", "delta_ball", "delta_rev", "delta_str",
    "delta_fb", "delta_bb", "delta_os", "delta_bsucc", "cur_bmid",
    "bmid_minus_career",
)
CTX_COLUMNS = tuple(
    f"dx_{rate}_{context}"
    for rate in ("succ", "mid")
    for context in ("adv", "onb", "sh", "bs")
)
H1_COLUMNS = tuple(
    f"lx_{rate}_{context}"
    for rate in ("ball", "rev", "str")
    for context in ("sh", "bs")
)
FAMILIES = {
    "hoo_d13": D_COLUMNS,
    "missing_state9": MISSING_COLUMNS,
    "delta_state11": DELTA_COLUMNS,
    "hoo_context14": (*CTX_COLUMNS, *H1_COLUMNS),
    "missing_plus_context23": (*MISSING_COLUMNS, *CTX_COLUMNS, *H1_COLUMNS),
    "full_dxh1_27": (*D_COLUMNS, *CTX_COLUMNS, *H1_COLUMNS),
}


def build_prior_bank(train: pd.DataFrame, forecast_year: int) -> dict[str, Any]:
    """Build the exact public-recipe pre-season count/event tables."""
    history = train.loc[train["season"].lt(forecast_year)]
    bank: dict[str, Any] = {"forecast_year": int(forecast_year), "kinds": {}}
    for kind, (n_column, id_column) in KIND_SPEC.items():
        ids = history[id_column]
        table = pd.DataFrame(index=pd.Index(ids.unique(), name=id_column))
        table["prior_n"] = history.groupby(id_column, observed=True, sort=False).size()
        for rate_column, rate_n, rate_id, label, rate_kind in RATE_SPEC:
            if rate_kind != kind:
                continue
            if rate_n != n_column or rate_id != id_column:
                raise ValueError("inconsistent current-state specification")
            events = (
                pd.to_numeric(history[n_column], errors="coerce").fillna(0.0)
                * pd.to_numeric(history[rate_column], errors="coerce").fillna(0.0)
            )
            table[f"prior_event_{label}"] = events.groupby(ids, observed=True, sort=False).max()
        bank["kinds"][kind] = table.reset_index()
    return bank


def current_state_features(
    train: pd.DataFrame, rows: pd.DataFrame, forecast_year: int
) -> pd.DataFrame:
    """Reconstruct D/X/H1 and explicit deltas from a single row and a frozen bank."""
    bank = build_prior_bank(train, forecast_year)
    output = pd.DataFrame(index=np.arange(len(rows)))
    current_n: dict[str, np.ndarray] = {}
    prior_tables: dict[str, pd.DataFrame] = {}
    for kind, (n_column, id_column) in KIND_SPEC.items():
        table = bank["kinds"][kind]
        query = rows[[id_column]].reset_index(drop=True).merge(
            table, on=id_column, how="left", sort=False, validate="many_to_one"
        )
        prior_tables[kind] = query
        n_now = pd.to_numeric(rows[n_column], errors="coerce").fillna(0.0).to_numpy(float)
        prior_n = query["prior_n"].fillna(0.0).to_numpy(float)
        current_n[kind] = np.maximum(n_now - prior_n, 0.0)
        output[f"cur_logn_{kind}"] = np.log1p(current_n[kind])

    for rate_column, n_column, _id_column, label, kind in RATE_SPEC:
        n_now = pd.to_numeric(rows[n_column], errors="coerce").fillna(0.0).to_numpy(float)
        rate_now = pd.to_numeric(rows[rate_column], errors="coerce").to_numpy(float)
        prior_event = prior_tables[kind][f"prior_event_{label}"].fillna(0.0).to_numpy(float)
        numerator = n_now * np.nan_to_num(rate_now, nan=0.0) - prior_event
        output[f"cur_{label}"] = np.divide(
            numerator,
            current_n[kind],
            out=np.full(len(rows), np.nan, dtype=np.float64),
            where=current_n[kind] > 0.0,
        )

    adv = (
        pd.to_numeric(rows["strikes_before"], errors="coerce").to_numpy(float)
        > pd.to_numeric(rows["balls_before"], errors="coerce").to_numpy(float)
    ).astype(float)
    onb = (pd.to_numeric(rows["num_runners_on"], errors="coerce").to_numpy(float) > 0).astype(float)
    same_hand = (
        rows["pitcher_hand"].astype(str).to_numpy()
        == rows["batter_hand"].astype(str).to_numpy()
    ).astype(float)
    ball_strike = (
        pd.to_numeric(rows["balls_before"], errors="coerce").to_numpy(float)
        - pd.to_numeric(rows["strikes_before"], errors="coerce").to_numpy(float)
    )
    contexts = {"adv": adv, "onb": onb, "sh": same_hand, "bs": ball_strike}
    for rate in ("succ", "mid"):
        for name, value in contexts.items():
            output[f"dx_{rate}_{name}"] = output[f"cur_{rate}"].to_numpy(float) * value
    for rate in ("ball", "rev", "str"):
        for name in ("sh", "bs"):
            output[f"lx_{rate}_{name}"] = (
                output[f"cur_{rate}"].to_numpy(float) * contexts[name]
            )

    legacy = _current_season_state(rows, _make_season_bank(train, forecast_year))
    legacy_map = {
        "succ": "season__pitcher_raw",
        "mid": "season__middle_rate",
        "ball": "season__ball_rate",
        "rev": "season__reverse_rate",
        "str": "season__strike_rate",
        "fb": "season__fastball_rate",
        "bb": "season__breaking_rate",
        "os": "season__offspeed_rate",
        "bsucc": "season__batter_raw",
    }
    for rate, legacy_column in legacy_map.items():
        output[f"delta_{rate}"] = (
            output[f"cur_{rate}"].to_numpy(float)
            - legacy[legacy_column].to_numpy(float)
        )
    output["bmid_minus_career"] = (
        output["cur_bmid"].to_numpy(float)
        - pd.to_numeric(rows["asof_batter_middle_rate"], errors="coerce").to_numpy(float)
    )
    return output.astype(np.float64)


@dataclass(frozen=True)
class RidgeSpec:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    missing_columns: np.ndarray
    coefficient: np.ndarray


def _design_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    missing = ~np.isfinite(x)
    filled = np.where(missing, median, x)
    mean = filled.mean(axis=0)
    scale = np.sqrt(np.mean(np.square(filled - mean), axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    valid_missing = np.where((missing.mean(axis=0) > 0.0) & (missing.mean(axis=0) < 1.0))[0]
    z = (filled - mean) / scale
    if len(valid_missing):
        z = np.column_stack((z, missing[:, valid_missing].astype(float)))
    return z, median, mean, scale, valid_missing


def fit_ridge(features: np.ndarray, residual: np.ndarray, alpha: float) -> RidgeSpec:
    z, median, mean, scale, missing_columns = _design_fit(features)
    y = np.asarray(residual, dtype=np.float64)
    coefficient = np.linalg.solve(
        z.T @ z + float(alpha) * np.eye(z.shape[1]), z.T @ y
    )
    return RidgeSpec(median, mean, scale, missing_columns, coefficient)


def predict_ridge(spec: RidgeSpec, features: np.ndarray, cap: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    missing = ~np.isfinite(x)
    filled = np.where(missing, spec.median, x)
    z = (filled - spec.mean) / spec.scale
    if len(spec.missing_columns):
        z = np.column_stack((z, missing[:, spec.missing_columns].astype(float)))
    return np.clip(z @ spec.coefficient, -float(cap), float(cap))


def route_mask(axis: dict[str, np.ndarray], route: str) -> np.ndarray:
    exact = axis["exact_mask"].astype(bool)
    if route == "ALL":
        return exact
    return exact & (axis["domain3"].astype(str) == route)


def apply_correction(
    parent: np.ndarray,
    axis: dict[str, np.ndarray],
    correction: np.ndarray,
    route: str,
    eta: float,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = route_mask(axis, route)
    output[active] = np.clip(
        output[active] + float(eta) * np.asarray(correction, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any], *, locked: bool) -> bool:
    gain_min = float(gate.get("gain_min", 0.0)) if locked else 0.0
    return bool(
        metrics["gain"] >= gain_min
        and metrics["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, low_memory=False)
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parent = {name: saved[name].astype(np.float64) for name in saved.files}
    metric_axes = {
        name: {**axis, "parent": parent[name]}
        for name, axis in axes.items()
    }
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    for name in frames:
        if len(frames[name]) != len(axes[name]["target"]):
            raise ValueError(f"frame/axis length mismatch: {name}")
        target = frames[name]["control_success"].to_numpy(float)
        if not np.array_equal(target, axes[name]["target"].astype(float)):
            raise ValueError(f"frame/axis target mismatch: {name}")
        if len(parent[name]) != len(target):
            raise ValueError(f"v104 parent length mismatch: {name}")

    feature_frames = {
        "full_2022": current_state_features(train, frames["full_2022"], 2022),
        "late_2023": current_state_features(train, frames["late_2023"], 2023),
        "full_2024": current_state_features(train, frames["full_2024"], 2024),
    }
    selected_families = [str(name) for name in config["feature_families"]]
    if any(name not in FAMILIES for name in selected_families):
        raise ValueError("unknown feature family")

    early22 = frames["full_2022"]["game_month"].le(7).to_numpy()
    late22 = ~early22
    axis_late22 = _slice_axis(metric_axes["full_2022"], late22)
    trials: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    source_candidates: dict[str, dict[str, np.ndarray]] = {}
    for family in selected_families:
        columns = list(FAMILIES[family])
        x22 = feature_frames["full_2022"][columns].to_numpy(float)
        x23 = feature_frames["late_2023"][columns].to_numpy(float)
        for route in config["routes"]:
            fit_early = early22 & route_mask(axes["full_2022"], str(route))
            fit_full22 = route_mask(axes["full_2022"], str(route))
            if fit_early.sum() < 1000 or fit_full22.sum() < 1000:
                continue
            residual22 = axes["full_2022"]["target"].astype(float) - parent["full_2022"]
            for alpha in config["ridge_alpha_grid"]:
                spec_early = fit_ridge(x22[fit_early], residual22[fit_early], float(alpha))
                correction_late22 = predict_ridge(
                    spec_early, x22[late22], float(config["correction_cap"])
                )
                spec22 = fit_ridge(x22[fit_full22], residual22[fit_full22], float(alpha))
                correction23 = predict_ridge(
                    spec22, x23, float(config["correction_cap"])
                )
                for eta in config["eta_grid"]:
                    key = f"{family}__{route}__a{float(alpha):g}__e{float(eta):g}"
                    candidate_late22 = apply_correction(
                        parent["full_2022"][late22], axis_late22, correction_late22,
                        str(route), float(eta),
                    )
                    candidate23 = apply_correction(
                        parent["late_2023"], axes["late_2023"], correction23,
                        str(route), float(eta),
                    )
                    metrics = {
                        "early22_to_late22": _axis_metrics(axis_late22, candidate_late22),
                        "full22_to_late23": _axis_metrics(metric_axes["late_2023"], candidate23),
                    }
                    passed = all(
                        _point_pass(item, config["source_gate"], locked=False)
                        for item in metrics.values()
                    )
                    trials.append({
                        "key": key,
                        "family": family,
                        "route": str(route),
                        "ridge_alpha": float(alpha),
                        "eta": float(eta),
                        "source_gate_passed": bool(passed),
                        "minimum_gain": float(min(item["gain"] for item in metrics.values())),
                        "mean_gain": float(np.mean([item["gain"] for item in metrics.values()])),
                        "minimum_month_fraction": float(min(
                            item["positive_month_fraction"] for item in metrics.values()
                        )),
                        "worst_month_gain": float(min(
                            item["worst_month_gain"] for item in metrics.values()
                        )),
                    })
                    source_metrics[key] = metrics
                    source_candidates[key] = {
                        "late_2022": candidate_late22,
                        "late_2023": candidate23,
                    }

    ranking = pd.DataFrame(trials).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    selected_key = str(selected["key"])
    family = str(selected["family"])
    route = str(selected["route"])
    alpha = float(selected["ridge_alpha"])
    eta = float(selected["eta"])

    x23 = feature_frames["late_2023"][list(FAMILIES[family])].to_numpy(float)
    x24 = feature_frames["full_2024"][list(FAMILIES[family])].to_numpy(float)
    fit23 = route_mask(axes["late_2023"], route)
    residual23 = axes["late_2023"]["target"].astype(float) - parent["late_2023"]
    spec23 = fit_ridge(x23[fit23], residual23[fit23], alpha)
    correction24 = predict_ridge(spec23, x24, float(config["correction_cap"]))
    candidate24 = apply_correction(
        parent["full_2024"], axes["full_2024"], correction24, route, eta
    )
    late24 = frames["full_2024"]["game_month"].ge(8).to_numpy()
    axis_late24 = _slice_axis(metric_axes["full_2024"], late24)
    locked_metrics = {
        "full_2024": _axis_metrics(metric_axes["full_2024"], candidate24),
        "late_2024": _axis_metrics(axis_late24, candidate24[late24]),
    }
    locked_pass = {
        name: _point_pass(item, config["locked_gate"], locked=True)
        for name, item in locked_metrics.items()
    }
    eligible = bool(selected["source_gate_passed"] and all(locked_pass.values()))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        late_2022=source_candidates[selected_key]["late_2022"],
        late_2023=source_candidates[selected_key]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
        correction_full_2024=correction24,
    )
    feature_audit = {
        name: {
            "columns": list(FAMILIES[name]),
            "missing_fraction": {
                axis: float(feature_frames[axis][list(FAMILIES[name])].isna().mean().mean())
                for axis in feature_frames
            },
        }
        for name in selected_families
    }
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "external_source_recipe": "hoo743-ui/LG_Aimers09 D/X/H1 formulas",
        "formula_differences_vs_v19": {
            "raw_failure_state": "D keeps unsmoothed current-season middle/ball/reverse/strike rates",
            "batter_middle_state": "D reconstructs current-season batter middle rate",
            "pitchmix_denominator": "D uses asof_pitcher_pitchmix_n rather than asof_pitcher_n",
            "explicit_context_products": "X/H1 multiplies state by count, runners and handedness",
        },
        "feature_audit": feature_audit,
        "n_trials": int(len(ranking)),
        "selected": selected,
        "source_ranking_top20": ranking.head(20).to_dict(orient="records"),
        "selected_source_metrics": source_metrics[selected_key],
        "locked_metrics": locked_metrics,
        "locked_point_gate_pass": locked_pass,
        "eligible_for_robust_audit": eligible,
        "eligible_for_packaging": False,
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.v104_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()

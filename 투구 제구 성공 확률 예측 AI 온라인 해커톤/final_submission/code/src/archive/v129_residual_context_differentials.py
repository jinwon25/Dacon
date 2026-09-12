"""Audit externally validated residual context differentials above v104.

The recipe estimates, per pitcher, the residual mean difference between the
two levels of a binary context.  Effective sample size is the harmonic cell
size and the difference is strongly shrunk.  Query rows receive +/- half the
shrunk difference, so pitcher level is removed and only the contrast remains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics
from src.champion.v127_catboost_current_state_rebase import apply_correction
from src.core.contract import _load_contract_axis


PROTOCOL = "V129_RESIDUAL_CONTEXT_DIFFERENTIALS_V1"
CONTEXTS = ("same_hand", "two_strike", "runner_on")
FAMILY_AXES = {
    "hand_only": ("same_hand",),
    "c3_hand_two_runner": CONTEXTS,
}


def context_values(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "same_hand": (
            rows["pitcher_hand"].astype(str).to_numpy()
            == rows["batter_hand"].astype(str).to_numpy()
        ).astype(np.int8),
        "two_strike": (
            pd.to_numeric(rows["strikes_before"], errors="coerce").to_numpy(float) == 2
        ).astype(np.int8),
        "runner_on": (
            pd.to_numeric(rows["num_runners_on"], errors="coerce").to_numpy(float) > 0
        ).astype(np.int8),
    }


def fit_differential(
    pitcher: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    shrink: float,
) -> pd.Series:
    frame = pd.DataFrame({
        "pitcher": np.asarray(pitcher),
        "context": np.asarray(context, dtype=np.int8),
        "residual": np.asarray(residual, dtype=np.float64),
    })
    grouped = frame.groupby(["pitcher", "context"], observed=True)["residual"].agg(
        ["mean", "size"]
    ).unstack()
    required = [("mean", 0), ("mean", 1), ("size", 0), ("size", 1)]
    if any(column not in grouped for column in required):
        return pd.Series(dtype=np.float64)
    n0 = grouped[("size", 0)].fillna(0.0).astype(float)
    n1 = grouped[("size", 1)].fillna(0.0).astype(float)
    effective = (n0 * n1) / (n0 + n1).replace(0.0, np.nan)
    difference = grouped[("mean", 1)] - grouped[("mean", 0)]
    return (difference * effective / (effective + float(shrink))).dropna()


def predict_differential(
    table: pd.Series,
    pitcher: np.ndarray,
    context: np.ndarray,
    cap: float,
) -> np.ndarray:
    difference = pd.Series(np.asarray(pitcher)).map(table).fillna(0.0).to_numpy(float)
    sign = np.where(np.asarray(context, dtype=np.int8) == 1, 0.5, -0.5)
    return np.clip(sign * difference, -float(cap), float(cap))


def build_corrections(
    source_rows: pd.DataFrame,
    source_residual: np.ndarray,
    query_rows: pd.DataFrame,
    shrink: dict[str, float],
    cap: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_context = context_values(source_rows)
    query_context = context_values(query_rows)
    source_pitcher = source_rows["pitcher_id"].to_numpy()
    query_pitcher = query_rows["pitcher_id"].to_numpy()
    corrections: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    for name in CONTEXTS:
        table = fit_differential(
            source_pitcher, source_context[name], source_residual, float(shrink[name])
        )
        correction = predict_differential(
            table, query_pitcher, query_context[name], float(cap)
        )
        corrections[name] = correction
        audit[name] = {
            "table_pitchers": int(len(table)),
            "median_abs_difference": float(table.abs().median()) if len(table) else 0.0,
            "query_nonzero_fraction": float(np.mean(correction != 0.0)),
            "query_sd": float(correction.std()),
        }
    return corrections, audit


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
    metric_axes = {name: {**axis, "parent": parent[name]} for name, axis in axes.items()}
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    for name in frames:
        if not np.array_equal(
            frames[name]["control_success"].to_numpy(float), axes[name]["target"].astype(float)
        ):
            raise ValueError(f"axis target mismatch: {name}")

    residual22 = axes["full_2022"]["target"].astype(float) - parent["full_2022"]
    exact22 = axes["full_2022"]["exact_mask"].astype(bool)
    early22 = frames["full_2022"]["game_month"].le(7).to_numpy() & exact22
    late22 = frames["full_2022"]["game_month"].ge(8).to_numpy()
    source_rows_early = frames["full_2022"].loc[early22].reset_index(drop=True)
    corr_late22, audit_late22 = build_corrections(
        source_rows_early, residual22[early22],
        frames["full_2022"].loc[late22].reset_index(drop=True),
        config["shrink"], float(config["correction_cap"]),
    )
    corr_late23, audit_late23 = build_corrections(
        frames["full_2022"].loc[exact22].reset_index(drop=True), residual22[exact22],
        frames["late_2023"], config["shrink"], float(config["correction_cap"]),
    )
    metric_late22 = _slice_axis(metric_axes["full_2022"], late22)
    parent_late22 = parent["full_2022"][late22]

    trials: list[dict[str, Any]] = []
    trial_metrics: dict[str, dict[str, Any]] = {}
    trial_candidates: dict[str, dict[str, np.ndarray]] = {}
    for family in config["families"]:
        family = str(family)
        family_axes = FAMILY_AXES[family]
        direction_late22 = np.sum(
            np.column_stack([corr_late22[name] for name in family_axes]), axis=1
        )
        direction_late23 = np.sum(
            np.column_stack([corr_late23[name] for name in family_axes]), axis=1
        )
        for route in config["routes"]:
            route = str(route)
            for eta in config["eta_grid"]:
                candidate_late22 = apply_correction(
                    parent_late22, metric_late22, direction_late22, route, float(eta)
                )
                candidate_late23 = apply_correction(
                    parent["late_2023"], axes["late_2023"], direction_late23,
                    route, float(eta),
                )
                metrics = {
                    "early22_to_late22": _axis_metrics(metric_late22, candidate_late22),
                    "full22_to_late23": _axis_metrics(
                        metric_axes["late_2023"], candidate_late23
                    ),
                }
                passed = all(
                    _point_pass(item, config["source_gate"], locked=False)
                    for item in metrics.values()
                )
                key = f"{family}__{route}__e{float(eta):g}"
                trials.append({
                    "key": key,
                    "family": family,
                    "route": route,
                    "eta": float(eta),
                    "source_gate_passed": bool(passed),
                    "minimum_gain": float(min(item["gain"] for item in metrics.values())),
                    "mean_gain": float(np.mean([item["gain"] for item in metrics.values()])),
                    "minimum_month_fraction": float(min(
                        item["positive_month_fraction"] for item in metrics.values()
                    )),
                    "worst_month_gain": float(min(item["worst_month_gain"] for item in metrics.values())),
                })
                trial_metrics[key] = metrics
                trial_candidates[key] = {
                    "late_2022": candidate_late22,
                    "late_2023": candidate_late23,
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
    eta = float(selected["eta"])

    residual23 = axes["late_2023"]["target"].astype(float) - parent["late_2023"]
    source_rows = pd.concat(
        [frames["full_2022"].loc[exact22], frames["late_2023"]], ignore_index=True
    )
    source_residual = np.concatenate([residual22[exact22], residual23])
    corr24, audit24 = build_corrections(
        source_rows, source_residual, frames["full_2024"],
        config["shrink"], float(config["correction_cap"]),
    )
    direction24 = np.sum(
        np.column_stack([corr24[name] for name in FAMILY_AXES[family]]), axis=1
    )
    candidate24 = apply_correction(
        parent["full_2024"], axes["full_2024"], direction24, route, eta
    )
    late24 = frames["full_2024"]["game_month"].ge(8).to_numpy()
    metric_late24 = _slice_axis(metric_axes["full_2024"], late24)
    locked_metrics = {
        "full_2024": _axis_metrics(metric_axes["full_2024"], candidate24),
        "late_2024": _axis_metrics(metric_late24, candidate24[late24]),
    }
    locked_pass = {
        name: _point_pass(item, config["locked_gate"], locked=True)
        for name, item in locked_metrics.items()
    }
    eligible = bool(selected["source_gate_passed"] and all(locked_pass.values()))
    axis_matrix = np.column_stack([corr24[name] for name in CONTEXTS])
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        late_2022=trial_candidates[selected_key]["late_2022"],
        late_2023=trial_candidates[selected_key]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
        correction_full_2024=direction24,
        **{f"correction_{name}_full_2024": corr24[name] for name in CONTEXTS},
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "external_recipe": "C2/C3 residual differentials; k=1000/1000/2000; +/- half contrast",
        "n_trials": int(len(ranking)),
        "selected": selected,
        "source_ranking": ranking.to_dict(orient="records"),
        "selected_source_metrics": trial_metrics[selected_key],
        "table_audit": {
            "early22_to_late22": audit_late22,
            "full22_to_late23": audit_late23,
            "full22_plus_late23_to_full24": audit24,
        },
        "locked_axis_correlation": pd.DataFrame(
            axis_matrix, columns=CONTEXTS
        ).corr().to_dict(),
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

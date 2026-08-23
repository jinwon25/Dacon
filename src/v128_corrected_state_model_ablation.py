"""Retrain the frozen v19 state model with public D/X/H1 feature ablations.

For each source origin, the original state model and an enhanced copy use the
same labels, baseline, recency weights and LightGBM hyperparameters.  Their
prediction difference is therefore a direct feature-ablation direction.  The
family, route and dose are selected on 2022 and late 2023; only the selected
family is trained at the locked 2024 origin.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.multi_year_state_model import _add_categories, _model, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.v103_fixed_union_robust import _axis_metrics
from src.v127_hoo_current_state_rebase import (
    FAMILIES,
    apply_correction,
    current_state_features,
)
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V128_CORRECTED_STATE_MODEL_ABLATION_V1"
SOURCE_AXES = ("full_2022", "late_2023")


def _all_external_features(train: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    seasons: list[np.ndarray] = []
    for year in sorted(train["season"].unique()):
        rows = train.loc[train["season"].eq(year)].reset_index(drop=True)
        part = current_state_features(train, rows, int(year))
        parts.append(part.astype(np.float32))
        seasons.append(np.full(len(part), int(year), dtype=np.int16))
    output = pd.concat(parts, ignore_index=True)
    if not np.array_equal(np.concatenate(seasons), train["season"].to_numpy(np.int16)):
        raise ValueError("train rows are not season ordered")
    return output


def _weights(season: np.ndarray, audit_year: int, half_life: float) -> np.ndarray:
    output = np.exp2(-(audit_year - 1.0 - season.astype(np.float64)) / float(half_life))
    return output / output.mean()


def _fit_predict(
    features: pd.DataFrame,
    categorical: list[str],
    target: np.ndarray,
    baseline: np.ndarray,
    season: np.ndarray,
    audit_year: int,
    half_life: float,
    leaves: int,
    seed: int,
) -> np.ndarray:
    fit = season < audit_year
    audit = season == audit_year
    model = _model(leaves=int(leaves), seed=int(seed))
    model.fit(
        features.loc[fit],
        target[fit] - baseline[fit],
        sample_weight=_weights(season[fit], audit_year, half_life),
        categorical_feature=categorical,
    )
    prediction = np.clip(baseline[audit] + model.predict(features.loc[audit]), 0.001, 0.999)
    del model
    gc.collect()
    return np.asarray(prediction, dtype=np.float64)


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
    train = _add_domain_and_pressure(pd.read_csv(train_csv, low_memory=False))
    target = train["control_success"].to_numpy(np.float64)
    season = train["season"].to_numpy(np.int16)
    numeric_state, _ = _state_features(train)
    baseline = (
        float(config["state_recipe"]["pitcher_weight"])
        * numeric_state["season__pitcher_rate_k80"].to_numpy(np.float64)
        + (1.0 - float(config["state_recipe"]["pitcher_weight"]))
        * numeric_state["season__batter_rate_k80"].to_numpy(np.float64)
    )
    base_features = _add_categories(train, numeric_state)
    categorical = [name for name in base_features if name.startswith("cat__")]
    external = _all_external_features(train)

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in (*SOURCE_AXES, "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parent = {name: saved[name].astype(np.float64) for name in saved.files}
    metric_axes = {name: {**axis, "parent": parent[name]} for name, axis in axes.items()}
    year_frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    for name in axes:
        if not np.array_equal(
            year_frames[name]["control_success"].to_numpy(float),
            axes[name]["target"].astype(float),
        ):
            raise ValueError(f"axis target mismatch: {name}")

    recipe = config["state_recipe"]
    base_prediction: dict[str, np.ndarray] = {}
    enhanced_prediction: dict[str, dict[str, np.ndarray]] = {
        family: {} for family in config["feature_families"]
    }
    standalone_ablation: dict[str, dict[str, Any]] = {
        family: {} for family in config["feature_families"]
    }
    for audit_year, axis_name in ((2022, "full_2022"), (2023, "late_2023")):
        print(f"[v128] base audit={audit_year}", flush=True)
        direct = _fit_predict(
            base_features, categorical, target, baseline, season, audit_year,
            float(recipe["half_life"]), int(recipe["leaves"]), 12800 + audit_year,
        )
        if axis_name == "late_2023":
            month_mask = train.loc[train["season"].eq(2023), "game_month"].ge(8).to_numpy()
            direct = direct[month_mask]
        base_prediction[axis_name] = direct
        for family in config["feature_families"]:
            print(f"[v128] family={family} audit={audit_year}", flush=True)
            columns = list(FAMILIES[str(family)])
            features = pd.concat([base_features, external[columns]], axis=1)
            enhanced = _fit_predict(
                features, categorical, target, baseline, season, audit_year,
                float(recipe["half_life"]), int(recipe["leaves"]),
                12900 + audit_year,
            )
            del features
            gc.collect()
            if axis_name == "late_2023":
                enhanced = enhanced[month_mask]
            enhanced_prediction[str(family)][axis_name] = enhanced
            standalone_axis = {**metric_axes[axis_name], "parent": direct}
            standalone_ablation[str(family)][axis_name] = _axis_metrics(
                standalone_axis, enhanced
            )

    trials: list[dict[str, Any]] = []
    trial_metrics: dict[str, dict[str, Any]] = {}
    trial_candidates: dict[str, dict[str, np.ndarray]] = {}
    for family in config["feature_families"]:
        family = str(family)
        correction = {
            name: enhanced_prediction[family][name] - base_prediction[name]
            for name in SOURCE_AXES
        }
        for route in config["routes"]:
            route = str(route)
            for eta in config["eta_grid"]:
                candidates = {
                    name: apply_correction(
                        parent[name], axes[name], correction[name], route, float(eta)
                    )
                    for name in SOURCE_AXES
                }
                metrics = {
                    name: _axis_metrics(metric_axes[name], candidates[name])
                    for name in SOURCE_AXES
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
                trial_candidates[key] = candidates
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

    print("[v128] locked base audit=2024", flush=True)
    base24 = _fit_predict(
        base_features, categorical, target, baseline, season, 2024,
        float(recipe["half_life"]), int(recipe["leaves"]), 12800 + 2024,
    )
    print(f"[v128] locked family={family} audit=2024", flush=True)
    locked_features = pd.concat([base_features, external[list(FAMILIES[family])]], axis=1)
    enhanced24 = _fit_predict(
        locked_features, categorical, target, baseline, season, 2024,
        float(recipe["half_life"]), int(recipe["leaves"]), 12900 + 2024,
    )
    del locked_features
    gc.collect()
    correction24 = enhanced24 - base24
    candidate24 = apply_correction(
        parent["full_2024"], axes["full_2024"], correction24, route, eta
    )
    late24 = year_frames["full_2024"]["game_month"].ge(8).to_numpy()
    metric_late24 = _slice_axis(metric_axes["full_2024"], late24)
    locked_metrics = {
        "full_2024": _axis_metrics(metric_axes["full_2024"], candidate24),
        "late_2024": _axis_metrics(metric_late24, candidate24[late24]),
    }
    locked_pass = {
        name: _point_pass(item, config["locked_gate"], locked=True)
        for name, item in locked_metrics.items()
    }
    locked_standalone_ablation = _axis_metrics(
        {**metric_axes["full_2024"], "parent": base24}, enhanced24
    )
    eligible = bool(selected["source_gate_passed"] and all(locked_pass.values()))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=trial_candidates[selected_key]["full_2022"],
        late_2023=trial_candidates[selected_key]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
        correction_full_2024=correction24,
        direct_base_full_2024=base24,
        direct_enhanced_full_2024=enhanced24,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "method": "same v19 LightGBM recipe; enhanced-minus-original feature ablation",
        "n_trials": int(len(ranking)),
        "selected": selected,
        "source_ranking": ranking.to_dict(orient="records"),
        "selected_source_metrics": trial_metrics[selected_key],
        "standalone_feature_ablation": standalone_ablation,
        "locked_standalone_feature_ablation": locked_standalone_ablation,
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

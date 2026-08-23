"""Forward audit of a shallow nonlinear residual model on v78 stable features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics
from src.v78_environment_stable_residual import (
    _center_residual,
    _current_axes,
    _historical_frame,
    _risk_weights,
    _source_recipe,
    build_features,
    environment_labels,
    select_stable_features,
)


PROTOCOL = "V81_STABLE_SHALLOW_GBDT_RESIDUAL_V1"


@dataclass
class TreeSpec:
    feature_names: tuple[str, ...]
    model: lgb.LGBMRegressor | None
    risk: str


def rolling_month_splits(
    frame: pd.DataFrame, config: dict[str, Any]
) -> list[tuple[np.ndarray, np.ndarray]]:
    month = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(np.int16)
    output = []
    minimum = int(config["minimum_fold_rows"])
    starts_late = int(month.min()) >= 8
    for fold in config["rolling_source_folds"]:
        train_month_max = int(fold["train_month_max"])
        if starts_late != (train_month_max >= 8):
            continue
        train = month <= train_month_max
        valid = (month >= int(fold["valid_month_min"])) & (
            month <= int(fold["valid_month_max"])
        )
        if int(train.sum()) >= minimum and int(valid.sum()) >= minimum:
            if int(month[train].max()) >= int(month[valid].min()):
                raise AssertionError("rolling source fold is not strictly forward")
            output.append((train, valid))
    return output


def fit_tree_spec(
    features: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    environments: np.ndarray,
    risk: str,
    config: dict[str, Any],
) -> tuple[TreeSpec, dict[str, Any]]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    stable = config["stable_selection"]
    selected, selection = select_stable_features(
        features.to_numpy(np.float64),
        residual,
        environments,
        list(features.columns),
        alpha=float(stable["ridge_alpha"]),
        minimum_environment_rows=int(stable["minimum_environment_rows"]),
        minimum_sign_consistency=float(stable["minimum_sign_consistency"]),
        minimum_median_abs_coefficient=float(
            stable["minimum_median_abs_standardized_coefficient"]
        ),
    )
    names = tuple(features.columns[selected])
    if not names:
        return TreeSpec(names, None, risk), selection
    parameters = dict(config["model"])
    parameters.update(verbosity=-1, n_jobs=6)
    model = lgb.LGBMRegressor(**parameters)
    model.fit(
        features.loc[:, list(names)],
        _center_residual(residual, environments),
        sample_weight=_risk_weights(risk, residual, environments),
    )
    return TreeSpec(names, model, risk), selection


def predict_correction(
    spec: TreeSpec, features: pd.DataFrame, correction_cap: float
) -> np.ndarray:
    if spec.model is None or not spec.feature_names:
        return np.zeros(len(features), dtype=np.float64)
    prediction = np.asarray(
        spec.model.predict(features.loc[:, list(spec.feature_names)]),
        dtype=np.float64,
    )
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise ValueError("invalid shallow GBDT correction")
    return np.clip(prediction, -float(correction_cap), float(correction_cap))


def crossfit_source(
    frame: pd.DataFrame,
    parent: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    features = build_features(frame)
    target = frame["target"].to_numpy(np.float64)
    environments = environment_labels(frame)
    corrections = {
        str(risk): np.zeros(len(frame), dtype=np.float64) for risk in config["risks"]
    }
    covered = np.zeros(len(frame), dtype=bool)
    audits = []
    for fold_index, (train, valid) in enumerate(
        rolling_month_splits(frame, config), start=1
    ):
        if np.any(covered & valid):
            raise AssertionError("rolling source validation folds overlap")
        covered |= valid
        fold_audit: dict[str, Any] = {
            "fold": fold_index,
            "train_rows": int(train.sum()),
            "valid_rows": int(valid.sum()),
            "train_month_max": int(frame.loc[train, "game_month"].max()),
            "valid_month_min": int(frame.loc[valid, "game_month"].min()),
            "risks": {},
        }
        for risk in config["risks"]:
            spec, selection = fit_tree_spec(
                features.loc[train].reset_index(drop=True),
                target[train],
                np.asarray(parent)[train],
                environments[train],
                str(risk),
                config,
            )
            corrections[str(risk)][valid] = predict_correction(
                spec,
                features.loc[valid].reset_index(drop=True),
                float(config["correction_cap"]),
            )
            fold_audit["risks"][str(risk)] = {
                "selected_feature_count": len(spec.feature_names),
                "selected_features": list(spec.feature_names),
                "selection_environment_count": int(
                    selection.get("environment_count", 0)
                ),
            }
        audits.append(fold_audit)
    return corrections, covered, audits


def run_transition(
    name: str,
    source_frame: pd.DataFrame,
    source_parent: np.ndarray,
    audit_frame: pd.DataFrame,
    audit_parent: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    corrections, covered, crossfit = crossfit_source(
        source_frame, source_parent, config
    )
    if covered.any():
        local_corrections = {
            risk: values[covered] for risk, values in corrections.items()
        }
        recipe, source_metrics = _source_recipe(
            source_frame.loc[covered].reset_index(drop=True),
            np.asarray(source_parent)[covered],
            local_corrections,
            config,
        )
    else:
        recipe = {"risk": str(config["risks"][0]), "eta": 0.0, "source_gate": False}
        source_metrics = pd.DataFrame()
    features = build_features(source_frame)
    spec, selection = fit_tree_spec(
        features,
        source_frame["target"].to_numpy(np.float64),
        source_parent,
        environment_labels(source_frame),
        str(recipe["risk"]),
        config,
    )
    correction = predict_correction(
        spec, build_features(audit_frame), float(config["correction_cap"])
    )
    candidate = np.clip(
        np.asarray(audit_parent) + float(recipe["eta"]) * correction,
        0.001,
        0.999,
    )
    audit = diagnostics(
        audit_frame,
        audit_parent,
        candidate,
        np.ones(len(audit_frame), dtype=bool),
    )
    return (
        {
            "axis": name,
            "source_rows": len(source_frame),
            "source_crossfit_rows": int(covered.sum()),
            "audit_rows": len(audit_frame),
            "selected_recipe": recipe,
            "selected_feature_count": len(spec.feature_names),
            "selected_features": list(spec.feature_names),
            "selection": selection,
            "crossfit": crossfit,
            "audit": audit,
        },
        candidate,
        source_metrics.assign(axis=name),
    )


def run(
    project: Path,
    state_dir: Path,
    final_parent_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project = project.resolve()
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    historical: dict[int, pd.DataFrame] = {}
    historical_parent: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        historical[year], historical_parent[year] = _historical_frame(
            raw, state_dir.resolve(), year
        )
    axes, current_parent = _current_axes(project, raw, final_parent_dir.resolve())
    early22 = historical[2022]["game_month"].le(7).to_numpy()
    late22 = historical[2022]["game_month"].ge(8).to_numpy()
    full24 = axes["outer_full_2024"]
    early24 = full24["game_month"].le(7).to_numpy()
    transitions = {
        "early22_to_late22": (
            historical[2022].loc[early22].reset_index(drop=True),
            historical_parent[2022][early22],
            historical[2022].loc[late22].reset_index(drop=True),
            historical_parent[2022][late22],
        ),
        "full22_to_full23": (
            historical[2022], historical_parent[2022],
            historical[2023], historical_parent[2023],
        ),
        "full23_to_full24_historical": (
            historical[2023], historical_parent[2023],
            historical[2024], historical_parent[2024],
        ),
        "late23_to_full24_exact": (
            axes["selection_late_2023"], current_parent["selection_late_2023"],
            full24, current_parent["outer_full_2024"],
        ),
        "early24_to_late24_exact": (
            full24.loc[early24].reset_index(drop=True),
            current_parent["outer_full_2024"][early24],
            axes["replication_late_2024"],
            current_parent["replication_late_2024"],
        ),
    }
    summaries: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    ledgers = []
    for name, values in transitions.items():
        print(f"[v81] {name}", flush=True)
        summary, prediction, ledger = run_transition(name, *values, config)
        summaries[name] = summary
        predictions[name] = prediction
        if not ledger.empty:
            ledgers.append(ledger)

    primary_names = ("early22_to_late22", "full22_to_full23")
    primary = [summaries[name] for name in primary_names]
    gates = {
        "primary_source_selected_nonzero_eta": all(
            float(item["selected_recipe"]["eta"]) > 0.0 for item in primary
        ),
        "primary_gains_positive": all(float(item["audit"]["gain"]) > 0.0 for item in primary),
        "primary_month_fraction_at_least_075": all(
            float(item["audit"]["positive_month_fraction"]) >= 0.75 for item in primary
        ),
        "primary_worst_month_above_minus_5": all(
            float(item["audit"]["worst_month_gain"]) > -5.0 for item in primary
        ),
        "primary_domains_nonnegative": all(
            float(item["audit"]["minimum_domain_gain"]) >= 0.0 for item in primary
        ),
    }
    compact = [
        {
            "axis": name,
            "selected_risk": item["selected_recipe"]["risk"],
            "selected_eta": item["selected_recipe"]["eta"],
            "selected_feature_count": item["selected_feature_count"],
            "gain": item["audit"]["gain"],
            "positive_month_fraction": item["audit"]["positive_month_fraction"],
            "worst_month_gain": item["audit"]["worst_month_gain"],
            "minimum_domain_gain": item["audit"]["minimum_domain_gain"],
            "mean_abs_shift": item["audit"]["mean_abs_shift"],
        }
        for name, item in summaries.items()
    ]
    passes = bool(all(gates.values()))
    result = {
        "protocol": PROTOCOL,
        "config": config,
        "primary_axes": list(primary_names),
        "audits": summaries,
        "gates": gates,
        "passes_primary_mechanism_gate": passes,
        "eligible_for_packaging": False,
        "packaging_reason": (
            "bootstrap and family Reality Check still required"
            if passes else "primary mechanism gate failed"
        ),
        "test_csv_read": False,
        "row_local_features_only": True,
        "test_aggregate_used": False,
        "current_pitch_physics_or_location_used": False,
    }
    if ledgers:
        pd.concat(ledgers, ignore_index=True).to_csv(
            output_dir / "source_trial_ledger.csv", index=False
        )
    pd.DataFrame(compact).to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": gates, "metrics": compact}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--final-parent-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.state_dir, args.final_parent_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()

"""Strict-forward paired removal of batter cumulative ASOF summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.v97_conditional_direct_forward import (
    CATEGORICAL,
    _align_categories,
    _bank,
    _diagnostics,
    _feature_frame,
    _load_contract_axis,
)
from src.v106_paired_player_identity import CONDITIONAL_COLUMNS, optimal_eta, paired_axis, subset_axis


PROTOCOL = "V107_BATTER_ASOF_ABLATION_V1"


def drop_batter_asof(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(features.columns))
    if missing:
        raise ValueError(f"missing batter ASOF columns: {missing}")
    output = features.drop(columns=columns)
    if "batter_hand" not in output or "batter_team_id" not in output:
        raise AssertionError("legitimate batter context was removed")
    return output


def fit_predict(
    raw: pd.DataFrame,
    features: dict[int, pd.DataFrame],
    audit_year: int,
    config: dict[str, Any],
) -> np.ndarray:
    fit_years = [year for year in sorted(features) if year < audit_year]
    fit = pd.concat([features[year] for year in fit_years], ignore_index=True)
    audit = features[audit_year]
    fit, audit = _align_categories(fit, audit)
    fit_mask = raw["season"].lt(audit_year).to_numpy()
    target = raw.loc[fit_mask, "control_success"].to_numpy(np.int8)
    seasons = raw.loc[fit_mask, "season"].to_numpy(np.int16)
    weights = np.exp2(
        -((audit_year - 1) - seasons) / float(config["recency_half_life"])
    )
    model = lgb.LGBMClassifier(**config["model"])
    model.fit(
        fit,
        target,
        sample_weight=weights,
        categorical_feature=list(CATEGORICAL),
    )
    return np.clip(model.predict_proba(audit)[:, 1], 0.001, 0.999)


def run(
    train_csv: Path,
    contract_dir: Path,
    v97_config_path: Path,
    v98_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    v97_config = json.loads(v97_config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    years = [int(year) for year in config["audit_years"]]
    feature_years = list(range(int(raw["season"].min()), max(years) + 1))
    features: dict[int, pd.DataFrame] = {}
    for year in feature_years:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        history = raw.loc[raw["season"].lt(year)]
        frame = _feature_frame(rows, _bank(history, v97_config["conditional_strength"]))
        frame = frame.drop(columns=list(CONDITIONAL_COLUMNS))
        features[year] = drop_batter_asof(frame, list(config["dropped_columns"]))
        print(
            f"[v107 features] year={year} rows={len(rows)} columns={len(features[year].columns)}",
            flush=True,
        )

    axes: dict[int, dict[str, np.ndarray]] = {}
    corrections: dict[int, np.ndarray] = {}
    for year in years:
        print(f"[v107 no-batter-ASOF model] audit_year={year}", flush=True)
        ablated = fit_predict(raw, features, year, config)
        with np.load(v98_dir / f"ablation_full_{year}.npz") as saved:
            baseline = saved["baseline"].astype(np.float64)
            expected_index = saved["raw_index"].astype(np.int64)
        common = _load_contract_axis(contract_dir / f"common_full_{year}.npz")
        if not np.array_equal(expected_index, common["raw_index"].astype(np.int64)):
            raise ValueError(f"v98/common row order mismatch for {year}")
        correction = ablated - baseline
        corrections[year] = correction
        axes[year] = paired_axis(common, correction)
        np.savez_compressed(
            output_dir / f"batter_ablation_full_{year}.npz",
            raw_index=expected_index,
            target=common["target"],
            baseline=baseline,
            ablated=ablated,
            correction=correction,
            domain3=common["domain3"],
            game_month=common["game_month"],
            pitcher_id=common["pitcher_id"],
            batter_id=common["batter_id"],
        )

    selection = [axes[int(year)] for year in config["selection_years"]]
    rows: list[dict[str, Any]] = []
    for name, domains in config["routes"].items():
        route = tuple(str(value) for value in domains)
        eta = optimal_eta(selection, route, float(config["eta_cap"]))
        metrics = [_diagnostics(axis, route, eta) for axis in selection]
        rows.append(
            {
                "route": name,
                "eta": eta,
                "selection_min_gain": min(item["gain"] for item in metrics),
                "selection_mean_gain": float(np.mean([item["gain"] for item in metrics])),
                "selection_min_month_fraction": min(item["positive_month_fraction"] for item in metrics),
                "selection_worst_month_gain": min(item["worst_month_gain"] for item in metrics),
            }
        )
    ledger = pd.DataFrame(rows).sort_values(
        ["selection_min_gain", "selection_mean_gain"], ascending=False
    ).reset_index(drop=True)
    ledger.to_csv(output_dir / "selection_ledger.csv", index=False)
    selected = ledger.iloc[0]
    route_name = str(selected["route"])
    route = tuple(str(value) for value in config["routes"][route_name])
    eta = float(selected["eta"])
    common_metrics = {str(year): _diagnostics(axes[year], route, eta) for year in years}

    exact_metrics: dict[str, Any] = {}
    for year in (2022, 2024):
        exact = _load_contract_axis(contract_dir / f"v84_full_{year}.npz")
        exact_metrics[f"full_{year}"] = _diagnostics(
            subset_axis(exact, corrections[year]), route, eta
        )
    late23 = _load_contract_axis(contract_dir / "v84_late_2023.npz")
    raw_index23 = axes[2023]["raw_index"].astype(np.int64)
    positions23 = pd.Series(np.arange(len(raw_index23)), index=raw_index23).loc[
        late23["raw_index"].astype(np.int64)
    ].to_numpy(np.int64)
    parent23 = late23["parent"].astype(np.float64)
    late23_axis = {
        **late23,
        "base_parent": parent23,
        "direct": parent23 + corrections[2023][positions23],
    }
    exact_metrics["late_2023"] = _diagnostics(late23_axis, route, eta)

    source_passed = bool(
        selected["selection_min_gain"] > 0.0
        and selected["selection_min_month_fraction"] >= 0.75
        and selected["selection_worst_month_gain"] > -5.0
    )
    locked = [exact_metrics["full_2022"], exact_metrics["late_2023"], exact_metrics["full_2024"]]
    locked_passed = bool(
        all(item["gain"] > 0.0 for item in locked)
        and all(item["positive_month_fraction"] >= 0.75 for item in locked)
        and all(item["worst_month_gain"] > -5.0 for item in locked)
        and all(item["minimum_domain_gain"] >= 0.0 for item in locked)
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if source_passed and locked_passed else "reject",
        "paired_model_contract": "same rows, weights, seed, hyperparameters and context; only four batter cumulative ASOF columns removed",
        "selected_recipe": {"route": route_name, "domains": list(route), "eta": eta},
        "source_metrics": {str(year): common_metrics[str(year)] for year in config["selection_years"]},
        "locked_common_metrics": {str(year): common_metrics[str(year)] for year in config["locked_years"]},
        "exact_v84_metrics": exact_metrics,
        "gates": {"source_stability": source_passed, "locked_stability": locked_passed},
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
        "eligible_for_packaging": False,
        "packaging_reason": "robust resampling, standalone refit, parity and runtime remain required even if promoted",
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
    parser.add_argument("--v97-config", type=Path, required=True)
    parser.add_argument("--v98-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v97_config,
        args.v98_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

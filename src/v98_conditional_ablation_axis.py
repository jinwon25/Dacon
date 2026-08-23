"""Paired-ablation isolation of v97 strict conditional information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.contract import _diagnostics, _load_contract_axis
from src.v97_conditional_direct_forward import _bank, _feature_frame, _fit_predict


PROTOCOL = "V98_CONDITIONAL_ABLATION_AXIS_V1"


def _optimal_eta(
    axes: list[dict[str, np.ndarray]], route: tuple[str, ...], cap: float
) -> float:
    numerator = 0.0
    denominator = 0.0
    for axis in axes:
        active = np.isin(axis["domain3"].astype(str), route)
        correction = axis["direct"][active] - axis["parent"][active]
        numerator += float(
            np.dot(axis["target"][active] - axis["base_parent"][active], correction)
        )
        denominator += float(np.dot(correction, correction))
    return 0.0 if denominator <= 0.0 else float(np.clip(numerator / denominator, 0.0, cap))


def run(
    train_csv: Path,
    contract_dir: Path,
    v97_config_path: Path,
    v97_prediction_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    v97_config = json.loads(v97_config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    audit_years = [int(value) for value in config["audit_years"]]
    feature_years = list(range(int(raw["season"].min()), max(audit_years) + 1))
    conditional_columns = [str(value) for value in config["conditional_columns"]]

    baseline_features: dict[int, pd.DataFrame] = {}
    for year in feature_years:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        history = raw.loc[raw["season"].lt(year)]
        conditional = _feature_frame(rows, _bank(history, v97_config["conditional_strength"]))
        missing = sorted(set(conditional_columns) - set(conditional.columns))
        if missing:
            raise ValueError(f"missing conditional columns: {missing}")
        baseline_features[year] = conditional.drop(columns=conditional_columns)
        print(
            f"[v98 features] year={year} rows={len(rows)} columns={len(baseline_features[year].columns)}",
            flush=True,
        )

    axes: dict[int, dict[str, np.ndarray]] = {}
    baselines: dict[int, np.ndarray] = {}
    corrections: dict[int, np.ndarray] = {}
    for year in audit_years:
        print(f"[v98 baseline model] audit_year={year}", flush=True)
        baseline = _fit_predict(raw, baseline_features, year, v97_config)
        with np.load(v97_prediction_dir / f"direct_full_{year}.npz", allow_pickle=True) as saved:
            conditional = saved["direct"].astype(np.float64)
            raw_index = saved["raw_index"].astype(np.int64)
        common = _load_contract_axis(contract_dir / f"common_full_{year}.npz")
        if not np.array_equal(raw_index, common["raw_index"].astype(np.int64)):
            raise ValueError(f"v97 prediction order mismatch for {year}")
        correction = conditional - baseline
        baselines[year] = baseline
        corrections[year] = correction
        # _diagnostics applies parent + eta * (direct - parent).  Set its
        # synthetic direct to common_parent + paired_ablation_correction.
        axes[year] = {
            **common,
            "base_parent": common["parent"].astype(np.float64),
            "direct": common["parent"].astype(np.float64) + correction,
        }
        np.savez_compressed(
            output_dir / f"ablation_full_{year}.npz",
            raw_index=raw_index,
            target=common["target"],
            baseline=baseline,
            conditional=conditional,
            correction=correction,
            domain3=common["domain3"],
            game_month=common["game_month"],
            pitcher_id=common["pitcher_id"],
            batter_id=common["batter_id"],
        )

    selection = [axes[int(year)] for year in config["selection_years"]]
    rows = []
    for name, values in config["routes"].items():
        route = tuple(str(value) for value in values)
        eta = _optimal_eta(selection, route, float(config["eta_cap"]))
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
    selected_name = str(selected["route"])
    selected_eta = float(selected["eta"])
    selected_route = tuple(str(value) for value in config["routes"][selected_name])

    common_metrics = {
        str(year): _diagnostics(axes[year], selected_route, selected_eta)
        for year in audit_years
    }
    exact_metrics: dict[str, Any] = {}
    for year in (2022, 2024):
        exact = _load_contract_axis(contract_dir / f"v84_full_{year}.npz")
        mask = exact["exact_mask"].astype(bool)
        exact_axis = {
            "target": exact["target"][mask],
            "parent": exact["parent"][mask].astype(np.float64),
            "direct": exact["parent"][mask].astype(np.float64) + corrections[year][mask],
            "domain3": exact["domain3"][mask],
            "game_month": exact["game_month"][mask],
        }
        exact_metrics[str(year)] = _diagnostics(exact_axis, selected_route, selected_eta)

    result = {
        "protocol": PROTOCOL,
        "selected_on_common_years_only": list(config["selection_years"]),
        "selected_recipe": {"route": selected_name, "domains": list(selected_route), "eta": selected_eta},
        "route_trial_count": int(len(ledger)),
        "paired_model_contract": "same training rows, weights, seed and hyperparameters; only ten strict conditional columns differ",
        "common_diagnostics": common_metrics,
        "exact_v84_diagnostics": exact_metrics,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
        "eligible_for_packaging": False,
        "packaging_reason": "screen only; promotion statistics and final standalone refit are pending",
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
    parser.add_argument("--v97-prediction-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v97_config,
        args.v97_prediction_dir, args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()

"""Paired LightGBM ablation for pitcher stretch/inning/score contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.contract import _diagnostics, _load_contract_axis
from src.v97_conditional_direct_forward import (
    _aggregate,
    _bank,
    _feature_frame,
    _fit_predict,
)


PROTOCOL = "V101_PITCHER_CONTEXT_ABLATION_V1"


def _contexts(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["stretch"] = (pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0) > 0).astype(np.int8)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(5).to_numpy()
    out["inning_band"] = np.where(inning <= 3, 0, np.where(inning <= 6, 1, 2)).astype(np.int8)
    score = np.abs(
        pd.to_numeric(frame["score_diff_pitcher_team"], errors="coerce").fillna(0).to_numpy()
    )
    out["score_band"] = np.where(score <= 1, 0, np.where(score >= 4, 2, 1)).astype(np.int8)
    return out


def _context_bank(
    history: pd.DataFrame,
    pitcher: pd.DataFrame | None,
    global_rate: float,
    strengths: dict[str, float],
) -> dict[str, pd.DataFrame | None]:
    if history.empty or pitcher is None:
        return {name: None for name in strengths}
    work = history[["pitcher_id", "control_success"]].copy()
    context = _contexts(history)
    for column in context:
        work[column] = context[column].to_numpy()
    output: dict[str, pd.DataFrame | None] = {}
    for name, strength in strengths.items():
        table = _aggregate(work, ["pitcher_id", name])
        table = table.merge(
            pitcher[["pitcher_id", "rate"]].rename(columns={"rate": "parent"}),
            on="pitcher_id", how="left", validate="many_to_one",
        )
        table["parent"] = table["parent"].fillna(global_rate)
        table["rate"] = (
            table["success"] + float(strength) * table["parent"]
        ) / (table["n"] + float(strength))
        table["dev"] = table["rate"] - table["parent"]
        table["reliability"] = table["n"] / (table["n"] + float(strength))
        output[name] = table[["pitcher_id", name, "rate", "dev", "n", "reliability"]]
    return output


def attach_context_features(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    tables: dict[str, pd.DataFrame | None],
    global_rate: float,
) -> pd.DataFrame:
    out = features.copy()
    context = _contexts(rows).reset_index(drop=True)
    for name, table in tables.items():
        keys = pd.DataFrame(
            {
                "pitcher_id": rows["pitcher_id"].reset_index(drop=True),
                name: context[name],
            }
        )
        if table is None:
            merged = pd.DataFrame(index=np.arange(len(rows)))
        else:
            merged = keys.merge(
                table, on=["pitcher_id", name], how="left",
                sort=False, validate="many_to_one",
            )
        out[f"ctx_{name}_rate"] = (
            merged["rate"].fillna(global_rate).to_numpy()
            if "rate" in merged else global_rate
        )
        out[f"ctx_{name}_dev"] = (
            merged["dev"].fillna(0.0).to_numpy() if "dev" in merged else 0.0
        )
        out[f"ctx_{name}_n"] = (
            merged["n"].fillna(0.0).to_numpy() if "n" in merged else 0.0
        )
        out[f"ctx_{name}_reliability"] = (
            merged["reliability"].fillna(0.0).to_numpy()
            if "reliability" in merged else 0.0
        )
    return out


def _optimal_eta(
    target: np.ndarray, parent: np.ndarray, correction: np.ndarray, cap: float
) -> float:
    denominator = float(np.dot(correction, correction))
    return 0.0 if denominator <= 0.0 else float(
        np.clip(np.dot(target - parent, correction) / denominator, 0.0, cap)
    )


def _align(
    common: dict[str, np.ndarray], exact: dict[str, np.ndarray], values: np.ndarray
) -> np.ndarray:
    location = {int(value): index for index, value in enumerate(common["raw_index"])}
    take = np.asarray([location[int(value)] for value in exact["raw_index"]], dtype=np.int64)
    return np.asarray(values)[take]


def _metrics(
    exact: dict[str, np.ndarray], correction: np.ndarray, eta: float
) -> dict[str, Any]:
    mask = exact["exact_mask"].astype(bool)
    parent = exact["parent"][mask].astype(np.float64)
    axis = {
        "target": exact["target"][mask].astype(np.float64),
        "parent": parent,
        "direct": parent + np.asarray(correction)[mask],
        "domain3": exact["domain3"][mask],
        "game_month": exact["game_month"][mask],
    }
    return _diagnostics(axis, ("R_CORE",), eta)


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
    features: dict[int, pd.DataFrame] = {}
    for year in feature_years:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        history = raw.loc[raw["season"].lt(year)]
        base_bank = _bank(history, v97_config["conditional_strength"])
        base = _feature_frame(rows, base_bank)
        tables = _context_bank(
            history, base_bank["pitcher"], float(base_bank["global"]),
            config["context_strength"],
        )
        features[year] = attach_context_features(
            rows, base, tables, float(base_bank["global"])
        )
        print(
            f"[v101 features] year={year} rows={len(rows)} columns={len(features[year].columns)}",
            flush=True,
        )

    corrections: dict[int, np.ndarray] = {}
    common_axes: dict[int, dict[str, np.ndarray]] = {}
    for year in audit_years:
        print(f"[v101 model] audit_year={year}", flush=True)
        extended = _fit_predict(raw, features, year, v97_config)
        with np.load(v97_prediction_dir / f"direct_full_{year}.npz") as saved:
            baseline = saved["direct"].astype(np.float64)
        corrections[year] = extended - baseline
        common_axes[year] = _load_contract_axis(contract_dir / f"common_full_{year}.npz")
        np.savez_compressed(
            output_dir / f"context_ablation_full_{year}.npz",
            raw_index=common_axes[year]["raw_index"],
            target=common_axes[year]["target"],
            extended=extended, baseline=baseline,
            correction=corrections[year],
        )

    exact22 = _load_contract_axis(contract_dir / "v84_full_2022.npz")
    core22 = exact22["exact_mask"].astype(bool) & (
        exact22["domain3"].astype(str) == "R_CORE"
    )
    eta = _optimal_eta(
        exact22["target"][core22], exact22["parent"][core22],
        corrections[2022][core22], float(config["eta_cap"]),
    )
    exact_files = {
        2022: "v84_full_2022.npz",
        2023: "v84_late_2023.npz",
        2024: "v84_full_2024.npz",
    }
    diagnostics: dict[str, Any] = {}
    for year, filename in exact_files.items():
        exact = _load_contract_axis(contract_dir / filename)
        correction = _align(common_axes[year], exact, corrections[year])
        diagnostics[str(year)] = _metrics(exact, correction, eta)
    selection = diagnostics["2022"]
    gate = bool(
        selection["gain"] > 0.0
        and selection["positive_month_fraction"]
        >= float(config["selection_gate"]["minimum_positive_month_fraction"])
        and selection["worst_month_gain"]
        > float(config["selection_gate"]["worst_month_gain_strictly_above"])
    )
    locked = [diagnostics["2023"], diagnostics["2024"]]
    locked_gate = bool(
        all(item["gain"] > 0.0 for item in locked)
        and all(
            item["positive_month_fraction"]
            >= float(config["selection_gate"]["minimum_positive_month_fraction"])
            for item in locked
        )
        and all(
            item["worst_month_gain"]
            > float(config["selection_gate"]["worst_month_gain_strictly_above"])
            for item in locked
        )
    )
    result = {
        "protocol": PROTOCOL,
        "selected_recipe": {"route": "R_CORE", "eta": eta, "context_bundle": list(config["context_strength"])},
        "selection_gate_passed": gate,
        "locked_gate_passed": locked_gate,
        "exact_v84_diagnostics": diagnostics,
        "point_gates_passed": bool(gate and locked_gate),
        "eligible_for_packaging": False,
        "packaging_reason": (
            "bootstrap and Reality Check pending"
            if gate and locked_gate else "selection or locked point gate failed"
        ),
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
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

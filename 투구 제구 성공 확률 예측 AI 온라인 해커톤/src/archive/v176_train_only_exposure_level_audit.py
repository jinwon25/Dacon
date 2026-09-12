"""Strict-forward audit of train-only player-exposure level directions.

This independently tests a public hypothesis that historical batter exposure
contains a level correction not fully absorbed by the model.  It deliberately
does not reuse leaderboard-fitted coefficients.  Every lookup table, center,
and scale is estimated from official training seasons strictly before the
audit season.  Unknown players receive zero adjustment, and test rows are
never grouped or used to estimate a distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import (
    _load_year_context,
    _locked_contract,
    _source_contracts,
    c3_mix,
    compose,
)
from src.archive.v173_h1_noncore_extension_audit import paired_metrics
from src.archive.v174_brier_h1_regressor import _jy_prediction, _robustness
from src.core.contract import _load_contract_axis


PROTOCOL = "V176_TRAIN_ONLY_EXPOSURE_LEVEL_AUDIT_V1"
SOURCE_AXES = ("full_2022", "late_2023")
ETAS = (0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004)


def entity_exposure_direction(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    entity: str,
    *,
    log_transform: bool,
) -> np.ndarray:
    """Standardize a history-only entity exposure table and query row-locally."""
    counts = history.groupby(entity, sort=True).size().astype(np.float64)
    if log_transform:
        values = np.log1p(counts.to_numpy(np.float64))
    else:
        values = counts.to_numpy(np.float64)
    center = float(values.mean())
    scale = float(values.std())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("exposure table has zero or invalid scale")
    table = pd.Series((values - center) / scale, index=counts.index)
    return rows[entity].map(table).fillna(0.0).to_numpy(np.float64)


def asof_level_direction(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    column: str,
) -> np.ndarray:
    reference = np.log1p(
        pd.to_numeric(history[column], errors="coerce").to_numpy(np.float64)
    )
    reference = reference[np.isfinite(reference)]
    center = float(reference.mean())
    scale = float(reference.std())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("ASOF reference has zero or invalid scale")
    query = np.log1p(
        pd.to_numeric(rows[column], errors="coerce").to_numpy(np.float64)
    )
    output = (query - center) / scale
    output[~np.isfinite(output)] = 0.0
    return output


def direction_library(
    raw: pd.DataFrame, rows: pd.DataFrame, year: int
) -> dict[str, np.ndarray]:
    two_year = raw.loc[
        raw["season"].ge(year - 2) & raw["season"].lt(year)
    ].reset_index(drop=True)
    last_year = raw.loc[raw["season"].eq(year - 1)].reset_index(drop=True)
    all_prior = raw.loc[raw["season"].lt(year)].reset_index(drop=True)
    return {
        "batter_exposure_2y_linear": entity_exposure_direction(
            two_year, rows, "batter_id", log_transform=False
        ),
        "batter_exposure_2y_log": entity_exposure_direction(
            two_year, rows, "batter_id", log_transform=True
        ),
        "batter_exposure_1y_linear": entity_exposure_direction(
            last_year, rows, "batter_id", log_transform=False
        ),
        "pitcher_exposure_2y_linear": entity_exposure_direction(
            two_year, rows, "pitcher_id", log_transform=False
        ),
        "asof_batter_n_log": asof_level_direction(
            all_prior, rows, "asof_batter_n"
        ),
    }


def apply_direction(
    base: np.ndarray, direction: np.ndarray, eta: float
) -> np.ndarray:
    return np.clip(
        np.asarray(base, dtype=np.float64)
        + float(eta) * np.asarray(direction, dtype=np.float64),
        0.001,
        0.999,
    )


def _source_base(
    values: dict[str, Any], axis: dict[str, np.ndarray]
) -> np.ndarray:
    return compose(
        values["component"], values["h1"],
        c3_mix(values["sign"], values["recent"], 0.15),
        axis, h1_weight=0.15,
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    columns = [
        "season", "game_month", "pitcher_id", "batter_id",
        "asof_batter_n", "balls_before", "strikes_before",
        "num_runners_on", "li", "control_success",
    ]
    raw = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    year_rows = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = year_rows[2023]["game_month"].ge(8).to_numpy()
    source_rows = {
        "full_2022": year_rows[2022],
        "late_2023": year_rows[2023].loc[late23].reset_index(drop=True),
    }
    source_directions = {
        "full_2022": direction_library(raw, year_rows[2022], 2022),
        "late_2023": {
            name: values[late23]
            for name, values in direction_library(raw, year_rows[2023], 2023).items()
        },
    }
    del source_rows
    source_base = {
        name: _source_base(source[name], axes[name]) for name in SOURCE_AXES
    }

    rows = []
    details = {}
    source_candidates = {}
    for direction_name in source_directions["full_2022"]:
        for eta in ETAS:
            key = f"{direction_name}__eta{eta:g}"
            per_axis = {}
            candidates = {}
            for axis_name in SOURCE_AXES:
                candidate = apply_direction(
                    source_base[axis_name],
                    source_directions[axis_name][direction_name], eta,
                )
                active = np.asarray(axes[axis_name]["exact_mask"], dtype=bool)
                per_axis[axis_name] = paired_metrics(
                    axes[axis_name], source_base[axis_name], candidate, active
                )
                candidates[axis_name] = candidate
            passed = all(
                per_axis[name]["overall_gain"] > 0.0
                and per_axis[name]["positive_month_fraction"] >= 0.75
                and per_axis[name]["worst_month_gain"] > -5.0
                for name in SOURCE_AXES
            )
            rows.append({
                "key": key,
                "direction": direction_name,
                "eta": eta,
                "source_gate_passed": passed,
                "source_min_gain": min(
                    per_axis[name]["overall_gain"] for name in SOURCE_AXES
                ),
                "source_mean_gain": float(np.mean([
                    per_axis[name]["overall_gain"] for name in SOURCE_AXES
                ])),
                "source_worst_month": min(
                    per_axis[name]["worst_month_gain"] for name in SOURCE_AXES
                ),
            })
            details[key] = per_axis
            source_candidates[key] = candidates

    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain"],
        ascending=False, kind="stable",
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    direction_name = str(selected["direction"])
    eta = float(selected["eta"])

    locked, parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024],
        h1_path, c3_path, v160_path,
    )
    official = _jy_prediction(
        axes["full_2024"], frames[2024], locked, locked["h1"], bridge_oof
    )
    locked_directions = direction_library(raw, year_rows[2024], 2024)
    locked_family = []
    locked_details = {}
    selected_candidate = None
    for candidate_eta in ETAS:
        candidate = apply_direction(
            official, locked_directions[direction_name], candidate_eta
        )
        locked_family.append(candidate)
        locked_details[f"{direction_name}__eta{candidate_eta:g}"] = paired_metrics(
            axes["full_2024"], official, candidate,
            np.asarray(axes["full_2024"]["exact_mask"], dtype=bool),
        )
        if candidate_eta == eta:
            selected_candidate = candidate
    assert selected_candidate is not None
    robust = _robustness(
        axes["full_2024"], official, selected_candidate, locked_family
    )
    locked_selected = locked_details[f"{direction_name}__eta{eta:g}"]
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] < 0.10
    )
    promote = bool(
        selected["source_gate_passed"]
        and locked_selected["overall_gain"] > 0.0
        and robust_pass
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=source_candidates[str(selected["key"])]["full_2022"],
        late_2023=source_candidates[str(selected["key"])]["late_2023"],
        full_2024=selected_candidate,
        locked_direction=locked_directions[direction_name],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_release_build" if promote else "reject",
        "public_hypothesis_only": (
            "historical player exposure may contain a residual level direction"
        ),
        "leaderboard_fitted_coefficient_reused": False,
        "selected": selected,
        "source_details": details,
        "parity": parity,
        "locked_details": locked_details,
        "locked_selected": locked_selected,
        "locked_robustness": robust,
        "eligible_for_packaging": promote,
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "test_distribution_center_or_scale_used": False,
            "leaderboard_score_used_for_selection": False,
            "full_2024_used_for_recipe_selection": False,
            "official_train_only": True,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected"],
        "source": result["source_details"][result["selected"]["key"]],
        "locked": result["locked_selected"],
        "robustness": result["locked_robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

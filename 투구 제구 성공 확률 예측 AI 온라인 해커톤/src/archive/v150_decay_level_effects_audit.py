"""Audit decayed contrast and residual-level effects above frozen v148."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics
from src.champion.v148_contract import reconstruct_deployed_v148
from src.champion.v130_hoo_independent_oof_blend import post4
from src.champion.v133_hoo_h1_c3_forward import CONTEXT_COLS
from src.core.contract import _load_contract_axis


PROTOCOL = "V150_DECAY_LEVEL_EFFECTS_AUDIT_V1"
SOURCE_AXES = ("full_2022", "late_2023")


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _active(axis: dict[str, np.ndarray]) -> np.ndarray:
    return axis["exact_mask"].astype(bool) & (axis["domain3"].astype(str) == "R_CORE")


def _weighted_level(
    keys: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    query_keys: np.ndarray,
    shrink: float,
) -> np.ndarray:
    table = pd.DataFrame({"key": keys, "sr": residual * weights, "w": weights})
    grouped = table.groupby("key", sort=False)[["sr", "w"]].sum()
    values = grouped["sr"] / (grouped["w"] + float(shrink))
    return pd.Series(query_keys).map(values).fillna(0.0).to_numpy(np.float64)


def _weighted_contrast(
    keys: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    query_keys: np.ndarray,
    query_context: np.ndarray,
    shrink: float,
) -> np.ndarray:
    table = pd.DataFrame({
        "key": keys, "context": context, "sr": residual * weights, "w": weights
    }).groupby(["key", "context"], sort=False)[["sr", "w"]].sum().unstack()
    for field in ("sr", "w"):
        for value in (0, 1):
            if (field, value) not in table:
                table[(field, value)] = 0.0
    n0 = table[("w", 0)].fillna(0.0)
    n1 = table[("w", 1)].fillna(0.0)
    m0 = table[("sr", 0)] / n0.replace(0.0, np.nan)
    m1 = table[("sr", 1)] / n1.replace(0.0, np.nan)
    effective = (n0 * n1) / (n0 + n1).replace(0.0, np.nan)
    values = ((m1 - m0) * effective / (effective + float(shrink))).dropna()
    magnitude = pd.Series(query_keys).map(values).fillna(0.0).to_numpy(np.float64)
    return np.where(query_context == 1, 0.5 * magnitude, -0.5 * magnitude)


def _effects_for_year(
    frames: dict[int, pd.DataFrame],
    residual: dict[int, np.ndarray],
    year: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = frames[year]
    # H1 is honest OOF from 2020 onward; 2019 has no earlier-season model and
    # therefore cannot supply a comparable residual table.
    all_years = tuple(value for value in sorted(residual) if value < year)
    latest = max(all_years)
    contrast_parts: list[pd.DataFrame] = []
    contrast_residual: list[np.ndarray] = []
    contrast_weights: list[np.ndarray] = []
    for source_year in all_years:
        weight = float(config["contrast_gamma"]) ** (latest - source_year)
        contrast_parts.append(frames[source_year])
        contrast_residual.append(residual[source_year])
        contrast_weights.append(np.full(len(frames[source_year]), weight, dtype=np.float64))
    history = pd.concat(contrast_parts, ignore_index=True)
    history_residual = np.concatenate(contrast_residual)
    history_weights = np.concatenate(contrast_weights)
    pitcher = history["pitcher_id"].to_numpy(np.int64)
    query_pitcher = query["pitcher_id"].to_numpy(np.int64)
    contexts = (
        (history["pitcher_hand"].to_numpy(np.int8) == history["batter_hand"].to_numpy(np.int8)).astype(np.int8),
        (history["strikes_before"].to_numpy(np.int8) == 2).astype(np.int8),
        (history["num_runners_on"].to_numpy(np.int8) > 0).astype(np.int8),
    )
    query_contexts = (
        (query["pitcher_hand"].to_numpy(np.int8) == query["batter_hand"].to_numpy(np.int8)).astype(np.int8),
        (query["strikes_before"].to_numpy(np.int8) == 2).astype(np.int8),
        (query["num_runners_on"].to_numpy(np.int8) > 0).astype(np.int8),
    )
    contrast = np.zeros(len(query), dtype=np.float64)
    for context, query_context, shrink in zip(
        contexts, query_contexts, config["contrast_shrink"]
    ):
        contrast += _weighted_contrast(
            pitcher, context, history_residual, history_weights,
            query_pitcher, query_context, float(shrink)
        )

    pitcher_years = all_years[-int(config["pitcher_history_seasons"]):]
    pitcher_history = pd.concat([frames[value] for value in pitcher_years], ignore_index=True)
    pitcher_residual = np.concatenate([residual[value] for value in pitcher_years])
    pitcher_level = _weighted_level(
        pitcher_history["pitcher_id"].to_numpy(np.int64), pitcher_residual,
        np.ones(len(pitcher_history), dtype=np.float64), query_pitcher,
        float(config["pitcher_level_shrink"]),
    )

    batter_parts: list[pd.DataFrame] = []
    batter_residual: list[np.ndarray] = []
    batter_weights: list[np.ndarray] = []
    for source_year in all_years:
        weight = float(config["batter_gamma"]) ** (latest - source_year)
        batter_parts.append(frames[source_year])
        batter_residual.append(residual[source_year])
        batter_weights.append(np.full(len(frames[source_year]), weight, dtype=np.float64))
    batter_history = pd.concat(batter_parts, ignore_index=True)
    batter_level = _weighted_level(
        batter_history["batter_id"].to_numpy(np.int64), np.concatenate(batter_residual),
        np.concatenate(batter_weights), query["batter_id"].to_numpy(np.int64),
        float(config["batter_level_shrink"]),
    )
    return contrast, pitcher_level, batter_level


def _apply(
    baseline: np.ndarray,
    active: np.ndarray,
    current_c3: np.ndarray,
    decay_c3: np.ndarray,
    pitcher_level: np.ndarray,
    batter_level: np.ndarray,
    contrast_replace: float,
    pitcher_weight: float,
    batter_weight: float,
) -> np.ndarray:
    output = np.asarray(baseline, dtype=np.float64).copy()
    delta = (
        0.5 * contrast_replace * (decay_c3 - current_c3)
        + pitcher_weight * pitcher_level
        + batter_weight * batter_level
    )
    output[active] = np.clip(output[active] + delta[active], 0.001, 0.999)
    return output


def _point(axis: dict[str, np.ndarray], parent: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    return _axis_metrics({**axis, "parent": parent}, candidate)


def _source_pass(total: dict[str, Any], inc: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        total["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and total["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and total["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
        and inc["gain"] >= float(gate["incremental_gain_min"])
        and inc["worst_month_gain"] > float(gate["incremental_worst_month_gain_min_exclusive"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v131_dir: Path,
    v133_dir: Path,
    v136_dir: Path,
    v139_dir: Path,
    v141_dir: Path,
    v147_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = list(CONTEXT_COLS) + ["batter_id"]
    context = pd.read_csv(train_csv, usecols=usecols, low_memory=False)
    season = context["season"].to_numpy(np.int16)
    frames = {year: context.loc[season == year].reset_index(drop=True) for year in range(2019, 2025)}
    with np.load(v133_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        h1 = {year: saved[f"h1_{year}"].astype(np.float64) for year in range(2020, 2025)}
    post = {
        year: post4(context.loc[season < year].reset_index(drop=True), frames[year])
        for year in range(2020, 2025)
    }
    residual = {
        year: frames[year]["control_success"].to_numpy(np.float64) - (h1[year] + post[year])
        for year in range(2020, 2025)
    }
    effects = {year: _effects_for_year(frames, residual, year, config) for year in (2022, 2023, 2024)}

    axes = {name: _load_contract_axis(contract_dir / f"v84_{name}.npz") for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v131_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v131 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v136_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v136 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v139_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v139 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v141_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v124_2024 = saved["v124_full_2024"].astype(np.float64)
    with np.load(v147_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v142_2024 = saved["v142_full_2024"].astype(np.float64)
        v138_2024 = saved["v138_full_2024"].astype(np.float64)
        v148_2024 = reconstruct_deployed_v148(v142_2024, v138_2024)

    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    effect_axes = {
        "full_2022": effects[2022],
        "late_2023": tuple(value[late23] for value in effects[2023]),
        "full_2024": effects[2024],
    }
    current: dict[str, np.ndarray] = {}
    current_c3: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for name in (*SOURCE_AXES, "full_2024"):
        active[name] = _active(axes[name])
        sign = np.zeros(len(v104[name]), dtype=np.float64)
        recent = np.zeros(len(v104[name]), dtype=np.float64)
        sign[active[name]] = (v139[name][active[name]] - v131[name][active[name]]) / 0.5
        recent[active[name]] = (v136[name][active[name]] - v131[name][active[name]]) / 0.5
        current_c3[name] = 0.85 * sign + 0.15 * recent
        current[name] = v131[name].copy()
        current[name][active[name]] = np.clip(
            current[name][active[name]] + 0.5 * current_c3[name][active[name]], 0.001, 0.999
        )
    current["full_2024"] = v148_2024

    rows: list[dict[str, Any]] = []
    metrics_by_key: dict[tuple[float, float, float], dict[str, Any]] = {}
    for contrast_replace in config["contrast_replace_grid"]:
        for pitcher_weight in config["pitcher_weight_grid"]:
            for batter_weight in config["batter_weight_grid"]:
                key = (float(contrast_replace), float(pitcher_weight), float(batter_weight))
                metrics: dict[str, Any] = {}
                passes = []
                for name in SOURCE_AXES:
                    decay, pitcher, batter = effect_axes[name]
                    candidate = _apply(current[name], active[name], current_c3[name], decay, pitcher, batter, *key)
                    total = _point(axes[name], v104[name], candidate)
                    inc = _point(axes[name], current[name], candidate)
                    metrics[name] = {"total": total, "incremental": inc}
                    passes.append(_source_pass(total, inc, config["source_gate"]))
                metrics_by_key[key] = metrics
                rows.append({
                    "contrast_replace": key[0], "pitcher_weight": key[1], "batter_weight": key[2],
                    "source_gate_passed": all(passes),
                    "minimum_total_gain": min(x["total"]["gain"] for x in metrics.values()),
                    "minimum_incremental_gain": min(x["incremental"]["gain"] for x in metrics.values()),
                    "mean_incremental_gain": np.mean([x["incremental"]["gain"] for x in metrics.values()]),
                    "worst_total_month_gain": min(x["total"]["worst_month_gain"] for x in metrics.values()),
                    "worst_incremental_month_gain": min(x["incremental"]["worst_month_gain"] for x in metrics.values()),
                })
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "minimum_incremental_gain", "mean_incremental_gain", "worst_incremental_month_gain"],
        ascending=False, kind="stable"
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)

    fixed = config["fixed_external_recipe"]
    fixed_key = (float(fixed["contrast_replace"]), float(fixed["pitcher_weight"]), float(fixed["batter_weight"]))
    selected_keys = [
        tuple(float(row[name]) for name in ("contrast_replace", "pitcher_weight", "batter_weight"))
        for row in ranking.loc[ranking["source_gate_passed"]].head(int(config["source_audit_top"])).to_dict("records")
    ]
    for key in (fixed_key, (1.0, 0.0, 0.0), (0.0, 2.0, 1.75), (0.0, 0.0, 0.0)):
        if key not in selected_keys:
            selected_keys.append(key)

    late24 = axes["full_2024"]["game_month"].astype(np.int16) >= 8
    locked_rows: list[dict[str, Any]] = []
    candidates: dict[tuple[float, float, float], np.ndarray] = {}
    decay, pitcher, batter = effect_axes["full_2024"]
    for key in selected_keys:
        candidate = _apply(v148_2024, active["full_2024"], current_c3["full_2024"], decay, pitcher, batter, *key)
        candidates[key] = candidate
        inc = _point(axes["full_2024"], v148_2024, candidate)
        late_inc = _point(_slice_axis(axes["full_2024"], late24), v148_2024[late24], candidate[late24])
        total = _point(axes["full_2024"], v124_2024, candidate)
        source_metric = metrics_by_key[key]
        locked_rows.append({
            "contrast_replace": key[0], "pitcher_weight": key[1], "batter_weight": key[2],
            "fixed_external_recipe": key == fixed_key,
            "source_minimum_incremental_gain": min(x["incremental"]["gain"] for x in source_metric.values()),
            "source_worst_incremental_month_gain": min(x["incremental"]["worst_month_gain"] for x in source_metric.values()),
            "incremental_gain_vs_v148": inc["gain"],
            "incremental_late_gain_vs_v148": late_inc["gain"],
            "incremental_positive_month_fraction": inc["positive_month_fraction"],
            "incremental_worst_month_gain": inc["worst_month_gain"],
            "gain_vs_v124": total["gain"],
        })
    locked = pd.DataFrame(locked_rows)
    gate = config["locked_gate"]
    locked["locked_gate_passed"] = (
        (locked["incremental_gain_vs_v148"] >= float(gate["incremental_gain_vs_v148_min"]))
        & (locked["incremental_late_gain_vs_v148"] >= float(gate["incremental_late_gain_vs_v148_min"]))
        & (locked["incremental_positive_month_fraction"] >= float(gate["incremental_positive_month_fraction_min"]))
        & (locked["incremental_worst_month_gain"] > float(gate["incremental_worst_month_gain_min_exclusive"]))
        & (locked["gain_vs_v124"] >= float(gate["gain_vs_v124_min"]))
    )
    calibration = config["public_calibration"]
    locked["estimated_public"] = float(calibration["v148_score"]) + (
        float(calibration["conservative_transfer_ratio"]) * locked["incremental_gain_vs_v148"]
    )
    locked = locked.sort_values(
        ["locked_gate_passed", "estimated_public", "source_minimum_incremental_gain"],
        ascending=False, kind="stable"
    ).reset_index(drop=True)
    locked.to_csv(output_dir / "locked_ranking.csv", index=False)
    passing = locked.loc[locked["locked_gate_passed"]]
    selected = (passing.iloc[0] if len(passing) else locked.iloc[0]).to_dict()
    selected_key = tuple(float(selected[name]) for name in ("contrast_replace", "pitcher_weight", "batter_weight"))
    np.savez_compressed(
        output_dir / "selected_axes.npz", full_2024=candidates[selected_key],
        late_2024=candidates[selected_key][late24], v148_full_2024=v148_2024,
        v124_full_2024=v124_2024, active_mask=active["full_2024"],
        decay_c3_full_2024=decay, pitcher_level_full_2024=pitcher,
        batter_level_full_2024=batter,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_packaging" if bool(selected["locked_gate_passed"]) else "reject",
        "n_source_points": int(len(ranking)), "n_locked_points": int(len(locked)),
        "selected": selected, "fixed_external_recipe_result": next(
            row for row in locked.to_dict("records") if bool(row["fixed_external_recipe"])
        ),
        "top20": locked.head(20).to_dict("records"),
        "eligible_for_packaging": bool(
            selected["locked_gate_passed"]
            and selected["estimated_public"] >= float(calibration["target_score_min"])
        ),
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v131-dir", type=Path, required=True)
    parser.add_argument("--v133-dir", type=Path, required=True)
    parser.add_argument("--v136-dir", type=Path, required=True)
    parser.add_argument("--v139-dir", type=Path, required=True)
    parser.add_argument("--v141-dir", type=Path, required=True)
    parser.add_argument("--v147-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.v104_dir, args.v131_dir, args.v133_dir,
        args.v136_dir, args.v139_dir, args.v141_dir, args.v147_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()

"""Select a deployed-H1 seed subset on source OOF, then audit current JY.

The production H1 is the mean of CatBoost seeds 42/43/44.  Exact strict-
forward predictions for each seed already exist from v157.  This experiment
enumerates the seven non-empty subsets, selects only on full-2022 and
late-2023, and opens full-2024 once after the recipe is fixed.  C3 tables,
the fixed H1 affine, post4, bridge, JY gate, and every non-H1 component are
held fixed.  No model is trained and no test row or leaderboard value is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import (
    BRIDGE_SCALE,
    C3_ACTIVE_RECENT_WEIGHT,
    C3_BASE_RECENT_WEIGHT,
    H1_ACTIVE_WEIGHT,
    H1_BASE_WEIGHT,
    _load_year_context,
    _locked_contract,
    _source_contracts,
    affine,
    c3_mix,
    compose,
    gate_library,
    metrics,
    overwrite_gate,
)
from src.core.contract import _load_contract_axis
from src.core.robustness import evaluate_robustness


PROTOCOL = "V172_H1_SEED_SUBSET_AUDIT_V1"
SEEDS = (42, 43, 44)


def seed_subsets() -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for size in range(1, len(SEEDS) + 1)
        for subset in itertools.combinations(SEEDS, size)
    )


def subset_mean(checkpoints: dict[int, np.ndarray], subset: tuple[int, ...]) -> np.ndarray:
    if not subset or any(seed not in checkpoints for seed in subset):
        raise ValueError(f"invalid seed subset: {subset}")
    return np.mean(np.column_stack([checkpoints[seed] for seed in subset]), axis=1)


def _checkpoint_predictions(
    checkpoint_dir: Path, year: int
) -> dict[int, np.ndarray]:
    return {
        seed: np.load(
            checkpoint_dir / f"h1_year{year}_seed{seed}.npy", allow_pickle=False
        ).astype(np.float64)
        for seed in SEEDS
    }


def _source_formula(
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    component: np.ndarray,
    h1: np.ndarray,
    sign: np.ndarray,
    recent: np.ndarray,
) -> np.ndarray:
    c3_base = c3_mix(sign, recent, C3_BASE_RECENT_WEIGHT)
    c3_active = c3_mix(sign, recent, C3_ACTIVE_RECENT_WEIGHT)
    base = compose(component, h1, c3_base, axis, h1_weight=H1_BASE_WEIGHT)
    proposal = compose(
        component, h1, c3_active, axis, h1_weight=H1_ACTIVE_WEIGHT
    )
    output, _active = overwrite_gate(
        base, proposal, axis, frame, gate_library()["runners_or_high_li"]
    )
    return output


def _locked_formula(
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    values: dict[str, np.ndarray],
    h1: np.ndarray,
    bridge_oof: Path,
) -> np.ndarray:
    with np.load(bridge_oof, allow_pickle=False) as saved:
        historical_parent = saved["parent"].astype(np.float64)
        bridge025_top = saved["v142_full_2024"].astype(np.float64) + 0.25 * (
            saved["v138_full_2024"].astype(np.float64)
            - saved["v142_full_2024"].astype(np.float64)
        )
    component_delta = (bridge025_top - historical_parent) / (
        1.0 - H1_BASE_WEIGHT
    )
    bridge_component = values["component"] + BRIDGE_SCALE * component_delta
    base = compose(
        values["component"], h1, values["c3_base"], axis,
        h1_weight=H1_BASE_WEIGHT,
    )
    proposal = compose(
        bridge_component, h1, values["c3_active"], axis,
        h1_weight=H1_ACTIVE_WEIGHT,
    )
    output, _active = overwrite_gate(
        base, proposal, axis, frame, gate_library()["runners_or_high_li"]
    )
    return output


def _point_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.75
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
    checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    positions = {
        "full_2022": np.arange(len(frames[2022])),
        "late_2023": np.flatnonzero(late23),
        "full_2024": np.arange(len(frames[2024])),
    }
    years = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    checkpoints = {
        year: _checkpoint_predictions(checkpoint_dir, year)
        for year in (2022, 2023, 2024)
    }
    source_current = {
        name: _source_formula(
            axes[name], values["frame"], values["component"], values["h1"],
            values["sign"], values["recent"],
        )
        for name, values in source.items()
    }

    locked_values, parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024], h1_path, c3_path,
        v160_path,
    )
    locked_current = _locked_formula(
        axes["full_2024"], frames[2024], locked_values, locked_values["h1"],
        bridge_oof,
    )

    rows: list[dict[str, Any]] = []
    source_candidates: dict[tuple[int, ...], dict[str, np.ndarray]] = {}
    for subset in seed_subsets():
        candidates: dict[str, np.ndarray] = {}
        source_results: dict[str, dict[str, Any]] = {}
        for name in ("full_2022", "late_2023"):
            year = years[name]
            raw = subset_mean(checkpoints[year], subset)[positions[name]]
            h1 = affine(raw + correction[year][positions[name]])
            values = source[name]
            candidate = _source_formula(
                axes[name], values["frame"], values["component"], h1,
                values["sign"], values["recent"],
            )
            candidates[name] = candidate
            source_results[name] = metrics(
                axes[name], source_current[name], candidate
            )
        source_candidates[subset] = candidates
        rows.append({
            "subset": "+".join(map(str, subset)),
            "seed_count": len(subset),
            "source_gate_passed": all(_point_gate(x) for x in source_results.values()),
            "source_min_gain": min(x["gain"] for x in source_results.values()),
            "source_mean_gain": float(np.mean([x["gain"] for x in source_results.values()])),
            "source_worst_month": min(
                x["worst_month_gain"] for x in source_results.values()
            ),
            "full_2022_gain": source_results["full_2022"]["gain"],
            "late_2023_gain": source_results["late_2023"]["gain"],
        })

    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain", "seed_count"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    selected = tuple(int(x) for x in str(ranking.iloc[0]["subset"]).split("+"))
    selected_raw24 = subset_mean(checkpoints[2024], selected)
    selected_h124 = affine(selected_raw24 + correction[2024])
    selected24 = _locked_formula(
        axes["full_2024"], frames[2024], locked_values, selected_h124, bridge_oof
    )
    locked_result = metrics(axes["full_2024"], locked_current, selected24)

    family24 = []
    for subset in seed_subsets():
        raw = subset_mean(checkpoints[2024], subset)
        family24.append(
            _locked_formula(
                axes["full_2024"], frames[2024], locked_values,
                affine(raw + correction[2024]), bridge_oof,
            )
        )
    robustness_axis = {**axes["full_2024"], "parent": locked_current}
    robust = evaluate_robustness(
        robustness_axis,
        selected24,
        family24,
        {"bootstrap_resamples": 1000, "block_size": 4096, "seed": 17242},
    )
    source_passed = bool(ranking.iloc[0]["source_gate_passed"])
    locked_passed = bool(
        _point_gate(locked_result)
        and robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )
    promotion = source_passed and locked_passed

    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=source_candidates[selected]["full_2022"],
        late_2023=source_candidates[selected]["late_2023"],
        full_2024=selected24,
        current_full_2024=locked_current,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_packaging" if promotion else "rejected",
        "selected_source_only_subset": list(selected),
        "selected_source_row": ranking.iloc[0].to_dict(),
        "locked_2024_metrics": locked_result,
        "locked_2024_robustness": robust,
        "source_gate_passed": source_passed,
        "locked_gate_passed": locked_passed,
        "promotion_gate_passed": promotion,
        "current_contract_parity": parity,
        "limitations": [
            "Only three pre-existing seeds are available, so subset selection has limited effective diversity.",
            "Full-2024 is development-contaminated and is opened only after source selection.",
            "C3 remains frozen from the original three-seed ensemble residuals.",
        ],
        "restrictions": {
            "model_refit": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "full_2024_used_for_recipe_selection": False,
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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.checkpoint_dir,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

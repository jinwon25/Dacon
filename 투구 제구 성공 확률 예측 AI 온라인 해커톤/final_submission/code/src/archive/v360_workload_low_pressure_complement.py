"""Audit the frozen v209 workload-H1 proposal on v345's unused R_CORE rows.

The v209 three-seed workload expert is already used by v345 only when runners
are on base or leverage is high.  This experiment does not tune the model,
weight, or gate.  It reconstructs the frozen 0.18 proposal and applies its
increment only to the disjoint complement: R_CORE, bases empty, LI < 1.5.
Full-2022 and late-2023 are source checks; full-2024 is the final audit above
the exact v345 parent.  Every deployment feature is current-row local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import (
    BRIDGE_SCALE,
    H1_BASE_WEIGHT,
    _load_year_context,
    _locked_contract,
    _source_contracts,
    affine,
    c3_mix,
    compose,
    gate_library,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v209_h1_workload_multiseed_audit import (
    PRIMARY_ACTIVE_WEIGHT,
    SEEDS,
    YEARS,
    load_seed_predictions,
)
from src.archive.v205_h1_workload_strict_forward_audit import (
    load_batter_ids_by_year,
)
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V360_WORKLOAD_LOW_PRESSURE_COMPLEMENT_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")


def _frozen_directions(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    context, frames, correction = _load_year_context(train_csv)
    season = context["season"].to_numpy(np.int16)
    batter_ids = load_batter_ids_by_year(train_csv, season)
    for year in YEARS:
        frames[year] = frames[year].assign(batter_id=batter_ids[year])
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    sources = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    locked, _ = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024],
        h1_path, c3_path, v160_path,
    )

    augmented_h1: dict[int, np.ndarray] = {}
    baseline_parity: dict[str, float] = {}
    for year in YEARS:
        baseline = []
        augmented = []
        for seed in SEEDS:
            base, aug = load_seed_predictions(
                year, seed, baseline_checkpoint_dir, seed42_dir, multiseed_dir
            )
            baseline.append(base)
            augmented.append(aug)
        baseline_h1 = affine(np.mean(baseline, axis=0) + correction[year])
        augmented_h1[year] = affine(np.mean(augmented, axis=0) + correction[year])
        reference = (
            sources["full_2022"]["h1"] if year == 2022
            else sources["late_2023"]["h1"] if year == 2023
            else locked["h1"]
        )
        compared = baseline_h1 if year != 2023 else baseline_h1[late23]
        baseline_parity[str(year)] = float(np.max(np.abs(compared - reference)))
    if max(baseline_parity.values()) > 1e-12:
        raise ValueError(f"v209 baseline H1 parity failed: {baseline_parity}")

    source_parts = {
        name: {
            "component": sources[name]["component"],
            "c3": c3_mix(sources[name]["sign"], sources[name]["recent"], 0.25),
        }
        for name in ("full_2022", "late_2023")
    }
    with np.load(bridge_oof, allow_pickle=False) as saved:
        historical_parent = saved["parent"].astype(np.float64)
        bridge025_top = saved["v142_full_2024"].astype(np.float64) + 0.25 * (
            saved["v138_full_2024"].astype(np.float64)
            - saved["v142_full_2024"].astype(np.float64)
        )
    component_delta = (bridge025_top - historical_parent) / (1.0 - H1_BASE_WEIGHT)
    locked_component = locked["component"] + BRIDGE_SCALE * component_delta

    proposals = {
        "full_2022": compose(
            source_parts["full_2022"]["component"], augmented_h1[2022],
            source_parts["full_2022"]["c3"], axes["full_2022"],
            h1_weight=PRIMARY_ACTIVE_WEIGHT,
        ),
        "late_2023": compose(
            source_parts["late_2023"]["component"], augmented_h1[2023][late23],
            source_parts["late_2023"]["c3"], axes["late_2023"],
            h1_weight=PRIMARY_ACTIVE_WEIGHT,
        ),
        "full_2024": compose(
            locked_component, augmented_h1[2024], locked["c3_active"],
            axes["full_2024"], h1_weight=PRIMARY_ACTIVE_WEIGHT,
        ),
    }
    directions = {name: proposals[name] - parents[name] for name in ORIGINS}
    return axis_frames, directions, {"parent_parity": parity, "h1_parity": baseline_parity}


def _complement_mask(frame: pd.DataFrame, domain: np.ndarray) -> np.ndarray:
    used = gate_library()["runners_or_high_li"](frame)
    return (np.asarray(domain).astype(str) == "R_CORE") & ~used


def run(
    train_csv: Path,
    baseline_checkpoint_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, directions, parity = _frozen_directions(
        train_csv, baseline_checkpoint_dir, seed42_dir, multiseed_dir,
        contract_dir, v104_path, h1_path, c3_path, v160_path, bridge_oof,
    )
    with np.load(v335_axes, allow_pickle=False) as saved:
        v335 = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in ORIGINS
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        if not np.array_equal(saved["parent_full_2024"], v335["full_2024"]):
            raise ValueError("v345/v335 parent mismatch")
        v345 = saved["candidate_full_2024"].astype(np.float64)

    parents = {
        "full_2022": v335["full_2022"],
        "late_2023": v335["late_2023"],
        "full_2024": v345,
    }
    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    domains = {
        name: np.where(
            frames[name]["game_type"].astype(str).eq("R"),
            np.where(
                frames[name]["pitcher_team_id"].eq(13)
                | frames[name]["batter_team_id"].eq(13),
                "R_ANCHOR", "R_CORE",
            ),
            "F",
        )
        for name in ORIGINS
    }
    for name in ORIGINS:
        active[name] = _complement_mask(frames[name], domains[name])
        candidates[name] = parents[name].copy()
        candidates[name][active[name]] = np.clip(
            parents[name][active[name]] + directions[name][active[name]],
            0.001, 0.999,
        )
        metrics[name] = axis_metrics(
            frames[name], parents[name], candidates[name], active[name]
        )

    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24, parents["full_2024"], candidates["full_2024"],
        active["full_2024"], [parents["full_2024"], candidates["full_2024"]],
    )
    source_pass = bool(
        metrics["full_2022"]["gain"] > 0.0
        and metrics["late_2023"]["gain"] > 0.0
        and metrics["full_2022"]["positive_month_fraction"] >= 4.0 / 7.0
        and metrics["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
    )
    locked = metrics["full_2024"]
    eligible = bool(
        source_pass
        and locked["gain"] >= 2.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0
        and robustness["pitcher"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parents[name] for name in ORIGINS},
        **{f"candidate_{name}": candidates[name] for name in ORIGINS},
        **{f"active_{name}": active[name] for name in ORIGINS},
        **{f"direction_{name}": directions[name] for name in ORIGINS},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if eligible else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "frozen_recipe": {
            "source": "v209 three-seed augmented H1",
            "h1_weight": PRIMARY_ACTIVE_WEIGHT,
            "scope": "R_CORE and not(runners_or_high_li)",
            "dose_retuned": False,
        },
        "metrics": metrics,
        "locked_robustness": robustness,
        "parity": parity,
        "source_gate_passed": source_pass,
        "eligible_for_packaging": eligible,
        "restrictions": {
            "official_train_only": True,
            "frozen_v209_model_weight_and_direction": True,
            "disjoint_from_v345_workload_gate": True,
            "v345_f_and_r_anchor_preserved_exactly": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed42-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.baseline_checkpoint_dir, args.seed42_dir,
        args.multiseed_dir, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v335_axes,
        args.v345_axes, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

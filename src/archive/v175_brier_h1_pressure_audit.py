"""Locked diagnostic for the source-positive RMSE-H1 pressure-count route.

The route and convex mixture are frozen from full-2022 and late-2023:
``strikes_before == 2 or balls_before == 3`` and alpha 0.50.  Because the
source monthly-positive fractions are below the standing 0.75 gate, this audit
cannot package a release; full-2024 only decides whether a multi-seed
replication is worth the cost.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import (
    _load_year_context,
    _locked_contract,
    _source_contracts,
    affine,
)
from src.archive.v173_h1_noncore_extension_audit import paired_metrics
from src.archive.v174_brier_h1_regressor import (
    _fit_year,
    _jy_prediction,
    _prepare_features,
    _robustness,
    _source_candidate,
    mix_h1,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V175_BRIER_H1_PRESSURE_AUDIT_V1"
ALPHA = 0.50


def pressure_mask(frame: Any) -> np.ndarray:
    return (
        frame["strikes_before"].to_numpy() >= 2
    ) | (frame["balls_before"].to_numpy() >= 3)


def route_candidate(
    base: np.ndarray,
    alternative: np.ndarray,
    axis: dict[str, np.ndarray],
    frame: Any,
) -> tuple[np.ndarray, np.ndarray]:
    active = (
        np.asarray(axis["exact_mask"], dtype=bool)
        & (np.asarray(axis["domain3"]).astype(str) == "R_CORE")
        & pressure_mask(frame)
    )
    output = np.asarray(base, dtype=np.float64).copy()
    output[active] = np.asarray(alternative, dtype=np.float64)[active]
    return output, active


def run(
    train_csv: Path,
    trackman_csv: Path,
    external_root: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    source_checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, external_root
    )
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    raw22 = np.load(
        source_checkpoint_dir / "rmse_h1_2022_seed17442.npy", allow_pickle=False
    )
    raw23 = np.load(
        source_checkpoint_dir / "rmse_h1_2023_seed17442.npy", allow_pickle=False
    )
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    challenger_source = {
        "full_2022": affine(raw22 + correction[2022]),
        "late_2023": affine((raw23 + correction[2023])[late23]),
    }
    source_metrics = {}
    for name in ("full_2022", "late_2023"):
        base, alternative, _rcore = _source_candidate(
            source[name], axes[name], challenger_source[name], ALPHA
        )
        candidate, active = route_candidate(
            base, alternative, axes[name], source[name]["frame"]
        )
        source_metrics[name] = paired_metrics(
            axes[name], base, candidate, active
        )

    raw24 = _fit_year(
        train, target, season, 2024, h1_features,
        output_dir / "checkpoints" / "rmse_h1_2024_seed17442.npy",
    )
    locked, parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024],
        h1_path, c3_path, v160_path,
    )
    challenger24 = affine(raw24 + correction[2024])
    official = _jy_prediction(
        axes["full_2024"], frames[2024], locked, locked["h1"], bridge_oof
    )
    family = []
    locked_family = {}
    selected_candidate = None
    selected_active = None
    for alpha in (0.25, 0.50, 1.0):
        mixed = mix_h1(locked["h1"], challenger24, alpha)
        alternative = _jy_prediction(
            axes["full_2024"], frames[2024], locked, mixed, bridge_oof
        )
        candidate, active = route_candidate(
            official, alternative, axes["full_2024"], frames[2024]
        )
        family.append(candidate)
        locked_family[f"pressure_a{alpha:g}"] = paired_metrics(
            axes["full_2024"], official, candidate, active
        )
        if alpha == ALPHA:
            selected_candidate, selected_active = candidate, active
    assert selected_candidate is not None and selected_active is not None
    robust = _robustness(
        axes["full_2024"], official, selected_candidate, family
    )
    selected = locked_family[f"pressure_a{ALPHA:g}"]
    result = {
        "protocol": PROTOCOL,
        "status": "diagnostic_only",
        "selected_recipe": {
            "gate": "R_CORE and (strikes_before >= 2 or balls_before >= 3)",
            "alpha": ALPHA,
            "selection_axes": ["full_2022", "late_2023"],
        },
        "source_metrics": source_metrics,
        "source_standing_gate_passed": False,
        "source_gate_failure": (
            "positive-month fractions are below 0.75 despite positive total "
            "gain on both source axes"
        ),
        "parity": parity,
        "locked_family": locked_family,
        "locked_selected": selected,
        "locked_robustness": robust,
        "eligible_for_packaging": False,
        "multi_seed_replication_warranted": bool(
            selected["overall_gain"] >= 1.0
            and robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
        ),
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "leaderboard_score_used_for_selection": False,
            "full_2024_used_for_recipe_selection": False,
            "official_train_only_for_fitting": True,
            "row_local_inference": True,
        },
    }
    np.savez_compressed(
        output_dir / "diagnostic_axes.npz",
        full_2024=selected_candidate,
        official_full_2024=official,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    del train
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--source-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.external_root,
        args.contract_dir, args.v104_path, args.h1_path, args.c3_path,
        args.v160_path, args.bridge_oof, args.source_checkpoint_dir,
        args.output_dir,
    )
    print(json.dumps({
        "source": result["source_metrics"],
        "locked": result["locked_selected"],
        "robustness": result["locked_robustness"],
        "multi_seed_replication_warranted": result[
            "multi_seed_replication_warranted"
        ],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

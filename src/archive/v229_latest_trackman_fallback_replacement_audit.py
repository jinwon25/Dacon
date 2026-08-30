"""Locked Public1175 replacement audit for the v228 latest-TrackMan XGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.archive.v226_recent2_fallback_replacement_audit import (
    full_axis,
    locked_point_gate,
    source_sanity,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V229_LATEST_TRACKMAN_FALLBACK_REPLACEMENT_LOCKED_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")


def restrictions() -> dict[str, bool]:
    return {
        "public1175_evidence_parent_frozen": True,
        "pressure_gate_frozen": True,
        "parent_probability_threshold_frozen_at_050": True,
        "fallback_weight_frozen_at_030": True,
        "active_rows_must_match_current_public1175": True,
        "single_latest_trackman_candidate_only": True,
        "locked_2024_not_used_for_feature_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    current_oof_dir: Path,
    latest_trackman_oof_dir: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
    candidate_filename_template: str = "hyunku_latest_tm_xgb_{year}.npy",
    candidate_label: str = "latest_trackman",
    protocol: str = PROTOCOL,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "season", "game_type", "pitcher_team_id", "batter_team_id",
        "num_runners_on", "li", "game_month", "pitcher_id",
    ]
    train = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    season = train["season"].to_numpy(np.int16)
    frames = {
        year: train.loc[season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
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
    score_axes = {name: full_axis(axes[name]) for name in AXES}
    parent = {name: axes[name]["parent"].astype(np.float64) for name in AXES}
    current_full = {
        year: align_regular_prediction(
            frames[year], current_oof_dir / f"hyunku_fallback_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    latest_full = {
        year: align_regular_prediction(
            frames[year],
            latest_trackman_oof_dir / candidate_filename_template.format(year=year),
        )
        for year in (2022, 2023, 2024)
    }
    current_xgb = {
        "full_2022": current_full[2022],
        "late_2023": current_full[2023][late23],
        "full_2024": current_full[2024],
    }
    latest_xgb = {
        "full_2022": latest_full[2022],
        "late_2023": latest_full[2023][late23],
        "full_2024": latest_full[2024],
    }

    current: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    active_match: dict[str, bool] = {}
    absolute: dict[str, dict[str, Any]] = {}
    incremental: dict[str, dict[str, Any]] = {}
    for name in AXES:
        pressure = pressure_gate(axis_frames[name])
        current[name], current_active = apply_fallback(
            parent[name], current_xgb[name], pressure
        )
        candidate[name], candidate_active = apply_fallback(
            parent[name], latest_xgb[name], pressure
        )
        active[name] = candidate_active
        active_match[name] = bool(np.array_equal(current_active, candidate_active))
        absolute[name] = metrics(score_axes[name], parent[name], candidate[name])
        incremental[name] = metrics(score_axes[name], current[name], candidate[name])
        for result in (absolute[name], incremental[name]):
            result["active_rows"] = int(candidate_active.sum())
        incremental[name]["changed_rows"] = int(np.sum(
            np.abs(candidate[name] - current[name]) > 1e-15
        ))

    source_pass = source_sanity(absolute)
    point_pass = locked_point_gate(absolute["full_2024"], incremental["full_2024"])
    robustness = _robustness(
        score_axes["full_2024"], current["full_2024"], candidate["full_2024"],
        active["full_2024"], [current["full_2024"], candidate["full_2024"]],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] <= 0.10
    )
    confirm = bool(
        all(active_match.values()) and source_pass and point_pass and robust_pass
    )
    np.savez_compressed(
        output_dir / "replacement_axes.npz",
        **{f"parent_{name}": parent[name] for name in AXES},
        **{f"current_{name}": current[name] for name in AXES},
        **{f"candidate_{name}": candidate[name] for name in AXES},
        **{f"active_{name}": active[name] for name in AXES},
    )
    summary = {
        "protocol": protocol,
        "status": "confirmed_for_full_fit" if confirm else (
            "active_contract_failure" if not all(active_match.values()) else (
                "source_sanity_reject" if not source_pass else (
                    "locked_point_reject" if not point_pass else "robust_reject"
                )
            )
        ),
        "current_public1175_proxy_to_parent": {
            name: metrics(score_axes[name], parent[name], current[name])
            for name in AXES
        },
        f"{candidate_label}_to_parent": absolute,
        f"{candidate_label}_incremental_over_current": incremental,
        "active_row_match": active_match,
        "source_sanity_passed": source_pass,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": confirm,
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--latest-trackman-oof-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.current_oof_dir, args.latest_trackman_oof_dir,
        args.contract_dir, args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "absolute": result["latest_trackman_to_parent"],
        "incremental": result["latest_trackman_incremental_over_current"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

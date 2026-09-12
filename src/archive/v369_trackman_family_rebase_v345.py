"""Source-select the frozen v259 TrackMan family above v345.

The four direct TrackMan experts and their six equal pairs were declared in
v259.  Their fixed 25% fallback replacement deltas are reconstructed exactly,
then rebased above v345.  Selection is repeated only because the parent has
changed, using full-2022 and late-2023; full-2024 is opened for one pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v205_h1_workload_strict_forward_audit import load_batter_ids_by_year
from src.archive.v218_public1175_joint_h1_incremental_audit import align_regular_prediction
from src.archive.v241_mechanism_aware_fallback_expansion import route_masks
from src.archive.v248_fixed_route_dual_tree_fallback import compose
from src.archive.v259_independent_feature_family_audit import MODEL_STEMS, SUBSETS, blend_feature_models
from src.archive.v324_player_transition_residual import axis_metrics
from src.archive.v362_pitcher_situation_lowrank_v345 import full_row_rms


PROTOCOL = "V369_TRACKMAN_FAMILY_REBASE_V345_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")


def run(
    train_csv: Path,
    base_oof_dir: Path,
    command_oof_dir: Path,
    pitchmix_oof_dir: Path,
    count_oof_dir: Path,
    batter_oof_dir: Path,
    exact_axes: Path,
    v244_axes: Path,
    v335_axes: Path,
    v345_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context, yearly, _ = _load_year_context(train_csv)
    batter_ids = load_batter_ids_by_year(
        train_csv, context["season"].to_numpy(np.int16)
    )
    for year in (2022, 2023, 2024):
        yearly[year] = yearly[year].assign(batter_id=batter_ids[year])
    late23 = yearly[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": yearly[2022],
        "late_2023": yearly[2023].loc[late23].reset_index(drop=True),
        "full_2024": yearly[2024],
    }
    directories = {
        "command": command_oof_dir,
        "pitchmix": pitchmix_oof_dir,
        "count": count_oof_dir,
        "batter": batter_oof_dir,
    }
    base_full = {
        year: align_regular_prediction(yearly[year], base_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy")
        for year in (2022, 2023, 2024)
    }
    expert_full = {
        family: {
            year: align_regular_prediction(yearly[year], directory / f"{MODEL_STEMS[family]}_{year}.npy")
            for year in (2022, 2023, 2024)
        }
        for family, directory in directories.items()
    }
    base = {
        "full_2022": base_full[2022],
        "late_2023": base_full[2023][late23],
        "full_2024": base_full[2024],
    }
    experts = {
        family: {
            "full_2022": values[2022],
            "late_2023": values[2023][late23],
            "full_2024": values[2024],
        }
        for family, values in expert_full.items()
    }
    with np.load(exact_axes, allow_pickle=False) as saved:
        exact_parent = {
            name: saved[f"jy_parent_{name}"].astype(np.float64)
            for name in ORIGINS
        }
    with np.load(v244_axes, allow_pickle=False) as saved:
        expected_v244 = {
            name: saved[f"candidate_runtime_faithful_exact_jy_{name}"].astype(np.float64)
            for name in ORIGINS
        }
    routes = {name: route_masks(exact_parent[name], base[name], frames[name]) for name in ORIGINS}
    reconstructed_v244: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    for name in ORIGINS:
        reconstructed_v244[name], _ = compose(exact_parent[name], base[name], routes[name])
        parity[name] = float(np.max(np.abs(reconstructed_v244[name] - expected_v244[name])))
        if parity[name] > 1e-12:
            raise ValueError(f"v244 parity failed on {name}: {parity[name]}")
    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            name: saved[f"candidate_{name}"].astype(np.float64)
            for name in ("full_2022", "late_2023")
        }
    with np.load(v345_axes, allow_pickle=False) as saved:
        parents["full_2024"] = saved["candidate_full_2024"].astype(np.float64)
        v345_increment = saved["candidate_full_2024"].astype(np.float64) - saved["parent_full_2024"].astype(np.float64)

    source_screen: list[dict[str, Any]] = []
    source_payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for subset in SUBSETS:
        label = "+".join(subset)
        source_payload[label] = {}
        record: dict[str, Any] = {"candidate": label}
        for name in ("full_2022", "late_2023"):
            mixed = blend_feature_models(base[name], [experts[item][name] for item in subset])
            historical_candidate, _ = compose(exact_parent[name], mixed, routes[name])
            direction = historical_candidate - reconstructed_v244[name]
            candidate = np.clip(parents[name] + direction, 0.001, 0.999)
            active = np.abs(direction) > 1e-15
            source_payload[label][name] = direction, candidate, active
            metrics = axis_metrics(frames[name], parents[name], candidate, active)
            metrics["full_row_rms_shift"] = full_row_rms(parents[name], candidate)
            record[name] = metrics
        record["minimum_source_gain"] = min(record["full_2022"]["gain"], record["late_2023"]["gain"])
        record["source_pass"] = bool(
            record["full_2022"]["gain"] > 0.0
            and record["late_2023"]["gain"] > 0.0
            and record["full_2022"]["positive_month_fraction"] >= 2.0 / 3.0
            and record["late_2023"]["positive_month_fraction"] >= 2.0 / 3.0
            and record["full_2022"]["worst_month_gain"] > -10.0
            and record["late_2023"]["worst_month_gain"] > -10.0
        )
        source_screen.append(record)
    passing = [row for row in source_screen if row["source_pass"]]
    passing.sort(key=lambda row: (row["minimum_source_gain"], row["full_2022"]["gain"], row["late_2023"]["gain"]), reverse=True)
    restrictions = {
        "official_train_and_trackman_only": True,
        "frozen_v259_family_and_total_weight": True,
        "source_reselection_only_because_parent_changed": True,
        "full_2024_opened_for_one_source_selected_candidate": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }
    if not passing:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "v244_reconstruction_max_abs": parity,
            "source_screen": source_screen,
            "locked_origin_opened": False,
            "eligible_for_packaging": False,
            "restrictions": restrictions,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        return summary

    label = passing[0]["candidate"]
    subset = tuple(label.split("+"))
    name = "full_2024"
    mixed = blend_feature_models(base[name], [experts[item][name] for item in subset])
    historical_candidate, _ = compose(exact_parent[name], mixed, routes[name])
    direction = historical_candidate - reconstructed_v244[name]
    candidate = np.clip(parents[name] + direction, 0.001, 0.999)
    active = np.abs(direction) > 1e-15
    locked = axis_metrics(frames[name], parents[name], candidate, active)
    locked["full_row_rms_shift"] = full_row_rms(parents[name], candidate)
    union = active | (np.abs(v345_increment) > 1e-15)
    correlation = float(np.corrcoef(direction[union], v345_increment[union])[0, 1])
    axes24 = {
        "target": frames[name][TARGET].to_numpy(np.float64),
        "game_month": frames[name]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames[name]["pitcher_id"].to_numpy(),
        "batter_id": frames[name]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames[name]), dtype=bool),
    }
    robustness = _robustness(axes24, parents[name], candidate, active, [parents[name], candidate])
    passed = bool(
        locked["gain"] >= 1.0
        and locked["positive_month_fraction"] >= 0.5
        and locked["worst_month_gain"] > -15.0
        and locked["full_row_rms_shift"] >= 0.0005
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        parent_full_2022=parents["full_2022"],
        candidate_full_2022=source_payload[label]["full_2022"][1],
        active_full_2022=source_payload[label]["full_2022"][2],
        direction_full_2022=source_payload[label]["full_2022"][0],
        parent_late_2023=parents["late_2023"],
        candidate_late_2023=source_payload[label]["late_2023"][1],
        active_late_2023=source_payload[label]["late_2023"][2],
        direction_late_2023=source_payload[label]["late_2023"][0],
        parent_full_2024=parents[name],
        candidate_full_2024=candidate,
        active_full_2024=active,
        direction_full_2024=direction,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if passed else "locked_reject",
        "selected_candidate": label,
        "v244_reconstruction_max_abs": parity,
        "source_screen": source_screen,
        "locked_full_2024": locked,
        "increment_correlation_with_v345": correlation,
        "locked_robustness": robustness,
        "eligible_for_packaging": passed,
        "restrictions": restrictions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--base-oof-dir", type=Path, required=True)
    parser.add_argument("--command-oof-dir", type=Path, required=True)
    parser.add_argument("--pitchmix-oof-dir", type=Path, required=True)
    parser.add_argument("--count-oof-dir", type=Path, required=True)
    parser.add_argument("--batter-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v345-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.base_oof_dir, args.command_oof_dir, args.pitchmix_oof_dir, args.count_oof_dir, args.batter_oof_dir, args.exact_axes, args.v244_axes, args.v335_axes, args.v345_axes, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

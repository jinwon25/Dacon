"""Rebase the frozen joint-workload H1 direction above the v335 champion.

The route and dose library is screened on full-2022 and late-2023 only.  The
selected pair is then evaluated once on full-2024.  The historical strict H1
screen is reproduced from the three frozen seeds before any v335 comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, affine
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v328_baseball_archetype_consensus_moe import axis_metrics


PROTOCOL = "V336_JOINT_WORKLOAD_H1_REBASE_V335_V1"
TARGET = "control_success"
ORIGINS = ("full_2022", "late_2023", "full_2024")
SOURCE_ORIGINS = ("full_2022", "late_2023")
YEARS = (2022, 2023, 2024)
SEEDS = (42, 43, 44)
PROMOTION_SCOPES = ("R_CORE_ALL", "R_ALL")
DIAGNOSTIC_SCOPES = ("R_CORE_RUNNERS_OR_HIGH_LI",)
DOSES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
HISTORICAL_STRICT_DOSE = 0.10


def route_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return fixed row-local route masks without target-derived cut points."""

    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    core = regular & ~anchor
    runners = pd.to_numeric(
        frame["num_runners_on"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64) > 0.0
    high_li = pd.to_numeric(
        frame["li"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64) >= 1.5
    return {
        "R_CORE_ALL": core,
        "R_ALL": regular,
        "R_CORE_RUNNERS_OR_HIGH_LI": core & (runners | high_li),
        "R_ANCHOR": anchor,
        "F": ~regular,
    }


def apply_direction(
    parent: np.ndarray,
    direction: np.ndarray,
    active: np.ndarray,
    dose: float,
) -> np.ndarray:
    """Add the frozen H1 direction only on the requested route."""

    parent = np.asarray(parent, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if parent.shape != direction.shape or parent.shape != active.shape:
        raise ValueError("parent, direction and route mask must align")
    candidate = parent.copy()
    candidate[active] = np.clip(
        parent[active] + float(dose) * direction[active], 0.001, 0.999
    )
    return candidate


def full_row_rms(parent: np.ndarray, candidate: np.ndarray) -> float:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        parent, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def source_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["gain"] > 0.0
        and metrics["positive_month_fraction"] >= 2.0 / 3.0
        and metrics["worst_month_gain"] > -5.0
    )


def select_source_candidate(
    screen: dict[str, dict[str, dict[str, Any]]]
) -> tuple[str, str]:
    """Select by source maximin; never accepts a locked-origin statistic."""

    ranked: list[tuple[tuple[float, float, float, int], str, str]] = []
    for scope in PROMOTION_SCOPES:
        for dose_key, origin_metrics in screen[scope].items():
            if set(origin_metrics) != set(SOURCE_ORIGINS):
                raise ValueError("source screen contains a non-source origin")
            if not all(source_gate(origin_metrics[name]) for name in SOURCE_ORIGINS):
                continue
            gains = [float(origin_metrics[name]["gain"]) for name in SOURCE_ORIGINS]
            dose = float(dose_key)
            scope_preference = int(scope == "R_CORE_ALL")
            key = (min(gains), float(np.mean(gains)), -dose, scope_preference)
            ranked.append((key, scope, dose_key))
    if not ranked:
        raise RuntimeError("no joint-workload source candidate passed")
    _rank, scope, dose_key = max(ranked, key=lambda item: item[0])
    return scope, dose_key


def _load_h1_directions(
    baseline_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    correction: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    baseline_h1: dict[int, np.ndarray] = {}
    joint_h1: dict[int, np.ndarray] = {}
    direction: dict[int, np.ndarray] = {}
    for year in YEARS:
        baseline_raw = []
        joint_raw = []
        for seed in SEEDS:
            baseline_raw.append(
                np.load(
                    baseline_dir / f"h1_year{year}_seed{seed}.npy",
                    allow_pickle=False,
                ).astype(np.float64)
            )
            joint_dir = seed42_dir if seed == 42 else multiseed_dir
            joint_raw.append(
                np.load(
                    joint_dir / f"joint_h1_year{year}_seed{seed}.npy",
                    allow_pickle=False,
                ).astype(np.float64)
            )
        baseline_h1[year] = affine(np.mean(baseline_raw, axis=0) + correction[year])
        joint_h1[year] = affine(np.mean(joint_raw, axis=0) + correction[year])
        direction[year] = joint_h1[year] - baseline_h1[year]
    return baseline_h1, joint_h1, direction


def _add_batter_id(
    train_csv: Path,
    context: pd.DataFrame,
    frames: dict[int, pd.DataFrame],
) -> None:
    identity = pd.read_csv(
        train_csv, usecols=["season", "batter_id"], low_memory=False
    )
    season = identity["season"].to_numpy(np.int16)
    if not np.array_equal(season, context["season"].to_numpy(np.int16)):
        raise ValueError("batter identity rows are not aligned")
    batter = identity["batter_id"].to_numpy()
    for year in YEARS:
        frames[year]["batter_id"] = batter[season == year]


def run(
    train_csv: Path,
    v335_axes: Path,
    v318_axes: Path,
    baseline_checkpoint_dir: Path,
    seed42_dir: Path,
    multiseed_dir: Path,
    v216_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context, full_frames, correction = _load_year_context(train_csv)
    _add_batter_id(train_csv, context, full_frames)
    late23 = full_frames[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": full_frames[2022],
        "late_2023": full_frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": full_frames[2024],
    }

    with np.load(v335_axes, allow_pickle=False) as saved:
        parents = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
        v335_increment = {
            origin: saved[f"candidate_{origin}"].astype(np.float64)
            - saved[f"parent_{origin}"].astype(np.float64)
            for origin in ORIGINS
        }
    with np.load(v318_axes, allow_pickle=False) as saved:
        v320_increment_2024 = saved["candidate_full_2024"].astype(np.float64) - saved[
            "parent_full_2024"
        ].astype(np.float64)

    baseline_h1, joint_h1, raw_direction = _load_h1_directions(
        baseline_checkpoint_dir, seed42_dir, multiseed_dir, correction
    )
    directions = {
        "full_2022": raw_direction[2022],
        "late_2023": raw_direction[2023][late23],
        "full_2024": raw_direction[2024],
    }

    historical = json.loads(v216_summary.read_text(encoding="utf-8"))
    strict_reproduction: dict[str, Any] = {}
    for year in YEARS:
        active = np.ones(len(full_frames[year]), dtype=bool)
        candidate = apply_direction(
            baseline_h1[year], raw_direction[year], active, HISTORICAL_STRICT_DOSE
        )
        reproduced = axis_metrics(
            full_frames[year], baseline_h1[year], candidate, active
        )
        expected = float(historical["strict_ensemble"][str(year)]["gain"])
        error = float(reproduced["gain"] - expected)
        if abs(error) > 1e-9:
            raise ValueError(f"v216 strict reproduction failed for {year}: {error}")
        strict_reproduction[str(year)] = {
            "gain": reproduced["gain"],
            "positive_month_fraction": reproduced["positive_month_fraction"],
            "worst_month_gain": reproduced["worst_month_gain"],
            "historical_gain_error": error,
        }

    masks = {origin: route_masks(frames[origin]) for origin in ORIGINS}
    screen: dict[str, dict[str, dict[str, Any]]] = {}
    for scope in (*PROMOTION_SCOPES, *DIAGNOSTIC_SCOPES):
        screen[scope] = {}
        for dose in DOSES:
            dose_key = f"{dose:.3f}"
            screen[scope][dose_key] = {}
            for origin in SOURCE_ORIGINS:
                candidate = apply_direction(
                    parents[origin], directions[origin], masks[origin][scope], dose
                )
                result = axis_metrics(
                    frames[origin], parents[origin], candidate, masks[origin][scope]
                )
                result["full_row_rms_shift"] = full_row_rms(
                    parents[origin], candidate
                )
                screen[scope][dose_key][origin] = result

    source_record = {
        "protocol": PROTOCOL,
        "status": "source_screen_only",
        "source_selection_contract": {
            "origins": list(SOURCE_ORIGINS),
            "promotion_scopes": list(PROMOTION_SCOPES),
            "diagnostic_scopes": list(DIAGNOSTIC_SCOPES),
            "doses": list(DOSES),
            "ranking": "maximize minimum source gain, then mean gain, then smaller dose",
            "locked_origin_used_for_selection": False,
        },
        "historical_strict_reproduction": strict_reproduction,
        "source_screen": screen,
    }
    (output_dir / "source_screen.json").write_text(
        json.dumps(source_record, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    try:
        selected_scope, selected_dose_key = select_source_candidate(screen)
    except RuntimeError as error:
        summary = {
            **source_record,
            "status": "source_reject",
            "reason": str(error),
            "locked_origin_opened": False,
            "restrictions": {
                "official_train_only": True,
                "frozen_three_seed_joint_h1": True,
                "source_only_route_and_dose_selection": True,
                "full_2024_opened_only_for_selected_pair": True,
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
    selected_dose = float(selected_dose_key)
    selected_candidates: dict[str, np.ndarray] = {}
    selected_metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        selected_candidates[origin] = apply_direction(
            parents[origin], directions[origin], masks[origin][selected_scope], selected_dose
        )
        result = axis_metrics(
            frames[origin], parents[origin], selected_candidates[origin],
            masks[origin][selected_scope],
        )
        result["full_row_rms_shift"] = full_row_rms(
            parents[origin], selected_candidates[origin]
        )
        selected_metrics[origin] = result

    active24 = masks["full_2024"][selected_scope]
    axes24 = {
        "target": frames["full_2024"][TARGET].to_numpy(np.float64),
        "game_month": frames["full_2024"]["game_month"].to_numpy(np.int16),
        "pitcher_id": frames["full_2024"]["pitcher_id"].to_numpy(),
        "batter_id": frames["full_2024"]["batter_id"].to_numpy(),
        "exact_mask": np.ones(len(frames["full_2024"]), dtype=bool),
    }
    robustness = _robustness(
        axes24,
        parents["full_2024"],
        selected_candidates["full_2024"],
        active24,
        [parents["full_2024"], selected_candidates["full_2024"]],
    )

    joint_increment24 = (
        selected_candidates["full_2024"] - parents["full_2024"]
    )
    support = {
        "joint_rows": int(np.count_nonzero(joint_increment24)),
        "overlap_with_v320_f_increment": int(
            np.count_nonzero(joint_increment24) and np.count_nonzero(
                (np.abs(joint_increment24) > 1e-15)
                & (np.abs(v320_increment_2024) > 1e-15)
            )
        ),
        "overlap_with_v335_anchor_increment": int(
            np.count_nonzero(
                (np.abs(joint_increment24) > 1e-15)
                & (np.abs(v335_increment["full_2024"]) > 1e-15)
            )
        ),
    }
    locked = selected_metrics["full_2024"]
    locked_gate = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0
    )
    rms_target_fraction = float(locked["full_row_rms_shift"] / 0.004551)

    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{origin}": parents[origin] for origin in ORIGINS},
        **{
            f"candidate_{origin}": selected_candidates[origin]
            for origin in ORIGINS
        },
        **{f"direction_{origin}": directions[origin] for origin in ORIGINS},
        **{
            f"active_{origin}": masks[origin][selected_scope]
            for origin in ORIGINS
        },
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "candidate" if locked_gate else "locked_reject",
        "parent": "v335-compatible forward axes",
        "historical_strict_reproduction": strict_reproduction,
        "source_selection_contract": {
            "origins": list(SOURCE_ORIGINS),
            "promotion_scopes": list(PROMOTION_SCOPES),
            "diagnostic_scopes": list(DIAGNOSTIC_SCOPES),
            "doses": list(DOSES),
            "ranking": "maximize minimum source gain, then mean gain, then smaller dose",
            "locked_origin_used_for_selection": False,
        },
        "source_screen": screen,
        "selected": {"scope": selected_scope, "dose": selected_dose},
        "metrics": selected_metrics,
        "locked_robustness": robustness,
        "support_orthogonality": support,
        "rms_goal_context": {
            "locked_full_row_rms": locked["full_row_rms_shift"],
            "single_candidate_rms_for_public_1190": 0.004551,
            "fraction_of_single_candidate_target": rms_target_fraction,
        },
        "locked_gate": {
            "gain_positive": bool(locked["gain"] > 0.0),
            "positive_month_fraction_ge_0_625": bool(
                locked["positive_month_fraction"] >= 0.625
            ),
            "worst_month_above_minus_5": bool(locked["worst_month_gain"] > -5.0),
            "passed": locked_gate,
            "crossed_bootstrap_and_reality_check_are_diagnostic_only": True,
        },
        "restrictions": {
            "official_train_only": True,
            "frozen_three_seed_joint_h1": True,
            "source_only_route_and_dose_selection": True,
            "full_2024_opened_only_for_selected_pair": True,
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
    parser.add_argument("--v335-axes", type=Path, required=True)
    parser.add_argument("--v318-axes", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed42-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--v216-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.v335_axes,
        args.v318_axes,
        args.baseline_checkpoint_dir,
        args.seed42_dir,
        args.multiseed_dir,
        args.v216_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

"""Audit an independent fallback specialist on v244-protected R_ANCHOR rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import align_regular_prediction
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.archive.v248_fixed_route_dual_tree_fallback import AXES, SOURCE_AXES
from src.core.contract import _load_contract_axis


PROTOCOL = "V251_ANCHOR_DUAL_TREE_SPECIALIST_V1"
FAMILIES = ("xgb", "lightgbm", "dual_tree_mean")
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30)


def anchor_mask(frame) -> np.ndarray:
    return (
        frame["game_type"].astype(str).eq("R").to_numpy()
        & (
            frame["pitcher_team_id"].eq(13).to_numpy()
            | frame["batter_team_id"].eq(13).to_numpy()
        )
    )


def apply_specialist(
    parent: np.ndarray,
    specialist: np.ndarray,
    active: np.ndarray,
    weight: float,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    specialist = np.asarray(specialist, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if not np.isfinite(specialist[active]).all():
        raise ValueError("missing specialist predictions on R_ANCHOR")
    output = parent.copy()
    output[active] = np.clip(
        parent[active] + float(weight) * (specialist[active] - parent[active]),
        0.001,
        0.999,
    )
    return output


def select_candidate(results: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, float] | None:
    eligible: list[tuple[float, float, float, str]] = []
    for family in FAMILIES:
        for weight in WEIGHTS:
            key = f"{weight:g}"
            items = [results[family][key][axis] for axis in SOURCE_AXES]
            gains = [item["gain"] for item in items]
            if all(gain > 0.0 for gain in gains) and all(
                item["positive_month_fraction"] >= 2.0 / 3.0 for item in items
            ):
                eligible.append((min(gains), float(np.mean(gains)), -weight, family))
    if not eligible:
        return None
    _minimum, _mean, negative_weight, family = max(eligible)
    return family, -negative_weight


def restrictions() -> dict[str, bool]:
    return {
        "v244_all_existing_rows_frozen": True,
        "new_route_is_disjoint_r_anchor_only": True,
        "three_predeclared_model_families": True,
        "family_and_weight_selected_on_full_2022_and_late_2023_only": True,
        "source_month_majority_required": True,
        "full_2024_locked_from_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def _load_axes(contract_dir: Path, bridge_oof: Path) -> dict[str, dict[str, np.ndarray]]:
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    return {
        name: {**axis, "exact_mask": np.ones(len(axis["target"]), dtype=bool)}
        for name, axis in axes.items()
    }


def run(
    train_csv: Path,
    xgb_oof_dir: Path,
    lightgbm_oof_dir: Path,
    exact_axes: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, _correction = _load_year_context(train_csv)
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = _load_axes(contract_dir, bridge_oof)
    with np.load(v244_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(np.float64)
            for axis in AXES
        }
    xgb_full = {
        year: align_regular_prediction(
            frames[year], xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    lgbm_full = {
        year: align_regular_prediction(
            frames[year], lightgbm_oof_dir / f"runtime_faithful_lgbm_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    xgb = {
        "full_2022": xgb_full[2022],
        "late_2023": xgb_full[2023][late23],
        "full_2024": xgb_full[2024],
    }
    lightgbm = {
        "full_2022": lgbm_full[2022],
        "late_2023": lgbm_full[2023][late23],
        "full_2024": lgbm_full[2024],
    }
    specialists = {
        "xgb": xgb,
        "lightgbm": lightgbm,
        "dual_tree_mean": {
            axis: 0.5 * (xgb[axis] + lightgbm[axis]) for axis in AXES
        },
    }
    active = {axis: anchor_mask(axis_frames[axis]) for axis in AXES}

    results: dict[str, dict[str, dict[str, Any]]] = {family: {} for family in FAMILIES}
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    for family in FAMILIES:
        for weight in WEIGHTS:
            key = f"{weight:g}"
            results[family][key] = {}
            for axis in AXES:
                candidate = apply_specialist(
                    parents[axis], specialists[family][axis], active[axis], weight
                )
                predictions[(family, key, axis)] = candidate
                results[family][key][axis] = paired_metrics(
                    axes[axis], parents[axis], candidate, active[axis]
                )

    selected = select_candidate(results)
    locked_pass = False
    robust_pass = False
    robustness = None
    if selected is not None:
        family, weight = selected
        key = f"{weight:g}"
        locked_pass = results[family][key]["full_2024"]["gain"] > 0.0
        candidate_family = [
            predictions[(candidate_family, f"{candidate_weight:g}", "full_2024")]
            for candidate_family in FAMILIES
            for candidate_weight in WEIGHTS
        ] + [parents["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"],
            parents["full_2024"],
            predictions[(family, key, "full_2024")],
            active["full_2024"],
            candidate_family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] < 0.05
        )
        np.savez_compressed(
            output_dir / "selected_candidate_axes.npz",
            **{f"parent_{axis}": parents[axis] for axis in AXES},
            **{
                f"candidate_{axis}": predictions[(family, key, axis)]
                for axis in AXES
            },
            **{f"active_{axis}": active[axis] for axis in AXES},
        )

    summary = {
        "protocol": PROTOCOL,
        "status": (
            "robust_candidate"
            if selected is not None and locked_pass and robust_pass
            else "locked_or_robust_reject"
            if selected is not None
            else "source_reject"
        ),
        "families": list(FAMILIES),
        "weights": list(WEIGHTS),
        "selected": None if selected is None else {
            "family": selected[0], "weight": selected[1]
        },
        "active_rows": {axis: int(active[axis].sum()) for axis in AXES},
        "results": results,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(selected is not None and locked_pass and robust_pass),
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
    parser.add_argument("--xgb-oof-dir", type=Path, required=True)
    parser.add_argument("--lightgbm-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.xgb_oof_dir,
        args.lightgbm_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    selected = result["selected"]
    print(json.dumps({
        "status": result["status"],
        "selected": selected,
        "active_rows": result["active_rows"],
        "selected_results": None if selected is None else result["results"]
        [selected["family"]][f"{selected['weight']:g}"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

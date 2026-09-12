"""Audit pseudo-deployment XGB on regular-season routes untouched by v244.

The family contains three structural scopes at one inherited low dose (15%):
the untouched R_CORE complement, the team-13 R_ANCHOR domain, and their union.
All are limited to the inherited April--September window.  Selection uses only
2022 and late-2023; full-2024 is evaluated after the source rule is fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import align_regular_prediction
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics, route_masks
from src.archive.v248_fixed_route_dual_tree_fallback import ROUTE_WEIGHTS, compose
from src.archive.v253_fixed_route_command_dispersion_audit import AXES, SOURCE_AXES, _load_axes


PROTOCOL = "V272_PSEUDO_DEPLOYMENT_NEW_ROUTE_AUDIT_V1"
ROUTE_DOSE = 0.15
FIRST_ACTIVE_MONTH = 4
LAST_ACTIVE_MONTH = 9
SCOPES = ("core_complement", "anchor", "core_plus_anchor")


def new_route_masks(frame, v244_active: np.ndarray, pseudo: np.ndarray) -> dict[str, np.ndarray]:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    core = regular & ~anchor
    calendar = frame["game_month"].between(
        FIRST_ACTIVE_MONTH, LAST_ACTIVE_MONTH
    ).to_numpy()
    finite = np.isfinite(np.asarray(pseudo, dtype=np.float64))
    core_complement = core & ~np.asarray(v244_active, dtype=bool) & calendar & finite
    anchor = anchor & calendar & finite
    return {
        "core_complement": core_complement,
        "anchor": anchor,
        "core_plus_anchor": core_complement | anchor,
    }


def apply_new_route(
    baseline: np.ndarray,
    parent: np.ndarray,
    pseudo: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    candidate = np.asarray(baseline, dtype=np.float64).copy()
    parent = np.asarray(parent, dtype=np.float64)
    pseudo = np.asarray(pseudo, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    candidate[selected] = np.clip(
        parent[selected] + ROUTE_DOSE * (pseudo[selected] - parent[selected]),
        0.001,
        0.999,
    )
    return candidate


def select_scope(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, int, str]] = []
    for scope in SCOPES:
        source = [results[scope][axis] for axis in SOURCE_AXES]
        if not all(item["gain"] > 0.0 for item in source):
            continue
        if not all(item["positive_month_fraction"] >= (2.0 / 3.0) for item in source):
            continue
        active_rows = min(int(item["active_rows"]) for item in source)
        if active_rows < 5000:
            continue
        gains = [float(item["gain"]) for item in source]
        eligible.append((min(gains), float(np.mean(gains)), active_rows, scope))
    return None if not eligible else max(eligible)[3]


def run(
    train_csv: Path,
    base_xgb_oof_dir: Path,
    pseudo_oof_dir: Path,
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
    with np.load(exact_axes, allow_pickle=False) as saved:
        parents = {axis: saved[f"jy_parent_{axis}"].astype(np.float64) for axis in AXES}
    with np.load(v244_axes, allow_pickle=False) as saved:
        expected_v244 = {
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(np.float64)
            for axis in AXES
        }

    year_by_axis = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    parity: dict[str, float] = {}
    baselines: dict[str, np.ndarray] = {}
    masks: dict[tuple[str, str], np.ndarray] = {}
    candidates: dict[tuple[str, str], np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {scope: {} for scope in SCOPES}
    for axis in AXES:
        year = year_by_axis[axis]
        base = align_regular_prediction(
            frames[year],
            base_xgb_oof_dir / f"hyunku_runtime_faithful_xgb_{year}.npy",
        )
        pseudo = align_regular_prediction(
            frames[year], pseudo_oof_dir / f"pseudo_deployment_xgb_{year}.npy"
        )
        if axis == "late_2023":
            base = base[late23]
            pseudo = pseudo[late23]
        routes = route_masks(parents[axis], base, axis_frames[axis])
        baselines[axis], active = compose(parents[axis], base, routes)
        parity[axis] = float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        scopes = new_route_masks(axis_frames[axis], active, pseudo)
        for scope in SCOPES:
            masks[(scope, axis)] = scopes[scope]
            candidates[(scope, axis)] = apply_new_route(
                baselines[axis], parents[axis], pseudo, scopes[scope]
            )
            results[scope][axis] = paired_metrics(
                axes[axis],
                baselines[axis],
                candidates[(scope, axis)],
                scopes[scope],
            )

    selected_scope = select_scope(results)
    source_pass = selected_scope is not None
    locked_pass = bool(
        selected_scope is not None and results[selected_scope]["full_2024"]["gain"] > 0.0
    )
    robustness = None
    robust_pass = False
    if selected_scope is not None:
        robustness = _robustness(
            axes["full_2024"],
            baselines["full_2024"],
            candidates[(selected_scope, "full_2024")],
            masks[(selected_scope, "full_2024")],
            [candidates[(scope, "full_2024")] for scope in SCOPES]
            + [baselines["full_2024"].copy()],
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] < 0.05
        )
        np.savez_compressed(
            output_dir / "selected_candidate_axes.npz",
            **{f"parent_{axis}": baselines[axis] for axis in AXES},
            **{
                f"candidate_{axis}": candidates[(selected_scope, axis)] for axis in AXES
            },
            **{f"active_{axis}": masks[(selected_scope, axis)] for axis in AXES},
        )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "local_gate_pass"
            if source_pass and locked_pass and robust_pass
            else "source_reject"
            if not source_pass
            else "locked_or_robust_reject"
        ),
        "route_dose": ROUTE_DOSE,
        "calendar_window": [FIRST_ACTIVE_MONTH, LAST_ACTIVE_MONTH],
        "scopes": list(SCOPES),
        "selection_rule": "source positivity and monthly majority, then maximin gain",
        "selected_scope": selected_scope,
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "source_gate_passed": source_pass,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(source_pass and locked_pass and robust_pass),
        "restrictions": {
            "one_inherited_low_dose_only": True,
            "three_disjoint_structural_scope_choices": True,
            "calendar_inherited_from_v70_v261": True,
            "v244_existing_routes_preserved": True,
            "source_axes_select_scope_before_locked_metric": True,
            "full_2024_pseudo_predictions_previously_generated": True,
            "exploratory_not_confirmatory": True,
            "test_csv_read": False,
            "public_score_used_for_selection": False,
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
    parser.add_argument("--base-xgb-oof-dir", type=Path, required=True)
    parser.add_argument("--pseudo-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.base_xgb_oof_dir,
        args.pseudo_oof_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

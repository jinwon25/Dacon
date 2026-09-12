"""Open one soft-source pressure-profile candidate on locked 2024.

The relaxation is fixed without looking at v187's unopened 2024 axis.  It
requires a source minimum above -0.5 BSS, positive mean source gain, worst
month above -1.0, and eta at most 0.1.  This is explicitly weaker evidence
than the standard promotion gate and is reported as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_parent_parents
from src.archive.v178_row_region_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
)
from src.archive.v187_pitcher_pressure_profile_residual import (
    V178_SCALE,
    apply_correction,
    attach_profile,
    fit_ridge,
    predict_ridge,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V188_PRESSURE_PROFILE_SOFT_TRANSITION_V1"


def soft_source_candidates(ranking: pd.DataFrame) -> pd.DataFrame:
    return ranking.loc[
        ranking["minimum_gain"].ge(-0.5)
        & ranking["mean_gain"].gt(0.0)
        & ranking["worst_month_gain"].gt(-1.0)
        & ranking["eta"].le(0.1)
    ].sort_values(
        ["minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=False,
        kind="stable",
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
    v187_output: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ranking = pd.read_csv(v187_output / "source_ranking.csv")
    passing = soft_source_candidates(ranking)
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "soft_source_reject",
            "restrictions": restrictions(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    selected = passing.iloc[0]
    profile_alpha = float(selected["profile_alpha"])
    ridge_alpha = float(selected["ridge_alpha"])
    eta = float(selected["eta"])

    full_train = pd.read_csv(train_csv, low_memory=False)
    _context, raw_frames, post4 = _load_year_context(train_csv)
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": full_train.loc[full_train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": full_train.loc[
            full_train["season"].eq(2023) & full_train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": full_train.loc[full_train["season"].eq(2024)].reset_index(drop=True),
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, post4, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    frozen_weights = load_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in ("full_2022", "late_2023", "full_2024")
        }
    directions = {
        name: load_direction(name, v158_base[name], frozen_weights, library_root)
        for name in v158_base
    }
    base = {
        name: apply_direction(parents[name], directions[name], V178_SCALE)
        for name in parents
    }
    features = {
        "full_2022": attach_profile(
            frames["full_2022"], full_train.loc[full_train["season"].lt(2022)], profile_alpha
        ),
        "late_2023": attach_profile(
            frames["late_2023"], full_train.loc[full_train["season"].lt(2023)], profile_alpha
        ),
        "full_2024": attach_profile(
            frames["full_2024"], full_train.loc[full_train["season"].lt(2024)], profile_alpha
        ),
    }
    exact22 = np.asarray(axes["full_2022"]["exact_mask"], dtype=bool)
    exact23 = np.asarray(axes["late_2023"]["exact_mask"], dtype=bool)
    exact24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
    residual22 = np.asarray(axes["full_2022"]["target"])[exact22] - base["full_2022"][exact22]
    residual23 = np.asarray(axes["late_2023"]["target"])[exact23] - base["late_2023"][exact23]
    fit_x = np.vstack(
        [
            features["full_2022"].to_numpy(np.float64)[exact22],
            features["late_2023"].to_numpy(np.float64)[exact23],
        ]
    )
    fit_y = np.concatenate([residual22, residual23])
    fit_weight = np.concatenate(
        [np.full(len(residual22), 0.55), np.ones(len(residual23))]
    )
    spec = fit_ridge(fit_x, fit_y, ridge_alpha, fit_weight)
    correction24 = predict_ridge(
        spec, features["full_2024"].to_numpy(np.float64)
    )
    candidate24 = apply_correction(base["full_2024"], correction24, exact24, eta)
    locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
    incremental = metrics(axes["full_2024"], base["full_2024"], candidate24)
    robust = _robustness(
        axes["full_2024"], parents["full_2024"], candidate24,
        exact24, [candidate24, base["full_2024"]],
    )
    point_pass = bool(
        locked["gain"] > 0.0 and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -5.0 and locked["minimum_domain_gain"] >= 0.0
    )
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] <= 0.10
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "selected_model.npz",
        mean=spec.mean, scale=spec.scale, coefficient=spec.coefficient,
        profile_alpha=profile_alpha, ridge_alpha=ridge_alpha, eta=eta,
    )
    np.savez_compressed(
        output_dir / "selected_axis.npz",
        parent=parents["full_2024"], base=base["full_2024"],
        candidate=candidate24, correction=correction24, active=exact24,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "soft_robust_pass" if point_pass and robust_pass else (
            "soft_point_pass_robust_reject" if point_pass else "locked_reject"
        ),
        "selected_on_soft_sources_only": selected.to_dict(),
        "soft_source_gate": {
            "minimum_gain_min": -0.5,
            "mean_gain_strict_min": 0.0,
            "worst_month_gain_strict_min": -1.0,
            "eta_max": 0.1,
        },
        "locked_2024": locked,
        "incremental_over_v178": incremental,
        "robustness": robust,
        "point_gate_passed": point_pass,
        "robust_gate_passed": robust_pass,
        "eligible_for_packaging": bool(point_pass and robust_pass),
        "evidence_tier": "soft_source",
        "parity": parity,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_labels_only": True,
        "soft_gate_frozen_before_locked_2024": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_prediction_used": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
        "locked_2024_development_contaminated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v158-path", type=Path, required=True)
    parser.add_argument("--v165-summary", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--v187-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.v187_output, args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

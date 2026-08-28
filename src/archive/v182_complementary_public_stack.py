"""Combine frozen v178 with the complementary public psych-regime OOF axis.

The v178 direction and scale remain frozen.  A small coefficient for the
independent public direction is selected using full-2022 (unchanged) and
late-2023 only, before the 2024 contract is opened.  This remains a research
screen: no public artifact is eligible for packaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v178_jy_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
    source_gate,
)
from src.archive.v181_public_independent_oof_screen import load_public_family
from src.core.contract import _load_contract_axis


PROTOCOL = "V182_COMPLEMENTARY_PUBLIC_STACK_V1"
V178_SCALE = 0.25
PUBLIC_FAMILY = "psych_regime_film"
PUBLIC_WEIGHTS = (0.0, 0.005, 0.01, 0.015, 0.025, 0.05)


def additive_stack(
    parent: np.ndarray,
    v178_direction: np.ndarray,
    independent: np.ndarray,
    active: np.ndarray,
    public_weight: float,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    output = parent + V178_SCALE * np.asarray(v178_direction, dtype=np.float64)
    public_delta = np.asarray(independent, dtype=np.float64) - parent
    active = np.asarray(active, dtype=bool)
    output[active] += float(public_weight) * public_delta[active]
    return np.clip(output, 0.001, 0.999)


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
    public_oof_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, correction = _load_year_context(train_csv)
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes,
        raw_frames,
        correction,
        v104_path,
        h1_path,
        c3_path,
        v160_path,
        bridge_oof,
    )
    weights = load_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in ("full_2022", "late_2023", "full_2024")
        }
    v178_direction = {
        name: load_direction(name, v158_base[name], weights, library_root)
        for name in ("full_2022", "late_2023", "full_2024")
    }
    public = load_public_family(
        public_oof_root,
        PUBLIC_FAMILY,
        late23,
        raw_frames[2023]["control_success"].to_numpy(np.int8),
        raw_frames[2024]["control_success"].to_numpy(np.int8),
        raw_frames[2023]["pitcher_id"].to_numpy(np.int64),
        raw_frames[2024]["pitcher_id"].to_numpy(np.int64),
    )

    exact22 = np.asarray(axes["full_2022"]["exact_mask"], dtype=bool)
    exact23 = np.asarray(axes["late_2023"]["exact_mask"], dtype=bool)
    v178_22 = apply_direction(
        parents["full_2022"], v178_direction["full_2022"], V178_SCALE
    )
    result22 = metrics(axes["full_2022"], parents["full_2022"], v178_22)
    rows: list[dict[str, Any]] = []
    source_details: dict[str, dict[str, Any]] = {}
    for public_weight in PUBLIC_WEIGHTS:
        candidate23 = additive_stack(
            parents["late_2023"],
            v178_direction["late_2023"],
            public["late_2023"],
            exact23,
            public_weight,
        )
        result23 = metrics(axes["late_2023"], parents["late_2023"], candidate23)
        passed = source_gate(result22) and source_gate(result23)
        key = str(public_weight)
        source_details[key] = {"full_2022": result22, "late_2023": result23}
        rows.append(
            {
                "public_weight": public_weight,
                "full_2022_gain": result22["gain"],
                "late_2023_gain": result23["gain"],
                "minimum_gain": min(result22["gain"], result23["gain"]),
                "minimum_positive_month_fraction": min(
                    result22["positive_month_fraction"],
                    result23["positive_month_fraction"],
                ),
                "worst_month_gain": min(
                    result22["worst_month_gain"], result23["worst_month_gain"]
                ),
                "source_gate_passed": passed,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain", "public_weight"],
        ascending=[False, False, False, True],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_weight_screen.csv", index=False, encoding="utf-8-sig")
    passing = ranking.loc[ranking["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "source": source_details,
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_weight = float(passing.iloc[0]["public_weight"])
        active24 = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        candidate24 = additive_stack(
            parents["full_2024"],
            v178_direction["full_2024"],
            public["full_2024"],
            active24,
            selected_weight,
        )
        locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
        robust = _robustness(
            axes["full_2024"],
            parents["full_2024"],
            candidate24,
            active24,
            [candidate24, parents["full_2024"]],
        )
        point_pass = bool(
            locked["gain"] > 0.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -5.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] <= 0.10
        )
        np.savez_compressed(
            output_dir / "selected_axis.npz",
            parent=parents["full_2024"],
            candidate=candidate24,
            v178_direction=v178_direction["full_2024"],
            public_prediction=public["full_2024"],
            active=active24,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "research_signal_pass" if point_pass and robust_pass else "locked_reject",
            "frozen_v178_scale": V178_SCALE,
            "public_family": PUBLIC_FAMILY,
            "selected_public_weight_on_sources_only": selected_weight,
            "source": source_details[str(selected_weight)],
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_local_reproduction": bool(point_pass and robust_pass),
            "eligible_for_packaging": False,
            "packaging_blocker": "public model must be independently reproduced from official train",
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
        "frozen_v178_direction_and_scale": True,
        "official_train_labels_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "public_artifact_in_submission": False,
        "row_local_candidate": True,
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
    parser.add_argument("--public-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv,
        args.contract_dir,
        args.v104_path,
        args.h1_path,
        args.c3_path,
        args.v160_path,
        args.bridge_oof,
        args.v158_path,
        args.v165_summary,
        args.library_root,
        args.public_oof_root,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

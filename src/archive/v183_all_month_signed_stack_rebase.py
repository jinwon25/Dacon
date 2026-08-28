"""Rebase the source-only v165 ``all_months`` signed solution above JY.

Unlike v178, this audit deliberately chooses the v165 solution that omitted no
source-month constraint.  The choice is semantic and frozen before 2024 is
opened.  Only a conservative attenuation is selected on full-2022 and
late-2023.
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
    source_gate,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V183_ALL_MONTH_SIGNED_STACK_REBASE_V1"
SOURCE_AXES = ("full_2022", "late_2023")
SCALES = (0.25, 0.5, 1.0)
VARIANT = "all_months"


def load_all_month_weights(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != "V165_SIGNED_GROUP_CONSTRAINED_STACK_V1":
        raise ValueError("expected the frozen v165 summary")
    matches = [row for row in summary["source_trials"] if row["variant"] == VARIANT]
    if len(matches) != 1:
        raise ValueError("expected exactly one all_months source solution")
    row = matches[0]
    if row.get("omitted_month_constraint") is not None or not row.get("source_gate_passed"):
        raise ValueError("all_months source solution is not valid")
    raw = row["net_weights"]
    weights = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if sum(abs(float(value)) for value in weights.values()) > 0.500001:
        raise ValueError("invalid signed L1 bound")
    return {str(name): float(value) for name, value in weights.items()}


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
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, correction = _load_year_context(train_csv)
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
    weights = load_all_month_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in (*SOURCE_AXES, "full_2024")
        }
    directions = {
        name: load_direction(name, v158_base[name], weights, library_root)
        for name in (*SOURCE_AXES, "full_2024")
    }

    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        details[str(scale)] = {}
        for axis_name in SOURCE_AXES:
            candidate = apply_direction(parents[axis_name], directions[axis_name], scale)
            details[str(scale)][axis_name] = metrics(
                axes[axis_name], parents[axis_name], candidate
            )
        values = details[str(scale)]
        rows.append(
            {
                "scale": scale,
                "full_2022_gain": values["full_2022"]["gain"],
                "late_2023_gain": values["late_2023"]["gain"],
                "minimum_gain": min(item["gain"] for item in values.values()),
                "minimum_positive_month_fraction": min(
                    item["positive_month_fraction"] for item in values.values()
                ),
                "worst_month_gain": min(item["worst_month_gain"] for item in values.values()),
                "minimum_domain_gain": min(item["minimum_domain_gain"] for item in values.values()),
                "source_gate_passed": all(source_gate(item) for item in values.values()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain", "scale"],
        ascending=[False, False, False, True],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_scale_audit.csv", index=False, encoding="utf-8-sig")
    passing = ranking.loc[ranking["source_gate_passed"]]
    if passing.empty:
        summary = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "weights": weights,
            "source": details,
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_scale = float(passing.iloc[0]["scale"])
        candidate24 = apply_direction(
            parents["full_2024"], directions["full_2024"], selected_scale
        )
        locked = metrics(axes["full_2024"], parents["full_2024"], candidate24)
        exact = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        active = exact & np.not_equal(directions["full_2024"], 0.0)
        robust = _robustness(
            axes["full_2024"],
            parents["full_2024"],
            candidate24,
            active,
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
            direction=directions["full_2024"],
            active=active,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else "locked_reject",
            "frozen_variant": VARIANT,
            "weights": weights,
            "selected_scale": selected_scale,
            "source": details[str(selected_scale)],
            "locked_2024": locked,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
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
        "all_month_variant_frozen_before_locked_2024": True,
        "official_train_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
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
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

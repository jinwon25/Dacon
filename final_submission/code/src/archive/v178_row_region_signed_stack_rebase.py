"""Rebase the source-selected v165 signed direction above the exact row-region parent.

The v165 component weights were optimized on full-2022 and late-2023 before
the row-region package existed.  This audit does not refit those component weights.
It selects only a conservative global attenuation on the same two source axes,
then opens the development-contaminated 2024 contract once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import (
    _load_year_context,
    metrics,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import (
    add_context,
    exact_parent_parents,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V178_JY_SIGNED_STACK_REBASE_V1"
SOURCE_AXES = ("full_2022", "late_2023")
SCALES = (0.25, 0.5, 1.0)


def load_weights(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != "V165_SIGNED_GROUP_CONSTRAINED_STACK_V1":
        raise ValueError("expected the frozen v165 summary")
    raw = summary["selected_on_sources_only"]["net_weights"]
    weights = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not weights or sum(abs(float(value)) for value in weights.values()) > 0.500001:
        raise ValueError("invalid frozen signed weights")
    return {str(name): float(value) for name, value in weights.items()}


def load_direction(
    axis_name: str,
    base: np.ndarray,
    weights: dict[str, float],
    library_root: Path,
) -> np.ndarray:
    direction = np.zeros(len(base), dtype=np.float64)
    for family, weight in weights.items():
        path = library_root / family / "selected_axes.npz"
        with np.load(path, allow_pickle=False) as saved:
            candidate = saved[axis_name].astype(np.float64)
        if candidate.shape != np.asarray(base).shape:
            raise ValueError(f"component alignment mismatch: {family}/{axis_name}")
        direction += float(weight) * (candidate - base)
    return direction


def apply_direction(parent: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(
        np.asarray(parent, dtype=np.float64)
        + float(scale) * np.asarray(direction, dtype=np.float64),
        0.001,
        0.999,
    )


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= 0.75
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
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
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, correction = _load_year_context(train_csv)
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": add_context(raw_frames[2022]),
        "late_2023": add_context(raw_frames[2023].loc[late23].reset_index(drop=True)),
        "full_2024": add_context(raw_frames[2024]),
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_parent_parents(
        axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    weights = load_weights(v165_summary)
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
    source_details: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        source_details[str(scale)] = {}
        for axis_name in SOURCE_AXES:
            candidate = apply_direction(parents[axis_name], directions[axis_name], scale)
            result = metrics(axes[axis_name], parents[axis_name], candidate)
            source_details[str(scale)][axis_name] = result
        axis_results = source_details[str(scale)]
        rows.append(
            {
                "scale": scale,
                "full_2022_gain": axis_results["full_2022"]["gain"],
                "late_2023_gain": axis_results["late_2023"]["gain"],
                "minimum_gain": min(item["gain"] for item in axis_results.values()),
                "minimum_positive_month_fraction": min(
                    item["positive_month_fraction"] for item in axis_results.values()
                ),
                "worst_month_gain": min(
                    item["worst_month_gain"] for item in axis_results.values()
                ),
                "minimum_domain_gain": min(
                    item["minimum_domain_gain"] for item in axis_results.values()
                ),
                "source_gate_passed": all(source_gate(item) for item in axis_results.values()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"], ascending=False
    )
    passing = ranking.loc[ranking["source_gate_passed"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_scale_audit.csv", index=False, encoding="utf-8-sig")
    if passing.empty:
        result = {
            "protocol": PROTOCOL,
            "status": "source_reject",
            "weights": weights,
            "source": source_details,
            "parity": parity,
            "restrictions": restrictions(),
        }
    else:
        selected_scale = float(passing.iloc[0]["scale"])
        # The Reality Check must cover only candidates that survived the
        # source-only gate.  Including source-rejected scales would let the
        # locked 2024 outcome effectively promote a recipe that was already
        # disqualified, and made the reported observed-best index refer to a
        # different scale than the actually selected candidate.
        passing_scales = [float(value) for value in passing["scale"].tolist()]
        locked_family = [
            apply_direction(parents["full_2024"], directions["full_2024"], scale)
            for scale in passing_scales
        ]
        # The shared White-test helper requires at least two columns.  The
        # no-op parent is the natural null arm when only one non-zero scale
        # survives source selection.
        reality_check_scales: list[float | str] = list(passing_scales)
        if len(locked_family) == 1:
            locked_family.append(parents["full_2024"].copy())
            reality_check_scales.append("no_op_parent")
        locked_candidate = apply_direction(
            parents["full_2024"], directions["full_2024"], selected_scale
        )
        locked_metrics = metrics(
            axes["full_2024"], parents["full_2024"], locked_candidate
        )
        exact = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool)
        active = exact & np.not_equal(directions["full_2024"], 0.0)
        robust = _robustness(
            axes["full_2024"], parents["full_2024"], locked_candidate,
            active, locked_family,
        )
        point_pass = bool(
            locked_metrics["gain"] > 0.0
            and locked_metrics["positive_month_fraction"] >= 0.625
            and locked_metrics["worst_month_gain"] > -5.0
            and locked_metrics["minimum_domain_gain"] >= 0.0
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
            candidate=locked_candidate,
            direction=directions["full_2024"],
            active=active,
        )
        result = {
            "protocol": PROTOCOL,
            "status": "robust_pass" if point_pass and robust_pass else (
                "point_pass_robust_reject" if point_pass else "locked_reject"
            ),
            "weights": weights,
            "selected_scale": selected_scale,
            "reality_check_scales": reality_check_scales,
            "source": source_details[str(selected_scale)],
            "locked_2024": locked_metrics,
            "robustness": robust,
            "point_gate_passed": point_pass,
            "robust_gate_passed": robust_pass,
            "eligible_for_packaging": bool(point_pass and robust_pass),
            "eligible_for_relaxed_submission_review": bool(point_pass),
            "parity": parity,
            "restrictions": restrictions(),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def restrictions() -> dict[str, bool]:
    return {
        "frozen_v165_component_weights": True,
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
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

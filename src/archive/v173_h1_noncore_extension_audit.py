"""Audit the deployed exact H1 model outside the historical R_CORE route.

The original v131 ``ALL`` screen changed F/R_ANCHOR predictions, but its
shared metric helper hard-coded R_CORE as the only active domain.  This audit
closes that blind spot with paired competition-BSS metrics.  Candidate weights
are fixed before looking at 2024; only full-2022 and late-2023 choose a dose.
R_ANCHOR remains diagnostic because the full-2022 exact contract has no anchor
support, while F has two independent source axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import (
    _load_year_context,
    _locked_contract,
    _source_contracts,
)
from src.core.contract import _load_contract_axis
from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    one_way_cluster_bootstrap,
    paired_score_summary,
    white_reality_check,
)


PROTOCOL = "V173_H1_NONCORE_EXTENSION_AUDIT_V1"
SOURCE_AXES = ("full_2022", "late_2023")
WEIGHTS = (0.05, 0.10, 0.15)


def extend_h1(
    base: np.ndarray,
    component: np.ndarray,
    h1: np.ndarray,
    domain: np.ndarray,
    exact_mask: np.ndarray,
    route: tuple[str, ...],
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend absolute H1 into selected non-core domains, row by row."""
    output = np.asarray(base, dtype=np.float64).copy()
    active = np.asarray(exact_mask, dtype=bool) & np.isin(
        np.asarray(domain).astype(str), route
    )
    output[active] = np.clip(
        (1.0 - float(weight)) * np.asarray(component, dtype=np.float64)[active]
        + float(weight) * np.asarray(h1, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output, active


def paired_metrics(
    axis: dict[str, np.ndarray],
    base: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    exact = np.asarray(axis["exact_mask"], dtype=bool)
    target = np.asarray(axis["target"], dtype=np.float64)
    result = paired_score_summary(target[exact], candidate[exact], base[exact])
    active_summary = paired_score_summary(
        target[active], candidate[active], base[active]
    )
    month_rows = []
    months = np.asarray(axis["game_month"])
    for month in sorted(np.unique(months[active]).tolist()):
        mask = active & (months == month)
        item = paired_score_summary(target[mask], candidate[mask], base[mask])
        month_rows.append(
            {
                "month": int(month),
                "n_rows": int(mask.sum()),
                "gain": item["unclipped_bss_equivalent_gain"],
            }
        )
    gains = [row["gain"] for row in month_rows]
    return {
        "overall_gain": result["unclipped_bss_equivalent_gain"],
        "active_domain_gain": active_summary["unclipped_bss_equivalent_gain"],
        "active_rows": int(active.sum()),
        "active_fraction_exact": float(active.sum() / exact.sum()),
        "mean_abs_shift_active": float(
            np.mean(np.abs(np.asarray(candidate)[active] - np.asarray(base)[active]))
        ),
        "positive_month_fraction": float(np.mean(np.asarray(gains) > 0.0)),
        "worst_month_gain": float(min(gains)),
        "months": month_rows,
    }


def _robustness(
    axis: dict[str, np.ndarray],
    base: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
    family: list[np.ndarray],
) -> dict[str, Any]:
    target = np.asarray(axis["target"], dtype=np.float64)[active]
    incumbent = np.asarray(base, dtype=np.float64)[active]
    trial = np.asarray(candidate, dtype=np.float64)[active]
    improvement = np.column_stack(
        [
            np.square(incumbent - target)
            - np.square(np.asarray(item, dtype=np.float64)[active] - target)
            for item in family
        ]
    )
    return {
        "pitcher": one_way_cluster_bootstrap(
            target, trial, incumbent, np.asarray(axis["pitcher_id"])[active],
            n_resamples=2000, seed=173,
        ),
        "crossed_pitcher_batter": crossed_pigeonhole_bootstrap(
            target, trial, incumbent,
            np.asarray(axis["pitcher_id"])[active],
            np.asarray(axis["batter_id"])[active],
            n_resamples=2000, seed=174,
        ),
        "chronological_block": circular_block_bootstrap(
            target, trial, incumbent, block_size=512,
            n_resamples=2000, seed=175,
        ),
        "reality_check": white_reality_check(
            target, improvement, block_size=512,
            n_resamples=2000, seed=176,
        ),
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    locked, parity = _locked_contract(
        axes["full_2024"], frames[2024], correction[2024],
        h1_path, c3_path, v160_path,
    )
    locked_values = {
        "frame": frames[2024],
        "component": locked["component"],
        "h1": locked["h1"],
        "base": locked["current"],
    }

    rows: list[dict[str, Any]] = []
    source_details: dict[str, Any] = {}
    source_candidates: dict[float, dict[str, np.ndarray]] = {}
    for weight in WEIGHTS:
        key = f"F__w{weight:g}"
        details = {}
        candidates = {}
        for name in SOURCE_AXES:
            values = source[name]
            base = values["component"].copy()
            # Source current contract changes R_CORE only; non-core remains v104.
            from src.archive.v168_jy_exact_contract_reaudit import c3_mix, compose
            base = compose(
                base, values["h1"],
                c3_mix(values["sign"], values["recent"], 0.15),
                axes[name], h1_weight=0.15,
            )
            candidate, active = extend_h1(
                base, values["component"], values["h1"],
                axes[name]["domain3"], axes[name]["exact_mask"],
                ("F",), weight,
            )
            details[name] = paired_metrics(axes[name], base, candidate, active)
            candidates[name] = candidate
        eligible = all(
            details[name]["overall_gain"] > 0.0
            and details[name]["positive_month_fraction"] >= 0.5
            for name in SOURCE_AXES
        )
        rows.append(
            {
                "key": key,
                "weight": weight,
                "source_gate_passed": eligible,
                "source_min_gain": min(
                    details[name]["overall_gain"] for name in SOURCE_AXES
                ),
                "source_mean_gain": float(np.mean([
                    details[name]["overall_gain"] for name in SOURCE_AXES
                ])),
                "source_worst_month": min(
                    details[name]["worst_month_gain"] for name in SOURCE_AXES
                ),
            }
        )
        source_details[key] = details
        source_candidates[weight] = candidates

    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain"],
        ascending=False, kind="stable",
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_weight = float(selected["weight"])

    locked_candidates = []
    locked_details = {}
    selected_candidate = None
    selected_active = None
    for weight in WEIGHTS:
        candidate, active = extend_h1(
            locked_values["base"], locked_values["component"],
            locked_values["h1"], axes["full_2024"]["domain3"],
            axes["full_2024"]["exact_mask"], ("F",), weight,
        )
        locked_candidates.append(candidate)
        locked_details[f"F__w{weight:g}"] = paired_metrics(
            axes["full_2024"], locked_values["base"], candidate, active
        )
        if weight == selected_weight:
            selected_candidate, selected_active = candidate, active

    assert selected_candidate is not None and selected_active is not None
    robust = _robustness(
        axes["full_2024"], locked_values["base"],
        selected_candidate, selected_active, locked_candidates,
    )

    # One-source R_ANCHOR result is explicitly diagnostic and cannot be selected.
    anchor = {}
    for name, values in (
        ("late_2023", source["late_2023"]),
        ("full_2024", locked_values),
    ):
        if name == "late_2023":
            base = values["component"].copy()
            from src.archive.v168_jy_exact_contract_reaudit import c3_mix, compose
            base = compose(
                base, values["h1"],
                c3_mix(values["sign"], values["recent"], 0.15),
                axes[name], h1_weight=0.15,
            )
        else:
            base = values["base"]
        candidate, active = extend_h1(
            base, values["component"], values["h1"],
            axes[name]["domain3"], axes[name]["exact_mask"],
            ("R_ANCHOR",), selected_weight,
        )
        anchor[name] = paired_metrics(axes[name], base, candidate, active)

    selected_locked = locked_details[f"F__w{selected_weight:g}"]
    robust_pass = bool(
        robust["pitcher"]["p05"] > 0.0
        and robust["crossed_pitcher_batter"]["p05"] > 0.0
        and robust["chronological_block"]["p05"] > 0.0
        and robust["reality_check"]["p_value"] < 0.10
    )
    promote = bool(
        selected["source_gate_passed"]
        and selected_locked["overall_gain"] > 0.0
        and robust_pass
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=source_candidates[selected_weight]["full_2022"],
        late_2023=source_candidates[selected_weight]["late_2023"],
        full_2024=selected_candidate,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_release_audit" if promote else "reject",
        "root_cause": (
            "v131 changed non-core rows, but its metric helper hard-coded "
            "R_CORE as the only scored active domain"
        ),
        "parity": parity,
        "selected": selected,
        "source_details": source_details,
        "locked_details": locked_details,
        "locked_robustness": robust,
        "anchor_diagnostic_only": anchor,
        "eligible_for_packaging": promote,
        "limitations": [
            "R_ANCHOR has only late-2023 source support and is never promotion-eligible.",
            "The 2024 axis is locked and is not used to choose route or weight.",
        ],
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "leaderboard_score_used_for_selection": False,
            "official_train_only_for_fitting": True,
            "row_local_inference": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected"],
        "locked": result["locked_details"][
            result["selected"]["key"]
        ],
        "robustness": result["locked_robustness"],
        "anchor": result["anchor_diagnostic_only"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

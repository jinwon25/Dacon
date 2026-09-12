"""Dependence-aware audit of a frozen v87 + v98 additive correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    grouped_gain_table,
    one_way_cluster_bootstrap,
    white_reality_check,
)
from src.archive.v86_reliability_gated_r_fm import apply_reliability_offset
from src.archive.v87_cross_season_consensus_r_fm import combine_corrections
from src.core.contract import _diagnostics, _load_contract_axis
from src.archive.v100_cross_family_consensus import cross_family_candidate
from src.core.axis_metrics import _axis_metrics, _robust_axis


PROTOCOL = "V103_FIXED_UNION_ROBUST_V1"


def fixed_union_candidate(
    parent: np.ndarray,
    domain3: np.ndarray,
    fm_candidate: np.ndarray,
    conditional_correction: np.ndarray,
    conditional_eta: float,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    fm_candidate = np.asarray(fm_candidate, dtype=np.float64)
    correction = np.asarray(conditional_correction, dtype=np.float64)
    active = np.asarray(domain3).astype(str) == "R_CORE"
    candidate = parent.copy()
    candidate[active] = np.clip(
        fm_candidate[active] + float(conditional_eta) * correction[active],
        0.001,
        0.999,
    )
    return candidate


def run(
    train_csv: Path,
    contract_dir: Path,
    v87_dir: Path,
    v98_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    exact = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(contract_dir / "v84_full_2024.npz"),
    }
    late24_mask = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = {
        key: np.asarray(value)[late24_mask]
        for key, value in exact["full_2024"].items()
    }
    frames["late_2024"] = frames["full_2024"].loc[late24_mask].reset_index(drop=True)
    with np.load(v87_dir / "source_corrections.npz") as saved:
        fm_parts = {name: saved[name].astype(np.float64) for name in saved.files}
    with np.load(v87_dir / "outer_full_2024.npz") as saved:
        fm_parts["correction24_old"] = saved["correction_old"].astype(np.float64)
        fm_parts["correction24_recent"] = saved["correction_recent"].astype(np.float64)
    corrections = {
        "full_2022": np.load(v98_dir / "ablation_full_2022.npz")["correction"].astype(np.float64),
        "full_2024": np.load(v98_dir / "ablation_full_2024.npz")["correction"].astype(np.float64),
    }
    corrections["late_2024"] = corrections["full_2024"][late24_mask]
    common23 = _load_contract_axis(contract_dir / "common_full_2023.npz")
    correction23_full = np.load(v98_dir / "ablation_full_2023.npz")["correction"].astype(np.float64)
    location23 = {int(value): index for index, value in enumerate(common23["raw_index"])}
    corrections["late_2023"] = correction23_full[
        np.asarray([location23[int(value)] for value in exact["late_2023"]["raw_index"]])
    ]
    fm_lookup = {
        "full_2022": (fm_parts["correction22_old"], fm_parts["correction22_recent"]),
        "late_2023": (fm_parts["correction23_old"], fm_parts["correction23_recent"]),
        "full_2024": (fm_parts["correction24_old"], fm_parts["correction24_recent"]),
        "late_2024": (
            fm_parts["correction24_old"][late24_mask],
            fm_parts["correction24_recent"][late24_mask],
        ),
    }
    metrics: dict[str, dict[str, Any]] = {}
    robust: dict[str, dict[str, Any]] = {}
    saved_axes: dict[str, np.ndarray] = {}
    for index, name in enumerate(
        ("full_2022", "late_2023", "full_2024", "late_2024")
    ):
        axis = exact[name]
        frame = frames[name]
        older, recent = fm_lookup[name]
        fm_correction, _ = combine_corrections(older, recent, str(config["fm_policy"]))
        fm_candidate, _, _ = apply_reliability_offset(
            frame, axis["parent"], fm_correction, "none", eta=float(config["fm_eta"])
        )
        conditional_only = fixed_union_candidate(
            axis["parent"], axis["domain3"], axis["parent"], corrections[name],
            float(config["conditional_eta"]),
        )
        sign_candidate, _ = cross_family_candidate(
            axis["parent"], axis["domain3"], fm_candidate, corrections[name],
            float(config["conditional_eta"]),
        )
        union = fixed_union_candidate(
            axis["parent"], axis["domain3"], fm_candidate, corrections[name],
            float(config["conditional_eta"]),
        )
        metrics[name] = _axis_metrics(axis, union)
        robust[name] = _robust_axis(
            frame, axis, union,
            [fm_candidate, conditional_only, sign_candidate, union], config, 10 * index,
        )
        saved_axes[f"candidate_{name}"] = union
    np.savez_compressed(output_dir / "fixed_union_axes.npz", **saved_axes)
    gate = config["selection_gate"]
    point_pass = {
        name: bool(
            item["gain"] > 0.0
            and item["positive_month_fraction"] >= float(gate["minimum_positive_month_fraction"])
            and item["worst_month_gain"] > float(gate["worst_month_gain_strictly_above"])
            and item["minimum_domain_gain"] >= float(gate["minimum_domain_gain"])
        )
        for name, item in metrics.items()
    }
    robust_pass = {
        name: bool(
            all(result[key]["p05"] > 0.0 for key in (
                "pitcher", "crossed_pitcher_batter", "chronological_block"
            ))
            and result["reality_check"]["p_value"] <= float(gate["reality_check_alpha"])
        )
        for name, result in robust.items()
    }
    eligible = bool(all(point_pass.values()) and all(robust_pass.values()))
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "metrics": metrics,
        "point_gate_pass": point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
        "eligible_for_packaging": eligible,
        "packaging_reason": "all gates passed" if eligible else "point or robustness gate failed",
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v87-dir", type=Path, required=True)
    parser.add_argument("--v98-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v87_dir, args.v98_dir,
        args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()

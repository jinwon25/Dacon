"""Audit a low-DOF convex bridge between stable v142 and higher-gain v138."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.v103_fixed_union_robust import _axis_metrics
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V147_V142_V138_BLEND_AUDIT_V1"


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def run(
    contract_path: Path,
    v141_dir: Path,
    v138_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = _load_contract_axis(contract_path)
    with np.load(v141_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v142 = saved["full_2024"].astype(np.float64)
        v124 = saved["v124_full_2024"].astype(np.float64)
    with np.load(v138_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v138 = saved["full_2024"].astype(np.float64)
    late = axis["game_month"].astype(np.int16) >= 8
    trials = []
    candidates = {}
    gate = config["development_gate"]
    for weight in (float(value) for value in config["blend_grid"]):
        candidate = np.clip(v142 + weight * (v138 - v142), 0.001, 0.999)
        axis_v124 = {**axis, "parent": v124}
        axis_v142 = {**axis, "parent": v142}
        vs_v124 = {
            "full_2024": _axis_metrics(axis_v124, candidate),
            "late_2024": _axis_metrics(_slice_axis(axis_v124, late), candidate[late]),
        }
        incremental = {
            "full_2024": _axis_metrics(axis_v142, candidate),
            "late_2024": _axis_metrics(_slice_axis(axis_v142, late), candidate[late]),
        }
        passed = bool(
            vs_v124["full_2024"]["gain"] >= float(gate["gain_vs_v124_min"])
            and vs_v124["late_2024"]["gain"] >= float(gate["late_gain_vs_v124_min"])
            and vs_v124["full_2024"]["worst_month_gain"]
            > float(gate["worst_month_gain_min_exclusive"])
            and vs_v124["full_2024"]["positive_month_fraction"]
            >= float(gate["positive_month_fraction_min"])
        )
        trials.append(
            {
                "weight": weight,
                "gate_passed": passed,
                "gain_vs_v124": vs_v124["full_2024"]["gain"],
                "late_gain_vs_v124": vs_v124["late_2024"]["gain"],
                "worst_month_gain_vs_v124": vs_v124["full_2024"]["worst_month_gain"],
                "incremental_gain_vs_v142": incremental["full_2024"]["gain"],
                "incremental_late_gain_vs_v142": incremental["late_2024"]["gain"],
                "incremental_worst_month_gain_vs_v142": incremental["full_2024"]["worst_month_gain"],
                "metrics_vs_v124": vs_v124,
                "incremental_metrics_vs_v142": incremental,
            }
        )
        candidates[weight] = candidate
    passing = [row for row in trials if row["gate_passed"]]
    selected = max(passing, key=lambda row: (row["gain_vs_v124"], row["late_gain_vs_v124"], -row["weight"]))
    weight = float(selected["weight"])
    eligible = bool(weight > 0.0 and selected["incremental_gain_vs_v142"] > 0.0)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2024=candidates[weight],
        late_2024=candidates[weight][late],
        v142_full_2024=v142,
        v124_full_2024=v124,
        v138_full_2024=v138,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "development_candidate" if eligible else "reject",
        "selected_weight": weight,
        "trials": trials,
        "selected": selected,
        "eligible_for_packaging": eligible,
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
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--v141-dir", type=Path, required=True)
    parser.add_argument("--v138-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract_path, args.v141_dir, args.v138_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()

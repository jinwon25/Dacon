"""Select a low-DOF v142 direction dose on source axes, then audit locked 2024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.axis_metrics import _axis_metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V143_V142_SOURCE_ONLY_DOSE_V1"
CONTRACT_FILES = {
    "full_2022": "v84_full_2022.npz",
    "late_2023": "v84_late_2023.npz",
    "full_2024": "v84_full_2024.npz",
}


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _source_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def run(
    contract_dir: Path,
    v104_dir: Path,
    v139_dir: Path,
    v141_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in config["source_axes"]}
    with np.load(v139_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v139 = {name: saved[name].astype(np.float64) for name in config["source_axes"]}
    source_axes = {
        name: _load_contract_axis(contract_dir / CONTRACT_FILES[name])
        for name in config["source_axes"]
    }
    for name, axis in source_axes.items():
        if len(axis["target"]) != len(v104[name]) or len(v104[name]) != len(v139[name]):
            raise ValueError(f"source axis length mismatch: {name}")
        axis["parent"] = v104[name]

    trials = []
    for scale in (float(value) for value in config["scale_grid"]):
        metrics = {}
        for name in config["source_axes"]:
            candidate = np.clip(v104[name] + scale * (v139[name] - v104[name]), 0.001, 0.999)
            metrics[name] = _axis_metrics(source_axes[name], candidate)
        trial = {
            "scale": scale,
            "source_metrics": metrics,
            "source_gate_passed": bool(
                all(_source_pass(item, config["source_gate"]) for item in metrics.values())
            ),
            "minimum_gain": float(min(item["gain"] for item in metrics.values())),
            "mean_gain": float(np.mean([item["gain"] for item in metrics.values()])),
            "minimum_month_fraction": float(
                min(item["positive_month_fraction"] for item in metrics.values())
            ),
            "worst_month_gain": float(
                min(item["worst_month_gain"] for item in metrics.values())
            ),
        }
        trials.append(trial)

    passing = [trial for trial in trials if trial["source_gate_passed"]]
    if not passing:
        raise ValueError("no source-stable dose passed")
    selected = max(passing, key=lambda row: (row["minimum_gain"], row["mean_gain"], -row["scale"]))
    scale = float(selected["scale"])

    locked_axis = _load_contract_axis(contract_dir / CONTRACT_FILES["full_2024"])
    with np.load(v141_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v142 = saved["full_2024"].astype(np.float64)
        v124 = saved["v124_full_2024"].astype(np.float64)
    candidate = np.clip(v124 + scale * (v142 - v124), 0.001, 0.999)
    locked_axis["parent"] = v124
    vs_v124_full = _axis_metrics(locked_axis, candidate)
    late = locked_axis["game_month"].astype(np.int16) >= 8
    vs_v124_late = _axis_metrics(_slice_axis(locked_axis, late), candidate[late])
    locked_axis["parent"] = v142
    vs_v142_full = _axis_metrics(locked_axis, candidate)
    vs_v142_late = _axis_metrics(_slice_axis(locked_axis, late), candidate[late])
    veto = config["locked_incremental_veto"]
    locked_pass = bool(
        vs_v142_full["gain"] >= float(veto["full_gain_min"])
        and vs_v142_late["gain"] >= float(veto["late_gain_min"])
        and vs_v142_full["worst_month_gain"]
        > float(veto["worst_month_gain_min_exclusive"])
        and vs_v142_late["worst_month_gain"]
        > float(veto["worst_month_gain_min_exclusive"])
    )
    eligible = bool(scale > 1.0 and locked_pass)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2024=candidate,
        late_2024=candidate[late],
        v142_full_2024=v142,
        v124_full_2024=v124,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_packaging" if eligible else "reject",
        "selection_basis": "full_2022 and late_2023 only",
        "selected_scale": scale,
        "deploy_formula": {
            "v124_weight": 1.0 - 0.15 * scale,
            "h1_weight": 0.15 * scale,
            "c3_weight": 0.5 * scale,
        },
        "source_trials": trials,
        "selected_source": selected,
        "locked_metrics_vs_v124": {
            "full_2024": vs_v124_full,
            "late_2024": vs_v124_late,
        },
        "locked_incremental_vs_v142": {
            "full_2024": vs_v142_full,
            "late_2024": vs_v142_late,
        },
        "locked_veto_passed": locked_pass,
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
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v139-dir", type=Path, required=True)
    parser.add_argument("--v141-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.contract_dir,
        args.v104_dir,
        args.v139_dir,
        args.v141_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

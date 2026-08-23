"""Audit the fixed v133 H1+C3 correction above the v124 OOF parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.axis_metrics import _axis_metrics
from src.v123_public_quadratic_stack import _directions
from src.core.contract import _load_contract_axis


PROTOCOL = "V134_V124_C3_REBASE_V1"


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["gain"] >= float(gate["gain_min"])
        and metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def run(
    main_project: Path,
    contract_path: Path,
    v104_dir: Path,
    v133_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target, directions, audit = _directions(main_project.resolve())
    axis = _load_contract_axis(contract_path)
    if not np.array_equal(target, axis["target"].astype(np.float64)):
        raise ValueError("v123 direction target is not aligned with v84 full-2024")
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = saved["full_2024"].astype(np.float64)
    with np.load(v133_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v133 = saved["full_2024"].astype(np.float64)
    current = np.asarray(config["v124_current_point"], dtype=np.float64)
    selected = np.asarray(config["v124_selected_point"], dtype=np.float64)
    v124 = np.clip(v104 + directions @ (selected - current), 0.001, 0.999)
    candidate = np.clip(v124 + (v133 - v104), 0.001, 0.999)
    metric_axis = {**axis, "parent": v124}
    late = axis["game_month"].astype(np.int16) >= 8
    metrics = {
        "full_2024": _axis_metrics(metric_axis, candidate),
        "late_2024": _axis_metrics(_slice_axis(metric_axis, late), candidate[late]),
    }
    passes = {
        name: _point_pass(item, config["locked_gate"]) for name, item in metrics.items()
    }
    eligible = bool(all(passes.values()))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2024=candidate,
        late_2024=candidate[late],
        v124_full_2024=v124,
        v104_full_2024=v104,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "v123_direction_audit": audit,
        "v133_selected_key": config["v133_selected_key"],
        "rebase_formula": config["rebase_formula"],
        "locked_metrics_vs_v124": metrics,
        "locked_point_gate_pass": passes,
        "eligible_for_robust_audit": eligible,
        "eligible_for_packaging": False,
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
    parser.add_argument("--main-project", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v133-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.main_project,
        args.contract_path,
        args.v104_dir,
        args.v133_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

"""Audit a source-frozen May maturity gate on the v142-to-v138 bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V152_MAY_MATURITY_BRIDGE_AUDIT_V1"


def _slice(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _metrics(axis: dict[str, np.ndarray], parent: np.ndarray, candidate: np.ndarray) -> dict:
    return _axis_metrics({**axis, "parent": parent}, candidate)


def run(contract_dir: Path, v104_dir: Path, v131_dir: Path, v136_dir: Path,
        v139_dir: Path, v141_dir: Path, v138_dir: Path, config_path: Path,
        output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("full_2022", "late_2023", "full_2024")
    axes = {name: _load_contract_axis(contract_dir / f"v84_{name}.npz") for name in names}
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in names}
    with np.load(v131_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v131 = {name: saved[name].astype(np.float64) for name in names}
    with np.load(v136_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v136 = {name: saved[name].astype(np.float64) for name in names}
    with np.load(v139_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v139 = {name: saved[name].astype(np.float64) for name in names}
    with np.load(v141_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v142 = saved["full_2024"].astype(np.float64)
        v124 = saved["v124_full_2024"].astype(np.float64)
    with np.load(v138_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v138 = saved["full_2024"].astype(np.float64)

    excluded = np.isin(axes["full_2024"]["game_month"], config["excluded_months"])
    source_rows = []
    locked_rows = []
    candidates = {}
    cal = config["public_calibration"]
    local_a = float(cal["local_quadratic_linear"])
    local_b = float(cal["local_quadratic_curvature"])
    observed_delta = float(cal["v148_score"]) - float(cal["v142_score"])
    public_b = local_b
    public_a = (observed_delta + public_b * float(cal["v148_weight"]) ** 2) / float(cal["v148_weight"])
    for weight in config["bridge_weight_grid"]:
        weight = float(weight)
        source_metrics = {}
        source_gate_increments = {}
        source_pass = []
        for name in ("full_2022", "late_2023"):
            axis = axes[name]
            active = axis["exact_mask"].astype(bool) & (axis["domain3"].astype(str) == "R_CORE")
            sign = (v139[name] - v131[name]) / 0.5
            recent = (v136[name] - v131[name]) / 0.5
            candidate = v104[name].copy()
            allowed = active & ~np.isin(axis["game_month"], config["excluded_months"])
            c3 = (1.0 - weight) * sign + weight * recent
            candidate[allowed] = np.clip(v131[name][allowed] + 0.5 * c3[allowed], 0.001, 0.999)
            metric = _metrics(axis, v104[name], candidate)
            ungated = v104[name].copy()
            ungated[active] = np.clip(
                v131[name][active] + 0.5 * c3[active], 0.001, 0.999
            )
            ungated_metric = _metrics(axis, v104[name], ungated)
            source_gate_increments[name] = metric["gain"] - ungated_metric["gain"]
            source_metrics[name] = metric
            gate = config["source_gate"]
            source_pass.append(
                metric["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
                and metric["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
                and metric["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
            )
        source_rows.append({
            "weight": weight, "source_gate_passed": all(source_pass),
            "source_minimum_gain": min(x["gain"] for x in source_metrics.values()),
            "source_worst_month_gain": min(x["worst_month_gain"] for x in source_metrics.values()),
            "source_minimum_month_fraction": min(x["positive_month_fraction"] for x in source_metrics.values()),
            "full_2022_gate_increment": source_gate_increments["full_2022"],
            "late_2023_gate_increment": source_gate_increments["late_2023"],
        })
        candidate24 = np.clip(v142 + weight * (v138 - v142), 0.001, 0.999)
        candidate24[excluded] = v124[excluded]
        candidates[weight] = candidate24
        late = axes["full_2024"]["game_month"].astype(np.int16) >= 8
        metric = _metrics(axes["full_2024"], v124, candidate24)
        late_metric = _metrics(_slice(axes["full_2024"], late), v124[late], candidate24[late])
        estimated_public_no_gate = float(cal["v142_score"]) + public_a * weight - public_b * weight * weight
        locked_rows.append({
            "weight": weight, "source_gate_passed": all(source_pass),
            "gain_vs_v124": metric["gain"], "late_gain_vs_v124": late_metric["gain"],
            "positive_month_fraction": metric["positive_month_fraction"],
            "worst_month_gain": metric["worst_month_gain"],
            "estimated_public_before_gate": estimated_public_no_gate,
            "may_gate_local_increment": metric["gain"] - (local_a * weight - local_b * weight * weight + 6.943060261765936),
            "full_2022_gate_increment": source_gate_increments["full_2022"],
            "late_2023_gate_increment": source_gate_increments["late_2023"],
        })
    source_frame = pd.DataFrame(source_rows)
    locked = pd.DataFrame(locked_rows)
    gate = config["locked_gate"]
    locked["locked_gate_passed"] = (
        locked["source_gate_passed"]
        & (locked["gain_vs_v124"] >= float(gate["gain_vs_v124_min"]))
        & (locked["late_gain_vs_v124"] >= float(gate["late_gain_vs_v124_min"]))
        & (locked["positive_month_fraction"] >= float(gate["positive_month_fraction_min"]))
        & (locked["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"]))
    )
    # The gate's hidden transfer is not identified; expose a conservative and
    # an optimistic bound rather than silently counting it at 100%.
    locked["estimated_public_conservative"] = locked["estimated_public_before_gate"] + 0.5 * locked["may_gate_local_increment"]
    locked["estimated_public_optimistic"] = locked["estimated_public_before_gate"] + locked["may_gate_local_increment"]
    locked = locked.sort_values(
        ["locked_gate_passed", "estimated_public_conservative", "gain_vs_v124"],
        ascending=False, kind="stable"
    ).reset_index(drop=True)
    source_frame.to_csv(output_dir / "source_ranking.csv", index=False)
    locked.to_csv(output_dir / "locked_ranking.csv", index=False)
    deployment_weight = float(config["deployment_weight"])
    selected_rows = locked.loc[np.isclose(locked["weight"], deployment_weight)]
    if len(selected_rows) != 1:
        raise ValueError("deployment_weight must identify exactly one grid row")
    selected = selected_rows.iloc[0].to_dict()
    selected_weight = float(selected["weight"])
    late = axes["full_2024"]["game_month"].astype(np.int16) >= 8
    np.savez_compressed(
        output_dir / "selected_axes.npz", full_2024=candidates[selected_weight],
        late_2024=candidates[selected_weight][late], v124_full_2024=v124,
        v142_full_2024=v142, v138_full_2024=v138, excluded_mask=excluded,
    )
    result = {
        "protocol": PROTOCOL, "status": "promote_to_packaging" if bool(selected["locked_gate_passed"]) else "reject",
        "public_quadratic": {"linear": public_a, "curvature": public_b,
                             "optimum_weight": public_a / (2.0 * public_b)},
        "selected": selected, "ranking": locked.to_dict("records"),
        "eligible_for_packaging": bool(
            selected["locked_gate_passed"]
            and selected["estimated_public_conservative"] >= float(cal["target_score_min"])
        ),
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v131-dir", type=Path, required=True)
    parser.add_argument("--v136-dir", type=Path, required=True)
    parser.add_argument("--v139-dir", type=Path, required=True)
    parser.add_argument("--v141-dir", type=Path, required=True)
    parser.add_argument("--v138-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract_dir, args.v104_dir, args.v131_dir, args.v136_dir, args.v139_dir,
        args.v141_dir, args.v138_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()

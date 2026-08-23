"""Audit joint H1/C3/base-response weights above the frozen v148 champion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics
from src.v148_contract import reconstruct_deployed_v148
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V149_JOINT_H1_C3_RESPONSE_AUDIT_V1"
SOURCE_AXES = ("full_2022", "late_2023")


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _active(axis: dict[str, np.ndarray]) -> np.ndarray:
    return axis["exact_mask"].astype(bool) & (axis["domain3"].astype(str) == "R_CORE")


def _recover_component(
    parent: np.ndarray,
    blended: np.ndarray,
    active: np.ndarray,
    weight: float,
) -> np.ndarray:
    component = np.asarray(parent, dtype=np.float64).copy()
    component[active] = (
        np.asarray(blended, dtype=np.float64)[active]
        - (1.0 - float(weight)) * np.asarray(parent, dtype=np.float64)[active]
    ) / float(weight)
    return component


def _recover_correction(
    base: np.ndarray,
    corrected: np.ndarray,
    active: np.ndarray,
    dose: float,
) -> np.ndarray:
    correction = np.zeros(len(base), dtype=np.float64)
    correction[active] = (
        np.asarray(corrected, dtype=np.float64)[active]
        - np.asarray(base, dtype=np.float64)[active]
    ) / float(dose)
    return correction


def _candidate(
    v124: np.ndarray,
    v104: np.ndarray,
    h1: np.ndarray,
    sign_all: np.ndarray,
    mean_recent: np.ndarray,
    active: np.ndarray,
    h1_weight: float,
    c3_weight: float,
    base_bridge: float,
    consensus_mix: float,
) -> np.ndarray:
    output = np.asarray(v124, dtype=np.float64).copy()
    base = (1.0 - base_bridge) * v124[active] + base_bridge * v104[active]
    c3 = (1.0 - consensus_mix) * sign_all[active] + consensus_mix * mean_recent[active]
    output[active] = np.clip(
        (1.0 - h1_weight) * base + h1_weight * h1[active] + c3_weight * c3,
        0.001,
        0.999,
    )
    return output


def _source_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def run(
    contract_dir: Path,
    v104_dir: Path,
    v131_dir: Path,
    v136_dir: Path,
    v139_dir: Path,
    v141_dir: Path,
    v147_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in (*SOURCE_AXES, "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v131_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v131 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v136_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v136 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v139_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v139 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, "full_2024")}
    with np.load(v141_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v124_2024 = saved["v124_full_2024"].astype(np.float64)
    with np.load(v147_dir / "selected_axes.npz", allow_pickle=False) as saved:
        # v147's selected point is the strict 5% audit bridge.  The deployed
        # v148 package intentionally uses the separately frozen 15% bridge.
        v142_2024 = saved["v142_full_2024"].astype(np.float64)
        v138_2024 = saved["v138_full_2024"].astype(np.float64)
        v148_2024 = reconstruct_deployed_v148(v142_2024, v138_2024)

    components: dict[str, dict[str, np.ndarray]] = {}
    for name in (*SOURCE_AXES, "full_2024"):
        active = _active(axes[name])
        components[name] = {
            "active": active,
            "h1": _recover_component(v104[name], v131[name], active, 0.15),
            "sign_all": _recover_correction(v131[name], v139[name], active, 0.5),
            "mean_recent": _recover_correction(v131[name], v136[name], active, 0.5),
        }

    source_rows: list[dict[str, Any]] = []
    source_pass: dict[tuple[float, float, float], bool] = {}
    source_metrics: dict[tuple[float, float, float], dict[str, Any]] = {}
    for h1_weight in config["h1_weight_grid"]:
        for c3_weight in config["c3_weight_grid"]:
            for consensus_mix in config["consensus_mix_grid"]:
                point = (float(h1_weight), float(c3_weight), float(consensus_mix))
                metrics: dict[str, Any] = {}
                for name in SOURCE_AXES:
                    item = components[name]
                    candidate = _candidate(
                        v104[name], v104[name], item["h1"], item["sign_all"],
                        item["mean_recent"], item["active"], *point[:2], 0.0, point[2]
                    )
                    metrics[name] = _axis_metrics({**axes[name], "parent": v104[name]}, candidate)
                passed = all(_source_pass(metric, config["source_gate"]) for metric in metrics.values())
                source_pass[point] = passed
                source_metrics[point] = metrics
                source_rows.append({
                    "h1_weight": point[0],
                    "c3_weight": point[1],
                    "consensus_mix": point[2],
                    "source_gate_passed": passed,
                    "minimum_gain": min(metric["gain"] for metric in metrics.values()),
                    "mean_gain": np.mean([metric["gain"] for metric in metrics.values()]),
                    "minimum_month_fraction": min(metric["positive_month_fraction"] for metric in metrics.values()),
                    "worst_month_gain": min(metric["worst_month_gain"] for metric in metrics.values()),
                })

    metric_axis_v124 = {**axes["full_2024"], "parent": v124_2024}
    metric_axis_v148 = {**axes["full_2024"], "parent": v148_2024}
    late = axes["full_2024"]["game_month"].astype(np.int16) >= 8
    item = components["full_2024"]
    calibration = config["public_calibration"]
    transfer_v142 = (
        float(calibration["v142_score"]) - float(calibration["v124_score"])
    ) / float(calibration["v142_local_gain_vs_v124"])
    transfer_v148 = (
        float(calibration["v148_score"]) - float(calibration["v142_score"])
    ) / float(calibration["v148_local_increment_vs_v142"])
    conservative_transfer = float(calibration["conservative_transfer_ratio"])
    locked_rows: list[dict[str, Any]] = []
    candidates: dict[tuple[float, float, float, float], np.ndarray] = {}
    for h1_weight in config["h1_weight_grid"]:
        for c3_weight in config["c3_weight_grid"]:
            for base_bridge in config["base_bridge_grid"]:
                for consensus_mix in config["consensus_mix_grid"]:
                    source_key = (float(h1_weight), float(c3_weight), float(consensus_mix))
                    if not source_pass[source_key]:
                        continue
                    key = (*source_key[:2], float(base_bridge), source_key[2])
                    candidate = _candidate(
                        v124_2024, v104["full_2024"], item["h1"], item["sign_all"],
                        item["mean_recent"], item["active"], *key
                    )
                    candidates[key] = candidate
                    vs_v124 = _axis_metrics(metric_axis_v124, candidate)
                    vs_v148 = _axis_metrics(metric_axis_v148, candidate)
                    late_vs_v124 = _axis_metrics(
                        _slice_axis(metric_axis_v124, late), candidate[late]
                    )
                    source_item = source_metrics[source_key]
                    estimated_score_anchor = float(calibration["v124_score"]) + transfer_v142 * vs_v124["gain"]
                    estimated_score_increment = float(calibration["v148_score"]) + conservative_transfer * vs_v148["gain"]
                    locked_rows.append({
                        "h1_weight": key[0], "c3_weight": key[1],
                        "base_bridge": key[2], "consensus_mix": key[3],
                        "gain_vs_v124": vs_v124["gain"],
                        "late_gain_vs_v124": late_vs_v124["gain"],
                        "worst_month_gain_vs_v124": vs_v124["worst_month_gain"],
                        "positive_month_fraction_vs_v124": vs_v124["positive_month_fraction"],
                        "incremental_gain_vs_v148": vs_v148["gain"],
                        "incremental_worst_month_gain_vs_v148": vs_v148["worst_month_gain"],
                        "source_minimum_gain": min(x["gain"] for x in source_item.values()),
                        "source_worst_month_gain": min(x["worst_month_gain"] for x in source_item.values()),
                        "estimated_public_from_v124": estimated_score_anchor,
                        "estimated_public_from_v148": estimated_score_increment,
                    })

    source_ranking = pd.DataFrame(source_rows).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=False, kind="stable"
    ).reset_index(drop=True)
    locked_ranking = pd.DataFrame(locked_rows)
    gate = config["locked_gate"]
    locked_ranking["locked_gate_passed"] = (
        (locked_ranking["gain_vs_v124"] >= float(gate["gain_vs_v124_min"]))
        & (locked_ranking["late_gain_vs_v124"] >= float(gate["late_gain_vs_v124_min"]))
        & (locked_ranking["positive_month_fraction_vs_v124"] >= float(gate["positive_month_fraction_min"]))
        & (locked_ranking["worst_month_gain_vs_v124"] > float(gate["worst_month_gain_min_exclusive"]))
        & (locked_ranking["incremental_gain_vs_v148"] >= float(gate["incremental_gain_vs_v148_min"]))
    )
    locked_ranking = locked_ranking.sort_values(
        ["locked_gate_passed", "estimated_public_from_v148", "gain_vs_v124", "source_minimum_gain"],
        ascending=False, kind="stable"
    ).reset_index(drop=True)
    source_ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    locked_ranking.to_csv(output_dir / "locked_ranking.csv", index=False)

    passing = locked_ranking.loc[locked_ranking["locked_gate_passed"]]
    selected_row = (passing.iloc[0] if len(passing) else locked_ranking.iloc[0]).to_dict()
    selected_key = tuple(float(selected_row[name]) for name in (
        "h1_weight", "c3_weight", "base_bridge", "consensus_mix"
    ))
    selected_candidate = candidates[selected_key]
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2024=selected_candidate,
        late_2024=selected_candidate[late],
        v124_full_2024=v124_2024,
        v148_full_2024=v148_2024,
        active_mask=item["active"],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_packaging" if bool(selected_row["locked_gate_passed"]) else "reject",
        "n_source_points": int(len(source_ranking)),
        "n_locked_points": int(len(locked_ranking)),
        "transfer_ratio_v142_vs_v124": transfer_v142,
        "transfer_ratio_v148_vs_v142": transfer_v148,
        "selected": selected_row,
        "top20": locked_ranking.head(20).to_dict(orient="records"),
        "target_score_min": float(calibration["target_score_min"]),
        "eligible_for_packaging": bool(
            selected_row["locked_gate_passed"]
            and selected_row["estimated_public_from_v148"] >= float(calibration["target_score_min"])
        ),
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
    parser.add_argument("--v131-dir", type=Path, required=True)
    parser.add_argument("--v136-dir", type=Path, required=True)
    parser.add_argument("--v139-dir", type=Path, required=True)
    parser.add_argument("--v141-dir", type=Path, required=True)
    parser.add_argument("--v147-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract_dir, args.v104_dir, args.v131_dir, args.v136_dir, args.v139_dir,
        args.v141_dir, args.v147_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()

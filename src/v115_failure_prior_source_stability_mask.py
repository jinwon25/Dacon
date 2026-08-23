"""Apply a source-only context stability mask to the frozen v113 failure prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics, _robust_axis
from src.v104_source_stability_mask import (
    _learn_safe_levels,
    _point_pass,
    context_labels,
    policy_mask,
)
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V115_FAILURE_PRIOR_SOURCE_STABILITY_MASK_V1"
AXES = ("full_2022", "late_2023", "full_2024", "late_2024")
SOURCE_AXES = ("full_2022", "late_2023")


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v113_dir: Path,
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
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    late24_mask = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = _slice_axis(exact["full_2024"], late24_mask)
    frames["late_2024"] = frames["full_2024"].loc[late24_mask].reset_index(drop=True)
    with np.load(v104_dir / "selected_axes.npz") as saved:
        v104 = {name: saved[name].astype(np.float64) for name in AXES}
    with np.load(v113_dir / "selected_axes.npz") as saved:
        raw_candidate = {name: saved[name].astype(np.float64) for name in AXES}
    rebased = {name: {**exact[name], "parent": v104[name]} for name in AXES}

    safe, source_tables = _learn_safe_levels(
        frames,
        rebased,
        raw_candidate,
        int(config["minimum_group_rows_per_source"]),
    )
    policies = [str(value) for value in config["policies"]]
    candidates: dict[str, dict[str, np.ndarray]] = {name: {} for name in policies}
    source_metrics: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in policies
    }
    mask_fraction: dict[str, dict[str, float]] = {name: {} for name in policies}
    for axis in AXES:
        labels = context_labels(frames[axis])
        for policy in policies:
            mask = policy_mask(labels, safe, policy)
            candidates[policy][axis] = np.where(
                mask, raw_candidate[axis], v104[axis]
            ).astype(np.float64)
            mask_fraction[policy][axis] = float(mask.mean())
            if axis in SOURCE_AXES:
                source_metrics[policy][axis] = _axis_metrics(
                    rebased[axis], candidates[policy][axis]
                )

    gate = config["selection_gate"]
    ranking_rows: list[dict[str, Any]] = []
    for policy in policies:
        items = [source_metrics[policy][name] for name in SOURCE_AXES]
        ranking_rows.append(
            {
                "policy": policy,
                "source_gate_passed": bool(all(_point_pass(item, gate) for item in items)),
                "minimum_gain": float(min(item["gain"] for item in items)),
                "worst_month_gain": float(min(item["worst_month_gain"] for item in items)),
                "minimum_month_fraction": float(
                    min(item["positive_month_fraction"] for item in items)
                ),
                "mean_mask_fraction": float(
                    np.mean([mask_fraction[policy][name] for name in SOURCE_AXES])
                ),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_policy_ranking.csv", index=False)
    selected = str(ranking.iloc[0]["policy"])
    selected_source_pass = bool(ranking.iloc[0]["source_gate_passed"])
    metrics = {
        name: _axis_metrics(rebased[name], candidates[selected][name]) for name in AXES
    }
    point_pass = {name: _point_pass(value, gate) for name, value in metrics.items()}
    family = {
        name: [candidates[policy][name] for policy in policies] for name in AXES
    }
    robust = {
        name: _robust_axis(
            frames[name], rebased[name], candidates[selected][name], family[name],
            config, 400 + 10 * index,
        )
        for index, name in enumerate(AXES)
    }
    robust_pass = {
        name: bool(
            all(
                value[key]["p05"] > 0.0
                for key in ("pitcher", "crossed_pitcher_batter", "chronological_block")
            )
            and value["reality_check"]["p_value"] <= float(gate["reality_check_alpha"])
        )
        for name, value in robust.items()
    }
    eligible = bool(
        selected_source_pass
        and all(point_pass.values())
        and all(robust_pass.values())
    )
    np.savez_compressed(output_dir / "selected_axes.npz", **candidates[selected])
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "safe_levels_learned_from_sources": safe,
        "source_group_tables": source_tables,
        "source_ranking": ranking.to_dict(orient="records"),
        "selected_policy": selected,
        "selected_source_gate_passed": selected_source_pass,
        "selected_mask_fraction": mask_fraction[selected],
        "metrics": metrics,
        "point_gate_pass": point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
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
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v113-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v104_dir, args.v113_dir,
        args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()

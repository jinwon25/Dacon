"""Rebase the frozen v105 architecture correction on the exact v104 OOF parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.champion.v104_source_stability_mask import (
    _axis_metrics,
    _load_contract_axis,
    _point_pass,
    _robust_axis,
    context_labels,
    policy_mask,
)


PROTOCOL = "V110_V104_CROSS_ARCHITECTURE_REBASE_V1"
SUPPORTED_PROTOCOLS = {PROTOCOL, "V111_V104_INACTIVE_ARCHITECTURE_ROUTE_V1"}


def fit_alpha(
    targets: list[np.ndarray],
    parents: list[np.ndarray],
    corrections: list[np.ndarray],
    cap: float,
) -> float:
    """Fit one source-only least-squares probability dose with a fixed bound."""
    error = np.concatenate(
        [np.asarray(y, float) - np.asarray(p, float) for y, p in zip(targets, parents)]
    )
    delta = np.concatenate([np.asarray(value, float) for value in corrections])
    denominator = float(np.dot(delta, delta))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(error, delta) / denominator, 0.0, cap))


def apply_correction(parent: np.ndarray, correction: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(
        np.asarray(parent, dtype=np.float64)
        + float(alpha) * np.asarray(correction, dtype=np.float64),
        0.001,
        0.999,
    )


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v105_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"config protocol must be one of {sorted(SUPPORTED_PROTOCOLS)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(train_csv, low_memory=False)
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    late_mask = axes["full_2024"]["game_month"].astype(np.int16) >= 8
    axes["late_2024"] = _slice_axis(axes["full_2024"], late_mask)
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    frames["late_2024"] = frames["full_2024"].loc[late_mask].reset_index(drop=True)

    with np.load(v104_dir / "selected_axes.npz") as saved:
        v104 = {name: saved[name].astype(np.float64) for name in axes}
    with np.load(v105_dir / "selected_axes.npz") as saved:
        v105 = {name: saved[name].astype(np.float64) for name in axes}

    scope = str(config.get("correction_scope", "all"))
    if scope == "all":
        scope_masks = {name: np.ones(len(frame), dtype=bool) for name, frame in frames.items()}
    elif scope == "v104_inactive":
        v104_summary = json.loads((v104_dir / "summary.json").read_text(encoding="utf-8"))
        safe = v104_summary["safe_levels_learned_from_sources"]
        scope_masks = {
            name: ~policy_mask(context_labels(frame), safe, "majority_two")
            for name, frame in frames.items()
        }
    else:
        raise ValueError(f"unknown correction_scope: {scope}")

    rebased_axes: dict[str, dict[str, np.ndarray]] = {}
    correction: dict[str, np.ndarray] = {}
    for name, axis in axes.items():
        if len(v104[name]) != len(axis["parent"]) or len(v105[name]) != len(axis["parent"]):
            raise ValueError(f"axis length mismatch: {name}")
        correction[name] = np.where(
            scope_masks[name],
            v105[name] - axis["parent"].astype(np.float64),
            0.0,
        )
        rebased_axes[name] = {**axis, "parent": v104[name]}

    source_names = ("full_2022", "late_2023")
    alpha = fit_alpha(
        [axes[name]["target"] for name in source_names],
        [v104[name] for name in source_names],
        [correction[name] for name in source_names],
        float(config["alpha_cap"]),
    )
    candidate = {
        name: apply_correction(v104[name], correction[name], alpha) for name in axes
    }
    metrics = {
        name: _axis_metrics(rebased_axes[name], candidate[name]) for name in axes
    }
    gate = config["selection_gate"]
    point_pass = {name: _point_pass(value, gate) for name, value in metrics.items()}

    family_alpha = sorted(
        set(float(value) for value in config["family_alpha"]) | {float(alpha)}
    )
    family = {
        name: [
            apply_correction(v104[name], correction[name], value)
            for value in family_alpha
        ]
        for name in axes
    }
    robust = {
        name: _robust_axis(
            frames[name], rebased_axes[name], candidate[name], family[name], config,
            100 + 10 * index,
        )
        for index, name in enumerate(axes)
    }
    robust_pass = {
        name: bool(
            all(
                result[key]["p05"] > 0.0
                for key in ("pitcher", "crossed_pitcher_batter", "chronological_block")
            )
            and result["reality_check"]["p_value"] <= float(gate["reality_check_alpha"])
        )
        for name, result in robust.items()
    }
    eligible = bool(all(point_pass.values()) and all(robust_pass.values()))

    np.savez_compressed(output_dir / "v104_exact_axes.npz", **v104)
    np.savez_compressed(output_dir / "selected_axes.npz", **candidate)
    result = {
        "protocol": config["protocol"],
        "status": "promote" if eligible else "reject",
        "correction_scope": scope,
        "scope_fraction": {
            name: float(mask.mean()) for name, mask in scope_masks.items()
        },
        "selected_alpha": alpha,
        "family_alpha": family_alpha,
        "metrics": metrics,
        "point_gate_pass": point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
        "eligible_for_packaging": eligible,
        "correction_correlation_with_v104": {
            name: float(np.corrcoef(v104[name] - axes[name]["parent"], correction[name])[0, 1])
            for name in axes
        },
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
    parser.add_argument("--v105-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v104_dir,
        args.v105_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

"""Audit the original public H1 affine on exact ensemble OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.axis_metrics import _axis_metrics
from src.core.contract import _load_contract_axis
from src.core.robustness import evaluate_robustness


PROTOCOL = "V160_ORIGINAL_H1_AFFINE_AUDIT_V1"
SOURCE_AXES = ("full_2022", "late_2023")


def _affine(prediction: np.ndarray, alpha: float, center: float) -> np.ndarray:
    return np.clip(center + alpha * (prediction - center), 0.001, 0.999)


def _delta(
    base: np.ndarray,
    axis: dict[str, np.ndarray],
    h1: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    adjusted = _affine(h1, float(config["alpha"]), float(config["center"]))
    active = axis["exact_mask"].astype(bool)
    active &= axis["domain3"].astype(str) == str(config["route"])
    output = np.asarray(base, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active]
        + float(config["h1_weight"]) * (adjusted[active] - h1[active]),
        0.001,
        0.999,
    )
    return output


def _metrics(
    axis: dict[str, np.ndarray], base: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    return _axis_metrics({**axis, "parent": base}, candidate)


def _pass(metrics: dict[str, Any], gate: dict[str, Any], *, locked: bool) -> bool:
    threshold = float(gate.get("gain_min", gate.get("gain_min_each", 0.0)))
    comparison = metrics["gain"] >= threshold if locked else metrics["gain"] > threshold
    return bool(
        comparison
        and metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["minimum_domain_gain"])
    )


def run(
    contract_dir: Path,
    v157_dir: Path,
    v158_dir: Path,
    v148_oof: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in SOURCE_AXES
    }
    with np.load(v157_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        h1 = {
            year: saved[f"exact_h1_{year}"].astype(np.float64)
            for year in (2022, 2023, 2024)
        }
        deployed_exact = saved["v148_exact_h1_candidate"].astype(np.float64)
    with np.load(v158_dir / "selected_axes.npz", allow_pickle=False) as saved:
        source_base = {
            "full_2022": saved["full_2022"].astype(np.float64),
            "late_2023": saved["late_2023"].astype(np.float64),
        }
    # v158 source arrays are exact-H1/exact-C3 formulas.  Replacing only the
    # H1 affine remains valid because C3 is held fixed in both arms.
    # Guard against relying on row contiguity: recover late-season positions
    # from the full-2023 contract's raw train indices.
    full23 = _load_contract_axis(contract_dir / "common_full_2023.npz")
    full23_raw = full23["raw_index"].astype(np.int64)
    late_raw = axes["late_2023"]["raw_index"].astype(np.int64)
    late_position = np.searchsorted(full23_raw, late_raw)
    if (
        np.any(late_position >= len(full23_raw))
        or not np.array_equal(full23_raw[late_position], late_raw)
    ):
        raise ValueError("late-2023 H1/contract alignment mismatch")
    late23_h1 = h1[2023][late_position]
    source_h1 = {"full_2022": h1[2022], "late_2023": late23_h1}
    source_candidate = {
        name: _delta(source_base[name], axes[name], source_h1[name], config)
        for name in SOURCE_AXES
    }
    source_metrics = {
        name: _metrics(axes[name], source_base[name], source_candidate[name])
        for name in SOURCE_AXES
    }
    source_gate = all(
        _pass(item, config["source_gate"], locked=False)
        for item in source_metrics.values()
    )

    axis148 = _load_contract_axis(v148_oof)
    candidate148 = _delta(deployed_exact, axis148, h1[2024], config)
    locked_metrics = _metrics(axis148, deployed_exact, candidate148)
    robust = evaluate_robustness(
        axis148,
        candidate148,
        [candidate148, deployed_exact],
        config["robustness"],
    )
    gate = config["locked_gate"]
    locked_gate = bool(
        _pass(locked_metrics, gate, locked=True)
        and robust["pitcher"]["p05"] > float(gate["bootstrap_p05_min"])
        and robust["crossed_pitcher_batter"]["p05"]
        > float(gate["bootstrap_p05_min"])
        and robust["chronological_block"]["p05"]
        > float(gate["bootstrap_p05_min"])
        and robust["reality_check"]["p_value"]
        <= float(gate["reality_check_alpha"])
    )
    promotion = bool(source_gate and locked_gate)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=source_candidate["full_2022"],
        late_2023=source_candidate["late_2023"],
        full_2024=candidate148,
        deployed_exact_full_2024=deployed_exact,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_packaging" if promotion else "rejected",
        "fixed_affine": {"alpha": config["alpha"], "center": config["center"]},
        "source_metrics": source_metrics,
        "source_gate_passed": source_gate,
        "locked_metrics": locked_metrics,
        "robustness": robust,
        "locked_gate_passed": locked_gate,
        "promotion_gate_passed": promotion,
        "eligible_for_packaging": promotion,
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
    parser.add_argument("--v157-dir", type=Path, required=True)
    parser.add_argument("--v158-dir", type=Path, required=True)
    parser.add_argument("--v148-oof", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.contract_dir,
        args.v157_dir,
        args.v158_dir,
        args.v148_oof,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

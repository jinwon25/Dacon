"""Measure affine calibration headroom of the new v148 model family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.core.axis_metrics import _axis_metrics
from src.champion.v148_contract import reconstruct_deployed_v148
from src.core.contract import _load_contract_axis


def _current_source(
    axis: dict[str, np.ndarray], v104: np.ndarray, v131: np.ndarray,
    v136: np.ndarray, v139: np.ndarray,
) -> np.ndarray:
    active = axis["exact_mask"].astype(bool) & (axis["domain3"].astype(str) == "R_CORE")
    sign = (v139 - v131) / 0.5
    recent = (v136 - v131) / 0.5
    output = v104.copy()
    output[active] = np.clip(
        v131[active] + 0.5 * (0.85 * sign[active] + 0.15 * recent[active]),
        0.001, 0.999,
    )
    return output


def _fit(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(mask.sum()), prediction[mask]])
    coefficient, *_ = np.linalg.lstsq(design, target[mask], rcond=None)
    return float(coefficient[0]), float(coefficient[1])


def _apply(prediction: np.ndarray, mask: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    output = prediction.copy()
    output[mask] = np.clip(intercept + slope * prediction[mask], 0.001, 0.999)
    return output


def run(contract_dir: Path, v104_dir: Path, v131_dir: Path, v136_dir: Path,
        v139_dir: Path, v147_dir: Path, output_dir: Path) -> dict:
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
    with np.load(v147_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v142 = saved["v142_full_2024"].astype(np.float64)
        v138 = saved["v138_full_2024"].astype(np.float64)
        v148 = reconstruct_deployed_v148(v142, v138)
    current = {
        name: _current_source(axes[name], v104[name], v131[name], v136[name], v139[name])
        for name in names[:-1]
    }
    current["full_2024"] = v148
    oracle = {}
    fitted = {}
    for name in names:
        axis, prediction = axes[name], current[name]
        target = axis["target"].astype(np.float64)
        exact = axis["exact_mask"].astype(bool)
        rcore = exact & (axis["domain3"].astype(str) == "R_CORE")
        fitted[name] = {}
        oracle[name] = {}
        for route, mask in (("ALL", exact), ("R_CORE", rcore)):
            intercept, slope = _fit(target, prediction, mask)
            candidate = _apply(prediction, mask, intercept, slope)
            fitted[name][route] = {"intercept": intercept, "slope": slope}
            oracle[name][route] = _axis_metrics({**axis, "parent": prediction}, candidate)
    transfer = {}
    for source in names[:-1]:
        transfer[source] = {}
        axis, prediction = axes["full_2024"], current["full_2024"]
        exact = axis["exact_mask"].astype(bool)
        rcore = exact & (axis["domain3"].astype(str) == "R_CORE")
        for route, mask in (("ALL", exact), ("R_CORE", rcore)):
            spec = fitted[source][route]
            candidate = _apply(prediction, mask, spec["intercept"], spec["slope"])
            transfer[source][route] = {
                **spec, **_axis_metrics({**axis, "parent": prediction}, candidate)
            }
    result = {
        "protocol": "V151_V148_CALIBRATION_HEADROOM_V1",
        "oracle_diagnostic_only": oracle,
        "source_fitted": fitted,
        "source_to_full_2024": transfer,
        "eligible_for_packaging": False,
        "test_csv_read": False,
        "full_2024_used_for_recipe_selection": False,
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
    parser.add_argument("--v147-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract_dir, args.v104_dir, args.v131_dir, args.v136_dir,
        args.v139_dir, args.v147_dir, args.output_dir)


if __name__ == "__main__":
    main()

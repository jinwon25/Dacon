"""Forward-only calibration audit for the deployed Public1175 fallback XGB.

Calibration form and dose are selected on two chronological source checks:
early-2022 -> late-2022 and full-2022 -> late-2023.  The selected recipe is
then refit on all source evidence and evaluated once on locked full-2024.
The Public1175 gate, threshold, and 30% XGB blend weight never change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    XGB_WEIGHT,
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.core.contract import _load_contract_axis


PROTOCOL = "V227_FALLBACK_FORWARD_CALIBRATION_LOCKED_AUDIT_V1"
MODES = ("offset", "affine")
DOSES = (0.25, 0.50, 0.75, 1.00)
AXES = ("full_2022", "late_2023", "full_2024")


def full_axis(axis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {**axis, "exact_mask": np.ones(len(axis["target"]), dtype=bool)}


def slice_axis(
    axis: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, np.ndarray]:
    selected = np.asarray(selected, dtype=bool)
    if len(selected) != len(axis["target"]):
        raise ValueError("axis slice length mismatch")
    output: dict[str, np.ndarray] = {}
    for key, value in axis.items():
        array = np.asarray(value)
        output[key] = array[selected] if array.ndim and len(array) == len(selected) else value
    return full_axis(output)


def fit_calibrator(
    parent: np.ndarray,
    xgb_prediction: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    mode: str,
) -> tuple[float, float]:
    """Fit ``a + b*xgb`` for the XGB leg of the frozen 70:30 blend."""
    active = np.asarray(active, dtype=bool)
    parent = np.asarray(parent, dtype=np.float64)[active]
    xgb_prediction = np.asarray(xgb_prediction, dtype=np.float64)[active]
    target = np.asarray(target, dtype=np.float64)[active]
    if len(target) == 0 or not np.isfinite(xgb_prediction).all():
        raise ValueError("calibrator fit support is empty or non-finite")
    pseudo_target = (
        target - (1.0 - XGB_WEIGHT) * parent
    ) / XGB_WEIGHT
    if mode == "offset":
        return float(np.mean(pseudo_target - xgb_prediction)), 1.0
    if mode == "affine":
        design = np.column_stack([np.ones(len(target)), xgb_prediction])
        coefficient, *_ = np.linalg.lstsq(design, pseudo_target, rcond=None)
        return float(coefficient[0]), float(coefficient[1])
    raise ValueError(f"unknown calibration mode: {mode}")


def calibrated_fallback(
    parent: np.ndarray,
    xgb_prediction: np.ndarray,
    pressure: np.ndarray,
    parameters: tuple[float, float],
    dose: float,
) -> tuple[np.ndarray, np.ndarray]:
    a, b = parameters
    raw = np.asarray(xgb_prediction, dtype=np.float64)
    fitted = a + b * raw
    calibrated = np.clip(raw + float(dose) * (fitted - raw), 0.001, 0.999)
    return apply_fallback(parent, calibrated, pressure)


def source_gate(results: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(results[name]["gain"] > 0.0 for name in ("late_2022", "late_2023"))
        and results["late_2022"]["positive_month_fraction"] >= (2.0 / 3.0)
        and results["late_2023"]["positive_month_fraction"] >= (2.0 / 3.0)
        and results["late_2022"]["worst_month_gain"] > -5.0
        and results["late_2023"]["worst_month_gain"] > -5.0
        and results["late_2022"]["minimum_domain_gain"] >= 0.0
        and results["late_2023"]["minimum_domain_gain"] >= 0.0
    )


def select_variant(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, int, str]] = []
    for key, axes in results.items():
        if source_gate(axes):
            gains = [axes[name]["gain"] for name in ("late_2022", "late_2023")]
            simpler = int(key.startswith("offset"))
            eligible.append((min(gains), float(np.mean(gains)), simpler, key))
    return None if not eligible else max(eligible)[-1]


def restrictions() -> dict[str, bool]:
    return {
        "fallback_xgb_model_frozen": True,
        "public1175_gate_frozen": True,
        "public1175_threshold_frozen_at_050": True,
        "public1175_weight_frozen_at_030": True,
        "source_only_calibration_selection": True,
        "chronological_source_fits": True,
        "locked_2024_not_used_for_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    fallback_oof_dir: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "season", "game_type", "pitcher_team_id", "batter_team_id",
        "num_runners_on", "li", "game_month", "pitcher_id",
    ]
    train = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    season = train["season"].to_numpy(np.int16)
    frames = {
        year: train.loc[season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = {
        "full_2022": full_axis(_load_contract_axis(
            contract_dir / "v84_full_2022.npz"
        )),
        "late_2023": full_axis(_load_contract_axis(
            contract_dir / "v84_late_2023.npz"
        )),
        "full_2024": full_axis(_load_contract_axis(bridge_oof)),
    }
    parent = {name: axes[name]["parent"].astype(np.float64) for name in AXES}
    full_xgb = {
        year: align_regular_prediction(
            frames[year], fallback_oof_dir / f"hyunku_fallback_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    xgb = {
        "full_2022": full_xgb[2022],
        "late_2023": full_xgb[2023][late23],
        "full_2024": full_xgb[2024],
    }
    pressure = {name: pressure_gate(axis_frames[name]) for name in AXES}
    current: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for name in AXES:
        current[name], active[name] = apply_fallback(
            parent[name], xgb[name], pressure[name]
        )

    late22 = frames[2022]["game_month"].ge(8).to_numpy()
    early22 = ~late22
    late22_axis = slice_axis(axes["full_2022"], late22)
    source_results: dict[str, dict[str, Any]] = {}
    source_parameters: dict[str, dict[str, list[float]]] = {}
    locked_candidates: dict[str, np.ndarray] = {}
    locked_parameters: dict[str, list[float]] = {}
    for mode in MODES:
        early_parameters = fit_calibrator(
            parent["full_2022"], xgb["full_2022"],
            axes["full_2022"]["target"], active["full_2022"] & early22, mode,
        )
        full22_parameters = fit_calibrator(
            parent["full_2022"], xgb["full_2022"],
            axes["full_2022"]["target"], active["full_2022"], mode,
        )
        pooled_parameters = fit_calibrator(
            np.concatenate([parent["full_2022"], parent["late_2023"]]),
            np.concatenate([xgb["full_2022"], xgb["late_2023"]]),
            np.concatenate([
                axes["full_2022"]["target"], axes["late_2023"]["target"]
            ]),
            np.concatenate([active["full_2022"], active["late_2023"]]),
            mode,
        )
        for dose in DOSES:
            key = f"{mode}_d{dose:.2f}"
            late22_full, _ = calibrated_fallback(
                parent["full_2022"], xgb["full_2022"],
                pressure["full_2022"], early_parameters, dose,
            )
            late23_candidate, _ = calibrated_fallback(
                parent["late_2023"], xgb["late_2023"],
                pressure["late_2023"], full22_parameters, dose,
            )
            source_results[key] = {
                "late_2022": metrics(
                    late22_axis, current["full_2022"][late22], late22_full[late22]
                ),
                "late_2023": metrics(
                    axes["late_2023"], current["late_2023"], late23_candidate
                ),
            }
            source_parameters[key] = {
                "early_2022_fit": list(early_parameters),
                "full_2022_fit": list(full22_parameters),
            }
            locked_candidates[key], _ = calibrated_fallback(
                parent["full_2024"], xgb["full_2024"],
                pressure["full_2024"], pooled_parameters, dose,
            )
            locked_parameters[key] = list(pooled_parameters)

    selected = select_variant(source_results)
    locked = None
    point_pass = False
    robustness = None
    robust_pass = False
    if selected is not None:
        locked = metrics(
            axes["full_2024"], current["full_2024"], locked_candidates[selected]
        )
        point_pass = bool(
            locked["gain"] >= 1.0
            and locked["positive_month_fraction"] >= 0.625
            and locked["worst_month_gain"] > -5.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        family_keys = [key for key in source_results if source_gate(source_results[key])]
        family = [locked_candidates[key] for key in family_keys]
        family.append(current["full_2024"])
        robustness = _robustness(
            axes["full_2024"], current["full_2024"],
            locked_candidates[selected], active["full_2024"], family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
    confirm = bool(selected is not None and point_pass and robust_pass)
    if selected is not None:
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            current_full_2024=current["full_2024"],
            candidate_full_2024=locked_candidates[selected],
            active_full_2024=active["full_2024"],
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_release_fit" if confirm else (
            "source_reject" if selected is None else (
                "locked_point_reject" if not point_pass else "robust_reject"
            )
        ),
        "source_results": source_results,
        "source_parameters": source_parameters,
        "selected_variant_from_sources": selected,
        "locked_refit_parameters": None if selected is None else locked_parameters[selected],
        "locked_2024_incremental": locked,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_release_fit": confirm,
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--fallback-oof-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.fallback_oof_dir, args.contract_dir,
        args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected_variant_from_sources"],
        "locked": result["locked_2024_incremental"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

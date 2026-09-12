"""Audit a small independent swimmer blend above the promoted v244 parent.

The swimmer OOF is rebuilt from official training rows with strict season-forward
splits.  Candidate family and weight are selected on full-2022 and late-2023
only; full-2024 remains locked confirmation.  Public results are never used for
model or weight selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V246_SWIMMER_INDEPENDENT_RESIDUAL_BLEND_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
MODELS = ("logistic", "random_forest", "histgb", "rf_histgb_mean")
WEIGHTS = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)


def blend(parent: np.ndarray, independent: np.ndarray, weight: float) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    independent = np.asarray(independent, dtype=np.float64)
    if parent.shape != independent.shape:
        raise ValueError("blend arrays have different shapes")
    return np.clip(
        (1.0 - float(weight)) * parent + float(weight) * independent,
        0.001,
        0.999,
    )


def select_candidate(results: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, float] | None:
    """Choose the largest worst-source gain, then mean gain, then lower dose."""
    eligible: list[tuple[float, float, float, str]] = []
    for model in MODELS:
        for weight in WEIGHTS:
            key = f"{weight:g}"
            gains = [results[model][key][axis]["gain"] for axis in SOURCE_AXES]
            if all(gain > 0.0 for gain in gains):
                eligible.append((min(gains), float(np.mean(gains)), -weight, model))
    if not eligible:
        return None
    _minimum, _mean, negative_weight, model = max(eligible)
    return model, -negative_weight


def _load_axes(contract_dir: Path, bridge_oof: Path) -> dict[str, dict[str, np.ndarray]]:
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    return {
        name: {**axis, "exact_mask": np.ones(len(axis["target"]), dtype=bool)}
        for name, axis in axes.items()
    }


def _align_independent_oof(
    train_csv: Path,
    oof_csv: Path,
    axes: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    row_ids = pd.read_csv(train_csv, usecols=["row_id"])["row_id"].astype(str)
    oof = pd.read_csv(
        oof_csv,
        usecols=["row_id", "control_success", "validation_season", "model", "prediction"],
    )
    if oof.duplicated(["row_id", "model"]).any():
        raise ValueError("duplicate swimmer row/model predictions")
    target_counts = oof.groupby("row_id", sort=False)["control_success"].nunique()
    if int(target_counts.max()) != 1:
        raise ValueError("inconsistent swimmer target by row_id")

    wide = oof.pivot(index="row_id", columns="model", values="prediction")
    targets = oof.drop_duplicates("row_id").set_index("row_id")["control_success"]
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for axis_name, axis in axes.items():
        ids = row_ids.iloc[np.asarray(axis["raw_index"], dtype=np.int64)].to_numpy()
        selected = wide.reindex(ids)
        required = ("logistic", "random_forest", "histgb")
        if selected.loc[:, required].isna().any().any():
            raise ValueError(f"missing swimmer OOF rows for {axis_name}")
        aligned_target = targets.reindex(ids).to_numpy(dtype=np.float64)
        if not np.array_equal(aligned_target, np.asarray(axis["target"], dtype=np.float64)):
            raise ValueError(f"swimmer target mismatch for {axis_name}")
        rf = selected["random_forest"].to_numpy(dtype=np.float64)
        histgb = selected["histgb"].to_numpy(dtype=np.float64)
        aligned[axis_name] = {
            "logistic": selected["logistic"].to_numpy(dtype=np.float64),
            "random_forest": rf,
            "histgb": histgb,
            "rf_histgb_mean": 0.5 * (rf + histgb),
        }
    return aligned


def restrictions() -> dict[str, bool]:
    return {
        "v244_parent_frozen": True,
        "independent_codebase_and_model_family": True,
        "official_train_only_strict_forward_oof": True,
        "family_and_weight_selected_on_full_2022_and_late_2023_only": True,
        "full_2024_locked_from_selection": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    swimmer_oof_csv: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _load_axes(contract_dir, bridge_oof)
    independent = _align_independent_oof(train_csv, swimmer_oof_csv, axes)
    with np.load(v244_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"candidate_runtime_faithful_exact_jy_{axis}"].astype(np.float64)
            for axis in AXES
        }

    results: dict[str, dict[str, dict[str, Any]]] = {}
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    all_rows = {axis: np.ones(len(axes[axis]["target"]), dtype=bool) for axis in AXES}
    for model in MODELS:
        results[model] = {}
        for weight in WEIGHTS:
            key = f"{weight:g}"
            results[model][key] = {}
            for axis in AXES:
                candidate = blend(parents[axis], independent[axis][model], weight)
                predictions[(model, key, axis)] = candidate
                results[model][key][axis] = paired_metrics(
                    axes[axis], parents[axis], candidate, all_rows[axis]
                )

    selected = select_candidate(results)
    locked_pass = False
    robust_pass = False
    robustness = None
    if selected is not None:
        model, weight = selected
        key = f"{weight:g}"
        locked_pass = results[model][key]["full_2024"]["gain"] > 0.0
        family = [
            predictions[(model, f"{candidate_weight:g}", "full_2024")]
            for candidate_weight in WEIGHTS
        ] + [parents["full_2024"].copy()]
        robustness = _robustness(
            axes["full_2024"],
            parents["full_2024"],
            predictions[(model, key, "full_2024")],
            all_rows["full_2024"],
            family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] < 0.05
        )
        np.savez_compressed(
            output_dir / "selected_candidate_axes.npz",
            **{
                f"parent_{axis}": parents[axis]
                for axis in AXES
            },
            **{
                f"independent_{axis}": independent[axis][model]
                for axis in AXES
            },
            **{
                f"candidate_{axis}": predictions[(model, key, axis)]
                for axis in AXES
            },
        )

    summary = {
        "protocol": PROTOCOL,
        "status": (
            "robust_candidate"
            if selected is not None and locked_pass and robust_pass
            else "locked_or_robust_reject"
            if selected is not None
            else "source_reject"
        ),
        "models": list(MODELS),
        "weights": list(WEIGHTS),
        "selection_objective": "maximise worst source gain, then mean source gain, then prefer lower dose",
        "selected": None if selected is None else {"model": selected[0], "weight": selected[1]},
        "results": results,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": bool(selected is not None and locked_pass and robust_pass),
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
    parser.add_argument("--swimmer-oof-csv", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.swimmer_oof_csv,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    selected = result["selected"]
    print(json.dumps({
        "status": result["status"],
        "selected": selected,
        "source": None if selected is None else {
            axis: result["results"][selected["model"]][f"{selected['weight']:g}"][axis]
            for axis in SOURCE_AXES
        },
        "locked_2024": None if selected is None else result["results"]
        [selected["model"]][f"{selected['weight']:g}"]["full_2024"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

"""Rebase preserved independent EXP-021 strict OOF onto Public1175."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v168_jy_exact_contract_reaudit import (
    _load_year_context,
    metrics,
    rcore_mask,
)
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v205_h1_workload_strict_forward_audit import load_batter_ids_by_year
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.core.contract import _load_contract_axis


PROTOCOL = "V223_EXP021_STRICT_PUBLIC1175_REBASE_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
MODES = ("probability", "logit")
WEIGHTS = (0.005, 0.010, 0.020, 0.035, 0.050, 0.075, 0.100)
STRICT_NAME = "lowrank_s300_r6"


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.001, 0.999)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def blend_strict(
    parent: np.ndarray,
    challenger: np.ndarray,
    axis: dict[str, np.ndarray],
    mode: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    challenger = np.asarray(challenger, dtype=np.float64)
    if parent.shape != challenger.shape:
        raise ValueError("strict blend shape mismatch")
    active = rcore_mask(axis)
    if mode == "probability":
        proposal = parent + float(weight) * (challenger - parent)
    elif mode == "logit":
        proposal = _sigmoid(
            _logit(parent) + float(weight) * (_logit(challenger) - _logit(parent))
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    output = parent.copy()
    output[active] = np.clip(proposal[active], 0.001, 0.999)
    return output, active


def source_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= (2.0 / 3.0)
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
    )


def select_recipe(results: dict[str, dict[str, Any]]) -> str | None:
    eligible: list[tuple[float, float, float, str]] = []
    for key, axes in results.items():
        if all(source_gate(axes[name]) for name in SOURCE_AXES):
            gains = [axes[name]["gain"] for name in SOURCE_AXES]
            weight = float(key.rsplit("_w", 1)[1])
            eligible.append((min(gains), float(np.mean(gains)), -weight, key))
    return None if not eligible else max(eligible)[3]


def restrictions() -> dict[str, bool]:
    return {
        "independent_exp021_recipe_frozen": True,
        "strictly_prior_season_oof": True,
        "source_only_blend_selection": True,
        "locked_2024_not_used_for_selection": True,
        "public1175_fallback_formula_frozen": True,
        "test_csv_read": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    strict_dir: Path,
    fallback_oof_dir: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context, frames, correction = _load_year_context(train_csv)
    season = context["season"].to_numpy(np.int16)
    target = context["control_success"].to_numpy(np.float64)
    batter_ids = load_batter_ids_by_year(train_csv, season)
    for year in (2022, 2023, 2024):
        frames[year] = frames[year].assign(batter_id=batter_ids[year])
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes, frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    strict_full: dict[int, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for year in (2022, 2023, 2024):
        prediction_path = strict_dir / f"predictions_{STRICT_NAME}_{year}.npy"
        target_path = strict_dir / f"targets_{year}.npy"
        prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
        saved_target = np.load(target_path, allow_pickle=False).astype(np.float64)
        expected_target = target[season == year]
        if not np.array_equal(saved_target, expected_target):
            raise ValueError(f"strict target/order mismatch: {year}")
        if len(prediction) != len(expected_target) or not np.isfinite(prediction).all():
            raise ValueError(f"strict prediction contract failed: {year}")
        strict_full[year] = prediction
        provenance[str(year)] = {"rows": len(prediction), "path": str(prediction_path)}
    strict = {
        "full_2022": strict_full[2022],
        "late_2023": strict_full[2023][late23],
        "full_2024": strict_full[2024],
    }

    exact_results: dict[str, dict[str, Any]] = {}
    exact_candidates: dict[str, dict[str, np.ndarray]] = {}
    exact_active: dict[str, dict[str, np.ndarray]] = {}
    for mode in MODES:
        for weight in WEIGHTS:
            key = f"{mode}_w{weight:.3f}"
            exact_results[key], exact_candidates[key], exact_active[key] = {}, {}, {}
            for name in AXES:
                candidate, active = blend_strict(
                    parents[name], strict[name], axes[name], mode, weight
                )
                exact_candidates[key][name] = candidate
                exact_active[key][name] = active
                exact_results[key][name] = metrics(axes[name], parents[name], candidate)
    selected = select_recipe(exact_results)

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
    evidence_parent = {name: axes[name]["parent"].astype(np.float64) for name in AXES}
    pressure = {name: pressure_gate(axis_frames[name]) for name in AXES}
    public1175_proxy = {
        name: apply_fallback(evidence_parent[name], xgb[name], pressure[name])[0]
        for name in AXES
    }
    proxy_results: dict[str, dict[str, Any]] = {}
    proxy_candidates: dict[str, dict[str, np.ndarray]] = {}
    for key in exact_results:
        mode, weight_text = key.rsplit("_w", 1)
        weight = float(weight_text)
        proxy_results[key], proxy_candidates[key] = {}, {}
        for name in AXES:
            pre_fallback, _active = blend_strict(
                evidence_parent[name], strict[name], axes[name], mode, weight
            )
            candidate, fallback_active = apply_fallback(
                pre_fallback, xgb[name], pressure[name]
            )
            proxy_candidates[key][name] = candidate
            result = metrics(axes[name], public1175_proxy[name], candidate)
            result["fallback_active_rows"] = int(fallback_active.sum())
            proxy_results[key][name] = result

    locked = None if selected is None else exact_results[selected]["full_2024"]
    proxy_locked = None if selected is None else proxy_results[selected]["full_2024"]
    robustness = None
    point_pass = False
    robust_pass = False
    if selected is not None:
        point_pass = bool(
            locked["gain"] >= 2.5
            and locked["positive_month_fraction"] >= 0.75
            and locked["worst_month_gain"] > -5.0
            and locked["minimum_domain_gain"] >= 0.0
        )
        passing = [
            key for key, values in exact_results.items()
            if all(source_gate(values[name]) for name in SOURCE_AXES)
        ]
        family = [exact_candidates[key]["full_2024"] for key in passing]
        family.append(parents["full_2024"].copy())
        robustness = _robustness(
            axes["full_2024"], parents["full_2024"],
            exact_candidates[selected]["full_2024"],
            exact_active[selected]["full_2024"], family,
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
    proxy_pass = bool(
        proxy_locked is not None
        and proxy_locked["gain"] >= 2.5
        and proxy_locked["positive_month_fraction"] >= 0.75
        and proxy_locked["worst_month_gain"] > -5.0
    )
    confirm = bool(point_pass and robust_pass and proxy_pass)
    if selected is not None:
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            **{f"strict_{name}": strict[name] for name in AXES},
            **{f"exact_parent_{name}": parents[name] for name in AXES},
            **{f"exact_candidate_{name}": exact_candidates[selected][name] for name in AXES},
            **{f"public1175_proxy_{name}": public1175_proxy[name] for name in AXES},
            **{f"proxy_candidate_{name}": proxy_candidates[selected][name] for name in AXES},
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_full_fit" if confirm else (
            "source_reject" if selected is None else (
                "locked_point_reject" if not point_pass else (
                    "robust_reject" if not robust_pass else "proxy_effect_reject"
                )
            )
        ),
        "strict_provenance": provenance,
        "selected_recipe_from_sources": selected,
        "exact_results": exact_results,
        "locked_2024": locked,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "public1175_proxy_results": proxy_results,
        "public1175_proxy_locked_2024": proxy_locked,
        "public1175_proxy_effect_passed": proxy_pass,
        "eligible_for_full_fit": confirm,
        "eligible_for_packaging": False,
        "parity": parity,
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
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--fallback-oof-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.strict_dir, args.fallback_oof_dir,
        args.contract_dir, args.v104_path, args.h1_path, args.c3_path,
        args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected_recipe_from_sources"],
        "locked": result["locked_2024"],
        "proxy_locked": result["public1175_proxy_locked_2024"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

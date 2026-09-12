"""Strict-forward rolling logistic stack of runtime-faithful XGB and LightGBM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import align_regular_prediction
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics, route_masks
from src.archive.v248_fixed_route_dual_tree_fallback import AXES, SOURCE_AXES, compose
from src.core.contract import _load_contract_axis


PROTOCOL = "V250_ROLLING_DUAL_TREE_STACKER_V1"
FIRST_META_YEAR = 2021
LOGISTIC_C = 0.1


def logit_features(xgb: np.ndarray, lightgbm: np.ndarray) -> np.ndarray:
    xgb = np.clip(np.asarray(xgb, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    lightgbm = np.clip(
        np.asarray(lightgbm, dtype=np.float64), 1e-5, 1.0 - 1e-5
    )
    if xgb.shape != lightgbm.shape:
        raise ValueError("meta feature arrays have different shapes")
    return np.column_stack((np.log(xgb / (1.0 - xgb)), np.log(lightgbm / (1.0 - lightgbm))))


def fit_rolling_meta(
    audit_year: int,
    frames: dict[int, Any],
    xgb_full: dict[int, np.ndarray],
    lgbm_full: dict[int, np.ndarray],
) -> tuple[LogisticRegression, dict[str, Any]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    training_years = tuple(range(FIRST_META_YEAR, audit_year))
    if not training_years:
        raise ValueError(f"no prior OOF years for meta audit {audit_year}")
    for year in training_years:
        regular = frames[year]["game_type"].astype(str).eq("R").to_numpy()
        xgb = xgb_full[year][regular]
        lightgbm = lgbm_full[year][regular]
        if not (np.isfinite(xgb).all() and np.isfinite(lightgbm).all()):
            raise ValueError(f"missing regular meta OOF for {year}")
        features.append(logit_features(xgb, lightgbm))
        targets.append(
            frames[year].loc[regular, "control_success"].to_numpy(np.int8)
        )
        year_weight = 0.5 ** ((audit_year - 1 - year) / 2.0)
        weights.append(np.full(int(regular.sum()), year_weight, dtype=np.float64))
    x = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0)
    sample_weight = np.concatenate(weights, axis=0)
    model = LogisticRegression(
        C=LOGISTIC_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=300,
        random_state=2050,
    )
    model.fit(x, y, sample_weight=sample_weight)
    metadata = {
        "audit_year": audit_year,
        "training_years": list(training_years),
        "training_rows": int(len(y)),
        "coefficient_xgb_logit": float(model.coef_[0, 0]),
        "coefficient_lightgbm_logit": float(model.coef_[0, 1]),
        "intercept": float(model.intercept_[0]),
    }
    return model, metadata


def restrictions() -> dict[str, bool]:
    return {
        "v244_parent_routes_and_weights_frozen": True,
        "meta_training_uses_only_prior_year_oof_and_labels": True,
        "fixed_two_logit_l2_logistic_meta_model": True,
        "fixed_recency_weight_formula": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_fit_or_selection": False,
        "full_2024_seen_during_model_family_development": True,
        "clean_locked_holdout_claim": False,
    }


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


def _checkpoint(directory: Path, family: str, year: int) -> Path:
    name = (
        f"runtime_faithful_xgb_{year}.npy"
        if family == "xgb"
        else f"runtime_faithful_lgbm_{year}.npy"
    )
    return directory / name


def run(
    train_csv: Path,
    xgb_2021_dir: Path,
    xgb_main_dir: Path,
    lightgbm_2021_dir: Path,
    lightgbm_main_dir: Path,
    exact_axes: Path,
    v244_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _context, frames, _correction = _load_year_context(train_csv)
    axes = _load_axes(contract_dir, bridge_oof)
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    with np.load(exact_axes, allow_pickle=False) as saved:
        parents = {
            axis: saved[f"jy_parent_{axis}"].astype(np.float64)
            for axis in AXES
        }
    with np.load(v244_axes, allow_pickle=False) as saved:
        expected_v244 = {
            axis: saved[f"candidate_runtime_faithful_exact_parent_{axis}"].astype(np.float64)
            for axis in AXES
        }

    xgb_full: dict[int, np.ndarray] = {}
    lgbm_full: dict[int, np.ndarray] = {}
    for year in range(FIRST_META_YEAR, 2025):
        xgb_dir = xgb_2021_dir if year == 2021 else xgb_main_dir
        lgbm_dir = lightgbm_2021_dir if year == 2021 else lightgbm_main_dir
        xgb_full[year] = align_regular_prediction(
            frames[year], _checkpoint(xgb_dir, "xgb", year)
        )
        lgbm_full[year] = align_regular_prediction(
            frames[year], _checkpoint(lgbm_dir, "lightgbm", year)
        )

    meta_full: dict[int, np.ndarray] = {}
    meta_fits: dict[str, Any] = {}
    for audit_year in (2022, 2023, 2024):
        model, metadata = fit_rolling_meta(
            audit_year, frames, xgb_full, lgbm_full
        )
        regular = frames[audit_year]["game_type"].astype(str).eq("R").to_numpy()
        output = np.full(len(frames[audit_year]), np.nan, dtype=np.float64)
        output[regular] = model.predict_proba(
            logit_features(
                xgb_full[audit_year][regular], lgbm_full[audit_year][regular]
            )
        )[:, 1]
        meta_full[audit_year] = output
        meta_fits[str(audit_year)] = metadata

    xgb = {
        "full_2022": xgb_full[2022],
        "late_2023": xgb_full[2023][late23],
        "full_2024": xgb_full[2024],
    }
    meta = {
        "full_2022": meta_full[2022],
        "late_2023": meta_full[2023][late23],
        "full_2024": meta_full[2024],
    }
    baselines: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    results: dict[str, Any] = {}
    for axis in AXES:
        routes = route_masks(parents[axis], xgb[axis], axis_frames[axis])
        baselines[axis], active[axis] = compose(parents[axis], xgb[axis], routes)
        parity[axis] = float(np.max(np.abs(baselines[axis] - expected_v244[axis])))
        if parity[axis] > 1e-12:
            raise ValueError(f"v244 reconstruction mismatch on {axis}: {parity[axis]}")
        candidates[axis], candidate_active = compose(
            parents[axis], meta[axis], routes
        )
        if not np.array_equal(candidate_active, active[axis]):
            raise AssertionError("fixed route support changed")
        results[axis] = paired_metrics(
            axes[axis], baselines[axis], candidates[axis], active[axis]
        )

    source_pass = all(results[axis]["gain"] > 0.0 for axis in SOURCE_AXES)
    locked_pass = results["full_2024"]["gain"] > 0.0
    robustness = _robustness(
        axes["full_2024"],
        baselines["full_2024"],
        candidates["full_2024"],
        active["full_2024"],
        [candidates["full_2024"], baselines["full_2024"].copy()],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.05
    )
    eligible = bool(source_pass and locked_pass and robust_pass)
    np.savez_compressed(
        output_dir / "candidate_axes.npz",
        **{f"parent_{axis}": baselines[axis] for axis in AXES},
        **{f"candidate_{axis}": candidates[axis] for axis in AXES},
        **{f"active_{axis}": active[axis] for axis in AXES},
        **{f"meta_fallback_{axis}": meta[axis] for axis in AXES},
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_candidate" if eligible else "source_locked_or_robust_reject",
        "meta_fits": meta_fits,
        "v244_reconstruction_max_abs": parity,
        "results": results,
        "source_gate_passed": source_pass,
        "locked_point_passed": locked_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": eligible,
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
    parser.add_argument("--xgb-2021-dir", type=Path, required=True)
    parser.add_argument("--xgb-main-dir", type=Path, required=True)
    parser.add_argument("--lightgbm-2021-dir", type=Path, required=True)
    parser.add_argument("--lightgbm-main-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v244-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.xgb_2021_dir,
        args.xgb_main_dir,
        args.lightgbm_2021_dir,
        args.lightgbm_main_dir,
        args.exact_axes,
        args.v244_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "meta_fits": result["meta_fits"],
        "results": result["results"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

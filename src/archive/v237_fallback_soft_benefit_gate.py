"""Strict-forward ID-free soft benefit gate for the Public1175 fallback.

The gate can only shrink the incumbent 30% XGB edit toward its frozen JY
parent; it cannot expand support or exceed the submitted dose.  One fixed
ridge/soft-scale recipe is fit on prior OOF row benefits and evaluated at two
forward source transitions before full 2024 is opened.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.archive.v168_jy_exact_contract_reaudit import metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
    apply_fallback,
)
from src.archive.v219_public1175_evidence_transport_audit import pressure_gate
from src.archive.v224_fallback_xgb_complement_scope_audit import full_axis
from src.archive.v78_environment_stable_residual import build_features
from src.core.contract import _load_contract_axis


PROTOCOL = "V237_FALLBACK_SOFT_BENEFIT_GATE_V1"
RIDGE_ALPHA = 200.0
SOFT_SCALE_QUANTILE = 0.90
AXES = ("full_2022", "late_2023", "full_2024")


@dataclass
class GateModel:
    scaler: StandardScaler
    model: Ridge
    soft_scale: float


def gate_features(
    frame: pd.DataFrame,
    parent: np.ndarray,
    incumbent: np.ndarray,
    xgb: np.ndarray,
) -> pd.DataFrame:
    output = build_features(frame)
    parent = np.asarray(parent, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    xgb = np.asarray(xgb, dtype=np.float64)
    direction = incumbent - parent
    safe_xgb = np.where(np.isfinite(xgb), xgb, parent)
    output["parent_probability"] = parent
    output["xgb_probability"] = safe_xgb
    output["fallback_direction"] = direction
    output["fallback_abs_direction"] = np.abs(direction)
    output["direction_x_parent_centered"] = direction * (parent - 0.5)
    output["direction_x_runners"] = direction * output["runners_scaled"].to_numpy()
    output["direction_x_log_li"] = direction * output["log_li"].to_numpy()
    if not np.isfinite(output.to_numpy(np.float64)).all():
        raise ValueError("non-finite fallback gate feature")
    return output


def _month_weights(frame: pd.DataFrame, active: np.ndarray) -> np.ndarray:
    month = frame["game_month"].to_numpy(np.int16)
    output = np.zeros(len(frame), dtype=np.float64)
    for value in np.unique(month[active]):
        mask = active & (month == value)
        output[mask] = 1.0 / float(mask.sum())
    output *= float(active.sum()) / float(output.sum())
    return output


def fit_gate(
    features: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    incumbent: np.ndarray,
    active: np.ndarray,
    frame: pd.DataFrame,
) -> GateModel:
    active = np.asarray(active, dtype=bool)
    if not active.any():
        raise ValueError("fallback gate has no active fitting rows")
    benefit = (
        np.square(np.asarray(parent) - np.asarray(target))
        - np.square(np.asarray(incumbent) - np.asarray(target))
    ) * 1_000_000.0
    weight = _month_weights(frame, active)
    x = features.to_numpy(np.float64)
    scaler = StandardScaler().fit(x[active], sample_weight=weight[active])
    model = Ridge(alpha=RIDGE_ALPHA).fit(
        scaler.transform(x[active]), benefit[active], sample_weight=weight[active]
    )
    fitted = model.predict(scaler.transform(x[active]))
    positive = fitted[fitted > 0.0]
    scale = (
        float(np.quantile(positive, SOFT_SCALE_QUANTILE))
        if len(positive) else 1.0
    )
    return GateModel(scaler=scaler, model=model, soft_scale=max(scale, 1e-9))


def apply_gate(
    fitted: GateModel,
    features: pd.DataFrame,
    parent: np.ndarray,
    incumbent: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = fitted.model.predict(
        fitted.scaler.transform(features.to_numpy(np.float64))
    )
    gate = np.zeros(len(features), dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    gate[active] = np.clip(expected[active] / fitted.soft_scale, 0.0, 1.0)
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        output[active]
        + gate[active]
        * (np.asarray(incumbent, dtype=np.float64)[active] - output[active]),
        0.001,
        0.999,
    )
    # Outside submitted fallback support, incumbent and parent must be exact.
    output[~active] = np.asarray(incumbent, dtype=np.float64)[~active]
    return output, gate, expected


def point_gate(result: dict[str, Any], minimum_month_fraction: float) -> bool:
    return bool(
        result["gain"] > 0.0
        and result["positive_month_fraction"] >= float(minimum_month_fraction)
        and result["worst_month_gain"] > -5.0
        and result["minimum_domain_gain"] >= 0.0
    )


def restrictions() -> dict[str, bool]:
    return {
        "public1175_formula_is_upper_dose": True,
        "fallback_support_frozen": True,
        "gate_never_exceeds_submitted_30pct": True,
        "player_ids_excluded_from_gate": True,
        "one_fixed_ridge_recipe": True,
        "strict_forward_benefit_fit": True,
        "locked_2024_not_used_for_fitting_or_selection": True,
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
    raw = pd.read_csv(train_csv, low_memory=False)
    frames = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = {
        "full_2022": full_axis(_load_contract_axis(contract_dir / "v84_full_2022.npz")),
        "late_2023": full_axis(_load_contract_axis(contract_dir / "v84_late_2023.npz")),
        "full_2024": full_axis(_load_contract_axis(bridge_oof)),
    }
    parent = {name: axes[name]["parent"].astype(np.float64) for name in AXES}
    xgb_full = {
        year: align_regular_prediction(
            frames[year], fallback_oof_dir / f"hyunku_fallback_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    xgb = {
        "full_2022": xgb_full[2022],
        "late_2023": xgb_full[2023][late23],
        "full_2024": xgb_full[2024],
    }
    incumbent: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    features: dict[str, pd.DataFrame] = {}
    for name in AXES:
        incumbent[name], active[name] = apply_fallback(
            parent[name], xgb[name], pressure_gate(axis_frames[name])
        )
        features[name] = gate_features(
            axis_frames[name], parent[name], incumbent[name], xgb[name]
        )

    early22 = axis_frames["full_2022"]["game_month"].le(7).to_numpy()
    late22 = axis_frames["full_2022"]["game_month"].ge(8).to_numpy()
    model22 = fit_gate(
        features["full_2022"].loc[early22].reset_index(drop=True),
        axes["full_2022"]["target"][early22], parent["full_2022"][early22],
        incumbent["full_2022"][early22], active["full_2022"][early22],
        axis_frames["full_2022"].loc[early22].reset_index(drop=True),
    )
    candidate22, gate22, expected22 = apply_gate(
        model22, features["full_2022"].loc[late22].reset_index(drop=True),
        parent["full_2022"][late22], incumbent["full_2022"][late22],
        active["full_2022"][late22],
    )
    late22_axis = {
        key: np.asarray(value)[late22] for key, value in axes["full_2022"].items()
    }
    source22 = metrics(
        late22_axis, incumbent["full_2022"][late22], candidate22
    )

    model23 = fit_gate(
        features["full_2022"], axes["full_2022"]["target"],
        parent["full_2022"], incumbent["full_2022"], active["full_2022"],
        axis_frames["full_2022"],
    )
    candidate23, gate23, expected23 = apply_gate(
        model23, features["late_2023"], parent["late_2023"],
        incumbent["late_2023"], active["late_2023"],
    )
    source23 = metrics(
        axes["late_2023"], incumbent["late_2023"], candidate23
    )
    source_pass = bool(
        point_gate(source22, 2.0 / 3.0) and point_gate(source23, 2.0 / 3.0)
    )

    joined_features = pd.concat(
        [features["full_2022"], features["late_2023"]], ignore_index=True
    )
    joined_frame = pd.concat(
        [axis_frames["full_2022"], axis_frames["late_2023"]], ignore_index=True
    )
    final_model = fit_gate(
        joined_features,
        np.concatenate([
            axes["full_2022"]["target"], axes["late_2023"]["target"]
        ]),
        np.concatenate([parent["full_2022"], parent["late_2023"]]),
        np.concatenate([incumbent["full_2022"], incumbent["late_2023"]]),
        np.concatenate([active["full_2022"], active["late_2023"]]),
        joined_frame,
    )
    candidate24, gate24, expected24 = apply_gate(
        final_model, features["full_2024"], parent["full_2024"],
        incumbent["full_2024"], active["full_2024"],
    )
    locked = metrics(
        axes["full_2024"], incumbent["full_2024"], candidate24
    )
    point_pass = point_gate(locked, 0.625)
    robustness = _robustness(
        axes["full_2024"], incumbent["full_2024"], candidate24,
        active["full_2024"], [candidate24, incumbent["full_2024"]],
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] <= 0.10
    )
    confirm = bool(source_pass and point_pass and robust_pass)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        incumbent_full_2024=incumbent["full_2024"],
        candidate_full_2024=candidate24,
        active_full_2024=active["full_2024"],
        gate_full_2024=gate24,
        expected_benefit_full_2024=expected24,
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_release_build" if confirm else (
            "source_reject" if not source_pass else (
                "locked_point_reject" if not point_pass else "robust_reject"
            )
        ),
        "model": {
            "ridge_alpha": RIDGE_ALPHA,
            "soft_scale_quantile": SOFT_SCALE_QUANTILE,
            "feature_count": int(features["full_2022"].shape[1]),
        },
        "source": {
            "early22_to_late22": source22,
            "full22_to_late23": source23,
        },
        "source_gate_passed": source_pass,
        "locked_2024_incremental": locked,
        "locked_point_passed": point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "gate_audit": {
            "late22_mean_active": float(np.mean(gate22[active["full_2022"][late22]])),
            "late23_mean_active": float(np.mean(gate23[active["late_2023"]])),
            "full24_mean_active": float(np.mean(gate24[active["full_2024"]])),
            "full24_zero_fraction_active": float(np.mean(gate24[active["full_2024"]] == 0.0)),
            "full24_one_fraction_active": float(np.mean(gate24[active["full_2024"]] == 1.0)),
        },
        "eligible_for_release_build": confirm,
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
        "source": result["source"],
        "locked": result["locked_2024_incremental"],
        "gate_audit": result["gate_audit"],
        "robustness": result["robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

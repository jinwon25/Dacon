"""Metric-aligned H1 challenger trained with squared-error CatBoost.

The deployed H1 classifier optimizes log loss even though the competition
metric is a monotone transform of Brier loss.  This experiment keeps its exact
features, temporal folds, tree budget, and row-local inference contract, but
fits a CatBoost regressor with RMSE.  Full-2022 and late-2023 alone select a
small mixture into the deployed H1 component.  Full-2024 is fitted and opened
only when a source candidate passes the predeclared gate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.archive.v168_jy_exact_contract_reaudit import (
    _load_year_context,
    _locked_contract,
    _source_contracts,
    affine,
    c3_mix,
    compose,
    gate_library,
    overwrite_gate,
)
from src.archive.v173_h1_noncore_extension_audit import paired_metrics
from src.champion.v131_hoo_h1_independent_oof import CAT_COLS, _prepare_features
from src.core.contract import _load_contract_axis
from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    one_way_cluster_bootstrap,
    white_reality_check,
)


PROTOCOL = "V174_BRIER_H1_REGRESSOR_V1"
ALPHAS = (0.25, 0.50, 0.75, 1.0)
SOURCE_AXES = ("full_2022", "late_2023")


def mix_h1(current: np.ndarray, challenger: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(
        (1.0 - float(alpha)) * np.asarray(current, dtype=np.float64)
        + float(alpha) * np.asarray(challenger, dtype=np.float64),
        0.001,
        0.999,
    )


def _pipeline(features: list[str], seed: int) -> Pipeline:
    categorical = [column for column in CAT_COLS if column in features]
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", "passthrough", numeric),
        ]
    )
    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=100.0,
        border_count=32,
        loss_function="RMSE",
        random_seed=int(seed),
        verbose=0,
        thread_count=16,
        allow_writing_files=False,
    )
    return Pipeline([("pre", preprocessor), ("reg", model)])


def _fit_year(
    train: pd.DataFrame,
    target: np.ndarray,
    season: np.ndarray,
    year: int,
    features: list[str],
    checkpoint: Path,
) -> np.ndarray:
    if checkpoint.exists():
        values = np.load(checkpoint, allow_pickle=False).astype(np.float64)
        if values.shape != (int(np.sum(season == year)),):
            raise ValueError(f"invalid checkpoint shape: {checkpoint}")
        print(f"[v174] resumed {checkpoint.name}", flush=True)
        return values
    fit = season < year
    audit = season == year
    print(
        f"[v174] fit RMSE-H1 year={year} train={int(fit.sum()):,} "
        f"audit={int(audit.sum()):,}",
        flush=True,
    )
    model = _pipeline(features, 17442)
    model.fit(train.loc[fit, features], target[fit])
    values = np.clip(
        np.asarray(model.predict(train.loc[audit, features]), dtype=np.float64),
        0.001,
        0.999,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint, values, allow_pickle=False)
    del model
    gc.collect()
    return values


def _source_base(
    values: dict[str, Any], axis: dict[str, np.ndarray]
) -> np.ndarray:
    return compose(
        values["component"], values["h1"],
        c3_mix(values["sign"], values["recent"], 0.15),
        axis, h1_weight=0.15,
    )


def _source_candidate(
    values: dict[str, Any],
    axis: dict[str, np.ndarray],
    challenger_h1: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _source_base(values, axis)
    candidate = compose(
        values["component"], mix_h1(values["h1"], challenger_h1, alpha),
        c3_mix(values["sign"], values["recent"], 0.15),
        axis, h1_weight=0.15,
    )
    active = np.asarray(axis["exact_mask"], dtype=bool) & (
        np.asarray(axis["domain3"]).astype(str) == "R_CORE"
    )
    return base, candidate, active


def _jy_prediction(
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    locked: dict[str, np.ndarray],
    h1: np.ndarray,
    bridge_oof: Path,
) -> np.ndarray:
    with np.load(bridge_oof, allow_pickle=False) as saved:
        bridge025_top = saved["v142_full_2024"].astype(np.float64) + 0.25 * (
            saved["v138_full_2024"].astype(np.float64)
            - saved["v142_full_2024"].astype(np.float64)
        )
        historical_parent = saved["parent"].astype(np.float64)
    component_delta = (bridge025_top - historical_parent) / 0.85
    bridge_component = locked["component"] + 1.2 * component_delta
    main = compose(
        locked["component"], h1, locked["c3_base"], axis, h1_weight=0.15
    )
    proposal = compose(
        bridge_component, h1, locked["c3_active"], axis, h1_weight=0.16
    )
    output, _active = overwrite_gate(
        main, proposal, axis, frame, gate_library()["runners_or_high_li"]
    )
    return output


def _robustness(
    axis: dict[str, np.ndarray],
    base: np.ndarray,
    candidate: np.ndarray,
    family: list[np.ndarray],
) -> dict[str, Any]:
    exact = np.asarray(axis["exact_mask"], dtype=bool)
    target = np.asarray(axis["target"], dtype=np.float64)[exact]
    incumbent = np.asarray(base, dtype=np.float64)[exact]
    trial = np.asarray(candidate, dtype=np.float64)[exact]
    improvement = np.column_stack([
        np.square(incumbent - target)
        - np.square(np.asarray(item, dtype=np.float64)[exact] - target)
        for item in family
    ])
    return {
        "pitcher": one_way_cluster_bootstrap(
            target, trial, incumbent, np.asarray(axis["pitcher_id"])[exact],
            n_resamples=2000, seed=17442,
        ),
        "crossed_pitcher_batter": crossed_pigeonhole_bootstrap(
            target, trial, incumbent,
            np.asarray(axis["pitcher_id"])[exact],
            np.asarray(axis["batter_id"])[exact],
            n_resamples=2000, seed=17443,
        ),
        "chronological_block": circular_block_bootstrap(
            target, trial, incumbent, block_size=4096,
            n_resamples=2000, seed=17444,
        ),
        "reality_check": white_reality_check(
            target, improvement, block_size=4096,
            n_resamples=2000, seed=17445,
        ),
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    external_root: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, target, season, _base_features, _dx, h1_features = _prepare_features(
        train_csv, trackman_csv, external_root
    )
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    raw22 = _fit_year(
        train, target, season, 2022, h1_features,
        output_dir / "checkpoints" / "rmse_h1_2022_seed17442.npy",
    )
    raw23 = _fit_year(
        train, target, season, 2023, h1_features,
        output_dir / "checkpoints" / "rmse_h1_2023_seed17442.npy",
    )
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    challenger = {
        "full_2022": affine(raw22 + correction[2022]),
        "late_2023": affine((raw23 + correction[2023])[late23]),
    }

    rows = []
    details = {}
    source_arrays = {}
    for alpha in ALPHAS:
        key = f"rmse_h1_mix_a{alpha:g}"
        per_axis = {}
        arrays = {}
        for name in SOURCE_AXES:
            base, candidate, active = _source_candidate(
                source[name], axes[name], challenger[name], alpha
            )
            per_axis[name] = paired_metrics(
                axes[name], base, candidate, active
            )
            arrays[name] = candidate
        passed = all(
            per_axis[name]["overall_gain"] > 0.0
            and per_axis[name]["positive_month_fraction"] >= 0.75
            and per_axis[name]["worst_month_gain"] > -5.0
            for name in SOURCE_AXES
        )
        rows.append({
            "key": key,
            "alpha": alpha,
            "source_gate_passed": passed,
            "source_min_gain": min(
                per_axis[name]["overall_gain"] for name in SOURCE_AXES
            ),
            "source_mean_gain": float(np.mean([
                per_axis[name]["overall_gain"] for name in SOURCE_AXES
            ])),
            "source_worst_month": min(
                per_axis[name]["worst_month_gain"] for name in SOURCE_AXES
            ),
        })
        details[key] = per_axis
        source_arrays[alpha] = arrays
    ranking = pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "source_min_gain", "source_mean_gain"],
        ascending=False, kind="stable",
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_alpha = float(selected["alpha"])

    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "source_reject",
        "model": {
            "kind": "CatBoostRegressor",
            "loss": "RMSE",
            "iterations": 1200,
            "depth": 8,
            "learning_rate": 0.02,
            "l2_leaf_reg": 100.0,
            "border_count": 32,
            "seed": 17442,
        },
        "selected": selected,
        "source_details": details,
        "locked_details": None,
        "locked_robustness": None,
        "eligible_for_packaging": False,
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "leaderboard_score_used_for_selection": False,
            "full_2024_fit_only_after_source_gate": True,
            "official_train_only_for_fitting": True,
            "row_local_inference": True,
        },
    }
    ranking.to_csv(output_dir / "source_ranking.csv", index=False, encoding="utf-8-sig")

    if bool(selected["source_gate_passed"]):
        raw24 = _fit_year(
            train, target, season, 2024, h1_features,
            output_dir / "checkpoints" / "rmse_h1_2024_seed17442.npy",
        )
        locked, parity = _locked_contract(
            axes["full_2024"], frames[2024], correction[2024],
            h1_path, c3_path, v160_path,
        )
        challenger24 = affine(raw24 + correction[2024])
        official = _jy_prediction(
            axes["full_2024"], frames[2024], locked, locked["h1"], bridge_oof
        )
        family = []
        locked_details = {}
        selected_candidate = None
        active = np.asarray(axes["full_2024"]["exact_mask"], dtype=bool) & (
            np.asarray(axes["full_2024"]["domain3"]).astype(str) == "R_CORE"
        )
        for alpha in ALPHAS:
            mixed = mix_h1(locked["h1"], challenger24, alpha)
            candidate = _jy_prediction(
                axes["full_2024"], frames[2024], locked, mixed, bridge_oof
            )
            family.append(candidate)
            locked_details[f"rmse_h1_mix_a{alpha:g}"] = paired_metrics(
                axes["full_2024"], official, candidate, active
            )
            if alpha == selected_alpha:
                selected_candidate = candidate
        assert selected_candidate is not None
        robust = _robustness(
            axes["full_2024"], official, selected_candidate, family
        )
        locked_metric = locked_details[f"rmse_h1_mix_a{selected_alpha:g}"]
        robust_pass = bool(
            robust["pitcher"]["p05"] > 0.0
            and robust["crossed_pitcher_batter"]["p05"] > 0.0
            and robust["chronological_block"]["p05"] > 0.0
            and robust["reality_check"]["p_value"] < 0.10
        )
        promote = bool(locked_metric["overall_gain"] > 0.0 and robust_pass)
        result.update({
            "status": "promote_to_release_build" if promote else "locked_reject",
            "parity": parity,
            "locked_details": locked_details,
            "locked_robustness": robust,
            "eligible_for_packaging": promote,
        })
        np.savez_compressed(
            output_dir / "selected_axes.npz",
            full_2022=source_arrays[selected_alpha]["full_2022"],
            late_2023=source_arrays[selected_alpha]["late_2023"],
            full_2024=selected_candidate,
        )

    del train
    gc.collect()
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.trackman_csv, args.external_root,
        args.contract_dir, args.v104_path, args.h1_path, args.c3_path,
        args.v160_path, args.bridge_oof, args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "selected": result["selected"],
        "source": result["source_details"][result["selected"]["key"]],
        "locked": result["locked_details"],
        "robustness": result["locked_robustness"],
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

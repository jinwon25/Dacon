"""Correct the one-pitch truncation in deployable success-rate anchors.

v279/v285 deliberately anchored every cumulative statistic at the final
*pre-pitch* state of prior history.  That is necessary for failure categories
whose final-pitch class is unavailable, but not for success: official train
contains ``control_success`` for that final pitch.  Pitcher and batter success
anchors can therefore be advanced to the exact completed-history state.

The trained fallback XGB models, v290 parent, route masks, and route weights
remain frozen.  Only success-derived runtime columns are recomputed.  The same
semantic correction is evaluated on full 2022, late 2023, and full 2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.archive.v168_row_region_exact_contract_reaudit import _load_year_context
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    align_regular_prediction,
)
from src.archive.v241_mechanism_aware_fallback_expansion import (
    paired_metrics,
    route_masks,
)
from src.archive.v248_fixed_route_dual_tree_fallback import AXES, compose, _load_axes


PROTOCOL = "V306_COMPLETED_SUCCESS_ANCHOR_AUDIT_V1"
TARGET = "control_success"
SOURCE_AXES = ("full_2022", "late_2023")
SUCCESS_SPECS = (
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "p_succ"),
    ("batter_id", "asof_batter_n", "asof_batter_success_rate", "b_succ"),
)


def completed_success_anchors(
    history: pd.DataFrame,
    id_column: str,
    n_column: str,
    rate_column: str,
) -> dict[int, tuple[float, float]]:
    """Return exact completed count/success sum at the end of history."""

    values = history[[id_column, n_column, rate_column, TARGET]].copy()
    values[n_column] = pd.to_numeric(values[n_column], errors="coerce")
    values[rate_column] = pd.to_numeric(values[rate_column], errors="coerce")
    latest_index = values.groupby(id_column, sort=False)[n_column].idxmax()
    latest = values.loc[latest_index].copy()
    latest["completed_n"] = latest[n_column] + 1.0
    latest["completed_sum"] = (
        latest[n_column] * latest[rate_column].fillna(0.0)
        + latest[TARGET].astype(float)
    )
    return {
        int(row[0]): (float(row[1]), float(row[2]))
        for row in latest[[id_column, "completed_n", "completed_sum"]].itertuples(
            index=False, name=None
        )
    }


def patch_success_features(
    audit: pd.DataFrame,
    history: pd.DataFrame,
    features: pd.DataFrame,
    priors: dict[str, float],
) -> pd.DataFrame:
    """Patch only columns derived from pitcher/batter cumulative success."""

    output = features.copy()
    state: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for id_column, n_column, rate_column, prefix in SUCCESS_SPECS:
        anchors = completed_success_anchors(
            history, id_column, n_column, rate_column
        )
        ids = audit[id_column].astype(int).to_numpy()
        pair = [anchors.get(int(value), (0.0, 0.0)) for value in ids]
        anchor_n = np.asarray([item[0] for item in pair], dtype=np.float64)
        anchor_sum = np.asarray([item[1] for item in pair], dtype=np.float64)
        n = pd.to_numeric(audit[n_column], errors="coerce").to_numpy(np.float64)
        rate = pd.to_numeric(audit[rate_column], errors="coerce").to_numpy(
            np.float64
        )
        prior = float(priors[prefix])
        delta_n = np.maximum(np.nan_to_num(n, nan=0.0) - anchor_n, 0.0)
        delta_sum = np.maximum(
            np.nan_to_num(n * rate, nan=0.0) - anchor_sum, 0.0
        )
        season_rate = (delta_sum + 150.0 * prior) / (delta_n + 150.0)
        output[f"{prefix}_ssn"] = season_rate.astype(np.float32)
        output[f"{prefix}_ssn_vs_car"] = (
            season_rate - np.nan_to_num(rate, nan=prior)
        ).astype(np.float32)
        output[f"{prefix}_ssn_n"] = delta_n.astype(np.float32)
        for shrink in (25, 75, 400, 1000):
            output[f"{prefix}_k{shrink}"] = (
                (delta_sum + shrink * prior) / (delta_n + shrink)
            ).astype(np.float32)
        state[prefix] = (delta_n, delta_sum)

    if "p_ppa" in output:
        ppa = output["p_ppa"].to_numpy(np.float64)
        pitcher_n = state["p_succ"][0]
        output["p_est_apps"] = (pitcher_n / np.clip(ppa, 5.0, None)).astype(
            np.float32
        )
        output["p_ssn_per_month"] = (
            pitcher_n
            / np.clip(audit["game_month"].to_numpy(np.float64), 3.0, None)
        ).astype(np.float32)
    return output.reindex(columns=features.columns).astype(np.float32)


def build_predictions(
    train: pd.DataFrame,
    old_feature_dir: Path,
    exact_model_dir: Path,
    output_dir: Path,
) -> dict[int, np.ndarray]:
    predictions: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        checkpoint = output_dir / f"completed_success_xgb_{year}.npy"
        if checkpoint.exists():
            predictions[year] = np.load(checkpoint, allow_pickle=False).astype(
                np.float64
            )
            continue
        history = train.loc[train["season"].lt(year)].reset_index(drop=True)
        audit = train.loc[
            train["season"].eq(year)
            & train["game_type"].astype(str).eq("R")
        ].reset_index(drop=True)
        features = pd.read_parquet(
            old_feature_dir / f"exact_runtime_features_{year}.parquet"
        )
        lookup = joblib.load(
            old_feature_dir / f"lookup_{year}" / "fallback_lookups.joblib"
        )
        patched = patch_success_features(
            audit, history, features, lookup["priors"]
        )
        model = xgb.XGBClassifier()
        model.load_model(
            exact_model_dir / f"training_parity_exact_xgb_{year}.json"
        )
        prediction = model.predict_proba(patched)[:, 1].astype(np.float64)
        np.save(checkpoint, prediction.astype(np.float32), allow_pickle=False)
        predictions[year] = prediction
    return predictions


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "only_success_derived_anchor_features_changed": True,
        "failure_category_anchors_unchanged": True,
        "fallback_xgb_models_frozen": True,
        "v290_parent_routes_and_route_weights_frozen": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    old_feature_dir: Path,
    exact_model_dir: Path,
    stale_oof_dir: Path,
    exact_axes: Path,
    v285_axes: Path,
    v290_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    completed_regular = build_predictions(
        train, old_feature_dir, exact_model_dir, output_dir
    )
    _context, frames, _correction = _load_year_context(train_csv)
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = _load_axes(contract_dir, bridge_oof)
    with np.load(exact_axes, allow_pickle=False) as saved:
        jy_parent = {
            axis: saved[f"jy_parent_{axis}"].astype(np.float64) for axis in AXES
        }
    stale_full = {
        year: align_regular_prediction(
            frames[year], stale_oof_dir / f"runtime_faithful_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    old_exact_full = {
        year: align_regular_prediction(
            frames[year], exact_model_dir / f"training_parity_exact_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    completed_full = {
        year: align_regular_prediction(
            frames[year], output_dir / f"completed_success_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    stale = {
        "full_2022": stale_full[2022],
        "late_2023": stale_full[2023][late23],
        "full_2024": stale_full[2024],
    }
    old_exact = {
        "full_2022": old_exact_full[2022],
        "late_2023": old_exact_full[2023][late23],
        "full_2024": old_exact_full[2024],
    }
    completed = {
        "full_2022": completed_full[2022],
        "late_2023": completed_full[2023][late23],
        "full_2024": completed_full[2024],
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        top_parent = {
            "full_2022": saved["candidate_full_2022"].astype(np.float64)
        }
    with np.load(v290_axes, allow_pickle=False) as saved:
        top_parent.update(
            {
                "late_2023": saved["candidate_late_2023"].astype(np.float64),
                "full_2024": saved["candidate_full_2024"].astype(np.float64),
            }
        )

    candidate: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    parity: dict[str, float] = {}
    details: dict[str, Any] = {}
    for axis in AXES:
        routes = route_masks(
            jy_parent[axis], stale[axis], axis_frames[axis]
        )
        old_output, active[axis] = compose(
            jy_parent[axis], old_exact[axis], routes
        )
        new_output, new_active = compose(
            jy_parent[axis], completed[axis], routes
        )
        if not np.array_equal(active[axis], new_active):
            raise AssertionError("completed anchor changed route support")
        parity[axis] = float(
            np.max(np.abs(top_parent[axis][active[axis]] - old_output[active[axis]]))
        )
        if parity[axis] > 1e-12:
            raise ValueError(f"v290 route parity mismatch: {axis} {parity[axis]}")
        candidate[axis] = np.clip(
            top_parent[axis] + (new_output - old_output), 0.001, 0.999
        )
        details[axis] = paired_metrics(
            axes[axis], top_parent[axis], candidate[axis], active[axis]
        )

    source_pass = bool(all(details[axis]["gain"] > 0.0 for axis in SOURCE_AXES))
    locked = details["full_2024"]
    robustness = _robustness(
        axes["full_2024"], top_parent["full_2024"], candidate["full_2024"],
        active["full_2024"], [top_parent["full_2024"], candidate["full_2024"]],
    )
    locked_pass = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -3.0
    )
    np.savez_compressed(
        output_dir / "completed_success_axes.npz",
        **{f"parent_{axis}": top_parent[axis] for axis in AXES},
        **{f"candidate_{axis}": candidate[axis] for axis in AXES},
        **{f"active_{axis}": active[axis] for axis in AXES},
    )
    result = {
        "protocol": PROTOCOL,
        "status": "semantic_candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "metrics": details,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "locked_robustness": robustness,
        "route_parity_max_abs": parity,
        "eligible_for_packaging": bool(source_pass and locked_pass),
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--old-feature-dir", type=Path, required=True)
    parser.add_argument("--exact-model-dir", type=Path, required=True)
    parser.add_argument("--stale-oof-dir", type=Path, required=True)
    parser.add_argument("--exact-axes", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.old_feature_dir, args.exact_model_dir,
        args.stale_oof_dir, args.exact_axes, args.v285_axes, args.v290_axes,
        args.contract_dir, args.bridge_oof, args.output_dir,
    ), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

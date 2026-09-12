"""Fresh V2-FROZEN-R1 and V2-NESTED-R1 outer predictions.

Only the engineered LightGBM + official RandomForest components are rebuilt;
Trackman and legacy cache artifacts are not reused.  All iteration and offset
selection for V2-NESTED-R1 is performed on prior inner years.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.calibration import apply_logit_offset
from src.data import TARGET_COL, read_main
from src.metrics import brier_score
from src.nested_v2 import OFFICIAL_FEATURES, VARIANT, _base_lgb_params
from src.train import FeatureBuilder, train_rf_holdout

ROUND_GRID = (50, 54, 100, 150, 200, 250, 300, 350)
FIXED_ROUNDS = 54
FIXED_OFFSET = -0.09579592585412458


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest().upper()


def fit_lgb_grid(train: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray) -> dict[int, np.ndarray]:
    builder = FeatureBuilder(feature_set="engineered")
    y = train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.int8)
    builder.fit(train.iloc[train_idx], y)
    x_train = builder.transform(train.iloc[train_idx])
    x_valid = builder.transform(train.iloc[valid_idx])
    cats = [c for c in builder.categorical_columns if c in x_train.columns]
    dtrain = lgb.Dataset(x_train, label=y, categorical_feature=cats, free_raw_data=True)
    booster = lgb.train(_base_lgb_params(VARIANT), dtrain, num_boost_round=max(ROUND_GRID), callbacks=[lgb.log_evaluation(0)])
    out = {int(r): np.asarray(booster.predict(x_valid, num_iteration=int(r)), dtype=np.float64) for r in ROUND_GRID}
    del builder, x_train, x_valid, dtrain, booster
    gc.collect()
    return out


def choose_offset(pred: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(pred, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    grid = np.linspace(-0.25, 0.25, 1001)
    scores = [float(np.mean((1 / (1 + np.exp(-np.clip(logits + x, -40, 40))) - target) ** 2)) for x in grid]
    return float(grid[int(np.argmin(scores))])


def weighted_median(values: list[int], weights: list[float]) -> int:
    order = np.argsort(values)
    values = [values[i] for i in order]
    weights = [weights[i] for i in order]
    total = sum(weights)
    cum = 0.0
    for v, w in zip(values, weights):
        cum += w
        if cum >= total / 2:
            return int(v)
    return int(values[-1])


def fit_lgb_fixed(train: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray, rounds: int) -> np.ndarray:
    return fit_lgb_grid(train, train_idx, valid_idx)[int(rounds)]


def inner_select(train: pd.DataFrame, inner_years: list[int]) -> tuple[int, float, list[dict]]:
    selected_rounds: list[int] = []
    offsets: list[float] = []
    details = []
    weights = []
    for rank, year in enumerate(inner_years, start=1):
        ti = np.flatnonzero(train["season"].to_numpy() < year)
        vi = np.flatnonzero(train["season"].to_numpy() == year)
        grid = fit_lgb_grid(train, ti, vi)
        rf = train_rf_holdout(train, ti, vi, OFFICIAL_FEATURES)
        rf_pred = np.asarray(rf["prediction"], dtype=np.float64)
        target = train.iloc[vi][TARGET_COL].to_numpy(dtype=np.float64)
        best = None
        for rounds, lgb_pred in grid.items():
            blend = 0.35 * lgb_pred + 0.65 * rf_pred
            offset = choose_offset(blend, target)
            cal = apply_logit_offset(blend, offset)
            score = brier_score(target, cal)
            if best is None or score < best[0]:
                best = (score, int(rounds), float(offset))
        assert best is not None
        score, rounds, offset = best
        selected_rounds.append(rounds); offsets.append(offset); weights.append(float(rank))
        details.append({"inner_validation_season":year,"selected_iteration":rounds,"offset":offset,"brier":score,"n_rows":len(target),"train_max_season":year-1,"outer_target_used_for_selection":False})
        del grid, rf
        gc.collect()
    rounds = weighted_median(selected_rounds, weights)
    offset = float(np.average(offsets, weights=weights))
    return rounds, offset, details


def run(project: Path, years=(2021, 2022, 2023), run_id="v2_r1_oof_20260809_01", data_project: Path | None = None) -> pd.DataFrame:
    project = project.resolve(); started = time.perf_counter()
    data_project = (data_project or project).resolve()
    out_dir = project / "artifacts" / "top1100" / "v2_r1_oof" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    train = read_main(data_project / "data" / "train.csv")
    rows = []
    for year in years:
        outer_train = np.flatnonzero(train["season"].to_numpy() < year)
        outer_valid = np.flatnonzero(train["season"].to_numpy() == year)
        target = train.iloc[outer_valid][TARGET_COL].to_numpy(dtype=np.float64)
        nested_rounds, nested_offset, inner = inner_select(train, list(range(2020, year)))
        frozen_lgb = fit_lgb_fixed(train, outer_train, outer_valid, FIXED_ROUNDS)
        nested_lgb = fit_lgb_fixed(train, outer_train, outer_valid, nested_rounds)
        rf = train_rf_holdout(train, outer_train, outer_valid, OFFICIAL_FEATURES)
        rf_pred = np.asarray(rf["prediction"], dtype=np.float64)
        frozen = np.clip(0.35 * apply_logit_offset(frozen_lgb, FIXED_OFFSET) + 0.65 * apply_logit_offset(rf_pred, FIXED_OFFSET), 1e-6, 1-1e-6)
        nested = np.clip(0.35 * apply_logit_offset(nested_lgb, nested_offset) + 0.65 * apply_logit_offset(rf_pred, nested_offset), 1e-6, 1-1e-6)
        valid = train.iloc[outer_valid].reset_index(drop=True)
        np.savez_compressed(out_dir / f"v2_r1_o{year}.npz", row_id=valid["row_id"].astype(str).to_numpy(), season=valid["season"].to_numpy(np.int16), target=target, p_v2_frozen=frozen, p_v2_nested=nested)
        for name, pred, rounds, offset in (("V2-FROZEN-R1",frozen,FIXED_ROUNDS,FIXED_OFFSET),("V2-NESTED-R1",nested,nested_rounds,nested_offset)):
            rows.append({"recipe_id":name,"outer_validation_season":year,"train_max_season":year-1,"n_rows":len(target),"brier":float(brier_score(target,pred)),"prediction_sha256":sha_array(pred),"row_id_sha256":sha_array(valid["row_id"].astype(str).to_numpy().astype('U')),"target_sha256":sha_array(target),"selected_iteration":rounds,"selected_offset":offset,"inner_details":json.dumps(inner),"trackman_used":False,"outer_target_used_for_selection":False})
        print(f"[v2-r1] outer={year} nested_rounds={nested_rounds} offset={nested_offset:.6f} frozen={brier_score(target,frozen):.9f} nested={brier_score(target,nested):.9f}", flush=True)
        del frozen_lgb, nested_lgb, rf, rf_pred, frozen, nested
        gc.collect()
    out = pd.DataFrame(rows); out.to_csv(project / "reports" / "top1100" / "v2_r1_oof_metrics.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({"run_id":run_id,"outer_years":list(years),"runtime_seconds":time.perf_counter()-started,"frozen_definition":"engineered LGB+official RF, fixed rounds 54 and fixed package offset; no Trackman","nested_definition":"engineered LGB+official RF, inner Brier selected rounds/offset; no Trackman","outer_target_used_for_selection":False}, indent=2), encoding="utf-8")
    print(out[["recipe_id","outer_validation_season","brier","selected_iteration","selected_offset"]].to_string(index=False), flush=True); return out


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--project-dir",type=Path,default=Path(".")); ap.add_argument("--data-project",type=Path); ap.add_argument("--years",default="2021,2022,2023"); ap.add_argument("--run-id",default="v2_r1_oof_20260809_01"); args=ap.parse_args(); run(args.project_dir,years=tuple(int(x) for x in args.years.split(",") if x),run_id=args.run_id,data_project=args.data_project)


if __name__ == "__main__": main()

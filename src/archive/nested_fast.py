"""Fast strict nested OOF using pre-recorded inner selections.

The inner cache is used only for the immediately preceding season's
iteration/offset choice. Outer LightGBM is then fit with that fixed iteration
count and outer target is never passed into training callbacks.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.calibration import apply_logit_offset
from src.archive.data import TARGET_COL, read_main
from src.metrics import brier_score
from src.archive.nested_v2 import OFFICIAL_FEATURES, _fit_fixed_lgb, _recency_weights, _select_offset
from src.archive.train import train_rf_holdout
from src.archive.validation import walk_forward_splits


def run(project: Path) -> pd.DataFrame:
    train = read_main(project / "data/train.csv")
    years = (2021, 2022, 2023, 2024)
    folds = {fold.validation_season: fold for fold in walk_forward_splits(train, validation_seasons=years)}
    rows = []; saved = {}
    for year in years:
        fold = folds[year]
        inner_path = project / f"artifacts/followup/oof/wave0_incumbent_validate_{year-1}.npz"
        with np.load(inner_path, allow_pickle=False) as z:
            inner_offset = _select_offset(z["incumbent"].astype(float), z["target"].astype(float))
            rounds = int(z["lgb_best_iteration"][0])
        print(f"[nested-fast] outer={year} fixed_rounds={rounds}", flush=True)
        lgb_raw = _fit_fixed_lgb(train, fold.train_idx, fold.valid_idx, rounds)
        rf = train_rf_holdout(train, fold.train_idx, fold.valid_idx, OFFICIAL_FEATURES)
        rec = train_rf_holdout(train, fold.train_idx, fold.valid_idx, OFFICIAL_FEATURES, sample_weight=_recency_weights(train.iloc[fold.train_idx]["season"].to_numpy(), year))
        target = train.iloc[fold.valid_idx][TARGET_COL].to_numpy(dtype=float)
        lgb_cal = apply_logit_offset(lgb_raw, inner_offset)
        a = np.clip(0.35*lgb_cal + 0.65*apply_logit_offset(np.asarray(rf["prediction"], dtype=float), inner_offset), 1e-4, 1-1e-4)
        b = np.clip(0.35*lgb_cal + 0.65*apply_logit_offset(np.asarray(rec["prediction"], dtype=float), inner_offset), 1e-4, 1-1e-4)
        saved.update({f"{year}_target": target, f"{year}_valid_idx": fold.valid_idx.astype(np.int64), f"{year}_A_original": a, f"{year}_B_recency": b})
        for arm, pred in (("A_original", a), ("B_R_recency", b)):
            rows.append({"baseline":"v2_nested", "arm":arm, "outer_validation_season":year, "n_rows":len(target), "brier":brier_score(target,pred), "delta_vs_A":0.0 if arm=="A_original" else brier_score(target,pred)-brier_score(target,a), "inner_validation_season":year-1, "selected_lgb_iterations":rounds, "selected_offset":inner_offset, "outer_target_used_for_selection":False, "prediction_source":"strict fixed outer fit using prior inner cache selection"})
        del rf, rec, lgb_raw, lgb_cal
        gc.collect()
    out = pd.DataFrame(rows); out.to_csv(project / "reports/champion_v2_nested_results.csv", index=False); np.savez_compressed(project / "artifacts/followup/v2_nested_predictions.npz", **saved)
    (project / "artifacts/followup/v2_nested_provenance.json").write_text('{"primary_status":"PASS_BUILDER","outer_target_used_for_selection":false,"selection":"prior inner cache only","trackman":"not included; C/D remain frozen diagnostic"}', encoding="utf-8")
    print(out.to_string(index=False)); return out


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--project-dir",type=Path,default=Path(".")); args=parser.parse_args(); run(args.project_dir.resolve())


if __name__=="__main__": main()

"""Fresh wave0 OOF only; original training routines, no cached model loading."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import read_main, ID_COL
from src.followup import _generate_wave0_fold
from src.validation import walk_forward_splits

ROOT = Path(__file__).resolve().parent
FIELDS = ("valid_idx", "target", "lgb_raw", "rf_raw", "blend_raw",
          "lgb_trend", "rf_trend", "incumbent", "offset", "forecast_rate",
          "trend_window", "lgb_best_iteration")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-oof-dir", type=Path)
    parser.add_argument("--years", type=int, nargs="+", default=[2020,2021,2022,2023,2024])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh directory; cached OOF must not substitute for training")
    args.output_dir.mkdir(parents=True)
    config = json.loads((ROOT / "configs/followup.json").read_text(encoding="utf-8"))
    train = read_main(args.data_dir / "train.csv")
    # Header only, as in run_wave0; no evaluation values or distribution read.
    features = [c for c in pd.read_csv(args.data_dir / "test.csv",nrows=0).columns if c != ID_COL]
    results = []
    for fold in walk_forward_splits(train,validation_seasons=tuple(args.years)):
        year = fold.validation_season
        output = _generate_wave0_fold(args.output_dir,train,fold.train_idx,fold.valid_idx,year,config,features)
        report = {"year":year,"fit_rows":len(fold.train_idx),"validation_rows":len(fold.valid_idx)}
        if args.reference_oof_dir:
            with np.load(args.reference_oof_dir / f"wave0_incumbent_validate_{year}.npz",allow_pickle=False) as old:
                errors = {key:float(np.max(np.abs(output[key].astype(float)-old[key].astype(float)))) for key in FIELDS}
            report.update({"max_abs_by_field":errors,"strict_parity":max(errors.values()) <= 1e-12})
        results.append(report)
        summary = {"protocol":"WAVE0_FRESH_FIT_COMPARISON_V1","folds":results,
                   "requested_years":args.years,"all_requested_completed":len(results)==len(args.years),
                   "reference_oof_used_for_training":False,"private_score_recomputed":False}
        (args.output_dir / "wave0_fresh_fit.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2),flush=True)
        if report.get("strict_parity") is False:
            raise ValueError("Fresh OOF differs from historical OOF; stop before downstream fitting")

if __name__ == "__main__":
    main()

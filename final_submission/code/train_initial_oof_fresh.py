"""Fresh initial V2 nested OOF; duplicate fits share only this run's in-memory predictions."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from train import sha256, DATA_HASHES
from src import v2_r1_oof as source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-project",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--reference-root",type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; no cached fits may be imported")
    if sha256(args.data_project/"data/train.csv") != DATA_HASHES["train.csv"]:
        raise ValueError("Official train mismatch")
    versions = {k:importlib.metadata.version(k) for k in ("numpy","pandas","lightgbm","scikit-learn")}
    expected = {"numpy":"2.2.6","pandas":"2.2.3","lightgbm":"4.6.0","scikit-learn":"1.6.1"}
    if versions != expected:
        raise ValueError("Use the pinned initial OOF environment")
    (args.output_dir/"reports/top1100").mkdir(parents=True)
    lgb_cache, rf_cache, fits = {}, {}, []
    original_lgb, original_rf = source.fit_lgb_grid, source.train_rf_holdout
    def key(train,ti,vi):
        seasons = train["season"].to_numpy()
        years = np.unique(seasons[vi])
        if len(years) != 1:
            raise ValueError("Expected a single complete validation season")
        year = int(years[0])
        if not np.array_equal(ti,np.flatnonzero(seasons < year)) or not np.array_equal(vi,np.flatnonzero(seasons == year)):
            raise ValueError("Refusing to reuse a nonidentical fold")
        return year
    def lgb_cached(train,ti,vi):
        year = key(train,ti,vi)
        if year not in lgb_cache:
            started = time.perf_counter()
            lgb_cache[year] = original_lgb(train,ti,vi)
            fits.append({"model":"LightGBM","validation_year":year,"train_rows":len(ti),
                         "validation_rows":len(vi),"seconds":time.perf_counter()-started})
            print("New initial LightGBM fold "+str(year),flush=True)
        return lgb_cache[year]
    def rf_cached(train,ti,vi,features):
        year = key(train,ti,vi)
        if list(features) != list(source.OFFICIAL_FEATURES):
            raise ValueError("Unexpected RF features")
        if year not in rf_cache:
            started = time.perf_counter()
            fitted = original_rf(train,ti,vi,features)
            rf_cache[year] = {"prediction":np.asarray(fitted["prediction"]).copy()}
            fits.append({"model":"RandomForest","validation_year":year,"train_rows":len(ti),
                         "validation_rows":len(vi),"seconds":time.perf_counter()-started})
            print("New initial RF fold "+str(year),flush=True)
        return rf_cache[year]
    source.fit_lgb_grid, source.train_rf_holdout = lgb_cached, rf_cached
    try:
        source.run(args.output_dir,years=(2021,2022,2023,2024),run_id="fresh",
                   data_project=args.data_project)
    finally:
        source.fit_lgb_grid, source.train_rf_holdout = original_lgb, original_rf
    report = {"fresh_training_complete":True,"versions":versions,"fits":fits,
              "cached_fits_from_other_runs_used":False,"reference_used_for_training":False,
              "train_sha256":DATA_HASHES["train.csv"],"folds":{},
              "private_score_recomputed":False}
    for year in (2021,2022,2023,2024):
        path = args.output_dir/"artifacts/top1100/v2_r1_oof/fresh"/f"v2_r1_o{year}.npz"
        record = {"sha256":sha256(path)}
        if args.reference_root:
            matches = list(args.reference_root.rglob(f"v2_r1_o{year}.npz"))
            if len(matches) != 1:
                raise ValueError("Expected one numeric reference OOF per year")
            with np.load(matches[0],allow_pickle=False) as old, np.load(path,allow_pickle=False) as new:
                record["comparison"] = {}
                for field in ("target","season","p_v2_frozen","p_v2_nested"):
                    if old[field].shape != new[field].shape:
                        raise ValueError("OOF row shape differs")
                    record["comparison"][field] = float(np.max(np.abs(old[field]-new[field])))
                # Object-typed row IDs are deliberately not deserialized.
            record["reference_sha256"] = sha256(matches[0])
        report["folds"][str(year)] = record
    report["numeric_reference_parity"] = (
        all(max(v["comparison"].values()) <= 1e-12 for v in report["folds"].values())
        if args.reference_root else None)
    (args.output_dir/"initial_oof_fresh_fit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    if report["numeric_reference_parity"] is False:
        raise ValueError("Fresh initial OOF differs; preserve the reference")


if __name__ == "__main__":
    main()


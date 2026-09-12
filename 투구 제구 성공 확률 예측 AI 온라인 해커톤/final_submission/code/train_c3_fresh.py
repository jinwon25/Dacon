"""Fresh five-fold H1 residual OOF and C3 tables, with fixed recipe and reference hash."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np

from train import DATA_HASHES, FINAL_SHA256, sha256
from src.champion.v131_catboost_h1_independent_oof import _prepare_features, _fit_year
from src.champion.v142_build_submission_package import build_c3_bundle

ROOT = Path(__file__).resolve().parent


def compare(left, right, path="root"):
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("C3 dictionary mismatch: " + path)
        return max((compare(left[k], right[k], path + "/" + str(k)) for k in left), default=0.)
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("C3 list mismatch: " + path)
        return max((compare(a,b,path) for a,b in zip(left,right)), default=0.)
    if isinstance(left, (float, np.floating)):
        if not np.isfinite(left) or not np.isfinite(right):
            raise ValueError("C3 contains a non-finite value: " + path)
        return abs(float(left)-float(right))
    if left != right:
        raise ValueError("C3 value mismatch: " + path)
    return 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-dir","reference-zip","output-dir"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--oof-path", type=Path,
                        help="Explicitly reuse OOF for table-only refit; this mode does not prove OOF training.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    if sha256(args.reference_zip) != FINAL_SHA256:
        raise ValueError("Only the original submitted reference ZIP is allowed")
    for name,expected in DATA_HASHES.items():
        if sha256(args.data_dir/name) != expected:
            raise ValueError("Official input mismatch: " + name)
    if __import__("catboost").__version__ != "1.2.8":
        raise ValueError("Use CatBoost 1.2.8")
    args.output_dir.mkdir(parents=True)
    recipe = json.loads((ROOT/"configs/v131_catboost_h1_independent_oof.json").read_text(encoding="utf-8"))["model"]
    table_recipe = json.loads((ROOT/"configs/v148_v142_v138_blend_package.json").read_text(encoding="utf-8"))["c3"]
    oof_path = args.oof_path or args.output_dir/"h1_oof.npz"
    if args.oof_path is None:
        frame, target, season, _base, _dx, features = _prepare_features(
            args.data_dir/"train.csv",args.data_dir/"trackman_history.csv",ROOT/"h1")
        predictions = {}
        for year in range(2020,2025):
            predictions[f"h1_{year}"] = _fit_year(frame,target,season,year,features,recipe,"C3-fresh")
            np.savez_compressed(oof_path,**predictions)
        del frame, target, season
    with np.load(oof_path, allow_pickle=False) as saved:
        for year in range(2020,2025):
            value = saved[f"h1_{year}"]
            if value.ndim != 1 or not np.all(np.isfinite(value) & (value >= 0) & (value <= 1)):
                raise ValueError("Invalid H1 OOF probabilities")
    fresh = build_c3_bundle(args.data_dir/"train.csv",oof_path,table_recipe)
    with zipfile.ZipFile(args.reference_zip) as archive:
        reference = joblib.load(io.BytesIO(archive.read("model/c3_sign_all.joblib")))
    maximum = compare(reference,fresh)
    joblib.dump(fresh,args.output_dir/"c3_sign_all.joblib",compress=3)
    result = {
        "strict_fit_parity":bool(np.isfinite(maximum) and maximum <= 1e-12),
        "maximum_table_abs_difference":maximum,
        "oof_years":list(range(2020,2025)),
        "oof_sha256":sha256(oof_path),
        "oof_fresh_training_proven_by_this_command":args.oof_path is None,
        "oof_source":("Five new CatBoost fits using official earlier-season training rows."
                      if args.oof_path is None else "Explicit table-only refit from the supplied OOF; verify its training separately."),
        "official_data_sha256":DATA_HASHES,
        "reference_inference_sha256":FINAL_SHA256,
        "fixed_h1_recipe":recipe,"fixed_c3_recipe":table_recipe,
        "reference_inference_modified":False,"private_score_recomputed":False
    }
    (args.output_dir/"c3_fresh_fit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    if not result["strict_fit_parity"]:
        raise ValueError("New C3 tables differ; preserve original inference")


if __name__ == "__main__":
    main()

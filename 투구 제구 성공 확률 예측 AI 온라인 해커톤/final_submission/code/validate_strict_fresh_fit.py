"""Compare freshly fitted strict native assets against the submitted v345 component."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd

from train import sha256, FINAL_SHA256, DATA_HASHES

PREFIX = "model/v124/model/parent/parent/strict/"
FILES = ("history_state.json", "multirate_state.json", "feature_schemas.json",
         "histgradientboosting.json", "group_effects.json", "team_effects.json",
         "pitcher_count_effects.json", "lowrank_effects.json")


def compare_native(left, right, tolerance=1e-12):
    errors = []
    maximum = 0.
    def walk(a, b, path):
        nonlocal maximum
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                errors.append(path + ":keys")
            for key in a.keys() & b.keys():
                walk(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(path + ":length")
            for index, (x,y) in enumerate(zip(a,b)):
                walk(x,y,path + "/" + str(index))
        elif isinstance(a, (int,float)) and isinstance(b, (int,float)):
            if a == b:
                return
            difference = abs(float(a)-float(b))
            if not math.isfinite(difference):
                errors.append(path + ":nonfinite")
            else:
                maximum = max(maximum,difference)
                if difference > tolerance:
                    errors.append(path + ":numeric")
        elif a != b:
            errors.append(path + ":value")
    walk(left,right,"root")
    return {"within_tolerance":not errors, "maximum_numeric_abs_difference":maximum,
            "mismatch_count":len(errors), "first_mismatches":sorted(errors)[:12]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference-zip","candidate-zip","train-csv","test-csv","output-dir"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh validation directory")
    if sha256(args.reference_zip) != FINAL_SHA256:
        raise ValueError("Unrecognized inference reference")
    if sha256(args.train_csv) != DATA_HASHES["train.csv"]:
        raise ValueError("Official training data mismatch")
    # Frozen endpoints are only valid for subsequent seasons with sufficient
    # cumulative counts. Historical training rows do not satisfy that contract.
    sample = pd.read_csv(args.test_csv)
    reports = {}
    predictions = []
    with zipfile.ZipFile(args.reference_zip) as original, zipfile.ZipFile(args.candidate_zip) as fresh:
        for name in FILES:
            reports[name] = compare_native(json.loads(original.read(PREFIX+name)),
                                           json.loads(fresh.read("model/"+name)))
        old_lgb = lgb.Booster(model_str=original.read(PREFIX+"rfull_lightgbm.txt").decode("utf-8"))
        new_lgb = lgb.Booster(model_str=fresh.read("model/rfull_lightgbm.txt").decode("utf-8"))
        reports["rfull_lightgbm.txt"] = compare_native(old_lgb.dump_model(),new_lgb.dump_model())
        old_meta = json.loads(original.read(PREFIX+"metadata.json"))
        new_meta = json.loads(fresh.read("model/metadata.json"))
        if old_meta["candidate"] != new_meta["candidate"]:
            raise ValueError("Different strict candidate")
        # Always use the hash-pinned original inference code for both sets of assets.
        runtime = original.read("model/v124/model/parent/parent/components/strict_script.py")
        with tempfile.TemporaryDirectory(prefix="strict-parity-") as temporary:
            root = Path(temporary)
            script_path = root/"runtime.py"
            script_path.write_bytes(runtime)
            sample.to_csv(root/"test.csv",index=False,encoding="utf-8")
            sample[["row_id"]].assign(control_success=0.).to_csv(root/"sample.csv",index=False)
            for label,archive,prefix in (("original",original,PREFIX),("fresh",fresh,"model/")):
                model_dir = root/label
                model_dir.mkdir()
                for name in (*FILES,"rfull_lightgbm.txt","metadata.json"):
                    (model_dir/name).write_bytes(archive.read(prefix+name))
                spec = importlib.util.spec_from_file_location("strict_"+label,script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.MODEL_DIR = model_dir
                module.TEST_PATH = root/"test.csv"
                module.SAMPLE_PATH = root/"sample.csv"
                module.OUTPUT_PATH = root/(label+".csv")
                module.main()
                predictions.append(pd.read_csv(module.OUTPUT_PATH)["control_success"].to_numpy())
    maximum = float(np.max(np.abs(predictions[0]-predictions[1])))
    result = {"reference_inference_sha256":FINAL_SHA256,
              "candidate_zip_sha256":sha256(args.candidate_zip),
              "native_assets":reports,"rows_compared":len(sample),
              "prediction_fixture":"Official public sample, not hidden test; complete native model/state comparison is reported separately.",
              "prediction_max_abs":maximum,
              "strict_fit_parity":bool(np.isfinite(maximum) and maximum <= 1e-12
                                      and all(r["within_tolerance"] for r in reports.values())),
              "whole_parent_ensemble_fresh_fit":False,"private_score_recomputed":False,
              "metadata_note":"Timing/version annotations excluded; candidate identity checked."}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"strict_fresh_fit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    if not result["strict_fit_parity"]:
        raise ValueError("Strict fresh fit differs; preserve the original inference component")


if __name__ == "__main__":
    main()

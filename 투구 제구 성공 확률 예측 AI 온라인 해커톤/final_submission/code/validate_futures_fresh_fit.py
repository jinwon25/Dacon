"""Compare newly fitted native CatBoost futures models, without loading pickle."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from src.champion.v289_finalize_recent_futures_expert import SEEDS
from src.archive.v287_recent_futures_direct_expert import build_features

REFERENCE_SHA256 = "d44578dc50220ce84dd4b8489bbae680afdcf93f931ed4236287b5a9e6f5aaaa"

def compare(reference_zip, candidate_dir, train_csv, output_dir):
    if output_dir.exists():
        raise FileExistsError("Use a new output directory")
    with reference_zip.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != REFERENCE_SHA256:
            raise ValueError("Not the submitted reference archive")
    raw = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    sample = raw.loc[raw.season.eq(2024) & raw.game_type.eq("F")].reset_index(drop=True)
    features = build_features(sample)
    results = []
    with tempfile.TemporaryDirectory(prefix="futures-fit-parity-") as temp, zipfile.ZipFile(reference_zip) as archive:
        stage = Path(temp)
        for seed in SEEDS:
            name = f"futures_expert_seed{seed}.cbm"
            old_path = stage / name
            old_path.write_bytes(archive.read("model/recent_futures/" + name))
            old, new = CatBoostClassifier(), CatBoostClassifier()
            old.load_model(old_path)
            new.load_model(candidate_dir / name)
            delta = old.predict_proba(features)[:,1] - new.predict_proba(features)[:,1]
            old.save_model(stage / "old.json", format="json")
            new.save_model(stage / "new.json", format="json")
            a = json.loads((stage / "old.json").read_text(encoding="utf-8"))
            b = json.loads((stage / "new.json").read_text(encoding="utf-8"))
            fields = ("oblivious_trees", "features_info", "scale_and_bias", "ctr_data")
            equal = {key: a.get(key) == b.get(key) for key in fields}
            results.append({"seed":seed,"model_sections_equal":equal,
                            "prediction_max_abs":float(np.abs(delta).max())})
            print(f"seed={seed} max_abs={results[-1]['prediction_max_abs']}",flush=True)
    report = {"rows_compared":len(sample),"sample":"all official 2024 F training rows",
              "models":results,"strict_fit_parity":all(all(r["model_sections_equal"].values())
                  and r["prediction_max_abs"] <= 1e-12 for r in results),
              "selection_reestimated":False,"private_score_recomputed":False,
              "reference_inference_modified":False}
    output_dir.mkdir(parents=True)
    (output_dir / "futures_fresh_fit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["strict_fit_parity"]:
        raise ValueError("New fit differs; retain original inference models")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-zip","candidate-dir","train-csv","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    compare(args.reference_zip,args.candidate_dir,args.train_csv,args.output_dir)

if __name__ == "__main__":
    main()

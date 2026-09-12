"""Compare a CPU/GPU fresh XGBoost fit on fixed inference features; never deploy it."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new validation directory")
    spec = importlib.util.spec_from_file_location("frozen", ROOT / "fallback_xgb_frozen_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = pd.read_csv(args.train_csv, encoding="utf-8-sig", low_memory=False)
    sample = raw.iloc[np.linspace(0, len(raw)-1, min(len(raw),32768), dtype=int)].copy()
    sample = pd.concat([sample,pd.read_csv(args.test_csv,encoding="utf-8-sig")],ignore_index=True)
    features = module.build(sample,ROOT)
    original, candidate = xgb.XGBClassifier(), xgb.XGBClassifier()
    original.load_model(ROOT / "fallback_xgb.json")
    candidate.load_model(args.model)
    old_p, new_p = original.predict_proba(features)[:,1], candidate.predict_proba(features)[:,1]
    old_json = json.loads((ROOT / "fallback_xgb.json").read_text(encoding="utf-8"))
    new_json = json.loads(args.model.read_text(encoding="utf-8"))
    a = old_json["learner"]["gradient_booster"]["model"]["trees"]
    b = new_json["learner"]["gradient_booster"]["model"]["trees"]
    delta = new_p.astype(float)-old_p.astype(float)
    report = {
        "rows_compared":len(sample),"xgboost_version":xgb.__version__,
        "reference_trees":len(a),"candidate_trees":len(b),
        "identical_trees":sum(x==y for x,y in zip(a,b)),
        "prediction_max_abs":float(np.max(np.abs(delta))),
        "prediction_rms":float(np.sqrt(np.mean(delta**2))),
        "exact_fit_parity":a==b and bool(np.array_equal(old_p,new_p)),
        "private_score_recomputed":False,"reference_inference_modified":False,
        "scope":"Evenly sampled train rows plus public format test; both models use the same frozen inference features. This does not measure Private Score or isolate the cause of a mismatch."
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "xgb_fresh_fit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()

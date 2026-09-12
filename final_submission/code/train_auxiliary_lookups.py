"""Fit auxiliary lookups from supplied OOF and compare safe NPZ arrays."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from train import sha256, FINAL_SHA256
from src.champion.v319_finalize_futures_lowrank import run as lowrank_fit
from src.champion.v334_finalize_player_transition import run as transition_fit

def compare_arrays(old, new):
    if set(old.files) != set(new.files):
        raise ValueError("Array keys differ")
    report = {}
    for key in old.files:
        a, b = old[key], new[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Array shape/dtype differs: " + key)
        exact = bool(np.array_equal(a,b))
        error = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.dtype.kind in "fiu" else (0. if exact else None)
        report[key] = {"exact":exact,"max_abs":error}
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-csv","oof-dir","reference-zip","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    if sha256(args.reference_zip) != FINAL_SHA256:
        raise ValueError("Not the submitted reference ZIP")
    args.output_dir.mkdir(parents=True)
    lowrank_fit(args.train_csv,args.oof_dir,args.output_dir/"lowrank")
    transition_fit(args.train_csv,args.oof_dir,args.output_dir/"transition")
    results = {}
    with zipfile.ZipFile(args.reference_zip) as archive:
        for key,member,candidate in (
            ("lowrank","model/futures_lowrank/lookup.npz","lowrank/futures_lowrank_lookup.npz"),
            ("transition","model/player_transition/lookup.npz","transition/player_transition_lookup.npz"),
        ):
            with np.load(io.BytesIO(archive.read(member)),allow_pickle=False) as old, np.load(args.output_dir/candidate,allow_pickle=False) as new:
                results[key] = compare_arrays(old,new)
    report = {"arrays":results,"all_arrays_exact":all(r["exact"] for group in results.values() for r in group.values()),
              "all_arrays_within_1e12":all(r["max_abs"] is not None and r["max_abs"] <= 1e-12 for group in results.values() for r in group.values()),
              "source_oof_sha256":{str(y):sha256(args.oof_dir/f"wave0_incumbent_validate_{y}.npz") for y in range(2020,2025)},
              "oof_fresh_training_proven_by_this_command":False,
              "reference_inference_modified":False,"private_score_recomputed":False}
    (args.output_dir/"lookup_refit_parity.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["all_arrays_within_1e12"]:
        raise ValueError("Lookup refit differs; preserve original inference")

if __name__ == "__main__":
    main()

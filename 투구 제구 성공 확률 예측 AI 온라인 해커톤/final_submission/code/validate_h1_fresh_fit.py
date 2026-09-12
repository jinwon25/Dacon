"""Compare a freshly trained H1 bundle with the saved H1 model, without replacing it."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import zipfile

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def prepared(module, raw, bundle, workload_transform=None):
    frame = module.attach_ctx(raw.copy(), bundle)
    if any(c in bundle["features"] for c in module.CAAFE_COLS):
        frame = module.attach_caafe(frame)
    if any(c in bundle["features"] for c in module.ASOF_COLS):
        frame = module.attach_asof_state(frame, bundle)
    if workload_transform is not None:
        return module.build_features(workload_transform(frame), bundle)
    return module.build_features(module.attach_aux(frame, bundle), bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-reference-zip", type=Path,
                        help="Compare workload H1 against the hash-pinned submitted v345 ZIP.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new validation directory")
    workload_transform = None
    reference_member = "model/rf.pkl"
    report_name = "h1_fresh_fit.json"
    if args.workload_reference_zip:
        from train import FINAL_SHA256
        from src.champion.v343_build_transition_workload_package import WORKLOAD_FUNCTIONS
        reference_zip = args.workload_reference_zip
        expected_hash = FINAL_SHA256
        reference_member = "model/workload_h1/model/rf.pkl"
        report_name = "workload_fresh_fit.json"
        # These are the audited local builder's function definitions, not downloaded code.
        namespace = {"np": np, "pd": pd}
        exec(compile(WORKLOAD_FUNCTIONS, "v343_workload_functions", "exec"), namespace)
        workload_transform = namespace["_attach_workload_features"]
    else:
        recipe = json.loads((ROOT / "h1/training_recipe.json").read_text(encoding="utf-8"))
        reference_zip = ROOT / "lineage_inputs/cand_asof_xl.zip"
        expected_hash = recipe["archive_sha256"]
    with reference_zip.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected_hash:
            raise ValueError("Unrecognized reference H1 archive")
    with zipfile.ZipFile(reference_zip) as archive:
        original = joblib.load(io.BytesIO(archive.read(reference_member)))
    candidate = joblib.load(args.model)
    spec = importlib.util.spec_from_file_location("h1_runtime", ROOT / "h1/script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = pd.read_csv(args.train_csv, encoding="utf-8-sig", low_memory=False)
    sample = raw.iloc[np.linspace(0, len(raw) - 1, min(len(raw), 32768), dtype=int)].copy()
    if args.test_csv:
        sample = pd.concat([sample, pd.read_csv(args.test_csv)], ignore_index=True)
    else:
        sample = sample.reset_index(drop=True)
    old_x = prepared(module, sample, original, workload_transform)
    new_x = prepared(module, sample, candidate, workload_transform)
    input_equal = old_x.equals(new_x)
    old_p = np.asarray(module.predict_proba(original, old_x), dtype=float)
    new_p = np.asarray(module.predict_proba(candidate, new_x), dtype=float)
    model_reports = []
    with tempfile.TemporaryDirectory(prefix="h1-model-compare-") as temporary:
        for index, (old, new) in enumerate(zip(original["models"], candidate["models"])):
            old_path, new_path = Path(temporary) / "old.json", Path(temporary) / "new.json"
            old.steps[-1][1].save_model(str(old_path), format="json")
            new.steps[-1][1].save_model(str(new_path), format="json")
            a = json.loads(old_path.read_text(encoding="utf-8"))
            b = json.loads(new_path.read_text(encoding="utf-8"))
            model_reports.append({
                "seed": 42 + index,
                "trees_equal": a["oblivious_trees"] == b["oblivious_trees"],
                "borders_equal": a["features_info"] == b["features_info"],
                "scale_and_bias_equal": a["scale_and_bias"] == b["scale_and_bias"],
                "leaf_values_max_abs": float(np.max(np.abs(
                    old.steps[-1][1].get_leaf_values() - new.steps[-1][1].get_leaf_values()))),
            })
    max_abs = float(np.max(np.abs(old_p - new_p)))
    strict = (len(original["models"]) == len(candidate["models"]) == 3
              and original["features"] == candidate["features"] and input_equal
              and max_abs <= 1e-12 and all(r["trees_equal"] and r["borders_equal"]
                  and r["scale_and_bias_equal"] for r in model_reports))
    with args.model.open("rb") as handle:
        candidate_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"rows_compared": len(sample), "features_equal": input_equal,
              "reference_archive_sha256": expected_hash, "reference_member": reference_member,
              "fresh_bundle_sha256": candidate_hash,
              "prediction_max_abs": max_abs, "models": model_reports,
              "strict_fit_parity": strict, "private_score_recomputed": False}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / report_name).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not strict:
        raise ValueError("Fresh H1 fit differs; keep the original inference bundle")


if __name__ == "__main__":
    main()

"""Full-fit workload H1 using only the hash-pinned submitted template."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from train import sha256, FINAL_SHA256, DATA_HASHES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--reference-zip",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    if sha256(args.reference_zip) != FINAL_SHA256:
        raise ValueError("Refusing to deserialize any archive except the submitted v345")
    for name, expected in DATA_HASHES.items():
        if sha256(args.data_dir / name) != expected:
            raise ValueError("Official input mismatch: " + name)
    versions = {name:importlib.metadata.version(name) for name in
                ("catboost","scikit-learn","numpy","pandas","joblib")}
    required = {"catboost":"1.2.8","scikit-learn":"1.6.1","numpy":"2.2.6","pandas":"2.2.3","joblib":"1.5.1"}
    if versions != required:
        raise ValueError("Use the verified workload training environment: " + json.dumps(required))
    from src.champion.v342_finalize_workload_h1 import run
    result = run(args.data_dir/"train.csv",args.data_dir/"trackman_history.csv",
                 Path(__file__).resolve().parent/"h1",args.reference_zip,args.output_dir)
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()

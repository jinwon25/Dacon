"""Offline training, checkpoint rebuild, and verification entry point."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINAL_SHA256 = "d44578dc50220ce84dd4b8489bbae680afdcf93f931ed4236287b5a9e6f5aaaa"
DATA_HASHES = {
    "train.csv": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "trackman_history.csv": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_inputs(data_dir):
    for name, expected in DATA_HASHES.items():
        actual = sha256(data_dir / name)
        if actual != expected:
            raise ValueError(f"Official data hash mismatch: {name}: {actual}")
    manifest = json.loads((ROOT / "checkpoints.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if item["role"] == "historical_checkpoint_not_fresh_training":
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {item['path']}")
    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "catboost",
                 "lightgbm", "xgboost", "torch", "pytest"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return {"os": platform.platform(), "python": platform.python_version(),
            "libraries": versions, "data_and_checkpoint_hashes": "pass",
            "reproduction_inputs_ready": True,
            "all_model_exact_fresh_fit_verified": False,
            "known_limitations": ["Fallback XGBoost exact CUDA training parity has not been established",
                         "The complete initial-to-v343 fresh training chain has not been verified as one uninterrupted run",
                         "Cross-version and CPU/GPU model-training parity is not established"]}


def run_python(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONIOENCODING": "utf-8",
                        "PYTHONDONTWRITEBYTECODE": "1"})


def fresh_output(path):
    if path.exists():
        raise FileExistsError(f"Use a new output directory: {path}")
    path.mkdir(parents=True)


def h1_training_arguments(data, output, versions):
    recipe = json.loads((ROOT / "h1/training_recipe.json").read_text(encoding="utf-8"))
    mismatches = {name: {"installed": versions.get(name), "required": version}
                  for name, version in recipe["required_versions"].items()
                  if versions.get(name) != version}
    if mismatches:
        raise RuntimeError("Use the separate H1 training environment; version mismatch: "
                           + json.dumps(mismatches))
    return ["h1/exp/build_asof.py", "--build", "--ctx", "--lvl",
            "--seeds", len(recipe["seeds"]), "--depth", recipe["depth"],
            "--skip-eval", "--center", recipe["center"],
            "--data-dir", data, "--output-dir", output]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "rebuild", "verify", "lookups", "train-h1", "train-strict", "train-futures"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--scale-rows", type=int, default=245789)
    parser.add_argument("--stage", choices=["all", "lgb_oof", "hgb_oof", "team", "pitcher_count", "lowrank", "package"], default="all")
    parser.add_argument("--check", action="store_true", help="Validate strict training prerequisites without fitting")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    report = check_inputs(data)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.command == "preflight":
        return
    if args.command == "train-strict":
        if not args.output_dir:
            parser.error("train-strict requires --output-dir")
        arguments = ["strict/reproduce_strict.py", "--stage", args.stage,
                     "--data-dir", data, "--work-dir", args.output_dir.resolve()]
        if args.check:
            arguments.append("--check")
        run_python(*arguments)
        return
    if args.command == "verify":
        if not args.package:
            parser.error("verify requires --package")
        package = args.package.resolve()
        if sha256(package) != FINAL_SHA256:
            raise ValueError("Not the submitted v345 archive")
        run_python("-m", "src.audit_standalone_release", "--package", package,
                   "--test-csv", data / "test.csv", "--scale-rows", args.scale_rows,
                   "--timeout-seconds", 600)
        return
    if not args.output_dir:
        parser.error(f"{args.command} requires --output-dir")
    output = args.output_dir.resolve()
    h1_arguments = (h1_training_arguments(data, output, report["libraries"])
                    if args.command == "train-h1" else None)
    if args.command == "train-futures":
        if report["libraries"]["catboost"] != "1.2.8":
            raise RuntimeError("The submitted futures models require CatBoost 1.2.8")
        audit = ROOT / "reproduction_inputs/futures_selection_summary.json"
        if sha256(audit) != "dc2cb6bdf9db89b30c5a0248d569f38ecb51a5ec7067d7f7175127e7779c800d":
            raise ValueError("Futures selection summary hash mismatch")
    fresh_output(output)
    (output / "environment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.command == "lookups":
        run_python("fallback_xgb/build_fallback_xgb_lookups.py",
                   "--train-csv", data / "train.csv", "--trackman-csv", data / "trackman_history.csv",
                   "--verify", "--output", output / "fallback_lookups.joblib")
    elif args.command == "train-h1":
        run_python(*h1_arguments)
    elif args.command == "train-futures":
        run_python("-m", "src.champion.v289_finalize_recent_futures_expert",
                   "--train-csv", data / "train.csv", "--v288-summary", audit,
                   "--output-dir", output)
    elif args.command == "rebuild":
        checkpoints = ROOT / "reproduction_inputs"
        run_python("-m", "src.champion.v345_build_transition_workload_beta_package",
                   "--source-zip", checkpoints / "v343.zip", "--train-csv", data / "train.csv",
                   "--beta-axes", checkpoints / "beta_axes.npz",
                   "--audit-summary", checkpoints / "v345_audit_summary.json",
                   "--output-dir", output)
        package = output / "submit_v345.zip"
        actual = sha256(package)
        with zipfile.ZipFile(package) as archive:
            if archive.testzip() is not None:
                raise ValueError("Output ZIP CRC failure")
        if actual != FINAL_SHA256:
            raise ValueError(f"Rebuild completed but exact archive parity FAILED: {actual}")
        print(f"EXACT CHECKPOINT REBUILD PASS: {actual}; not an all-model fresh fit.")


if __name__ == "__main__":
    main()

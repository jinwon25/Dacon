"""Portable EXP-019/020/021 training chain; separate from whole-v345 reproduction.

Preserves recovered algorithms and selected configurations. All four source
seasons are generated (the old local R-LightGBM script defaulted to 2024 only).
No pretrained checkpoint is fetched or used by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
STAGES = {
    "lgb_oof": "train_exp019_r_full_residual",
    "hgb_oof": "train_exp019_histgb_residual",
    "team": "train_exp019_team_eb_ensemble",
    "pitcher_count": "train_exp020_pitcher_count_eb_atop_team",
    "lowrank": "train_exp020_low_rank_pitcher_context_eb",
    "package": "build_exp021_final_candidates",
}
YEARS = (2021, 2022, 2023, 2024)
TRAIN_HASH = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
_PATH_DEFAULTS = {}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_sources():
    manifest = json.loads((ROOT / "source_integrity.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if sha256(EXPERIMENTS / record["file"]) != record["sha256"]:
            raise ValueError(f"Recovered source hash mismatch: {record['file']}")
    return len(manifest["files"])


def configure(stage, data_dir, work_dir):
    sys.path.insert(0, str(EXPERIMENTS))
    module = importlib.import_module(STAGES[stage])
    # Imported helpers retain their own globals, so rebase those as well.
    for helper in tuple(sys.modules.values()):
        location = getattr(helper, "__file__", None)
        if not location or Path(location).resolve().parent != EXPERIMENTS:
            continue
        for name, value in tuple(vars(helper).items()):
            if not isinstance(value, Path):
                continue
            value = _PATH_DEFAULTS.setdefault((helper.__name__, name), value)
            if name == "DATA_DIR":
                setattr(helper, name, data_dir)
            elif name == "DATA_PATH":
                setattr(helper, name, data_dir / "train.csv")
            elif not value.is_absolute() and value.parts and value.parts[0] == "artifacts":
                setattr(helper, name, work_dir / value)
    if stage == "lgb_oof":
        module.VALIDATION_SEASONS = list(YEARS)
        module.REPORT_SEASONS = [2022, 2023, 2024]
    if stage == "package":
        module.DATA_DIR = data_dir
        module.ROOT = work_dir
        module.TEMPLATE = EXPERIMENTS / "exp021_submission_inference.py"
        module.TEAM_ROOT = work_dir / "artifacts/EXP-019/team_eb_ensemble"
        module.LOWRANK_METRICS = work_dir / "artifacts/EXP-020/low_rank_pitcher_context_eb/validation_metrics.json"
        module.PC_METRICS = work_dir / "artifacts/EXP-020/pitcher_count_eb_atop_team/validation_metrics.json"
        module.PYTHON = Path(sys.executable)
        module.VARIANTS = {"strict": {
            "directory": work_dir / "final/EXP-021-STRICT",
            "zip": work_dir / "submit_exp021_strict.zip",
            "candidate": "strict_lowrank_s300_r6"}}
        module.smoke_test = lambda path: smoke_test(path, data_dir)
    return module


def prerequisite_files(stage, work_dir):
    artifacts = work_dir / "artifacts"
    lgb = artifacts / "EXP-019/r_full_residual/rfull_l63_m1000_i300"
    hgb = artifacts / "EXP-019/histgb_residual/hist_l15_d4_m3000_i160"
    team = artifacts / "EXP-019/team_eb_ensemble"
    files = []
    if stage in ("hgb_oof", "team"):
        for year in YEARS:
            files += [lgb / f"targets_{year}.npy", lgb / f"predictions_branch_w075_{year}.npy"]
            if stage == "team":
                files += [hgb / f"targets_{year}.npy", hgb / f"predictions_branch_w100_{year}.npy"]
    if stage in ("pitcher_count", "lowrank", "package"):
        for year in YEARS:
            files += [team / f"targets_{year}.npy", team / f"base_ensemble_predictions_{year}.npy",
                      team / f"predictions_all_prior_s1000_{year}.npy"]
    if stage in ("lowrank", "package"):
        files += [artifacts / "EXP-020/pitcher_count_eb_atop_team/validation_metrics.json"]
    if stage == "lowrank":
        files += [artifacts / f"EXP-020/pitcher_count_eb_atop_team/predictions_team_pc_all_{year}.npy"
                  for year in YEARS]
    if stage == "package":
        files += [artifacts / "EXP-020/low_rank_pitcher_context_eb/validation_metrics.json"]
    return files


def smoke_test(package, data_dir):
    """Use small CSV copies, not Windows symlinks or the full training CSV."""
    import numpy as np
    import pandas as pd
    with tempfile.TemporaryDirectory(prefix="strict-smoke-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(package) as archive:
            for name in archive.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                    raise ValueError(f"Unsafe component ZIP path: {name}")
            archive.extractall(stage)
        (stage / "data").mkdir()
        for name in ("test.csv", "sample_submission.csv"):
            shutil.copy2(data_dir / name, stage / "data" / name)
        subprocess.run([sys.executable, "script.py"], cwd=stage, check=True, timeout=600,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        prediction = pd.read_csv(stage / "output/submission.csv")
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        values = prediction["control_success"].to_numpy()
        if not prediction["row_id"].equals(sample["row_id"]):
            raise ValueError("Smoke row order mismatch")
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Smoke probabilities invalid")
        return {"rows": len(values), "status": "pass", "fixture": "public sample"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate sources, imports and paths; no fitting")
    args = parser.parse_args()
    data, work = args.data_dir.resolve(), args.work_dir.resolve()
    count = verify_sources()
    if sha256(data / "train.csv") != TRAIN_HASH:
        raise ValueError("Not the official train.csv")
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    if args.check:
        for stage in selected:
            configure(stage, data, work)
        print(json.dumps({"source_files_verified": count, "import_check": "pass", "stages": selected,
                          "missing_prerequisites_before_training": {
                              s: [str(p.relative_to(work)) for p in prerequisite_files(s, work) if not p.exists()]
                              for s in selected}, "full_training_executed": False}, indent=2))
        return
    if args.stage == "all":
        if work.exists():
            raise FileExistsError("Full fresh training requires a new work directory")
        work.mkdir(parents=True)
        for stage in selected:
            subprocess.run([sys.executable, __file__, "--stage", stage,
                            "--data-dir", str(data), "--work-dir", str(work)], check=True)
        return
    missing = [str(p) for p in prerequisite_files(args.stage, work) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Run preceding stages first: " + ", ".join(missing))
    work.mkdir(parents=True, exist_ok=True)
    marker = work / (args.stage + ".complete.json")
    if marker.exists():
        raise FileExistsError(f"Completed stage is immutable: {marker}")
    module = configure(args.stage, data, work)
    module.main()
    marker.write_text(json.dumps({"stage": args.stage, "source_files": count,
                                   "train_sha256": TRAIN_HASH, "status": "completed",
                                   "whole_v345_reproduction": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

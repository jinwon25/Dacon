"""Prove the flat v148 package predicts identically to the original package.

Both packages are executed in separate subprocesses (their component modules
share module names, so they cannot safely coexist in one interpreter) and their
raw probability vectors are compared elementwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "V148_FLAT_PARITY_AUDIT_V1"
ANCHOR_TEAM = 13
TARGET_COL = "control_success"
PARITY_TOLERANCE = 1e-12

_RUNNER = """
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1]).resolve()
frame_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("v148_package_entry", root / "script.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
frame = pd.read_csv(frame_path, encoding="utf-8-sig")
prediction = np.asarray(module.predict_dataframe(frame), dtype=np.float64)
np.save(output_path, prediction)
"""


def domain_of(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).to_numpy() == "R"
    anchor = (
        frame["pitcher_team_id"].to_numpy() == ANCHOR_TEAM
    ) | (frame["batter_team_id"].to_numpy() == ANCHOR_TEAM)
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def build_sample(train_csv: Path, rows: int, seed: int) -> pd.DataFrame:
    full = pd.read_csv(train_csv, encoding="utf-8-sig")
    if TARGET_COL in full.columns:
        full = full.drop(columns=[TARGET_COL])
    domain = domain_of(full)
    cold = (
        full["asof_pitcher_n"].isna()
        | (pd.to_numeric(full["asof_pitcher_n"], errors="coerce").fillna(0) == 0)
        | full["asof_pitcher_success_rate"].isna()
    ).to_numpy()

    rng = np.random.default_rng(seed)
    chosen: set[int] = set()

    # proportional base draw
    base = rng.choice(len(full), size=min(rows, len(full)), replace=False)
    chosen.update(int(i) for i in base)

    # guarantee every domain and cold-start rows are represented
    for label in ("R_CORE", "R_ANCHOR", "F"):
        pool = np.flatnonzero(domain == label)
        take = min(3000, len(pool))
        if take:
            chosen.update(int(i) for i in rng.choice(pool, size=take, replace=False))
    cold_pool = np.flatnonzero(cold)
    if len(cold_pool):
        chosen.update(
            int(i) for i in rng.choice(cold_pool, size=min(5000, len(cold_pool)), replace=False)
        )
    # guarantee every season is represented
    for season in sorted(full["season"].dropna().unique()):
        pool = np.flatnonzero(full["season"].to_numpy() == season)
        take = min(1500, len(pool))
        if take:
            chosen.update(int(i) for i in rng.choice(pool, size=take, replace=False))

    index = np.sort(np.fromiter(chosen, dtype=np.int64))
    sample = full.iloc[index].reset_index(drop=True)
    del full
    return sample


def apply_inference_season(
    sample: pd.DataFrame,
    season: int,
    strict_dir: Path,
    cold_rows: int,
) -> pd.DataFrame:
    """Retarget the sample to the only season the frozen package accepts.

    ``strict`` enforces contract invariants that a historical training row
    cannot satisfy: ``season > through_season`` (2024), every ``asof_*_n``
    counter must be at least the entity's frozen 2024-end total, and the
    reconstructed in-season successes must lie in ``[0, in-season pitches]``.
    Real 2025 evaluation rows satisfy these by construction.  This rebuilds the
    counters as ``frozen_career + in_season_progress``, deriving the progress
    and rates from the original row so per-row diversity survives.  Both
    packages receive this identical frame, so the parity comparison is
    unaffected by the reconstruction.
    """
    history = json.loads((strict_dir / "history_state.json").read_text(encoding="utf-8"))
    multirate = json.loads((strict_dir / "multirate_state.json").read_text(encoding="utf-8"))
    tables = multirate["tables"]

    def indexed(records: list[dict[str, object]], key: str) -> pd.DataFrame:
        return pd.DataFrame.from_records(records).set_index(key)

    hist = {
        "pitcher": indexed(history["pitcher"], "pitcher_id"),
        "batter": indexed(history["batter"], "batter_id"),
    }
    mrate = {
        "pitcher_control": indexed(tables["pitcher_control"], "pitcher_id"),
        "batter_control": indexed(tables["batter_control"], "batter_id"),
        "pitcher_pitchmix": indexed(tables["pitcher_pitchmix"], "pitcher_id"),
    }

    out = sample.copy()
    out["season"] = season

    def numeric(column: str) -> np.ndarray:
        return pd.to_numeric(out[column], errors="coerce").fillna(0.0).to_numpy(float)

    for entity, control_key in (("pitcher", "pitcher_control"), ("batter", "batter_control")):
        ids = out[f"{entity}_id"]
        hist_prior_n = ids.map(hist[entity]["prior_n"]).fillna(0.0).to_numpy(float)
        hist_prior_successes = (
            ids.map(hist[entity]["prior_successes"]).fillna(0.0).to_numpy(float)
        )
        control_prior_n = ids.map(mrate[control_key]["prior_n"]).fillna(0.0).to_numpy(float)
        base_n = np.maximum(hist_prior_n, control_prior_n)

        season_n = np.minimum(numeric(f"asof_{entity}_n"), 600.0)
        career_n = base_n + season_n
        # headroom measured against the stricter history_state floor
        headroom = career_n - hist_prior_n
        original_rate = (
            pd.to_numeric(out[f"asof_{entity}_success_rate"], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        gained = np.clip(np.rint(headroom * original_rate), 0.0, headroom)
        career_successes = hist_prior_successes + gained
        out[f"asof_{entity}_n"] = career_n
        out[f"asof_{entity}_success_rate"] = np.divide(
            career_successes,
            career_n,
            out=np.full(len(out), np.nan),
            where=career_n > 0,
        )

    mix_prior = (
        out["pitcher_id"].map(mrate["pitcher_pitchmix"]["prior_n"]).fillna(0.0).to_numpy(float)
    )
    out["asof_pitcher_pitchmix_n"] = mix_prior + np.minimum(
        numeric("asof_pitcher_pitchmix_n"), 600.0
    )

    # Genuine cold start cannot be drawn from train: every entity there is
    # already in the frozen tables.  A 2025 debut is synthesised instead by
    # assigning never-seen ids and zeroed counters, which exercises the
    # fillna/global-rate fallback paths.
    ceiling = int(
        max(
            out["pitcher_id"].max(),
            out["batter_id"].max(),
            max(hist["pitcher"].index.max(), hist["batter"].index.max()),
            max(frame.index.max() for frame in mrate.values()),
        )
    )
    positions = out.index[:cold_rows]
    debut = ceiling + 1000 + np.arange(len(positions), dtype=np.int64)
    out.loc[positions, "pitcher_id"] = debut
    out.loc[positions, "batter_id"] = debut
    for entity in ("pitcher", "batter"):
        out.loc[positions, f"asof_{entity}_n"] = 0.0
        out.loc[positions, f"asof_{entity}_success_rate"] = np.nan
    out.loc[positions, "asof_pitcher_pitchmix_n"] = 0.0
    return out


def predict_with_package(
    package_root: Path, frame_path: Path, output_path: Path
) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(package_root), str(frame_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"package inference failed ({package_root}):\n{completed.stdout}\n{completed.stderr}"
        )
    return time.perf_counter() - started


def extract(zip_path: Path, destination: Path) -> Path:
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    return destination


def run(
    original_zip: Path,
    flat_zip: Path,
    train_csv: Path,
    workspace: Path,
    rows: int,
    seed: int,
    invariance_rows: int,
    inference_season: int,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    original_root = extract(original_zip, workspace / "original")
    flat_root = extract(flat_zip, workspace / "flat")

    sample = build_sample(train_csv, rows, seed)
    source_seasons = sorted(int(s) for s in sample["season"].dropna().unique())
    sample = apply_inference_season(
        sample,
        inference_season,
        original_root / "model/v124/model/parent/parent/strict",
        cold_rows=2000,
    )
    frame_path = workspace / "sample.csv"
    sample.to_csv(frame_path, index=False, encoding="utf-8")

    original_npy = workspace / "original.npy"
    flat_npy = workspace / "flat.npy"
    original_seconds = predict_with_package(original_root, frame_path, original_npy)
    flat_seconds = predict_with_package(flat_root, frame_path, flat_npy)

    repeat_npy = workspace / "original_repeat.npy"
    predict_with_package(original_root, frame_path, repeat_npy)

    original_prediction = np.load(original_npy)
    flat_prediction = np.load(flat_npy)
    determinism_max = float(np.abs(np.load(repeat_npy) - original_prediction).max())
    difference = np.abs(original_prediction - flat_prediction)
    domain = domain_of(sample)

    per_domain = {}
    for label in ("R_CORE", "R_ANCHOR", "F"):
        mask = domain == label
        per_domain[label] = {
            "rows": int(mask.sum()),
            "max_abs_diff": float(difference[mask].max()) if mask.any() else 0.0,
        }

    cold = (
        (pd.to_numeric(sample["asof_pitcher_n"], errors="coerce").fillna(0) == 0)
        & sample["asof_pitcher_success_rate"].isna()
    ).to_numpy()

    # invariance checks on the flat package
    subset = sample.iloc[:invariance_rows].reset_index(drop=True)
    subset_path = workspace / "subset.csv"
    subset.to_csv(subset_path, index=False, encoding="utf-8")
    base_npy = workspace / "subset_base.npy"
    predict_with_package(flat_root, subset_path, base_npy)
    base_prediction = np.load(base_npy)

    order = np.random.default_rng(seed + 1).permutation(len(subset))
    shuffled_path = workspace / "subset_shuffled.csv"
    subset.iloc[order].reset_index(drop=True).to_csv(
        shuffled_path, index=False, encoding="utf-8"
    )
    shuffled_npy = workspace / "subset_shuffled.npy"
    predict_with_package(flat_root, shuffled_path, shuffled_npy)
    shuffled_max = float(np.abs(np.load(shuffled_npy) - base_prediction[order]).max())

    half = len(subset) // 2
    partition_parts = []
    for index, piece in enumerate((subset.iloc[:half], subset.iloc[half:])):
        piece_path = workspace / f"subset_part{index}.csv"
        piece.reset_index(drop=True).to_csv(piece_path, index=False, encoding="utf-8")
        piece_npy = workspace / f"subset_part{index}.npy"
        predict_with_package(flat_root, piece_path, piece_npy)
        partition_parts.append(np.load(piece_npy))
    partition_max = float(
        np.abs(np.concatenate(partition_parts) - base_prediction).max()
    )

    max_abs_diff = float(difference.max())
    result = {
        "protocol": PROTOCOL,
        "rows": int(len(sample)),
        "seed": seed,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": float(difference.mean()),
        "exact_match_fraction": float((difference == 0.0).mean()),
        "tolerance": PARITY_TOLERANCE,
        "parity_passed": bool(max_abs_diff <= PARITY_TOLERANCE),
        "original_vs_original_max_abs": determinism_max,
        "per_domain": per_domain,
        "cold_start_rows": int(cold.sum()),
        "cold_start_max_abs_diff": float(difference[cold].max()) if cold.any() else 0.0,
        "source_seasons_sampled": source_seasons,
        "inference_season": inference_season,
        "months": sorted(int(m) for m in sample["game_month"].dropna().unique()),
        "original_prediction_mean": float(original_prediction.mean()),
        "flat_prediction_mean": float(flat_prediction.mean()),
        "original_seconds": original_seconds,
        "flat_seconds": flat_seconds,
        "invariance_rows": int(len(subset)),
        "shuffle_max_abs": shuffled_max,
        "partition_max_abs": partition_max,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-zip", type=Path, required=True)
    parser.add_argument("--flat-zip", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--invariance-rows", type=int, default=20000)
    parser.add_argument("--inference-season", type=int, default=2025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        args.original_zip,
        args.flat_zip,
        args.train_csv,
        args.workspace,
        args.rows,
        args.seed,
        args.invariance_rows,
        args.inference_season,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

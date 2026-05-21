"""Phase B: seed-averaged boundary correction at test time.

Given a trained selector output directory (containing both
oof_selector_scores.npz and test_selector_scores.npz), train K boundary
correction models on different seeds and average their test predictions.
Saves a blended submission CSV.

Usage:
  python scripts/seed_avg_boundary.py <SELECTOR_OUT_DIR> --seeds 20260622 20260623 20260624 \
      --cap 0.006 --apply 1.0 [--name TAG]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def _read_submission(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    xyz: list[list[float]] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return ids, np.asarray(xyz, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("selector_out", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--cap", type=float, default=0.006)
    ap.add_argument("--apply", type=float, default=1.0)
    ap.add_argument("--name", type=str, default=None)
    args = ap.parse_args()

    selector_out = args.selector_out.resolve()
    score_bank = selector_out / "oof_selector_scores.npz"
    test_bank = selector_out / "test_selector_scores.npz"
    assert score_bank.exists(), score_bank
    assert test_bank.exists(), test_bank

    tag = args.name or f"seedavg_cap{args.cap:g}_apply{args.apply:g}_n{len(args.seeds)}"
    out_root = ROOT / "outputs" / "07_seed_averaged" / f"{selector_out.name}_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    seed_preds: list[np.ndarray] = []
    seed_ids_ref: list[str] = []
    for seed in args.seeds:
        seed_dir = out_root / f"seed{seed}"
        if not (seed_dir / "submission_boundary_tiny_gate.csv").exists():
            seed_dir.mkdir(parents=True, exist_ok=True)
            P.call_main(P.BOUNDARY_MAIN, [
                "--root", P.DATA_ROOT,
                "--out-dir", seed_dir,
                "--fold", 0, "--folds", 5,
                "--score-bank", score_bank,
                "--make-test",
                "--test-score-bank", test_bank,
                "--epochs", 1, "--fine-epochs", 1, "--min-epochs", 1, "--patience", 1,
                "--hidden", 64, "--batch", 8192,
                "--lr", 0.001, "--fine-lr-scale", 0.18,
                "--cap", args.cap, "--apply-scale", args.apply,
                "--device", "cpu", "--seed", seed, "--save-val-pred",
            ])
        gate_csv = seed_dir / "submission_boundary_tiny_gate.csv"
        ids, xyz = _read_submission(gate_csv)
        if not seed_ids_ref:
            seed_ids_ref = ids
        else:
            assert ids == seed_ids_ref, "test ids changed between seed runs"
        seed_preds.append(xyz)
    avg = np.mean(np.stack(seed_preds, axis=0), axis=0)
    out_csv = out_root / "submission_seedavg_gate.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "z"])
        for sid, row in zip(seed_ids_ref, avg):
            w.writerow([sid, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])

    # Distance vs first seed prediction (sanity)
    diffs = np.linalg.norm(avg - seed_preds[0], axis=1)
    print(f"\n[SEED_AVG] wrote {out_csv}")
    print(f"  seeds={args.seeds}")
    print(f"  mean shift vs seed[0]: {float(np.mean(diffs)) * 1000:.3f} mm")
    print(f"  p95 shift:             {float(np.quantile(diffs, 0.95)) * 1000:.3f} mm")
    print(f"  max shift:             {float(np.max(diffs)) * 1000:.3f} mm")


if __name__ == "__main__":
    main()

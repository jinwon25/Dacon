"""Verify the integrated candidate bank with the new poly/smoothed candidates.

Checks:
1) make_candidates() runs at horizons 1 and 2 for various end_idx without error.
2) Oracle at end_idx=10, horizon=2 matches the standalone probe expectation.
3) Per-fold oracle (to catch high-variance per-fold behavior).
4) Per-candidate hit on training set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def main() -> None:
    data_root = P.DATA_ROOT
    train_ids, train_y = P.read_labels(data_root / "train_labels.csv")
    train_x = P.load_stack(data_root / "train", train_ids)

    print(f"CANDIDATES count: {len(P.CANDIDATES)}")
    print(f"FAMILY_NAMES: {P.FAMILY_NAMES}")
    print(f"CANDIDATE_FAMILY for new ones (last 4): {P.CANDIDATE_FAMILY[-4:]}")
    print(f"New kinds: {[c.kind for c in P.CANDIDATES[-4:]]}")

    # Smoke: horizons 1 and 2, various end_idx
    for horizon in (1, 2):
        for end_idx in (3, 5, 8, 10):
            try:
                cands = P.make_candidates(train_x[:16], end_idx, horizon=horizon)
                assert cands.shape == (16, len(P.CANDIDATES), 3), cands.shape
                assert np.isfinite(cands).all(), "non-finite values in cands"
            except Exception as e:
                print(f"FAIL end_idx={end_idx} horizon={horizon}: {e}")
                raise
    print("smoke (horizons 1/2, end_idx 3/5/8/10): OK")

    # Final bank: end_idx=10, horizon=2
    cands = P.make_candidates(train_x, 10, horizon=2)
    err = np.linalg.norm(cands - train_y[:, None, :], axis=2)
    oracle = float(np.mean(np.min(err, axis=1) <= P.R_HIT))
    print(f"\nfull bank oracle (end_idx=10, horizon=2): {oracle:.4f}")
    print(f"  (probe predicted >= 0.7460 with these 4 new candidates)")

    # Per-cand hit
    print("\nPer-candidate hit@1cm:")
    for i, spec in enumerate(P.CANDIDATES):
        h = float(np.mean(err[:, i] <= P.R_HIT))
        flag = "  NEW" if spec.kind != "motion" else ""
        print(f"  [{i:2d}] {spec.name:<36} hit={h:.4f}{flag}")

    # Best-arg by candidate counts (which candidates would be the oracle pick most often)
    best_arg = np.argmin(err, axis=1)
    counts = np.bincount(best_arg, minlength=len(P.CANDIDATES))
    order = np.argsort(counts)[::-1]
    print("\nOracle-best argmax counts (top 10):")
    for j in order[:10]:
        print(f"  [{j:2d}] {P.CANDIDATES[int(j)].name:<36} {int(counts[j])}")

    # Per-fold oracle (stable_fold_id)
    print("\nPer-fold oracle (5-fold stable hash):")
    fold_ids = np.asarray([P.stable_fold_id(s, 5) for s in train_ids])
    for f in range(5):
        mask = fold_ids == f
        sub_err = err[mask]
        o = float(np.mean(np.min(sub_err, axis=1) <= P.R_HIT))
        print(f"  fold {f}: oracle={o:.4f} n={int(mask.sum())}")

    # Compare against subset oracle WITHOUT the 4 new candidates
    base_cands = cands[:, : len(P.CANDIDATES) - 4]
    base_err = np.linalg.norm(base_cands - train_y[:, None, :], axis=2)
    base_oracle = float(np.mean(np.min(base_err, axis=1) <= P.R_HIT))
    print(f"\nbaseline 27-cand oracle: {base_oracle:.4f}")
    print(f"with 4 smoothed:         {oracle:.4f}  delta={oracle - base_oracle:+.4f}")


if __name__ == "__main__":
    main()

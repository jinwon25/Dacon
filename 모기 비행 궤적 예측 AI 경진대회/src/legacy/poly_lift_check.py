"""Quantify how much the poly proposals RESCUE current selector misses.

Using the existing 27-cand OOF selector scores, for each row pick the soft / gate
predictions the selector is actually producing. Then check:
  - on rows where selector misses (>1cm), how many would a poly candidate save?
  - if we extended the selector pick to "selector pick OR best poly candidate (by
    its standalone confidence proxy)", how much does hit-rate move?

This is a conservative upper-bound: it assumes the *retrained* selector can
recognize poly candidates as the right pick. Real gain will be lower, but the
upper bound tells us whether further work is worth it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402

# Reuse build_proposals from probe script:
sys.path.insert(0, str(ROOT / "scripts"))
from poly_oracle_probe import build_proposals  # noqa: E402


def main() -> None:
    data_root = P.DATA_ROOT
    train_ids, train_y = P.read_labels(data_root / "train_labels.csv")
    train_x = P.load_stack(data_root / "train", train_ids)

    # Load existing OOF selector scores (27-cand)
    bank_path = ROOT / "outputs" / "selector_full" / "oof_selector_scores.npz"
    data = np.load(bank_path, allow_pickle=True)
    covered = data["covered"]
    cands = data["cands"]              # (10000, 27, 3)
    ens_scores = data["ens_scores"]    # (10000, 27)
    y = data["y"]
    print(f"covered: {int(covered.sum())} / {len(covered)}")

    # Restrict to covered rows
    idx = np.flatnonzero(covered)
    cands = cands[idx]
    scores = ens_scores[idx]
    truth = y[idx]
    x = train_x[idx]
    print(f"working set: {len(idx)} rows")

    # Selector argmax pred
    argmax = np.argmax(scores, axis=1)
    pred_argmax = cands[np.arange(len(idx)), argmax]
    err_argmax = np.linalg.norm(pred_argmax - truth, axis=1)
    hit_argmax = err_argmax <= P.R_HIT
    print(f"current argmax hit: {float(np.mean(hit_argmax)):.4f}")

    # Soft pick with selector best temperature ~ 0.05 (from notebooks)
    soft = P.soft_select(cands, scores, temperature=0.05)
    err_soft = np.linalg.norm(soft - truth, axis=1)
    hit_soft = err_soft <= P.R_HIT
    print(f"current soft hit (temp=0.05): {float(np.mean(hit_soft)):.4f}")

    # Build poly proposals on the working subset
    proposals = build_proposals(x)
    poly_stack = np.stack(list(proposals.values()), axis=1)  # (N, M, 3)
    poly_err = np.linalg.norm(poly_stack - truth[:, None, :], axis=2)
    print(f"\nproposals: {list(proposals.keys())}")
    print(f"poly bank shape: {poly_stack.shape}")

    # Upper bound: if we can perfectly pick among existing argmax + all poly
    miss_argmax = ~hit_argmax
    poly_hit_any = np.min(poly_err, axis=1) <= P.R_HIT
    rescue = miss_argmax & poly_hit_any
    print(f"argmax-miss rows that any poly cand can rescue: {int(rescue.sum())} ({float(np.mean(rescue)):.4f})")

    # Per-poly rescue counts
    print("\nPer-poly rescue (among current argmax-miss rows):")
    for i, name in enumerate(proposals.keys()):
        rescue_i = miss_argmax & (poly_err[:, i] <= P.R_HIT)
        print(f"  {name:<24} rescues {int(rescue_i.sum()):>4} ({float(np.mean(rescue_i)):.4f})")

    # Selector "agreement" check: when current top candidate is already close
    # to a poly cand, the poly probably matches existing physics → low marginal value.
    # When poly cand is far from current argmax pick AND closer to truth → high value.
    dist_pick_to_poly = np.min(np.linalg.norm(pred_argmax[:, None, :] - poly_stack, axis=2), axis=1)
    print(f"\nmean dist(argmax pick, nearest poly): {float(np.mean(dist_pick_to_poly)):.4f} m")
    print(f"median dist:                          {float(np.median(dist_pick_to_poly)):.4f} m")

    # Conservative composite: switch to nearest poly candidate ONLY when
    # all 27 score candidates have low max score (low confidence) AND poly is
    # close to argmax pick (consistent). This is a simulation, not a real prediction.
    # Just to show: what fraction of low-confidence misses can be rescued?
    top1 = np.max(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    low_conf = margin < np.quantile(margin, 0.50)
    low_conf_miss = low_conf & miss_argmax
    low_conf_miss_rescuable = low_conf_miss & poly_hit_any
    print(f"\nlow-margin (bottom 50%) argmax-misses: {int(low_conf_miss.sum())}")
    print(f"  of those, rescuable by some poly: {int(low_conf_miss_rescuable.sum())} ({float(np.mean(low_conf_miss_rescuable)) * len(low_conf_miss) / max(int(low_conf_miss.sum()), 1):.2%})")


if __name__ == "__main__":
    main()

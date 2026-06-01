"""Post-hoc ensemble: blend the proven selector_full (attn_gru, seed 20260506)
with the new tcn from multimodel_attn_tcn_seed20260621.

Both banks share the 27-cand structure. We do a weighted convex combination of
their OOF scores, then search the optimal weight + global temperature/gate
parameters on the OOF data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def main() -> None:
    old = np.load(ROOT / "outputs" / "selector_full" / "oof_selector_scores.npz", allow_pickle=True)
    new = np.load(ROOT / "outputs" / "06_selector_experiments" / "multimodel_attn_tcn_seed20260621" / "oof_selector_scores.npz", allow_pickle=True)

    # Sanity: same candidate ordering, same y, same cands geometry.
    assert list(old["candidate_names"]) == list(new["candidate_names"]), "candidate order mismatch"
    assert old["covered"].shape == new["covered"].shape, "covered mismatch"
    assert np.array_equal(old["y"], new["y"]), "y mismatch"
    covered = old["covered"] & new["covered"]
    print(f"covered both: {int(covered.sum())}")

    cands = old["cands"][covered]
    y = old["y"][covered]
    s_old_attn = old["attn_gru_scores"][covered]
    s_new_attn = new["attn_gru_scores"][covered]
    s_new_tcn = new["tcn_scores"][covered]

    baselines = {
        "old_attn_gru (selector_full)": s_old_attn,
        "new_attn_gru (seed 20260621)": s_new_attn,
        "new_tcn (seed 20260621)":      s_new_tcn,
    }

    print("\n=== individual ===")
    for name, s in baselines.items():
        m = P.search_argmax_soft_gate(cands, s, y)
        print(f"  {name:<35s} gate hit={m['metrics']['hit']:.4f} hits={m['metrics']['hits']}")

    print("\n=== pairwise blends (weighted mean) ===")
    pair_results = []
    for a_name, s_a in baselines.items():
        for b_name, s_b in baselines.items():
            if a_name >= b_name:
                continue
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                s = w * s_a + (1 - w) * s_b
                m = P.search_argmax_soft_gate(cands, s, y)
                pair_results.append((m["metrics"]["hit"], a_name, b_name, w, m))
    pair_results.sort(key=lambda x: x[0], reverse=True)
    for hit, a, b, w, m in pair_results[:8]:
        print(f"  w*{a[:18]}+(1-w)*{b[:18]} w={w}: hit={hit:.4f}")

    print("\n=== 3-way blends (uniform + various weights) ===")
    s1, s2, s3 = s_old_attn, s_new_attn, s_new_tcn
    three_results = []
    for w1 in np.arange(0.2, 0.9, 0.1):
        for w2 in np.arange(0.0, 0.9, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue
            s = w1 * s1 + w2 * s2 + w3 * s3
            m = P.search_argmax_soft_gate(cands, s, y)
            three_results.append((m["metrics"]["hit"], w1, w2, w3, m))
    three_results.sort(key=lambda x: x[0], reverse=True)
    print(f"  best 3-way (w_oldattn, w_newattn, w_newtcn):")
    for hit, w1, w2, w3, m in three_results[:8]:
        print(f"    w=({w1:.1f}, {w2:.1f}, {w3:.1f}) gate hit={hit:.4f}  argmax_rate={m.get('argmax_rate', 0):.2f}")

    # Save the best blend OOF predictions for downstream boundary use
    best_hit, w1, w2, w3, best_m = three_results[0]
    best_scores = w1 * s1 + w2 * s2 + w3 * s3
    print(f"\nBest blend weights: ({w1:.1f}, {w2:.1f}, {w3:.1f}) gate hit={best_hit:.4f}")
    print(f"Baseline selector_full gate hit: 0.6688")
    print(f"Delta: {best_hit - 0.6688:+.4f}")


if __name__ == "__main__":
    main()

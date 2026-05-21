"""4-way ensemble search: old_attn_gru + new_attn_gru + new_tcn + transformer.

Greedy + random search over convex weights. Compares to current best.
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
    old = np.load(ROOT / "outputs" / "selector_full" / "oof_selector_scores.npz", allow_pickle=True)
    mm = np.load(ROOT / "outputs" / "06_selector_experiments" / "multimodel_attn_tcn_seed20260621" / "oof_selector_scores.npz", allow_pickle=True)
    tr = np.load(ROOT / "outputs" / "06_selector_experiments" / "transformer_seed20260622" / "oof_selector_scores.npz", allow_pickle=True)

    assert list(old["candidate_names"]) == list(mm["candidate_names"]) == list(tr["candidate_names"])
    covered = old["covered"] & mm["covered"] & tr["covered"]
    cands = old["cands"][covered]
    y = old["y"][covered]
    s1 = old["attn_gru_scores"][covered]  # old attn_gru
    s2 = mm["attn_gru_scores"][covered]   # new attn_gru
    s3 = mm["tcn_scores"][covered]        # new tcn
    s4 = tr["transformer_scores"][covered]  # transformer

    individuals = {
        "old_attn_gru": s1,
        "new_attn_gru": s2,
        "new_tcn": s3,
        "transformer": s4,
    }

    print("=== individual ===")
    for name, s in individuals.items():
        m = P.search_argmax_soft_gate(cands, s, y)
        print(f"  {name:>15s}: gate hit={m['metrics']['hit']:.4f}  margin_threshold={m['margin_threshold']:.3f}")

    # Greedy random search: sample 200 random simplex points
    rng = np.random.default_rng(42)
    best = -1.0
    best_w = None
    # Try uniform; pairs with leftover weight; full random simplex
    grid_w = []
    # Coarse grid 0.0..1.0 by 0.1
    for w1 in np.arange(0.0, 1.01, 0.1):
        for w2 in np.arange(0.0, 1.01 - w1 + 1e-9, 0.1):
            for w3 in np.arange(0.0, 1.01 - w1 - w2 + 1e-9, 0.1):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -1e-6 or w4 > 1 + 1e-6:
                    continue
                w4 = max(0.0, w4)
                grid_w.append((round(w1, 2), round(w2, 2), round(w3, 2), round(w4, 2)))
    print(f"\nsearching {len(grid_w)} coarse-grid weights")
    grid_results = []
    for w in grid_w:
        s = w[0] * s1 + w[1] * s2 + w[2] * s3 + w[3] * s4
        m = P.search_argmax_soft_gate(cands, s, y)
        grid_results.append((m["metrics"]["hit"], w, m))
    grid_results.sort(key=lambda r: r[0], reverse=True)
    print("\nTop 10 grid weights (w1=old_attn, w2=new_attn, w3=new_tcn, w4=transformer):")
    for hit, w, m in grid_results[:10]:
        print(f"  w=({w[0]:.1f},{w[1]:.1f},{w[2]:.1f},{w[3]:.1f}) hit={hit:.4f}  thr={m['margin_threshold']:.3f}  argmax_rate={m['argmax_rate']:.2f}")

    best_hit, best_w, best_m = grid_results[0]
    print(f"\nBest weights: {best_w} gate hit={best_hit:.4f}")
    print(f"Best individual: old_attn_gru 0.6561  new_attn_gru 0.6572")
    print(f"Delta vs new_attn_gru: {best_hit - 0.6572:+.4f}")

    # Refinement near best
    print("\nRefining around best...")
    refine_results = []
    grid = np.arange(-0.05, 0.051, 0.025)
    for d1 in grid:
        for d2 in grid:
            for d3 in grid:
                w1 = best_w[0] + d1
                w2 = best_w[1] + d2
                w3 = best_w[2] + d3
                w4 = 1.0 - w1 - w2 - w3
                if min(w1, w2, w3, w4) < -1e-6:
                    continue
                if max(w1, w2, w3, w4) > 1.0 + 1e-6:
                    continue
                s = w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4
                m = P.search_argmax_soft_gate(cands, s, y)
                refine_results.append((m["metrics"]["hit"], (w1, w2, w3, w4), m))
    refine_results.sort(key=lambda r: r[0], reverse=True)
    for hit, w, m in refine_results[:5]:
        print(f"  w=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f},{w[3]:.3f}) hit={hit:.4f}")

    rh, rw, rm = refine_results[0]
    # Save the best blend OOF + test for downstream boundary use
    s_blend = rw[0] * s1 + rw[1] * s2 + rw[2] * s3 + rw[3] * s4
    blend_full = np.zeros_like(old["attn_gru_scores"])
    blend_full[covered] = s_blend
    out_dir = ROOT / "outputs" / "08_blend_selector_4way"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "oof_selector_scores.npz",
        covered=covered,
        y=old["y"],
        cands=old["cands"],
        ens_scores=blend_full.astype(np.float32),
        ens_prior=old["ens_prior"].astype(np.float32),  # not used by boundary, placeholder
        candidate_names=old["candidate_names"],
    )

    # Same for test
    old_t = np.load(ROOT / "outputs" / "selector_full" / "test_selector_scores.npz", allow_pickle=True)
    mm_t = np.load(ROOT / "outputs" / "06_selector_experiments" / "multimodel_attn_tcn_seed20260621" / "test_selector_scores.npz", allow_pickle=True)
    tr_t = np.load(ROOT / "outputs" / "06_selector_experiments" / "transformer_seed20260622" / "test_selector_scores.npz", allow_pickle=True)
    t1 = old_t["attn_gru_scores"]
    t2 = mm_t["attn_gru_scores"]
    t3 = mm_t["tcn_scores"]
    t4 = tr_t["transformer_scores"]
    blend_test = rw[0] * t1 + rw[1] * t2 + rw[2] * t3 + rw[3] * t4
    np.savez_compressed(
        out_dir / "test_selector_scores.npz",
        cands=old_t["cands"],
        ens_scores=blend_test.astype(np.float32),
        ens_prior=old_t["ens_prior"].astype(np.float32),  # placeholder
        candidate_names=old_t["candidate_names"],
    )
    print(f"\nWrote 4-way blend bank to {out_dir}")
    print(f"OOF gate: {rh:.4f}  weights={rw}")


if __name__ == "__main__":
    main()

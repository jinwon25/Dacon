"""v132_final_blend.py — 최종 블렌드 + 정직한 nested-CV (제출 결정).

기존 pool(v110 load_pool 45) + 신규 decorrelated 멤버(v131 frenet/gru + 그 boundary).
두 가지를 한 번에:
  1) BLEND: full-pool DE + conservative DE(v122c 패턴: top-K + force diverse).
  2) NESTED-CV: 선택 recipe의 정직한 일반화 hit 추정 (blender 과적합 정량화).
     - v122c-style(소수 diverse) vs v122d-style(full pool)의 OOF→일반화 gap 비교.
     - 과거: v122c(9 active) LB 변환 +0.0143, v110_v3(29w) +0.0109(과적합).

usage:
  python scripts/v132_final_blend.py --mode both --tag v132
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass
from v110_de_ensemble import load_pool, softmax_weights, load_y_and_sub

CACHE = Path("data/cache"); DATA = Path("data")

def hit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())

def add_new_members(pool, y):
    """v131/v135 신규 멤버 + boundary 버전을 pool에 추가 (존재하는 것만).
    raw schema: oof_global/test_global/fold_mask. boundary schema: oof_v91/test_v91."""
    added = []
    # (prefix, [tags]) — raw members
    raw_groups = [
        ("v131", ["frenet_mlp", "frenet_gru", "yaw_gru", "frenet_gru_s4", "frenet_gru_n2", "frenet_gru_big", "frenet_gru_n2s4", "frenet_gru_n4", "frenet_transformer", "frenet_lru", "frenet_tcn", "frenet_itransformer", "frenet_flow", "yaw_flow", "frenet_sonode", "yaw_sonode"]),
        ("v135", ["ch_frenet_accel", "ch_yaw_accel", "ch_frenet_jerk"]),
    ]
    for prefix, tags in raw_groups:
        for tag in tags:
            p = CACHE / f"{prefix}_{tag}_state.npz"
            if p.exists():
                st = np.load(p)
                if not st["fold_mask"].all():
                    continue
                oof = st["oof_global"].astype(np.float64); test = st["test_global"].astype(np.float64)
                nm = f"{prefix}_{tag}"
                pool.append((nm, oof, test, hit(oof, y))); added.append(nm)
    # boundary versions: out-tag was f"{prefix}{tag}" -> file f"{prefix}{tag}_cap{cap}_state.npz"
    for prefix, tags in raw_groups:
        for tag in tags:
            for cap in ["10", "15"]:
                p = CACHE / f"{prefix}{tag}_cap{cap}_state.npz"
                if p.exists():
                    st = np.load(p)
                    oof = st["oof_v91"].astype(np.float64); test = st["test_v91"].astype(np.float64)
                    nm = f"{prefix}{tag}_c{cap}"
                    pool.append((nm, oof, test, hit(oof, y))); added.append(nm)
    return pool, added

def de_fit(oofs, y, n_iter=200, popsize=30, n_starts=3, seed0=0):
    N = len(oofs)
    def neg(z):
        w = softmax_weights(z)
        return -hit((w[:, None, None] * oofs).sum(0), y)
    bounds = [(-5., 5.)] * N
    best = (None, np.inf)
    for s in range(n_starts):
        r = differential_evolution(neg, bounds, seed=seed0 + s, maxiter=n_iter, popsize=popsize,
                                   tol=1e-6, mutation=(0.3, 1.5), recombination=0.8, init="sobol",
                                   polish=True, workers=1)
        if r.fun < best[1]: best = (r.x, r.fun)
    return softmax_weights(best[0]), -best[1]

def conservative_subset(pool, y, force_names, top_k=8, oof_floor=0.67):
    """v122c 패턴: OOF>=floor top-K + force-include diverse 멤버."""
    idx_by_oof = sorted(range(len(pool)), key=lambda i: -pool[i][3])
    chosen = []
    for i in idx_by_oof:
        if pool[i][3] >= oof_floor and len(chosen) < top_k:
            chosen.append(i)
    for i, p in enumerate(pool):
        if p[0] in force_names and i not in chosen:
            chosen.append(i)
    return chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["blend", "nestedcv", "both"])
    ap.add_argument("--tag", default="v132")
    ap.add_argument("--outer-folds", type=int, default=5)
    args = ap.parse_args()

    pool, y = load_pool(include_mdn=True)
    pool, added = add_new_members(pool, y)
    print(f"pool={len(pool)} (+{len(added)} new: {added})")
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])

    # decorrelation of new members vs v120/v122c base
    ref = {n: i for i, n in enumerate(names)}
    if "v120" in ref:
        for nm in added:
            if nm in ref:
                L2 = np.linalg.norm(tests[ref[nm]] - tests[ref["v120"]], axis=-1).mean() * 1000
                print(f"  decorr L2({nm} vs v120)={L2:.2f}mm  OOF={pool[ref[nm]][3]:.4f}")

    # force-include the new diverse members + existing ODE family
    force = [n for n in names if n.startswith("v131") or n.startswith("v135")
             or n in ("v120", "v121", "v121c5", "v120_big")]

    if args.mode in ("blend", "both"):
        # conservative (v122c-style)
        cons = conservative_subset(pool, y, force, top_k=8, oof_floor=0.67)
        wc, rhc = de_fit(oofs[cons], y, n_iter=200, popsize=30, n_starts=4)
        test_c = (wc[:, None, None] * tests[cons]).sum(0)
        print(f"\n[conservative] {len(cons)} members  OOF={rhc:.4f}")
        for j in np.argsort(-wc):
            if wc[j] >= 0.01: print(f"    {names[cons[j]]:<22} w={wc[j]:.3f} (OOF={pool[cons[j]][3]:.4f})")
        n_active_c = int((wc >= 0.01).sum())

        # full pool (v122d-style)
        wf, rhf = de_fit(oofs, y, n_iter=150, popsize=30, n_starts=2)
        test_f = (wf[:, None, None] * tests).sum(0)
        n_active_f = int((wf >= 0.01).sum())
        print(f"[full pool]    {len(pool)} members  OOF={rhf:.4f}  active={n_active_f}")

        for tag, tp, rh, na in [("conservative", test_c, rhc, n_active_c), ("full", test_f, rhf, n_active_f)]:
            _, sub, _ = load_y_and_sub()
            out = DATA / f"submission_{args.tag}_{tag}_oof{rh:.4f}.csv"
            pd.DataFrame({"id": sub["id"], "x": tp[:,0], "y": tp[:,1], "z": tp[:,2]}).to_csv(out, index=False)
            print(f"  saved {out.name}  (active={na})")
        # L2 vs v122c
        c22 = np.load(CACHE / "v122c_v121diverse_weights.npz")["test_pred"]
        print(f"  L2(cons vs v122c)={np.linalg.norm(test_c-c22,axis=-1).mean()*1000:.3f}mm")
        print(f"  L2(full vs v122c)={np.linalg.norm(test_f-c22,axis=-1).mean()*1000:.3f}mm")

    if args.mode in ("nestedcv", "both"):
        print(f"\n=== NESTED-CV (honest generalization, {args.outer_folds} outer folds) ===")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.outer_folds, shuffle=True, random_state=42)
        N = y.shape[0]
        # compare two recipes honestly: conservative-subset vs full-pool
        recipes = {
            "conservative": conservative_subset(pool, y, force, top_k=8, oof_floor=0.67),
            "full": list(range(len(pool))),
        }
        for rname, idxs in recipes.items():
            sub_oofs = oofs[idxs]
            val_hits = []
            for tr, va in kf.split(np.arange(N)):
                w, _ = de_fit(sub_oofs[:, tr], y[tr], n_iter=120, popsize=25, n_starts=2)
                pred_va = (w[:, None, None] * sub_oofs[:, va]).sum(0)
                val_hits.append(hit(pred_va, y[va]))
            insample_w, insample_rh = de_fit(sub_oofs, y, n_iter=150, popsize=25, n_starts=2)
            honest = float(np.mean(val_hits))
            print(f"  [{rname:<12}] in-sample OOF={insample_rh:.4f}  honest nested-CV={honest:.4f}  "
                  f"overfit gap={insample_rh-honest:+.4f}  active={int((insample_w>=0.01).sum())}")

if __name__ == "__main__":
    main()

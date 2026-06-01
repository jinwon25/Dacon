"""v141_decision.py — 최종 제출 결정: 새 conservative 블렌드 vs 구pool 기준선 (정직 nested-CV).

질문: 신규 Frenet/GRU/control-head 멤버를 넣은 conservative 블렌드가 정직한 일반화(nested-CV)로
구pool(v122c-era) 대비 실제로 개선되는가? 둘 다 동일 recipe(top-K + force diverse)로 nested-CV.

출력: 새 conservative 블렌드 submission 저장 + in-sample/honest OOF 비교 + 신규멤버 weight.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except: pass
from v132_final_blend import (load_pool, add_new_members, conservative_subset, de_fit, hit)
from v110_de_ensemble import load_y_and_sub
from sklearn.model_selection import KFold
CACHE = Path("data/cache"); DATA = Path("data")

def nested_cv(oofs_subset, y, outer=5, n_iter=110, popsize=22, n_starts=2):
    kf = KFold(n_splits=outer, shuffle=True, random_state=42)
    N = y.shape[0]; vh = []
    for k, (tr, va) in enumerate(kf.split(np.arange(N))):
        w, _ = de_fit(oofs_subset[:, tr], y[tr], n_iter=n_iter, popsize=popsize, n_starts=n_starts)
        vh.append(hit((w[:, None, None] * oofs_subset[:, va]).sum(0), y[va]))
        print(f"    outer fold {k+1}/{outer}: val hit={vh[-1]:.4f}", flush=True)
    return float(np.mean(vh))

def main():
    pool, y = load_pool(include_mdn=True)
    pool, added = add_new_members(pool, y)
    names = [p[0] for p in pool]; oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    print(f"pool={len(pool)}  new added={len(added)}", flush=True)
    force = [n for n in names if n.startswith("v131") or n.startswith("v135")
             or n in ("v120", "v121", "v121c5", "v120_big")]

    # ========== NEW pool conservative blend ==========
    cons = conservative_subset(pool, y, force, top_k=8, oof_floor=0.67)
    w, rh = de_fit(oofs[cons], y, n_iter=200, popsize=30, n_starts=5)
    test_c = (w[:, None, None] * tests[cons]).sum(0)
    n_active = int((w >= 0.01).sum())
    print(f"\n[NEW conservative] in-sample OOF={rh:.4f}  active={n_active}", flush=True)
    new_w = 0.0
    for j in np.argsort(-w):
        if w[j] >= 0.01:
            nm = names[cons[j]]; tag = "NEW" if (nm.startswith("v131") or nm.startswith("v135")) else ""
            if tag: new_w += w[j]
            print(f"    {nm:<22} w={w[j]:.3f} oof={pool[cons[j]][3]:.4f} {tag}", flush=True)
    print(f"    new-member weight sum = {new_w:.3f}", flush=True)

    # save submission
    _, sub, _ = load_y_and_sub()
    out = DATA / f"submission_v141_newconservative_oof{rh:.4f}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_c[:,0], "y": test_c[:,1], "z": test_c[:,2]}).to_csv(out, index=False)
    print(f"    saved {out.name}", flush=True)
    # L2 vs v122c
    c22 = np.load(CACHE / "v122c_v121diverse_weights.npz")["test_pred"]
    print(f"    L2(new-cons vs v122c)={np.linalg.norm(test_c-c22,axis=-1).mean()*1000:.2f}mm", flush=True)

    # ========== OLD pool (v122c-era) conservative blend — baseline ==========
    old_idx = [i for i, n in enumerate(names) if not (n.startswith("v131") or n.startswith("v135"))]
    old_pool = [pool[i] for i in old_idx]; old_oofs = oofs[old_idx]
    old_names = [p[0] for p in old_pool]
    old_force = [n for n in old_names if n in ("v120", "v121", "v121c5", "v120_big")]
    old_cons_local = conservative_subset(old_pool, y, old_force, top_k=8, oof_floor=0.67)
    w_o, rh_o = de_fit(old_oofs[old_cons_local], y, n_iter=200, popsize=30, n_starts=5)
    print(f"\n[OLD conservative] in-sample OOF={rh_o:.4f}  active={int((w_o>=0.01).sum())}", flush=True)

    # ========== NESTED-CV honest comparison ==========
    print(f"\n=== NESTED-CV honest generalization (5 outer folds) ===", flush=True)
    print(f"  [NEW conservative] (subset n={len(cons)}):", flush=True)
    honest_new = nested_cv(oofs[cons], y)
    print(f"  [OLD conservative] (subset n={len(old_cons_local)}):", flush=True)
    honest_old = nested_cv(old_oofs[old_cons_local], y)

    print(f"\n========== DECISION SUMMARY ==========", flush=True)
    print(f"  v122c reference:      in-sample OOF 0.6769 -> LB 0.6912 (변환 +0.0143)", flush=True)
    print(f"  OLD conservative:     in-sample {rh_o:.4f}  honest nested-CV {honest_old:.4f}  (overfit {rh_o-honest_old:+.4f})", flush=True)
    print(f"  NEW conservative:     in-sample {rh:.4f}  honest nested-CV {honest_new:.4f}  (overfit {rh-honest_new:+.4f})", flush=True)
    print(f"  --> honest lift NEW vs OLD = {honest_new-honest_old:+.4f}", flush=True)
    print(f"  --> new-member weight {new_w:.3f}, active {n_active}", flush=True)
    if honest_new > honest_old:
        est = 0.6912 + (honest_new - honest_old)
        print(f"  ==> NEW blend honestly beats OLD. 예상 LB ~ {est:.4f} (v122c 0.6912 + honest lift)", flush=True)
    else:
        print(f"  ==> no honest lift; v122c 0.6912 안전 제출 유지 권고", flush=True)

if __name__ == "__main__":
    main()

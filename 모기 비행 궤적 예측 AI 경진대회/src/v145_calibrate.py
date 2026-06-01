"""v145_calibrate — hit@1cm 직접 최적화 후처리 (무학습 quick win).

리서치 근거:
- Trojan_Horse 공개: per-axis 보정 [1,0.95,1] (수직축 5% 축소) — DE 블렌드가 못 하는 post-multiply.
- hit@1cm은 hard-threshold라 variance가 bias를 지배하면 예측을 anchor로 당길수록 ε-ball 적중↑.
- Caruana greedy(실제 hit 직접 최적화, bagging)로 DE softmax-surrogate 대비 비교.

방법: 현재 best 블렌드(v143b: s4+n2 conservative)를 재구성 → OOF/test 예측 →
  (A) per-axis α: pred_cal = anchor + α⊙(pred-anchor), anchor∈{last_obs, CV}. OOF hit로 그리드.
  (B) Caruana greedy hit@1cm (bagged) vs DE.
출력: 보정 전/후 OOF hit + 보정된 test 제출 저장.
usage: python scripts/v145_calibrate.py
"""
from __future__ import annotations
import sys, itertools, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception: pass
from v132_final_blend import add_new_members, de_fit, hit
from v110_de_ensemble import load_pool, load_y_and_sub
from v23_train import load_data, DT, T_PRED
DATA = Path("data")
def isnew(nm): return any(t in nm for t in ["frenet_gru_s4","frenet_gru_n2","frenet_gru_big","frenet_gru_n2s4","frenet_gru_n4"])
def hit3(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())

def build_subset(names, pool, allowed, force_names, top_k=8, floor=0.67):
    aidx = [i for i in range(len(names)) if names[i] in allowed]
    chosen = []
    for i in sorted(aidx, key=lambda i: -pool[i][3]):
        if pool[i][3] >= floor and len(chosen) < top_k: chosen.append(i)
    for i in aidx:
        if names[i] in force_names and i not in chosen: chosen.append(i)
    return chosen

def main():
    X_train, X_test, y_train, sub = load_data()
    pool, y = load_pool(include_mdn=True)
    pool, _ = add_new_members(pool, y)
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    print(f"pool={len(pool)}  y_train={y_train.shape}", flush=True)

    # --- rebuild v143b (s4+n2) conservative blend ---
    nonnew = {n for n in names if not isnew(n)}
    allowed = nonnew | {n for n in names if ("frenet_gru_s4" in n or "frenet_gru_n2" in n)}
    force_nonnew = [n for n in names if (n.startswith("v131") or n.startswith("v135")
                    or n in ("v120","v121","v121c5","v120_big")) and not isnew(n)]
    force = force_nonnew + [n for n in allowed if isnew(n)]
    cons = build_subset(names, pool, allowed, force)
    w, rh = de_fit(oofs[cons], y, n_iter=180, popsize=26, n_starts=3)
    oof_blend = (w[:, None, None] * oofs[cons]).sum(0)
    test_blend = (w[:, None, None] * tests[cons]).sum(0)
    base_hit = hit3(oof_blend, y_train)
    print(f"[v143b blend] OOF hit={base_hit:.4f}  active={int((w>=0.01).sum())}", flush=True)

    last_tr = X_train[:, -1, :]; last_te = X_test[:, -1, :]
    v_tr = (X_train[:, -1] - X_train[:, -2]) / DT; v_te = (X_test[:, -1] - X_test[:, -2]) / DT
    cv_tr = last_tr + v_tr * T_PRED; cv_te = last_te + v_te * T_PRED   # constant-velocity anchor

    # === (A) per-axis shrinkage: pred_cal = anchor + alpha*(pred-anchor) ===
    grid = np.round(np.arange(0.80, 1.121, 0.02), 3)
    for aname, anc_tr, anc_te in [("last_obs", last_tr, last_te), ("CV", cv_tr, cv_te)]:
        d = oof_blend - anc_tr  # OOF displacement from anchor
        best = (base_hit, (1.0, 1.0, 1.0))
        # coordinate-ascent ×3 passes (axes interact via 3D distance)
        alpha = [1.0, 1.0, 1.0]
        for _ in range(3):
            for ax in range(3):
                hits = []
                for a in grid:
                    al = alpha.copy(); al[ax] = a
                    p = anc_tr + np.array(al) * d
                    hits.append(hit3(p, y_train))
                alpha[ax] = float(grid[int(np.argmax(hits))])
            cur = hit3(anc_tr + np.array(alpha) * d, y_train)
            if cur > best[0]: best = (cur, tuple(alpha))
        print(f"\n[per-axis α | anchor={aname}] best OOF hit={best[0]:.4f} (Δ{best[0]-base_hit:+.4f})  α={best[1]}", flush=True)
        if best[0] > base_hit + 0.0002:
            al = np.array(best[1])
            test_cal = anc_te + al * (test_blend - anc_te)
            out = DATA / f"submission_v145cal_{aname}_a{al[0]:.2f}_{al[1]:.2f}_{al[2]:.2f}_oof{best[0]:.4f}.csv"
            pd.DataFrame({"id": sub["id"], "x": test_cal[:,0], "y": test_cal[:,1], "z": test_cal[:,2]}).to_csv(out, index=False)
            print(f"    saved {out.name}", flush=True)

    # === (B) Caruana greedy hit@1cm (bagged) vs DE ===
    print("\n[Caruana greedy hit@1cm, 20 bags × 60 picks-with-replacement]", flush=True)
    rng = np.random.RandomState(0); M = len(pool); N = len(y_train)
    bag_test = np.zeros_like(test_blend); bag_oof = np.zeros_like(oof_blend); nb = 0
    for b in range(20):
        members = rng.choice(M, size=max(8, M//2), replace=False)
        ens = []  # indices into pool
        cur = np.zeros((N, 3))
        for pick in range(60):
            best_i, best_h = None, -1
            for mi in members:
                cand = (cur * len(ens) + oofs[mi]) / (len(ens) + 1)
                h = hit3(cand, y_train)
                if h > best_h: best_h, best_i = h, mi
            ens.append(best_i); cur = (cur * (len(ens)-1) + oofs[best_i]) / len(ens)
        bag_oof += cur; bag_test += np.mean([tests[i] for i in ens], axis=0); nb += 1
    bag_oof /= nb; bag_test /= nb
    greedy_hit = hit3(bag_oof, y_train)
    print(f"  Caruana bagged OOF hit={greedy_hit:.4f} (vs DE {base_hit:.4f}, Δ{greedy_hit-base_hit:+.4f})", flush=True)
    if greedy_hit > base_hit + 0.0002:
        out = DATA / f"submission_v145greedy_oof{greedy_hit:.4f}.csv"
        pd.DataFrame({"id": sub["id"], "x": bag_test[:,0], "y": bag_test[:,1], "z": bag_test[:,2]}).to_csv(out, index=False)
        print(f"  saved {out.name}", flush=True)

    print("\n[done] 보정으로 OOF hit가 의미있게(+0.0003↑) 오르면 제출 후보, 아니면 v143b 유지.", flush=True)

if __name__ == "__main__":
    main()

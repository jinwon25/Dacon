"""v144_reblend_calibrate — n2s4(multistep+시드앙상블)/n4(4-step) 포함 재블렌드 + per-axis hit@1cm 보정.

전략: v143b(LB 0.699)는 s4+n2였음. n2s4(OOF 0.6801, 역대최고)+n4 추가 → DE 재블렌드.
이어서 Trojan_Horse식 per-axis shrinkage 보정(hit@1cm 직접 최적화, DE가 못 하는 post-multiply).
usage: python scripts/v144_reblend_calibrate.py
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception: pass
from v132_final_blend import add_new_members, de_fit, hit
from v110_de_ensemble import load_pool, load_y_and_sub
from v23_train import load_data, DT, T_PRED
DATA = Path("data"); FINAL = Path("final_candidates"); CACHE = Path("data/cache")
NEW_KEEP = ["frenet_gru_s4", "frenet_gru_n2", "frenet_gru_n2s4", "frenet_gru_n4"]   # big 제외(DE가 거부했었음)
def isnew(nm): return any(t in nm for t in ["frenet_gru_s4","frenet_gru_n2","frenet_gru_big","frenet_gru_n2s4","frenet_gru_n4","frenet_transformer","frenet_lru","frenet_tcn"])
def hit3(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())
def decorr(a, b): return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)

def main():
    X_train, X_test, y_train, sub = load_data()
    pool, y = load_pool(include_mdn=True)
    pool, added = add_new_members(pool, y)
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    print(f"pool={len(pool)}", flush=True)

    # 새 멤버 진단 (버그검증)
    v120i = names.index("v120")
    print("=== new member diagnostic ===")
    for nm in [n for n in names if any(k in n for k in NEW_KEEP) and ("_c1" in n)]:
        i = names.index(nm); d_o = decorr(oofs[i], oofs[v120i]); d_t = decorr(tests[i], tests[v120i])
        ok = d_t <= 3.0 * max(d_o, 0.3) + 2.0
        print(f"  {nm:<26} OOF={pool[i][3]:.4f} dTEST={d_t:.2f}mm {'OK' if ok else '!!BUG'}", flush=True)

    # === best conservative blend (force 강한 multistep/seed 멤버) ===
    nonnew = {n for n in names if not isnew(n)}
    allowed = nonnew | {n for n in names if any(k in n for k in NEW_KEEP)}
    force_nonnew = [n for n in names if (n.startswith("v131") or n.startswith("v135")
                    or n in ("v120","v121","v121c5","v120_big")) and not isnew(n)]
    force = force_nonnew + [n for n in allowed if isnew(n)]
    aidx = [i for i in range(len(names)) if names[i] in allowed]
    chosen = []
    for i in sorted(aidx, key=lambda i: -pool[i][3]):
        if pool[i][3] >= 0.67 and len(chosen) < 8: chosen.append(i)
    for i in aidx:
        if names[i] in force and i not in chosen: chosen.append(i)
    w, rh = de_fit(oofs[chosen], y, n_iter=200, popsize=30, n_starts=4)
    oof_blend = (w[:, None, None] * oofs[chosen]).sum(0)
    test_blend = (w[:, None, None] * tests[chosen]).sum(0)
    base_hit = hit3(oof_blend, y_train)
    print(f"\n[v144 blend] OOF hit={base_hit:.4f}  active={int((w>=0.01).sum())}", flush=True)
    neww = 0.0
    for j in np.argsort(-w):
        if w[j] >= 0.01:
            nm = names[chosen[j]]
            if isnew(nm): neww += w[j]
            print(f"    {nm:<26} w={w[j]:.3f} oof={pool[chosen[j]][3]:.4f} {'NEW' if isnew(nm) else ''}", flush=True)
    print(f"    new-member weight={neww:.3f}", flush=True)
    v143b = pd.read_csv(DATA / "submission_v143b_s4n2_oof0.6819.csv")[["x","y","z"]].to_numpy()
    print(f"    L2 vs v143b(LB0.699)={decorr(test_blend, v143b):.2f}mm", flush=True)
    out_raw = DATA / f"submission_v144blend_oof{rh:.4f}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_blend[:,0], "y": test_blend[:,1], "z": test_blend[:,2]}).to_csv(out_raw, index=False)
    print(f"    saved {out_raw.name}", flush=True)

    # === per-axis shrinkage 보정 (hit@1cm 직접) ===
    last_tr = X_train[:, -1]; last_te = X_test[:, -1]
    v_tr = (X_train[:,-1]-X_train[:,-2])/DT; v_te = (X_test[:,-1]-X_test[:,-2])/DT
    cv_tr = last_tr + v_tr*T_PRED; cv_te = last_te + v_te*T_PRED
    grid = np.round(np.arange(0.80, 1.121, 0.01), 3)
    best_overall = (base_hit, None, None, "none")
    for aname, anc_tr, anc_te in [("last_obs", last_tr, last_te), ("CV", cv_tr, cv_te)]:
        d = oof_blend - anc_tr; alpha = [1.0,1.0,1.0]
        for _ in range(4):
            for ax in range(3):
                hits = [hit3(anc_tr + np.array([alpha[0] if k!=ax else a for k in range(3)])*d, y_train) for a in grid]
                alpha[ax] = float(grid[int(np.argmax(hits))])
        h = hit3(anc_tr + np.array(alpha)*d, y_train)
        print(f"\n[per-axis α | {aname}] OOF hit={h:.4f} (Δ{h-base_hit:+.4f}) α={tuple(round(a,3) for a in alpha)}", flush=True)
        if h > best_overall[0]:
            test_cal = anc_te + np.array(alpha)*(test_blend - anc_te)
            best_overall = (h, test_cal, tuple(round(a,3) for a in alpha), aname)

    # === 최종 후보 저장 ===
    if best_overall[1] is not None and best_overall[0] > base_hit + 0.0002:
        h, test_cal, alpha, aname = best_overall
        out = DATA / f"submission_v144cal_{aname}_oof{h:.4f}.csv"
        pd.DataFrame({"id": sub["id"], "x": test_cal[:,0], "y": test_cal[:,1], "z": test_cal[:,2]}).to_csv(out, index=False)
        print(f"\n★ 보정 후보 저장: {out.name}  (anchor={aname}, α={alpha}, OOF {base_hit:.4f}→{h:.4f})", flush=True)
        print(f"  L2(보정 vs v143b)={decorr(test_cal, v143b):.2f}mm", flush=True)
    else:
        print(f"\n보정 효과 미미 → raw 블렌드({out_raw.name}) 사용. OOF {base_hit:.4f}", flush=True)
    print(f"\n[요약] v144 블렌드 OOF {rh:.4f} (v143b 0.6819 / LB 0.699 대비). 신규 multistep weight {neww:.3f}.", flush=True)

if __name__ == "__main__":
    main()

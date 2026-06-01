"""v146_reblend_calibrate — 새 인코더(transformer/lru/tcn) + 최강 multistep(n2s4) 포함 재블렌드 + 보정.

핵심 검증: 인코더 멤버가 frenet_gru/multistep과 decorrelated해서 블렌드에 새 정보를 더하는가?
(multistep은 소진됐었음 — 같은 메커니즘 정제라). 인코더는 GRU/ODE와 다른 메커니즘이라 기대.
usage: python scripts/v146_reblend_calibrate.py
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
DATA = Path("data"); CACHE = Path("data/cache")
# 강한 multistep + 새 인코더를 후보로 (n4/big 제외: 약함/거부)
NEW_KEEP = ["frenet_gru_s4", "frenet_gru_n2", "frenet_gru_n2s4",
            "frenet_transformer", "frenet_lru", "frenet_tcn"]
ALL_NEW = ["frenet_gru_s4","frenet_gru_n2","frenet_gru_big","frenet_gru_n2s4","frenet_gru_n4",
           "frenet_transformer","frenet_lru","frenet_tcn","frenet_itransformer"]
def isnew(nm): return any(t in nm for t in ALL_NEW)
def hit3(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())
def decorr(a, b): return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)

def main():
    X_train, X_test, y_train, sub = load_data()
    pool, y = load_pool(include_mdn=True)
    pool, _ = add_new_members(pool, y)
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    idx = {n: i for i, n in enumerate(names)}
    print(f"pool={len(pool)}", flush=True)

    # 진단: 인코더 멤버가 dominant(frenet_gru_c10)와 decorrelated한가?
    dom = idx.get("v131frenet_gru_c10"); v120i = idx["v120"]
    print("=== encoder member diagnostic (dTEST vs v120 / vs frenet_gru_c10) ===")
    for nm in [n for n in names if any(e in n for e in ["frenet_transformer","frenet_lru","frenet_tcn"]) and "_c1" in n]:
        i = idx[nm]; d120 = decorr(tests[i], tests[v120i]); ddom = decorr(tests[i], tests[dom]) if dom else -1
        ok = d120 <= 3.0*max(decorr(oofs[i],oofs[v120i]),0.3)+2.0
        print(f"  {nm:<28} OOF={pool[i][3]:.4f} dV120={d120:.2f} dGRU={ddom:.2f}mm {'OK' if ok else '!!BUG'}", flush=True)

    # 블렌드: nonnew + 강한 multistep + 인코더, force-include
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
    w, rh = de_fit(oofs[chosen], y, n_iter=180, popsize=28, n_starts=3)
    oof_blend = (w[:,None,None]*oofs[chosen]).sum(0); test_blend = (w[:,None,None]*tests[chosen]).sum(0)
    base_hit = hit3(oof_blend, y_train)
    print(f"\n[v146 blend] OOF hit={base_hit:.4f} active={int((w>=0.01).sum())}", flush=True)
    enc_w = 0.0
    for j in np.argsort(-w):
        if w[j] >= 0.01:
            nm = names[chosen[j]]
            is_enc = any(e in nm for e in ["frenet_transformer","frenet_lru","frenet_tcn"])
            if is_enc: enc_w += w[j]
            print(f"    {nm:<28} w={w[j]:.3f} oof={pool[chosen[j]][3]:.4f} {'ENC' if is_enc else ('NEW' if isnew(nm) else '')}", flush=True)
    v143b = pd.read_csv(DATA/"submission_v143b_s4n2_oof0.6819.csv")[["x","y","z"]].to_numpy()
    print(f"    encoder weight={enc_w:.3f}   L2 vs v143b(LB0.699)={decorr(test_blend,v143b):.2f}mm", flush=True)
    out = DATA/f"submission_v146blend_oof{rh:.4f}.csv"
    pd.DataFrame({"id":sub["id"],"x":test_blend[:,0],"y":test_blend[:,1],"z":test_blend[:,2]}).to_csv(out,index=False)
    print(f"    saved {out.name}", flush=True)

    # per-axis 보정
    last_tr=X_train[:,-1]; last_te=X_test[:,-1]
    v_tr=(X_train[:,-1]-X_train[:,-2])/DT; v_te=(X_test[:,-1]-X_test[:,-2])/DT
    cv_tr=last_tr+v_tr*T_PRED; cv_te=last_te+v_te*T_PRED
    grid=np.round(np.arange(0.80,1.121,0.01),3); best=(base_hit,None,None,"none")
    for an,atr,ate in [("last_obs",last_tr,last_te),("CV",cv_tr,cv_te)]:
        d=oof_blend-atr; al=[1.,1.,1.]
        for _ in range(4):
            for ax in range(3):
                hh=[hit3(atr+np.array([al[k] if k!=ax else a for k in range(3)])*d,y_train) for a in grid]
                al[ax]=float(grid[int(np.argmax(hh))])
        h=hit3(atr+np.array(al)*d,y_train)
        print(f"[per-axis α|{an}] OOF={h:.4f} (Δ{h-base_hit:+.4f}) α={tuple(round(a,3) for a in al)}", flush=True)
        if h>best[0]: best=(h, ate+np.array(al)*(test_blend-ate), tuple(round(a,3) for a in al), an)
    if best[1] is not None and best[0]>base_hit+0.0002:
        h,tc,al,an=best
        o2=DATA/f"submission_v146cal_{an}_oof{h:.4f}.csv"
        pd.DataFrame({"id":sub["id"],"x":tc[:,0],"y":tc[:,1],"z":tc[:,2]}).to_csv(o2,index=False)
        print(f"\n★ 보정 후보: {o2.name} (α={al}, OOF {base_hit:.4f}→{h:.4f}, L2vs143b={decorr(tc,v143b):.2f}mm)", flush=True)
    print(f"\n[요약] v146 OOF {rh:.4f} (v143b 0.6819 / LB 0.699). 인코더 weight {enc_w:.3f}. "
          f"{'인코더가 새 정보 추가 → 제출 후보' if enc_w>=0.08 else '인코더 기여 미미'}", flush=True)

if __name__ == "__main__":
    main()

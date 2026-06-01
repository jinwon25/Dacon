"""v148_reblend — CREE HyperPhysics_xy2(회전 기반 turn-rate 물리, frenet과 2.82mm decorrelated) 멤버를
블렌드에 통합. 강한 multistep(n2s4/n2) + CREE를 force → DE 재블렌드 + per-axis 보정.

CREE는 v131 계열이 아니라 별도 로딩(cree_xy2_state + cap10/15). 우리 frenet-neural 클러스터 밖이라
인코더(거부)와 달리 weight를 받을 것으로 기대 → 0.699 돌파 후보.
usage: python scripts/v148_reblend.py
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
CACHE = Path("data/cache"); DATA = Path("data")
NEW_KEEP = ["frenet_gru_s4", "frenet_gru_n2", "frenet_gru_n2s4"]   # 검증된 강한 multistep
CREE = ["cree_xy2", "cree_xy2_c10", "cree_xy2_c15"]
def iskeep(nm): return any(k in nm for k in NEW_KEEP) or nm in CREE
def iscree(nm): return nm in CREE
def hit3(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())
def decorr(a, b): return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)

def load_cree(pool, y):
    """cree_xy2 raw + boundary 멤버를 pool에 추가 (존재 시)."""
    added = []
    p = CACHE / "cree_xy2_state.npz"
    if p.exists():
        st = np.load(p); oof = st["oof_global"].astype(np.float64); test = st["test_global"].astype(np.float64)
        pool.append(("cree_xy2", oof, test, hit(oof, y))); added.append("cree_xy2")
    for cap in ["10", "15"]:
        p = CACHE / f"cree_xy2_cap{cap}_state.npz"
        if p.exists():
            st = np.load(p); oof = st["oof_v91"].astype(np.float64); test = st["test_v91"].astype(np.float64)
            pool.append((f"cree_xy2_c{cap}", oof, test, hit(oof, y))); added.append(f"cree_xy2_c{cap}")
    return pool, added

def main():
    X_train, X_test, y_train, sub = load_data()
    pool, y = load_pool(include_mdn=True)
    pool, _ = add_new_members(pool, y)
    pool, cree_added = load_cree(pool, y)
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    idx = {n: i for i, n in enumerate(names)}
    print(f"pool={len(pool)}  cree added={cree_added}", flush=True)
    if not cree_added:
        print("[!!] CREE 멤버 없음 — 학습 미완료. 중단."); return

    # CREE 진단
    fg = idx.get("v131frenet_gru_c10")
    print("=== CREE diagnostic ===")
    for nm in CREE:
        if nm in idx:
            i = idx[nm]; d_fg = decorr(tests[i], tests[fg]) if fg else -1
            print(f"  {nm:<14} OOF={pool[i][3]:.4f}  dFRENET={d_fg:.2f}mm", flush=True)

    # 블렌드: nonnew + 강한 multistep + CREE
    keep_neural = {n for n in names if not (any(t in n for t in ["frenet_gru_s4","frenet_gru_n2","frenet_gru_big","frenet_gru_n2s4","frenet_gru_n4","frenet_transformer","frenet_lru","frenet_tcn","frenet_itransformer"]) or n in CREE)}
    allowed = keep_neural | {n for n in names if iskeep(n)}
    # diverse anchor만 force (top-8 OOF가 강한 frenet 멤버는 알아서 선택) → subset 축소로 de_fit 가속
    force_base = [n for n in names if n in ("v120","v121","v121c5") and n in keep_neural]
    force = force_base + [n for n in allowed if iskeep(n)]
    aidx = [i for i in range(len(names)) if names[i] in allowed]
    chosen = []
    for i in sorted(aidx, key=lambda i: -pool[i][3]):
        if pool[i][3] >= 0.67 and len(chosen) < 8: chosen.append(i)
    for i in aidx:
        if names[i] in force and i not in chosen: chosen.append(i)
    print(f"  subset size={len(chosen)}, fitting DE (lighter)...", flush=True)
    w, rh = de_fit(oofs[chosen], y, n_iter=140, popsize=22, n_starts=2)
    oof_blend = (w[:,None,None]*oofs[chosen]).sum(0); test_blend = (w[:,None,None]*tests[chosen]).sum(0)
    base_hit = hit3(oof_blend, y_train)
    print(f"\n[v148 blend] OOF hit={base_hit:.4f} active={int((w>=0.01).sum())}", flush=True)
    cw = 0.0
    for j in np.argsort(-w):
        if w[j] >= 0.01:
            nm = names[chosen[j]]
            if iscree(nm): cw += w[j]
            print(f"    {nm:<26} w={w[j]:.3f} oof={pool[chosen[j]][3]:.4f} {'CREE' if iscree(nm) else ('keep' if iskeep(nm) else '')}", flush=True)
    _v143b_path = DATA/"submission_v143b_s4n2_oof0.6819.csv"   # 진단용 참조(선택). 없으면 base 자기참조로 폴백.
    v143b = pd.read_csv(_v143b_path)[["x","y","z"]].to_numpy() if _v143b_path.exists() else test_blend
    print(f"    CREE weight={cw:.3f}   L2 vs v143b(LB0.699)={decorr(test_blend,v143b):.2f}mm", flush=True)
    out = DATA/f"submission_v148blend_oof{rh:.4f}.csv"
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
        o2=DATA/f"submission_v148cal_{an}_oof{h:.4f}.csv"
        pd.DataFrame({"id":sub["id"],"x":tc[:,0],"y":tc[:,1],"z":tc[:,2]}).to_csv(o2,index=False)
        print(f"\n★ 보정 후보: {o2.name} (α={al}, OOF {base_hit:.4f}→{h:.4f}, L2vs143b={decorr(tc,v143b):.2f}mm)", flush=True)
    # === CREE-forward: 가장 decorrelated한 raw cree(2.82mm)를 더 주입 (over-conversion 베팅) ===
    if "cree_xy2" in idx:
        co = oofs[idx["cree_xy2"]]; ct = tests[idx["cree_xy2"]]
        print("\n=== CREE-forward mix (raw cree 추가 주입, L2 키우기) ===", flush=True)
        for a in [0.12, 0.18, 0.25, 0.32]:
            mo = (1-a)*oof_blend + a*co; mt = (1-a)*test_blend + a*ct
            print(f"  α={a}: OOF={hit3(mo,y_train):.4f}  L2vs143b={decorr(mt,v143b):.2f}mm", flush=True)
        a = 0.18; mt = (1-a)*test_blend + a*ct; mo = hit3((1-a)*oof_blend + a*co, y_train)
        o3 = DATA/f"submission_v148creefwd_a{a}_oof{mo:.4f}.csv"
        pd.DataFrame({"id":sub["id"],"x":mt[:,0],"y":mt[:,1],"z":mt[:,2]}).to_csv(o3,index=False)
        print(f"  ★ saved {o3.name}  (L2vs143b={decorr(mt,v143b):.2f}mm — v143b가 v141 이긴 0.66mm와 비교)", flush=True)
    print(f"\n[요약] v148 OOF {rh:.4f}. CREE weight {cw:.3f}. "
          f"{'★ CREE가 새 정보 추가 → 강력한 제출 후보' if cw>=0.08 else 'CREE 기여 미미(예상밖)'}", flush=True)

if __name__ == "__main__":
    main()

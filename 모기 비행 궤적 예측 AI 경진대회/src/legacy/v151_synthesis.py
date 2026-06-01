"""v151_synthesis — 3개 독립 메커니즘 종합: frenet-RK4 + CREE-회전물리 + Flow-직접.
base 블렌드(frenet+CREE-bd+Flow-bd) + raw CREE/Flow 수동주입(over-conversion). 5슬롯용 후보 스프레드 생성.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception: pass
from v132_final_blend import add_new_members, de_fit, hit
from v110_de_ensemble import load_pool
from v23_train import load_data
from v148_reblend import load_cree, hit3, decorr
DATA = Path("data"); CACHE = Path("data/cache")
KEEP = ["frenet_gru_s4", "frenet_gru_n2", "frenet_gru_n2s4", "frenet_flow", "yaw_flow", "cree_xy2"]
def iskeep(nm): return any(k in nm for k in KEEP)

def main():
    X_train, X_test, y_train, sub = load_data()
    pool, y = load_pool(include_mdn=True)
    pool, _ = add_new_members(pool, y)
    pool, cree_added = load_cree(pool, y)
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    idx = {n: i for i, n in enumerate(names)}
    print(f"pool={len(pool)} cree={cree_added}", flush=True)
    frg = idx.get("v131frenet_gru_c10")
    print("=== mechanism members (OOF / dFRENET) ===")
    for nm in ["cree_xy2", "cree_xy2_c10", "v131frenet_flow_c10", "v131yaw_flow_c10", "v131_yaw_flow"]:
        if nm in idx:
            i = idx[nm]; print(f"  {nm:<22} OOF={pool[i][3]:.4f} dFRENET={decorr(tests[i], tests[frg]):.2f}mm", flush=True)

    allowed = {n for n in names if iskeep(n) or n in ("v120","v121","v121c5","v53","v131frenet_gru_c10","v131frenet_gru_c15")}
    force = ["v120","v121","v121c5"] + [n for n in names if iskeep(n) and ("_c1" in n or n == "cree_xy2")]
    aidx = [i for i in range(len(names)) if names[i] in allowed]
    chosen = []
    for i in sorted(aidx, key=lambda i: -pool[i][3]):
        if pool[i][3] >= 0.67 and len(chosen) < 8: chosen.append(i)
    for i in aidx:
        if names[i] in force and i not in chosen: chosen.append(i)
    print(f"\nsubset={len(chosen)}, fitting DE...", flush=True)
    w, rh = de_fit(oofs[chosen], y, n_iter=140, popsize=22, n_starts=2)
    oof_b = (w[:,None,None]*oofs[chosen]).sum(0); test_b = (w[:,None,None]*tests[chosen]).sum(0)
    print(f"[base blend] OOF={hit3(oof_b,y_train):.4f} active={int((w>=0.01).sum())}", flush=True)
    for j in np.argsort(-w):
        if w[j] >= 0.01: print(f"    {names[chosen[j]]:<22} w={w[j]:.3f} oof={pool[chosen[j]][3]:.4f}", flush=True)

    v143b = pd.read_csv(DATA/"submission_v143b_s4n2_oof0.6819.csv")[["x","y","z"]].to_numpy()
    cree = tests[idx["cree_xy2"]]; cree_o = oofs[idx["cree_xy2"]]
    flow = tests[idx["v131_yaw_flow"]]; flow_o = oofs[idx["v131_yaw_flow"]]   # 가장 직교한 raw flow

    print("\n=== injection candidates (3-mechanism) ===", flush=True)
    for a, b, nm in [(0.25,0,"cree25"), (0.30,0,"cree30"), (0.20,0.10,"cree20flow10"),
                     (0.25,0.10,"cree25flow10"), (0.15,0.15,"cree15flow15"), (0,0.20,"flow20")]:
        mt = (1-a-b)*test_b + a*cree + b*flow
        mo = hit3((1-a-b)*oof_b + a*cree_o + b*flow_o, y_train)
        L2 = decorr(mt, v143b)
        out = DATA/f"submission_v151_{nm}_oof{mo:.4f}.csv"
        pd.DataFrame({"id":sub["id"],"x":mt[:,0],"y":mt[:,1],"z":mt[:,2]}).to_csv(out, index=False)
        print(f"  {nm:<16} OOF={mo:.4f} L2vs143b={L2:.2f}mm  saved {out.name}", flush=True)
    print("\n[done] v151 종합 후보 준비 (frenet+CREE+Flow 3메커니즘)", flush=True)

if __name__ == "__main__":
    main()

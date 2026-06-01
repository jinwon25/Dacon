"""v79_grid_v78.py — v78 추가 후 exhaustive 3/4-way grid + SoftStacker.

v78 OOF 0.6730 (v35 0.6725 + 0.0005, paradigm 다름).
v48 3-way (v48_9m / v46_7m / v35)에 v78 추가 → 4-way / 5-way grid.
"""
import sys, glob, os, json, datetime as _dt
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data

PROJECT = SCRIPT_DIR.parent
CACHE = PROJECT / "data/cache"
DATA = PROJECT / "data"


def rh(p, y): return (np.linalg.norm(p - y, axis=-1) <= 0.01).mean()


def main():
    X_train, X_test, y_train, sub = load_data()
    v48s = np.load(CACHE / "v48_state.npz"); v46s = np.load(CACHE / "v46_state.npz")
    st35 = np.load(CACHE / "v35_state.npz")
    st78 = np.load(CACHE / "v78_state.npz")
    st44 = np.load(CACHE / "v44_state.npz")
    st39 = np.load(CACHE / "v39_state.npz")
    pool_o = {
        "v48_9m": v48s["oof_v48"], "v46_7m": v46s["oof_v46"],
        "v35": st35["oof_v35"].astype(np.float64),
        "v78": st78["oof_v78"].astype(np.float64),
        "v44": st44["oof_v44"].astype(np.float64),
        "v39": st39["oof_v39"].astype(np.float64),
    }
    pool_t = {
        "v48_9m": v48s["test_v48"], "v46_7m": v46s["test_v46"],
        "v35": st35["test_v35"].astype(np.float64),
        "v78": st78["test_v78"].astype(np.float64),
        "v44": st44["test_v44"].astype(np.float64),
        "v39": st39["test_v39"].astype(np.float64),
    }
    base_o = 0.70*pool_o["v48_9m"] + 0.12*pool_o["v46_7m"] + 0.18*pool_o["v35"]
    base_t = 0.70*pool_t["v48_9m"] + 0.12*pool_t["v46_7m"] + 0.18*pool_t["v35"]
    rh_base = rh(base_o, y_train)
    print(f"base v48 3-way: {rh_base:.4f}")

    for n, p in pool_o.items():
        print(f"  {n}: {rh(p, y_train):.4f}")
    print(f"  v78 hit & base miss: {(((np.linalg.norm(pool_o['v78']-y_train,axis=-1)<=0.01)) & (np.linalg.norm(base_o-y_train,axis=-1)>0.01)).sum()}")

    best = (rh_base, None, None)
    print("\n=== 3-way exhaustive (all triples) ===")
    names = list(pool_o.keys())
    for trip in itertools.combinations(names, 3):
        po = [pool_o[n] for n in trip]; pt = [pool_t[n] for n in trip]
        b_r, b_w = rh_base, None
        for a in np.linspace(0, 1, 21):
            for b in np.linspace(0, 1-a, 21):
                c = 1 - a - b
                if c < 0: continue
                ens = a*po[0] + b*po[1] + c*po[2]
                r = rh(ens, y_train)
                if r > b_r: b_r, b_w = r, (a, b, c)
        if b_w and b_r > best[0]:
            best = (b_r, f"3way_{'_'.join(trip)}", b_w[0]*pt[0] + b_w[1]*pt[1] + b_w[2]*pt[2])
            print(f"  ★ {trip} ({b_w[0]:.2f}/{b_w[1]:.2f}/{b_w[2]:.2f}): {b_r:.4f}  Δ {b_r - rh_base:+.4f}")

    print("\n=== 4-way exhaustive (all quads) ===")
    for quad in itertools.combinations(names, 4):
        po = [pool_o[n] for n in quad]; pt = [pool_t[n] for n in quad]
        b_r, b_w = rh_base, None
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1-a, 11):
                for c in np.linspace(0, 1-a-b, 11):
                    d = 1-a-b-c
                    if d < 0: continue
                    ens = a*po[0] + b*po[1] + c*po[2] + d*po[3]
                    r = rh(ens, y_train)
                    if r > b_r: b_r, b_w = r, (a, b, c, d)
        if b_w and b_r > best[0]:
            best = (b_r, f"4way_{'_'.join(quad)}",
                    b_w[0]*pt[0] + b_w[1]*pt[1] + b_w[2]*pt[2] + b_w[3]*pt[3])
            print(f"  ★ {quad} ({b_w[0]:.2f}/{b_w[1]:.2f}/{b_w[2]:.2f}/{b_w[3]:.2f}): {b_r:.4f}  Δ {b_r - rh_base:+.4f}")

    # 5-way (top 5 names): v48_9m, v46_7m, v35, v78, v39 (top OOF)
    print("\n=== 5-way (top 5, coarser grid) ===")
    top5_names = ["v48_9m", "v46_7m", "v35", "v78", "v39"]
    po = [pool_o[n] for n in top5_names]
    pt = [pool_t[n] for n in top5_names]
    b_r, b_w = rh_base, None
    for a in np.linspace(0.4, 0.9, 11):
        for b in np.linspace(0, min(0.3, 1-a), 7):
            for c in np.linspace(0, min(0.3, 1-a-b), 7):
                for d in np.linspace(0, min(0.3, 1-a-b-c), 7):
                    e = 1 - a - b - c - d
                    if e < 0: continue
                    ens = a*po[0] + b*po[1] + c*po[2] + d*po[3] + e*po[4]
                    r = rh(ens, y_train)
                    if r > b_r: b_r, b_w = r, (a, b, c, d, e)
    if b_w:
        ws = b_w
        print(f"  best (v48/v46/v35/v78/v39) = ({ws[0]:.2f}/{ws[1]:.2f}/{ws[2]:.2f}/{ws[3]:.2f}/{ws[4]:.2f}): {b_r:.4f}  Δ {b_r - rh_base:+.4f}")
        if b_r > best[0]:
            final_t = sum(w*p for w, p in zip(ws, pt))
            best = (b_r, "5way_top5", final_t)

    # 6-way (all 6 models, very coarse)
    print("\n=== 6-way (all 6, very coarse) ===")
    all6 = list(pool_o.keys())
    po = [pool_o[n] for n in all6]; pt = [pool_t[n] for n in all6]
    b_r, b_w = rh_base, None
    for a in np.linspace(0.3, 0.9, 7):
        for b in np.linspace(0, min(0.3, 1-a), 4):
            for c in np.linspace(0, min(0.3, 1-a-b), 4):
                for d in np.linspace(0, min(0.3, 1-a-b-c), 4):
                    for e in np.linspace(0, min(0.3, 1-a-b-c-d), 4):
                        f = 1-a-b-c-d-e
                        if f < 0: continue
                        ens = a*po[0]+b*po[1]+c*po[2]+d*po[3]+e*po[4]+f*po[5]
                        r = rh(ens, y_train)
                        if r > b_r: b_r, b_w = r, (a,b,c,d,e,f)
    if b_w:
        ws = b_w
        print(f"  best 6-way = {tuple(round(w,2) for w in ws)}: {b_r:.4f}  Δ {b_r - rh_base:+.4f}")
        if b_r > best[0]:
            final_t = sum(w*p for w, p in zip(ws, pt))
            best = (b_r, "6way_all", final_t)

    print(f"\n{'='*60}")
    if best[1] is None:
        print(f"NO LIFT — all grids ≤ base {rh_base:.4f}")
    else:
        print(f"★ BEST: {best[1]}  OOF={best[0]:.4f}  Δ {best[0] - rh_base:+.4f}")
        out = DATA / f"submission_v79_{best[1]}.csv"
        pd.DataFrame({"id": sub["id"], "x": best[2][:,0], "y": best[2][:,1], "z": best[2][:,2]}).to_csv(out, index=False)
        print(f"  [submission] {out.name}")
    print("="*60)

    entry = {"version": "v79_grid_v78", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "rh_base": float(rh_base), "best_oof": float(best[0]),
             "best_scheme": str(best[1]) if best[1] else None,
             "delta": float(best[0] - rh_base)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

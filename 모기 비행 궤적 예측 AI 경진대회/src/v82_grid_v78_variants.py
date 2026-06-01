"""v82_grid_v78_variants.py — v78 + cap variants 추가 exhaustive grid.

v79: 4-way (v48_9m/v46_7m/v35/v78) best 0.6749 (+0.0001)
v78 cap 0.5 OOF 0.6725, cap 1.5 OOF 0.6727 (variants)
→ 5/6-way grid에서 더 큰 lift 가능성
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
    st78_05 = np.load(CACHE / "v78_cap0p5_state.npz")
    st78_15 = np.load(CACHE / "v78_cap1p5_state.npz")
    st44 = np.load(CACHE / "v44_state.npz")
    st39 = np.load(CACHE / "v39_state.npz")
    st52_05 = np.load(CACHE / "v52_cap0p5_state.npz") if (CACHE / "v52_cap0p5_state.npz").exists() else None
    st52_15 = np.load(CACHE / "v52_cap1p5_state.npz") if (CACHE / "v52_cap1p5_state.npz").exists() else None

    pool_o = {
        "v48_9m": v48s["oof_v48"], "v46_7m": v46s["oof_v46"],
        "v35": st35["oof_v35"].astype(np.float64),
        "v78": st78["oof_v78"].astype(np.float64),
        "v78_05": st78_05["oof_v78"].astype(np.float64),
        "v78_15": st78_15["oof_v78"].astype(np.float64),
        "v44": st44["oof_v44"].astype(np.float64),
        "v39": st39["oof_v39"].astype(np.float64),
    }
    pool_t = {
        "v48_9m": v48s["test_v48"], "v46_7m": v46s["test_v46"],
        "v35": st35["test_v35"].astype(np.float64),
        "v78": st78["test_v78"].astype(np.float64),
        "v78_05": st78_05["test_v78"].astype(np.float64),
        "v78_15": st78_15["test_v78"].astype(np.float64),
        "v44": st44["test_v44"].astype(np.float64),
        "v39": st39["test_v39"].astype(np.float64),
    }
    if st52_05 is not None:
        pool_o["v52_05"] = st52_05["oof_v52"].astype(np.float64) if "oof_v52" in st52_05.files else None
        pool_t["v52_05"] = st52_05["test_v52"].astype(np.float64) if "test_v52" in st52_05.files else None
        if pool_o["v52_05"] is None:
            # try different key names
            for k in st52_05.files:
                if "oof" in k:
                    pool_o["v52_05"] = st52_05[k].astype(np.float64)
                    break
            for k in st52_05.files:
                if "test" in k:
                    pool_t["v52_05"] = st52_05[k].astype(np.float64)
                    break

    base_o = 0.70*pool_o["v48_9m"] + 0.12*pool_o["v46_7m"] + 0.18*pool_o["v35"]
    base_t = 0.70*pool_t["v48_9m"] + 0.12*pool_t["v46_7m"] + 0.18*pool_t["v35"]
    rh_base = rh(base_o, y_train)
    print(f"base v48 3-way: {rh_base:.4f}")
    for n, p in pool_o.items():
        if p is not None:
            print(f"  {n}: {rh(p, y_train):.4f}")
    # cleanup nones
    pool_o = {k: v for k, v in pool_o.items() if v is not None}
    pool_t = {k: v for k, v in pool_t.items() if v is not None}
    names = list(pool_o.keys())

    best = (rh_base, None, None, None)
    print("\n=== 4-way exhaustive ===")
    for quad in itertools.combinations(names, 4):
        po = [pool_o[n] for n in quad]; pt = [pool_t[n] for n in quad]
        b_r, b_w = rh_base, None
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1-a, 11):
                for c in np.linspace(0, 1-a-b, 11):
                    d = 1-a-b-c
                    if d < 0: continue
                    ens = a*po[0]+b*po[1]+c*po[2]+d*po[3]
                    r = rh(ens, y_train)
                    if r > b_r: b_r, b_w = r, (a,b,c,d)
        if b_w and b_r > best[0]:
            ws = b_w
            print(f"  ★ {quad} ({ws[0]:.2f}/{ws[1]:.2f}/{ws[2]:.2f}/{ws[3]:.2f}): {b_r:.4f}  Δ {b_r - rh_base:+.4f}")
            ft = sum(w*p for w, p in zip(ws, pt))
            best = (b_r, f"4w_{'_'.join(quad)}", ws, ft)

    print("\n=== 5-way (top quad + 1 extra, finer around best) ===")
    # 단순 5-way (top quad + extra from {v78_05, v78_15, v44, v39})
    base_quad = ["v48_9m", "v46_7m", "v35", "v78"]
    for extra in [n for n in names if n not in base_quad]:
        names_5 = base_quad + [extra]
        po = [pool_o[n] for n in names_5]; pt = [pool_t[n] for n in names_5]
        b_r, b_w = rh_base, None
        # fine grid around (0.70/0.09/0.19/0.02/0) → mostly base 거의 유지
        for a in [0.60, 0.65, 0.70, 0.75]:
            for b in [0.05, 0.09, 0.12, 0.15]:
                for c in [0.10, 0.15, 0.19, 0.23]:
                    for d in [0.00, 0.02, 0.05, 0.08]:
                        e = 1 - a - b - c - d
                        if e < 0 or e > 0.15: continue
                        ens = a*po[0]+b*po[1]+c*po[2]+d*po[3]+e*po[4]
                        r = rh(ens, y_train)
                        if r > b_r: b_r, b_w = r, (a,b,c,d,e)
        if b_w and b_r > best[0]:
            ws = b_w
            print(f"  ★ {names_5} ({'/'.join(f'{w:.2f}' for w in ws)}): {b_r:.4f}  Δ {b_r - rh_base:+.4f}")
            ft = sum(w*p for w, p in zip(ws, pt))
            best = (b_r, f"5w_{'_'.join(names_5)}", ws, ft)
        elif b_w:
            print(f"  {names_5}: best {b_r:.4f} (no global lift)")

    print("\n=== 6-way (full exhaustive coarse) ===")
    sixes = list(itertools.combinations(names, 6))
    print(f"  testing {len(sixes)} combinations...")
    for six in sixes[:30]:  # 처음 30개만
        po = [pool_o[n] for n in six]; pt = [pool_t[n] for n in six]
        b_r, b_w = rh_base, None
        for a in np.linspace(0.5, 0.9, 5):
            for b in np.linspace(0, min(0.2, 1-a), 4):
                for c in np.linspace(0, min(0.25, 1-a-b), 5):
                    for d in np.linspace(0, min(0.15, 1-a-b-c), 4):
                        for e in np.linspace(0, min(0.15, 1-a-b-c-d), 4):
                            f = 1-a-b-c-d-e
                            if f < 0 or f > 0.15: continue
                            ens = a*po[0]+b*po[1]+c*po[2]+d*po[3]+e*po[4]+f*po[5]
                            r = rh(ens, y_train)
                            if r > b_r: b_r, b_w = r, (a,b,c,d,e,f)
        if b_w and b_r > best[0]:
            ws = b_w
            print(f"  ★ {six} ({'/'.join(f'{w:.2f}' for w in ws)}): {b_r:.4f}  Δ {b_r - rh_base:+.4f}")
            ft = sum(w*p for w, p in zip(ws, pt))
            best = (b_r, f"6w", ws, ft)

    print(f"\n{'='*60}")
    if best[1] is None:
        print(f"NO LIFT — all grids ≤ base {rh_base:.4f}")
    else:
        print(f"★ BEST: {best[1]}")
        print(f"   weights: {best[2]}")
        print(f"   OOF: {best[0]:.4f}  Δ {best[0] - rh_base:+.4f}")
        out = DATA / f"submission_v82_{best[1]}.csv"
        pd.DataFrame({"id": sub["id"], "x": best[3][:,0], "y": best[3][:,1], "z": best[3][:,2]}).to_csv(out, index=False)
        print(f"  [submission] {out.name}")
    print("="*60)

    entry = {"version": "v82_grid_v78_variants", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "rh_base": float(rh_base), "best_oof": float(best[0]),
             "best_scheme": str(best[1]) if best[1] else None,
             "delta": float(best[0] - rh_base)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

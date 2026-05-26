"""v76_median_perAxis.py — median ensemble + per-axis blend + top-K subset 빠른 진단.

지금까지 plateau 0.6748 못 넘음. 새 paradigm 학습 전, 간과한 단순 패턴 점검:
  1. Element-wise median (vs mean) — multi-modal robust
  2. Per-axis (x/y/z) 다른 weight blend
  3. Top-K subset only (v35, v44, v39 등 top 3)
  4. Combined: median of top-K + per-axis cal
"""
import sys, glob, os, json, datetime as _dt
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data, yaw_angle, inverse_rotate_xy

PROJECT = SCRIPT_DIR.parent
CACHE = PROJECT / "cache"
DATA = PROJECT / "open"
DT = 0.040


def rh(p, y): return (np.linalg.norm(p - y, axis=-1) <= 0.01).mean()


def main():
    X_train, X_test, y_train, sub = load_data()
    kc = np.load(CACHE / "kalman.npz")
    kt, ke = kc["kalman_train"], kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]

    st30 = np.load(CACHE / "v30_state.npz")
    st35 = np.load(CACHE / "v35_state.npz")
    st41 = np.load(CACHE / "v41_state.npz")
    st44 = np.load(CACHE / "v44_state.npz")
    st39 = np.load(CACHE / "v39_state.npz")
    BO = PROJECT / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
    BEST_TEST = PROJECT / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
    train_files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    bo = np.load(BO, allow_pickle=True); gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(bo["ids"], train_ids):
        m = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([m[i] for i in bo["ids"]]); gate_o = gate_o[perm]
    gate_t = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    v48s = np.load(CACHE / "v48_state.npz"); v46s = np.load(CACHE / "v46_state.npz")

    pool_o = {
        "v30A": kt + st30["oof_A"]*ALPHA, "v30B": kt + st30["oof_B"]*ALPHA,
        "v35":  st35["oof_v35"].astype(np.float64),
        "v41A": kt + st41["oof_A"]*ALPHA, "v41B": kt + st41["oof_B"]*ALPHA,
        "v44":  st44["oof_v44"].astype(np.float64),
        "gate": gate_o,
        "v39":  st39["oof_v39"].astype(np.float64),
        "v48_9m": v48s["oof_v48"],
        "v46_7m": v46s["oof_v46"],
    }
    pool_t = {
        "v30A": ke + st30["test_A"]*ALPHA, "v30B": ke + st30["test_B"]*ALPHA,
        "v35":  st35["test_v35"].astype(np.float64),
        "v41A": ke + st41["test_A"]*ALPHA, "v41B": ke + st41["test_B"]*ALPHA,
        "v44":  st44["test_v44"].astype(np.float64),
        "gate": gate_t,
        "v39":  st39["test_v39"].astype(np.float64),
        "v48_9m": v48s["test_v48"],
        "v46_7m": v46s["test_v46"],
    }
    base_o = 0.70*v48s["oof_v48"] + 0.12*v46s["oof_v46"] + 0.18*st35["oof_v35"]
    base_t = 0.70*v48s["test_v48"] + 0.12*v46s["test_v46"] + 0.18*st35["test_v35"]
    rh_base = rh(base_o, y_train)
    print(f"base v48 3-way: {rh_base:.4f}")

    # Single model OOF
    for n, p in pool_o.items():
        print(f"  {n:8s}: {rh(p, y_train):.4f}")

    best = (rh_base, None, None)

    # ===== 1. Element-wise median over subsets =====
    print("\n=== Median ensemble ===")
    subsets = [
        ["v35", "v44", "v39"],
        ["v35", "v44", "v39", "v48_9m"],
        ["v35", "v44", "v39", "v48_9m", "v46_7m"],
        ["v35", "v44", "v39", "v48_9m", "v46_7m", "gate"],
        ["v35", "v44", "v39", "v48_9m", "v46_7m", "gate", "v30A", "v30B"],
        ["v35", "v44", "v39", "v48_9m", "v46_7m", "gate", "v30A", "v30B", "v41A", "v41B"],
        ["v35", "v44", "v39", "v48_9m"],  # dup for trimmed
    ]
    for names in subsets:
        stack_o = np.stack([pool_o[n] for n in names], axis=0)
        stack_t = np.stack([pool_t[n] for n in names], axis=0)
        med_o = np.median(stack_o, axis=0)
        med_t = np.median(stack_t, axis=0)
        r = rh(med_o, y_train)
        flag = " ★" if r > best[0] else ""
        print(f"  median[{','.join(names)}]: {r:.4f}{flag}")
        if r > best[0]:
            best = (r, f"median_{len(names)}m", med_t)

    # ===== 2. Per-axis blend search (base + each rescue) =====
    print(f"\n=== Per-axis blend (base + each pool member, axis별 다른 weight) ===")
    # 각 축에 대해 best blend weight 따로 찾기
    for n in ["v35", "v44", "v39", "v48_9m"]:
        ro, rt = pool_o[n], pool_t[n]
        # search per-axis w on [0, 1]
        w_per = []
        for ax in range(3):
            best_r_ax = rh(base_o, y_train)  # 전체 R-Hit metric이라 axis-specific 비교 어려움
            # 대신: full 3D R-Hit 측정 하면서 1축씩 변화
            best_w_ax, best_full = 1.0, rh_base
            for w in np.linspace(0, 1, 21):
                final = base_o.copy()
                final[:, ax] = w * base_o[:, ax] + (1 - w) * ro[:, ax]
                r = rh(final, y_train)
                if r > best_full: best_full, best_w_ax = r, w
            w_per.append(best_w_ax)
        # 동시 적용
        final_o = base_o.copy(); final_t = base_t.copy()
        for ax in range(3):
            final_o[:, ax] = w_per[ax] * base_o[:, ax] + (1 - w_per[ax]) * ro[:, ax]
            final_t[:, ax] = w_per[ax] * base_t[:, ax] + (1 - w_per[ax]) * rt[:, ax]
        r = rh(final_o, y_train)
        flag = " ★" if r > best[0] else ""
        print(f"  base + {n} per-axis (x,y,z)={tuple(round(w,2) for w in w_per)}: {r:.4f}{flag}")
        if r > best[0]:
            best = (r, f"perAxis_{n}", final_t)

    # ===== 3. Top-K subset mean (no v48 3-way) =====
    print(f"\n=== Top-K mean ===")
    sorted_pool = sorted(pool_o.items(), key=lambda kv: -rh(kv[1], y_train))
    for K in [2, 3, 4, 5]:
        top_names = [n for n, _ in sorted_pool[:K]]
        stack_o = np.stack([pool_o[n] for n in top_names], axis=0)
        stack_t = np.stack([pool_t[n] for n in top_names], axis=0)
        m_o = stack_o.mean(axis=0); m_t = stack_t.mean(axis=0)
        r = rh(m_o, y_train)
        flag = " ★" if r > best[0] else ""
        print(f"  top{K} mean [{','.join(top_names)}]: {r:.4f}{flag}")
        if r > best[0]: best = (r, f"top{K}_mean", m_t)

    # ===== 4. Trimmed mean (drop max/min per coordinate) =====
    print(f"\n=== Trimmed mean (drop max+min per coord) ===")
    for names in [["v35","v44","v39","v48_9m","v46_7m"],
                  ["v35","v44","v39","v48_9m","v46_7m","gate","v30A","v30B"]]:
        stack_o = np.stack([pool_o[n] for n in names], axis=0)  # (M, N, 3)
        stack_t = np.stack([pool_t[n] for n in names], axis=0)
        if len(names) < 3: continue
        # trimmed mean (drop top 1 and bottom 1 per coord)
        s_o = np.sort(stack_o, axis=0)
        s_t = np.sort(stack_t, axis=0)
        trimmed_o = s_o[1:-1].mean(axis=0)
        trimmed_t = s_t[1:-1].mean(axis=0)
        r = rh(trimmed_o, y_train)
        flag = " ★" if r > best[0] else ""
        print(f"  trimmed[{len(names)}m]: {r:.4f}{flag}")
        if r > best[0]: best = (r, f"trimmed_{len(names)}m", trimmed_t)

    # ===== 5. Median of top-K =====
    print(f"\n=== Median of top-K (robust) ===")
    for K in [3, 4, 5, 6]:
        top_names = [n for n, _ in sorted_pool[:K]]
        stack_o = np.stack([pool_o[n] for n in top_names], axis=0)
        stack_t = np.stack([pool_t[n] for n in top_names], axis=0)
        med_o = np.median(stack_o, axis=0); med_t = np.median(stack_t, axis=0)
        r = rh(med_o, y_train)
        flag = " ★" if r > best[0] else ""
        print(f"  median top{K} [{','.join(top_names)}]: {r:.4f}{flag}")
        if r > best[0]: best = (r, f"median_top{K}", med_t)

    # ===== 6. 3-way exhaustive grid search (top model triples) =====
    print(f"\n=== Exhaustive 3-way blend grid (top 6 models) ===")
    top6 = [n for n, _ in sorted_pool[:6]]
    for triple in itertools.combinations(top6, 3):
        po = [pool_o[n] for n in triple]
        pt = [pool_t[n] for n in triple]
        best_combo_r, best_combo_w = rh_base, None
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1 - a, 11):
                c = 1 - a - b
                if c < 0: continue
                ens = a*po[0] + b*po[1] + c*po[2]
                r = rh(ens, y_train)
                if r > best_combo_r:
                    best_combo_r, best_combo_w = r, (a, b, c)
        if best_combo_w is not None and best_combo_r > rh_base + 1e-5:
            a, b, c = best_combo_w
            final_t = a*pt[0] + b*pt[1] + c*pt[2]
            flag = " ★" if best_combo_r > best[0] else ""
            print(f"  {triple} ({a:.2f}/{b:.2f}/{c:.2f}): {best_combo_r:.4f}  Δ {best_combo_r - rh_base:+.4f}{flag}")
            if best_combo_r > best[0]:
                best = (best_combo_r, f"3way_{'_'.join(triple)}", final_t)

    # ===== 7. 4-way grid =====
    print(f"\n=== 4-way grid (top 5 models, exhaustive quadruple) ===")
    top5 = [n for n, _ in sorted_pool[:5]]
    for quad in itertools.combinations(top5, 4):
        po = [pool_o[n] for n in quad]
        pt = [pool_t[n] for n in quad]
        best_combo_r, best_combo_w = rh_base, None
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1 - a, 11):
                for c in np.linspace(0, 1 - a - b, 11):
                    d = 1 - a - b - c
                    if d < 0: continue
                    ens = a*po[0] + b*po[1] + c*po[2] + d*po[3]
                    r = rh(ens, y_train)
                    if r > best_combo_r:
                        best_combo_r, best_combo_w = r, (a, b, c, d)
        if best_combo_w is not None and best_combo_r > rh_base + 1e-5:
            a, b, c, d = best_combo_w
            final_t = a*pt[0] + b*pt[1] + c*pt[2] + d*pt[3]
            flag = " ★" if best_combo_r > best[0] else ""
            print(f"  {quad} ({a:.2f}/{b:.2f}/{c:.2f}/{d:.2f}): {best_combo_r:.4f}{flag}")
            if best_combo_r > best[0]:
                best = (best_combo_r, f"4way_{'_'.join(quad)}", final_t)

    print(f"\n{'='*60}")
    if best[1] is None:
        print(f"NO LIFT — all simple ensembles ≤ base {rh_base:.4f}")
    else:
        print(f"BEST: {best[1]}  OOF={best[0]:.4f}  Δ {best[0] - rh_base:+.4f}")
        out = DATA / f"submission_v76_{best[1]}.csv"
        pd.DataFrame({"id": sub["id"], "x": best[2][:,0], "y": best[2][:,1], "z": best[2][:,2]}).to_csv(out, index=False)
        print(f"  [submission] {out.name}")
    print("="*60)

    entry = {"version": "v76_median_perAxis", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "rh_base": rh_base, "best_oof": best[0],
             "best_scheme": str(best[1]) if best[1] else None,
             "delta": float(best[0] - rh_base)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

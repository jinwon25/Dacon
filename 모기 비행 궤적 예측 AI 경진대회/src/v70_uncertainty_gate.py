"""v70_uncertainty_gate.py — Uncertainty-gated routing (cheating-free).

진단:
  - 사후 base-miss 알면 routing → OOF 0.6935 (+0.0187)
  - test에선 base hit/miss 모름 → uncertainty proxy로 high-risk sample identification

Proxies (cheating-free, test 적용 가능):
  P1. pool prediction variance (10-model std) — disagreement
  P2. base ↔ v65 distance — base와 rescue 모델 disagreement
  P3. base ↔ pool mean distance
  P4. learned uncertainty (binary classifier on base-miss)

Routing rule:
  uncertainty(x) ≥ T → use rescue (v65 hard / v62 / v68 selector pick)
  else                → use base (v48 3-way)

OOF grid sweep으로 best (proxy, rescue, T) 선택.
"""
import sys, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data, yaw_angle, inverse_rotate_xy

CACHE = SCRIPT_DIR.parent / "data/cache"
DATA = SCRIPT_DIR.parent / "data"
PROJECT = SCRIPT_DIR.parent
BO_PATH = PROJECT / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
BEST_TEST = PROJECT / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rh(p, y): return (np.linalg.norm(p - y, axis=-1) <= 0.01)


def load_pool():
    X_train, X_test, y_train, sub = load_data()
    kc = np.load(CACHE / "kalman.npz")
    kt, ke = kc["kalman_train"], kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]

    st30 = np.load(CACHE / "v30_state.npz")
    st35 = np.load(CACHE / "v35_state.npz")
    st41 = np.load(CACHE / "v41_state.npz")
    st44 = np.load(CACHE / "v44_state.npz")
    st39 = np.load(CACHE / "v39_state.npz")
    train_files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    bo = np.load(BO_PATH, allow_pickle=True); gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(bo["ids"], train_ids):
        m = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([m[i] for i in bo["ids"]]); gate_o = gate_o[perm]
    gate_t = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)

    st65 = np.load(CACHE / "v65_K64_state.npz")
    st62 = np.load(CACHE / "v62_state.npz")
    v_last_tr = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_te = (X_test[:, -1] - X_test[:, -2]) / DT
    th_tr, th_te = yaw_angle(v_last_tr), yaw_angle(v_last_te)
    v62o = st62["kalman_train"] + inverse_rotate_xy((st62["oof_A"] + st62["oof_B"])/2, th_tr)
    v62t = st62["kalman_test"] + inverse_rotate_xy((st62["test_A"] + st62["test_B"])/2, th_te)

    pool_oof = {
        "v30A": kt + st30["oof_A"]*ALPHA, "v30B": kt + st30["oof_B"]*ALPHA,
        "v35": st35["oof_v35"].astype(np.float64),
        "v41A": kt + st41["oof_A"]*ALPHA, "v41B": kt + st41["oof_B"]*ALPHA,
        "v44": st44["oof_v44"].astype(np.float64),
        "gate": gate_o, "v39": st39["oof_v39"].astype(np.float64),
        "v65s": st65["oof_soft"].astype(np.float64),
        "v65h": st65["oof_hard"].astype(np.float64),
        "v62": v62o,
    }
    pool_te = {
        "v30A": ke + st30["test_A"]*ALPHA, "v30B": ke + st30["test_B"]*ALPHA,
        "v35": st35["test_v35"].astype(np.float64),
        "v41A": ke + st41["test_A"]*ALPHA, "v41B": ke + st41["test_B"]*ALPHA,
        "v44": st44["test_v44"].astype(np.float64),
        "gate": gate_t, "v39": st39["test_v39"].astype(np.float64),
        "v65s": st65["test_soft"].astype(np.float64),
        "v65h": st65["test_hard"].astype(np.float64),
        "v62": v62t,
    }
    return X_train, X_test, y_train, sub, pool_oof, pool_te


def sweep_threshold(uncertainty, rescue_o, rescue_t, base_o, base_t, y_train, name):
    """OOF에서 T sweep, best 찾기, test 동일 T 적용"""
    rh_base = rh(base_o, y_train).mean()
    best_T, best_r, best_n = None, rh_base, 0
    grid = np.percentile(uncertainty, [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95])
    print(f"\n  [{name}]  uncertainty percentile T sweep:")
    for q, T in zip([10,20,30,40,50,60,70,75,80,85,90,95], grid):
        use_rescue = uncertainty >= T
        n = use_rescue.sum()
        final = np.where(use_rescue[:, None], rescue_o, base_o)
        r = rh(final, y_train).mean()
        flag = " ★" if r > best_r else ""
        print(f"    p{q:2d} T={T:.4f}: rescue {n:5d} ({n/len(y_train)*100:5.1f}%) → OOF {r:.4f} (Δ {r - rh_base:+.4f}){flag}")
        if r > best_r:
            best_r, best_T, best_n = r, T, n
    if best_T is None:
        return None, None, None, None
    # test
    use_rescue_te = uncertainty_test_proxy(name, base_t, rescue_t) >= best_T
    final_te = np.where(use_rescue_te[:, None], rescue_t, base_t)
    return best_T, best_r, best_n, final_te


# placeholder, will compute per-name
def uncertainty_test_proxy(name, base_t, rescue_t):
    # Same logic as OOF: distance between base and rescue. But for other proxies we need per-name.
    # Caller should not rely on this — we re-compute test uncertainty per proxy explicitly below.
    return np.zeros(len(base_t))


def main():
    X_train, X_test, y_train, sub, pool_oof, pool_te = load_pool()
    N, Nt = len(y_train), len(X_test)

    # base
    v48 = np.load(CACHE / "v48_state.npz"); v46 = np.load(CACHE / "v46_state.npz")
    base_o = 0.70*v48["oof_v48"] + 0.12*v46["oof_v46"] + 0.18*pool_oof["v35"]
    base_t = 0.70*v48["test_v48"] + 0.12*v46["test_v46"] + 0.18*pool_te["v35"]
    rh_base = rh(base_o, y_train).mean()
    print(f"base v48 3-way OOF: {rh_base:.4f}")

    # rescue candidates
    rescues = {
        "v65h":   (pool_oof["v65h"], pool_te["v65h"]),
        "v65s":   (pool_oof["v65s"], pool_te["v65s"]),
        "v62":    (pool_oof["v62"],  pool_te["v62"]),
        "v65h+v65s": ((pool_oof["v65h"] + pool_oof["v65s"]) / 2,
                     (pool_te["v65h"] + pool_te["v65s"]) / 2),
        "v65h+v62": ((pool_oof["v65h"] + pool_oof["v62"]) / 2,
                    (pool_te["v65h"] + pool_te["v62"]) / 2),
        "v65s+v62": ((pool_oof["v65s"] + pool_oof["v62"]) / 2,
                    (pool_te["v65s"] + pool_te["v62"]) / 2),
        "all3":    ((pool_oof["v65h"] + pool_oof["v65s"] + pool_oof["v62"]) / 3,
                   (pool_te["v65h"] + pool_te["v65s"] + pool_te["v62"]) / 3),
    }

    # uncertainty proxies
    pool_names_strong = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate", "v39"]
    pool_stack_o = np.stack([pool_oof[n] for n in pool_names_strong], axis=0)  # (8, N, 3)
    pool_stack_t = np.stack([pool_te[n] for n in pool_names_strong], axis=0)

    # P1: strong pool prediction std (across 8 strong models, per coord summed)
    std_o = pool_stack_o.std(axis=0).sum(axis=-1)  # (N,)
    std_t = pool_stack_t.std(axis=0).sum(axis=-1)
    # P2: base ↔ v65h distance
    d_b_v65h_o = np.linalg.norm(base_o - pool_oof["v65h"], axis=-1)
    d_b_v65h_t = np.linalg.norm(base_t - pool_te["v65h"], axis=-1)
    # P3: base ↔ v62 distance
    d_b_v62_o = np.linalg.norm(base_o - pool_oof["v62"], axis=-1)
    d_b_v62_t = np.linalg.norm(base_t - pool_te["v62"], axis=-1)
    # P4: base ↔ rescue mean distance (per rescue, computed below)

    proxies = {
        "pool_std": (std_o, std_t),
        "d_base_v65h": (d_b_v65h_o, d_b_v65h_t),
        "d_base_v62": (d_b_v62_o, d_b_v62_t),
    }

    best_overall = (rh_base, None, None, None)
    test_outputs = {}
    print("\n" + "=" * 60)
    print("SWEEP: (rescue) × (proxy) × T")
    print("=" * 60)
    for rname, (resc_o, resc_t) in rescues.items():
        # add proxy: distance base ↔ this rescue
        proxies_rescue = dict(proxies)
        d_o = np.linalg.norm(base_o - resc_o, axis=-1)
        d_t = np.linalg.norm(base_t - resc_t, axis=-1)
        proxies_rescue["d_base_rescue"] = (d_o, d_t)

        for pname, (u_o, u_t) in proxies_rescue.items():
            label = f"{rname}/{pname}"
            grid = np.percentile(u_o, [50, 60, 70, 75, 80, 85, 90, 92, 95])
            best_T, best_r, best_n = None, rh_base, 0
            for q, T in zip([50,60,70,75,80,85,90,92,95], grid):
                use_rescue = u_o >= T
                n = use_rescue.sum()
                final = np.where(use_rescue[:, None], resc_o, base_o)
                r = rh(final, y_train).mean()
                if r > best_r:
                    best_r, best_T, best_n = r, T, n
                    best_qi = q
            if best_T is not None and best_r > rh_base:
                print(f"  {label:30s}  p{best_qi:2d}  T={best_T:.4f}  n={best_n:5d}  OOF={best_r:.4f}  Δ {best_r - rh_base:+.4f}")
                # apply test
                use_rescue_te = u_t >= best_T
                final_te = np.where(use_rescue_te[:, None], resc_t, base_t)
                key = f"{rname}_{pname}_p{best_qi}"
                test_outputs[key] = (final_te, best_r, best_T)
                if best_r > best_overall[0]:
                    best_overall = (best_r, label, best_T, key)

    print("\n" + "=" * 60)
    if best_overall[1] is None:
        print(f"NO IMPROVEMENT — all (rescue × proxy × T) combos ≤ base {rh_base:.4f}")
        print(f"진단: unconditional uncertainty proxy로는 base miss 예측 불가.")
        print(f"      → v71 learned base-miss classifier 필요.")
    else:
        print(f"BEST: rescue/proxy = {best_overall[1]}, T={best_overall[2]:.4f}, OOF={best_overall[0]:.4f}")
        print(f"      (base {rh_base:.4f}, Δ {best_overall[0] - rh_base:+.4f})")
    print("=" * 60)

    # Save top 5 by OOF
    top5 = sorted(test_outputs.items(), key=lambda kv: -kv[1][1])[:5]
    print(f"\nTop 5 submissions (OOF desc):")
    for key, (final_te, oof, T) in top5:
        out = DATA / f"submission_v70_{key}.csv"
        pd.DataFrame({"id": sub["id"], "x": final_te[:,0], "y": final_te[:,1], "z": final_te[:,2]}).to_csv(out, index=False)
        print(f"  OOF {oof:.4f} → {out.name}")

    # save best state
    if best_overall[3] is not None:
        bkey = best_overall[3]
        final_te = test_outputs[bkey][0]
        np.savez(CACHE / "v70_best_state.npz",
                 best_rescue_proxy=best_overall[1], best_T=best_overall[2],
                 best_oof=best_overall[0], final_test=final_te)


if __name__ == "__main__":
    main()

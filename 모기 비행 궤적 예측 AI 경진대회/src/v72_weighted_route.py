"""v72_weighted_route.py — Soft routing: per-sample blend weight = miss_prob.

진단:
  - v71 miss AUC 0.7826 (좋음), hard routing 실패 (rescue 약함, false-positive 손실)
  - oracle upper bound: base miss & rescue hit = 607 sample (max +6%)
  - hard routing: T threshold로 cutoff → rescue 잘못 적용 시 손해 큼
  - soft routing: per-sample weight 부드럽게 → false-positive 손실 작게

설계:
  final = (1 - α·p_miss) · base + α·p_miss · rescue
  α grid: [0.1, 0.2, 0.3, ..., 1.0]
  rescue grid: v65h, v65s, v62, mean variants

또한:
  - p_miss 보정 (calibration): linear, sigmoid scale
  - 다양한 rescue 시도
"""
import sys, glob, os, json, datetime as _dt
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data, yaw_angle, inverse_rotate_xy

PROJECT = SCRIPT_DIR.parent
CACHE = PROJECT / "data/cache"
DATA = PROJECT / "data"
BO_PATH = PROJECT / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
BEST_TEST = PROJECT / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rh(p, y): return (np.linalg.norm(p - y, axis=-1) <= 0.01).mean()


def main():
    X_train, X_test, y_train, sub = load_data()
    kc = np.load(CACHE / "kalman.npz")
    kt, ke = kc["kalman_train"], kc["kalman_test"]
    ALPHA_CAL = np.array([1.000, 0.950, 1.000])[None, :]

    st35 = np.load(CACHE / "v35_state.npz")
    st65 = np.load(CACHE / "v65_K64_state.npz")
    st62 = np.load(CACHE / "v62_state.npz")
    v48s = np.load(CACHE / "v48_state.npz"); v46s = np.load(CACHE / "v46_state.npz")

    v_last_tr = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_te = (X_test[:, -1] - X_test[:, -2]) / DT
    th_tr, th_te = yaw_angle(v_last_tr), yaw_angle(v_last_te)
    v62o = st62["kalman_train"] + inverse_rotate_xy((st62["oof_A"] + st62["oof_B"])/2, th_tr)
    v62t = st62["kalman_test"] + inverse_rotate_xy((st62["test_A"] + st62["test_B"])/2, th_te)

    base_o = 0.70*v48s["oof_v48"] + 0.12*v46s["oof_v46"] + 0.18*st35["oof_v35"]
    base_t = 0.70*v48s["test_v48"] + 0.12*v46s["test_v46"] + 0.18*st35["test_v35"]
    v65h_o, v65h_t = st65["oof_hard"].astype(np.float64), st65["test_hard"].astype(np.float64)
    v65s_o, v65s_t = st65["oof_soft"].astype(np.float64), st65["test_soft"].astype(np.float64)
    v62o_, v62t_ = v62o, v62t

    rh_base = rh(base_o, y_train)
    print(f"base v48 3-way OOF: {rh_base:.4f}")

    v71 = np.load(CACHE / "v71_miss_state.npz")
    p_miss_o = v71["prob_oof"]; p_miss_t = v71["prob_te"]
    print(f"miss prob OOF mean: {p_miss_o.mean():.3f}, miss AUC {float(v71['auc']):.4f}")

    rescues = {
        "v65h": (v65h_o, v65h_t),
        "v65s": (v65s_o, v65s_t),
        "v62":  (v62o_, v62t_),
        "v65hs_avg": ((v65h_o + v65s_o)/2, (v65h_t + v65s_t)/2),
        "v65h_v62": ((v65h_o + v62o_)/2, (v65h_t + v62t_)/2),
        "v65s_v62": ((v65s_o + v62o_)/2, (v65s_t + v62t_)/2),
        "all3_avg": ((v65h_o + v65s_o + v62o_)/3, (v65h_t + v65s_t + v62t_)/3),
    }

    print("\n=== Soft routing sweep: final = (1 - α·p_miss)·base + α·p_miss·rescue ===")
    best = (rh_base, None, None, None)
    grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
    for rname, (ro, rt) in rescues.items():
        for alpha in grid:
            w = (alpha * p_miss_o)[:, None]  # (N, 1)
            final_o = (1 - w) * base_o + w * ro
            r = rh(final_o, y_train)
            if r > best[0]:
                w_t = (alpha * p_miss_t)[:, None]
                final_t = (1 - w_t) * base_t + w_t * rt
                best = (r, rname, alpha, final_t)
                print(f"  ★ {rname:12s}  α={alpha:.2f}  OOF={r:.4f}  Δ {r - rh_base:+.4f}")

    print("\n=== Soft routing with calibrated miss_prob (sigmoid steeper) ===")
    # try sharper: p_sharp = sigmoid((p - 0.5) * gain)
    import scipy.special as sp
    for gain in [2, 3, 5, 8]:
        p_sharp_o = sp.expit((p_miss_o - 0.5) * gain)
        p_sharp_t = sp.expit((p_miss_t - 0.5) * gain)
        for rname, (ro, rt) in rescues.items():
            for alpha in [0.3, 0.5, 0.75, 1.0]:
                w = (alpha * p_sharp_o)[:, None]
                final_o = (1 - w) * base_o + w * ro
                r = rh(final_o, y_train)
                if r > best[0]:
                    w_t = (alpha * p_sharp_t)[:, None]
                    final_t = (1 - w_t) * base_t + w_t * rt
                    best = (r, f"{rname}_g{gain}", alpha, final_t)
                    print(f"  ★ {rname:12s}  α={alpha:.2f}  gain={gain}  OOF={r:.4f}  Δ {r - rh_base:+.4f}")

    print(f"\n{'='*60}")
    if best[1] is None:
        print(f"NO LIFT — soft routing도 base 못 넘음. rescue가 절대적으로 약함.")
    else:
        print(f"BEST: rescue={best[1]}  α={best[2]:.2f}  OOF={best[0]:.4f}  Δ {best[0] - rh_base:+.4f}")
        out = DATA / f"submission_v72_{best[1]}_a{best[2]:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": best[3][:,0], "y": best[3][:,1], "z": best[3][:,2]}).to_csv(out, index=False)
        print(f"  [submission] {out.name}")
    print("="*60)

    entry = {
        "version": "v72_weighted_route", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "soft routing: per-sample blend weight = α·p_miss",
        "best_oof": float(best[0]), "rh_base": float(rh_base),
        "delta": float(best[0] - rh_base),
        "best_rescue": str(best[1]) if best[1] else None,
        "best_alpha": float(best[2]) if best[2] is not None else None,
    }
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

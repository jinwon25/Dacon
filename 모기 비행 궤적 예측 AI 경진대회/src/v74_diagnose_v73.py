"""v74_diagnose_v73.py — v73 paradigm rescue 종합 진단 + routing 재시도.

v73 OOF soft 0.6244 (v65 0.6137 → +0.0107), hard 0.6058.
다음 진단:
  1. v73 hit / base miss 분석 (oracle rescue potential 갱신)
  2. v71 miss prob + v73 rescue routing
  3. v73 + v65 + v62 mean rescue (더 강한 rescue?)
  4. blend 다양한 조합
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


def rh_mask(p, y): return (np.linalg.norm(p - y, axis=-1) <= 0.01)
def rh(p, y): return rh_mask(p, y).mean()


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
    train_files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    bo = np.load(BO_PATH, allow_pickle=True); gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(bo["ids"], train_ids):
        m = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([m[i] for i in bo["ids"]]); gate_o = gate_o[perm]
    gate_t = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    st65 = np.load(CACHE / "v65_K64_state.npz")
    st73 = np.load(CACHE / "v73_K64_state.npz")
    st62 = np.load(CACHE / "v62_state.npz")
    v_last_tr = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_te = (X_test[:, -1] - X_test[:, -2]) / DT
    th_tr, th_te = yaw_angle(v_last_tr), yaw_angle(v_last_te)
    v62o = st62["kalman_train"] + inverse_rotate_xy((st62["oof_A"] + st62["oof_B"])/2, th_tr)
    v62t = st62["kalman_test"] + inverse_rotate_xy((st62["test_A"] + st62["test_B"])/2, th_te)

    v48 = np.load(CACHE / "v48_state.npz"); v46 = np.load(CACHE / "v46_state.npz")
    base_o = 0.70*v48["oof_v48"] + 0.12*v46["oof_v46"] + 0.18*st35["oof_v35"]
    base_t = 0.70*v48["test_v48"] + 0.12*v46["test_v46"] + 0.18*st35["test_v35"]
    rh_base = rh(base_o, y_train)
    print(f"base OOF: {rh_base:.4f}")

    print("\n=== Rescue candidates OOF ===")
    rescues = {
        "v65h": (st65["oof_hard"], st65["test_hard"]),
        "v65s": (st65["oof_soft"], st65["test_soft"]),
        "v73h": (st73["oof_hard"], st73["test_hard"]),
        "v73s": (st73["oof_soft"], st73["test_soft"]),
        "v62":  (v62o, v62t),
    }
    for n, (o, _) in rescues.items():
        print(f"  {n}: OOF {rh(o, y_train):.4f}")

    # combo rescues
    print("\n=== Combo rescues (mean) ===")
    combo = {
        "v73s+v65s": ((st73["oof_soft"] + st65["oof_soft"])/2, (st73["test_soft"] + st65["test_soft"])/2),
        "v73h+v65h": ((st73["oof_hard"] + st65["oof_hard"])/2, (st73["test_hard"] + st65["test_hard"])/2),
        "v73s+v62":  ((st73["oof_soft"] + v62o)/2, (st73["test_soft"] + v62t)/2),
        "v73h+v62":  ((st73["oof_hard"] + v62o)/2, (st73["test_hard"] + v62t)/2),
        "v73s+v65h": ((st73["oof_soft"] + st65["oof_hard"])/2, (st73["test_soft"] + st65["test_hard"])/2),
        "v73h+v65s+v62": ((st73["oof_hard"] + st65["oof_soft"] + v62o)/3,
                          (st73["test_hard"] + st65["test_soft"] + v62t)/3),
        "v73s+v65s+v62": ((st73["oof_soft"] + st65["oof_soft"] + v62o)/3,
                          (st73["test_soft"] + st65["test_soft"] + v62t)/3),
        "v73h+v65h+v62": ((st73["oof_hard"] + st65["oof_hard"] + v62o)/3,
                          (st73["test_hard"] + st65["test_hard"] + v62t)/3),
    }
    for n, (o, _) in combo.items():
        print(f"  {n}: OOF {rh(o, y_train):.4f}")
    rescues.update(combo)

    # base hit / rescue oracle gain
    base_hit = rh_mask(base_o, y_train)
    print(f"\n=== Sample-wise oracle gain (base miss & rescue hit) ===")
    for n, (o, _) in rescues.items():
        r_hit = rh_mask(o, y_train)
        rescued = (~base_hit) & r_hit
        n_rescued = rescued.sum()
        print(f"  {n:18s}: rescued {n_rescued:4d}  (any-hit {(base_hit | r_hit).mean():.4f}, oracle Δ {(base_hit | r_hit).mean() - rh_base:+.4f})")

    # v71 miss prob
    v71 = np.load(CACHE / "v71_miss_state.npz")
    p_miss_o = v71["prob_oof"]; p_miss_t = v71["prob_te"]
    print(f"\nmiss prob OOF mean: {p_miss_o.mean():.3f}, AUC {float(v71['auc']):.4f}")

    # ★ Soft routing sweep ALL rescues × α grid
    print(f"\n=== Soft routing: final = (1 - α·p_miss)·base + α·p_miss·rescue ===")
    best = (rh_base, None, None, None)
    for rname, (ro, rt) in rescues.items():
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]:
            w = (alpha * p_miss_o)[:, None]
            final_o = (1 - w) * base_o + w * ro
            r = rh(final_o, y_train)
            if r > best[0]:
                w_t = (alpha * p_miss_t)[:, None]
                final_t = (1 - w_t) * base_t + w_t * rt
                best = (r, rname, alpha, final_t)
                print(f"  ★ {rname:18s}  α={alpha:.2f}  OOF={r:.4f}  Δ {r - rh_base:+.4f}")

    # ★ Sharpened miss prob
    import scipy.special as sp
    for gain in [2, 3, 5, 8, 12]:
        p_s_o = sp.expit((p_miss_o - 0.5) * gain)
        p_s_t = sp.expit((p_miss_t - 0.5) * gain)
        for rname, (ro, rt) in rescues.items():
            for alpha in [0.2, 0.3, 0.5, 0.75, 1.0]:
                w = (alpha * p_s_o)[:, None]
                final_o = (1 - w) * base_o + w * ro
                r = rh(final_o, y_train)
                if r > best[0]:
                    w_t = (alpha * p_s_t)[:, None]
                    final_t = (1 - w_t) * base_t + w_t * rt
                    best = (r, f"{rname}_g{gain}", alpha, final_t)
                    print(f"  ★ {rname:18s}  α={alpha:.2f}  g={gain}  OOF={r:.4f}  Δ {r - rh_base:+.4f}")

    # ★ Hard threshold routing
    print(f"\n=== Hard threshold routing ===")
    for rname, (ro, rt) in rescues.items():
        grid = np.percentile(p_miss_o, [50, 60, 70, 75, 80, 85, 88, 90, 92, 95])
        for q, T in zip([50,60,70,75,80,85,88,90,92,95], grid):
            use_r = p_miss_o >= T
            final_o = np.where(use_r[:, None], ro, base_o)
            r = rh(final_o, y_train)
            if r > best[0]:
                use_r_t = p_miss_t >= T
                final_t = np.where(use_r_t[:, None], rt, base_t)
                best = (r, f"hard_{rname}_p{q}", T, final_t)
                print(f"  ★ {rname:18s}  T={T:.3f} (p{q})  OOF={r:.4f}  Δ {r - rh_base:+.4f}")

    # ★ Linear blend (without miss prob) — paranoid check
    print(f"\n=== Plain linear blend (no miss prob) ===")
    for rname, (ro, rt) in rescues.items():
        for w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
            final_o = (1 - w) * base_o + w * ro
            r = rh(final_o, y_train)
            if r > best[0]:
                final_t = (1 - w) * base_t + w * rt
                best = (r, f"linear_{rname}", w, final_t)
                print(f"  ★ {rname:18s}  w={w:.2f}  OOF={r:.4f}  Δ {r - rh_base:+.4f}")

    print(f"\n{'='*60}")
    if best[1] is None:
        print(f"NO LIFT — all rescue × routing combos ≤ base {rh_base:.4f}")
    else:
        print(f"BEST: {best[1]}  param={best[2]}  OOF={best[0]:.4f}  Δ {best[0] - rh_base:+.4f}")
        out = DATA / f"submission_v74_{best[1]}_{best[2]:.3f}.csv"
        pd.DataFrame({"id": sub["id"], "x": best[3][:,0], "y": best[3][:,1], "z": best[3][:,2]}).to_csv(out, index=False)
        print(f"  [submission] {out.name}")
    print("="*60)

    entry = {
        "version": "v74_diagnose_v73", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "rh_base": float(rh_base),
        "best_oof": float(best[0]),
        "delta": float(best[0] - rh_base),
        "best_scheme": str(best[1]) if best[1] else None,
        "best_param": float(best[2]) if best[2] is not None else None,
    }
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

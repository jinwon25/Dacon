"""v66_diagnose_v65.py — v65 anchor head paradigm 다양성 진단.

목적:
  v65 OOF 0.6137 (soft) / 0.6037 (hard). 단독 약함.
  그러나 paradigm 다름 (classification). pool에 추가 효과 검증.

진단 항목:
  1. v48 9-model + v65 oracle gain (any-hit upper bound 증가량)
  2. v65 + v35 / v48 3-way linear blend grid
  3. v65 hit과 v48 3-way hit의 sample-wise correlation
  4. v65가 v48 3-way가 놓치는 sample을 잡는 비율
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data

CACHE = SCRIPT_DIR.parent / "cache"
DATA = SCRIPT_DIR.parent / "open"


def rh(p, y):
    return (np.linalg.norm(p - y, axis=-1) <= 0.01)


def main():
    X_train, X_test, y_train, sub = load_data()

    # v48 3-way (base)
    v48 = np.load(CACHE / "v48_state.npz"); v48o, v48t = v48["oof_v48"], v48["test_v48"]
    v46 = np.load(CACHE / "v46_state.npz"); v46o, v46t = v46["oof_v46"], v46["test_v46"]
    v35 = np.load(CACHE / "v35_state.npz"); v35o, v35t = v35["oof_v35"], v35["test_v35"]
    base_o = 0.70 * v48o + 0.12 * v46o + 0.18 * v35o
    base_t = 0.70 * v48t + 0.12 * v46t + 0.18 * v35t

    # v65 K=64
    v65 = np.load(CACHE / "v65_K64_state.npz")
    v65s_o, v65s_t = v65["oof_soft"], v65["test_soft"]
    v65h_o, v65h_t = v65["oof_hard"], v65["test_hard"]

    print("=== Single-model OOF ===")
    for name, p in [("v48 3-way", base_o), ("v48 9m", v48o), ("v35", v35o),
                    ("v65 soft", v65s_o), ("v65 hard", v65h_o)]:
        r = rh(p, y_train).mean()
        print(f"  {name:14s}: {r:.4f}")

    # Sample-wise diagnostic
    base_hit = rh(base_o, y_train)
    v65s_hit = rh(v65s_o, y_train)
    v65h_hit = rh(v65h_o, y_train)

    print(f"\n=== sample-wise hit analysis ===")
    print(f"  base hit: {base_hit.mean():.4f}  miss: {(~base_hit).sum()}")
    print(f"  v65 soft hit: {v65s_hit.mean():.4f}")
    print(f"  base miss & v65 soft hit (v65 새로 잡음): {((~base_hit) & v65s_hit).sum()} = {((~base_hit) & v65s_hit).mean():.4f}")
    print(f"  base hit & v65 soft miss (v65 놓침): {(base_hit & (~v65s_hit)).sum()}")
    print(f"  any-hit (base OR v65 soft): {(base_hit | v65s_hit).mean():.4f}  (oracle gain {(base_hit | v65s_hit).mean() - base_hit.mean():+.4f})")
    print(f"  any-hit (base OR v65 hard): {(base_hit | v65h_hit).mean():.4f}  (oracle gain {(base_hit | v65h_hit).mean() - base_hit.mean():+.4f})")

    # 9-model + v65 oracle
    st30 = np.load(CACHE / "v30_state.npz")
    nc = np.load(CACHE / "xtrain_xtest.npz"); kc = np.load(CACHE / "kalman.npz")
    kt = kc["kalman_train"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    v30A_o = kt + st30["oof_A"] * ALPHA
    v30B_o = kt + st30["oof_B"] * ALPHA
    st41 = np.load(CACHE / "v41_state.npz")
    v41A_o = kt + st41["oof_A"] * ALPHA
    v41B_o = kt + st41["oof_B"] * ALPHA
    st44 = np.load(CACHE / "v44_state.npz"); v44o = st44["oof_v44"].astype(np.float64)
    st39 = np.load(CACHE / "v39_state.npz"); v39o = st39["oof_v39"].astype(np.float64)
    st32 = np.load(CACHE / "v32_mdn_state.npz"); v32o = st32["oof_weighted"].astype(np.float64)
    bo = np.load(SCRIPT_DIR.parent / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz", allow_pickle=True)
    import glob, os
    train_files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    best_ids = bo["ids"]
    gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        m = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([m[i] for i in best_ids])
        gate_o = gate_o[perm]

    pool9 = [v30A_o, v30B_o, v35o, v41A_o, v41B_o, v44o, gate_o, v39o, v32o]
    hits9 = np.stack([rh(p, y_train) for p in pool9])
    oracle9 = hits9.any(axis=0).mean()
    hits10s = np.vstack([hits9, rh(v65s_o, y_train)[None]])
    hits10h = np.vstack([hits9, rh(v65h_o, y_train)[None]])
    oracle10s = hits10s.any(axis=0).mean()
    oracle10h = hits10h.any(axis=0).mean()
    print(f"\n=== 9-model + v65 oracle ===")
    print(f"  9-model oracle: {oracle9:.4f}")
    print(f"  9m + v65 soft:  {oracle10s:.4f}  (Δ {oracle10s - oracle9:+.4f})")
    print(f"  9m + v65 hard:  {oracle10h:.4f}  (Δ {oracle10h - oracle9:+.4f})")

    # Linear blend grid: v48 3-way + v65 soft
    print(f"\n=== Linear blend: w*v48_3way + (1-w)*v65_soft ===")
    best_w, best_r = 1.0, rh(base_o, y_train).mean()
    for w in np.linspace(0, 1, 21):
        ens = w * base_o + (1 - w) * v65s_o
        r = rh(ens, y_train).mean()
        if r > best_r: best_r, best_w = r, w
    print(f"  best: w={best_w:.2f} → OOF {best_r:.4f}  (vs base {rh(base_o, y_train).mean():.4f}, Δ {best_r - rh(base_o, y_train).mean():+.4f})")

    # v35 + v65 soft
    print(f"\n=== Linear blend: w*v35 + (1-w)*v65_soft ===")
    best_w2, best_r2 = 1.0, rh(v35o, y_train).mean()
    for w in np.linspace(0, 1, 21):
        ens = w * v35o + (1 - w) * v65s_o
        r = rh(ens, y_train).mean()
        if r > best_r2: best_r2, best_w2 = r, w
    print(f"  best: w={best_w2:.2f} → OOF {best_r2:.4f}  (vs v35 {rh(v35o, y_train).mean():.4f}, Δ {best_r2 - rh(v35o, y_train).mean():+.4f})")

    # Submission for best blend if positive
    if best_r > rh(base_o, y_train).mean():
        blend_t = best_w * base_t + (1 - best_w) * v65s_t
        out = DATA / f"submission_v66_v48x{best_w:.2f}_v65soft.csv"
        pd.DataFrame({"id": sub["id"], "x": blend_t[:,0], "y": blend_t[:,1], "z": blend_t[:,2]}).to_csv(out, index=False)
        print(f"\n  [submission] {out.name}")


if __name__ == "__main__":
    main()

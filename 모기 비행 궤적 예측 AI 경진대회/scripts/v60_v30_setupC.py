"""v60_v30_setupC.py — v30 framework + new setup C (paradigm pool 확장).

v30 setup A: lr=5e-4, hidden=64, do=0.3 → OOF 0.6557 (cal)
v30 setup B: lr=1e-3, hidden=64, do=0.1 → OOF 0.6587 (cal)
v60 setup C: lr=3e-4, hidden=80, do=0.4 → 새 paradigm (lr 더 작고 dropout 더 크게)

목적: stacker pool에 paradigm-orthogonal 새 모델 추가. 14 stacker 카드 plateau 우회.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"

from v23_train import (
    DT, build_seq, build_tier3, build_scalar_feats,
    yaw_angle, rotate_xy, inverse_rotate_xy,
)
from v30_advanced_v23 import (
    compute_adv_weights, run_5fold_multiseed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v60: v30 framework + Setup C (lr=3e-4, hidden=80, do=0.4)  {args.n_seeds}-seed × 5-fold")
    print("=" * 60)

    # --- Data ---
    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test, kalman_train_alt = kc["kalman_train"], kc["kalman_test"], kc["kalman_train_alt"]
    rh_kal = float((np.linalg.norm(kalman_train - y_train, axis=-1) <= 0.01).mean())
    print(f"[kalman] R-Hit train: {rh_kal:.4f}")

    nc_noise = np.load(CACHE_DIR / "noise_fast.npz")
    noise_p, noise_s, noise_l = nc_noise["noise_p"], nc_noise["noise_s"], nc_noise["noise_l"]
    noise_p_te, noise_s_te = nc_noise["noise_p_test"], nc_noise["noise_s_test"]

    scal_tr = build_scalar_feats(X_train, noise_p, noise_s, noise_l)
    scal_te = build_scalar_feats(X_test, noise_p_te, noise_s_te)
    LOG_COLS = ["mean_speed","max_speed","speed_std","mean_acc","max_acc","max_jerk",
                "net_disp","|v_last|","|a_last|","|a_recent|","jerk_last","jerk_recent",
                "noise_poly2","noise_savgol","noise_loo"]
    for c in LOG_COLS:
        scal_tr[c] = np.log1p(scal_tr[c]); scal_te[c] = np.log1p(scal_te[c])
    X_scal_base_tr = scal_tr.values.astype(np.float32)
    X_scal_base_te = scal_te.values.astype(np.float32)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    X_scal_tr = np.concatenate([X_scal_base_tr, tier3_tr], axis=-1)
    X_scal_te = np.concatenate([X_scal_base_te, tier3_te], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)

    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8 = rotate_xy(y_train - kalman_train, theta_train)
    target_F  = rotate_xy(y_train - X_train[:, -1], theta_train)
    target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)

    # Reuse v30 adv weight (cached in v30_state.npz to save 1 min)
    try:
        st30 = np.load(CACHE_DIR / "v30_state.npz")
        sample_w = st30["sample_weight"].astype(np.float32)
        adv_auc = float(st30["adv_auc"])
        print(f"[adv] reused from v30 cache, AUC={adv_auc:.4f}")
    except Exception:
        print("[adv] computing fresh adv weights...")
        sample_w, adv_auc = compute_adv_weights(X_train, X_test)

    # --- Setup C ---
    CFG_C = dict(hidden=80, layers=1, fc=128, lr=3e-4, p=0.4, wd=1e-4)
    state_file = CACHE_DIR / "v60_state.npz"

    if state_file.exists() and not args.force:
        st = np.load(state_file)
        oof_C, test_C, rh_C = st["oof_C"], st["test_C"], float(st["rh_C"])
        print(f"[state] cached: C={rh_C:.4f}")
    else:
        print("\n" + "=" * 60)
        print(f"Setup C (lr=3e-4, hidden=80, dropout=0.4) × {args.n_seeds} seed × 5-fold")
        print("=" * 60)
        oof_C, test_C, fold_rh_C, rh_C = run_5fold_multiseed(
            target_T8, target_F, target_W, seq_tr, X_scal_tr, seq_te, X_scal_te, sample_w,
            kalman_train, theta_train, theta_test, y_train,
            config=CFG_C, n_seeds=args.n_seeds, n_folds=args.n_folds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
        )
        np.savez(state_file,
                  oof_C=oof_C, test_C=test_C, rh_C=rh_C,
                  fold_rh_C=np.array(fold_rh_C),
                  sample_weight=sample_w, adv_auc=adv_auc,
                  config_C=str(CFG_C))
        print(f"[state] saved: {state_file}")

    # Calibration (same ALPHA as v30)
    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_C_cal = oof_C * ALPHA[None, :]
    test_C_cal = test_C * ALPHA[None, :]
    pred_oof = kalman_train + oof_C_cal
    rh_C_cal = float((np.linalg.norm(pred_oof - y_train, axis=-1) <= 0.01).mean())

    # Reference
    rh_v30_A = float((np.linalg.norm(kalman_train + st30["oof_A"] * ALPHA - y_train, axis=-1) <= 0.01).mean()) if 'st30' in locals() else None
    rh_v30_B = float((np.linalg.norm(kalman_train + st30["oof_B"] * ALPHA - y_train, axis=-1) <= 0.01).mean()) if 'st30' in locals() else None

    print(f"\n=== v60 setup C 결과 ===")
    print(f"  v30 setup A: {rh_v30_A:.4f}")
    print(f"  v30 setup B: {rh_v30_B:.4f}")
    print(f"  v60 setup C (calibrated): {rh_C_cal:.4f}")

    # Submission
    test_pos = kalman_test + test_C_cal
    out_csv = DATA_DIR / "submission_v60_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    # Log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v60_v30_setupC",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "v30 framework + Setup C (lr=3e-4, hidden=80, dropout=0.4) paradigm 추가",
        "config_C": str(CFG_C),
        "n_folds": args.n_folds, "n_seeds": args.n_seeds,
        "v30_setup_A_oof_cal": float(rh_v30_A) if rh_v30_A else None,
        "v30_setup_B_oof_cal": float(rh_v30_B) if rh_v30_B else None,
        "v60_setup_C_oof_cal": float(rh_C_cal),
        "alpha": ALPHA.tolist(),
        "submission_path": str(out_csv),
    }
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    if not isinstance(logs, list): logs = [logs]
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"  [run_log] appended v60_v30_setupC")


if __name__ == "__main__":
    main()

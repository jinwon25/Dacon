"""v62_ca_paradigm.py — CA Kalman base + v23 framework (paradigm card).

목적:
  v46~v61 stacker plateau (OOF 0.6748, LB 0.6876) 돌파.
  모든 OOF 모델 풀이 CV Kalman 기반이라 sample-wise 다양성 소진.
  CA (constant acceleration) Kalman base = motion model paradigm 자체가 다름.
  → ensemble pool diversity 증가 기대.

차이점 (vs v23):
  - base Kalman: k_cv_so0p3mm → k_ca_so0p3mm
  - alt Kalman: k_cv_so1mm → k_ca_so1mm (residual W head용)
  - state file: cache/v62_state.npz (v23 cache 안 덮어씀)
  - 그 외 GRU+F+W, yaw rotation, scalar features 모두 v23 동일

사용:
  python scripts/v62_ca_paradigm.py --mode fast   # ~20분 (5-fold 1-seed setup A only)
  python scripts/v62_ca_paradigm.py --mode full   # ~60분 (5-fold 3-seed A+B)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# v23 framework 그대로 import
from v23_train import (
    MODE_CONFIGS, CACHE_DIR, DATA_DIR,
    load_data, get_scalar_feats, build_tier3, build_seq,
    yaw_angle, rotate_xy, inverse_rotate_xy, run_kfold,
)


def get_kalman_ca():
    """CA Kalman base (v23의 CV Kalman 자리에 들어감)."""
    km = np.load(CACHE_DIR / "kalmans_multi.npz")
    k_tr = km["k_ca_so0p3mm_train"].astype(np.float64)
    k_te = km["k_ca_so0p3mm_test"].astype(np.float64)
    k_tr_alt = km["k_ca_so1mm_train"].astype(np.float64)
    return k_tr, k_te, k_tr_alt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fast", choices=list(MODE_CONFIGS.keys()))
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    M = MODE_CONFIGS[mode]

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v62 CA-Kalman paradigm  MODE={mode}  config={M}")
    print(f"device={device}, threads={torch.get_num_threads()}, torch={torch.__version__}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()

    # --- 핵심 차이: CA Kalman base ---
    kalman_train, kalman_test, kalman_train_alt = get_kalman_ca()
    rh_kal = float((np.linalg.norm(kalman_train - y_train, axis=-1) <= 0.01).mean())
    rh_kal_alt = float((np.linalg.norm(kalman_train_alt - y_train, axis=-1) <= 0.01).mean())
    print(f"[CA kalman baseline] R-Hit train: {rh_kal:.4f}  (CV는 0.5964)")
    print(f"[CA alt    baseline] R-Hit train: {rh_kal_alt:.4f}")

    # scalar features + tier3 + seq (v23와 동일)
    X_scal_base_tr, X_scal_base_te = get_scalar_feats(X_train, X_test, M, mode)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    X_scal_tr = np.concatenate([X_scal_base_tr, tier3_tr], axis=-1)
    X_scal_te = np.concatenate([X_scal_base_te, tier3_te], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)
    print(f"[feat] X_scal {X_scal_tr.shape}, seq {seq_tr.shape}")

    # yaw + targets (CA base에 맞춰 residual 재계산)
    DT = 0.040
    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8 = rotate_xy(y_train - kalman_train,     theta_train)
    target_F  = rotate_xy(y_train - X_train[:, -1],   theta_train)
    target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)

    # state file (v23과 별도)
    state_file = CACHE_DIR / "v62_state.npz"
    if state_file.exists() and not args.force_retrain:
        st = np.load(state_file)
        oof_A, test_A = st["oof_A"], st["test_A"]
        oof_rhit_A = float(st["oof_rhit_A"]); fold_mask_A = st["fold_mask_A"]
        fold_rh_A = st["fold_rh_A"].tolist()
        has_B = bool(st.get("has_B", np.array(False)))
        if has_B:
            oof_B, test_B = st["oof_B"], st["test_B"]
            oof_rhit_B = float(st["oof_rhit_B"]); fold_mask_B = st["fold_mask_B"]
            fold_rh_B = st["fold_rh_B"].tolist()
        print(f"[state] cache 로드: A={oof_rhit_A:.4f}" + (f", B={oof_rhit_B:.4f}" if has_B else ""))
    else:
        CONFIG_A = dict(hidden=64, layers=1, fc=128, lr=5e-4, p=0.3, wd=1e-4)
        CONFIG_B = dict(hidden=64, layers=1, fc=128, lr=1e-3, p=0.1, wd=1e-4)

        print("=" * 60); print(f"Setup A 학습 (lr=5e-4, do=0.3)"); print("=" * 60)
        oof_A, test_A, fold_rh_A, oof_rhit_A, fold_mask_A = run_kfold(
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            config=CONFIG_A, mode_cfg=M, device=device,
        )

        if M["run_setup_B"]:
            print("\n" + "=" * 60); print(f"Setup B 학습 (lr=1e-3, do=0.1)"); print("=" * 60)
            oof_B, test_B, fold_rh_B, oof_rhit_B, fold_mask_B = run_kfold(
                target_T8, target_F, target_W,
                seq_tr, X_scal_tr, seq_te, X_scal_te,
                kalman_train, theta_train, theta_test, y_train,
                config=CONFIG_B, mode_cfg=M, device=device,
            )
            has_B = True
        else:
            oof_B = test_B = oof_rhit_B = None; fold_rh_B = []; fold_mask_B = None
            has_B = False

        kwargs = dict(oof_A=oof_A, test_A=test_A, oof_rhit_A=oof_rhit_A,
                      fold_rh_A=np.array(fold_rh_A), fold_mask_A=fold_mask_A,
                      has_B=np.array(has_B),
                      kalman_train=kalman_train, kalman_test=kalman_test)
        if has_B:
            kwargs.update(oof_B=oof_B, test_B=test_B, oof_rhit_B=oof_rhit_B,
                          fold_rh_B=np.array(fold_rh_B), fold_mask_B=fold_mask_B)
        np.savez(state_file, **kwargs)
        print(f"[state] 저장: {state_file}")

    if has_B:
        oof_avg = (oof_A + oof_B) / 2
        test_avg = (test_A + test_B) / 2
        eval_mask = fold_mask_A & fold_mask_B
    else:
        oof_avg = oof_A.copy(); test_avg = test_A.copy()
        eval_mask = fold_mask_A

    pred = kalman_train[eval_mask] + oof_avg[eval_mask]
    rh_avg = float((np.linalg.norm(pred - y_train[eval_mask], axis=-1) <= 0.01).mean())
    print(f"\n[v62 avg] OOF (covered {eval_mask.sum()}): {rh_avg:.4f}")

    # per-axis calibration (v23과 동일 출발점)
    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_cal  = oof_avg  * ALPHA[None, :]
    test_cal = test_avg * ALPHA[None, :]
    pred = kalman_train[eval_mask] + oof_cal[eval_mask]
    rh_cal = float((np.linalg.norm(pred - y_train[eval_mask], axis=-1) <= 0.01).mean())
    print(f"[v62 cal] OOF: {rh_cal:.4f}  (Δ {rh_cal - rh_avg:+.4f})")

    # 제출 CSV
    test_pos = kalman_test + test_cal
    out_csv = DATA_DIR / f"submission_v62_cpu_{mode}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"[submission] {out_csv}")

    # run_log
    log_path = Path(__file__).resolve().parents[1] / "run_log.json"
    entry = {
        "version": f"v62_ca_paradigm_{mode}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode, "mode_config": M,
        "approach": "v23 framework, base=CA Kalman (so0p3mm) + alt=CA so1mm",
        "ca_kalman_baseline_rhit": rh_kal,
        "ca_alt_baseline_rhit": rh_kal_alt,
        "setup_A_oof_rhit": oof_rhit_A,
        "setup_B_oof_rhit": oof_rhit_B if has_B else None,
        "avg_oof_rhit": rh_avg,
        "calibrated_oof_rhit": rh_cal,
        "alpha": ALPHA.tolist(),
        "covered_rows": int(eval_mask.sum()),
        "submission_path": str(out_csv),
    }
    logs = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            if not isinstance(logs, list): logs = [logs]
        except Exception: logs = []
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"v62 CA paradigm ({mode}) 완료")
    print("=" * 60)
    print(f"  OOF calibrated: {rh_cal:.4f}")
    print(f"  CA Kalman base R-Hit:  {rh_kal:.4f}  (CV는 0.5964)")
    print(f"  covered rows:   {int(eval_mask.sum())}/{len(y_train)}")
    print(f"  제출 CSV:       {out_csv}")


if __name__ == "__main__":
    main()

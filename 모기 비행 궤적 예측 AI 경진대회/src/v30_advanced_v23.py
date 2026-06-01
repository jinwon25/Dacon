"""v30_advanced_v23.py — v23 Kalman+GRU+F+W를 Kaggle 상위권 패턴으로 강화.

문헌 기반 강화점:
  1. **5-fold proper OOF** (vs fast 2-fold). Generalization gap ↓.
  2. **Multi-seed averaging (fold split 고정)**: KFold(random_state=0), model seed=0,1만 변경.
     v27 실패 (random_state=seed) 교훈 반영.
  3. **Adversarial sample reweighting** (Pan 2020, Cortes 2008):
     - LGB classifier로 train sample별 P(test|x) 추정
     - 학습 loss weight = clip(P(test|x), 0.3, 3.0) 정규화
     - Covariate shift 보정 (Adv AUC 0.5854 입증)
  4. **2-setup (lr=5e-4 do=0.3 / lr=1e-3 do=0.1)** — 기존 v23 패턴 유지
  5. **Cosine annealing + early stopping by R-Hit**

총 학습: 5 fold × 2 seed × 2 setup = 20 trainings × ~3분 = ~60분 CPU.

출력:
  - cache/v30_state.npz (oof_A_s0/A_s1/B_s0/B_s1, test_A/B, fold_mask 5-fold)
  - open/submission_v30_cpu.csv (single best after calibration)
  - run_log.json append
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# v23 모듈 재사용
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    DT, T_PRED, T_OBS, build_seq, build_tier3, build_scalar_feats,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
    GRUModelMultiAux, loss_combo, loss_euclid,
    kalman_predict, noise_poly2, noise_savgol, noise_loo_subset,
)


PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# Adversarial validation
# ============================================================
def compute_adv_weights(X_train, X_test, clip_low=0.3, clip_high=3.0):
    """Train sample별 P(test|x) 기반 weight. AUC도 반환."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    def feat(X):
        last = X[:, -1, :]
        v = (X[:, -1, :] - X[:, -2, :]) / DT
        a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
        v_recent = np.diff(X[:, -4:], axis=1) / DT
        v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
        nd = np.linalg.norm(X[:, -1] - X[:, 0], axis=-1, keepdims=True)
        return np.column_stack([last, v, a, v_mean, v_std, speed, a_norm, nd])

    ft_tr = feat(X_train); ft_te = feat(X_test)
    n_tr, n_te = len(ft_tr), len(ft_te)
    X_all = np.vstack([ft_tr, ft_te])
    y_all = np.concatenate([np.zeros(n_tr), np.ones(n_te)])

    # Shuffle for fold
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X_all))
    inv = np.argsort(perm)

    oof_proba = np.zeros(len(X_all))
    aucs = []
    kf = KFold(5, shuffle=True, random_state=0)
    for fi, (tr, va) in enumerate(kf.split(perm)):
        gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=300,
                                   min_child_samples=20, verbose=-1, random_state=0)
        gbm.fit(X_all[perm[tr]], y_all[perm[tr]])
        proba = gbm.predict_proba(X_all[perm[va]])[:, 1]
        oof_proba[perm[va]] = proba
        aucs.append(roc_auc_score(y_all[perm[va]], proba))

    print(f"[adv] AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # train의 proba
    train_proba = oof_proba[:n_tr]
    # weight = clip(proba / (1 - proba), clip_low, clip_high)
    weight = np.clip(train_proba / np.maximum(1 - train_proba, 1e-6), clip_low, clip_high)
    weight /= weight.mean()  # normalize
    print(f"[adv] train weight: mean={weight.mean():.3f}, std={weight.std():.3f}, "
          f"min={weight.min():.3f}, max={weight.max():.3f}")
    return weight.astype(np.float32), float(np.mean(aucs))


# ============================================================
# Combo loss with sample weighting
# ============================================================
def loss_combo_weighted(p, t, sw):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    if sw is None:
        return d.mean() + 0.3 * sh.mean()
    return ((d * sw).mean() + 0.3 * (sh * sw).mean()) / sw.mean()


def loss_aux_weighted(p, t, sw):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    return (d * sw).mean() / sw.mean() if sw is not None else d.mean()


# ============================================================
# K-fold trainer (proper 5-fold + multi-seed)
# ============================================================
def run_5fold_multiseed(target_main, target_F, target_W,
                         seq_arr, scal_arr, seq_te, scal_te,
                         sample_weight,
                         kalman_train, theta_train, theta_test, y_train,
                         config, n_seeds=2, n_folds=5, max_epochs=100, patience=15,
                         batch=256, lambda_F=0.3, lambda_W=0.3, device="cpu"):
    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold_seed = []
    fold_rh = []

    # KFold random_state 고정 (multi-seed 시 fold split 동일)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(len(target_main))))
    t0 = time.time()

    for fi, (tr, va) in enumerate(fold_iter):
        sc_seq = StandardScaler().fit(seq_arr[tr].reshape(-1, seq_arr.shape[2]))
        sc_scal = StandardScaler().fit(scal_arr[tr])
        seq_tr_n = normalize_seq(seq_arr[tr], sc_seq)
        seq_va_n = normalize_seq(seq_arr[va], sc_seq)
        seq_te_n = normalize_seq(seq_te, sc_seq)
        scal_tr_n = sc_scal.transform(scal_arr[tr]).astype(np.float32)
        scal_va_n = sc_scal.transform(scal_arr[va]).astype(np.float32)
        scal_te_n = sc_scal.transform(scal_te).astype(np.float32)

        def T(a): return torch.from_numpy(a).to(device)
        seq_tr_t, scal_tr_t = T(seq_tr_n), T(scal_tr_n)
        tgt_tr_t = T(target_main[tr].astype(np.float32))
        F_tr_t = T(target_F[tr].astype(np.float32))
        W_tr_t = T(target_W[tr].astype(np.float32))
        sw_tr_t = T(sample_weight[tr])
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)

        seed_val_rot, seed_test_rot = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = GRUModelMultiAux(
                n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                hidden=config["hidden"], layers=config["layers"],
                fc=config["fc"], p=config["p"],
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

            best_rh, best_state, no_improve = -1.0, None, 0
            n_tr = seq_tr_t.shape[0]
            for ep in range(max_epochs):
                model.train()
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr, batch):
                    idx = perm[i:i+batch]
                    opt.zero_grad()
                    out_main, outs_aux = model(seq_tr_t[idx], scal_tr_t[idx])
                    sw_b = sw_tr_t[idx]
                    loss = loss_combo_weighted(out_main, tgt_tr_t[idx], sw_b)
                    loss = loss + lambda_F * loss_aux_weighted(outs_aux[0], F_tr_t[idx], sw_b)
                    loss = loss + lambda_W * loss_aux_weighted(outs_aux[1], W_tr_t[idx], sw_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    out_va_rot, _ = model(seq_va_t, scal_va_t)
                    out_va_rot = out_va_rot.cpu().numpy()
                out_va = inverse_rotate_xy(out_va_rot, theta_train[va])
                pred = kalman_train[va] + out_va
                rh = float((np.linalg.norm(pred - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience: break

            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pv_rot, _ = model(seq_va_t, scal_va_t)
                pt_rot, _ = model(seq_te_t, scal_te_t)
            seed_val_rot.append(pv_rot.cpu().numpy())
            seed_test_rot.append(pt_rot.cpu().numpy())
            del model; gc.collect()

        # Multi-seed average (fold-fixed → safe)
        val_mean_rot = np.mean(seed_val_rot, axis=0)
        test_mean_rot = np.mean(seed_test_rot, axis=0)
        oof_rot[va] = val_mean_rot
        fold_mask[va] = True
        test_per_fold_seed.append(test_mean_rot)

        val_unrot = inverse_rotate_xy(val_mean_rot, theta_train[va])
        pred_pos = kalman_train[va] + val_unrot
        rh_fold = float((np.linalg.norm(pred_pos - y_train[va], axis=-1) <= 0.01).mean())
        fold_rh.append(rh_fold)
        print(f"  fold{fi+1}/{n_folds}: R-Hit={rh_fold:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)

    oof_unrot = np.zeros_like(target_main)
    oof_unrot[fold_mask] = inverse_rotate_xy(oof_rot[fold_mask], theta_train[fold_mask])
    pred_oof = kalman_train + oof_unrot
    oof_rhit = float((np.linalg.norm(pred_oof - y_train, axis=-1) <= 0.01).mean())
    test_unrot = np.mean([inverse_rotate_xy(rot, theta_test) for rot in test_per_fold_seed], axis=0)
    print(f"  OOF R-Hit (5-fold full): {oof_rhit:.4f}  소요 {(time.time()-t0)/60:.1f}m")
    return oof_unrot, test_unrot, fold_rh, oof_rhit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--adv-clip-low", type=float, default=0.3)
    parser.add_argument("--adv-clip-high", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v30 advanced v23: 5-fold × {args.n_seeds}-seed × 2-setup + adversarial reweight")
    print("=" * 60)

    # --- Data ---
    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    # --- Kalman ---
    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test, kalman_train_alt = kc["kalman_train"], kc["kalman_test"], kc["kalman_train_alt"]
    rh_kal = float((np.linalg.norm(kalman_train - y_train, axis=-1) <= 0.01).mean())
    print(f"[kalman] R-Hit train: {rh_kal:.4f}")

    # --- Noise + features (fast mode 캐시 재사용) ---
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

    # --- Yaw targets ---
    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8 = rotate_xy(y_train - kalman_train, theta_train)
    target_F  = rotate_xy(y_train - X_train[:, -1], theta_train)
    target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)

    # --- Adversarial reweighting ---
    sample_w, adv_auc = compute_adv_weights(X_train, X_test,
                                              clip_low=args.adv_clip_low, clip_high=args.adv_clip_high)

    # --- 2-setup × multi-seed × 5-fold ---
    CFG_A = dict(hidden=64, layers=1, fc=128, lr=5e-4, p=0.3, wd=1e-4)
    CFG_B = dict(hidden=64, layers=1, fc=128, lr=1e-3, p=0.1, wd=1e-4)

    state_file = CACHE_DIR / "v30_state.npz"
    if state_file.exists() and not args.force:
        st = np.load(state_file)
        oof_A, test_A, rh_A = st["oof_A"], st["test_A"], float(st["rh_A"])
        oof_B, test_B, rh_B = st["oof_B"], st["test_B"], float(st["rh_B"])
        print(f"[state] cached: A={rh_A:.4f}, B={rh_B:.4f}")
    else:
        print("\n" + "=" * 60); print("Setup A (lr=5e-4, do=0.3) ×", args.n_seeds, "seed × 5-fold")
        print("=" * 60)
        oof_A, test_A, fold_rh_A, rh_A = run_5fold_multiseed(
            target_T8, target_F, target_W, seq_tr, X_scal_tr, seq_te, X_scal_te, sample_w,
            kalman_train, theta_train, theta_test, y_train,
            config=CFG_A, n_seeds=args.n_seeds, n_folds=args.n_folds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
        )

        print("\n" + "=" * 60); print("Setup B (lr=1e-3, do=0.1) ×", args.n_seeds, "seed × 5-fold")
        print("=" * 60)
        oof_B, test_B, fold_rh_B, rh_B = run_5fold_multiseed(
            target_T8, target_F, target_W, seq_tr, X_scal_tr, seq_te, X_scal_te, sample_w,
            kalman_train, theta_train, theta_test, y_train,
            config=CFG_B, n_seeds=args.n_seeds, n_folds=args.n_folds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
        )

        np.savez(state_file,
                  oof_A=oof_A, test_A=test_A, rh_A=rh_A, fold_rh_A=np.array(fold_rh_A),
                  oof_B=oof_B, test_B=test_B, rh_B=rh_B, fold_rh_B=np.array(fold_rh_B),
                  sample_weight=sample_w, adv_auc=adv_auc)
        print(f"[state] saved: {state_file}")

    # --- sub_09 (A+B 평균) + calibration ---
    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_sub09 = (oof_A + oof_B) / 2
    test_sub09 = (test_A + test_B) / 2
    oof_cal  = oof_sub09  * ALPHA[None, :]
    test_cal = test_sub09 * ALPHA[None, :]

    pred_oof = kalman_train + oof_cal
    rh_sub09 = float((np.linalg.norm(kalman_train + oof_sub09 - y_train, axis=-1) <= 0.01).mean())
    rh_cal = float((np.linalg.norm(pred_oof - y_train, axis=-1) <= 0.01).mean())

    print(f"\n=== v30 결과 ===")
    print(f"  Setup A OOF (5-fold, {args.n_seeds}-seed avg): {rh_A:.4f}")
    print(f"  Setup B OOF: {rh_B:.4f}")
    print(f"  sub_09 OOF (A+B avg): {rh_sub09:.4f}")
    print(f"  calibrated OOF: {rh_cal:.4f}")
    print(f"  adv AUC: {adv_auc:.4f}")
    print(f"  ★ v23 fast 2-setup OOF 0.6516 → v30 OOF {rh_cal:.4f}, Δ {rh_cal - 0.6516:+.4f}")

    # --- Submission ---
    test_pos = kalman_test + test_cal
    out_csv = DATA_DIR / "submission_v30_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    # --- run_log ---
    log_path = PROJECT_DIR / "run_log.json"
    import datetime as _dt
    entry = {
        "version": "v30_cpu_advanced",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"v23 + 5-fold + {args.n_seeds}-seed (fold-fixed) + adversarial reweight + 2-setup",
        "n_folds": args.n_folds, "n_seeds": args.n_seeds,
        "adv_auc": float(adv_auc),
        "setup_A_oof": float(rh_A), "setup_B_oof": float(rh_B),
        "sub09_oof": float(rh_sub09),
        "calibrated_oof": float(rh_cal),
        "alpha": ALPHA.tolist(),
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


if __name__ == "__main__":
    main()

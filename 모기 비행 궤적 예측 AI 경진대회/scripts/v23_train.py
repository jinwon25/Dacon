"""v23_train.py — Kalman residual + GRU+F+W (Trojan_Horse LB 0.6780) 로컬 CPU 재현.

사용법:
  python scripts/v23_train.py --mode micro   # 15~30분, sanity
  python scripts/v23_train.py --mode fast    # 1~2h, 알고리즘 검증
  python scripts/v23_train.py --mode full    # 15~30h, LB 검증

핵심 파이프라인:
  1. Kalman CV (σ_obs=0.3mm, σ_proc=1.0) base prediction
  2. NN은 잔차만 학습 (yaw-rotated frame, last velocity → +x)
  3. GRU(h=64, l=1) + main(T+8 잔차) + F(T+7 직접변위) + W(다른 σ Kalman 잔차)
  4. combo loss = euclid + 0.3 × softhit
  5. 2 setups (lr=5e-4 do=0.3 / lr=1e-3 do=0.1)
  6. Per-axis α = (1.0, 0.95, 1.0) hard calibration

출력:
  cache/v23_state_{mode}.npz, cache/kalman.npz, cache/noise_{mode}.npz
  open/submission_v23_cpu_{mode}.csv
  run_log.json append
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# Mode config
# ============================================================
MODE_CONFIGS = {
    "micro": dict(n_folds=1, n_seeds=1, max_epochs=30, patience=10,
                  batch=256, loo_sample=500, run_setup_B=False),
    "fast":  dict(n_folds=2, n_seeds=1, max_epochs=80, patience=15,
                  batch=256, loo_sample=2000, run_setup_B=True),
    "full":  dict(n_folds=5, n_seeds=3, max_epochs=200, patience=30,
                  batch=256, loo_sample=None, run_setup_B=True),
}

DT = 0.040
T_PRED = 0.080
T_OBS = np.arange(-400, 1, 40) / 1000.0


# ============================================================
# Data loading
# ============================================================
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    cache = CACHE_DIR / "xtrain_xtest.npz"
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    test_files  = sorted(glob.glob(str(DATA_DIR / "test" / "*.csv")))
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub    = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x", "y", "z"]].values.astype(np.float64)

    if cache.exists():
        nc = np.load(cache)
        X_train, X_test = nc["X_train"], nc["X_test"]
        print(f"[data] cache 로드: X_train {X_train.shape}")
    else:
        def _read(f): return pd.read_csv(f)[["x", "y", "z"]].values
        def _load(files, desc, workers=8):
            arrays = [None] * len(files)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_read, f): i for i, f in enumerate(files)}
                for fu in tqdm(futs, desc=desc, total=len(futs)):
                    arrays[futs[fu]] = fu.result()
            return np.stack(arrays, axis=0).astype(np.float64)
        X_train = _load(train_files, "train")
        X_test  = _load(test_files, "test")
        np.savez(cache, X_train=X_train, X_test=X_test)
        print(f"[data] cache 저장: {cache}")

    assert X_train.shape[0] == y_train.shape[0] == 10000
    return X_train, X_test, y_train, sub


# ============================================================
# Kalman
# ============================================================
def kalman_predict(X, sigma_obs=0.30e-3, sigma_proc=1.0, P0=1.0):
    N, T, _ = X.shape
    F = np.array([[1, DT], [0, 1]])
    F_pred = np.array([[1, T_PRED], [0, 1]])
    Q = sigma_proc ** 2 * np.array([[DT**4/4, DT**3/2], [DT**3/2, DT**2]])
    R = sigma_obs ** 2
    pred = np.zeros((N, 3))
    for j in range(3):
        z_all = X[:, :, j]
        state = np.zeros((N, 2)); state[:, 0] = z_all[:, 0]
        P = np.eye(2) * P0
        for t in range(1, T):
            state = state @ F.T
            P = F @ P @ F.T + Q
            innov = z_all[:, t] - state[:, 0]
            S = P[0, 0] + R; K = P[:, 0] / S
            state = state + np.outer(innov, K)
            P = P - np.outer(K, P[0, :])
        pred[:, j] = (state @ F_pred.T)[:, 0]
    return pred


def get_kalman(X_train, X_test):
    cache = CACHE_DIR / "kalman.npz"
    if cache.exists():
        kc = np.load(cache)
        print("[kalman] cache 로드")
        return kc["kalman_train"], kc["kalman_test"], kc["kalman_train_alt"]
    print("[kalman] 계산 중 (10~30초)…")
    k_tr = kalman_predict(X_train, sigma_obs=0.30e-3, sigma_proc=1.0)
    k_te = kalman_predict(X_test,  sigma_obs=0.30e-3, sigma_proc=1.0)
    k_tr_alt = kalman_predict(X_train, sigma_obs=1.0e-3, sigma_proc=1.0)
    np.savez(cache, kalman_train=k_tr, kalman_test=k_te, kalman_train_alt=k_tr_alt)
    print(f"[kalman] cache 저장: {cache}")
    return k_tr, k_te, k_tr_alt


# ============================================================
# Noise + scalar features
# ============================================================
def noise_poly2(X):
    V = np.vander(T_OBS, 3, increasing=False)
    out = np.zeros(X.shape[0])
    for j in range(3):
        coef = np.linalg.lstsq(V, X[:, :, j].T, rcond=None)[0]
        out += (X[:, :, j] - (V @ coef).T).std(axis=1)
    return out / 3


def noise_savgol(X, w=5, p=2):
    Xs = savgol_filter(X, window_length=w, polyorder=p, axis=1)
    return (X - Xs).std(axis=1).mean(axis=-1)


def noise_loo_subset(X, sample_idx):
    sav = noise_savgol(X)
    out = sav.copy()
    idx_all = np.arange(len(T_OBS))
    for i in tqdm(sample_idx, desc="LOO spline subset"):
        s = 0.0
        for k in range(1, len(T_OBS) - 1):
            mask = idx_all != k
            for j in range(3):
                cs = CubicSpline(T_OBS[mask], X[i, mask, j])
                s += (X[i, k, j] - cs(T_OBS[k])) ** 2
        out[i] = np.sqrt(s / ((len(T_OBS) - 2) * 3))
    return out


def cos_safe(a, b):
    na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
    return np.clip((a * b).sum(-1) / np.maximum(na * nb, 1e-12), -1, 1)


def build_scalar_feats(X, noise_p, noise_s, noise_l=None):
    delta_ = np.diff(X, axis=1)
    v_ = delta_ / DT
    a_ = np.diff(v_, axis=1) / DT
    jerk_ = np.diff(a_, axis=1) / DT
    sp_ = np.linalg.norm(v_, axis=-1)
    ac_ = np.linalg.norm(a_, axis=-1)
    jk_ = np.linalg.norm(jerk_, axis=-1)
    v_l = v_[:, -1, :]; a_l = a_[:, -1, :]
    a_r = a_[:, -3:, :].mean(axis=1)
    nd_vec = X[:, -1] - X[:, 0]; nd = np.linalg.norm(nd_vec, axis=-1)
    pl = np.linalg.norm(delta_, axis=-1).sum(axis=1)
    straight = np.where(pl > 1e-12, nd / np.maximum(pl, 1e-12), 0.0)
    turn = cos_safe(v_l, v_[:, :-1, :].mean(axis=1))
    if noise_l is None: noise_l = noise_s
    return pd.DataFrame({
        "mean_speed": sp_.mean(1), "max_speed": sp_.max(1),
        "speed_std": sp_.std(1), "mean_acc": ac_.mean(1),
        "max_acc": ac_.max(1), "max_jerk": jk_.max(1),
        "straightness": straight, "net_disp": nd,
        "turn_cos": turn, "|v_last|": np.linalg.norm(v_l, axis=-1),
        "|a_last|": np.linalg.norm(a_l, axis=-1),
        "|a_recent|": np.linalg.norm(a_r, axis=-1),
        "jerk_last": jk_[:, -1], "jerk_recent": jk_[:, -3:].mean(1),
        "noise_poly2": noise_p, "noise_savgol": noise_s, "noise_loo": noise_l,
        "hard_turn": (turn < 0.5).astype(np.float32),
        "high_speed": (np.linalg.norm(v_l, axis=-1) > 1.0).astype(np.float32),
        "high_acc": (ac_.max(axis=1) > 15).astype(np.float32),
        "log_max_acc": np.log1p(ac_.max(1)),
    })


def get_scalar_feats(X_train, X_test, mode_cfg, mode_name):
    cache = CACHE_DIR / f"noise_{mode_name}.npz"
    if cache.exists():
        nc = np.load(cache)
        np_tr, ns_tr, nl_tr = nc["noise_p"], nc["noise_s"], nc["noise_l"]
        np_te, ns_te = nc["noise_p_test"], nc["noise_s_test"]
        print("[noise] cache 로드")
    else:
        print("[noise] 계산 중…")
        np_tr = noise_poly2(X_train); np_te = noise_poly2(X_test)
        ns_tr = noise_savgol(X_train); ns_te = noise_savgol(X_test)
        if mode_cfg["loo_sample"] is None:
            loo_idx = np.arange(X_train.shape[0])
        else:
            rng = np.random.RandomState(0)
            loo_idx = rng.choice(X_train.shape[0], size=mode_cfg["loo_sample"], replace=False)
        nl_tr = noise_loo_subset(X_train, loo_idx)
        np.savez(cache, noise_p=np_tr, noise_s=ns_tr, noise_l=nl_tr,
                  noise_p_test=np_te, noise_s_test=ns_te)
        print(f"[noise] cache 저장: {cache}")

    scal_tr = build_scalar_feats(X_train, np_tr, ns_tr, nl_tr)
    scal_te = build_scalar_feats(X_test,  np_te, ns_te)
    LOG_COLS = ["mean_speed","max_speed","speed_std","mean_acc","max_acc","max_jerk",
                "net_disp","|v_last|","|a_last|","|a_recent|","jerk_last","jerk_recent",
                "noise_poly2","noise_savgol","noise_loo"]
    for c in LOG_COLS:
        scal_tr[c] = np.log1p(scal_tr[c])
        scal_te[c] = np.log1p(scal_te[c])
    return scal_tr.values.astype(np.float32), scal_te.values.astype(np.float32)


# ============================================================
# Yaw rotation
# ============================================================
def yaw_angle(v):
    return np.arctan2(v[:, 1], v[:, 0])


def rotate_xy(vec, theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.stack([vec[:,0]*c + vec[:,1]*s,
                      -vec[:,0]*s + vec[:,1]*c,
                      vec[:,2]], axis=-1)


def inverse_rotate_xy(vec, theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.stack([vec[:,0]*c - vec[:,1]*s,
                      vec[:,0]*s + vec[:,1]*c,
                      vec[:,2]], axis=-1)


# ============================================================
# Tier 3 + sequence
# ============================================================
def build_tier3(X):
    disp = np.diff(X, axis=1); v = disp / DT
    speed = np.linalg.norm(v, axis=-1)
    roll = np.stack([speed[:, i:i+3].mean(axis=1) for i in range(8)], axis=1) * DT
    cum = np.concatenate([np.zeros((X.shape[0], 1)),
                           np.cumsum(np.linalg.norm(disp, axis=-1), axis=1)], axis=1)
    return np.concatenate([roll, cum], axis=1).astype(np.float32)


def build_seq(X):
    N = X.shape[0]
    rel = X - X[:, -1:, :]
    disp = np.diff(X, axis=1); v = disp / DT
    v_pad = np.concatenate([np.zeros((N,1,3)), v], axis=1)
    a = np.diff(v, axis=1) / DT
    a_pad = np.concatenate([np.zeros((N,2,3)), a], axis=1)
    return np.concatenate([rel, v_pad, a_pad], axis=-1).astype(np.float32)


def normalize_seq(arr, sc):
    N, T, C = arr.shape
    return sc.transform(arr.reshape(-1, C)).astype(np.float32).reshape(N, T, C)


# ============================================================
# Loss + model
# ============================================================
def loss_euclid(pred, true):
    return torch.sqrt(((pred - true) ** 2).sum(dim=-1) + 1e-12).mean()


def loss_softhit(pred, true, beta=0.002):
    d = torch.sqrt(((pred - true) ** 2).sum(dim=-1) + 1e-12)
    return torch.sigmoid((d - 0.01) / beta).mean()


def loss_combo(pred, true):
    return loss_euclid(pred, true) + 0.3 * loss_softhit(pred, true)


class GRUModelMultiAux(nn.Module):
    def __init__(self, n_channels=9, scal_dim=40, hidden=64, layers=1,
                 fc=128, p=0.2, aux_dims=(3, 3), main_scale_cm=2.0):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=layers, batch_first=True,
                           dropout=p if layers > 1 else 0)
        self.fc1 = nn.Linear(hidden + scal_dim, fc)
        self.fc2 = nn.Linear(fc, fc // 2)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.head_main = nn.Linear(fc // 2, 3)
        self.aux_heads = nn.ModuleList([nn.Linear(fc // 2, d) for d in aux_dims])
        self.main_scale = main_scale_cm / 100.0

    def forward(self, seq, scal):
        out, _ = self.gru(seq)
        z = torch.cat([out[:, -1, :], scal], dim=1)
        z = self.act(self.fc1(z)); z = self.drop(z)
        z = self.act(self.fc2(z))
        out_main = torch.tanh(self.head_main(z)) * self.main_scale
        return out_main, [h(z) for h in self.aux_heads]


# ============================================================
# K-fold runner
# ============================================================
def run_kfold(target_main, target_F, target_W,
              seq_arr, scal_arr, seq_te, scal_te,
              kalman_train, theta_train, theta_test, y_train,
              config, mode_cfg, lambda_F=0.3, lambda_W=0.3, device="cpu"):
    n_folds, n_seeds = mode_cfg["n_folds"], mode_cfg["n_seeds"]
    max_epochs, patience, batch = mode_cfg["max_epochs"], mode_cfg["patience"], mode_cfg["batch"]

    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold = []
    fold_rh = []
    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(len(target_main))))
    if n_folds == 1:
        fold_iter = fold_iter[:1]
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
        F_tr_t   = T(target_F[tr].astype(np.float32))
        W_tr_t   = T(target_W[tr].astype(np.float32))
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)

        seed_val, seed_test = [], []
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
                    loss = loss_combo(out_main, tgt_tr_t[idx])
                    loss = loss + lambda_F * loss_euclid(outs_aux[0], F_tr_t[idx])
                    loss = loss + lambda_W * loss_euclid(outs_aux[1], W_tr_t[idx])
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
                if ep == 0 or (ep + 1) % 5 == 0:
                    print(f"  fold{fi+1} seed{s} ep{ep+1:3d}: rhit={rh:.4f} (best {best_rh:.4f})  "
                          f"[{(time.time()-t0)/60:.1f}m]", flush=True)

            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pv_rot, _ = model(seq_va_t, scal_va_t)
                pt_rot, _ = model(seq_te_t, scal_te_t)
            seed_val.append(pv_rot.cpu().numpy())
            seed_test.append(pt_rot.cpu().numpy())
            del model; gc.collect()

        val_mean_rot = np.mean(seed_val, axis=0)
        test_mean_rot = np.mean(seed_test, axis=0)
        oof_rot[va] = val_mean_rot
        fold_mask[va] = True
        test_per_fold.append(test_mean_rot)

        val_unrot = inverse_rotate_xy(val_mean_rot, theta_train[va])
        pred_pos = kalman_train[va] + val_unrot
        rh_fold = float((np.linalg.norm(pred_pos - y_train[va], axis=-1) <= 0.01).mean())
        fold_rh.append(rh_fold)
        print(f"  ★ fold {fi+1}/{len(fold_iter)}: R-Hit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    if fold_mask.sum() == 0:
        oof_rhit = float("nan"); oof_unrot_all = np.zeros_like(target_main)
    else:
        oof_unrot_all = np.zeros_like(target_main)
        oof_unrot_all[fold_mask] = inverse_rotate_xy(oof_rot[fold_mask], theta_train[fold_mask])
        pred = kalman_train[fold_mask] + oof_unrot_all[fold_mask]
        oof_rhit = float((np.linalg.norm(pred - y_train[fold_mask], axis=-1) <= 0.01).mean())

    test_unrot = np.mean([inverse_rotate_xy(rot, theta_test) for rot in test_per_fold], axis=0)
    print(f"  OOF R-Hit (covered {fold_mask.sum()}): {oof_rhit:.4f}  소요 {(time.time()-t0)/60:.1f}m")
    return oof_unrot_all, test_unrot, fold_rh, oof_rhit, fold_mask


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="micro", choices=list(MODE_CONFIGS.keys()))
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
    print(f"v23 CPU MODE={mode}  config={M}")
    print(f"device={device}, threads={torch.get_num_threads()}, torch={torch.__version__}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()
    kalman_train, kalman_test, kalman_train_alt = get_kalman(X_train, X_test)
    rh_kal = float((np.linalg.norm(kalman_train - y_train, axis=-1) <= 0.01).mean())
    print(f"[kalman baseline] R-Hit train: {rh_kal:.4f}  (원본 0.5964 기대)")

    X_scal_base_tr, X_scal_base_te = get_scalar_feats(X_train, X_test, M, mode)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    X_scal_tr = np.concatenate([X_scal_base_tr, tier3_tr], axis=-1)
    X_scal_te = np.concatenate([X_scal_base_te, tier3_te], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)
    print(f"[feat] X_scal {X_scal_tr.shape}, seq {seq_tr.shape}")

    # Yaw + targets
    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8    = rotate_xy(y_train - kalman_train, theta_train)
    target_F     = rotate_xy(y_train - X_train[:, -1], theta_train)
    target_W     = rotate_xy(y_train - kalman_train_alt, theta_train)

    # state file
    state_file = CACHE_DIR / f"v23_state_{mode}.npz"
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
                       has_B=np.array(has_B))
        if has_B:
            kwargs.update(oof_B=oof_B, test_B=test_B, oof_rhit_B=oof_rhit_B,
                           fold_rh_B=np.array(fold_rh_B), fold_mask_B=fold_mask_B)
        np.savez(state_file, **kwargs)
        print(f"[state] 저장: {state_file}")

    # sub_09 = average
    if has_B:
        oof_sub09  = (oof_A + oof_B) / 2
        test_sub09 = (test_A + test_B) / 2
        eval_mask = fold_mask_A & fold_mask_B
    else:
        oof_sub09  = oof_A.copy(); test_sub09 = test_A.copy()
        eval_mask = fold_mask_A

    pred = kalman_train[eval_mask] + oof_sub09[eval_mask]
    rh_sub09 = float((np.linalg.norm(pred - y_train[eval_mask], axis=-1) <= 0.01).mean())
    print(f"\n[sub_09] OOF (covered {eval_mask.sum()}): {rh_sub09:.4f}  (원본 full 0.6612)")

    # Calibration
    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_cal  = oof_sub09  * ALPHA[None, :]
    test_cal = test_sub09 * ALPHA[None, :]
    pred = kalman_train[eval_mask] + oof_cal[eval_mask]
    rh_cal = float((np.linalg.norm(pred - y_train[eval_mask], axis=-1) <= 0.01).mean())
    print(f"[calibrated] OOF: {rh_cal:.4f}  (Δ {rh_cal - rh_sub09:+.4f}, 원본 full 0.6625)")

    # Submission
    test_pos = kalman_test + test_cal
    out_csv = DATA_DIR / f"submission_v23_cpu_{mode}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"[submission] {out_csv}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    import datetime as _dt
    entry = {
        "version": f"v23_cpu_{mode}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode, "mode_config": M,
        "approach": "Kalman residual + GRU+F+W (Trojan_Horse 0.6780 CPU)",
        "kalman_baseline_rhit": rh_kal,
        "setup_A_oof_rhit": oof_rhit_A,
        "setup_B_oof_rhit": oof_rhit_B if has_B else None,
        "sub09_oof_rhit": rh_sub09,
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
    print(f"v23 CPU ({mode}) 완료")
    print("=" * 60)
    print(f"  OOF calibrated: {rh_cal:.4f}")
    print(f"  covered rows:   {int(eval_mask.sum())}/{len(y_train)}")
    print(f"  제출 CSV:       {out_csv}")


if __name__ == "__main__":
    main()

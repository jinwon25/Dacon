"""v126_fft_neural_ode.py — v120 Neural ODE에 FFT magnitude/phase feature 추가.

목적: v120과 동일 backbone (RK4 ODE)이지만 frequency-domain feature를 scalar에 합쳐
       paradigm 다양성 추가. corr_3d v112 < 0.93 가능성 + 새 base feature.

추가 feature:
  - 11 obs의 x/y/z 각각 rfft → magnitude (6 bins) + phase (6 bins, 첫 bin 제외 5)
  - 36 + 18 = 36 추가 scalar feature (대략)

사용:
  python scripts/v126_fft_neural_ode.py --mode smoke
  python scripts/v126_fft_neural_ode.py --mode fast
  python scripts/v126_fft_neural_ode.py --mode full
"""
from __future__ import annotations
import argparse, gc, json, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    DT, T_PRED, load_data, get_kalman, get_scalar_feats,
    build_seq, build_tier3,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
)
from v120_neural_ode import NeuralODEModel, mirror_seq, mirror_target, rotate_xy_seq, loss_combined

PROJ = SCRIPT_DIR.parent
DATA = PROJ / "open"
CACHE = PROJ / "cache"
OUT = PROJ / "open"

MODE = {
    "smoke": dict(n_folds=1, n_seeds=1, max_epochs=30, patience=8, batch=256, lr=2e-3, wd=1e-3, mirror=False),
    "fast":  dict(n_folds=2, n_seeds=1, max_epochs=60, patience=12, batch=256, lr=2e-3, wd=1e-3, mirror=True),
    "full":  dict(n_folds=5, n_seeds=2, max_epochs=80, patience=15, batch=256, lr=2e-3, wd=1e-3, mirror=True),
}


def build_fft_feats(X):
    """X: (N, 11, 3). Return (N, F) where F = 3*(6+5) = 33.
    각 축에 대해 rfft → mag (6 bins) + phase (5 non-DC bins).
    """
    N = X.shape[0]
    # de-trend: subtract linear fit per sample per axis
    t = np.arange(11)
    A = np.stack([np.ones(11), t], axis=1)  # (11, 2)
    # X residual after linear fit
    out_feats = []
    for k in range(3):
        Xk = X[..., k]  # (N, 11)
        coef = np.linalg.lstsq(A, Xk.T, rcond=None)[0]  # (2, N)
        trend = (A @ coef).T  # (N, 11)
        resid = Xk - trend
        fft = np.fft.rfft(resid, axis=1)  # (N, 6)
        mag = np.abs(fft).astype(np.float32)
        ph = np.angle(fft[:, 1:]).astype(np.float32)  # exclude DC phase
        out_feats.append(mag); out_feats.append(np.cos(ph)); out_feats.append(np.sin(ph))
    return np.concatenate(out_feats, axis=1).astype(np.float32)  # (N, 3*(6 + 5 + 5)) = (N, 48)


def run_kfold(X_train, X_test, y_train,
              theta_train, theta_test,
              X_scal_tr, X_scal_te, cfg, device="cpu"):
    n_folds = cfg["n_folds"]; n_seeds = cfg["n_seeds"]
    max_epochs = cfg["max_epochs"]; patience = cfg["patience"]
    batch = cfg["batch"]; lr = cfg["lr"]; wd = cfg["wd"]
    mirror_on = cfg["mirror"]
    N = X_train.shape[0]

    seq_tr_raw = build_seq(X_train); seq_te_raw = build_seq(X_test)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    fft_tr = build_fft_feats(X_train); fft_te = build_fft_feats(X_test)
    print(f"[v126] FFT feats shape: tr={fft_tr.shape}, te={fft_te.shape}")

    def rot_seq(seq, theta):
        out = seq.copy()
        for k in range(3):
            blk = out[..., 3*k:3*k+3]
            out[..., 3*k:3*k+3] = rotate_xy_seq(blk, theta)
        return out
    seq_tr = rot_seq(seq_tr_raw, theta_train)
    seq_te = rot_seq(seq_te_raw, theta_test)

    init_vel_tr = seq_tr[:, -1, 3:6].astype(np.float32)
    init_vel_te = seq_te[:, -1, 3:6].astype(np.float32)
    speed_tr = np.linalg.norm(init_vel_tr, axis=-1).astype(np.float32)
    speed_te = np.linalg.norm(init_vel_te, axis=-1).astype(np.float32)

    target_local = rotate_xy(y_train - X_train[:, -1], theta_train).astype(np.float32)

    # scalar features: X_scal + tier3 + fft (NEW)
    scal_tr_full = np.concatenate([X_scal_tr, tier3_tr, fft_tr], axis=-1).astype(np.float32)
    scal_te_full = np.concatenate([X_scal_te, tier3_te, fft_te], axis=-1).astype(np.float32)

    seq_flat_dim = seq_tr.shape[1] * seq_tr.shape[2]
    scal_dim = scal_tr_full.shape[1]
    print(f"[v126] N={N}, seq_flat_dim={seq_flat_dim}, scal_dim={scal_dim} (with FFT)")

    if mirror_on:
        seq_tr_m = mirror_seq(seq_tr)
        init_vel_tr_m = seq_tr_m[:, -1, 3:6].astype(np.float32)
        target_local_m = mirror_target(target_local)

    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))[:n_folds]

    oof_local = np.zeros((N, 3), dtype=np.float32)
    fold_mask = np.zeros(N, dtype=bool)
    test_per_fold = []
    fold_rh_list = []
    t0 = time.time()

    for fi, (tr, va) in enumerate(fold_iter):
        sc_seq = StandardScaler().fit(seq_tr[tr].reshape(-1, seq_tr.shape[2]))
        sc_scal = StandardScaler().fit(scal_tr_full[tr])
        seq_n = normalize_seq(seq_tr, sc_seq)
        seq_te_n = normalize_seq(seq_te, sc_seq)
        scal_n = sc_scal.transform(scal_tr_full).astype(np.float32)
        scal_te_n = sc_scal.transform(scal_te_full).astype(np.float32)
        if mirror_on:
            seq_m_n = normalize_seq(seq_tr_m, sc_seq)

        seq_flat = seq_n.reshape(N, -1)
        seq_te_flat = seq_te_n.reshape(seq_te.shape[0], -1)
        if mirror_on:
            seq_m_flat = seq_m_n.reshape(N, -1)

        def T(a, d=device): return torch.from_numpy(a).to(d)
        tr_idx = tr
        if mirror_on:
            seq_tr_t = torch.from_numpy(np.concatenate([seq_flat[tr_idx], seq_m_flat[tr_idx]], axis=0)).to(device)
            scal_tr_t = torch.from_numpy(np.concatenate([scal_n[tr_idx], scal_n[tr_idx]], axis=0)).to(device)
            vel_tr_t = torch.from_numpy(np.concatenate([init_vel_tr[tr_idx], init_vel_tr_m[tr_idx]], axis=0)).to(device)
            sp_tr_t = torch.from_numpy(np.concatenate([speed_tr[tr_idx], speed_tr[tr_idx]], axis=0)).to(device)
            tgt_tr_t = torch.from_numpy(np.concatenate([target_local[tr_idx], target_local_m[tr_idx]], axis=0)).to(device)
        else:
            seq_tr_t = T(seq_flat[tr_idx]); scal_tr_t = T(scal_n[tr_idx])
            vel_tr_t = T(init_vel_tr[tr_idx]); sp_tr_t = T(speed_tr[tr_idx])
            tgt_tr_t = T(target_local[tr_idx])

        seq_va_t = T(seq_flat[va]); scal_va_t = T(scal_n[va])
        vel_va_t = T(init_vel_tr[va]); sp_va_t = T(speed_tr[va])

        seq_te_t = T(seq_te_flat); scal_te_t = T(scal_te_n)
        vel_te_t = T(init_vel_te); sp_te_t = T(speed_te)

        test_fold = np.zeros((seq_te.shape[0], 3), dtype=np.float32)

        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = NeuralODEModel(seq_dim=seq_flat_dim, scal_dim=scal_dim,
                                    latent_dim=64, hidden=64, n_steps=1).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_rh = -1; best_state = None; best_ep = 0; bad = 0
            n_tr = seq_tr_t.shape[0]
            for ep in range(1, max_epochs + 1):
                model.train()
                perm = torch.randperm(n_tr)
                ep_loss = 0; nb = 0
                for s in range(0, n_tr, batch):
                    idx = perm[s:s+batch]
                    pred = model(seq_tr_t[idx], scal_tr_t[idx], vel_tr_t[idx], sp_tr_t[idx])
                    loss, h, hit = loss_combined(pred, tgt_tr_t[idx], model._last_accels,
                                                  w_huber=100.0, w_hit=1.0, w_reg=1e-4)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item(); nb += 1
                sch.step()
                model.eval()
                with torch.no_grad():
                    pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                pv_global = X_train[va, -1] + inverse_rotate_xy(pv, theta_train[va])
                rh = float((np.linalg.norm(pv_global - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh; best_ep = ep; bad = 0
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad += 1
                if ep <= 5 or ep % 5 == 0 or bad >= patience:
                    print(f"  fold{fi} seed{seed} ep{ep:3d}/{max_epochs}: loss={ep_loss/nb:.4f} va R-Hit={rh:.4f} best={best_rh:.4f}@ep{best_ep}")
                if bad >= patience: break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                if mirror_on:
                    seq_va_m_flat = seq_m_flat[va]
                    init_vel_va_m = init_vel_tr_m[va]
                    pv_m = model(T(seq_va_m_flat), scal_va_t, T(init_vel_va_m), sp_va_t).cpu().numpy()
                    pv = 0.5 * (pv + mirror_target(pv_m))
                pte = model(seq_te_t, scal_te_t, vel_te_t, sp_te_t).cpu().numpy()
                if mirror_on:
                    seq_te_m_flat = mirror_seq(seq_te)
                    init_vel_te_m = seq_te_m_flat[:, -1, 3:6].astype(np.float32)
                    seq_te_m_n = normalize_seq(seq_te_m_flat, sc_seq).reshape(seq_te.shape[0], -1)
                    pte_m = model(T(seq_te_m_n), scal_te_t, T(init_vel_te_m), sp_te_t).cpu().numpy()
                    pte = 0.5 * (pte + mirror_target(pte_m))

            oof_local[va] += pv / n_seeds
            test_fold += pte / n_seeds

        fold_mask[va] = True
        test_per_fold.append(test_fold)
        fold_rh_list.append(best_rh)
        print(f"[v126] fold{fi} best R-Hit={best_rh:.4f}  elapsed {(time.time()-t0)/60:.1f}m")

    oof_global = X_train[fold_mask, -1] + inverse_rotate_xy(oof_local[fold_mask], theta_train[fold_mask])
    rh_oof = float((np.linalg.norm(oof_global - y_train[fold_mask], axis=-1) <= 0.01).mean())
    print(f"[v126] OOF R-Hit = {rh_oof:.4f}  (covered {fold_mask.sum()}/{N})")
    test_local = np.mean(test_per_fold, axis=0)
    test_global = X_test[:, -1] + inverse_rotate_xy(test_local, theta_test)
    oof_global_full = np.zeros((N, 3), dtype=np.float32)
    oof_global_full[fold_mask] = oof_global
    return oof_local, oof_global_full, fold_mask, test_global, rh_oof, fold_rh_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=list(MODE.keys()))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = MODE[args.mode]
    tag = args.tag or args.mode
    state_file = CACHE / f"v126_{tag}_state.npz"
    sub_file = OUT / f"submission_v126_{tag}.csv"

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"v126 FFT Neural ODE mode={args.mode}  threads={torch.get_num_threads()}")

    X_train, X_test, y_train, sub = load_data()
    kalman_train, kalman_test, _ = get_kalman(X_train, X_test)
    X_scal_tr, X_scal_te = get_scalar_feats(X_train, X_test, {"loo_sample":2000}, "fast")
    v_last_tr = (X_train[:,-1]-X_train[:,-2])/DT
    v_last_te = (X_test[:,-1]-X_test[:,-2])/DT
    theta_train, theta_test = yaw_angle(v_last_tr), yaw_angle(v_last_te)

    t0 = time.time()
    oof_local, oof_global, fold_mask, test_global, rh_oof, fold_rh = run_kfold(
        X_train, X_test, y_train, theta_train, theta_test, X_scal_tr, X_scal_te, cfg, device)
    print(f"[v126] total {(time.time()-t0)/60:.1f}m")

    np.savez(state_file,
              oof_local=oof_local, oof_global=oof_global, fold_mask=fold_mask,
              test_global=test_global, rh_oof=rh_oof,
              fold_rh=np.array(fold_rh), theta_train=theta_train, theta_test=theta_test)
    print(f"[v126] state saved: {state_file}")
    sub = pd.read_csv(DATA / "sample_submission.csv")
    sub[["x","y","z"]] = test_global
    sub.to_csv(sub_file, index=False)
    print(f"[v126] submission saved: {sub_file}")

if __name__ == "__main__":
    main()

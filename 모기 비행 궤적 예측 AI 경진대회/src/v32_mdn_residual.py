"""v32_mdn_residual.py — Mixture Density Network residual head on v23 base.

Bishop 1994 (MDN) + Chai et al. CVPR 2019 (MultiPath) + Waymo 2022 (MultiPath++).

문제: v23/v26 deterministic single-mode → multi-modal future (hover vs sharp turn) 사이
평균 fallback. 1cm threshold에서 mode 사이 평균은 fail.

해결: K=8 Gaussian mixture over 3D residual.
  Loss = NLL + 0.5 × softhit_min_over_mixture (1cm threshold direct surrogate)
  Inference: argmax mode (best π_k의 mean) — 가장 likely future 선택.

학습 시간: 5-fold × 2-seed × 1-setup × 80ep ≈ 30~50분 (single setup, MDN larger)

출력:
  - cache/v32_mdn_state.npz
  - open/submission_v32_cpu.csv
  - run_log append
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    DT, build_seq, build_tier3, build_scalar_feats,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
)


PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"

LOG_2PI = math.log(2 * math.pi)


# ============================================================
# MDN Model
# ============================================================
class GRUModelMDN(nn.Module):
    """GRU encoder + MDN head + aux heads (F, W) for stable learning."""
    def __init__(self, n_channels=9, scal_dim=40, hidden=64, layers=1,
                  fc=128, p=0.2, K=8, aux_dims=(3, 3),
                  main_scale_cm=2.0, sigma_min_cm=0.1, sigma_max_cm=3.0):
        super().__init__()
        self.K = K
        self.gru = nn.GRU(n_channels, hidden, num_layers=layers, batch_first=True,
                           dropout=p if layers > 1 else 0)
        self.fc1 = nn.Linear(hidden + scal_dim, fc)
        self.fc2 = nn.Linear(fc, fc // 2)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)

        # MDN head: K × (3 mean + 3 log_sigma + 1 mix_logit) = 7K
        self.mdn_means = nn.Linear(fc // 2, K * 3)
        self.mdn_log_sigmas = nn.Linear(fc // 2, K * 3)
        self.mdn_mix = nn.Linear(fc // 2, K)

        # aux F, W heads (deterministic, for regularization)
        self.aux_heads = nn.ModuleList([nn.Linear(fc // 2, d) for d in aux_dims])

        self.main_scale = main_scale_cm / 100.0
        self.sigma_min = sigma_min_cm / 100.0
        self.sigma_max = sigma_max_cm / 100.0

    def forward(self, seq, scal):
        out, _ = self.gru(seq)
        z = torch.cat([out[:, -1, :], scal], dim=1)
        z = self.act(self.fc1(z)); z = self.drop(z)
        z = self.act(self.fc2(z))

        # MDN: means tanh-bounded
        means = torch.tanh(self.mdn_means(z)).view(-1, self.K, 3) * self.main_scale
        # log_sigma clamp to reasonable range
        log_sigmas = self.mdn_log_sigmas(z).view(-1, self.K, 3)
        log_sigmas = torch.clamp(log_sigmas,
                                   math.log(self.sigma_min), math.log(self.sigma_max))
        mix_logits = self.mdn_mix(z)  # (B, K)

        # aux outputs
        aux_outs = [h(z) for h in self.aux_heads]
        return means, log_sigmas, mix_logits, aux_outs


# ============================================================
# Loss: MDN NLL + soft-hit
# ============================================================
def mdn_nll(means, log_sigmas, mix_logits, y_true):
    """means (B,K,3), log_sigmas (B,K,3), mix_logits (B,K), y_true (B,3)."""
    sigmas = torch.exp(log_sigmas)  # (B, K, 3)
    diff = means - y_true.unsqueeze(1)  # (B, K, 3)
    # log p(y | component k) = -0.5 (diff/σ)^2 - log σ - 0.5 log 2π, summed over 3 dims
    log_prob = -0.5 * (diff / sigmas) ** 2 - log_sigmas - 0.5 * LOG_2PI  # (B, K, 3)
    log_prob = log_prob.sum(-1)  # (B, K) — joint log prob of 3D residual under component k
    log_mix = F.log_softmax(mix_logits, dim=-1)  # (B, K)
    log_p_y = torch.logsumexp(log_mix + log_prob, dim=-1)  # (B,)
    return -log_p_y.mean()


def mdn_softhit_bonus(means, mix_logits, y_true, beta=0.002):
    """가장 가까운 mixture mode의 distance를 1cm 안으로 유도하는 soft-hit term.

    각 component의 mean과 y_true의 거리 → soft-hit per component → mix-weighted soft-hit max.
    """
    diff = means - y_true.unsqueeze(1)  # (B, K, 3)
    d_per_k = torch.sqrt((diff ** 2).sum(-1) + 1e-12)  # (B, K)
    # soft hit: 1 if d < 1cm, 0 otherwise (smooth)
    hit_per_k = torch.sigmoid((0.01 - d_per_k) / beta)  # (B, K) ∈ [0, 1]
    # mix-weighted (component being used) hit
    mix = F.softmax(mix_logits, dim=-1)  # (B, K)
    # max-weighted: 가장 likely component의 hit rate
    # argmax 대신 weighted to keep gradient
    hit = (mix * hit_per_k).sum(-1)  # (B,)
    return -hit.mean()  # minimize negative hit = maximize hit


def loss_mdn(means, log_sigmas, mix_logits, y_true, lambda_softhit=0.5):
    return mdn_nll(means, log_sigmas, mix_logits, y_true) + \
           lambda_softhit * mdn_softhit_bonus(means, mix_logits, y_true)


def loss_aux_euclid(p, t):
    return torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12).mean()


# ============================================================
# Inference
# ============================================================
def mdn_argmax_mode(means, mix_logits):
    """argmax mode: mixture weight 가장 큰 component의 mean (deterministic)."""
    best_k = mix_logits.argmax(-1)  # (B,)
    return means[torch.arange(means.shape[0]), best_k]  # (B, 3)


def mdn_weighted_mean(means, mix_logits):
    """Weighted mean across components."""
    weights = F.softmax(mix_logits, dim=-1).unsqueeze(-1)  # (B, K, 1)
    return (weights * means).sum(1)  # (B, 3)


# ============================================================
# K-fold trainer
# ============================================================
def run_5fold_mdn(target_main, target_F, target_W,
                   seq_arr, scal_arr, seq_te, scal_te,
                   kalman_train, theta_train, theta_test, y_train,
                   config, n_seeds=2, n_folds=5, max_epochs=80, patience=15,
                   batch=256, K=8, lambda_softhit=0.5, lambda_F=0.3, lambda_W=0.3, device="cpu"):
    oof_argmax = np.zeros((len(target_main), 3))
    oof_weighted = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_argmax_folds, test_weighted_folds = [], []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    t0 = time.time()

    for fi, (tr, va) in enumerate(kf.split(np.arange(len(target_main)))):
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
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)

        seed_va_argmax, seed_va_weighted = [], []
        seed_te_argmax, seed_te_weighted = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = GRUModelMDN(
                n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                hidden=config["hidden"], layers=config["layers"],
                fc=config["fc"], p=config["p"], K=K,
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
                    means, log_sigmas, mix_logits, aux_outs = model(seq_tr_t[idx], scal_tr_t[idx])
                    loss = loss_mdn(means, log_sigmas, mix_logits, tgt_tr_t[idx],
                                     lambda_softhit=lambda_softhit)
                    loss = loss + lambda_F * loss_aux_euclid(aux_outs[0], F_tr_t[idx])
                    loss = loss + lambda_W * loss_aux_euclid(aux_outs[1], W_tr_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    means, log_sigmas, mix_logits, _ = model(seq_va_t, scal_va_t)
                    va_argmax = mdn_argmax_mode(means, mix_logits).cpu().numpy()
                va_unrot = inverse_rotate_xy(va_argmax, theta_train[va])
                pred = kalman_train[va] + va_unrot
                rh = float((np.linalg.norm(pred - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience: break

                if ep == 0 or (ep + 1) % 10 == 0:
                    print(f"  fold{fi+1} seed{s} ep{ep+1:3d}: rhit={rh:.4f} (best {best_rh:.4f}) "
                          f"[{(time.time()-t0)/60:.1f}m]", flush=True)

            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                m_va, _, l_va, _ = model(seq_va_t, scal_va_t)
                m_te, _, l_te, _ = model(seq_te_t, scal_te_t)
                va_argmax = mdn_argmax_mode(m_va, l_va).cpu().numpy()
                va_weighted = mdn_weighted_mean(m_va, l_va).cpu().numpy()
                te_argmax = mdn_argmax_mode(m_te, l_te).cpu().numpy()
                te_weighted = mdn_weighted_mean(m_te, l_te).cpu().numpy()
            seed_va_argmax.append(va_argmax); seed_va_weighted.append(va_weighted)
            seed_te_argmax.append(te_argmax); seed_te_weighted.append(te_weighted)
            del model; gc.collect()

        # Multi-seed avg
        va_argmax = np.mean(seed_va_argmax, axis=0)
        va_weighted = np.mean(seed_va_weighted, axis=0)
        te_argmax = np.mean(seed_te_argmax, axis=0)
        te_weighted = np.mean(seed_te_weighted, axis=0)
        oof_argmax[va] = va_argmax
        oof_weighted[va] = va_weighted
        fold_mask[va] = True
        test_argmax_folds.append(te_argmax)
        test_weighted_folds.append(te_weighted)

        val_unrot = inverse_rotate_xy(va_argmax, theta_train[va])
        pred_pos = kalman_train[va] + val_unrot
        rh_fold = float((np.linalg.norm(pred_pos - y_train[va], axis=-1) <= 0.01).mean())
        print(f"  ★ fold{fi+1}/{n_folds}: argmax R-Hit={rh_fold:.4f}  [{(time.time()-t0)/60:.1f}m]", flush=True)

    # Unrotate OOF
    oof_argmax_unrot = inverse_rotate_xy(oof_argmax, theta_train)
    oof_weighted_unrot = inverse_rotate_xy(oof_weighted, theta_train)
    pred_argmax = kalman_train + oof_argmax_unrot
    pred_weighted = kalman_train + oof_weighted_unrot
    rh_argmax = float((np.linalg.norm(pred_argmax - y_train, axis=-1) <= 0.01).mean())
    rh_weighted = float((np.linalg.norm(pred_weighted - y_train, axis=-1) <= 0.01).mean())

    test_argmax = np.mean([inverse_rotate_xy(rot, theta_test) for rot in test_argmax_folds], axis=0)
    test_weighted = np.mean([inverse_rotate_xy(rot, theta_test) for rot in test_weighted_folds], axis=0)

    print(f"\n  OOF argmax (5-fold): {rh_argmax:.4f}")
    print(f"  OOF weighted: {rh_weighted:.4f}")
    print(f"  소요: {(time.time()-t0)/60:.1f}m")
    return (oof_argmax_unrot, test_argmax, rh_argmax,
             oof_weighted_unrot, test_weighted, rh_weighted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--lambda-softhit", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v32 MDN residual: K={args.K} × 5-fold × {args.n_seeds}-seed")
    print(f"  loss = NLL + {args.lambda_softhit} × softhit_bonus")
    print(f"  inference: argmax mode + weighted mean (둘 다 평가)")
    print("=" * 60)

    # Load (v23 cache 재사용)
    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test, kalman_train_alt = kc["kalman_train"], kc["kalman_test"], kc["kalman_train_alt"]

    nc_noise = np.load(CACHE_DIR / "noise_fast.npz")
    scal_tr = build_scalar_feats(X_train, nc_noise["noise_p"], nc_noise["noise_s"], nc_noise["noise_l"])
    scal_te = build_scalar_feats(X_test, nc_noise["noise_p_test"], nc_noise["noise_s_test"])
    LOG_COLS = ["mean_speed","max_speed","speed_std","mean_acc","max_acc","max_jerk",
                "net_disp","|v_last|","|a_last|","|a_recent|","jerk_last","jerk_recent",
                "noise_poly2","noise_savgol","noise_loo"]
    for c in LOG_COLS:
        scal_tr[c] = np.log1p(scal_tr[c]); scal_te[c] = np.log1p(scal_te[c])
    X_scal_tr = np.concatenate([scal_tr.values.astype(np.float32), build_tier3(X_train)], axis=-1)
    X_scal_te = np.concatenate([scal_te.values.astype(np.float32), build_tier3(X_test)], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)

    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8 = rotate_xy(y_train - kalman_train, theta_train)
    target_F  = rotate_xy(y_train - X_train[:, -1], theta_train)
    target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)

    state_file = CACHE_DIR / "v32_mdn_state.npz"
    if state_file.exists() and not args.force:
        st = np.load(state_file)
        rh_argmax = float(st["rh_argmax"]); rh_weighted = float(st["rh_weighted"])
        oof_argmax = st["oof_argmax"]; test_argmax = st["test_argmax"]
        oof_weighted = st["oof_weighted"]; test_weighted = st["test_weighted"]
        print(f"[state] cached: argmax={rh_argmax:.4f}, weighted={rh_weighted:.4f}")
    else:
        # Single setup (lr=5e-4, do=0.3) — MDN larger model, A 패턴
        CFG = dict(hidden=64, layers=1, fc=128, lr=5e-4, p=0.3, wd=1e-4)
        result = run_5fold_mdn(
            target_T8, target_F, target_W, seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            config=CFG, n_seeds=args.n_seeds, n_folds=args.n_folds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
            K=args.K, lambda_softhit=args.lambda_softhit,
        )
        oof_argmax, test_argmax, rh_argmax, oof_weighted, test_weighted, rh_weighted = result

        np.savez(state_file,
                  oof_argmax=oof_argmax, test_argmax=test_argmax, rh_argmax=rh_argmax,
                  oof_weighted=oof_weighted, test_weighted=test_weighted, rh_weighted=rh_weighted)
        print(f"[state] saved: {state_file}")

    # Calibration
    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_arg_cal = oof_argmax * ALPHA[None, :]
    oof_wgt_cal = oof_weighted * ALPHA[None, :]
    test_arg_cal = test_argmax * ALPHA[None, :]
    test_wgt_cal = test_weighted * ALPHA[None, :]

    rh_arg_cal = float((np.linalg.norm(kalman_train + oof_arg_cal - y_train, axis=-1) <= 0.01).mean())
    rh_wgt_cal = float((np.linalg.norm(kalman_train + oof_wgt_cal - y_train, axis=-1) <= 0.01).mean())

    print(f"\n=== v32 MDN 결과 ===")
    print(f"  argmax OOF (raw): {rh_argmax:.4f}")
    print(f"  argmax OOF (cal): {rh_arg_cal:.4f}")
    print(f"  weighted OOF (raw): {rh_weighted:.4f}")
    print(f"  weighted OOF (cal): {rh_wgt_cal:.4f}")

    # Best inference mode
    if rh_arg_cal >= rh_wgt_cal:
        best_mode = "argmax"
        oof_best = oof_arg_cal; test_best = test_arg_cal; rh_best = rh_arg_cal
    else:
        best_mode = "weighted"
        oof_best = oof_wgt_cal; test_best = test_wgt_cal; rh_best = rh_wgt_cal
    print(f"  ★ best: {best_mode} OOF {rh_best:.4f}")

    test_pos = kalman_test + test_best
    out_csv = DATA_DIR / "submission_v32_mdn_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    import datetime as _dt
    entry = {
        "version": "v32_mdn_cpu",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"MDN K={args.K} + softhit + 5-fold × {args.n_seeds}-seed",
        "K": args.K, "lambda_softhit": args.lambda_softhit,
        "argmax_oof_raw": float(rh_argmax),
        "argmax_oof_cal": float(rh_arg_cal),
        "weighted_oof_raw": float(rh_weighted),
        "weighted_oof_cal": float(rh_wgt_cal),
        "best_mode": best_mode,
        "best_oof": float(rh_best),
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

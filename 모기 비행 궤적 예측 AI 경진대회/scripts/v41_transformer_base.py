"""v41_transformer_base.py — Causal Transformer base (v30 framework + Transformer 인코더).

v30이 GRU 인코더 + multi-aux head로 OOF 0.6588 도달. v41은 같은 framework
(Kalman 잔차 + yaw rotation + F/W aux + adv reweight)에 Transformer 인코더 적용.

v15에서 Transformer 단독은 약했음 (val 0.6460, public 0.629). 차이:
  - v15는 5-fold proper OOF 없음
  - adversarial reweighting 없음
  - multi-seed averaging 없음
  - boundary correction 없음

이번에는 v30/v35 검증된 framework 모두 적용 → paradigm 다양성 base.

학습: 5-fold × 2-seed × 2-setup × 80ep = 20 trainings × ~3분 = ~60분
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

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    DT, build_seq, build_tier3, build_scalar_feats,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
    loss_combo as _loss_combo_v23,
)
from v30_advanced_v23 import (
    compute_adv_weights, loss_combo_weighted, loss_aux_weighted,
)


PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"


class CausalTransformerMultiAux(nn.Module):
    """Causal Transformer encoder + multi-aux head (v30 framework 호환)."""
    def __init__(self, n_channels=9, scal_dim=40, d_model=64, nhead=4, num_layers=2,
                  dim_ff=128, fc=128, p=0.2, aux_dims=(3, 3), main_scale_cm=2.0):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 11, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=p, batch_first=True, activation='gelu',
            norm_first=True,  # Pre-LN (학습 안정)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model + scal_dim, fc)
        self.fc2 = nn.Linear(fc, fc // 2)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.head_main = nn.Linear(fc // 2, 3)
        self.aux_heads = nn.ModuleList([nn.Linear(fc // 2, d) for d in aux_dims])
        self.main_scale = main_scale_cm / 100.0

    def forward(self, seq, scal):
        # seq: (B, 11, n_channels)
        x = self.input_proj(seq) + self.pos_emb
        T = x.size(1)
        # Causal mask (마지막 step이 이전 모든 step 볼 수 있음)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x[:, -1, :])  # 마지막 step만 사용 (GRU last hidden 처럼)
        z = torch.cat([x, scal], dim=1)
        z = self.act(self.fc1(z)); z = self.drop(z)
        z = self.act(self.fc2(z))
        out_main = torch.tanh(self.head_main(z)) * self.main_scale
        outs_aux = [h(z) for h in self.aux_heads]
        return out_main, outs_aux


def run_5fold_multiseed_trans(target_main, target_F, target_W,
                                seq_arr, scal_arr, seq_te, scal_te,
                                sample_weight,
                                kalman_train, theta_train, theta_test, y_train,
                                config, n_seeds=2, n_folds=5, max_epochs=80, patience=15,
                                batch=256, lambda_F=0.3, lambda_W=0.3, device="cpu"):
    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold_seed = []
    fold_rh = []

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
            model = CausalTransformerMultiAux(
                n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                d_model=config["d_model"], nhead=config["nhead"],
                num_layers=config["layers"], dim_ff=config["dim_ff"],
                fc=config["fc"], p=config["p"],
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                      weight_decay=config["wd"])
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
    print(f"  OOF R-Hit (5-fold full): {oof_rhit:.4f}  {(time.time()-t0)/60:.1f}m")
    return oof_unrot, test_unrot, fold_rh, oof_rhit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v41 = Causal Transformer base + v30 framework (5-fold × {args.n_seeds}-seed × 2-setup)")
    print("=" * 60)

    # Load (v30 cache 재사용)
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
    print(f"[kalman] R-Hit: {rh_kal:.4f}")

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

    sample_w, adv_auc = compute_adv_weights(X_train, X_test)

    # Transformer configs (작은 모델 - 11 step 짧음)
    CFG_A = dict(d_model=64, nhead=4, layers=2, dim_ff=128, fc=128,
                  lr=3e-4, p=0.2, wd=1e-4)  # 안정 (낮은 lr)
    CFG_B = dict(d_model=64, nhead=4, layers=2, dim_ff=128, fc=128,
                  lr=8e-4, p=0.1, wd=1e-4)  # 공격

    state_file = CACHE_DIR / "v41_state.npz"
    if state_file.exists() and not args.force:
        st = np.load(state_file)
        oof_A, test_A, rh_A = st["oof_A"], st["test_A"], float(st["rh_A"])
        oof_B, test_B, rh_B = st["oof_B"], st["test_B"], float(st["rh_B"])
        print(f"[state] cached: A={rh_A:.4f}, B={rh_B:.4f}")
    else:
        print("\n" + "=" * 60); print(f"Setup A (lr=3e-4, p=0.2) × {args.n_seeds}-seed × 5-fold")
        print("=" * 60)
        oof_A, test_A, fold_rh_A, rh_A = run_5fold_multiseed_trans(
            target_T8, target_F, target_W, seq_tr, X_scal_tr, seq_te, X_scal_te, sample_w,
            kalman_train, theta_train, theta_test, y_train,
            config=CFG_A, n_seeds=args.n_seeds, n_folds=args.n_folds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
        )
        print("\n" + "=" * 60); print(f"Setup B (lr=8e-4, p=0.1) × {args.n_seeds}-seed × 5-fold")
        print("=" * 60)
        oof_B, test_B, fold_rh_B, rh_B = run_5fold_multiseed_trans(
            target_T8, target_F, target_W, seq_tr, X_scal_tr, seq_te, X_scal_te, sample_w,
            kalman_train, theta_train, theta_test, y_train,
            config=CFG_B, n_seeds=args.n_seeds, n_folds=args.n_folds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
        )
        np.savez(state_file,
                  oof_A=oof_A, test_A=test_A, rh_A=rh_A,
                  oof_B=oof_B, test_B=test_B, rh_B=rh_B,
                  sample_weight=sample_w, adv_auc=adv_auc)
        print(f"[state] saved")

    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_sub = (oof_A + oof_B) / 2
    test_sub = (test_A + test_B) / 2
    oof_cal = oof_sub * ALPHA[None, :]
    test_cal = test_sub * ALPHA[None, :]
    rh_cal = float((np.linalg.norm(kalman_train + oof_cal - y_train, axis=-1) <= 0.01).mean())

    # Reference
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    rh_v30 = float((np.linalg.norm(
        kalman_train + (st30["oof_A"] + st30["oof_B"])/2 * ALPHA[None,:] - y_train, axis=-1
    ) <= 0.01).mean())

    print(f"\n=== v41 결과 ===")
    print(f"  Setup A OOF: {rh_A:.4f}")
    print(f"  Setup B OOF: {rh_B:.4f}")
    print(f"  Calibrated OOF: {rh_cal:.4f}")
    print(f"  ★ v30 (GRU) baseline OOF: {rh_v30:.4f}")
    print(f"  ★ v41 (Transformer) OOF: {rh_cal:.4f}, Δ vs v30: {rh_cal - rh_v30:+.4f}")

    test_pos = kalman_test + test_cal
    out_csv = DATA_DIR / "submission_v41_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    log_path = PROJECT_DIR / "run_log.json"
    import datetime as _dt
    entry = {
        "version": "v41_transformer_base",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Causal Transformer + v30 framework",
        "v30_oof": float(rh_v30),
        "v41_setup_A_oof": float(rh_A),
        "v41_setup_B_oof": float(rh_B),
        "v41_calibrated_oof": float(rh_cal),
        "delta_vs_v30": float(rh_cal - rh_v30),
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

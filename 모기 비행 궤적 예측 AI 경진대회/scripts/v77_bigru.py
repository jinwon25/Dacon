"""v77_bigru.py — Bidirectional GRU + v23 framework (새 paradigm).

진단:
  - 모든 mean/median/per-axis/multi-way blend OOF 0.6748 plateau
  - 13-model SoftStacker weight도 새 paradigm 모델 거부 (0.04 합)
  - 결론: 현 pool 절대적 ceiling. 새 paradigm 학습 외 답 없음.

설계:
  - v23 framework (Kalman residual + GRU + F + W aux heads) 그대로
  - 핵심 차이: 단방향 GRU → 양방향 GRU (bidirectional)
  - 11 step 짧은 시퀀스에 양방향 효과 클 가능성
  - 5-fold × 3-seed full mode
"""
from __future__ import annotations

import argparse, datetime as _dt, gc, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    load_data, get_kalman, get_scalar_feats, build_tier3, build_seq,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
    MODE_CONFIGS, CACHE_DIR, DATA_DIR,
    loss_combo, loss_euclid,
)

PROJECT = SCRIPT_DIR.parent
DT = 0.040


class BiGRUMultiAux(nn.Module):
    """양방향 GRU + v23 동일한 main/F/W 헤드"""
    def __init__(self, n_channels=9, scal_dim=40, hidden=64, fc=128, p=0.2,
                 aux_dims=(3, 3), main_scale_cm=2.0):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=1, batch_first=True, bidirectional=True)
        # bidirectional → output dim = 2 * hidden
        # last step 양방향 concat: forward[-1] + backward[0] = 2*hidden
        self.fc1 = nn.Linear(2 * hidden + scal_dim, fc)
        self.fc2 = nn.Linear(fc, fc // 2)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.head_main = nn.Linear(fc // 2, 3)
        self.aux_heads = nn.ModuleList([nn.Linear(fc // 2, d) for d in aux_dims])
        self.main_scale = main_scale_cm / 100.0

    def forward(self, seq, scal):
        out, _ = self.gru(seq)
        # out shape: (B, T, 2 * hidden)
        # last step concat: forward at t=-1 (out[:, -1, :hidden])
        # backward at t=0 reversed → out[:, 0, hidden:]
        # but PyTorch bidirectional returns concat at each time step
        # so out[:, -1, :] already has [fwd_last, bwd_last(= bwd at t=-1)]
        # 더 정확한 양방향 요약: forward at t=-1 + backward at t=0 (각 방향의 마지막 정보)
        fwd_last = out[:, -1, :out.shape[-1] // 2]
        bwd_first = out[:, 0, out.shape[-1] // 2:]
        h_cat = torch.cat([fwd_last, bwd_first], dim=1)
        z = torch.cat([h_cat, scal], dim=1)
        z = self.act(self.fc1(z)); z = self.drop(z)
        z = self.act(self.fc2(z))
        out_main = torch.tanh(self.head_main(z)) * self.main_scale
        return out_main, [h(z) for h in self.aux_heads]


def run_kfold_bigru(target_main, target_F, target_W,
                   seq_arr, scal_arr, seq_te, scal_te,
                   kalman_train, theta_train, theta_test, y_train,
                   config, mode_cfg, lambda_F=0.3, lambda_W=0.3, device="cpu"):
    n_folds, n_seeds = mode_cfg["n_folds"], mode_cfg["n_seeds"]
    max_epochs, patience, batch = mode_cfg["max_epochs"], mode_cfg["patience"], mode_cfg["batch"]
    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold = []; fold_rh = []
    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(len(target_main))))
    if n_folds == 1: fold_iter = fold_iter[:1]
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
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)
        seed_val, seed_test = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = BiGRUMultiAux(
                n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                hidden=config["hidden"], fc=config["fc"], p=config["p"],
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
                else: no_improve += 1
                if no_improve >= patience: break
                if ep == 0 or (ep + 1) % 10 == 0:
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
    oof_unrot_all = np.zeros_like(target_main)
    oof_unrot_all[fold_mask] = inverse_rotate_xy(oof_rot[fold_mask], theta_train[fold_mask])
    pred = kalman_train[fold_mask] + oof_unrot_all[fold_mask]
    oof_rhit = float((np.linalg.norm(pred - y_train[fold_mask], axis=-1) <= 0.01).mean())
    test_unrot = np.mean([inverse_rotate_xy(rot, theta_test) for rot in test_per_fold], axis=0)
    print(f"  OOF R-Hit: {oof_rhit:.4f}  소요 {(time.time()-t0)/60:.1f}m")
    return oof_unrot_all, test_unrot, fold_rh, oof_rhit, fold_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=list(MODE_CONFIGS.keys()))
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    mode = args.mode; M = MODE_CONFIGS[mode]
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v77 BiGRU + v23 framework  mode={mode}")
    print("=" * 60)
    X_train, X_test, y_train, sub = load_data()
    kalman_train, kalman_test, kalman_train_alt = get_kalman(X_train, X_test)
    X_scal_b_tr, X_scal_b_te = get_scalar_feats(X_train, X_test, M, mode)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    X_scal_tr = np.concatenate([X_scal_b_tr, tier3_tr], axis=-1)
    X_scal_te = np.concatenate([X_scal_b_te, tier3_te], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)

    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8 = rotate_xy(y_train - kalman_train, theta_train)
    target_F  = rotate_xy(y_train - X_train[:, -1], theta_train)
    target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)

    state_file = CACHE_DIR / f"v77_bigru_state.npz"
    if state_file.exists() and not args.force_retrain:
        st = np.load(state_file)
        oof_A, test_A = st["oof_A"], st["test_A"]
        oof_rhit_A = float(st["oof_rhit_A"])
        oof_B, test_B = st["oof_B"], st["test_B"]
        oof_rhit_B = float(st["oof_rhit_B"])
        print(f"[state] cache 로드: A={oof_rhit_A:.4f}, B={oof_rhit_B:.4f}")
    else:
        CONFIG_A = dict(hidden=64, fc=128, lr=5e-4, p=0.3, wd=1e-4)
        CONFIG_B = dict(hidden=64, fc=128, lr=1e-3, p=0.1, wd=1e-4)
        print("=" * 60); print("Setup A (lr=5e-4, do=0.3)"); print("=" * 60)
        oof_A, test_A, fold_rh_A, oof_rhit_A, mask_A = run_kfold_bigru(
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            config=CONFIG_A, mode_cfg=M, device=device,
        )
        print("\n" + "=" * 60); print("Setup B (lr=1e-3, do=0.1)"); print("=" * 60)
        oof_B, test_B, fold_rh_B, oof_rhit_B, mask_B = run_kfold_bigru(
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            config=CONFIG_B, mode_cfg=M, device=device,
        )
        np.savez(state_file, oof_A=oof_A, test_A=test_A, oof_rhit_A=oof_rhit_A,
                 oof_B=oof_B, test_B=test_B, oof_rhit_B=oof_rhit_B)
        print(f"[state] 저장: {state_file}")

    oof_avg = (oof_A + oof_B) / 2
    test_avg = (test_A + test_B) / 2
    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_cal = oof_avg * ALPHA[None, :]; test_cal = test_avg * ALPHA[None, :]
    pred = kalman_train + oof_cal
    rh_cal = float((np.linalg.norm(pred - y_train, axis=-1) <= 0.01).mean())
    print(f"\n[v77 BiGRU] OOF avg cal: {rh_cal:.4f}  (v23 GRU full: 0.6557~0.6587)")
    test_pos = kalman_test + test_cal
    out = DATA_DIR / f"submission_v77_bigru.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}).to_csv(out, index=False)
    print(f"  [submission] {out.name}")

    entry = {"version": "v77_bigru", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": "v23 framework with bidirectional GRU encoder",
             "rh_A": float(oof_rhit_A), "rh_B": float(oof_rhit_B), "rh_avg_cal": rh_cal,
             "submission": str(out)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

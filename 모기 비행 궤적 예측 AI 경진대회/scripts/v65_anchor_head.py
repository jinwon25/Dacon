"""v65_anchor_head.py — K-means Anchor Classification + per-anchor Residual head.

Plateau 본질 진단:
  - 우리 metric R-Hit@1cm = 본질이 classification (hit/miss)
  - 우리 모델 = 모두 regression (MSE/L1 + soft-hit)
  - Argoverse/Waymo SOTA (MultiPath++, TNT, DenseTNT) 모두 anchor classification + residual
  - 우리 v32 MDN 실패 = anchor 미고정 → mode collapse (Makansi CVPR 2019)

설계:
  1. train y_disp (yaw-rotated frame) K-means → K anchor (사전 고정, 학습 무관)
  2. v23 GRU encoder (재사용) → feature
  3. 두 head:
     - cls_head: Linear → K logit (softmax)
     - res_head: Linear → K × 3 residual
  4. Loss:
     - CE (nearest anchor 분류)
     - Smooth-L1 (assigned anchor의 residual)
     - + soft prediction direct loss (가중 0.5)
  5. Inference: hard (argmax + residual) vs soft (weighted sum) 둘 다 측정

핵심:
  - anchor frame: yaw-rotated (모든 sample의 forward axis를 +x로 정렬)
  - K=64 default (10000 samples / 64 = ~156 per cluster, 충분)
  - residual cap: atanh로 ±3cm
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    load_data, get_kalman, get_scalar_feats, build_tier3, build_seq,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
    MODE_CONFIGS, CACHE_DIR, DATA_DIR,
)

PROJECT_DIR = SCRIPT_DIR.parent

DT = 0.040


def rhit(pred, y):
    return float((np.linalg.norm(pred - y, axis=-1) <= 0.01).mean())


class AnchorModel(nn.Module):
    """v23 GRU + scalar encoder + (anchor_cls, anchor_residual) head."""
    def __init__(self, n_ch=9, scal_dim=40, K=64, hidden=64, fc=128, p=0.3,
                 res_cap_cm=3.0):
        super().__init__()
        self.K = K
        self.res_cap = res_cap_cm / 100.0
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden + scal_dim, fc)
        self.fc2 = nn.Linear(fc, fc // 2)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.cls_head = nn.Linear(fc // 2, K)
        self.res_head = nn.Linear(fc // 2, K * 3)

    def forward(self, seq, scal):
        out, _ = self.gru(seq)
        z = torch.cat([out[:, -1, :], scal], dim=1)
        z = self.act(self.fc1(z)); z = self.drop(z)
        z = self.act(self.fc2(z))
        logits = self.cls_head(z)                # (B, K)
        res = torch.tanh(self.res_head(z).view(-1, self.K, 3)) * self.res_cap  # (B, K, 3)
        return logits, res


def compute_anchors(y_disp_rot, K, seed=0):
    """K-means anchor (yaw-rotated displacement)."""
    print(f"[anchor] K-means K={K} on yaw-rotated y_disp...")
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    km.fit(y_disp_rot)
    anchors = km.cluster_centers_.astype(np.float32)  # (K, 3)
    nearest = km.labels_  # (N,) anchor assignment
    # diagnostics
    nearest_disp = anchors[nearest]
    err = np.linalg.norm(nearest_disp - y_disp_rot, axis=-1)
    print(f"[anchor] anchor 단독 R-Hit (residual=0) = {(err <= 0.01).mean():.4f}")
    print(f"[anchor] mean err to nearest anchor = {err.mean()*100:.2f} cm, p95 = {np.percentile(err, 95)*100:.2f} cm")
    return anchors, nearest


def loss_anchor(logits, res, target_disp_rot, anchors_t, w_cls=1.0, w_res=1.0, w_soft=0.5):
    """
    logits: (B, K)
    res: (B, K, 3)
    target_disp_rot: (B, 3)
    anchors_t: (K, 3)
    """
    B, K, _ = res.shape
    # nearest anchor
    d = torch.cdist(target_disp_rot.unsqueeze(1), anchors_t.unsqueeze(0).expand(B, K, 3))  # (B, 1, K)
    nearest_idx = d.squeeze(1).argmin(dim=-1)  # (B,)
    # CE loss
    ce = F.cross_entropy(logits, nearest_idx)
    # residual loss (assigned anchor만)
    target_res = target_disp_rot - anchors_t[nearest_idx]  # (B, 3)
    idx_exp = nearest_idx.view(-1, 1, 1).expand(-1, 1, 3)
    pred_res_assigned = res.gather(1, idx_exp).squeeze(1)
    res_loss = F.smooth_l1_loss(pred_res_assigned, target_res, beta=0.005)
    # soft prediction loss
    prob = F.softmax(logits, dim=-1)  # (B, K)
    pred_disp_soft = (prob.unsqueeze(-1) * (anchors_t.unsqueeze(0) + res)).sum(dim=1)  # (B, 3)
    soft_loss = F.smooth_l1_loss(pred_disp_soft, target_disp_rot, beta=0.005)
    return w_cls * ce + w_res * res_loss + w_soft * soft_loss


def predict(model, seq, scal, anchors_t, mode="soft", device="cpu"):
    model.eval()
    with torch.no_grad():
        logits, res = model(seq, scal)
        prob = F.softmax(logits, dim=-1)
        if mode == "soft":
            pred_disp = (prob.unsqueeze(-1) * (anchors_t.unsqueeze(0) + res)).sum(dim=1)
        else:  # hard
            idx = prob.argmax(dim=-1)  # (B,)
            idx_exp = idx.view(-1, 1, 1).expand(-1, 1, 3)
            pred_res = res.gather(1, idx_exp).squeeze(1)
            pred_disp = anchors_t[idx] + pred_res
    return pred_disp.cpu().numpy(), prob.cpu().numpy()


def run_kfold_anchor(seq_arr, scal_arr, seq_te, scal_te,
                    y_disp_rot, kalman_train, theta_train, theta_test, y_train,
                    anchors, K, mode_cfg, hidden=64, fc=128, p=0.3, lr=5e-4, wd=1e-4,
                    device="cpu"):
    n_folds, n_seeds = mode_cfg["n_folds"], mode_cfg["n_seeds"]
    max_epochs, patience, batch = mode_cfg["max_epochs"], mode_cfg["patience"], mode_cfg["batch"]
    N = y_disp_rot.shape[0]

    oof_soft = np.zeros((N, 3))
    oof_hard = np.zeros((N, 3))
    test_per_fold_soft, test_per_fold_hard = [], []
    fold_mask = np.zeros(N, dtype=bool)
    fold_rh_soft, fold_rh_hard = [], []

    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))
    if n_folds == 1: fold_iter = fold_iter[:1]
    anchors_t = torch.from_numpy(anchors).to(device)
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
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)
        y_tr_t = T(y_disp_rot[tr].astype(np.float32))

        seed_val_soft, seed_val_hard = [], []
        seed_test_soft, seed_test_hard = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = AnchorModel(n_ch=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                                K=K, hidden=hidden, fc=fc, p=p).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

            best_rh, best_state, no_improve = -1.0, None, 0
            n_tr = seq_tr_t.shape[0]
            for ep in range(max_epochs):
                model.train()
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr, batch):
                    idx = perm[i:i+batch]
                    opt.zero_grad()
                    logits, res = model(seq_tr_t[idx], scal_tr_t[idx])
                    loss = loss_anchor(logits, res, y_tr_t[idx], anchors_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                # validation: anchor head는 y_disp 직접 학습 → pred_pos = x_last + inverse_rotate(pred_disp)
                pv_hard, _ = predict(model, seq_va_t, scal_va_t, anchors_t, mode="hard", device=device)
                pv_unrot = inverse_rotate_xy(pv_hard, theta_train[va])
                pred_pos = x_last_train[va] + pv_unrot
                rh_hard = float((np.linalg.norm(pred_pos - y_train[va], axis=-1) <= 0.01).mean())

                if rh_hard > best_rh:
                    best_rh = rh_hard
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience: break
                if ep == 0 or (ep + 1) % 5 == 0:
                    print(f"  fold{fi+1} seed{s} ep{ep+1:3d}: hard_rhit={rh_hard:.4f} (best {best_rh:.4f})  "
                          f"[{(time.time()-t0)/60:.1f}m]", flush=True)

            model.load_state_dict(best_state)
            pv_s, _ = predict(model, seq_va_t, scal_va_t, anchors_t, mode="soft", device=device)
            pv_h, _ = predict(model, seq_va_t, scal_va_t, anchors_t, mode="hard", device=device)
            pt_s, _ = predict(model, seq_te_t, scal_te_t, anchors_t, mode="soft", device=device)
            pt_h, _ = predict(model, seq_te_t, scal_te_t, anchors_t, mode="hard", device=device)
            seed_val_soft.append(pv_s); seed_val_hard.append(pv_h)
            seed_test_soft.append(pt_s); seed_test_hard.append(pt_h)
            del model; gc.collect()

        val_s = np.mean(seed_val_soft, axis=0); val_h = np.mean(seed_val_hard, axis=0)
        test_s = np.mean(seed_test_soft, axis=0); test_h = np.mean(seed_test_hard, axis=0)
        oof_soft[va] = val_s; oof_hard[va] = val_h
        fold_mask[va] = True
        test_per_fold_soft.append(test_s); test_per_fold_hard.append(test_h)

        # fold report
        pred_s = x_last_train[va] + inverse_rotate_xy(val_s, theta_train[va])
        pred_h = x_last_train[va] + inverse_rotate_xy(val_h, theta_train[va])
        rh_s = float((np.linalg.norm(pred_s - y_train[va], axis=-1) <= 0.01).mean())
        rh_h = float((np.linalg.norm(pred_h - y_train[va], axis=-1) <= 0.01).mean())
        fold_rh_soft.append(rh_s); fold_rh_hard.append(rh_h)
        print(f"  ★ fold {fi+1}/{len(fold_iter)}: soft={rh_s:.4f}, hard={rh_h:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    # final OOF
    if fold_mask.sum() == 0:
        return None, None, None, None, None, None
    oof_soft_pos = np.zeros_like(oof_soft)
    oof_hard_pos = np.zeros_like(oof_hard)
    oof_soft_pos[fold_mask] = x_last_train[fold_mask] + inverse_rotate_xy(oof_soft[fold_mask], theta_train[fold_mask])
    oof_hard_pos[fold_mask] = x_last_train[fold_mask] + inverse_rotate_xy(oof_hard[fold_mask], theta_train[fold_mask])
    rh_soft = rhit(oof_soft_pos[fold_mask], y_train[fold_mask])
    rh_hard = rhit(oof_hard_pos[fold_mask], y_train[fold_mask])
    test_soft_pos = x_last_test + np.mean([inverse_rotate_xy(t, theta_test) for t in test_per_fold_soft], axis=0)
    test_hard_pos = x_last_test + np.mean([inverse_rotate_xy(t, theta_test) for t in test_per_fold_hard], axis=0)
    print(f"  OOF soft: {rh_soft:.4f}  hard: {rh_hard:.4f}")
    return oof_soft_pos, oof_hard_pos, test_soft_pos, test_hard_pos, rh_soft, rh_hard


# module-level globals so run_kfold_anchor sees them after main() sets
x_last_train = None
x_last_test = None


def main():
    global x_last_train, x_last_test
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fast", choices=list(MODE_CONFIGS.keys()))
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    mode = args.mode; M = MODE_CONFIGS[mode]; K = args.K

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v65 Anchor Head — K={K}, mode={mode}, {M}")
    print(f"device={device}, threads={torch.get_num_threads()}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()
    x_last_train = X_train[:, -1, :]
    x_last_test = X_test[:, -1, :]

    # scalar features (v23 그대로, kalman cache 의존 우회용)
    _, _, _ = get_kalman(X_train, X_test)  # ensure cache
    X_scal_base_tr, X_scal_base_te = get_scalar_feats(X_train, X_test, M, mode)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    X_scal_tr = np.concatenate([X_scal_base_tr, tier3_tr], axis=-1)
    X_scal_te = np.concatenate([X_scal_base_te, tier3_te], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)
    print(f"[feat] X_scal {X_scal_tr.shape}, seq {seq_tr.shape}")

    # yaw + y_disp (rotated frame, no kalman)
    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    y_disp_rot = rotate_xy(y_train - x_last_train, theta_train)
    print(f"[y_disp] median |y_disp| = {np.linalg.norm(y_disp_rot, axis=-1).mean()*100:.2f} cm")

    # K-means anchors
    anchors, nearest = compute_anchors(y_disp_rot, K=K)

    # kalman은 안 씀 (anchor head는 x_last에서 직접 학습)
    state_file = CACHE_DIR / f"v65_K{K}_state.npz"
    if state_file.exists() and not args.force_retrain:
        st = np.load(state_file)
        oof_s, oof_h = st["oof_soft"], st["oof_hard"]
        test_s, test_h = st["test_soft"], st["test_hard"]
        rh_s, rh_h = float(st["rh_soft"]), float(st["rh_hard"])
        anchors_loaded = st["anchors"]
        print(f"[state] cache 로드: soft={rh_s:.4f}, hard={rh_h:.4f}")
    else:
        oof_s, oof_h, test_s, test_h, rh_s, rh_h = run_kfold_anchor(
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            y_disp_rot, None, theta_train, theta_test, y_train,
            anchors, K, M, device=device,
        )
        np.savez(state_file, oof_soft=oof_s, oof_hard=oof_h,
                 test_soft=test_s, test_hard=test_h,
                 rh_soft=rh_s, rh_hard=rh_h, anchors=anchors)
        print(f"[state] 저장: {state_file}")

    # submission
    out_soft = DATA_DIR / f"submission_v65_K{K}_soft.csv"
    out_hard = DATA_DIR / f"submission_v65_K{K}_hard.csv"
    pd.DataFrame({"id": sub["id"], "x": test_s[:,0], "y": test_s[:,1], "z": test_s[:,2]}).to_csv(out_soft, index=False)
    pd.DataFrame({"id": sub["id"], "x": test_h[:,0], "y": test_h[:,1], "z": test_h[:,2]}).to_csv(out_hard, index=False)
    print(f"\n[v65 결과 K={K}]")
    print(f"  OOF soft: {rh_s:.4f}")
    print(f"  OOF hard: {rh_h:.4f}")
    print(f"  submission: {out_soft.name}, {out_hard.name}")

    # run log
    entry = {
        "version": f"v65_anchor_K{K}_{mode}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"K-means anchor classification + per-anchor residual (K={K}, yaw-rotated frame)",
        "K": K, "mode": mode, "mode_config": M,
        "rh_soft": rh_s, "rh_hard": rh_h,
        "submission_soft": str(out_soft),
        "submission_hard": str(out_hard),
    }
    log_path = PROJECT_DIR / "run_log.json"
    logs = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f: logs = json.load(f)
            if not isinstance(logs, list): logs = [logs]
        except Exception: logs = []
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f: json.dump(logs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

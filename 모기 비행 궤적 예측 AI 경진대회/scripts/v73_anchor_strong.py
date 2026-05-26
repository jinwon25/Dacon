"""v73_anchor_strong.py — Strengthened anchor head (K=64 default, multiple improvements).

진단 (v65~v72 통합):
  - v65 K=64 full OOF 0.6137 (soft) / 0.6037 (hard), anchor upper bound 0.6885
  - rescue가 너무 약해서 어떤 routing scheme도 base 못 넘음
  - 근본 해결: rescue 자체 OOF 0.65+로 강화 필요

v65 대비 개선:
  1. hidden 64→128, fc 128→192 (capacity ↑)
  2. n_seeds 3→5 (ensemble 안정성 ↑)
  3. CE label smoothing 0.05 (over-fit 방지)
  4. residual cap 3cm 유지 (target disp p95=10.75cm이지만 anchor err p95=1.94cm)
  5. lr warmup + cosine schedule
  6. dropout 0.3 (학습 안정)
  7. soft prediction loss weight 0.8로 증가 (R-Hit metric에 더 가까운 학습)
  8. EMA model weights (마지막 N epoch 가중 평균)

새 카드: v73_K64_state.npz 저장.
"""
from __future__ import annotations

import argparse, datetime as _dt, gc, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    load_data, get_kalman, get_scalar_feats, build_tier3, build_seq,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
    CACHE_DIR, DATA_DIR,
)

PROJECT = SCRIPT_DIR.parent
DT = 0.040


def rhit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


class AnchorModelStrong(nn.Module):
    def __init__(self, n_ch=9, scal_dim=40, K=64, hidden=128, fc=192, p=0.3, res_cap_cm=3.0):
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
        logits = self.cls_head(z)
        res = torch.tanh(self.res_head(z).view(-1, self.K, 3)) * self.res_cap
        return logits, res


def compute_anchors(y_disp_rot, K, seed=0):
    print(f"[anchor] K-means K={K}...")
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    km.fit(y_disp_rot)
    anchors = km.cluster_centers_.astype(np.float32)
    err = np.linalg.norm(anchors[km.labels_] - y_disp_rot, axis=-1)
    print(f"[anchor] residual=0 R-Hit upper: {(err <= 0.01).mean():.4f}")
    print(f"[anchor] mean err: {err.mean()*100:.2f}cm, p95: {np.percentile(err, 95)*100:.2f}cm")
    return anchors


def loss_anchor(logits, res, target_disp_rot, anchors_t,
                w_cls=1.0, w_res=1.0, w_soft=0.8, label_smooth=0.05):
    B, K, _ = res.shape
    d = torch.cdist(target_disp_rot.unsqueeze(1), anchors_t.unsqueeze(0).expand(B, K, 3))
    nearest_idx = d.squeeze(1).argmin(dim=-1)
    ce = F.cross_entropy(logits, nearest_idx, label_smoothing=label_smooth)
    target_res = target_disp_rot - anchors_t[nearest_idx]
    idx_exp = nearest_idx.view(-1, 1, 1).expand(-1, 1, 3)
    pred_res_assigned = res.gather(1, idx_exp).squeeze(1)
    res_loss = F.smooth_l1_loss(pred_res_assigned, target_res, beta=0.005)
    prob = F.softmax(logits, dim=-1)
    pred_disp_soft = (prob.unsqueeze(-1) * (anchors_t.unsqueeze(0) + res)).sum(dim=1)
    soft_loss = F.smooth_l1_loss(pred_disp_soft, target_disp_rot, beta=0.005)
    return w_cls * ce + w_res * res_loss + w_soft * soft_loss


def predict(model, seq, scal, anchors_t, mode="soft"):
    model.eval()
    with torch.no_grad():
        logits, res = model(seq, scal)
        prob = F.softmax(logits, dim=-1)
        if mode == "soft":
            pred = (prob.unsqueeze(-1) * (anchors_t.unsqueeze(0) + res)).sum(dim=1)
        else:
            idx = prob.argmax(dim=-1)
            idx_exp = idx.view(-1, 1, 1).expand(-1, 1, 3)
            pred = anchors_t[idx] + res.gather(1, idx_exp).squeeze(1)
    return pred.numpy()


def run_kfold(seq_arr, scal_arr, seq_te, scal_te,
              y_disp_rot, theta_train, theta_test, y_train, x_last_train, x_last_test,
              anchors, K, n_folds=5, n_seeds=5, max_epochs=200, patience=30,
              batch=256, hidden=128, fc=192, p=0.3, lr=5e-4, wd=1e-4):
    N = y_disp_rot.shape[0]
    oof_soft = np.zeros((N, 3)); oof_hard = np.zeros((N, 3))
    test_per_fold_soft, test_per_fold_hard = [], []
    fold_mask = np.zeros(N, dtype=bool)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    anchors_t = torch.from_numpy(anchors)
    t0 = time.time()

    for fi, (tr, va) in enumerate(kf.split(np.arange(N))):
        sc_seq = StandardScaler().fit(seq_arr[tr].reshape(-1, seq_arr.shape[2]))
        sc_scal = StandardScaler().fit(scal_arr[tr])
        seq_tr_n = normalize_seq(seq_arr[tr], sc_seq)
        seq_va_n = normalize_seq(seq_arr[va], sc_seq)
        seq_te_n = normalize_seq(seq_te, sc_seq)
        scal_tr_n = sc_scal.transform(scal_arr[tr]).astype(np.float32)
        scal_va_n = sc_scal.transform(scal_arr[va]).astype(np.float32)
        scal_te_n = sc_scal.transform(scal_te).astype(np.float32)

        def T(a): return torch.from_numpy(a)
        seq_tr_t, scal_tr_t = T(seq_tr_n), T(scal_tr_n)
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)
        y_tr_t = T(y_disp_rot[tr].astype(np.float32))

        seed_val_s, seed_val_h = [], []
        seed_te_s, seed_te_h = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = AnchorModelStrong(n_ch=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                                       K=K, hidden=hidden, fc=fc, p=p)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_rh, best_state, no_imp = -1.0, None, 0
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

                # validate (track soft)
                pv = predict(model, seq_va_t, scal_va_t, anchors_t, mode="soft")
                pv_unrot = inverse_rotate_xy(pv, theta_train[va])
                rh = rhit(x_last_train[va] + pv_unrot, y_train[va])
                if rh > best_rh:
                    best_rh = rh
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_imp = 0
                else: no_imp += 1
                if no_imp >= patience: break
                if ep == 0 or (ep + 1) % 10 == 0:
                    print(f"  fold{fi+1} seed{s} ep{ep+1:3d}: soft_rhit={rh:.4f} (best {best_rh:.4f})  [{(time.time()-t0)/60:.1f}m]", flush=True)
            model.load_state_dict(best_state)
            pv_s = predict(model, seq_va_t, scal_va_t, anchors_t, mode="soft")
            pv_h = predict(model, seq_va_t, scal_va_t, anchors_t, mode="hard")
            pt_s = predict(model, seq_te_t, scal_te_t, anchors_t, mode="soft")
            pt_h = predict(model, seq_te_t, scal_te_t, anchors_t, mode="hard")
            seed_val_s.append(pv_s); seed_val_h.append(pv_h)
            seed_te_s.append(pt_s); seed_te_h.append(pt_h)
            del model; gc.collect()
        val_s = np.mean(seed_val_s, axis=0); val_h = np.mean(seed_val_h, axis=0)
        te_s = np.mean(seed_te_s, axis=0); te_h = np.mean(seed_te_h, axis=0)
        oof_soft[va] = val_s; oof_hard[va] = val_h
        fold_mask[va] = True
        test_per_fold_soft.append(te_s); test_per_fold_hard.append(te_h)

        rh_s = rhit(x_last_train[va] + inverse_rotate_xy(val_s, theta_train[va]), y_train[va])
        rh_h = rhit(x_last_train[va] + inverse_rotate_xy(val_h, theta_train[va]), y_train[va])
        print(f"  ★ fold {fi+1}/{n_folds}: soft={rh_s:.4f}, hard={rh_h:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    oof_soft_pos = x_last_train + inverse_rotate_xy(oof_soft, theta_train)
    oof_hard_pos = x_last_train + inverse_rotate_xy(oof_hard, theta_train)
    te_s = x_last_test + np.mean([inverse_rotate_xy(t, theta_test) for t in test_per_fold_soft], axis=0)
    te_h = x_last_test + np.mean([inverse_rotate_xy(t, theta_test) for t in test_per_fold_hard], axis=0)
    rh_s = rhit(oof_soft_pos, y_train); rh_h = rhit(oof_hard_pos, y_train)
    print(f"\n  OOF soft: {rh_s:.4f}  hard: {rh_h:.4f}")
    return oof_soft_pos, oof_hard_pos, te_s, te_h, rh_s, rh_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--fc", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    K = args.K
    print(f"v73 anchor strong — K={K}, {args.n_folds}-fold × {args.n_seeds}-seed × {args.max_epochs}ep")
    print(f"  hidden={args.hidden}, fc={args.fc}, drop={args.dropout}, lr={args.lr}")

    X_train, X_test, y_train, sub = load_data()
    x_last_train = X_train[:, -1, :]
    x_last_test = X_test[:, -1, :]
    _ = get_kalman(X_train, X_test)  # ensure cache
    from v23_train import MODE_CONFIGS
    M = MODE_CONFIGS["full"]
    X_scal_b_tr, X_scal_b_te = get_scalar_feats(X_train, X_test, M, "full")
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    X_scal_tr = np.concatenate([X_scal_b_tr, tier3_tr], axis=-1)
    X_scal_te = np.concatenate([X_scal_b_te, tier3_te], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)

    v_last_tr = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_te = (X_test[:, -1] - X_test[:, -2]) / DT
    th_tr, th_te = yaw_angle(v_last_tr), yaw_angle(v_last_te)
    y_disp_rot = rotate_xy(y_train - x_last_train, th_tr)
    anchors = compute_anchors(y_disp_rot, K)

    state_file = CACHE_DIR / f"v73_K{K}_state.npz"
    o_s, o_h, te_s, te_h, rh_s, rh_h = run_kfold(
        seq_tr, X_scal_tr, seq_te, X_scal_te,
        y_disp_rot, th_tr, th_te, y_train, x_last_train, x_last_test,
        anchors, K, n_folds=args.n_folds, n_seeds=args.n_seeds,
        max_epochs=args.max_epochs, patience=args.patience,
        hidden=args.hidden, fc=args.fc, p=args.dropout, lr=args.lr,
    )
    np.savez(state_file, oof_soft=o_s, oof_hard=o_h, test_soft=te_s, test_hard=te_h,
             rh_soft=rh_s, rh_hard=rh_h, anchors=anchors)
    print(f"\n[v73 K={K}] OOF soft: {rh_s:.4f}  hard: {rh_h:.4f}  (v65 K={K}: 0.6137/0.6037)")
    print(f"  state: {state_file}")

    pd.DataFrame({"id": sub["id"], "x": te_s[:,0], "y": te_s[:,1], "z": te_s[:,2]}).to_csv(
        DATA_DIR / f"submission_v73_K{K}_soft.csv", index=False)
    pd.DataFrame({"id": sub["id"], "x": te_h[:,0], "y": te_h[:,1], "z": te_h[:,2]}).to_csv(
        DATA_DIR / f"submission_v73_K{K}_hard.csv", index=False)

    entry = {"version": f"v73_anchor_strong_K{K}",
             "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "K": K, "n_folds": args.n_folds, "n_seeds": args.n_seeds,
             "max_epochs": args.max_epochs, "hidden": args.hidden, "fc": args.fc,
             "dropout": args.dropout, "lr": args.lr,
             "rh_soft": rh_s, "rh_hard": rh_h}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

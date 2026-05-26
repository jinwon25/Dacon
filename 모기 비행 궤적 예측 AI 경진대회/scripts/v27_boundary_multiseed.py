"""v27_boundary_multiseed.py — v26 boundary MLP × multi-seed averaging.

v26 (OOF 0.6561) + 3 seeds 평균 → +0.002~0.005 안정성 기대.
같은 architecture, 5-fold OOF, seed=0/1/2 학습 후 평균.

사용법:
  python scripts/v27_boundary_multiseed.py --v23-mode fast --n-seeds 3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"
DT = 0.040


def loss_combo(p, t, sample_w=None):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    if sample_w is not None:
        return ((d * sample_w).mean() + 0.3 * (sh * sample_w).mean()) / sample_w.mean()
    return d.mean() + 0.3 * sh.mean()


class BoundaryMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, p=0.2, cap_cm=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 3)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.cap = cap_cm / 100.0

    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z))
        return torch.tanh(self.head(z)) * self.cap


def build_features(X, kalman, v23_pred, v16_pred):
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    diff_v23_v16 = v23_pred - v16_pred
    dist_v23_v16 = np.linalg.norm(diff_v23_v16, axis=-1, keepdims=True)
    res_v23_kal = v23_pred - kalman
    return np.concatenate([
        v23_pred, v16_pred, diff_v23_v16,
        kalman, res_v23_kal,
        last_pos, v, a,
        v_mean, v_std,
        speed, a_norm, dist_v23_v16,
    ], axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, v23_tr, feat_test, test_v23, sample_w_tr,
                    args, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    device = torch.device("cpu")
    oof_pred = np.zeros_like(v23_tr)
    test_per_fold = []
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        feat_tr_n = sc.transform(feat_tr[tr]).astype(np.float32)
        feat_va_n = sc.transform(feat_tr[va]).astype(np.float32)
        feat_te_n = sc.transform(feat_test).astype(np.float32)

        def T(a): return torch.from_numpy(a).to(device)
        x_tr, x_va, x_te = T(feat_tr_n), T(feat_va_n), T(feat_te_n)
        v23_tr_t = T(v23_tr[tr].astype(np.float32))
        v23_va_t = T(v23_tr[va].astype(np.float32))
        v23_te_t = T(test_v23.astype(np.float32))
        y_tr_t = T(y_tr[tr].astype(np.float32))
        sw_t = T(sample_w_tr[tr])

        torch.manual_seed(seed); np.random.seed(seed)
        model = BoundaryMLP(in_dim=feat_tr_n.shape[1], hidden=64, p=0.2,
                              cap_cm=args.cap_cm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
        best_rh, best_state, no_improve = -1.0, None, 0
        for ep in range(args.max_epochs):
            model.train()
            perm = torch.randperm(x_tr.shape[0])
            for i in range(0, x_tr.shape[0], 256):
                idx = perm[i:i+256]
                opt.zero_grad()
                delta = model(x_tr[idx])
                pred = v23_tr_t[idx] + delta
                loss = loss_combo(pred, y_tr_t[idx], sample_w=sw_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                pred_va = (v23_va_t + model(x_va)).cpu().numpy()
            rh = float((np.linalg.norm(pred_va - y_tr[va], axis=-1) <= 0.01).mean())
            if rh > best_rh:
                best_rh = rh
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience: break

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_pred[va] = (v23_va_t + model(x_va)).cpu().numpy()
            test_per_fold.append((v23_te_t + model(x_te)).cpu().numpy())
        rh_fold = float((np.linalg.norm(oof_pred[va] - y_tr[va], axis=-1) <= 0.01).mean())
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)")
        del model; gc.collect()

    rh_seed = float((np.linalg.norm(oof_pred - y_tr, axis=-1) <= 0.01).mean())
    test_pred = np.mean(test_per_fold, axis=0)
    return oof_pred, test_pred, rh_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23-mode", default="fast")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cap-cm", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    args = parser.parse_args()
    mode = args.v23_mode

    print("=" * 60)
    print(f"v27 multi-seed boundary MLP ({args.n_seeds} seeds) | base = v23/{mode}")
    print("=" * 60)
    torch.set_num_threads(os.cpu_count() or 4)

    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]

    st = np.load(CACHE_DIR / f"v23_state_{mode}.npz")
    oof_A, test_A = st["oof_A"], st["test_A"]
    fold_mask_A = st["fold_mask_A"]
    has_B = bool(st.get("has_B", np.array(False)))
    if has_B:
        oof_B, test_B = st["oof_B"], st["test_B"]
        fold_mask_B = st["fold_mask_B"]
        eval_mask = fold_mask_A & fold_mask_B
        oof_res = (oof_A + oof_B) / 2
        test_res = (test_A + test_B) / 2
    else:
        eval_mask = fold_mask_A
        oof_res, test_res = oof_A.copy(), test_A.copy()
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    oof_v23  = kalman_train + oof_res  * ALPHA
    test_v23 = kalman_test  + test_res * ALPHA

    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)

    # boundary mask
    d_v23 = np.linalg.norm(oof_v23 - y_train, axis=-1)
    boundary_mask = (d_v23 > 0.005) & (d_v23 <= 0.03) & eval_mask

    eval_idx = np.where(eval_mask)[0]
    feat_train_full = build_features(X_train, kalman_train, oof_v23, oof_v16)
    feat_test_full  = build_features(X_test, kalman_test, test_v23, test_v16)
    feat_tr = feat_train_full[eval_idx]
    y_tr = y_train[eval_idx]
    v23_tr = oof_v23[eval_idx]
    sample_w_tr = np.ones(len(eval_idx), dtype=np.float32)
    sample_w_tr[np.where(boundary_mask[eval_idx])[0]] = args.boundary_weight

    # Multi-seed
    oofs, tests, rh_seeds = [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} ===")
        oof_s, test_s, rh_s = train_one_seed(s, feat_tr, y_tr, v23_tr,
                                                feat_test_full, test_v23, sample_w_tr, args)
        oofs.append(oof_s); tests.append(test_s); rh_seeds.append(rh_s)
        print(f"  seed{s} OOF R-Hit: {rh_s:.4f}")

    # Averaged predictions
    oof_avg = np.mean(oofs, axis=0)
    test_avg = np.mean(tests, axis=0)
    rh_avg = float((np.linalg.norm(oof_avg - y_tr, axis=-1) <= 0.01).mean())

    rh_v23_full = float((np.linalg.norm(v23_tr - y_tr, axis=-1) <= 0.01).mean())

    print("\n" + "=" * 60)
    print(f"=== v27 결과 (seed-averaged, {args.n_seeds} seeds) ===")
    print("=" * 60)
    print(f"  v23 alone     : {rh_v23_full:.4f}")
    for i, rh in enumerate(rh_seeds):
        print(f"  v27 seed{i}     : {rh:.4f}")
    print(f"  v27 avg       : {rh_avg:.4f}  (Δ vs v23 {rh_avg - rh_v23_full:+.4f}, "
          f"Δ vs best-seed {rh_avg - max(rh_seeds):+.4f})")

    # Submission
    assert test_avg.shape == (10000, 3) and np.isfinite(test_avg).all()
    out_csv = DATA_DIR / f"submission_v27_cpu_{mode}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_avg[:,0], "y": test_avg[:,1], "z": test_avg[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n[submission] {out_csv}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": f"v27_cpu_{mode}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"v26 boundary MLP × {args.n_seeds}-seed averaging",
        "v23_mode": mode,
        "v23_oof_rhit": rh_v23_full,
        "v27_oof_rhit_per_seed": rh_seeds,
        "v27_oof_rhit_avg": rh_avg,
        "delta_vs_v23": rh_avg - rh_v23_full,
        "n_seeds": args.n_seeds,
        "cap_cm": args.cap_cm,
        "boundary_weight": args.boundary_weight,
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

    print(f"\n  CSV: {out_csv}")


if __name__ == "__main__":
    main()

"""v46_soft_stacker.py — Per-sample softmax-weighted ensemble over OOF models.

기존 v45 grid는 global weight. Oracle 3way (v35|v44|gate) = 0.6988 → 모델 disagreement에
sample-wise 시그널 0.025+ 잠재. 이를 soft weight MLP로 학습한다.

설계:
- Models (7): v30_A, v30_B, v35, v41_A, v41_B, v44, gate
- Features (~40): last_pos, v, a, speed, pairwise model distances, residuals vs Kalman
- MLP 41→64→32→7 → softmax → output = Σ w_i × pred_i
- combo loss (euclid + 0.3 softhit), 5-fold OOF (random_state=0), 3-seed avg

Why softmax (not residual): convex combination of model preds → oracle bound 0.6988,
over-fit 위험 작음. v35 alone 0.6725 + alpha 시그널 흡수.
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
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"

BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rhit(p, y):
    return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def loss_combo(pred, target):
    d = torch.sqrt(((pred - target) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    return d.mean() + 0.3 * sh.mean()


class SoftStacker(nn.Module):
    """입력 feat → 7 모델에 대한 softmax weight."""

    def __init__(self, in_dim, n_models, hidden=64, p=0.3, temp=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, n_models)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.temp = temp

    def forward(self, x, preds):
        """x: (B, F) features; preds: (B, M, 3) — return (B, 3) weighted."""
        z = self.act(self.fc1(x))
        z = self.drop(z)
        z = self.act(self.fc2(z))
        logits = self.head(z) / self.temp
        w = F.softmax(logits, dim=-1)  # (B, M)
        out = (w.unsqueeze(-1) * preds).sum(dim=1)  # (B, 3)
        return out, w


def build_features(X, kalman, model_preds: list):
    """
    X: (N, 11, 3) input traj
    kalman: (N, 3) Kalman CV +80ms prediction
    model_preds: list of (N, 3) OOF preds (M models)
    """
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1)
    v_std = v_recent.std(axis=1)

    # residuals vs Kalman base (each model relative to Kalman)
    feats = [last_pos, v, a, v_mean, v_std, speed, a_norm, kalman]
    for p in model_preds:
        feats.append(p - kalman)  # 3-dim residual per model
    # pairwise disagreement (L2 distance between selected key models)
    n = len(model_preds)
    for i in range(n):
        for j in range(i+1, n):
            d_ij = np.linalg.norm(model_preds[i] - model_preds[j], axis=-1, keepdims=True)
            feats.append(d_ij)
    return np.concatenate(feats, axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, preds_tr, feat_te, preds_te, args, kf):
    device = torch.device("cpu")
    N = feat_tr.shape[0]
    M = preds_tr.shape[1]
    oof_pred = np.zeros((N, 3), dtype=np.float64)
    test_per_fold = []
    weights_oof = np.zeros((N, M), dtype=np.float32)
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        ftn_tr = sc.transform(feat_tr[tr]).astype(np.float32)
        ftn_va = sc.transform(feat_tr[va]).astype(np.float32)
        ftn_te = sc.transform(feat_te).astype(np.float32)

        def T(a): return torch.from_numpy(a).to(device)
        x_tr = T(ftn_tr); x_va = T(ftn_va); x_te = T(ftn_te)
        p_tr = T(preds_tr[tr].astype(np.float32))
        p_va = T(preds_tr[va].astype(np.float32))
        p_te = T(preds_te.astype(np.float32))
        y_tr_t = T(y_tr[tr].astype(np.float32))

        torch.manual_seed(seed); np.random.seed(seed)
        model = SoftStacker(in_dim=ftn_tr.shape[1], n_models=M,
                            hidden=args.hidden, p=args.dropout, temp=args.temp).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

        best_rh, best_state, no_improve = -1.0, None, 0
        n_tr = ftn_tr.shape[0]
        for ep in range(args.max_epochs):
            model.train()
            perm = torch.randperm(n_tr)
            for i in range(0, n_tr, 512):
                idx = perm[i:i+512]
                opt.zero_grad()
                pred, _ = model(x_tr[idx], p_tr[idx])
                loss = loss_combo(pred, y_tr_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pred_va, _ = model(x_va, p_va)
                pred_va_np = pred_va.cpu().numpy()
            rh = rhit(pred_va_np, y_tr[va])
            if rh > best_rh:
                best_rh = rh
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pv, wv = model(x_va, p_va)
            oof_pred[va] = pv.cpu().numpy()
            weights_oof[va] = wv.cpu().numpy()
            pt, _ = model(x_te, p_te)
            test_per_fold.append(pt.cpu().numpy())
        rh_fold = rhit(oof_pred[va], y_tr[va])
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f}  best_ep_rh={best_rh:.4f}  ({(time.time()-t0)/60:.1f}m)")
        del model; gc.collect()
    return oof_pred, np.mean(test_per_fold, axis=0), weights_oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v46 SoftStacker — {args.n_seeds} seeds × {args.n_folds}-fold")
    print("=" * 60)

    # Load core data
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y_train = labels.set_index("id").loc[list(train_ids)][["x","y","z"]].values.astype(np.float64)

    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]

    # Models (7): v30_A, v30_B, v35, v41_A, v41_B, v44, gate
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    v30A_oof = kalman_train + st30["oof_A"] * ALPHA
    v30B_oof = kalman_train + st30["oof_B"] * ALPHA
    v30A_te  = kalman_test  + st30["test_A"] * ALPHA
    v30B_te  = kalman_test  + st30["test_B"] * ALPHA

    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v35_oof, v35_te = st35["oof_v35"].astype(np.float64), st35["test_v35"].astype(np.float64)

    st41 = np.load(CACHE_DIR / "v41_state.npz")
    v41A_oof = kalman_train + st41["oof_A"] * ALPHA
    v41B_oof = kalman_train + st41["oof_B"] * ALPHA
    v41A_te  = kalman_test  + st41["test_A"] * ALPHA
    v41B_te  = kalman_test  + st41["test_B"] * ALPHA

    st44 = np.load(CACHE_DIR / "v44_state.npz")
    v44_oof, v44_te = st44["oof_v44"].astype(np.float64), st44["test_v44"].astype(np.float64)

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    gate_te = df_best[["x","y","z"]].values.astype(np.float64)

    # baseline R-Hits
    models_train = [v30A_oof, v30B_oof, v35_oof, v41A_oof, v41B_oof, v44_oof, gate_oof]
    models_test  = [v30A_te,  v30B_te,  v35_te,  v41A_te,  v41B_te,  v44_te,  gate_te]
    names = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate"]
    print("\n=== Model OOF R-Hit ===")
    for n, p in zip(names, models_train):
        print(f"  {n}: {rhit(p, y_train):.4f}")

    # Oracle (any model hit @1cm)
    hits = np.stack([np.linalg.norm(p - y_train, axis=-1) <= 0.01 for p in models_train])
    oracle = hits.any(axis=0).mean()
    print(f"\n  Oracle (any of 7): {oracle:.4f}")

    # Features
    feat_tr = build_features(X_train, kalman_train, models_train)
    feat_te = build_features(X_test,  kalman_test,  models_test)
    print(f"\n  feat dim: {feat_tr.shape[1]}")

    # stack preds: (N, M, 3)
    preds_tr = np.stack(models_train, axis=1).astype(np.float64)
    preds_te = np.stack(models_test,  axis=1).astype(np.float64)
    print(f"  preds_tr: {preds_tr.shape}")

    # Multi-seed training
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests, all_w = [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} ===")
        oof_s, test_s, w_s = train_one_seed(s, feat_tr, y_train, preds_tr,
                                            feat_te, preds_te, args, kf)
        oofs.append(oof_s); tests.append(test_s); all_w.append(w_s)
        rh_s = rhit(oof_s, y_train)
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v46 = np.mean(oofs, axis=0)
    test_v46 = np.mean(tests, axis=0)
    w_mean = np.mean(all_w, axis=0)
    rh_v46 = rhit(oof_v46, y_train)

    print(f"\n=== v46 SoftStacker 결과 ===")
    print(f"  v35 alone OOF: {rhit(v35_oof, y_train):.4f}")
    print(f"  v46 OOF: {rh_v46:.4f}  (Δ vs v35: {rh_v46 - rhit(v35_oof, y_train):+.4f})")
    print(f"  weight avg over OOF (per model):")
    for n, wi in zip(names, w_mean.mean(axis=0)):
        print(f"    {n}: {wi:.3f}")

    # Hybrid with v35 alone (보수적)
    print(f"\n=== Hybrid with v35 alone ===")
    for alpha in np.linspace(0.0, 1.0, 11):
        ens = alpha * oof_v46 + (1 - alpha) * v35_oof
        r = rhit(ens, y_train)
        print(f"  v46×{alpha:.1f} + v35×{1-alpha:.1f}: {r:.4f}")

    # Save state
    np.savez(CACHE_DIR / "v46_state.npz",
             oof_v46=oof_v46, test_v46=test_v46, rh_v46=rh_v46,
             weights_oof_mean=w_mean)

    # Submissions
    out_csv = DATA_DIR / "submission_v46_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v46[:,0], "y": test_v46[:,1], "z": test_v46[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n  [submission] {out_csv}")

    # Best hybrid submission (search best alpha on OOF)
    best_alpha, best_r = 1.0, rh_v46
    for alpha in np.linspace(0.0, 1.0, 21):
        ens = alpha * oof_v46 + (1 - alpha) * v35_oof
        r = rhit(ens, y_train)
        if r > best_r:
            best_r, best_alpha = r, alpha
    hyb_test = best_alpha * test_v46 + (1 - best_alpha) * v35_te
    hyb_csv = DATA_DIR / f"submission_v46_hybrid_v46x{best_alpha:.2f}_v35x{1-best_alpha:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": hyb_test[:,0], "y": hyb_test[:,1], "z": hyb_test[:,2]}
                 ).to_csv(hyb_csv, index=False)
    print(f"  [hybrid] α={best_alpha:.2f} OOF={best_r:.4f} → {hyb_csv.name}")

    # Log
    entry = {
        "version": "v46_soft_stacker",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Per-sample softmax weighted ensemble over 7 model OOF preds",
        "n_models": len(names),
        "model_names": names,
        "n_seeds": args.n_seeds,
        "n_folds": args.n_folds,
        "feat_dim": int(feat_tr.shape[1]),
        "model_oof": {n: rhit(p, y_train) for n, p in zip(names, models_train)},
        "oracle_any7": float(oracle),
        "v46_oof": float(rh_v46),
        "v35_alone_oof": float(rhit(v35_oof, y_train)),
        "delta_vs_v35": float(rh_v46 - rhit(v35_oof, y_train)),
        "weight_avg": {n: float(wi) for n, wi in zip(names, w_mean.mean(axis=0))},
        "hybrid_best_alpha": float(best_alpha),
        "hybrid_best_oof": float(best_r),
        "submission": str(out_csv),
        "hybrid_submission": str(hyb_csv),
    }
    log_path = PROJECT_DIR / "run_log.json"
    log = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    log.append(entry)
    json.dump(log, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  [run_log] appended v46_soft_stacker")


if __name__ == "__main__":
    main()

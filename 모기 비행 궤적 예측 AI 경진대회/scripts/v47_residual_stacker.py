"""v47_residual_stacker.py — v46 SoftStacker + capped residual MLP.

v46 alone OOF 0.6734 (+0.0009 vs v35). Oracle 7-way 0.7211 갭 480bp.
soft selector만으로는 sample-wise routing 신호를 다 못 잡음.
→ residual head 추가 (capped tanh × 0.5cm). v35 boundary와 동일 패턴 적용.

설계:
- shared trunk: feat → 64 → 32 (GELU + dropout 0.3)
- soft head: 7 logits → softmax → Σ w_i × pred_i = base
- residual head: 3 channels → tanh × cap_cm
- output = base + residual
- loss: combo (euclid + 0.3 softhit)
- boundary sample weight 2.0 (d_base > 0.5cm and ≤ 3cm)
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
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"

BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rhit(p, y):
    return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def loss_combo(pred, target, sw=None):
    d = torch.sqrt(((pred - target) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    if sw is None:
        return d.mean() + 0.3 * sh.mean()
    return ((d * sw).mean() + 0.3 * (sh * sw).mean()) / sw.mean()


class ResidualStacker(nn.Module):
    """Soft selector + capped residual."""

    def __init__(self, in_dim, n_models, hidden=64, p=0.3, cap_cm=0.5, temp=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head_w = nn.Linear(hidden // 2, n_models)
        self.head_r = nn.Linear(hidden // 2, 3)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.temp = temp
        self.cap = cap_cm / 100.0

    def forward(self, x, preds):
        z = self.act(self.fc1(x))
        z = self.drop(z)
        z = self.act(self.fc2(z))
        logits = self.head_w(z) / self.temp
        w = F.softmax(logits, dim=-1)
        base = (w.unsqueeze(-1) * preds).sum(dim=1)
        residual = torch.tanh(self.head_r(z)) * self.cap
        return base + residual, w, residual


def build_features(X, kalman, model_preds: list):
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1)
    v_std = v_recent.std(axis=1)
    feats = [last_pos, v, a, v_mean, v_std, speed, a_norm, kalman]
    for p in model_preds:
        feats.append(p - kalman)
    n = len(model_preds)
    for i in range(n):
        for j in range(i+1, n):
            d_ij = np.linalg.norm(model_preds[i] - model_preds[j], axis=-1, keepdims=True)
            feats.append(d_ij)
    return np.concatenate(feats, axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, preds_tr, feat_te, preds_te,
                   sample_w_tr, args, kf):
    device = torch.device("cpu")
    N = feat_tr.shape[0]
    M = preds_tr.shape[1]
    oof_pred = np.zeros((N, 3), dtype=np.float64)
    test_per_fold = []
    weights_oof = np.zeros((N, M), dtype=np.float32)
    residual_oof = np.zeros((N, 3), dtype=np.float32)
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
        sw_t = T(sample_w_tr[tr])

        torch.manual_seed(seed); np.random.seed(seed)
        model = ResidualStacker(in_dim=ftn_tr.shape[1], n_models=M,
                                hidden=args.hidden, p=args.dropout,
                                cap_cm=args.cap_cm, temp=args.temp).to(device)
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
                pred, _, _ = model(x_tr[idx], p_tr[idx])
                loss = loss_combo(pred, y_tr_t[idx], sw=sw_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pred_va, _, _ = model(x_va, p_va)
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
            pv, wv, rv = model(x_va, p_va)
            oof_pred[va] = pv.cpu().numpy()
            weights_oof[va] = wv.cpu().numpy()
            residual_oof[va] = rv.cpu().numpy()
            pt, _, _ = model(x_te, p_te)
            test_per_fold.append(pt.cpu().numpy())
        rh_fold = rhit(oof_pred[va], y_tr[va])
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)")
        del model; gc.collect()
    return oof_pred, np.mean(test_per_fold, axis=0), weights_oof, residual_oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cap-cm", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v47 ResidualStacker — {args.n_seeds} seeds × {args.n_folds}-fold (cap {args.cap_cm}cm)")
    print("=" * 60)

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

    models_train = [v30A_oof, v30B_oof, v35_oof, v41A_oof, v41B_oof, v44_oof, gate_oof]
    models_test  = [v30A_te,  v30B_te,  v35_te,  v41A_te,  v41B_te,  v44_te,  gate_te]
    names = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate"]

    print("\n=== Model OOF R-Hit ===")
    for n, p in zip(names, models_train):
        print(f"  {n}: {rhit(p, y_train):.4f}")

    # boundary mask (on best base = v35)
    d_v35 = np.linalg.norm(v35_oof - y_train, axis=-1)
    boundary_mask = (d_v35 > 0.005) & (d_v35 <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"\n  boundary samples (0.5cm<d<=3cm vs v35): {boundary_mask.sum()} ({args.boundary_weight}× wt)")

    feat_tr = build_features(X_train, kalman_train, models_train)
    feat_te = build_features(X_test,  kalman_test,  models_test)
    preds_tr = np.stack(models_train, axis=1).astype(np.float64)
    preds_te = np.stack(models_test,  axis=1).astype(np.float64)
    print(f"  feat dim: {feat_tr.shape[1]}, preds: {preds_tr.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests, all_w, all_r = [], [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} ===")
        oof_s, test_s, w_s, r_s = train_one_seed(s, feat_tr, y_train, preds_tr,
                                                  feat_te, preds_te, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s); all_w.append(w_s); all_r.append(r_s)
        rh_s = rhit(oof_s, y_train)
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v47 = np.mean(oofs, axis=0)
    test_v47 = np.mean(tests, axis=0)
    w_mean = np.mean(all_w, axis=0)
    r_mean = np.mean(all_r, axis=0)
    rh_v47 = rhit(oof_v47, y_train)

    # residual stats
    r_norm = np.linalg.norm(r_mean, axis=-1)
    print(f"\n  Residual magnitude: mean={r_norm.mean()*1000:.2f}mm, "
          f"median={np.median(r_norm)*1000:.2f}mm, p95={np.percentile(r_norm, 95)*1000:.2f}mm")

    print(f"\n=== v47 ResidualStacker 결과 ===")
    print(f"  v35 alone OOF: {rhit(v35_oof, y_train):.4f}")
    print(f"  v46 OOF      : 0.6734")
    print(f"  v47 OOF      : {rh_v47:.4f}  (Δ vs v35: {rh_v47 - rhit(v35_oof, y_train):+.4f})")
    print(f"  weight avg over OOF (per model):")
    for n, wi in zip(names, w_mean.mean(axis=0)):
        print(f"    {n}: {wi:.3f}")

    # Hybrid sweep with v35 and v46
    print(f"\n=== Hybrid with v35 alone ===")
    best_alpha35, best_r35 = 1.0, rh_v47
    for alpha in np.linspace(0.0, 1.0, 21):
        ens = alpha * oof_v47 + (1 - alpha) * v35_oof
        r = rhit(ens, y_train)
        if r > best_r35: best_r35, best_alpha35 = r, alpha
    print(f"  best: v47×{best_alpha35:.2f} + v35×{1-best_alpha35:.2f}: OOF {best_r35:.4f}")

    # Save
    np.savez(CACHE_DIR / "v47_state.npz",
             oof_v47=oof_v47, test_v47=test_v47, rh_v47=rh_v47,
             weights_oof_mean=w_mean, residual_oof_mean=r_mean)

    out_csv = DATA_DIR / "submission_v47_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v47[:,0], "y": test_v47[:,1], "z": test_v47[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n  [submission] {out_csv}")

    if best_alpha35 < 1.0:
        hyb_test = best_alpha35 * test_v47 + (1 - best_alpha35) * v35_te
        hyb_csv = DATA_DIR / f"submission_v47_hybrid_v47x{best_alpha35:.2f}_v35x{1-best_alpha35:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb_test[:,0], "y": hyb_test[:,1], "z": hyb_test[:,2]}
                     ).to_csv(hyb_csv, index=False)
        print(f"  [hybrid] α={best_alpha35:.2f} OOF={best_r35:.4f} → {hyb_csv.name}")

    entry = {
        "version": "v47_residual_stacker",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "SoftStacker + capped residual MLP (0.5cm cap)",
        "n_models": len(names),
        "model_names": names,
        "cap_cm": args.cap_cm,
        "boundary_weight": args.boundary_weight,
        "v47_oof": float(rh_v47),
        "v35_alone_oof": float(rhit(v35_oof, y_train)),
        "v46_alone_oof": 0.6734,
        "delta_vs_v35": float(rh_v47 - rhit(v35_oof, y_train)),
        "weight_avg": {n: float(wi) for n, wi in zip(names, w_mean.mean(axis=0))},
        "residual_mean_mm": float(r_norm.mean()*1000),
        "hybrid_best_alpha_v47": float(best_alpha35),
        "hybrid_best_oof": float(best_r35),
        "submission": str(out_csv),
    }
    log_path = PROJECT_DIR / "run_log.json"
    log = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    log.append(entry)
    json.dump(log, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  [run_log] appended v47_residual_stacker")


if __name__ == "__main__":
    main()

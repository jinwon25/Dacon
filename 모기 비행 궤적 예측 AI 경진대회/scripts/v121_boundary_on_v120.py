"""v121_boundary_on_v120.py — Boundary MLP on v120 Neural ODE.

v120 OOF 0.6610 (5-fold mirror). v91/v94 패턴 (base + small δ) 적용 시
~+0.01 lift 기대 → 0.670+ 단독 가능. Pool 멤버 자격.

v111_boundary_on_v109.py 패턴 그대로 사용, base만 v120_full로 변경.
"""
from __future__ import annotations
import argparse, datetime as _dt, gc, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"
BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def loss_combo(p, t, sw=None):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    if sw is None:
        return d.mean() + 0.3 * sh.mean()
    return ((d * sw).mean() + 0.3 * (sh * sw).mean()) / sw.mean()


class BoundaryMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, p=0.2, cap_cm=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 3)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.cap = cap_cm / 100.0

    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z))
        return torch.tanh(self.head(z)) * self.cap


def build_features(X, kalman, base_pred, gate_pred, v16_pred):
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    diff_bg = base_pred - gate_pred
    diff_bv = base_pred - v16_pred
    dist_bg = np.linalg.norm(diff_bg, axis=-1, keepdims=True)
    res_b_kal = base_pred - kalman
    return np.concatenate([
        base_pred, gate_pred, v16_pred,
        diff_bg, diff_bv, dist_bg,
        kalman, res_b_kal,
        last_pos, v, a,
        v_mean, v_std,
        speed, a_norm,
    ], axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, base_tr, feat_test, test_base, sample_w_tr,
                    args, kf):
    device = torch.device("cpu")
    oof_pred = np.zeros_like(base_tr)
    test_per_fold = []
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        feat_tr_n = sc.transform(feat_tr[tr]).astype(np.float32)
        feat_va_n = sc.transform(feat_tr[va]).astype(np.float32)
        feat_te_n = sc.transform(feat_test).astype(np.float32)
        def T(a): return torch.from_numpy(a).to(device)
        x_tr, x_va, x_te = T(feat_tr_n), T(feat_va_n), T(feat_te_n)
        base_tr_t = T(base_tr[tr].astype(np.float32))
        base_va_t = T(base_tr[va].astype(np.float32))
        base_te_t = T(test_base.astype(np.float32))
        y_tr_t = T(y_tr[tr].astype(np.float32))
        sw_t = T(sample_w_tr[tr])

        torch.manual_seed(seed); np.random.seed(seed)
        model = BoundaryMLP(in_dim=feat_tr_n.shape[1], hidden=args.hidden, p=0.2,
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
                pred = base_tr_t[idx] + model(x_tr[idx])
                loss = loss_combo(pred, y_tr_t[idx], sw=sw_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pred_va = (base_va_t + model(x_va)).cpu().numpy()
            rh = float((np.linalg.norm(pred_va - y_tr[va], axis=-1) <= 0.01).mean())
            if rh > best_rh:
                best_rh = rh
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else: no_improve += 1
            if no_improve >= args.patience: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_pred[va] = (base_va_t + model(x_va)).cpu().numpy()
            test_per_fold.append((base_te_t + model(x_te)).cpu().numpy())
        rh_fold = float((np.linalg.norm(oof_pred[va] - y_tr[va], axis=-1) <= 0.01).mean())
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)",
              flush=True)
        del model; gc.collect()
    return oof_pred, np.mean(test_per_fold, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cap-cm", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--v120-tag", default="full", help="v120 state suffix (full/fast/smoke)")
    parser.add_argument("--out-tag", default="v121", help="output tag prefix")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    cap_lbl = f"{int(round(args.cap_cm*10)):02d}"
    tag = f"{args.out_tag}_cap{cap_lbl}"
    print("=" * 60)
    print(f"v121 boundary MLP on v120_{args.v120_tag} (cap {args.cap_cm}cm)")
    print(f"  out tag = {tag}")
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

    # v120 base
    v120 = np.load(CACHE_DIR / f"v120_{args.v120_tag}_state.npz", allow_pickle=True)
    oof_base = v120["oof_global"].astype(np.float64)
    test_base = v120["test_global"].astype(np.float64)
    fold_mask = v120["fold_mask"]
    if not fold_mask.all():
        # smoke/fast: only partial OOF; can't run 5-fold boundary
        print(f"[WARN] v120 only covers {fold_mask.sum()}/{len(fold_mask)} samples — boundary requires full")
    rh_base = float((np.linalg.norm(oof_base - y_train, axis=-1) <= 0.01).mean())
    print(f"v120 base OOF: {rh_base:.4f}")

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    test_gate = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64); test_v16 = st16["test"].astype(np.float64)

    d_base = np.linalg.norm(oof_base - y_train, axis=-1)
    boundary_mask = (d_base > 0.005) & (d_base <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"boundary samples (0.5cm < d ≤ 3cm): {boundary_mask.sum()}")

    feat_train = build_features(X_train, kalman_train, oof_base, gate_oof, oof_v16)
    feat_test = build_features(X_test, kalman_test, test_base, test_gate, test_v16)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} (cap={args.cap_cm}cm) ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, oof_base,
                                        feat_test, test_base, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_pred = np.mean(oofs, axis=0)
    test_pred = np.mean(tests, axis=0)
    rh = float((np.linalg.norm(oof_pred - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== {tag} ===")
    print(f"  base OOF: {rh_base:.4f}")
    print(f"  {tag}: {rh:.4f}  (Δ vs base: {rh - rh_base:+.4f})")

    np.savez(CACHE_DIR / f"{tag}_state.npz",
              oof_v91=oof_pred, test_v91=test_pred, rh_v91=rh,
              n_seeds=args.n_seeds, cap_cm=args.cap_cm)
    out_csv = DATA_DIR / f"submission_{tag}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pred[:,0], "y": test_pred[:,1], "z": test_pred[:,2]}
                  ).to_csv(out_csv, index=False)
    print(f"  [state] {tag}_state.npz")
    print(f"  [submission] {out_csv.name}")


if __name__ == "__main__":
    main()

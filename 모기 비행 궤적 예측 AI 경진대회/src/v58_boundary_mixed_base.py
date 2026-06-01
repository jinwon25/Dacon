"""v58_boundary_mixed_base.py — Boundary MLP on (v30 + v41) avg base.

v35 (v30 GRU base + boundary)  OOF 0.6725
v44 (v41 Trans base + boundary) OOF 0.6713
v58 (mixed v30+v41 avg base + boundary): paradigm 합성 가능성

base = (v30 + v41) / 2 → R-Hit measure 먼저 확인, boundary 학습 후 평가.
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


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
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


def build_features(X, kalman, base_pred, gate_pred, v16_pred, v30_pred, v41_pred):
    """v35/v44 features + mixed-base specific: v30 vs v41 disagreement."""
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    diff_vg = base_pred - gate_pred
    diff_vv = base_pred - v16_pred
    dist_vg = np.linalg.norm(diff_vg, axis=-1, keepdims=True)
    res_base_kal = base_pred - kalman
    # mixed-base specific: v30 vs v41 disagreement
    diff_v30_v41 = v30_pred - v41_pred
    dist_v30_v41 = np.linalg.norm(diff_v30_v41, axis=-1, keepdims=True)
    return np.concatenate([
        base_pred, gate_pred, v16_pred, v30_pred, v41_pred,
        diff_vg, diff_vv, dist_vg,
        diff_v30_v41, dist_v30_v41,
        kalman, res_base_kal,
        last_pos, v, a, v_mean, v_std,
        speed, a_norm,
    ], axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, base_tr, feat_test, test_base, sample_w_tr, args, kf):
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
            else:
                no_improve += 1
            if no_improve >= args.patience: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_pred[va] = (base_va_t + model(x_va)).cpu().numpy()
            test_per_fold.append((base_te_t + model(x_te)).cpu().numpy())
        rh_fold = float((np.linalg.norm(oof_pred[va] - y_tr[va], axis=-1) <= 0.01).mean())
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f} ({(time.time()-t0)/60:.1f}m)")
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
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v58 = mixed (v30+v41)/2 base + boundary MLP ({args.n_seeds} seeds, cap {args.cap_cm}cm)")
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
    v30_oof = kalman_train + (st30["oof_A"] + st30["oof_B"])/2 * ALPHA
    v30_te  = kalman_test  + (st30["test_A"] + st30["test_B"])/2 * ALPHA

    st41 = np.load(CACHE_DIR / "v41_state.npz")
    v41_oof = kalman_train + (st41["oof_A"] + st41["oof_B"])/2 * ALPHA
    v41_te  = kalman_test  + (st41["test_A"] + st41["test_B"])/2 * ALPHA

    # mixed base
    base_oof = (v30_oof + v41_oof) / 2
    base_te  = (v30_te  + v41_te) / 2
    rh_base = float((np.linalg.norm(base_oof - y_train, axis=-1) <= 0.01).mean())
    rh_v30 = float((np.linalg.norm(v30_oof - y_train, axis=-1) <= 0.01).mean())
    rh_v41 = float((np.linalg.norm(v41_oof - y_train, axis=-1) <= 0.01).mean())
    print(f"  v30 base OOF: {rh_v30:.4f}")
    print(f"  v41 base OOF: {rh_v41:.4f}")
    print(f"  mixed (v30+v41)/2 base OOF: {rh_base:.4f}")

    # gate
    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    test_gate = df_best[["x","y","z"]].values.astype(np.float64)

    # v16
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64); test_v16 = st16["test"].astype(np.float64)

    # boundary mask
    d_base = np.linalg.norm(base_oof - y_train, axis=-1)
    boundary_mask = (d_base > 0.005) & (d_base <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"  boundary samples: {boundary_mask.sum()} ({args.boundary_weight}× wt)")

    feat_train = build_features(X_train, kalman_train, base_oof, gate_oof, oof_v16, v30_oof, v41_oof)
    feat_test  = build_features(X_test, kalman_test, base_te, test_gate, test_v16, v30_te, v41_te)
    print(f"  feat dim: {feat_train.shape[1]}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, base_oof,
                                          feat_test, base_te, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v58 = np.mean(oofs, axis=0)
    test_v58 = np.mean(tests, axis=0)
    rh_v58 = float((np.linalg.norm(oof_v58 - y_train, axis=-1) <= 0.01).mean())

    st35 = np.load(CACHE_DIR / "v35_state.npz")
    rh_v35 = float((np.linalg.norm(st35["oof_v35"] - y_train, axis=-1) <= 0.01).mean())
    st44 = np.load(CACHE_DIR / "v44_state.npz")
    rh_v44 = float((np.linalg.norm(st44["oof_v44"] - y_train, axis=-1) <= 0.01).mean())

    print(f"\n=== v58 결과 ===")
    print(f"  v35 (v30 base) : {rh_v35:.4f}")
    print(f"  v44 (v41 base) : {rh_v44:.4f}")
    print(f"  mixed base alone: {rh_base:.4f}")
    print(f"  v58 (mixed + boundary): {rh_v58:.4f}  (Δ vs v35: {rh_v58-rh_v35:+.4f}, vs v44: {rh_v58-rh_v44:+.4f})")

    # Δ stats
    delta = oof_v58 - base_oof
    delta_norm = np.linalg.norm(delta, axis=-1)
    print(f"  Δ stats: mean={delta_norm.mean()*1000:.2f}mm, median={np.median(delta_norm)*1000:.2f}mm, "
          f"p99={np.percentile(delta_norm, 99)*1000:.2f}mm")

    np.savez(CACHE_DIR / "v58_state.npz",
             oof_v58=oof_v58, test_v58=test_v58, rh_v58=rh_v58)

    out_csv = DATA_DIR / "submission_v58_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v58[:,0], "y": test_v58[:,1], "z": test_v58[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    # Hybrid sweep
    v35_oof = st35["oof_v35"].astype(np.float64); v35_te = st35["test_v35"].astype(np.float64)
    print(f"\n=== Hybrid ===")
    best_a, best_r = 1.0, rh_v58
    for alpha in np.linspace(0, 1, 21):
        ens = alpha*oof_v58 + (1-alpha)*v35_oof
        r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
        if r > best_r: best_r, best_a = r, alpha
    print(f"  v58 × v35: α={best_a:.2f} OOF={best_r:.4f}")

    # v58 + v48_3way reconstructed
    try:
        st48 = np.load(CACHE_DIR / "v48_state.npz")
        st46 = np.load(CACHE_DIR / "v46_state.npz")
        oof_v48_3way = 0.70*st48["oof_v48"] + 0.12*st46["oof_v46"] + 0.18*v35_oof
        test_v48_3way = 0.70*st48["test_v48"] + 0.12*st46["test_v46"] + 0.18*v35_te
        best_2 = (1.0, 0.0); best_2r = rh_v58
        for a in np.linspace(0, 1, 21):
            ens = a*oof_v58 + (1-a)*oof_v48_3way
            r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
            if r > best_2r: best_2r, best_2 = r, (a, 1-a)
        print(f"  v58 × v48_3way: {best_2} OOF={best_2r:.4f}")
        if best_2r > rh_v58:
            a, b = best_2
            hyb = a*test_v58 + b*test_v48_3way
            pd.DataFrame({"id": sub["id"], "x": hyb[:,0], "y": hyb[:,1], "z": hyb[:,2]}
                         ).to_csv(DATA_DIR / f"submission_v58_2way_v58x{a:.2f}_v48_3wayx{b:.2f}.csv", index=False)
            print(f"  [2way] saved")
    except Exception as e:
        print(f"  hybrid err: {e}")
        best_2, best_2r = (1.0, 0.0), rh_v58

    entry = {
        "version": "v58_boundary_mixed_base",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Boundary MLP on (v30+v41)/2 mixed base",
        "base_oof": float(rh_base),
        "v35_oof": float(rh_v35),
        "v44_oof": float(rh_v44),
        "v58_oof": float(rh_v58),
        "delta_v58_v35": float(rh_v58 - rh_v35),
        "delta_v58_v44": float(rh_v58 - rh_v44),
        "delta_mm_mean": float(delta_norm.mean()*1000),
        "hybrid_v35_alpha": float(best_a),
        "hybrid_v35_oof": float(best_r),
        "hybrid_v48_3way": list(best_2),
        "hybrid_v48_3way_oof": float(best_2r),
        "submission_path": str(out_csv),
    }
    log_path = PROJECT_DIR / "run_log.json"
    log = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    log.append(entry)
    json.dump(log, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  [run_log] appended v58_boundary_mixed_base")


if __name__ == "__main__":
    main()

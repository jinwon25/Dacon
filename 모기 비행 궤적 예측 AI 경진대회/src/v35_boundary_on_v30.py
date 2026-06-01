"""v35_boundary_on_v30.py — Boundary refinement MLP on v30 base.

v26 (v23 base + boundary MLP) = +0.0045 OOF 향상 입증.
v30 (5-fold + multi-seed + adv reweight) OOF 0.6588 > v23 fast 0.6516.
같은 boundary MLP 적용 → v30 base는 더 강하므로 추가 향상 가능.

설계:
  - Input features: v30 + gate + v16 + Kalman + trajectory features (~30 dim)
  - MLP: 30→64→32→3, tanh × 1cm cap output
  - Loss: combo (euclid + 0.3 softhit) + sample_weight (boundary 2x)
  - 5-fold OOF, 3 seeds avg (fold-fixed)

학습 시간: ~3분 (CPU MLP 빠름)
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


def build_features(X, kalman, v30_pred, gate_pred, v16_pred):
    """v23 boundary feats + gate feats."""
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    diff_vg = v30_pred - gate_pred
    diff_vv = v30_pred - v16_pred
    dist_vg = np.linalg.norm(diff_vg, axis=-1, keepdims=True)
    res_v30_kal = v30_pred - kalman
    return np.concatenate([
        v30_pred, gate_pred, v16_pred,
        diff_vg, diff_vv, dist_vg,
        kalman, res_v30_kal,
        last_pos, v, a,
        v_mean, v_std,
        speed, a_norm,
    ], axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, v30_tr, feat_test, test_v30, sample_w_tr,
                    args, kf):
    device = torch.device("cpu")
    oof_pred = np.zeros_like(v30_tr)
    test_per_fold = []
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        feat_tr_n = sc.transform(feat_tr[tr]).astype(np.float32)
        feat_va_n = sc.transform(feat_tr[va]).astype(np.float32)
        feat_te_n = sc.transform(feat_test).astype(np.float32)

        def T(a): return torch.from_numpy(a).to(device)
        x_tr, x_va, x_te = T(feat_tr_n), T(feat_va_n), T(feat_te_n)
        v30_tr_t, v30_va_t, v30_te_t = (T(v30_tr[tr].astype(np.float32)),
                                          T(v30_tr[va].astype(np.float32)),
                                          T(test_v30.astype(np.float32)))
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
                pred = v30_tr_t[idx] + model(x_tr[idx])
                loss = loss_combo(pred, y_tr_t[idx], sw=sw_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pred_va = (v30_va_t + model(x_va)).cpu().numpy()
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
            oof_pred[va] = (v30_va_t + model(x_va)).cpu().numpy()
            test_per_fold.append((v30_te_t + model(x_te)).cpu().numpy())
        rh_fold = float((np.linalg.norm(oof_pred[va] - y_tr[va], axis=-1) <= 0.01).mean())
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)")
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
    print(f"v35 boundary MLP on v30 base ({args.n_seeds} seeds, cap {args.cap_cm}cm)")
    print("=" * 60)

    # --- Load ---
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

    # v30
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    oof_v30 = kalman_train + (st30["oof_A"] + st30["oof_B"])/2 * ALPHA
    test_v30 = kalman_test + (st30["test_A"] + st30["test_B"])/2 * ALPHA

    # gate
    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    test_gate = df_best[["x","y","z"]].values.astype(np.float64)

    # v16
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)

    # baseline R-Hits
    rh_v30 = float((np.linalg.norm(oof_v30 - y_train, axis=-1) <= 0.01).mean())
    print(f"v30 base OOF: {rh_v30:.4f}")

    # boundary mask
    d_v30 = np.linalg.norm(oof_v30 - y_train, axis=-1)
    boundary_mask = (d_v30 > 0.005) & (d_v30 <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"boundary samples (0.5cm < d ≤ 3cm): {boundary_mask.sum()} ({args.boundary_weight}× loss weight)")

    # Features
    feat_train = build_features(X_train, kalman_train, oof_v30, gate_oof, oof_v16)
    feat_test  = build_features(X_test, kalman_test, test_v30, test_gate, test_v16)
    print(f"feat dim: {feat_train.shape[1]}")

    # Multi-seed training
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, oof_v30,
                                          feat_test, test_v30, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v35 = np.mean(oofs, axis=0)
    test_v35 = np.mean(tests, axis=0)
    rh_v35 = float((np.linalg.norm(oof_v35 - y_train, axis=-1) <= 0.01).mean())

    print(f"\n=== v35 결과 (avg over {args.n_seeds} seeds) ===")
    print(f"  v30 base OOF: {rh_v30:.4f}")
    print(f"  v35 OOF: {rh_v35:.4f}  (Δ vs v30: {rh_v35 - rh_v30:+.4f})")

    # Δ stats
    delta = oof_v35 - oof_v30
    delta_norm = np.linalg.norm(delta, axis=-1)
    print(f"  Δ stats: mean={delta_norm.mean()*100:.3f}cm, median={np.median(delta_norm)*100:.3f}cm, "
          f"p99={np.percentile(delta_norm, 99)*100:.3f}cm")

    # Save state
    np.savez(CACHE_DIR / "v35_state.npz",
              oof_v35=oof_v35, test_v35=test_v35, rh_v35=rh_v35,
              n_seeds=args.n_seeds, cap_cm=args.cap_cm, boundary_weight=args.boundary_weight)

    # Submission
    out_csv = DATA_DIR / "submission_v35_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v35[:,0], "y": test_v35[:,1], "z": test_v35[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    # v33 baseline ensemble + v35 추가
    print(f"\n=== v35 ensemble candidates ===")
    a3, b3, c3 = 0.60, 0.38, 0.02
    ens_3way = a3*oof_v30 + b3*gate_oof + c3*oof_v16
    rh_3way = float((np.linalg.norm(ens_3way - y_train, axis=-1) <= 0.01).mean())
    print(f"  v33 3way baseline: {rh_3way:.4f}")

    # Replace v30 with v35 in 3way
    ens_3way_v35 = a3*oof_v35 + b3*gate_oof + c3*oof_v16
    rh_3way_v35 = float((np.linalg.norm(ens_3way_v35 - y_train, axis=-1) <= 0.01).mean())
    print(f"  3way with v35 instead of v30: {rh_3way_v35:.4f}  (Δ {rh_3way_v35 - rh_3way:+.4f})")
    test_3way_v35 = a3*test_v35 + b3*test_gate + c3*test_v16

    # 4-way with v30 + v35 + gate + v16 (full grid)
    print(f"\n=== 4-way grid (v30 + v35 + gate + v16) ===")
    best_4 = None; best_4r = 0
    for a in np.linspace(0.0, 0.7, 15):
        for b in np.linspace(0.0, 0.8 - a, 17):
            for c in np.linspace(0.0, 0.7, 15):
                d = 1 - a - b - c
                if d < 0 or d > 0.3: continue
                ens = a*oof_v30 + b*oof_v35 + c*gate_oof + d*oof_v16
                r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
                if r > best_4r:
                    best_4r, best_4 = r, (a, b, c, d)
    a, b, c, d = best_4
    print(f"  best: v30={a:.2f}, v35={b:.2f}, gate={c:.2f}, v16={d:.2f} → OOF {best_4r:.4f}")
    test_4 = a*test_v30 + b*test_v35 + c*test_gate + d*test_v16

    # Save best CSV
    ensembles_to_save = {
        "v35_alone": (test_v35, rh_v35),
        "3way_with_v35": (test_3way_v35, rh_3way_v35),
        f"4way_v30{a:.2f}_v35{b:.2f}_gate{c:.2f}_v16{d:.2f}": (test_4, best_4r),
    }
    for name, (tp, r) in ensembles_to_save.items():
        safe = name.replace(".","p").replace("=","").replace("(","").replace(")","").replace(",","").replace(" ","")[:45]
        out = DATA_DIR / f"submission_v35_{safe}.csv"
        pd.DataFrame({"id": sub["id"], "x": tp[:,0], "y": tp[:,1], "z": tp[:,2]}
                     ).to_csv(out, index=False)
        print(f"  saved: {out.name}  (OOF {r:.4f})")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v35_boundary_v30",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"Boundary MLP on v30 base × {args.n_seeds}-seed",
        "v30_oof": float(rh_v30),
        "v35_oof": float(rh_v35),
        "delta_v35_v30": float(rh_v35 - rh_v30),
        "v33_3way_oof": float(rh_3way),
        "3way_with_v35_oof": float(rh_3way_v35),
        "4way_grid_best": {"v30": a, "v35": b, "gate": c, "v16": d, "oof": float(best_4r)},
        "cap_cm": args.cap_cm, "boundary_weight": args.boundary_weight, "hidden": args.hidden,
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

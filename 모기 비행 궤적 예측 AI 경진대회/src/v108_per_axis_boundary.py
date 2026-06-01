"""v108_per_axis_boundary.py - Boundary MLP with per-axis cap.

motivation:
  - per-axis residual analysis on v97 OOF:
      x abs mean 0.624cm, y 0.641cm, z 0.454cm
      z axis is already well-predicted by base; current global cap
      over-corrects z and saturates x/y.
  - post-hoc per-axis scaling shows in-sample +0.002 but CV-unstable.
  - solution: apply per-axis cap at training time so the MLP head learns
    axis-specific residual distributions natively.

design:
  - v91 BoundaryMLP base, replace scalar cap with (3,) vector cap.
  - CLI --cap-xyz "x,y,z" in cm (e.g. "1.0,1.0,0.5").
  - features identical to v91 (base/gate/v16 + diffs + kalman + last_pos + v/a).
  - 5-fold × n-seeds.
  - separate cache: v108_capXxYyZz_state.npz where Xx etc are in 10cm units.
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


class BoundaryMLPPerAxis(nn.Module):
    def __init__(self, in_dim, hidden=64, p=0.2, cap_xyz_cm=(1.0, 1.0, 0.5)):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 3)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.register_buffer("cap", torch.tensor([c / 100.0 for c in cap_xyz_cm],
                                                  dtype=torch.float32))

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
                   args, kf, cap_xyz):
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
        model = BoundaryMLPPerAxis(in_dim=feat_tr_n.shape[1], hidden=args.hidden,
                                   p=0.2, cap_xyz_cm=cap_xyz).to(device)
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


def cap_tag(cap_xyz):
    return "_".join(f"{int(round(c*10)):02d}" for c in cap_xyz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cap-xyz", default="1.0,1.0,0.5",
                        help="per-axis cap in cm (comma-separated x,y,z)")
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--base-cache", default="v90_mirror_state.npz,v90_mirror_setupB_state.npz",
                        help="base cache(s), comma-separated will average")
    parser.add_argument("--out-tag", default="v108", help="prefix for output cache/csv")
    args = parser.parse_args()

    cap_xyz = tuple(float(c) for c in args.cap_xyz.split(","))
    assert len(cap_xyz) == 3, "cap-xyz must have 3 comma-separated values"
    tag = f"{args.out_tag}_{cap_tag(cap_xyz)}"

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v108 per-axis boundary MLP")
    print(f"  cap (x,y,z) cm = {cap_xyz}, tag = {tag}")
    print(f"  base = {args.base_cache}, seeds={args.n_seeds}, folds={args.n_folds}")
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

    cache_names = [c.strip() for c in args.base_cache.split(",")]
    oofs_residual, tests_residual = [], []
    for cn in cache_names:
        st = np.load(CACHE_DIR / cn)
        oofs_residual.append(st["oof"])
        tests_residual.append(st["test_pred"])
    oof_residual = np.mean(oofs_residual, axis=0)
    test_residual = np.mean(tests_residual, axis=0)
    oof_base = kalman_train + oof_residual * ALPHA
    test_base = kalman_test + test_residual * ALPHA

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    test_gate = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)

    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64); test_v16 = st16["test"].astype(np.float64)

    rh_base = float((np.linalg.norm(oof_base - y_train, axis=-1) <= 0.01).mean())
    print(f"v90 base OOF: {rh_base:.4f}")

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
        print(f"\n=== Training seed {s} (cap={cap_xyz}) ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, oof_base,
                                        feat_test, test_base, sample_w, args, kf, cap_xyz)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_pred = np.mean(oofs, axis=0)
    test_pred = np.mean(tests, axis=0)
    rh = float((np.linalg.norm(oof_pred - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== {tag} 결과 ===")
    print(f"  base OOF        : {rh_base:.4f}")
    print(f"  {tag} : {rh:.4f}  (Δ vs base: {rh - rh_base:+.4f}, "
          f"vs v97 cap1.0: {rh - 0.6741:+.4f}, vs v97 cap1.5: {rh - 0.6749:+.4f})")

    np.savez(CACHE_DIR / f"{tag}_state.npz",
             oof=oof_pred, test_pred=test_pred, rh=rh,
             n_seeds=args.n_seeds, cap_xyz=np.array(cap_xyz))
    out_csv = DATA_DIR / f"submission_{tag}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pred[:,0], "y": test_pred[:,1], "z": test_pred[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  state: {tag}_state.npz")
    print(f"  [submission] {out_csv.name}")

    entry = {"version": tag, "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": "per-axis cap boundary MLP on v90 mirror base",
             "cap_xyz": list(cap_xyz),
             "rh_base": float(rh_base), "rh": float(rh), "delta_base": float(rh - rh_base)}
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

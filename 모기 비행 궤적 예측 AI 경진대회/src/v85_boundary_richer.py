"""v85_boundary_richer.py — v78 boundary MLP + richer features (v30/v35/v44 predictions).

v78 base = v77 BiGRU + boundary MLP. feature 39 dim (base, gate, v16, kalman + trajectory).
v85: 추가 features = v30/v35/v44/v39 prediction + 모든 pairwise distances → boundary MLP가
sample-wise paradigm convergence/divergence 정보 활용 가능.

핵심:
  - boundary MLP가 "여러 모델의 disagreement" 시그널을 학습 → multi-modal sample identifier
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


class BoundaryMLP(nn.Module):
    def __init__(self, in_dim, hidden=96, p=0.25, cap_cm=1.0):
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


def build_features_rich(X, kalman, base_pred, gate_pred, v16_pred,
                          v30_pred, v35_pred, v44_pred, v39_pred):
    """v78 features + v30/v35/v44/v39 + pairwise distances among bases"""
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    # original v78 features
    diff_bg = base_pred - gate_pred
    diff_bv = base_pred - v16_pred
    dist_bg = np.linalg.norm(diff_bg, axis=-1, keepdims=True)
    res_b_kal = base_pred - kalman
    # NEW: 다른 base 모델 prediction과의 disagreement
    bases = [v30_pred, v35_pred, v44_pred, v39_pred]
    names = ["v30", "v35", "v44", "v39"]
    base_mean = np.mean(bases, axis=0)  # (N, 3)
    extra_feats = []
    # 각 base prediction (vs current base)
    for bp in bases:
        extra_feats.append(bp - base_pred)  # (N, 3)
        extra_feats.append(np.linalg.norm(bp - base_pred, axis=-1, keepdims=True))  # (N, 1)
    # base vs mean
    extra_feats.append(base_pred - base_mean)
    extra_feats.append(np.linalg.norm(base_pred - base_mean, axis=-1, keepdims=True))
    # pool std
    pool_stack = np.stack(bases + [base_pred, gate_pred], axis=0)  # (6, N, 3)
    pool_std = pool_stack.std(axis=0)
    extra_feats.append(pool_std)
    extra_feats.append(np.linalg.norm(pool_std, axis=-1, keepdims=True))
    return np.concatenate([
        base_pred, gate_pred, v16_pred,
        diff_bg, diff_bv, dist_bg,
        kalman, res_b_kal,
        last_pos, v, a,
        v_mean, v_std,
        speed, a_norm,
    ] + extra_feats, axis=-1).astype(np.float32)


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
        model = BoundaryMLP(in_dim=feat_tr_n.shape[1], hidden=args.hidden, p=args.dropout,
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
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--base", default="v77", choices=["v77", "v30"])
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v85 richer boundary MLP on {args.base} (cap {args.cap_cm}cm, hidden {args.hidden})")
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

    if args.base == "v77":
        st = np.load(CACHE_DIR / "v77_bigru_state.npz")
        oof_base = kalman_train + (st["oof_A"] + st["oof_B"])/2 * ALPHA
        test_base = kalman_test + (st["test_A"] + st["test_B"])/2 * ALPHA
    else:
        st = np.load(CACHE_DIR / "v30_state.npz")
        oof_base = kalman_train + (st["oof_A"] + st["oof_B"])/2 * ALPHA
        test_base = kalman_test + (st["test_A"] + st["test_B"])/2 * ALPHA

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    test_gate = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64); test_v16 = st16["test"].astype(np.float64)

    # Extra base predictions
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    v30_oof = kalman_train + (st30["oof_A"] + st30["oof_B"])/2 * ALPHA
    v30_test = kalman_test + (st30["test_A"] + st30["test_B"])/2 * ALPHA
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v35_oof = st35["oof_v35"].astype(np.float64); v35_test = st35["test_v35"].astype(np.float64)
    st44 = np.load(CACHE_DIR / "v44_state.npz")
    v44_oof = st44["oof_v44"].astype(np.float64); v44_test = st44["test_v44"].astype(np.float64)
    st39 = np.load(CACHE_DIR / "v39_state.npz")
    v39_oof = st39["oof_v39"].astype(np.float64); v39_test = st39["test_v39"].astype(np.float64)

    rh_base = float((np.linalg.norm(oof_base - y_train, axis=-1) <= 0.01).mean())
    print(f"{args.base} base OOF: {rh_base:.4f}")

    d_base = np.linalg.norm(oof_base - y_train, axis=-1)
    boundary_mask = (d_base > 0.005) & (d_base <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"boundary samples: {boundary_mask.sum()} ({args.boundary_weight}× weight)")

    feat_train = build_features_rich(X_train, kalman_train, oof_base, gate_oof, oof_v16,
                                        v30_oof, v35_oof, v44_oof, v39_oof)
    feat_test = build_features_rich(X_test, kalman_test, test_base, test_gate, test_v16,
                                       v30_test, v35_test, v44_test, v39_test)
    print(f"feat dim: {feat_train.shape[1]} (v78 was 39)")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, oof_base,
                                        feat_test, test_base, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v85 = np.mean(oofs, axis=0); test_v85 = np.mean(tests, axis=0)
    rh_v85 = float((np.linalg.norm(oof_v85 - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== v85 결과 ===")
    print(f"  base    : {rh_base:.4f}")
    print(f"  v85     : {rh_v85:.4f}  (Δ vs base: {rh_v85 - rh_base:+.4f})")
    print(f"  vs v78 0.6730: {rh_v85 - 0.6730:+.4f}")
    print(f"  vs v35 0.6725: {rh_v85 - 0.6725:+.4f}")

    state_name = f"v85_{args.base}_state.npz"
    np.savez(CACHE_DIR / state_name,
             oof_v85=oof_v85, test_v85=test_v85, rh_v85=rh_v85)
    out_csv = DATA_DIR / f"submission_v85_{args.base}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v85[:,0], "y": test_v85[:,1], "z": test_v85[:,2]}).to_csv(out_csv, index=False)
    print(f"  state: {state_name}, submission: {out_csv.name}")

    # blend with v48 3-way
    v48s = np.load(CACHE_DIR / "v48_state.npz"); v46s = np.load(CACHE_DIR / "v46_state.npz")
    bo_o = 0.70*v48s["oof_v48"] + 0.12*v46s["oof_v46"] + 0.18*v35_oof
    bo_t = 0.70*v48s["test_v48"] + 0.12*v46s["test_v46"] + 0.18*v35_test
    rh_b = float((np.linalg.norm(bo_o - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== blend v48 3-way + v85 ===")
    best_w, best_r = 1.0, rh_b
    for w in np.linspace(0, 1, 21):
        ens = w * bo_o + (1 - w) * oof_v85
        r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
        if r > best_r: best_r, best_w = r, w
    print(f"  best 2way w={best_w:.2f} → OOF {best_r:.4f}  Δ {best_r - rh_b:+.4f}")

    # 4-way: v48_9m / v46_7m / v35 / v85
    print(f"\n=== 4-way grid (v48_9m/v46_7m/v35/v85) ===")
    best_w4, best_r4 = None, rh_b
    po = [v48s["oof_v48"], v46s["oof_v46"], v35_oof, oof_v85]
    pt = [v48s["test_v48"], v46s["test_v46"], v35_test, test_v85]
    for a in np.linspace(0.55, 0.85, 7):
        for b in np.linspace(0.05, 0.20, 6):
            for c in np.linspace(0.10, 0.25, 6):
                d = 1 - a - b - c
                if d < 0 or d > 0.20: continue
                ens = a*po[0]+b*po[1]+c*po[2]+d*po[3]
                r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
                if r > best_r4: best_r4, best_w4 = r, (a, b, c, d)
    if best_w4:
        a, b, c, d = best_w4
        ens_t = a*pt[0]+b*pt[1]+c*pt[2]+d*pt[3]
        out4 = DATA_DIR / f"submission_v85_4way_{a:.2f}_{b:.2f}_{c:.2f}_{d:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out4, index=False)
        print(f"  best 4way ({a:.2f}/{b:.2f}/{c:.2f}/{d:.2f}): OOF {best_r4:.4f}  Δ {best_r4 - rh_b:+.4f}")
        print(f"  [submission] {out4.name}")

    entry = {"version": "v85_boundary_richer", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "base": args.base, "rh_base": rh_base, "rh_v85": rh_v85,
             "feat_dim": feat_train.shape[1],
             "blend_2way_oof": float(best_r), "blend_2way_w": float(best_w),
             "blend_4way_oof": float(best_r4) if best_w4 else rh_b,
             "blend_4way_weights": list(best_w4) if best_w4 else None}
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

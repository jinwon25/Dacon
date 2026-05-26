"""v91_boundary_on_v90.py — Boundary MLP on v90 mirror BiGRU base.

v78 패턴 그대로: base prediction → boundary refinement MLP (5-fold × 3-seed).
v90 base = v77 BiGRU + y-mirror augmentation + TTA (paradigm 다양성).

v77 base OOF 0.6597 → v78 boundary 0.6730 (+0.0133).
v90 base OOF ~ 0.66+ (mirror) → v91 boundary ~ 0.68+ 기대.

별도 cache 사용: {args.out_tag}_state.npz
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
    parser.add_argument("--base-cache", default="v90_mirror_state.npz",
                        help="v90 base cache file (콤마 구분 시 평균)")
    parser.add_argument("--out-tag", default="v91",
                        help="output state/csv suffix (e.g. v91, v94)")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v91 boundary MLP on v90 mirror base "
          f"({args.n_seeds} seeds × {args.n_folds} folds, cap {args.cap_cm}cm)")
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

    # v90 base (mirror augmented BiGRU) — 단일/다중 cache 평균 지원
    cache_names = [c.strip() for c in args.base_cache.split(",")]
    oofs_residual, tests_residual = [], []
    for cn in cache_names:
        st = np.load(CACHE_DIR / cn)
        oofs_residual.append(st["oof"])
        tests_residual.append(st["test_pred"])
    oof_residual = np.mean(oofs_residual, axis=0)
    test_residual = np.mean(tests_residual, axis=0)
    oof_v90 = kalman_train + oof_residual * ALPHA
    test_v90 = kalman_test + test_residual * ALPHA
    print(f"[base] {len(cache_names)} cache 평균: {cache_names}")

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    test_gate = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)

    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64); test_v16 = st16["test"].astype(np.float64)

    rh_v90 = float((np.linalg.norm(oof_v90 - y_train, axis=-1) <= 0.01).mean())
    print(f"v90 mirror base OOF: {rh_v90:.4f}")

    d_v90 = np.linalg.norm(oof_v90 - y_train, axis=-1)
    boundary_mask = (d_v90 > 0.005) & (d_v90 <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"boundary samples (0.5cm < d ≤ 3cm): {boundary_mask.sum()} ({args.boundary_weight}× loss weight)")

    feat_train = build_features(X_train, kalman_train, oof_v90, gate_oof, oof_v16)
    feat_test  = build_features(X_test, kalman_test, test_v90, test_gate, test_v16)
    print(f"feat dim: {feat_train.shape[1]}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, oof_v90,
                                        feat_test, test_v90, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v91 = np.mean(oofs, axis=0)
    test_v91 = np.mean(tests, axis=0)
    rh_v91 = float((np.linalg.norm(oof_v91 - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== v91 결과 ===")
    print(f"  v90 mirror base : {rh_v90:.4f}")
    print(f"  v91 boundary    : {rh_v91:.4f}  (Δ vs v90: {rh_v91 - rh_v90:+.4f}, "
          f"vs v35: {rh_v91 - 0.6725:+.4f}, vs v78: {rh_v91 - 0.6730:+.4f})")

    np.savez(CACHE_DIR / f"{args.out_tag}_state.npz",
             oof_v91=oof_v91, test_v91=test_v91, rh_v91=rh_v91,
             n_seeds=args.n_seeds, cap_cm=args.cap_cm)
    print(f"  state: {args.out_tag}_state.npz")

    out_csv = DATA_DIR / f"submission_{args.out_tag}_boundary_on_v90.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v91[:,0], "y": test_v91[:,1], "z": test_v91[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv.name}")

    # blend with current best stacker (v48 3-way) and v35
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v48s = np.load(CACHE_DIR / "v48_state.npz"); v46s = np.load(CACHE_DIR / "v46_state.npz")
    base_o = 0.70*v48s["oof_v48"] + 0.12*v46s["oof_v46"] + 0.18*st35["oof_v35"]
    base_t = 0.70*v48s["test_v48"] + 0.12*v46s["test_v46"] + 0.18*st35["test_v35"]
    rh_base = float((np.linalg.norm(base_o - y_train, axis=-1) <= 0.01).mean())
    print(f"\nv48 3-way base OOF: {rh_base:.4f}")

    # 2-way blend
    best_w, best_r = 1.0, rh_base
    for w in np.linspace(0, 1, 41):
        ens = w * base_o + (1 - w) * oof_v91
        r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
        if r > best_r: best_r, best_w = r, w
    print(f"  2-way blend best: w={best_w:.3f}*base + (1-w)*v91 → OOF {best_r:.4f}  Δ {best_r-rh_base:+.4f}")
    if best_w < 1.0:
        ens_t = best_w*base_t + (1-best_w)*test_v91
        out2 = DATA_DIR / f"submission_{args.out_tag}_2way_base{best_w:.2f}_v_{1-best_w:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}
                     ).to_csv(out2, index=False)
        print(f"  [submission] {out2.name}")

    # 4-way grid: base + v91 + v35 + v78 (paradigm diversity)
    try:
        st78 = np.load(CACHE_DIR / "v78_state.npz")
        oof_v78 = st78["oof_v78"]; test_v78 = st78["test_v78"]
    except FileNotFoundError:
        oof_v78 = test_v78 = None
    if oof_v78 is not None:
        best4 = (rh_base, None)
        for a in np.linspace(0.3, 1.0, 15):
            for b in np.linspace(0, 1-a, 11):
                for c in np.linspace(0, 1-a-b, 6):
                    d = 1 - a - b - c
                    if d < 0: continue
                    ens = a*base_o + b*oof_v91 + c*st35["oof_v35"] + d*oof_v78
                    r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
                    if r > best4[0]: best4 = (r, (a, b, c, d))
        if best4[1]:
            a, b, c, d = best4[1]
            ens_t = a*base_t + b*test_v91 + c*st35["test_v35"] + d*test_v78
            out4 = DATA_DIR / f"submission_{args.out_tag}_4w_base{a:.2f}_v_{b:.2f}_v35{c:.2f}_v78{d:.2f}.csv"
            pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}
                         ).to_csv(out4, index=False)
            print(f"\n  4-way grid best: ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}) → OOF {best4[0]:.4f}  "
                  f"Δ {best4[0]-rh_base:+.4f}")
            print(f"  [submission] {out4.name}")

    entry = {"version": "v91_boundary_on_v90", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": "boundary MLP on v90 mirror BiGRU base",
             "rh_v90": float(rh_v90), "rh_v91": float(rh_v91),
             "delta_vs_v90": float(rh_v91 - rh_v90),
             "delta_vs_v35": float(rh_v91 - 0.6725),
             "delta_vs_v78": float(rh_v91 - 0.6730),
             "blend_2way_best_oof": float(best_r),
             "blend_2way_best_w": float(best_w)}
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

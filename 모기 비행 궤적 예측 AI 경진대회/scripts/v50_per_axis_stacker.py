"""v50_per_axis_stacker.py — Per-axis SoftStacker (axis-wise sample-wise weights).

v46/v48 SoftStacker: 3D joint weight (1 weight × 3 coords).
v50: 각 axis (x, y, z) 별 독립 softmax weight (3 × M logits).

Hypothesis: 모델별 강점이 axis 따라 다름 (예: v35 x축 강, v44 z축 강).
9-model pool 동일. v48 base와 동일 framework.

v25 per-axis LGB는 axis joint correlation 손실로 실패 (joint R-Hit 0.40).
v50은 convex combination 유지, axis는 weight만 분리 → 안전.
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


def loss_combo(pred, target):
    d = torch.sqrt(((pred - target) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    return d.mean() + 0.3 * sh.mean()


class PerAxisStacker(nn.Module):
    """Per-axis softmax weighted ensemble.

    output[b, c] = Σ_m softmax(logits[b, c, m])_m × preds[b, m, c]
    """

    def __init__(self, in_dim, n_models, hidden=96, p=0.4, temp=1.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, n_models * 3)  # M × 3
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.n_models = n_models
        self.temp = temp

    def forward(self, x, preds):
        # preds: (B, M, 3)
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z)); z = self.drop(z)
        logits = self.head(z).view(-1, 3, self.n_models) / self.temp  # (B, 3, M)
        w = F.softmax(logits, dim=-1)  # (B, 3, M) — softmax over M for each axis
        # preds: (B, M, 3) → (B, 3, M) for axis-aligned dot
        preds_t = preds.transpose(1, 2)  # (B, 3, M)
        out = (w * preds_t).sum(dim=-1)  # (B, 3)
        return out, w


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
    key_idx = list(range(min(7, len(model_preds))))
    for i_pos, i in enumerate(key_idx):
        for j in key_idx[i_pos+1:]:
            d_ij = np.linalg.norm(model_preds[i] - model_preds[j], axis=-1, keepdims=True)
            feats.append(d_ij)
    return np.concatenate(feats, axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, preds_tr, feat_te, preds_te, args, kf):
    device = torch.device("cpu")
    N = feat_tr.shape[0]
    M = preds_tr.shape[1]
    oof_pred = np.zeros((N, 3), dtype=np.float64)
    test_per_fold = []
    weights_oof = np.zeros((N, 3, M), dtype=np.float32)
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
        model = PerAxisStacker(in_dim=ftn_tr.shape[1], n_models=M,
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
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f} ({(time.time()-t0)/60:.1f}m)")
        del model; gc.collect()
    return oof_pred, np.mean(test_per_fold, axis=0), weights_oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--temp", type=float, default=1.5)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v50 Per-Axis SoftStacker — 9 models, hid {args.hidden}, drop {args.dropout}, temp {args.temp}")
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

    st39 = np.load(CACHE_DIR / "v39_state.npz")
    v39_oof, v39_te = st39["oof_v39"].astype(np.float64), st39["test_v39"].astype(np.float64)

    st32 = np.load(CACHE_DIR / "v32_mdn_state.npz")
    v32_oof, v32_te = st32["oof_weighted"].astype(np.float64), st32["test_weighted"].astype(np.float64)

    models_train = [v30A_oof, v30B_oof, v35_oof, v41A_oof, v41B_oof, v44_oof, gate_oof, v39_oof, v32_oof]
    models_test  = [v30A_te,  v30B_te,  v35_te,  v41A_te,  v41B_te,  v44_te,  gate_te,  v39_te,  v32_te]
    names = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate", "v39", "v32"]

    print("\n=== Per-axis hit rate ===")
    for n, p in zip(names, models_train):
        ax_hit = [((p[:, c] - y_train[:, c])**2 < 0.01**2).mean() for c in range(3)]
        joint_hit = rhit(p, y_train)
        print(f"  {n}: x={ax_hit[0]:.3f} y={ax_hit[1]:.3f} z={ax_hit[2]:.3f}  joint={joint_hit:.4f}")

    feat_tr = build_features(X_train, kalman_train, models_train)
    feat_te = build_features(X_test,  kalman_test,  models_test)
    preds_tr = np.stack(models_train, axis=1).astype(np.float64)
    preds_te = np.stack(models_test,  axis=1).astype(np.float64)
    print(f"\n  feat dim: {feat_tr.shape[1]}, preds: {preds_tr.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests, all_w = [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} ===")
        oof_s, test_s, w_s = train_one_seed(s, feat_tr, y_train, preds_tr,
                                             feat_te, preds_te, args, kf)
        oofs.append(oof_s); tests.append(test_s); all_w.append(w_s)
        rh_s = rhit(oof_s, y_train)
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v50 = np.mean(oofs, axis=0)
    test_v50 = np.mean(tests, axis=0)
    w_mean = np.mean(all_w, axis=0)  # (N, 3, M)
    rh_v50 = rhit(oof_v50, y_train)

    print(f"\n=== v50 Per-Axis SoftStacker 결과 ===")
    print(f"  v35 alone       : {rhit(v35_oof, y_train):.4f}")
    print(f"  v46 (joint, 7m) : 0.6734")
    print(f"  v48 (joint, 9m) : 0.6734")
    print(f"  v50 (per-axis)  : {rh_v50:.4f}  (Δ vs v48: {rh_v50 - 0.6734:+.4f})")
    print(f"  weight avg per axis (M-vector):")
    for c, axis_name in enumerate(["x", "y", "z"]):
        print(f"    {axis_name}: ", end="")
        for n, wi in zip(names, w_mean.mean(axis=0)[c]):
            print(f"{n}:{wi:.2f} ", end="")
        print()

    # Hybrid sweeps
    print(f"\n=== Hybrid sweeps ===")
    best_alpha35, best_r35 = 1.0, rh_v50
    for alpha in np.linspace(0.0, 1.0, 21):
        ens = alpha * oof_v50 + (1 - alpha) * v35_oof
        r = rhit(ens, y_train)
        if r > best_r35: best_r35, best_alpha35 = r, alpha
    print(f"  v50+v35: α={best_alpha35:.2f} OOF={best_r35:.4f}")

    try:
        st46 = np.load(CACHE_DIR / "v46_state.npz")
        oof_v46, test_v46 = st46["oof_v46"], st46["test_v46"]
        st48 = np.load(CACHE_DIR / "v48_state.npz")
        oof_v48, test_v48 = st48["oof_v48"], st48["test_v48"]

        best_3 = (1.0, 0.0, 0.0); best_3r = rh_v50
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1-a, 11):
                c = 1 - a - b
                ens = a*oof_v50 + b*oof_v48 + c*v35_oof
                r = rhit(ens, y_train)
                if r > best_3r:
                    best_3r, best_3 = r, (a, b, c)
        print(f"  3-way v50/v48/v35: {best_3[0]:.2f}/{best_3[1]:.2f}/{best_3[2]:.2f} → {best_3r:.4f}")

        best_4 = (1.0, 0.0, 0.0, 0.0); best_4r = rh_v50
        for a in np.linspace(0, 1, 6):
            for b in np.linspace(0, 1-a, 6):
                for c in np.linspace(0, 1-a-b, 6):
                    d = 1 - a - b - c
                    if d < 0: continue
                    ens = a*oof_v50 + b*oof_v48 + c*v35_oof + d*oof_v46
                    r = rhit(ens, y_train)
                    if r > best_4r:
                        best_4r, best_4 = r, (a, b, c, d)
        print(f"  4-way v50/v48/v35/v46: {best_4[0]:.2f}/{best_4[1]:.2f}/{best_4[2]:.2f}/{best_4[3]:.2f} → {best_4r:.4f}")
    except Exception as e:
        print(f"  hybrid cache err: {e}")
        oof_v46 = test_v46 = oof_v48 = test_v48 = None
        best_3, best_3r = (1.0, 0.0, 0.0), rh_v50
        best_4, best_4r = (1.0, 0.0, 0.0, 0.0), rh_v50

    np.savez(CACHE_DIR / "v50_state.npz",
             oof_v50=oof_v50, test_v50=test_v50, rh_v50=rh_v50,
             weights_oof_mean=w_mean)

    out_csv = DATA_DIR / "submission_v50_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v50[:,0], "y": test_v50[:,1], "z": test_v50[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n  [submission] {out_csv}")

    if best_3r > rh_v50 and oof_v48 is not None:
        a, b, c = best_3
        hyb = a*test_v50 + b*test_v48 + c*v35_te
        hyb_csv = DATA_DIR / f"submission_v50_3way_v50x{a:.2f}_v48x{b:.2f}_v35x{c:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb[:,0], "y": hyb[:,1], "z": hyb[:,2]}
                     ).to_csv(hyb_csv, index=False)
        print(f"  [3way] {hyb_csv.name}")

    if best_4r > rh_v50 and oof_v46 is not None and oof_v48 is not None:
        a, b, c, d = best_4
        hyb = a*test_v50 + b*test_v48 + c*v35_te + d*test_v46
        hyb_csv = DATA_DIR / f"submission_v50_4way_v50x{a:.2f}_v48x{b:.2f}_v35x{c:.2f}_v46x{d:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb[:,0], "y": hyb[:,1], "z": hyb[:,2]}
                     ).to_csv(hyb_csv, index=False)
        print(f"  [4way] {hyb_csv.name}")

    entry = {
        "version": "v50_per_axis_stacker",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Per-axis softmax weighted ensemble (9 models)",
        "n_models": len(names),
        "model_names": names,
        "v50_oof": float(rh_v50),
        "delta_vs_v48": float(rh_v50 - 0.6734),
        "weight_avg_x": {n: float(wi) for n, wi in zip(names, w_mean.mean(axis=0)[0])},
        "weight_avg_y": {n: float(wi) for n, wi in zip(names, w_mean.mean(axis=0)[1])},
        "weight_avg_z": {n: float(wi) for n, wi in zip(names, w_mean.mean(axis=0)[2])},
        "hybrid_v35_alpha": float(best_alpha35),
        "hybrid_v35_oof": float(best_r35),
        "3way_v50_v48_v35": list(best_3),
        "3way_oof": float(best_3r),
        "4way_v50_v48_v35_v46": list(best_4),
        "4way_oof": float(best_4r),
        "submission": str(out_csv),
    }
    log_path = PROJECT_DIR / "run_log.json"
    log = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    log.append(entry)
    json.dump(log, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  [run_log] appended v50_per_axis_stacker")


if __name__ == "__main__":
    main()

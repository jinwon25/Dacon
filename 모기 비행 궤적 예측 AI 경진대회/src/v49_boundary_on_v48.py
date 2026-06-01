"""v49_boundary_on_v48.py — Boundary refinement MLP on v48 SoftStacker base.

v35 (boundary on v30 OOF 0.6588) → OOF 0.6725 (+0.0137).
v44 (boundary on v41 OOF 0.6608) → OOF 0.6713 (+0.0105).
v48 (9-model SoftStacker) OOF 0.6734.
같은 boundary 적용 → OOF 0.680~0.685 추정. LB 변환 시 0.694+ 도전.

v44 framework 그대로, base만 v48로 변경.
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


def build_features(X, kalman, base_pred, gate_pred, v16_pred):
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
    return np.concatenate([
        base_pred, gate_pred, v16_pred,
        diff_vg, diff_vv, dist_vg,
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
    print(f"v49 = v48 (9-model SoftStacker) base + boundary MLP ({args.n_seeds} seeds)")
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

    # v48 (SoftStacker) base — already final prediction (ALPHA applied via component models)
    st48 = np.load(CACHE_DIR / "v48_state.npz")
    oof_v48 = st48["oof_v48"].astype(np.float64)
    test_v48 = st48["test_v48"].astype(np.float64)
    rh_v48 = float((np.linalg.norm(oof_v48 - y_train, axis=-1) <= 0.01).mean())
    print(f"v48 base OOF: {rh_v48:.4f}")

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
    d_v48 = np.linalg.norm(oof_v48 - y_train, axis=-1)
    boundary_mask = (d_v48 > 0.005) & (d_v48 <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"boundary samples (0.5cm<d<=3cm vs v48): {boundary_mask.sum()} ({args.boundary_weight}× wt)")

    feat_train = build_features(X_train, kalman_train, oof_v48, gate_oof, oof_v16)
    feat_test  = build_features(X_test, kalman_test, test_v48, test_gate, test_v16)
    print(f"feat dim: {feat_train.shape[1]}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, oof_v48,
                                          feat_test, test_v48, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v49 = np.mean(oofs, axis=0)
    test_v49 = np.mean(tests, axis=0)
    rh_v49 = float((np.linalg.norm(oof_v49 - y_train, axis=-1) <= 0.01).mean())

    # Compare
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    rh_v35 = float(st35["rh_v35"])
    v35_oof = st35["oof_v35"].astype(np.float64)
    v35_te = st35["test_v35"].astype(np.float64)

    print(f"\n=== v49 결과 ===")
    print(f"  v35 (boundary on v30): {rh_v35:.4f}  (LB 0.6874)")
    print(f"  v48 base OOF         : {rh_v48:.4f}")
    print(f"  v49 (boundary on v48): {rh_v49:.4f}  (Δ vs v48: {rh_v49 - rh_v48:+.4f})")
    print(f"  Δ vs v35: {rh_v49 - rh_v35:+.4f}")
    print(f"  → LB 추정: {rh_v49 + 0.0146:.4f} (보수) ~ {rh_v49 + 0.0150:.4f}")

    # Δ stats
    delta = oof_v49 - oof_v48
    delta_norm = np.linalg.norm(delta, axis=-1)
    print(f"  Δ stats: mean={delta_norm.mean()*1000:.2f}mm, median={np.median(delta_norm)*1000:.2f}mm, "
          f"p99={np.percentile(delta_norm, 99)*1000:.2f}mm")

    np.savez(CACHE_DIR / "v49_state.npz",
             oof_v49=oof_v49, test_v49=test_v49, rh_v49=rh_v49)

    out_csv = DATA_DIR / "submission_v49_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v49[:,0], "y": test_v49[:,1], "z": test_v49[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    # Hybrid sweep with v35 and v48
    print(f"\n=== Hybrid sweeps ===")
    # v49 + v35
    best_alpha35, best_r35 = 1.0, rh_v49
    for alpha in np.linspace(0.0, 1.0, 21):
        ens = alpha * oof_v49 + (1 - alpha) * v35_oof
        r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
        if r > best_r35: best_r35, best_alpha35 = r, alpha
    print(f"  v49+v35: α={best_alpha35:.2f} OOF={best_r35:.4f}")

    # v49 + v48 (sanity, expect v49 wins)
    best_alpha48, best_r48 = 1.0, rh_v49
    for alpha in np.linspace(0.0, 1.0, 21):
        ens = alpha * oof_v49 + (1 - alpha) * oof_v48
        r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
        if r > best_r48: best_r48, best_alpha48 = r, alpha
    print(f"  v49+v48: α={best_alpha48:.2f} OOF={best_r48:.4f}")

    # 3-way v49 + v35 + v48
    best_3 = (1.0, 0.0, 0.0); best_3r = rh_v49
    for a in np.linspace(0, 1, 11):
        for b in np.linspace(0, 1-a, 11):
            c = 1 - a - b
            ens = a*oof_v49 + b*v35_oof + c*oof_v48
            r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
            if r > best_3r:
                best_3r, best_3 = r, (a, b, c)
    print(f"  3-way v49/v35/v48: {best_3[0]:.2f}/{best_3[1]:.2f}/{best_3[2]:.2f} → {best_3r:.4f}")

    # Save hybrid CSV
    if best_alpha35 < 1.0 and best_r35 > rh_v49:
        hyb = best_alpha35 * test_v49 + (1 - best_alpha35) * v35_te
        hyb_csv = DATA_DIR / f"submission_v49_hybrid_v49x{best_alpha35:.2f}_v35x{1-best_alpha35:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb[:,0], "y": hyb[:,1], "z": hyb[:,2]}
                     ).to_csv(hyb_csv, index=False)
        print(f"  [hybrid v35] {hyb_csv.name}")

    if best_3r > rh_v49 and (best_3[0] < 1.0 or best_3[1] > 0 or best_3[2] > 0):
        a, b, c = best_3
        hyb3 = a*test_v49 + b*v35_te + c*test_v48
        hyb3_csv = DATA_DIR / f"submission_v49_3way_v49x{a:.2f}_v35x{b:.2f}_v48x{c:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb3[:,0], "y": hyb3[:,1], "z": hyb3[:,2]}
                     ).to_csv(hyb3_csv, index=False)
        print(f"  [3way] {hyb3_csv.name}")

    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v49_boundary_v48",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Boundary MLP on v48 (9-model SoftStacker) base",
        "v48_oof": float(rh_v48), "v49_oof": float(rh_v49),
        "v35_oof": float(rh_v35),
        "delta_v49_v48": float(rh_v49 - rh_v48),
        "delta_v49_v35": float(rh_v49 - rh_v35),
        "delta_mm_mean": float(delta_norm.mean()*1000),
        "hybrid_v35_best_alpha": float(best_alpha35),
        "hybrid_v35_best_oof": float(best_r35),
        "3way_v49_v35_v48": list(best_3),
        "3way_oof": float(best_3r),
        "submission_path": str(out_csv),
    }
    logs = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"\n  [run_log] appended v49_boundary_v48")


if __name__ == "__main__":
    main()

"""v64_softstacker_v62.py — v48 9-model SoftStacker + v62 (CA paradigm) = 10-model.

목적:
  v48 9-model SoftStacker OOF 0.6734 plateau. v62 (CA Kalman paradigm) 추가하여
  oracle 4-model any-hit 0.7039 (3-model 0.6840 대비 +0.0199) sample-wise 다양성 추출.

설계:
  - v48 9 models + v62 = 10-model SoftStacker
  - 같은 hyperparams (hidden 96, drop 0.4, temp 1.5)
  - 5-fold × 3-seed
  - hybrid grid (v48 3-way 기준 blend 포함)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import glob
import json
import os
import sys
import time
from pathlib import Path

# Windows cp949 console에서 unicode print 가능하도록
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import yaw_angle, inverse_rotate_xy

PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rhit(p, y):
    return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def loss_combo(pred, target):
    d = torch.sqrt(((pred - target) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    return d.mean() + 0.3 * sh.mean()


class SoftStacker(nn.Module):
    def __init__(self, in_dim, n_models, hidden=96, p=0.4, temp=1.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, n_models)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.temp = temp

    def forward(self, x, preds):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z)); z = self.drop(z)
        logits = self.head(z) / self.temp
        w = F.softmax(logits, dim=-1)
        out = (w.unsqueeze(-1) * preds).sum(dim=1)
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
    # pairwise distances among first 7 only (limit feat dim)
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
    print(f"v64 SoftStacker — 10 models (v48 9 + v62 CA paradigm)")
    print(f"   hidden {args.hidden}, drop {args.dropout}, temp {args.temp}, "
          f"{args.n_folds}-fold × {args.n_seeds}-seed")
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

    # 기존 9 models (v48 base)
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

    # NEW: v62 (CA Kalman paradigm) - residual를 final position으로 변환
    st62 = np.load(CACHE_DIR / "v62_state.npz")
    v62_oof_res = (st62["oof_A"] + st62["oof_B"]) / 2
    v62_te_res = (st62["test_A"] + st62["test_B"]) / 2
    v62_kal_tr = st62["kalman_train"]
    v62_kal_te = st62["kalman_test"]
    # CA base에 사용된 yaw rotation 복원
    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train = yaw_angle(v_last_train); theta_test = yaw_angle(v_last_test)
    v62_oof = v62_kal_tr + inverse_rotate_xy(v62_oof_res, theta_train)
    v62_te  = v62_kal_te + inverse_rotate_xy(v62_te_res, theta_test)

    models_train = [v30A_oof, v30B_oof, v35_oof, v41A_oof, v41B_oof, v44_oof,
                    gate_oof, v39_oof, v32_oof, v62_oof]
    models_test  = [v30A_te, v30B_te, v35_te, v41A_te, v41B_te, v44_te,
                    gate_te, v39_te, v32_te, v62_te]
    names = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate", "v39", "v32", "v62"]

    print("\n=== Model OOF R-Hit ===")
    for n, p in zip(names, models_train):
        print(f"  {n}: {rhit(p, y_train):.4f}")

    hits = np.stack([np.linalg.norm(p - y_train, axis=-1) <= 0.01 for p in models_train])
    oracle = hits.any(axis=0).mean()
    hits9 = hits[:9]
    oracle9 = hits9.any(axis=0).mean()
    print(f"\n  Oracle 9-model:  {oracle9:.4f}")
    print(f"  Oracle 10-model: {oracle:.4f}  (Δ v62 adds {oracle - oracle9:+.4f})")

    feat_tr = build_features(X_train, kalman_train, models_train)
    feat_te = build_features(X_test,  kalman_test,  models_test)
    preds_tr = np.stack(models_train, axis=1).astype(np.float64)
    preds_te = np.stack(models_test,  axis=1).astype(np.float64)
    print(f"  feat dim: {feat_tr.shape[1]}, preds: {preds_tr.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests, all_w = [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== Training seed {s} ===")
        oof_s, test_s, w_s = train_one_seed(s, feat_tr, y_train, preds_tr,
                                             feat_te, preds_te, args, kf)
        oofs.append(oof_s); tests.append(test_s); all_w.append(w_s)
        rh_s = rhit(oof_s, y_train)
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v64 = np.mean(oofs, axis=0)
    test_v64 = np.mean(tests, axis=0)
    w_mean = np.mean(all_w, axis=0)
    rh_v64 = rhit(oof_v64, y_train)

    # base reference
    base_oof = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["oof_v48"] \
             + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["oof_v46"] \
             + 0.18 * v35_oof
    base_te  = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["test_v48"] \
             + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["test_v46"] \
             + 0.18 * v35_te
    rh_base = rhit(base_oof, y_train)

    print(f"\n=== v64 결과 ===")
    print(f"  v35 alone           : {rhit(v35_oof, y_train):.4f}")
    print(f"  v48 (9m) alone      : {rhit(np.load(CACHE_DIR / 'v48_state.npz')['oof_v48'], y_train):.4f}")
    print(f"  v48 3-way (base)    : {rh_base:.4f}")
    print(f"  v64 (10m)           : {rh_v64:.4f}  (Δ vs v48: {rh_v64 - 0.6734:+.4f}, vs base: {rh_v64 - rh_base:+.4f})")
    print(f"\n  weight avg over OOF:")
    for n, wi in zip(names, w_mean.mean(axis=0)):
        marker = " ← NEW" if n == "v62" else ""
        print(f"    {n}: {wi:.3f}{marker}")

    # Hybrid grid: v64 vs v48 3-way blend
    print(f"\n=== Hybrid: w*v48_3way + (1-w)*v64 ===")
    best_w, best_rh = 0.0, rh_v64  # w=0 → all v64
    for w in np.linspace(0, 1, 21):
        ens = w * base_oof + (1 - w) * oof_v64
        r = rhit(ens, y_train)
        if r > best_rh: best_rh, best_w = r, w
    print(f"  best: w={best_w:.2f} → OOF {best_rh:.4f}  (vs v48 3-way {rh_base:.4f}: {best_rh - rh_base:+.4f})")

    # save state
    np.savez(CACHE_DIR / "v64_state.npz",
             oof_v64=oof_v64, test_v64=test_v64, rh_v64=rh_v64,
             weights_oof_mean=w_mean)

    # alone submission
    out_csv = DATA_DIR / "submission_v64_10m.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v64[:,0], "y": test_v64[:,1], "z": test_v64[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n  [submission] {out_csv}")

    if best_w > 0 and best_rh > rh_v64:
        blend_te = best_w * base_te + (1 - best_w) * test_v64
        hyb_csv = DATA_DIR / f"submission_v64_hybrid_v48x{best_w:.2f}_v64x{1-best_w:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": blend_te[:,0], "y": blend_te[:,1], "z": blend_te[:,2]}
                     ).to_csv(hyb_csv, index=False)
        print(f"  [hybrid] OOF {best_rh:.4f} → {hyb_csv.name}")

    entry = {
        "version": "v64_softstacker_v62",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "10-model SoftStacker (v48 9m + v62 CA paradigm)",
        "n_models": len(names), "model_names": names,
        "model_oof": {n: rhit(p, y_train) for n, p in zip(names, models_train)},
        "oracle_any9": float(oracle9),
        "oracle_any10": float(oracle),
        "v64_oof": float(rh_v64),
        "delta_vs_v48_9m": float(rh_v64 - 0.6734),
        "delta_vs_v48_3way": float(rh_v64 - rh_base),
        "weight_avg": {n: float(wi) for n, wi in zip(names, w_mean.mean(axis=0))},
        "hybrid_best_w_v48_3way": float(best_w),
        "hybrid_best_oof": float(best_rh),
        "submission": str(out_csv),
    }
    log_path = PROJECT_DIR / "run_log.json"
    log = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    log.append(entry)
    json.dump(log, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  [run_log] appended v64_softstacker_v62")


if __name__ == "__main__":
    main()

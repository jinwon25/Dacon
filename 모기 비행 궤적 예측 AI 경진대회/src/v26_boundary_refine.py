"""v26_boundary_refine.py — Boundary refinement MLP on top of v23 + v16.

문제: v23 (0.6516) + v24 (0.6535)에서 멈춤. Oracle = 0.6874 → ceiling 격차 0.034.
v23이 못 잡는 sample 중 일부는 1cm~3cm boundary에 있을 가능성 → 작은 보정으로 hit 가능.

설계:
  Input features (sample별):
    - v23 pred (3) + v16 pred (3) + diff (3)
    - Kalman pred (3) + Kalman residual from v23 (3)
    - last position (3) + last velocity (3) + last acceleration (3)
    - speed, accel norm, jerk norm (3)
    - axis-distance v23↔v16 (3)
  Total = ~30 features

  Model: small MLP (30 → 64 → 32 → 3) with tanh × 1cm cap output
    → 예측 = v23_pred + tanh(MLP(feats)) × 0.01  (최대 1cm 보정)

  Loss: combo = euclid + 0.3 × softhit (β=0.002, 1cm threshold sigmoid surrogate)
  5-fold OOF, early stopping on R-Hit.

  강한 prior: 작은 cap (1cm)으로 v23 정상 예측 망가뜨리지 않음.
  Focus: only v23-miss (d > 0.5cm) sample에 weight 2배 (boundary mining).

사용법:
  python scripts/v26_boundary_refine.py --v23-mode fast
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"

DT = 0.040


# ============================================================
# Loss
# ============================================================
def loss_euclid(p, t):
    return torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12).mean()

def loss_softhit(p, t, beta=0.002):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    return torch.sigmoid((d - 0.01) / beta).mean()

def loss_combo(p, t, sample_w=None):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    if sample_w is not None:
        return ((d * sample_w).mean() + 0.3 * (sh * sample_w).mean()) / sample_w.mean()
    return d.mean() + 0.3 * sh.mean()


# ============================================================
# Model: ResMLP with cap
# ============================================================
class BoundaryMLP(nn.Module):
    """v23 prediction에 cap된 작은 Δ 보정."""
    def __init__(self, in_dim, hidden=64, p=0.2, cap_cm=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 3)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.cap = cap_cm / 100.0

    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z))
        delta = torch.tanh(self.head(z)) * self.cap
        return delta


# ============================================================
# Feature builder
# ============================================================
def build_features(X, kalman, v23_pred, v16_pred):
    """sample별 ~30 features."""
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)

    # 최근 3 step velocity 평균 (smoothing)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1)
    v_std = v_recent.std(axis=1)

    diff_v23_v16 = v23_pred - v16_pred
    dist_v23_v16 = np.linalg.norm(diff_v23_v16, axis=-1, keepdims=True)
    res_v23_kal  = v23_pred - kalman

    feats = np.concatenate([
        v23_pred, v16_pred, diff_v23_v16,
        kalman, res_v23_kal,
        last_pos, v, a,
        v_mean, v_std,
        speed, a_norm, dist_v23_v16,
    ], axis=-1).astype(np.float32)
    return feats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23-mode", default="fast", choices=["micro", "fast", "full"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cap-cm", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0,
                          help="v23-miss(d>0.5cm) sample loss weight")
    args = parser.parse_args()
    mode = args.v23_mode

    print("=" * 60)
    print(f"v26 Boundary refinement MLP (base = v23/{mode})")
    print(f"  cap={args.cap_cm}cm, boundary_w={args.boundary_weight}")
    print("=" * 60)

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cpu")

    # --- Load ---
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]

    # v23
    st = np.load(CACHE_DIR / f"v23_state_{mode}.npz")
    oof_A, test_A = st["oof_A"], st["test_A"]
    fold_mask_A = st["fold_mask_A"]
    has_B = bool(st.get("has_B", np.array(False)))
    if has_B:
        oof_B, test_B = st["oof_B"], st["test_B"]
        fold_mask_B = st["fold_mask_B"]
        eval_mask = fold_mask_A & fold_mask_B
        oof_res = (oof_A + oof_B) / 2
        test_res = (test_A + test_B) / 2
    else:
        eval_mask = fold_mask_A
        oof_res = oof_A.copy(); test_res = test_A.copy()
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    oof_v23  = kalman_train + oof_res  * ALPHA
    test_v23 = kalman_test  + test_res * ALPHA

    # v16
    st16 = np.load(V16_PATH)
    oof_v16  = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)

    # --- Distance diagnostics ---
    d_v23 = np.linalg.norm(oof_v23 - y_train, axis=-1)
    hit_v23 = d_v23 <= 0.01
    boundary_mask = (d_v23 > 0.005) & (d_v23 <= 0.03) & eval_mask  # 0.5cm < d ≤ 3cm
    print(f"\n--- v23 OOF distance distribution (covered {eval_mask.sum()}) ---")
    for thr in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        pct = ((d_v23[eval_mask] <= thr).mean()) * 100
        print(f"  d ≤ {thr*100:.1f}cm : {pct:.2f}%")
    print(f"  boundary (0.5cm < d ≤ 3cm): {boundary_mask.sum()} samples → 보정 잠재성")
    print(f"  v23 hit@1cm: {hit_v23[eval_mask].mean():.4f}")

    # --- Build features ---
    print("\n[feat] 학습용 feature 구축…")
    feat_train_full = build_features(X_train, kalman_train, oof_v23, oof_v16)
    feat_test_full  = build_features(X_test, kalman_test, test_v23, test_v16)
    print(f"  shape: train {feat_train_full.shape}, test {feat_test_full.shape}")

    # Target: residual from v23 (작은 보정량 학습)
    target_residual = y_train - oof_v23

    # --- 5-fold OOF MLP ---
    print("\n[train] 5-fold OOF MLP boundary refinement…")
    eval_idx = np.where(eval_mask)[0]
    feat_tr = feat_train_full[eval_idx]
    target_tr = target_residual[eval_idx]
    y_tr = y_train[eval_idx]
    v23_tr = oof_v23[eval_idx]

    # Sample weight: boundary sample 강조
    sample_w_tr = np.ones(len(eval_idx), dtype=np.float32)
    bndry_idx_local = np.where(boundary_mask[eval_idx])[0]
    sample_w_tr[bndry_idx_local] = args.boundary_weight
    print(f"  boundary weight {args.boundary_weight}x applied to {len(bndry_idx_local)} samples")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oof_pred = np.zeros_like(v23_tr)
    test_per_fold = []
    t0 = time.time()

    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        feat_tr_n = sc.transform(feat_tr[tr]).astype(np.float32)
        feat_va_n = sc.transform(feat_tr[va]).astype(np.float32)
        feat_te_n = sc.transform(feat_test_full).astype(np.float32)

        def T(a): return torch.from_numpy(a).to(device)
        x_tr_t = T(feat_tr_n); x_va_t = T(feat_va_n); x_te_t = T(feat_te_n)
        v23_va_t = T(v23_tr[va].astype(np.float32))
        v23_te_t = T(test_v23.astype(np.float32))
        y_va_t = T(y_tr[va].astype(np.float32))
        v23_tr_t = T(v23_tr[tr].astype(np.float32))
        y_train_t = T(y_tr[tr].astype(np.float32))
        sw_tr_t = T(sample_w_tr[tr])

        model = BoundaryMLP(in_dim=feat_tr_n.shape[1], hidden=64, p=0.2,
                              cap_cm=args.cap_cm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

        best_rh, best_state, no_improve = -1.0, None, 0
        n_tr = x_tr_t.shape[0]
        for ep in range(args.max_epochs):
            model.train()
            perm = torch.randperm(n_tr)
            for i in range(0, n_tr, 256):
                idx = perm[i:i+256]
                opt.zero_grad()
                delta = model(x_tr_t[idx])
                pred = v23_tr_t[idx] + delta
                loss = loss_combo(pred, y_train_t[idx], sample_w=sw_tr_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                delta_va = model(x_va_t)
                pred_va = (v23_va_t + delta_va).cpu().numpy()
            rh = float((np.linalg.norm(pred_va - y_tr[va], axis=-1) <= 0.01).mean())
            if rh > best_rh:
                best_rh = rh
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience: break

            if ep == 0 or (ep+1) % 10 == 0:
                print(f"  fold{fi+1} ep{ep+1:3d}: rhit={rh:.4f} (best {best_rh:.4f})  "
                      f"[{(time.time()-t0)/60:.1f}m]", flush=True)

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pred_va = (v23_va_t + model(x_va_t)).cpu().numpy()
            pred_te = (v23_te_t + model(x_te_t)).cpu().numpy()
        oof_pred[va] = pred_va
        test_per_fold.append(pred_te)
        rh_fold = float((np.linalg.norm(pred_va - y_tr[va], axis=-1) <= 0.01).mean())
        rh_v23_va = float((np.linalg.norm(v23_tr[va] - y_tr[va], axis=-1) <= 0.01).mean())
        print(f"  ★ fold{fi+1}: rhit={rh_fold:.4f}  (vs v23 alone {rh_v23_va:.4f}, Δ {rh_fold - rh_v23_va:+.4f})  ({(time.time()-t0)/60:.1f}m)")
        del model; gc.collect()

    rh_v26 = float((np.linalg.norm(oof_pred - y_tr, axis=-1) <= 0.01).mean())
    rh_v23_full = float((np.linalg.norm(v23_tr - y_tr, axis=-1) <= 0.01).mean())
    print(f"\n=== v26 OOF R-Hit (covered): {rh_v26:.4f}  (vs v23 alone {rh_v23_full:.4f}, Δ {rh_v26 - rh_v23_full:+.4f}) ===")

    # --- Δ 통계 ---
    delta_oof = oof_pred - v23_tr
    delta_norm = np.linalg.norm(delta_oof, axis=-1)
    print(f"\n[Δ] mean={delta_norm.mean()*100:.3f}cm, median={np.median(delta_norm)*100:.3f}cm, "
          f"p90={np.percentile(delta_norm, 90)*100:.3f}cm, p99={np.percentile(delta_norm, 99)*100:.3f}cm")

    # --- v26 + v16 ensemble (선택지) ---
    print("\n=== v26 + v16 ensemble candidates (OOF) ===")
    oof_v16_em = oof_v16[eval_idx]
    candidates = {
        "v26 alone": oof_pred,
        "simple_avg(v26, v16)": (oof_pred + oof_v16_em) / 2,
    }
    # global α grid
    best_a, best_r = 1.0, rh_v26
    for a in np.linspace(0.0, 1.0, 21):
        ens = a * oof_pred + (1-a) * oof_v16_em
        r = float((np.linalg.norm(ens - y_tr, axis=-1) <= 0.01).mean())
        if r > best_r: best_r, best_a = r, a
    candidates[f"global_α(v26)={best_a:.2f}"] = best_a * oof_pred + (1-best_a) * oof_v16_em

    for name, ens in candidates.items():
        r = float((np.linalg.norm(ens - y_tr, axis=-1) <= 0.01).mean())
        print(f"  {name:<30}: {r:.4f}")

    best_name = max(candidates, key=lambda k: float((np.linalg.norm(candidates[k] - y_tr, axis=-1) <= 0.01).mean()))
    best_rh = float((np.linalg.norm(candidates[best_name] - y_tr, axis=-1) <= 0.01).mean())
    print(f"\n★ Best: {best_name}  {best_rh:.4f}")

    # --- Test prediction ---
    test_v26 = np.mean(test_per_fold, axis=0)  # 5-fold average
    if best_name == "v26 alone":
        test_final = test_v26
    elif best_name == "simple_avg(v26, v16)":
        test_final = (test_v26 + test_v16) / 2
    else:  # global_α
        test_final = best_a * test_v26 + (1 - best_a) * test_v16

    # --- Submission ---
    assert test_final.shape == (10000, 3) and np.isfinite(test_final).all()
    out_csv = DATA_DIR / f"submission_v26_cpu_{mode}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_final[:,0], "y": test_final[:,1], "z": test_final[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n[submission] {out_csv}")

    # --- run_log ---
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": f"v26_cpu_{mode}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Boundary refinement MLP on v23 + v16 features (capped Δ)",
        "v23_mode": mode,
        "v23_oof_rhit": rh_v23_full,
        "v26_oof_rhit": rh_v26,
        "delta_oof_v26_v23": rh_v26 - rh_v23_full,
        "cap_cm": args.cap_cm,
        "boundary_weight": args.boundary_weight,
        "delta_stats": {
            "mean_cm": float(delta_norm.mean()*100),
            "p90_cm": float(np.percentile(delta_norm,90)*100),
            "p99_cm": float(np.percentile(delta_norm,99)*100),
        },
        "best_ensemble": best_name,
        "best_oof_rhit": float(best_rh),
        "covered_rows": int(eval_mask.sum()),
        "submission_path": str(out_csv),
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

    print("\n" + "=" * 60)
    print(f"v26 boundary MLP ({mode}) 완료")
    print("=" * 60)
    print(f"  v23 alone : {rh_v23_full:.4f}")
    print(f"  v26 alone : {rh_v26:.4f}  (Δ {rh_v26 - rh_v23_full:+.4f})")
    print(f"  best ens  : {best_rh:.4f}  ({best_name})")
    print(f"  CSV       : {out_csv}")


if __name__ == "__main__":
    main()

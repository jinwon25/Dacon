"""v39_pseudo_label.py — v35 confident pseudo-labels로 v35 재학습.

v5 실패 분석:
  - v5는 모든 test prediction을 label로 사용 (confidence 미고려)
  - Iterative round (round 2) → 회귀
v39 차이:
  - **Round 1 only** (iterative 금지)
  - **Top 20% confident만** (uncertainty score 기반)
  - confidence proxy = v35와 best gate의 거리 + v35 자체 score

이론적 근거:
  - Adv AUC 0.5854 (mild covariate shift) — pseudo-labeling이 test distribution을 train으로 흡수
  - Confident sample만 사용 → noise 회피
  - Bestfitting Kaggle 우승 솔루션 표준 패턴

학습:
  - v35 boundary MLP 그대로 + train에 confident test pseudo 추가 (~2000개)
  - 5-fold OOF × 3-seed (test pseudo는 fold에 영향 없음, 항상 train side)
  - 학습 시간: ~5~10분

출력: submission_v39_cpu.csv
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


def build_features(X, kalman, v30_pred, gate_pred, v16_pred):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cap-cm", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pseudo-pct", type=float, default=0.20,
                          help="Top X%% confident test samples to use as pseudo")
    parser.add_argument("--pseudo-weight", type=float, default=0.5,
                          help="Loss weight for pseudo samples (0.5 < 1.0 보수적)")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v39 pseudo-labeling round 1: top {args.pseudo_pct*100:.0f}% confident test")
    print(f"  pseudo loss weight = {args.pseudo_weight} (보수)")
    print("=" * 60)

    # Load
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

    # v35 OOF + test (pseudo-labeling source)
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    oof_v35 = st35["oof_v35"]; test_v35 = st35["test_v35"]
    rh_v35 = float((np.linalg.norm(oof_v35 - y_train, axis=-1) <= 0.01).mean())
    print(f"v35 base OOF: {rh_v35:.4f} (LB 0.6874)")

    # --- Confidence score for test samples ---
    # v35 vs best gate prediction 거리 작을수록 confident
    dist_v35_gate = np.linalg.norm(test_v35 - test_gate, axis=-1)
    # v35 vs Kalman 거리 작을수록 (Kalman과 일치 = 안정적 motion) confident
    dist_v35_kal = np.linalg.norm(test_v35 - kalman_test, axis=-1)
    # 종합 confidence: 두 distance 모두 작아야 confident
    # rank-based combination
    rank_vg = np.argsort(np.argsort(dist_v35_gate)) / len(dist_v35_gate)
    rank_vk = np.argsort(np.argsort(dist_v35_kal)) / len(dist_v35_kal)
    confidence_score = 1.0 - (rank_vg + rank_vk) / 2  # 1=most confident

    n_pseudo = int(args.pseudo_pct * len(test_v35))
    pseudo_idx = np.argsort(-confidence_score)[:n_pseudo]
    print(f"Top {args.pseudo_pct*100:.0f}% confident: {n_pseudo} test samples")
    print(f"  confidence threshold: {confidence_score[pseudo_idx].min():.3f}")
    print(f"  v35-gate dist range: {dist_v35_gate[pseudo_idx].min()*100:.2f}~{dist_v35_gate[pseudo_idx].max()*100:.2f}cm")

    # Pseudo-labels = v35 test prediction
    X_pseudo = X_test[pseudo_idx]
    y_pseudo = test_v35[pseudo_idx]
    v30_pseudo = test_v30[pseudo_idx]
    gate_pseudo = test_gate[pseudo_idx]
    v16_pseudo = test_v16[pseudo_idx]
    kalman_pseudo = kalman_test[pseudo_idx]

    # --- Build features for train + pseudo + test (eval) ---
    feat_train = build_features(X_train, kalman_train, oof_v30, gate_oof, oof_v16)
    feat_pseudo = build_features(X_pseudo, kalman_pseudo, v30_pseudo, gate_pseudo, v16_pseudo)
    feat_test = build_features(X_test, kalman_test, test_v30, test_gate, test_v16)
    print(f"feat: train={feat_train.shape}, pseudo={feat_pseudo.shape}, test={feat_test.shape}")

    # --- 5-fold OOF + multi-seed ---
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    device = torch.device("cpu")

    # boundary weight on train
    d_v30 = np.linalg.norm(oof_v30 - y_train, axis=-1)
    boundary_mask = (d_v30 > 0.005) & (d_v30 <= 0.03)
    sw_train = np.ones(len(y_train), dtype=np.float32)
    sw_train[boundary_mask] = args.boundary_weight
    # boundary on pseudo (v35-gate distance가 0.5cm~3cm)
    pseudo_boundary = (dist_v35_gate[pseudo_idx] > 0.005) & (dist_v35_gate[pseudo_idx] <= 0.03)
    sw_pseudo = np.ones(len(y_pseudo), dtype=np.float32) * args.pseudo_weight
    sw_pseudo[pseudo_boundary] *= args.boundary_weight

    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        oof_s = np.zeros_like(oof_v30); test_per_fold = []
        t0 = time.time()
        for fi, (tr, va) in enumerate(kf.split(feat_train)):
            # Standardize using train fold + pseudo
            feat_combined = np.concatenate([feat_train[tr], feat_pseudo], axis=0)
            sc = StandardScaler().fit(feat_combined)
            feat_tr_n = sc.transform(feat_train[tr]).astype(np.float32)
            feat_pseudo_n = sc.transform(feat_pseudo).astype(np.float32)
            feat_va_n = sc.transform(feat_train[va]).astype(np.float32)
            feat_te_n = sc.transform(feat_test).astype(np.float32)

            def T(a): return torch.from_numpy(a).to(device)
            x_tr, x_pseudo_t, x_va, x_te = T(feat_tr_n), T(feat_pseudo_n), T(feat_va_n), T(feat_te_n)
            v30_tr_t = T(oof_v30[tr].astype(np.float32))
            v30_pseudo_t = T(v30_pseudo.astype(np.float32))
            v30_va_t = T(oof_v30[va].astype(np.float32))
            v30_te_t = T(test_v30.astype(np.float32))
            y_tr_t = T(y_train[tr].astype(np.float32))
            y_pseudo_t = T(y_pseudo.astype(np.float32))
            sw_tr_t = T(sw_train[tr])
            sw_pseudo_t = T(sw_pseudo)

            torch.manual_seed(s); np.random.seed(s)
            model = BoundaryMLP(in_dim=feat_tr_n.shape[1], hidden=args.hidden, p=0.2,
                                  cap_cm=args.cap_cm).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

            best_rh, best_state, no_improve = -1.0, None, 0
            n_tr = x_tr.shape[0]
            n_pseudo = x_pseudo_t.shape[0]
            for ep in range(args.max_epochs):
                model.train()
                # Train + pseudo 합쳐서 shuffle
                perm_tr = torch.randperm(n_tr)
                perm_ps = torch.randperm(n_pseudo)
                # Per batch: real samples + small chunk of pseudo
                for i in range(0, n_tr, 256):
                    idx_tr = perm_tr[i:i+256]
                    # Pseudo chunk size proportional
                    n_ps_chunk = max(1, int(256 * n_pseudo / n_tr))
                    ps_start = (i // 256) * n_ps_chunk % n_pseudo
                    idx_ps = perm_ps[ps_start:ps_start + n_ps_chunk]
                    opt.zero_grad()
                    # Real samples loss
                    pred_real = v30_tr_t[idx_tr] + model(x_tr[idx_tr])
                    loss_real = loss_combo(pred_real, y_tr_t[idx_tr], sw=sw_tr_t[idx_tr])
                    # Pseudo samples loss
                    if len(idx_ps) > 0:
                        pred_ps = v30_pseudo_t[idx_ps] + model(x_pseudo_t[idx_ps])
                        loss_ps = loss_combo(pred_ps, y_pseudo_t[idx_ps], sw=sw_pseudo_t[idx_ps])
                        loss = loss_real + 0.3 * loss_ps  # pseudo 영향 보수
                    else:
                        loss = loss_real
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    pred_va = (v30_va_t + model(x_va)).cpu().numpy()
                rh = float((np.linalg.norm(pred_va - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= args.patience: break

            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                oof_s[va] = (v30_va_t + model(x_va)).cpu().numpy()
                test_per_fold.append((v30_te_t + model(x_te)).cpu().numpy())
            rh_fold = float((np.linalg.norm(oof_s[va] - y_train[va], axis=-1) <= 0.01).mean())
            print(f"  seed{s} fold{fi+1}: rhit={rh_fold:.4f} ({(time.time()-t0)/60:.1f}m)")
            del model; gc.collect()

        oofs.append(oof_s); tests.append(np.mean(test_per_fold, axis=0))
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v39 = np.mean(oofs, axis=0)
    test_v39 = np.mean(tests, axis=0)
    rh_v39 = float((np.linalg.norm(oof_v39 - y_train, axis=-1) <= 0.01).mean())

    print(f"\n=== v39 결과 ===")
    print(f"  v35 base OOF: {rh_v35:.4f} (LB 0.6874)")
    print(f"  v39 OOF: {rh_v39:.4f}  (Δ vs v35: {rh_v39 - rh_v35:+.4f})")

    # Save
    np.savez(CACHE_DIR / "v39_state.npz", oof_v39=oof_v39, test_v39=test_v39, rh_v39=rh_v39)
    out_csv = DATA_DIR / "submission_v39_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v39[:,0], "y": test_v39[:,1], "z": test_v39[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v39_pseudo_label",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"v35 framework + top {args.pseudo_pct*100:.0f}% confident pseudo-labels",
        "n_pseudo": int(n_pseudo) if 'n_pseudo' in dir() else 0,
        "pseudo_weight": args.pseudo_weight,
        "v35_oof": float(rh_v35),
        "v39_oof": float(rh_v39),
        "delta_v39_v35": float(rh_v39 - rh_v35),
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


if __name__ == "__main__":
    main()

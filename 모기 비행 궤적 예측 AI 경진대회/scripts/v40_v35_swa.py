"""v40_v35_swa.py — v35 framework + SWA (Stochastic Weight Averaging).

Izmailov et al. NeurIPS 2018. 학습 후반 30% epoch에서 매 N epoch마다 weight 평균.
- Loss landscape flat region으로 수렴 → generalization 좋음
- Kaggle 우승 솔루션 표준 패턴

구현:
  - 일반 학습으로 best R-Hit epoch 저장 (early stopping)
  - 그 후 추가 epoch에 SWA collect (cyclic LR 또는 high constant LR)
  - 최종: SWA model vs best epoch model 둘 다 OOF 평가 → best 선택

학습 시간: v35 (3 seed × 5 fold = 15 trainings × 2분) + SWA epochs ≈ 10~15분
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import glob
import json
import os
import time
from copy import deepcopy
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


def train_one_seed_swa(seed, feat_tr, y_tr, v30_tr, feat_test, test_v30, sample_w_tr, args, kf):
    """학습 후반 SWA collect → SWA model 사용."""
    from torch.optim.swa_utils import AveragedModel, SWALR
    device = torch.device("cpu")
    oof_pred = np.zeros_like(v30_tr)
    test_per_fold = []
    swa_oof_pred = np.zeros_like(v30_tr)
    swa_test_per_fold = []
    t0 = time.time()

    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        feat_tr_n = sc.transform(feat_tr[tr]).astype(np.float32)
        feat_va_n = sc.transform(feat_tr[va]).astype(np.float32)
        feat_te_n = sc.transform(feat_test).astype(np.float32)

        def T(a): return torch.from_numpy(a).to(device)
        x_tr, x_va, x_te = T(feat_tr_n), T(feat_va_n), T(feat_te_n)
        v30_tr_t = T(v30_tr[tr].astype(np.float32))
        v30_va_t = T(v30_tr[va].astype(np.float32))
        v30_te_t = T(test_v30.astype(np.float32))
        y_tr_t = T(y_tr[tr].astype(np.float32))
        sw_t = T(sample_w_tr[tr])

        torch.manual_seed(seed); np.random.seed(seed)
        model = BoundaryMLP(in_dim=feat_tr_n.shape[1], hidden=args.hidden, p=0.2,
                              cap_cm=args.cap_cm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

        # SWA tracking
        swa_model = AveragedModel(model)
        swa_start_ep = int(args.swa_start_frac * args.max_epochs)  # 70% of epochs
        swa_scheduler = SWALR(opt, swa_lr=args.swa_lr, anneal_epochs=5)

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

            if ep >= swa_start_ep:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
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
            if no_improve >= args.patience and ep < swa_start_ep:
                # SWA 시작 전이면 early stop
                break

        # Best epoch model
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_pred[va] = (v30_va_t + model(x_va)).cpu().numpy()
            test_per_fold.append((v30_te_t + model(x_te)).cpu().numpy())

        # SWA model (averaged)
        # SWA model uses BN stats so we need to update BN once (none in our MLP, skip)
        swa_model.eval()
        with torch.no_grad():
            swa_oof_pred[va] = (v30_va_t + swa_model(x_va)).cpu().numpy()
            swa_test_per_fold.append((v30_te_t + swa_model(x_te)).cpu().numpy())

        rh_best = float((np.linalg.norm(oof_pred[va] - y_tr[va], axis=-1) <= 0.01).mean())
        rh_swa = float((np.linalg.norm(swa_oof_pred[va] - y_tr[va], axis=-1) <= 0.01).mean())
        print(f"  seed{seed} fold{fi+1}: best={rh_best:.4f}, SWA={rh_swa:.4f}  ({(time.time()-t0)/60:.1f}m)")
        del model, swa_model; gc.collect()

    return (oof_pred, np.mean(test_per_fold, axis=0),
             swa_oof_pred, np.mean(swa_test_per_fold, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cap-cm", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--swa-start-frac", type=float, default=0.7,
                          help="Start SWA collection at this fraction of total epochs")
    parser.add_argument("--swa-lr", type=float, default=5e-4)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v40 = v35 framework + SWA (start at {args.swa_start_frac*100:.0f}% epochs)")
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
    oof_v30 = kalman_train + (st30["oof_A"] + st30["oof_B"])/2 * ALPHA
    test_v30 = kalman_test + (st30["test_A"] + st30["test_B"])/2 * ALPHA

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    test_gate = df_best[["x","y","z"]].values.astype(np.float64)

    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)

    rh_v30 = float((np.linalg.norm(oof_v30 - y_train, axis=-1) <= 0.01).mean())
    print(f"v30 base OOF: {rh_v30:.4f}")

    d_v30 = np.linalg.norm(oof_v30 - y_train, axis=-1)
    boundary_mask = (d_v30 > 0.005) & (d_v30 <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight

    feat_train = build_features(X_train, kalman_train, oof_v30, gate_oof, oof_v16)
    feat_test  = build_features(X_test, kalman_test, test_v30, test_gate, test_v16)
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)

    oofs_best, tests_best, oofs_swa, tests_swa = [], [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        oof_b, test_b, oof_s, test_s = train_one_seed_swa(
            s, feat_train, y_train, oof_v30, feat_test, test_v30, sample_w, args, kf)
        oofs_best.append(oof_b); tests_best.append(test_b)
        oofs_swa.append(oof_s); tests_swa.append(test_s)

    oof_best_avg = np.mean(oofs_best, axis=0)
    test_best_avg = np.mean(tests_best, axis=0)
    oof_swa_avg = np.mean(oofs_swa, axis=0)
    test_swa_avg = np.mean(tests_swa, axis=0)

    rh_best = float((np.linalg.norm(oof_best_avg - y_train, axis=-1) <= 0.01).mean())
    rh_swa = float((np.linalg.norm(oof_swa_avg - y_train, axis=-1) <= 0.01).mean())

    # Blend best + SWA
    print(f"\n=== Blend best epoch model + SWA model ===")
    best_alpha = 0.5; best_blend = 0
    for a in np.linspace(0.0, 1.0, 11):
        ens = a * oof_best_avg + (1-a) * oof_swa_avg
        r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
        if r > best_blend:
            best_blend = r; best_alpha = a
    print(f"  best epoch alone: {rh_best:.4f}")
    print(f"  SWA alone: {rh_swa:.4f}")
    print(f"  blend (best α={best_alpha:.2f}): {best_blend:.4f}")

    # Choose best
    if best_blend > max(rh_best, rh_swa):
        chosen_oof = best_alpha * oof_best_avg + (1-best_alpha) * oof_swa_avg
        chosen_test = best_alpha * test_best_avg + (1-best_alpha) * test_swa_avg
        chosen_name = f"blend_α={best_alpha:.2f}"
        chosen_rh = best_blend
    elif rh_swa > rh_best:
        chosen_oof, chosen_test = oof_swa_avg, test_swa_avg
        chosen_name = "swa_alone"; chosen_rh = rh_swa
    else:
        chosen_oof, chosen_test = oof_best_avg, test_best_avg
        chosen_name = "best_epoch_alone"; chosen_rh = rh_best

    # Reference comparison
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    rh_v35 = float(st35["rh_v35"])
    print(f"\n  ★ v35 baseline OOF: {rh_v35:.4f} (LB 0.6874)")
    print(f"  ★ v40 best: {chosen_rh:.4f} ({chosen_name}, Δ vs v35: {chosen_rh - rh_v35:+.4f})")

    # Save
    np.savez(CACHE_DIR / "v40_state.npz",
              oof_best=oof_best_avg, test_best=test_best_avg, rh_best=rh_best,
              oof_swa=oof_swa_avg, test_swa=test_swa_avg, rh_swa=rh_swa,
              chosen_oof=chosen_oof, chosen_test=chosen_test, chosen_name=chosen_name,
              chosen_rh=chosen_rh)

    out_csv = DATA_DIR / "submission_v40_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": chosen_test[:,0], "y": chosen_test[:,1], "z": chosen_test[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv}")

    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v40_v35_swa",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "v35 framework + SWA (Izmailov 2018)",
        "v35_oof": float(rh_v35),
        "best_epoch_oof": float(rh_best),
        "swa_oof": float(rh_swa),
        "blend_oof": float(best_blend),
        "chosen_name": chosen_name,
        "chosen_oof": float(chosen_rh),
        "delta_vs_v35": float(chosen_rh - rh_v35),
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

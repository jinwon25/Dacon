"""v68_selector.py — Per-sample best-model classifier (hard MoE-style).

진단:
  - SoftStacker가 v62/v65 paradigm weight 거의 0 → mean blend는 약한 모델 거부
  - oracle pool 0.7355 (9m + v65 hard) vs blend 0.6748 = 5% gap 추출 안 됨
  - 핵심: sample-conditional hard routing 필요

설계:
  1. Per-sample "best model" label = OOF에서 가장 y에 가까운 model index (argmin distance)
  2. Classifier: input 11-step + scalar features → softmax(K) over models
  3. Loss: CrossEntropy (smoothed)
  4. Inference:
     - hard: argmax → 그 모델의 prediction
     - top-2 soft: top-2 prob weighted blend
  5. 5-fold × 3-seed
"""
from __future__ import annotations

import argparse, datetime as _dt, gc, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import yaw_angle, inverse_rotate_xy

PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"
BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rhit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


class Selector(nn.Module):
    def __init__(self, in_dim, K, hidden=128, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, K)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)

    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z)); z = self.drop(z)
        return self.head(z)  # logits


def build_features_seq(X, model_preds):
    """sequence + scalar + per-model anchor info"""
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    speed_seq = np.linalg.norm(np.diff(X, axis=1) / DT, axis=-1)  # (N, 10)
    speed_mean = speed_seq.mean(axis=1, keepdims=True)
    speed_max = speed_seq.max(axis=1, keepdims=True)
    speed_std_seq = speed_seq.std(axis=1, keepdims=True)
    # turn
    v_seq = np.diff(X, axis=1) / DT
    v_norm = v_seq / (np.linalg.norm(v_seq, axis=-1, keepdims=True) + 1e-9)
    cos_turn = (v_norm[:, :-1] * v_norm[:, 1:]).sum(axis=-1)  # (N, 9)
    turn_mean = cos_turn.mean(axis=1, keepdims=True)
    turn_min = cos_turn.min(axis=1, keepdims=True)
    feats = [last_pos, v, a, v_mean, v_std, speed, a_norm,
             speed_mean, speed_max, speed_std_seq, turn_mean, turn_min]
    # per-model relative offsets
    mean_pred = np.mean(model_preds, axis=0)
    for p in model_preds:
        feats.append(p - last_pos)              # absolute displacement per model
        feats.append(p - mean_pred)             # divergence from mean
    return np.concatenate(feats, axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--use-v62", action="store_true")
    parser.add_argument("--v65-mode", default="hard", choices=["soft", "hard", "both"])
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v68 Selector — per-sample best model classifier")
    print(f"  v65: {args.v65_mode}, v62: {args.use_v62}, "
          f"hid {args.hidden}, drop {args.dropout}, smooth {args.label_smooth}")
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

    # base 9 (skip v32 since OOF=0, useless)
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    v30A_o = kalman_train + st30["oof_A"] * ALPHA; v30A_t = kalman_test + st30["test_A"] * ALPHA
    v30B_o = kalman_train + st30["oof_B"] * ALPHA; v30B_t = kalman_test + st30["test_B"] * ALPHA
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v35o = st35["oof_v35"].astype(np.float64); v35t = st35["test_v35"].astype(np.float64)
    st41 = np.load(CACHE_DIR / "v41_state.npz")
    v41A_o = kalman_train + st41["oof_A"] * ALPHA; v41A_t = kalman_test + st41["test_A"] * ALPHA
    v41B_o = kalman_train + st41["oof_B"] * ALPHA; v41B_t = kalman_test + st41["test_B"] * ALPHA
    st44 = np.load(CACHE_DIR / "v44_state.npz")
    v44o = st44["oof_v44"].astype(np.float64); v44t = st44["test_v44"].astype(np.float64)
    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        m = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([m[i] for i in best_ids]); gate_o = gate_o[perm]
    gate_t = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    st39 = np.load(CACHE_DIR / "v39_state.npz")
    v39o = st39["oof_v39"].astype(np.float64); v39t = st39["test_v39"].astype(np.float64)

    models = [v30A_o, v30B_o, v35o, v41A_o, v41B_o, v44o, gate_o, v39o]
    tests  = [v30A_t, v30B_t, v35t, v41A_t, v41B_t, v44t, gate_t, v39t]
    names  = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate", "v39"]

    # v65
    st65 = np.load(CACHE_DIR / "v65_K64_state.npz")
    if args.v65_mode in ("soft", "both"):
        models.append(st65["oof_soft"].astype(np.float64))
        tests.append(st65["test_soft"].astype(np.float64))
        names.append("v65s")
    if args.v65_mode in ("hard", "both"):
        models.append(st65["oof_hard"].astype(np.float64))
        tests.append(st65["test_hard"].astype(np.float64))
        names.append("v65h")

    # v62 (optional)
    if args.use_v62:
        st62 = np.load(CACHE_DIR / "v62_state.npz")
        v62_or = (st62["oof_A"] + st62["oof_B"]) / 2
        v62_tr = (st62["test_A"] + st62["test_B"]) / 2
        v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
        v_last_test = (X_test[:, -1] - X_test[:, -2]) / DT
        th_tr = yaw_angle(v_last_train); th_te = yaw_angle(v_last_test)
        v62o = st62["kalman_train"] + inverse_rotate_xy(v62_or, th_tr)
        v62t = st62["kalman_test"] + inverse_rotate_xy(v62_tr, th_te)
        models.append(v62o); tests.append(v62t); names.append("v62")

    K = len(names)
    print("\n=== Pool ===")
    for n, p in zip(names, models): print(f"  {n}: {rhit(p, y_train):.4f}")
    hits = np.stack([np.linalg.norm(p - y_train, axis=-1) <= 0.01 for p in models])
    oracle = hits.any(axis=0).mean()
    print(f"  Oracle ({K}-model): {oracle:.4f}")

    # per-sample best model label = argmin distance
    dists = np.stack([np.linalg.norm(p - y_train, axis=-1) for p in models])  # (K, N)
    best_idx = dists.argmin(axis=0)  # (N,)
    print(f"\n  best-model distribution:")
    for i, n in enumerate(names):
        cnt = (best_idx == i).sum()
        hit_rate = hits[i][best_idx == i].mean() if cnt > 0 else 0
        print(f"    {n:6s}: {cnt:5d} ({cnt/len(best_idx)*100:5.1f}%)  hit-rate in own bucket: {hit_rate:.4f}")
    # oracle if perfect selector
    perfect_sel = np.zeros((len(y_train), 3))
    for i in range(len(y_train)): perfect_sel[i] = models[best_idx[i]][i]
    print(f"  Perfect selector OOF R-Hit: {rhit(perfect_sel, y_train):.4f}  (= 1-cm oracle)")

    # features
    feat_tr = build_features_seq(X_train, models)
    feat_te = build_features_seq(X_test, tests)
    preds_tr = np.stack(models, axis=1).astype(np.float64)
    preds_te = np.stack(tests, axis=1).astype(np.float64)
    print(f"  feat dim: {feat_tr.shape[1]}, preds: {preds_tr.shape}")

    # 5-fold × n-seeds
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    N = feat_tr.shape[0]
    oof_pred_hard = np.zeros((N, 3))
    oof_pred_top2 = np.zeros((N, 3))
    test_preds_hard, test_preds_top2 = [], []
    sel_idx_oof = np.zeros(N, dtype=np.int64)
    prob_oof = np.zeros((N, K), dtype=np.float32)
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        ft = lambda a: sc.transform(a).astype(np.float32)
        def T(a): return torch.from_numpy(a)
        x_tr, x_va, x_te = T(ft(feat_tr[tr])), T(ft(feat_tr[va])), T(ft(feat_te))
        p_tr = T(preds_tr[tr].astype(np.float32))
        p_va = T(preds_tr[va].astype(np.float32))
        p_te = T(preds_te.astype(np.float32))
        lab_tr = T(best_idx[tr])
        y_va = y_train[va]

        seed_probs_va = []; seed_probs_te = []
        for s in range(args.n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = Selector(in_dim=x_tr.shape[1], K=K,
                             hidden=args.hidden, p=args.dropout)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
            best_rh, best_st, no_imp = -1.0, None, 0
            n_tr = x_tr.shape[0]
            for ep in range(args.max_epochs):
                model.train()
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr, 512):
                    idx = perm[i:i+512]
                    opt.zero_grad()
                    logits = model(x_tr[idx])
                    loss = F.cross_entropy(logits, lab_tr[idx], label_smoothing=args.label_smooth)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                # eval = hard selector
                model.eval()
                with torch.no_grad():
                    logits_va = model(x_va)
                    sel = logits_va.argmax(dim=-1).numpy()
                pred_hard = preds_tr[va][np.arange(len(va)), sel]
                rh = rhit(pred_hard, y_va)
                if rh > best_rh:
                    best_rh, best_st, no_imp = rh, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
                else: no_imp += 1
                if no_imp >= args.patience: break
            model.load_state_dict(best_st); model.eval()
            with torch.no_grad():
                pv = F.softmax(model(x_va), dim=-1).numpy()
                pt = F.softmax(model(x_te), dim=-1).numpy()
            seed_probs_va.append(pv); seed_probs_te.append(pt)
        prob_va = np.mean(seed_probs_va, axis=0)  # (Nv, K)
        prob_te = np.mean(seed_probs_te, axis=0)  # (Nte, K)

        # hard
        sel_va = prob_va.argmax(axis=-1)
        sel_te = prob_te.argmax(axis=-1)
        oof_pred_hard[va] = preds_tr[va][np.arange(len(va)), sel_va]
        test_preds_hard.append(preds_te[np.arange(len(preds_te)), sel_te])
        # top-2 soft (renormalize top-2 prob)
        top2_idx = np.argsort(-prob_va, axis=-1)[:, :2]  # (Nv, 2)
        top2_p = np.take_along_axis(prob_va, top2_idx, axis=-1)
        top2_p = top2_p / top2_p.sum(axis=-1, keepdims=True)
        oof_pred_top2[va] = sum(top2_p[:, k:k+1] * preds_tr[va][np.arange(len(va)), top2_idx[:, k]] for k in range(2))
        top2_idx_te = np.argsort(-prob_te, axis=-1)[:, :2]
        top2_p_te = np.take_along_axis(prob_te, top2_idx_te, axis=-1)
        top2_p_te = top2_p_te / top2_p_te.sum(axis=-1, keepdims=True)
        test_preds_top2.append(sum(top2_p_te[:, k:k+1] * preds_te[np.arange(len(preds_te)), top2_idx_te[:, k]] for k in range(2)))

        sel_idx_oof[va] = sel_va
        prob_oof[va] = prob_va
        print(f"  fold{fi+1}: hard={rhit(oof_pred_hard[va], y_va):.4f}  top2={rhit(oof_pred_top2[va], y_va):.4f}  "
              f"sel acc={(sel_va == best_idx[va]).mean():.3f}  ({(time.time()-t0)/60:.1f}m)")
        gc.collect()

    test_hard = np.mean(test_preds_hard, axis=0)
    test_top2 = np.mean(test_preds_top2, axis=0)
    rh_hard = rhit(oof_pred_hard, y_train)
    rh_top2 = rhit(oof_pred_top2, y_train)

    base_o = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["oof_v48"] \
           + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["oof_v46"] \
           + 0.18 * v35o
    base_t = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["test_v48"] \
           + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["test_v46"] \
           + 0.18 * v35t
    rh_base = rhit(base_o, y_train)

    print(f"\n=== v68 결과 ===")
    print(f"  Pool oracle ({K}-m): {oracle:.4f}")
    print(f"  v48 3-way base    : {rh_base:.4f}")
    print(f"  v68 hard          : {rh_hard:.4f}  (Δ vs base: {rh_hard - rh_base:+.4f})")
    print(f"  v68 top-2 soft    : {rh_top2:.4f}  (Δ vs base: {rh_top2 - rh_base:+.4f})")
    print(f"\n  selector pick distribution:")
    for i, n in enumerate(names):
        cnt = (sel_idx_oof == i).sum()
        print(f"    {n:6s}: picked {cnt:5d} ({cnt/N*100:5.1f}%)")

    # hybrid with base
    for label, ens_o, ens_t in [("hard", oof_pred_hard, test_hard), ("top2", oof_pred_top2, test_top2)]:
        best_w, best_r = 1.0, rh_base
        for w in np.linspace(0, 1, 21):
            r = rhit(w * base_o + (1 - w) * ens_o, y_train)
            if r > best_r: best_r, best_w = r, w
        print(f"  hybrid base×w + v68_{label}: best w={best_w:.2f} → OOF {best_r:.4f}  (Δ {best_r - rh_base:+.4f})")
        if best_r > rh_base:
            hyb_t = best_w * base_t + (1 - best_w) * ens_t
            csv = DATA_DIR / f"submission_v68_{label}_hybrid_basex{best_w:.2f}.csv"
            pd.DataFrame({"id": sub["id"], "x": hyb_t[:,0], "y": hyb_t[:,1], "z": hyb_t[:,2]}).to_csv(csv, index=False)
            print(f"    [submission] {csv.name}")

    # save state + alone submissions
    np.savez(CACHE_DIR / "v68_state.npz",
             oof_hard=oof_pred_hard, oof_top2=oof_pred_top2,
             test_hard=test_hard, test_top2=test_top2,
             rh_hard=rh_hard, rh_top2=rh_top2,
             sel_idx_oof=sel_idx_oof, prob_oof=prob_oof, names=np.array(names))
    pd.DataFrame({"id": sub["id"], "x": test_hard[:,0], "y": test_hard[:,1], "z": test_hard[:,2]}).to_csv(
        DATA_DIR / "submission_v68_hard.csv", index=False)
    pd.DataFrame({"id": sub["id"], "x": test_top2[:,0], "y": test_top2[:,1], "z": test_top2[:,2]}).to_csv(
        DATA_DIR / "submission_v68_top2.csv", index=False)

    entry = {
        "version": "v68_selector", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "per-sample best-model classifier (CE) → hard/top-2 selection",
        "K": K, "models": names, "oracle": float(oracle),
        "rh_hard": rh_hard, "rh_top2": rh_top2, "rh_base": rh_base,
        "delta_hard_vs_base": rh_hard - rh_base, "delta_top2_vs_base": rh_top2 - rh_base,
    }
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

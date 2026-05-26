"""v67_softstacker_v65.py — 11-model SoftStacker (v48 9m + v62 + v65 soft).

v66 진단: 9m + v65 soft oracle +0.0091, hard +0.0140.
linear blend 실패 → SoftStacker로 sample-wise blending 시도.
v68 MoE top-1 router는 별도 카드.
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
        self.act = nn.GELU(); self.drop = nn.Dropout(p); self.temp = temp

    def forward(self, x, preds):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z)); z = self.drop(z)
        logits = self.head(z) / self.temp
        w = F.softmax(logits, dim=-1)
        return (w.unsqueeze(-1) * preds).sum(dim=1), w


def build_features(X, kalman, model_preds):
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    feats = [last_pos, v, a, v_mean, v_std, speed, a_norm, kalman]
    for p in model_preds: feats.append(p - kalman)
    key = list(range(min(7, len(model_preds))))
    for i_pos, i in enumerate(key):
        for j in key[i_pos+1:]:
            feats.append(np.linalg.norm(model_preds[i] - model_preds[j], axis=-1, keepdims=True))
    return np.concatenate(feats, axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, preds_tr, feat_te, preds_te, args, kf):
    device = torch.device("cpu")
    N, M = feat_tr.shape[0], preds_tr.shape[1]
    oof = np.zeros((N, 3)); tests = []; w_all = np.zeros((N, M), dtype=np.float32)
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        ft = lambda a: sc.transform(a).astype(np.float32)
        def T(a): return torch.from_numpy(a).to(device)
        x_tr, x_va, x_te = T(ft(feat_tr[tr])), T(ft(feat_tr[va])), T(ft(feat_te))
        p_tr = T(preds_tr[tr].astype(np.float32))
        p_va = T(preds_tr[va].astype(np.float32))
        p_te = T(preds_te.astype(np.float32))
        y_tr_t = T(y_tr[tr].astype(np.float32))

        torch.manual_seed(seed); np.random.seed(seed)
        m = SoftStacker(in_dim=x_tr.shape[1], n_models=M,
                        hidden=args.hidden, p=args.dropout, temp=args.temp).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

        best_rh, best_st, no_imp = -1.0, None, 0
        n_tr = x_tr.shape[0]
        for ep in range(args.max_epochs):
            m.train()
            perm = torch.randperm(n_tr)
            for i in range(0, n_tr, 512):
                idx = perm[i:i+512]
                opt.zero_grad()
                pred, _ = m(x_tr[idx], p_tr[idx])
                loss_combo(pred, y_tr_t[idx]).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                pv, _ = m(x_va, p_va)
                r = rhit(pv.cpu().numpy(), y_tr[va])
            if r > best_rh:
                best_rh, best_st, no_imp = r, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
            else: no_imp += 1
            if no_imp >= args.patience: break
        m.load_state_dict(best_st); m.eval()
        with torch.no_grad():
            pv, wv = m(x_va, p_va)
            oof[va] = pv.cpu().numpy(); w_all[va] = wv.cpu().numpy()
            pt, _ = m(x_te, p_te); tests.append(pt.cpu().numpy())
        print(f"  seed{seed} fold{fi+1}: rhit={rhit(oof[va], y_tr[va]):.4f} ({(time.time()-t0)/60:.1f}m)")
        del m; gc.collect()
    return oof, np.mean(tests, axis=0), w_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--temp", type=float, default=1.5)
    parser.add_argument("--use-v62", action="store_true", help="v62도 추가 (12m)")
    parser.add_argument("--v65-mode", default="soft", choices=["soft", "hard"])
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v67 SoftStacker — base 9m + v65 {args.v65_mode}" + (" + v62" if args.use_v62 else ""))
    print(f"  hid {args.hidden}, drop {args.dropout}, temp {args.temp}, {args.n_folds}-fold × {args.n_seeds}-seed")
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

    # 9 base models
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
    st32 = np.load(CACHE_DIR / "v32_mdn_state.npz")
    v32o = st32["oof_weighted"].astype(np.float64); v32t = st32["test_weighted"].astype(np.float64)

    models = [v30A_o, v30B_o, v35o, v41A_o, v41B_o, v44o, gate_o, v39o, v32o]
    tests  = [v30A_t, v30B_t, v35t, v41A_t, v41B_t, v44t, gate_t, v39t, v32t]
    names = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate", "v39", "v32"]

    # v65
    st65 = np.load(CACHE_DIR / "v65_K64_state.npz")
    if args.v65_mode == "soft":
        v65o, v65t = st65["oof_soft"].astype(np.float64), st65["test_soft"].astype(np.float64)
    else:
        v65o, v65t = st65["oof_hard"].astype(np.float64), st65["test_hard"].astype(np.float64)
    models.append(v65o); tests.append(v65t); names.append(f"v65{args.v65_mode}")

    # optional v62
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

    print("\n=== Model OOF R-Hit ===")
    for n, p in zip(names, models): print(f"  {n}: {rhit(p, y_train):.4f}")
    hits = np.stack([np.linalg.norm(p - y_train, axis=-1) <= 0.01 for p in models])
    oracle = hits.any(axis=0).mean()
    print(f"\n  Oracle ({len(names)}-model): {oracle:.4f}")

    feat_tr = build_features(X_train, kalman_train, models)
    feat_te = build_features(X_test, kalman_test, tests)
    preds_tr = np.stack(models, axis=1).astype(np.float64)
    preds_te = np.stack(tests, axis=1).astype(np.float64)
    print(f"  feat dim: {feat_tr.shape[1]}, preds: {preds_tr.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests_l, w_all = [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        o, t, w = train_one_seed(s, feat_tr, y_train, preds_tr, feat_te, preds_te, args, kf)
        oofs.append(o); tests_l.append(t); w_all.append(w)
        print(f"  seed{s} OOF: {rhit(o, y_train):.4f}")

    oof_v67 = np.mean(oofs, axis=0); test_v67 = np.mean(tests_l, axis=0)
    w_mean = np.mean(w_all, axis=0)
    rh_v67 = rhit(oof_v67, y_train)

    base_o = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["oof_v48"] \
           + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["oof_v46"] \
           + 0.18 * v35o
    base_t = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["test_v48"] \
           + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["test_v46"] \
           + 0.18 * v35t
    rh_base = rhit(base_o, y_train)

    print(f"\n=== v67 결과 ({len(names)}m) ===")
    print(f"  v48 9m alone   : {rhit(np.load(CACHE_DIR / 'v48_state.npz')['oof_v48'], y_train):.4f}")
    print(f"  v48 3-way base : {rh_base:.4f}")
    print(f"  v67            : {rh_v67:.4f}  (Δ vs base: {rh_v67 - rh_base:+.4f})")
    print(f"\n  weights:")
    for n, wi in zip(names, w_mean.mean(axis=0)):
        flag = " ★" if "v65" in n else ""
        print(f"    {n:8s}: {wi:.3f}{flag}")

    # hybrid grid
    print(f"\n=== hybrid: w*base + (1-w)*v67 ===")
    best_w, best_r = 1.0, rh_base
    for w in np.linspace(0, 1, 21):
        ens = w * base_o + (1 - w) * oof_v67
        r = rhit(ens, y_train)
        if r > best_r: best_r, best_w = r, w
    print(f"  best: w={best_w:.2f} → OOF {best_r:.4f}  (Δ {best_r - rh_base:+.4f})")

    name = f"v67_{args.v65_mode}" + ("_v62" if args.use_v62 else "")
    np.savez(CACHE_DIR / f"{name}_state.npz",
             oof=oof_v67, test=test_v67, rh=rh_v67, w_mean=w_mean)

    out = DATA_DIR / f"submission_{name}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v67[:,0], "y": test_v67[:,1], "z": test_v67[:,2]}).to_csv(out, index=False)
    print(f"\n  [submission] {out.name}")
    if best_w < 1.0 and best_r > rh_base:
        hyb_t = best_w * base_t + (1 - best_w) * test_v67
        h_csv = DATA_DIR / f"submission_{name}_hybrid_basex{best_w:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb_t[:,0], "y": hyb_t[:,1], "z": hyb_t[:,2]}).to_csv(h_csv, index=False)
        print(f"  [hybrid] OOF {best_r:.4f} → {h_csv.name}")

    entry = {
        "version": name, "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"{len(names)}-model SoftStacker", "models": names,
        "rh": rh_v67, "delta_vs_base": rh_v67 - rh_base,
        "oracle": float(oracle),
        "weights": {n: float(w) for n, w in zip(names, w_mean.mean(axis=0))},
        "hybrid_best_w": float(best_w), "hybrid_best_oof": float(best_r),
    }
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

"""v75_softstacker_v73.py — 12-model SoftStacker (v48 9m + v73 soft + v73 hard + v65 hard).

v73 강화된 anchor head OOF 0.6244 — stacker 추가 효과 재시도.
v67 (v65 hard 추가): weight 0.014로 사실상 무시.
v73이 더 강하므로 weight 더 받을 가능성.
"""
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
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
BO_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
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
    N, M = feat_tr.shape[0], preds_tr.shape[1]
    oof = np.zeros((N, 3)); tests = []; w_all = np.zeros((N, M), dtype=np.float32)
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        ft = lambda a: sc.transform(a).astype(np.float32)
        def T(a): return torch.from_numpy(a)
        x_tr, x_va, x_te = T(ft(feat_tr[tr])), T(ft(feat_tr[va])), T(ft(feat_te))
        p_tr = T(preds_tr[tr].astype(np.float32))
        p_va = T(preds_tr[va].astype(np.float32))
        p_te = T(preds_te.astype(np.float32))
        y_tr_t = T(y_tr[tr].astype(np.float32))
        torch.manual_seed(seed); np.random.seed(seed)
        m = SoftStacker(in_dim=x_tr.shape[1], n_models=M,
                        hidden=args.hidden, p=args.dropout, temp=args.temp)
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
    parser.add_argument("--config", default="full",
                        help="full=v48 9m + v73 soft/hard + v65 hard + v62 (12m)\n"
                             "v73only=v48 9m + v73 soft (10m)\n"
                             "v73both=v48 9m + v73 soft + v73 hard (11m)")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)

    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y_train = labels.set_index("id").loc[list(train_ids)][["x","y","z"]].values.astype(np.float64)

    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    kc = np.load(CACHE_DIR / "kalman.npz")
    kt, ke = kc["kalman_train"], kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]

    st30 = np.load(CACHE_DIR / "v30_state.npz")
    v30A_o = kt + st30["oof_A"]*ALPHA; v30A_t = ke + st30["test_A"]*ALPHA
    v30B_o = kt + st30["oof_B"]*ALPHA; v30B_t = ke + st30["test_B"]*ALPHA
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v35o, v35t = st35["oof_v35"].astype(np.float64), st35["test_v35"].astype(np.float64)
    st41 = np.load(CACHE_DIR / "v41_state.npz")
    v41A_o = kt + st41["oof_A"]*ALPHA; v41A_t = ke + st41["test_A"]*ALPHA
    v41B_o = kt + st41["oof_B"]*ALPHA; v41B_t = ke + st41["test_B"]*ALPHA
    st44 = np.load(CACHE_DIR / "v44_state.npz")
    v44o, v44t = st44["oof_v44"].astype(np.float64), st44["test_v44"].astype(np.float64)
    bo = np.load(BO_PATH, allow_pickle=True); gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(bo["ids"], train_ids):
        mm = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([mm[i] for i in bo["ids"]]); gate_o = gate_o[perm]
    gate_t = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    st39 = np.load(CACHE_DIR / "v39_state.npz")
    v39o, v39t = st39["oof_v39"].astype(np.float64), st39["test_v39"].astype(np.float64)
    st32 = np.load(CACHE_DIR / "v32_mdn_state.npz")
    v32o, v32t = st32["oof_weighted"].astype(np.float64), st32["test_weighted"].astype(np.float64)

    models = [v30A_o, v30B_o, v35o, v41A_o, v41B_o, v44o, gate_o, v39o, v32o]
    tests  = [v30A_t, v30B_t, v35t, v41A_t, v41B_t, v44t, gate_t, v39t, v32t]
    names  = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate", "v39", "v32"]

    st73 = np.load(CACHE_DIR / "v73_K64_state.npz")
    st65 = np.load(CACHE_DIR / "v65_K64_state.npz")
    st62 = np.load(CACHE_DIR / "v62_state.npz")
    v_last_tr = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_te = (X_test[:, -1] - X_test[:, -2]) / DT
    th_tr, th_te = yaw_angle(v_last_tr), yaw_angle(v_last_te)
    v62o = st62["kalman_train"] + inverse_rotate_xy((st62["oof_A"] + st62["oof_B"])/2, th_tr)
    v62t = st62["kalman_test"] + inverse_rotate_xy((st62["test_A"] + st62["test_B"])/2, th_te)

    if args.config == "v73only":
        models += [st73["oof_soft"].astype(np.float64)]
        tests  += [st73["test_soft"].astype(np.float64)]
        names  += ["v73s"]
    elif args.config == "v73both":
        models += [st73["oof_soft"].astype(np.float64), st73["oof_hard"].astype(np.float64)]
        tests  += [st73["test_soft"].astype(np.float64), st73["test_hard"].astype(np.float64)]
        names  += ["v73s", "v73h"]
    else:  # full
        models += [st73["oof_soft"].astype(np.float64), st73["oof_hard"].astype(np.float64),
                   st65["oof_hard"].astype(np.float64), v62o]
        tests  += [st73["test_soft"].astype(np.float64), st73["test_hard"].astype(np.float64),
                   st65["test_hard"].astype(np.float64), v62t]
        names  += ["v73s", "v73h", "v65h", "v62"]

    K = len(names)
    print("=" * 60)
    print(f"v75 SoftStacker — {K} models ({args.config})")
    print("=" * 60)
    for n, p in zip(names, models): print(f"  {n}: {rhit(p, y_train):.4f}")
    hits = np.stack([np.linalg.norm(p - y_train, axis=-1) <= 0.01 for p in models])
    print(f"\n  Oracle ({K}-m): {hits.any(axis=0).mean():.4f}")

    feat_tr = build_features(X_train, kt, models)
    feat_te = build_features(X_test, ke, tests)
    preds_tr = np.stack(models, axis=1).astype(np.float64)
    preds_te = np.stack(tests, axis=1).astype(np.float64)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, ts, ws = [], [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        o, t, w = train_one_seed(s, feat_tr, y_train, preds_tr, feat_te, preds_te, args, kf)
        oofs.append(o); ts.append(t); ws.append(w)
        print(f"  seed{s} OOF: {rhit(o, y_train):.4f}")
    oof = np.mean(oofs, axis=0); test = np.mean(ts, axis=0); w_mean = np.mean(ws, axis=0)
    rh_v = rhit(oof, y_train)

    base_o = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["oof_v48"] \
           + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["oof_v46"] \
           + 0.18 * v35o
    base_t = 0.70 * np.load(CACHE_DIR / "v48_state.npz")["test_v48"] \
           + 0.12 * np.load(CACHE_DIR / "v46_state.npz")["test_v46"] \
           + 0.18 * v35t
    rh_base = rhit(base_o, y_train)

    print(f"\n=== v75 결과 ===")
    print(f"  base v48 3-way : {rh_base:.4f}")
    print(f"  v75 ({K}m)      : {rh_v:.4f}  (Δ vs base: {rh_v - rh_base:+.4f})")
    print(f"  weights:")
    for n, wi in zip(names, w_mean.mean(axis=0)):
        flag = " ★" if any(t in n for t in ("v73", "v65", "v62")) else ""
        print(f"    {n:8s}: {wi:.3f}{flag}")

    print(f"\n=== hybrid: w*base + (1-w)*v75 ===")
    best_w, best_r = 1.0, rh_base
    for w in np.linspace(0, 1, 21):
        ens = w * base_o + (1 - w) * oof
        r = rhit(ens, y_train)
        if r > best_r: best_r, best_w = r, w
    print(f"  best: w={best_w:.2f} → OOF {best_r:.4f}  (Δ {best_r - rh_base:+.4f})")

    out = DATA_DIR / f"submission_v75_{args.config}.csv"
    pd.DataFrame({"id": sub["id"], "x": test[:,0], "y": test[:,1], "z": test[:,2]}).to_csv(out, index=False)
    print(f"  [submission] {out.name}")
    if best_w < 1.0 and best_r > rh_base:
        hyb_t = best_w * base_t + (1 - best_w) * test
        h_csv = DATA_DIR / f"submission_v75_{args.config}_hyb_basex{best_w:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb_t[:,0], "y": hyb_t[:,1], "z": hyb_t[:,2]}).to_csv(h_csv, index=False)
        print(f"  [hybrid] OOF {best_r:.4f} → {h_csv.name}")

    np.savez(CACHE_DIR / f"v75_{args.config}_state.npz",
             oof=oof, test=test, rh=rh_v, w_mean=w_mean)
    entry = {"version": f"v75_{args.config}", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "K": K, "models": names, "rh": rh_v, "delta_vs_base": rh_v - rh_base,
             "weights": {n: float(w) for n, w in zip(names, w_mean.mean(axis=0))},
             "hybrid_best_w": float(best_w), "hybrid_best_oof": float(best_r)}
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

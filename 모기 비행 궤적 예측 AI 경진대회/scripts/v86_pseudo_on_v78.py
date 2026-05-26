"""v86_pseudo_on_v78.py — pseudo-label v2 on v77 base + boundary MLP.

v39 패턴: base v30 + boundary + v35 test prediction pseudo (top 20% confident) → OOF 0.6720
v86: base v77 BiGRU + boundary + v78 test prediction pseudo (top 20% confident)
     = paradigm 다른 pseudo-label 강화 → v78 기반 OOF 0.674+ 기대
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
BO_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def loss_combo(p, t, sw=None):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    if sw is None: return d.mean() + 0.3 * sh.mean()
    return ((d * sw).mean() + 0.3 * (sh * sw).mean()) / sw.mean()


class BoundaryMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, p=0.2, cap_cm=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 3); self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.cap = cap_cm / 100.0
    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z))
        return torch.tanh(self.head(z)) * self.cap


def build_features(X, kalman, base_pred, gate_pred, v16_pred):
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True); a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    diff_bg = base_pred - gate_pred; diff_bv = base_pred - v16_pred
    dist_bg = np.linalg.norm(diff_bg, axis=-1, keepdims=True)
    res_b_kal = base_pred - kalman
    return np.concatenate([base_pred, gate_pred, v16_pred, diff_bg, diff_bv, dist_bg,
                           kalman, res_b_kal, last_pos, v, a, v_mean, v_std,
                           speed, a_norm], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cap-cm", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pseudo-pct", type=float, default=0.20)
    parser.add_argument("--pseudo-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-loss-weight", type=float, default=0.3)
    args = parser.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"v86 pseudo-label on v77 base + v78 boundary pseudo (top {args.pseudo_pct*100:.0f}% confident)")

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

    # v77 base
    st77 = np.load(CACHE_DIR / "v77_bigru_state.npz")
    oof_v77 = kalman_train + (st77["oof_A"] + st77["oof_B"])/2 * ALPHA
    test_v77 = kalman_test + (st77["test_A"] + st77["test_B"])/2 * ALPHA

    # gate
    bo = np.load(BO_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids]); gate_oof = gate_oof[perm]
    test_gate = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    # v16
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64); test_v16 = st16["test"].astype(np.float64)

    # v78 OOF + test (pseudo source)
    st78 = np.load(CACHE_DIR / "v78_state.npz")
    oof_v78 = st78["oof_v78"]; test_v78 = st78["test_v78"]
    rh_v78 = float((np.linalg.norm(oof_v78 - y_train, axis=-1) <= 0.01).mean())
    print(f"v78 source OOF: {rh_v78:.4f}")

    # confidence: v78 vs gate + v78 vs Kalman 거리 작을수록 confident
    dist_vg = np.linalg.norm(test_v78 - test_gate, axis=-1)
    dist_vk = np.linalg.norm(test_v78 - kalman_test, axis=-1)
    rank_vg = np.argsort(np.argsort(dist_vg)) / len(dist_vg)
    rank_vk = np.argsort(np.argsort(dist_vk)) / len(dist_vk)
    confidence = 1.0 - (rank_vg + rank_vk) / 2
    n_pseudo = int(args.pseudo_pct * len(test_v78))
    pseudo_idx = np.argsort(-confidence)[:n_pseudo]
    print(f"Top {args.pseudo_pct*100:.0f}% confident: {n_pseudo} pseudo")

    X_pseudo = X_test[pseudo_idx]
    y_pseudo = test_v78[pseudo_idx]
    v77_pseudo = test_v77[pseudo_idx]
    gate_pseudo = test_gate[pseudo_idx]
    v16_pseudo = test_v16[pseudo_idx]
    kalman_pseudo = kalman_test[pseudo_idx]

    feat_train = build_features(X_train, kalman_train, oof_v77, gate_oof, oof_v16)
    feat_pseudo = build_features(X_pseudo, kalman_pseudo, v77_pseudo, gate_pseudo, v16_pseudo)
    feat_test = build_features(X_test, kalman_test, test_v77, test_gate, test_v16)
    print(f"feat: train={feat_train.shape}, pseudo={feat_pseudo.shape}")

    d_v77 = np.linalg.norm(oof_v77 - y_train, axis=-1)
    bm = (d_v77 > 0.005) & (d_v77 <= 0.03)
    sw_train = np.ones(len(y_train), dtype=np.float32); sw_train[bm] = args.boundary_weight
    pb = (dist_vg[pseudo_idx] > 0.005) & (dist_vg[pseudo_idx] <= 0.03)
    sw_pseudo = np.ones(len(y_pseudo), dtype=np.float32) * args.pseudo_weight
    sw_pseudo[pb] *= args.boundary_weight

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        oof_s = np.zeros_like(oof_v77); test_pf = []
        t0 = time.time()
        for fi, (tr, va) in enumerate(kf.split(feat_train)):
            sc = StandardScaler().fit(np.concatenate([feat_train[tr], feat_pseudo], axis=0))
            ft = lambda a: sc.transform(a).astype(np.float32)
            def T(a): return torch.from_numpy(a)
            x_tr, x_ps, x_va, x_te = T(ft(feat_train[tr])), T(ft(feat_pseudo)), T(ft(feat_train[va])), T(ft(feat_test))
            v77_tr_t = T(oof_v77[tr].astype(np.float32))
            v77_ps_t = T(v77_pseudo.astype(np.float32))
            v77_va_t = T(oof_v77[va].astype(np.float32))
            v77_te_t = T(test_v77.astype(np.float32))
            y_tr_t = T(y_train[tr].astype(np.float32))
            y_ps_t = T(y_pseudo.astype(np.float32))
            sw_tr_t = T(sw_train[tr]); sw_ps_t = T(sw_pseudo)

            torch.manual_seed(s); np.random.seed(s)
            model = BoundaryMLP(in_dim=x_tr.shape[1], hidden=args.hidden, p=0.2, cap_cm=args.cap_cm)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
            best_rh, best_st, no_imp = -1.0, None, 0
            n_tr = x_tr.shape[0]; n_ps = x_ps.shape[0]
            for ep in range(args.max_epochs):
                model.train()
                perm_tr = torch.randperm(n_tr); perm_ps = torch.randperm(n_ps)
                for i in range(0, n_tr, 256):
                    idx_tr = perm_tr[i:i+256]
                    n_ps_chunk = max(1, int(256 * n_ps / n_tr))
                    ps_start = (i // 256) * n_ps_chunk % n_ps
                    idx_ps = perm_ps[ps_start:ps_start + n_ps_chunk]
                    opt.zero_grad()
                    pred_real = v77_tr_t[idx_tr] + model(x_tr[idx_tr])
                    loss_real = loss_combo(pred_real, y_tr_t[idx_tr], sw=sw_tr_t[idx_tr])
                    if len(idx_ps) > 0:
                        pred_ps = v77_ps_t[idx_ps] + model(x_ps[idx_ps])
                        loss_ps = loss_combo(pred_ps, y_ps_t[idx_ps], sw=sw_ps_t[idx_ps])
                        loss = loss_real + args.pseudo_loss_weight * loss_ps
                    else:
                        loss = loss_real
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    pv = (v77_va_t + model(x_va)).cpu().numpy()
                rh = float((np.linalg.norm(pv - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh, best_st, no_imp = rh, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
                else: no_imp += 1
                if no_imp >= args.patience: break
            model.load_state_dict(best_st); model.eval()
            with torch.no_grad():
                oof_s[va] = (v77_va_t + model(x_va)).cpu().numpy()
                test_pf.append((v77_te_t + model(x_te)).cpu().numpy())
            rh_f = float((np.linalg.norm(oof_s[va] - y_train[va], axis=-1) <= 0.01).mean())
            print(f"  seed{s} fold{fi+1}: rhit={rh_f:.4f}  ({(time.time()-t0)/60:.1f}m)")
            del model; gc.collect()
        oofs.append(oof_s); tests.append(np.mean(test_pf, axis=0))
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v86 = np.mean(oofs, axis=0); test_v86 = np.mean(tests, axis=0)
    rh_v86 = float((np.linalg.norm(oof_v86 - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== v86 결과 ===")
    print(f"  v77 base : {float((np.linalg.norm(oof_v77 - y_train, axis=-1) <= 0.01).mean()):.4f}")
    print(f"  v78      : 0.6730")
    print(f"  v86 pseudo: {rh_v86:.4f}  (Δ vs v78: {rh_v86 - 0.6730:+.4f}, vs v39 0.6720: {rh_v86 - 0.6720:+.4f})")

    np.savez(CACHE_DIR / "v86_state.npz",
             oof_v86=oof_v86, test_v86=test_v86, rh_v86=rh_v86,
             pseudo_pct=args.pseudo_pct)
    out_csv = DATA_DIR / "submission_v86_pseudo_on_v78.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v86[:,0], "y": test_v86[:,1], "z": test_v86[:,2]}).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv.name}")

    # quick blend with base v48 3-way
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v48s = np.load(CACHE_DIR / "v48_state.npz"); v46s = np.load(CACHE_DIR / "v46_state.npz")
    bo_o = 0.70*v48s["oof_v48"] + 0.12*v46s["oof_v46"] + 0.18*st35["oof_v35"]
    bo_t = 0.70*v48s["test_v48"] + 0.12*v46s["test_v46"] + 0.18*st35["test_v35"]
    rh_b = float((np.linalg.norm(bo_o - y_train, axis=-1) <= 0.01).mean())

    # 5-way grid: v48_9m, v46_7m, v35, v78, v86
    print(f"\n=== 5-way grid (v48_9m/v46_7m/v35/v78/v86) ===")
    po = [v48s["oof_v48"], v46s["oof_v46"], st35["oof_v35"], st78["oof_v78"], oof_v86]
    pt = [v48s["test_v48"], v46s["test_v46"], st35["test_v35"], st78["test_v78"], test_v86]
    best_w, best_r = None, rh_b
    for a in np.linspace(0.55, 0.80, 6):
        for b in np.linspace(0.05, 0.18, 5):
            for c in np.linspace(0.10, 0.25, 6):
                for d in np.linspace(0, 0.08, 4):
                    e = 1 - a - b - c - d
                    if e < 0 or e > 0.15: continue
                    ens = a*po[0]+b*po[1]+c*po[2]+d*po[3]+e*po[4]
                    r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
                    if r > best_r: best_r, best_w = r, (a,b,c,d,e)
    if best_w:
        a,b,c,d,e = best_w
        ens_t = a*pt[0]+b*pt[1]+c*pt[2]+d*pt[3]+e*pt[4]
        out5 = DATA_DIR / f"submission_v86_5way_{a:.2f}_{b:.2f}_{c:.2f}_{d:.2f}_{e:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out5, index=False)
        print(f"  best ({a:.2f}/{b:.2f}/{c:.2f}/{d:.2f}/{e:.2f}): OOF {best_r:.4f}  Δ {best_r - rh_b:+.4f}")
        print(f"  [submission] {out5.name}")

    entry = {"version": "v86_pseudo_on_v78", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "rh_v86": rh_v86, "delta_vs_v78": rh_v86 - 0.6730, "delta_vs_v39": rh_v86 - 0.6720,
             "blend_5way_oof": float(best_r) if best_w else rh_b,
             "blend_5way_weights": list(best_w) if best_w else None}
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

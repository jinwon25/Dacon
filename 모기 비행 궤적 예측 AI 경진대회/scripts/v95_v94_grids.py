"""v95_v94_grids.py — v94 + 기존 pool 확장 grid + SoftStacker.

v94 OOF 0.6738 (paradigm 최강), v94 4-way OOF 0.6752 (+0.0004 over v82).
이 script: v94 추가 후 더 큰 grid + SoftStacker 재학습.
"""
from __future__ import annotations

import datetime as _dt, gc, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "open"
CACHE_DIR = PROJECT / "cache"

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


def rh(pred):
    return float((np.linalg.norm(pred - y_train, axis=-1) <= 0.01).mean())


pool = {}
def load_v(name, oof_key, test_key, cache_name=None):
    cn = cache_name or f"{name}_state.npz"
    st = np.load(CACHE_DIR / cn)
    pool[name] = (st[oof_key], st[test_key])

load_v("v35", "oof_v35", "test_v35")
load_v("v44", "oof_v44", "test_v44")
load_v("v48", "oof_v48", "test_v48")
load_v("v46", "oof_v46", "test_v46")
load_v("v53", "oof_v53", "test_v53")
load_v("v78", "oof_v78", "test_v78")
load_v("v91", "oof_v91", "test_v91")  # boundary on v90 single A
load_v("v94", "oof_v91", "test_v91", cache_name="v94_state.npz")  # boundary on v90 A+B
load_v("v94_cap0p5", "oof_v91", "test_v91", cache_name="v94_cap0p5_state.npz")
load_v("v94_cap1p5", "oof_v91", "test_v91", cache_name="v94_cap1p5_state.npz")
load_v("v97", "oof_v91", "test_v91", cache_name="v97_state.npz")  # boundary on v96 4-view
load_v("v97_cap0p5", "oof_v91", "test_v91", cache_name="v97_cap0p5_state.npz")
load_v("v97_cap1p5", "oof_v91", "test_v91", cache_name="v97_cap1p5_state.npz")
load_v("v101", "oof_v91", "test_v91", cache_name="v101_state.npz")  # boundary on v100 SWA+smooth
load_v("v101_cap1p5", "oof_v91", "test_v91", cache_name="v101_cap1p5_state.npz")

# v90 single A base
st90A = np.load(CACHE_DIR / "v90_mirror_state.npz")
pool["v90A"] = (kalman_train + st90A["oof"] * ALPHA, kalman_test + st90A["test_pred"] * ALPHA)
# v90 A+B avg base
st90B = np.load(CACHE_DIR / "v90_mirror_setupB_state.npz")
pool["v90AB"] = (kalman_train + (st90A["oof"] + st90B["oof"])/2 * ALPHA,
                  kalman_test + (st90A["test_pred"] + st90B["test_pred"])/2 * ALPHA)

# cap variants
for tag, cn in [("v52_cap0p5", "v52_cap0p5_state.npz"),
                ("v52_cap1p5", "v52_cap1p5_state.npz")]:
    try:
        st = np.load(CACHE_DIR / cn)
        keys = list(st.keys())
        oof_k = [k for k in keys if "oof" in k][0]
        test_k = [k for k in keys if "test" in k][0]
        pool[tag] = (st[oof_k], st[test_k])
    except Exception: pass

print("=" * 60)
print("Pool OOF R-Hit")
print("=" * 60)
for k, (o, _) in sorted(pool.items(), key=lambda kv: rh(kv[1][0]), reverse=True):
    print(f"  {k:15s}: {rh(o):.4f}")

v48_o, v48_t = pool["v48"]
v46_o, v46_t = pool["v46"]
v35_o, v35_t = pool["v35"]
v78_o, v78_t = pool["v78"]
v91_o, v91_t = pool["v91"]
v94_o, v94_t = pool["v94"]
v97_o, v97_t = pool["v97"]
v97c5_o, v97c5_t = pool["v97_cap1p5"]  # 단일 최강 0.6749
v97c05_o, v97c05_t = pool["v97_cap0p5"]
v101_o, v101_t = pool["v101"]
v101c5_o, v101c5_t = pool["v101_cap1p5"]

base_o = 0.70*v48_o + 0.12*v46_o + 0.18*v35_o
base_t = 0.70*v48_t + 0.12*v46_t + 0.18*v35_t
rh_base = rh(base_o)
print(f"\nv48 3-way base OOF: {rh_base:.4f}")
print(f"v94 단독 OOF: {rh(v94_o):.4f}")

# ============================================================
# 1) 5-way grid: base + v94 + v91 + v78 + v35 (확장)
# ============================================================
print("\n--- 5-way grid: base + v94 + v91 + v78 + v35 ---")
best5 = (rh_base, None)
for a in np.linspace(0.1, 1.0, 19):
    for b in np.linspace(0, 1-a, 13):
        for c in np.linspace(0, 1-a-b, 7):
            for d in np.linspace(0, 1-a-b-c, 5):
                e = 1 - a - b - c - d
                if e < 0: continue
                ens = a*base_o + b*v94_o + c*v91_o + d*v78_o + e*v35_o
                r = rh(ens)
                if r > best5[0]: best5 = (r, (a, b, c, d, e))
if best5[1]:
    a, b, c, d, e = best5[1]
    ens_t = a*base_t + b*v94_t + c*v91_t + d*v78_t + e*v35_t
    out = DATA_DIR / f"submission_v95_5w_base{a:.2f}_v94{b:.2f}_v91{c:.2f}_v78{d:.2f}_v35{e:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best: ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}, {e:.2f}) → OOF {best5[0]:.4f}  Δ {best5[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# ============================================================
# 2) v94 + v78 + v91 paradigm trio
# ============================================================
print("\n--- v94 + v78 + v91 paradigm trio ---")
best_t = (max(rh(v94_o), rh(v78_o), rh(v91_o)), None)
for a in np.linspace(0, 1, 21):
    for b in np.linspace(0, 1-a, 11):
        c = 1 - a - b
        if c < 0: continue
        ens = a*v94_o + b*v78_o + c*v91_o
        r = rh(ens)
        if r > best_t[0]: best_t = (r, (a, b, c))
if best_t[1]:
    a, b, c = best_t[1]
    ens_t = a*v94_t + b*v78_t + c*v91_t
    out = DATA_DIR / f"submission_v95_trio_v94{a:.2f}_v78{b:.2f}_v91{c:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best: ({a:.2f}, {b:.2f}, {c:.2f}) → OOF {best_t[0]:.4f}")
    print(f"  [submission] {out.name}")

# ============================================================
# 3) 12m SoftStacker (v48+v46+v35+v44+v78+v91+v94+v90AB+cap variants)
# ============================================================
print("\n" + "=" * 60)
print("12m SoftStacker: full pool with v94/v91/v90AB")
print("=" * 60)

stk_keys = ["v48", "v46", "v35", "v44", "v78", "v94", "v94_cap1p5",
            "v97", "v97_cap1p5", "v101", "v101_cap1p5", "v90AB"]
for k in ["v52_cap0p5", "v52_cap1p5"]:
    if k in pool: stk_keys.append(k)
print(f"Stacker pool ({len(stk_keys)}m): {stk_keys}")

stk_oof = np.stack([pool[k][0] for k in stk_keys], axis=1)
stk_test = np.stack([pool[k][1] for k in stk_keys], axis=1)
M = stk_oof.shape[1]

def build_stk_feat(stk):
    N, M, _ = stk.shape
    mean = stk.mean(axis=1, keepdims=True)
    res = stk - mean
    flat = res.reshape(N, M*3)
    pair_dist = np.zeros((N, M, M))
    for i in range(M):
        for j in range(M):
            pair_dist[:, i, j] = np.linalg.norm(stk[:, i] - stk[:, j], axis=-1)
    pair_flat = pair_dist.reshape(N, M*M)
    return np.concatenate([flat, pair_flat, mean.reshape(N, 3)], axis=-1).astype(np.float32)

feat_oof = build_stk_feat(stk_oof)
feat_test = build_stk_feat(stk_test)
print(f"Stacker feat dim: {feat_oof.shape[1]}")


class SoftStacker(nn.Module):
    def __init__(self, feat_dim, M, hidden=96, p=0.4, temp=1.5):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, M)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.temp = temp

    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z))
        logits = self.head(z) / self.temp
        return torch.softmax(logits, dim=-1)


def softhit_loss(p, t, beta=0.002):
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    return torch.sigmoid((d - 0.01) / beta).mean()


def train_stacker(seeds=3, folds=5, epochs=80, patience=15):
    device = torch.device("cpu")
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    test_pred_seeds = []
    oofs_seeds = []
    weight_log = []
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        oof_pred = np.zeros_like(stk_oof[:, 0])
        test_per_fold = []
        for fi, (tr, va) in enumerate(kf.split(feat_oof)):
            sc = StandardScaler().fit(feat_oof[tr])
            x_tr = torch.from_numpy(sc.transform(feat_oof[tr]).astype(np.float32))
            x_va = torch.from_numpy(sc.transform(feat_oof[va]).astype(np.float32))
            x_te = torch.from_numpy(sc.transform(feat_test).astype(np.float32))
            stk_tr = torch.from_numpy(stk_oof[tr].astype(np.float32))
            stk_va = torch.from_numpy(stk_oof[va].astype(np.float32))
            stk_te = torch.from_numpy(stk_test.astype(np.float32))
            y_tr_t = torch.from_numpy(y_train[tr].astype(np.float32))

            model = SoftStacker(feat_oof.shape[1], M).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            best_rh, best_state, no_improve = -1.0, None, 0
            for ep in range(epochs):
                model.train()
                perm = torch.randperm(x_tr.shape[0])
                for i in range(0, x_tr.shape[0], 256):
                    idx = perm[i:i+256]
                    opt.zero_grad()
                    w = model(x_tr[idx])
                    pred = (w.unsqueeze(-1) * stk_tr[idx]).sum(dim=1)
                    d = torch.sqrt(((pred - y_tr_t[idx]) ** 2).sum(dim=-1) + 1e-12)
                    loss = d.mean() + 0.3 * softhit_loss(pred, y_tr_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    w_va = model(x_va)
                    pred_va = (w_va.unsqueeze(-1) * stk_va).sum(dim=1).cpu().numpy()
                rh_va = float((np.linalg.norm(pred_va - y_train[va], axis=-1) <= 0.01).mean())
                if rh_va > best_rh:
                    best_rh = rh_va
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                w_va = model(x_va); w_te = model(x_te)
                oof_pred[va] = (w_va.unsqueeze(-1) * stk_va).sum(dim=1).cpu().numpy()
                test_per_fold.append((w_te.unsqueeze(-1) * stk_te).sum(dim=1).cpu().numpy())
                weight_log.append(w_va.mean(dim=0).cpu().numpy())
            del model; gc.collect()
        rh_s = rh(oof_pred)
        print(f"  seed {s}: OOF {rh_s:.4f}")
        oofs_seeds.append(oof_pred)
        test_pred_seeds.append(np.mean(test_per_fold, axis=0))
    oof_pred_all = np.mean(oofs_seeds, axis=0)
    test_pred_all = np.mean(test_pred_seeds, axis=0)
    avg_w = np.mean(weight_log, axis=0)
    print(f"\nAvg weights:")
    for k, w in zip(stk_keys, avg_w):
        print(f"  {k:15s}: {w:.4f}")
    return oof_pred_all, test_pred_all


oof_stk, test_stk = train_stacker(seeds=3, folds=5, epochs=80, patience=15)
rh_stk = rh(oof_stk)
print(f"\n{len(stk_keys)}m SoftStacker OOF: {rh_stk:.4f}  Δ vs v48 3-way: {rh_stk - rh_base:+.4f}")

if rh_stk > rh_base + 0.0005:
    out = DATA_DIR / f"submission_v95_{len(stk_keys)}m_stacker.csv"
    pd.DataFrame({"id": sub["id"], "x": test_stk[:,0], "y": test_stk[:,1], "z": test_stk[:,2]}).to_csv(out, index=False)
    print(f"  [submission] {out.name}")

# ============================================================
# 4) hybrid: stacker + v94 + v78 grid
# ============================================================
print("\n--- v95 hybrid: stacker + v94 + v78 + v35 grid ---")
best_h = (max(rh_stk, rh_base), None, None)
for src_o, src_t, name in [(oof_stk, test_stk, "stk"), (base_o, base_t, "base")]:
    for a in np.linspace(0.1, 1.0, 19):
        for b in np.linspace(0, 1-a, 13):
            for c in np.linspace(0, 1-a-b, 7):
                d = 1 - a - b - c
                if d < 0: continue
                ens = a*src_o + b*v94_o + c*v78_o + d*v35_o
                r = rh(ens)
                if r > best_h[0]: best_h = (r, (a, b, c, d), name)
if best_h[1]:
    a, b, c, d = best_h[1]
    src_t = test_stk if best_h[2] == "stk" else base_t
    ens_t = a*src_t + b*v94_t + c*v78_t + d*v35_t
    out = DATA_DIR / f"submission_v95_hybrid_{best_h[2]}{a:.2f}_v94{b:.2f}_v78{c:.2f}_v35{d:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best ({best_h[2]} src): ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}) → OOF {best_h[0]:.4f}  Δ {best_h[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# ============================================================
# 5) v97 추가 hybrid: stacker + v94 + v97 + v78 + v35 (5 ways)
# ============================================================
print("\n--- v95_v97 hybrid: stacker/base + v94 + v97 + v78 + v35 ---")
best_h97 = (best_h[0], None, None)
for src_o, src_t, name in [(oof_stk, test_stk, "stk"), (base_o, base_t, "base")]:
    for a in np.linspace(0.0, 0.6, 13):
        for b in np.linspace(0, 1-a, 13):
            for c in np.linspace(0, 1-a-b, 11):
                for d in np.linspace(0, 1-a-b-c, 7):
                    e = 1 - a - b - c - d
                    if e < 0: continue
                    ens = a*src_o + b*v94_o + c*v97_o + d*v78_o + e*v35_o
                    r = rh(ens)
                    if r > best_h97[0]: best_h97 = (r, (a, b, c, d, e), name)
if best_h97[1]:
    a, b, c, d, e = best_h97[1]
    src_t = test_stk if best_h97[2] == "stk" else base_t
    ens_t = a*src_t + b*v94_t + c*v97_t + d*v78_t + e*v35_t
    out = DATA_DIR / f"submission_v98_5w_{best_h97[2]}{a:.2f}_v94{b:.2f}_v97{c:.2f}_v78{d:.2f}_v35{e:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best ({best_h97[2]} src): ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}, {e:.2f}) → OOF {best_h97[0]:.4f}  Δ {best_h97[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# v94 + v97 single paradigm pair (paradigm 최강 두 단독)
print("\n--- v94 + v97 paradigm pair ---")
best_p97 = (max(rh(v94_o), rh(v97_o)), None)
for a in np.linspace(0, 1, 41):
    ens = a*v94_o + (1-a)*v97_o
    r = rh(ens)
    if r > best_p97[0]: best_p97 = (r, a)
print(f"  best: w_v94={best_p97[1]:.3f}  OOF {best_p97[0]:.4f}")

# v97_cap1p5 (단일 최강 0.6749) 활용 새 hybrid
print("\n--- v99 hybrid: stacker/base + v97_cap1p5 + v94 + v78 + v35 ---")
best_h99 = (best_h97[0] if best_h97[1] else max(rh_stk, rh_base), None, None)
for src_o, src_t, name in [(oof_stk, test_stk, "stk"), (base_o, base_t, "base")]:
    for a in np.linspace(0.0, 1.0, 21):
        for b in np.linspace(0, 1-a, 13):
            for c in np.linspace(0, 1-a-b, 11):
                for d in np.linspace(0, 1-a-b-c, 7):
                    e = 1 - a - b - c - d
                    if e < 0: continue
                    ens = a*src_o + b*v97c5_o + c*v94_o + d*v78_o + e*v35_o
                    r = rh(ens)
                    if r > best_h99[0]: best_h99 = (r, (a, b, c, d, e), name)
if best_h99[1]:
    a, b, c, d, e = best_h99[1]
    src_t = test_stk if best_h99[2] == "stk" else base_t
    ens_t = a*src_t + b*v97c5_t + c*v94_t + d*v78_t + e*v35_t
    out = DATA_DIR / f"submission_v99_5w_{best_h99[2]}{a:.2f}_v97c5{b:.2f}_v94{c:.2f}_v78{d:.2f}_v35{e:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best ({best_h99[2]} src): ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}, {e:.2f}) → OOF {best_h99[0]:.4f}  Δ {best_h99[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# v102 = 6-way grid with v97_cap1p5 + v94 + v101_cap1p5 + v78 + v35 + base (all paradigm)
print("\n--- v102 6-way grid: base + v97c5 + v94 + v101c5 + v78 + v35 ---")
best_v102 = (best_h99[0] if best_h99[1] else max(rh_stk, rh_base), None)
for a in np.linspace(0.0, 0.6, 13):
    for b in np.linspace(0, 1-a, 11):
        for c in np.linspace(0, 1-a-b, 9):
            for d in np.linspace(0, 1-a-b-c, 7):
                for e in np.linspace(0, 1-a-b-c-d, 5):
                    f = 1 - a - b - c - d - e
                    if f < 0: continue
                    ens = a*base_o + b*v97c5_o + c*v94_o + d*v101c5_o + e*v78_o + f*v35_o
                    r = rh(ens)
                    if r > best_v102[0]: best_v102 = (r, (a, b, c, d, e, f))
if best_v102[1]:
    a, b, c, d, e, f = best_v102[1]
    ens_t = a*base_t + b*v97c5_t + c*v94_t + d*v101c5_t + e*v78_t + f*v35_t
    out = DATA_DIR / f"submission_v102_6w_base{a:.2f}_v97c5{b:.2f}_v94{c:.2f}_v101c5{d:.2f}_v78{e:.2f}_v35{f:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best: ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}, {e:.2f}, {f:.2f}) → OOF {best_v102[0]:.4f}  Δ {best_v102[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("=== v95 grid summary ===")
print("=" * 60)
print(f"  v48 3-way base OOF: {rh_base:.4f}  (= v82 = LB 0.6876 검증)")
print(f"  v94 단독          : {rh(v94_o):.4f}")
print(f"  5-way blend       : {best5[0]:.4f}  (Δ {best5[0]-rh_base:+.4f})")
print(f"  paradigm trio     : {best_t[0]:.4f}")
print(f"  {len(stk_keys)}m SoftStacker  : {rh_stk:.4f}  (Δ {rh_stk-rh_base:+.4f})")
print(f"  hybrid best       : {best_h[0]:.4f}  (Δ {best_h[0]-rh_base:+.4f})")

entry = {"version": "v95_grids", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
         "rh_v48_3way_base": float(rh_base),
         "rh_v94": float(rh(v94_o)),
         "rh_v91": float(rh(v91_o)),
         "rh_5way": float(best5[0]),
         "rh_trio": float(best_t[0]),
         "rh_stacker": float(rh_stk),
         "rh_hybrid": float(best_h[0])}
log_path = PROJECT / "run_log.json"
logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
logs.append(entry)
json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

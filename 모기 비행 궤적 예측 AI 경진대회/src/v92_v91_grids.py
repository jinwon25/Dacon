"""v92_v91_grids.py — v91 blend/stacker quick grid.

v91 OOF 0.6734 (단일 최강, v78 0.6730 위, v35 0.6725 위).
v90 mirror base OOF 0.6643.

빠른 분석:
  1) v91 단독 cap variants (cap 0.5/1.0/1.5 — 학습은 안 함, 분석만)
  2) 4-way grid: v48 3-way + v91 + v35 + v78
  3) 5-way grid: + v90 base
  4) 12m SoftStacker = v48 stacker pool + v91
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
DATA_DIR = PROJECT / "data"
CACHE_DIR = PROJECT / "data/cache"

# ============================================================
# Load all OOFs
# ============================================================
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


# ============================================================
# Load model pool
# ============================================================
pool = {}
def load_v(name, oof_key, test_key):
    st = np.load(CACHE_DIR / f"{name}_state.npz")
    pool[name] = (st[oof_key], st[test_key])

# v35, v44, v48, v46, v53 (기본 stacker pool)
load_v("v35", "oof_v35", "test_v35")
load_v("v44", "oof_v44", "test_v44")
load_v("v48", "oof_v48", "test_v48")
load_v("v46", "oof_v46", "test_v46")
load_v("v53", "oof_v53", "test_v53")

# v78 (boundary on v77)
st78 = np.load(CACHE_DIR / "v78_state.npz")
pool["v78"] = (st78["oof_v78"], st78["test_v78"])

# v91 (boundary on v90)
st91 = np.load(CACHE_DIR / "v91_state.npz")
pool["v91"] = (st91["oof_v91"], st91["test_v91"])

# v90 base
st90 = np.load(CACHE_DIR / "v90_mirror_state.npz")
oof_v90 = kalman_train + st90["oof"] * ALPHA
test_v90 = kalman_test + st90["test_pred"] * ALPHA
pool["v90"] = (oof_v90, test_v90)

# v52 cap variants
for tag, cache_name in [("v52_cap0p5", "v52_cap0p5_state.npz"),
                        ("v52_cap1p5", "v52_cap1p5_state.npz")]:
    try:
        st = np.load(CACHE_DIR / cache_name)
        keys = list(st.keys())
        # 보통 oof_v52_cap0p5 같은 구조 아닐 수 있음, 동적 처리
        oof_k = [k for k in keys if "oof" in k][0]
        test_k = [k for k in keys if "test" in k][0]
        pool[tag] = (st[oof_k], st[test_k])
    except Exception as e:
        print(f"[skip] {tag}: {e}")

print("=" * 60)
print("Pool OOF R-Hit")
print("=" * 60)
for k, (o, _) in pool.items():
    print(f"  {k:15s}: {rh(o):.4f}")

# ============================================================
# 1) v48 3-way base + v91 2-way blend
# ============================================================
v48_o, v48_t = pool["v48"]
v46_o, v46_t = pool["v46"]
v35_o, v35_t = pool["v35"]
v78_o, v78_t = pool["v78"]
v91_o, v91_t = pool["v91"]
v90_o, v90_t = pool["v90"]

base_o = 0.70*v48_o + 0.12*v46_o + 0.18*v35_o
base_t = 0.70*v48_t + 0.12*v46_t + 0.18*v35_t
rh_base = rh(base_o)
print(f"\nv48 3-way base OOF: {rh_base:.4f}")

# ============================================================
# 2) 4-way grid: base + v91 + v35 + v78
# ============================================================
print("\n--- 4-way grid: base + v91 + v35 + v78 ---")
best = (rh_base, None)
for a in np.linspace(0.3, 1.0, 15):
    for b in np.linspace(0, 1-a, 11):
        for c in np.linspace(0, 1-a-b, 6):
            d = 1 - a - b - c
            if d < 0: continue
            ens = a*base_o + b*v91_o + c*v35_o + d*v78_o
            r = rh(ens)
            if r > best[0]: best = (r, (a, b, c, d))
if best[1]:
    a, b, c, d = best[1]
    ens_t = a*base_t + b*v91_t + c*v35_t + d*v78_t
    out = DATA_DIR / f"submission_v92_4w_base{a:.2f}_v91{b:.2f}_v35{c:.2f}_v78{d:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best: a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f} → OOF {best[0]:.4f}  Δ +{best[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# ============================================================
# 3) 5-way grid: base + v91 + v35 + v78 + v90
# ============================================================
print("\n--- 5-way grid: base + v91 + v35 + v78 + v90 ---")
best5 = (rh_base, None)
for a in np.linspace(0.4, 1.0, 13):
    for b in np.linspace(0, 1-a, 9):
        for c in np.linspace(0, 1-a-b, 6):
            for d in np.linspace(0, 1-a-b-c, 5):
                e = 1 - a - b - c - d
                if e < 0: continue
                ens = a*base_o + b*v91_o + c*v35_o + d*v78_o + e*v90_o
                r = rh(ens)
                if r > best5[0]: best5 = (r, (a, b, c, d, e))
if best5[1]:
    a, b, c, d, e = best5[1]
    ens_t = a*base_t + b*v91_t + c*v35_t + d*v78_t + e*v90_t
    out = DATA_DIR / f"submission_v92_5w_base{a:.2f}_v91{b:.2f}_v35{c:.2f}_v78{d:.2f}_v90{e:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best: a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f} e={e:.2f} → OOF {best5[0]:.4f}  Δ +{best5[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# ============================================================
# 4) v91 + v78 only blend
# ============================================================
print("\n--- v91 + v78 (paradigm pair) ---")
best_p = (max(rh(v91_o), rh(v78_o)), None)
for a in np.linspace(0, 1, 41):
    ens = a*v91_o + (1-a)*v78_o
    r = rh(ens)
    if r > best_p[0]: best_p = (r, a)
if best_p[1] is not None:
    print(f"  best: w_v91={best_p[1]:.3f}  OOF {best_p[0]:.4f}")

# ============================================================
# 5) 12m SoftStacker (NN selector)
# ============================================================
print("\n" + "=" * 60)
print("12m SoftStacker: v48 + v46 + v35 + v44 + v78 + v91 + v90 + cap variants")
print("=" * 60)

stk_keys = ["v48", "v46", "v35", "v44", "v78", "v91", "v90"]
for k in ["v52_cap0p5", "v52_cap1p5"]:
    if k in pool: stk_keys.append(k)
print(f"Stacker pool ({len(stk_keys)}m): {stk_keys}")

stk_oof = np.stack([pool[k][0] for k in stk_keys], axis=1)  # (N, M, 3)
stk_test = np.stack([pool[k][1] for k in stk_keys], axis=1)
M = stk_oof.shape[1]

# build features per sample: residuals to each model
def build_stk_feat(stk):
    N, M, _ = stk.shape
    mean = stk.mean(axis=1, keepdims=True)
    res = stk - mean  # (N, M, 3)
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
    oof_pred_all = np.zeros_like(stk_oof[:, 0])
    test_pred_seeds = []
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
                    w = model(x_tr[idx])  # (B, M)
                    pred = (w.unsqueeze(-1) * stk_tr[idx]).sum(dim=1)  # (B, 3)
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
                rh_va = rh(pred_va) if len(va) == len(y_train) else float((np.linalg.norm(pred_va - y_train[va], axis=-1) <= 0.01).mean())
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
        oof_pred_all = oof_pred if s == 0 else (oof_pred_all * s + oof_pred) / (s + 1)
        test_pred_seeds.append(np.mean(test_per_fold, axis=0))
    test_pred_all = np.mean(test_pred_seeds, axis=0)
    avg_w = np.mean(weight_log, axis=0)
    print(f"\nAvg weights:")
    for k, w in zip(stk_keys, avg_w):
        print(f"  {k:15s}: {w:.4f}")
    return oof_pred_all, test_pred_all


oof_stk, test_stk = train_stacker(seeds=3, folds=5, epochs=80, patience=15)
rh_stk = rh(oof_stk)
print(f"\n12m SoftStacker OOF: {rh_stk:.4f}  Δ vs v48 3-way {rh_base:.4f}: {rh_stk - rh_base:+.4f}")

if rh_stk > rh_base + 0.0005:
    out = DATA_DIR / "submission_v92_12m_stacker.csv"
    pd.DataFrame({"id": sub["id"], "x": test_stk[:,0], "y": test_stk[:,1], "z": test_stk[:,2]}).to_csv(out, index=False)
    print(f"  [submission] {out.name}")

# ============================================================
# 6) v92 hybrid: stacker + v91/v78/v35 grid
# ============================================================
print("\n--- v92 hybrid: stacker + v91/v78/v35 grid ---")
best_h = (max(rh_stk, rh_base, rh(v91_o)), None, None)
for src_o, src_t, name in [(oof_stk, test_stk, "stk"), (base_o, base_t, "base")]:
    src_rh = rh(src_o)
    for a in np.linspace(0.3, 1.0, 15):
        for b in np.linspace(0, 1-a, 11):
            for c in np.linspace(0, 1-a-b, 6):
                d = 1 - a - b - c
                if d < 0: continue
                ens = a*src_o + b*v91_o + c*v35_o + d*v78_o
                r = rh(ens)
                if r > best_h[0]: best_h = (r, (a, b, c, d), name)
if best_h[1]:
    a, b, c, d = best_h[1]
    src_t = test_stk if best_h[2] == "stk" else base_t
    ens_t = a*src_t + b*v91_t + c*v35_t + d*v78_t
    out = DATA_DIR / f"submission_v92_hybrid_{best_h[2]}{a:.2f}_v91{b:.2f}_v35{c:.2f}_v78{d:.2f}.csv"
    pd.DataFrame({"id": sub["id"], "x": ens_t[:,0], "y": ens_t[:,1], "z": ens_t[:,2]}).to_csv(out, index=False)
    print(f"  best ({best_h[2]} src): a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f} → OOF {best_h[0]:.4f}  Δ +{best_h[0]-rh_base:+.4f}")
    print(f"  [submission] {out.name}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("=== v92 grid summary ===")
print("=" * 60)
print(f"  v48 3-way base OOF: {rh_base:.4f}")
print(f"  v91 단독         : {rh(v91_o):.4f}")
print(f"  v78 단독         : {rh(v78_o):.4f}")
print(f"  v35 단독         : {rh(v35_o):.4f}")
print(f"  4-way blend best : {best[0]:.4f}  (Δ {best[0]-rh_base:+.4f})")
print(f"  5-way blend best : {best5[0]:.4f}  (Δ {best5[0]-rh_base:+.4f})")
print(f"  v91+v78 pair best: {best_p[0]:.4f}")
print(f"  12m SoftStacker  : {rh_stk:.4f}  (Δ {rh_stk-rh_base:+.4f})")
print(f"  v92 hybrid best  : {best_h[0]:.4f}  (Δ {best_h[0]-rh_base:+.4f})")

entry = {"version": "v92_grids", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
         "rh_v48_3way_base": float(rh_base),
         "rh_v91": float(rh(v91_o)),
         "rh_v78": float(rh(v78_o)),
         "rh_v90_base": float(rh(v90_o)),
         "rh_4way": float(best[0]),
         "rh_5way": float(best5[0]),
         "rh_v91_v78_pair": float(best_p[0]),
         "rh_12m_stacker": float(rh_stk),
         "rh_hybrid": float(best_h[0])}
log_path = PROJECT / "run_log.json"
logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
logs.append(entry)
json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

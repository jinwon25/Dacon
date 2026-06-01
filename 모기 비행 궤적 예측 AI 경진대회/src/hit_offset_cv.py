"""STEP B follow-up — speed-conditional offset의 leave-out CV 검증.

In-sample +0.0026 lift는 fit-on-same-data overfit. 진짜 lift는 KFold로 확인해야 함.
"""
from __future__ import annotations
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/cache"
DT = 0.040

xc = np.load(CACHE / "xtrain_xtest.npz")
X_train = xc["X_train"]
labels = pd.read_csv(ROOT / "data" / "train_labels.csv")
y_train = labels.set_index("id").loc[[f"TRAIN_{i:05d}" for i in range(1,10001)]][["x","y","z"]].values.astype(np.float64)
oof = np.load(CACHE / "v112_v107_diverse_weights.npz")["oof_pred"].astype(np.float64)

v_last = (X_train[:, -1] - X_train[:, -2]) / DT
sp_last = np.linalg.norm(v_last, axis=-1)

def best_delta_for_bin(o_b, y_b, span_mm=2, step_mm=0.25):
    rng = np.arange(-span_mm, span_mm+0.001, step_mm) * 1e-3
    base = (np.linalg.norm(o_b - y_b, axis=-1) <= 0.01).mean()
    best = (base, np.array([0.0,0.0,0.0]))
    for dx in rng:
        for dy in rng:
            for dz in rng:
                d = np.linalg.norm(o_b + np.array([dx,dy,dz]) - y_b, axis=-1)
                h = (d<=0.01).mean()
                if h > best[0]: best = (h, np.array([dx,dy,dz]))
    return best[1]

# 5-fold: bin edges는 전체에서 고정 (per-fold bin def 일관)
bins = np.quantile(sp_last, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
bin_idx = np.clip(np.searchsorted(bins[1:-1], sp_last), 0, 4)
print(f"bins: {bins}")

base_hit = (np.linalg.norm(oof - y_train, axis=-1) <= 0.01).mean()
print(f"base v112 hit = {base_hit:.4f}")

oof_corrected = oof.copy()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(kf.split(np.arange(10000))):
    for b in range(5):
        m_tr = (bin_idx[tr] == b)
        if m_tr.sum() < 50: continue
        # estimate delta on tr ∩ bin b
        delta_b = best_delta_for_bin(oof[tr][m_tr], y_train[tr][m_tr])
        # apply to va ∩ bin b
        m_va = (bin_idx[va] == b)
        idxs = va[m_va]
        oof_corrected[idxs] += delta_b
    fold_hit = (np.linalg.norm(oof_corrected[va] - y_train[va], axis=-1) <= 0.01).mean()
    fold_base = (np.linalg.norm(oof[va] - y_train[va], axis=-1) <= 0.01).mean()
    print(f"  fold {fold}: base {fold_base:.4f} → cv-corrected {fold_hit:.4f}  Δ {fold_hit-fold_base:+.4f}")

cv_hit = (np.linalg.norm(oof_corrected - y_train, axis=-1) <= 0.01).mean()
print(f"\nCV-honest speed-conditional hit = {cv_hit:.4f}  Δ vs base = {cv_hit - base_hit:+.4f}")
print(f"In-sample (last script) hit = 0.6794  Δ = +0.0026  (over-fit if CV-honest << in-sample)")

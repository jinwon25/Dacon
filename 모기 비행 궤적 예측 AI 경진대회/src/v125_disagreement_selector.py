"""v125_disagreement_selector.py — 모델 disagreement 기반 selector.

핵심 아이디어:
  - 메타특징만으로 only_v112 vs only_v120 식별: Δz < 0.14 → 거의 불가능 (EDA 확인)
  - 그러나 두 prediction의 차이 |v112 - v120| 자체가 강력한 신호일 가능성
  - 거리 큰 sample = 두 paradigm 불일치 = 누군가 틀림
  - 어느 쪽이 옳은지를 메타+disagreement+prediction 좌표 정보로 light MLP 학습

학습 설계:
  - X: [meta_features (8), |v112-v120| (1), v112_pred_local (3), v120_pred_local (3),
        last_obs (3), v_last (3), |v112-last_obs|, |v120-last_obs|]  = ~24 features
  - y_soft: target = (d_v112 - d_v120) / (|v112-v120| + eps), in [-1, 1]
           (양수: v120이 더 가까움)
  - 또는 y_hard: pick_v120 = (d_v120 < d_v112)  binary

  - 5-fold OOF training
  - 출력: per-sample p(v120) ∈ [0,1]
  - 최종: blend = (1-p) * v112 + p * v120
"""
from __future__ import annotations
import sys, time
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

SCRIPT = Path(__file__).resolve().parent
PROJ = SCRIPT.parent
CACHE = PROJ / "data/cache"
OPEN = PROJ / "data"
OUT = PROJ / "data"

DT = 0.040
T_PRED = 0.080


def meta_feats(X):
    delta = np.diff(X, axis=1) / DT
    accel = np.diff(delta, axis=1) / DT
    jerk = np.diff(accel, axis=1) / DT
    speed = np.linalg.norm(delta, axis=-1)
    va, vb = delta[:, :-1], delta[:, 1:]
    na = np.linalg.norm(va, axis=-1) + 1e-12
    nb = np.linalg.norm(vb, axis=-1) + 1e-12
    cos = np.clip((va * vb).sum(-1) / (na * nb), -1, 1)
    ang = np.arccos(cos)
    speed_diff = np.diff(speed, axis=1)
    feats = np.stack([
        np.linalg.norm(accel, axis=-1).mean(axis=1),
        np.linalg.norm(jerk, axis=-1).mean(axis=1),
        speed[:, -1], speed.max(axis=1), speed.mean(axis=1),
        ang.max(axis=1), ang.mean(axis=1),
        (-speed_diff).max(axis=1),
    ], axis=1).astype(np.float32)
    return feats  # (N, 8)


class SelectorMLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def build_features(X, pred_a, pred_b):
    """v112 = pred_a, v120 = pred_b 기준. last_obs 좌표 기반 normalize."""
    N = X.shape[0]
    last_obs = X[:, -1]
    v_last = (X[:, -1] - X[:, -2]) / DT
    speed_l = np.linalg.norm(v_last, axis=-1)
    # local rel coords
    a_local = pred_a - last_obs
    b_local = pred_b - last_obs
    diff = pred_a - pred_b
    diff_norm = np.linalg.norm(diff, axis=-1)
    meta = meta_feats(X)
    a_norm = np.linalg.norm(a_local, axis=-1)
    b_norm = np.linalg.norm(b_local, axis=-1)
    cos_ab = (a_local * b_local).sum(-1) / (a_norm * b_norm + 1e-12)
    feats = np.concatenate([
        meta,                             # 8
        diff_norm[:, None],              # 1
        a_local, b_local,                # 6
        v_last,                          # 3
        speed_l[:, None],                # 1
        a_norm[:, None], b_norm[:, None], # 2
        cos_ab[:, None],                  # 1
    ], axis=-1).astype(np.float32)
    return feats  # (N, 22)


def main():
    X_tr = np.load(CACHE / "xtrain_xtest.npz")["X_train"]
    X_te = np.load(CACHE / "xtrain_xtest.npz")["X_test"]
    y_tr = pd.read_csv(OPEN / "train_labels.csv").sort_values("id")[["x","y","z"]].values

    c12 = np.load(CACHE / "v112_v107_diverse_weights.npz", allow_pickle=True)
    c20 = np.load(CACHE / "v120_full_state.npz", allow_pickle=True)
    oof_a = c12["oof_pred"].astype(np.float32)
    oof_b = c20["oof_global"].astype(np.float32)
    test_a = c12["test_pred"].astype(np.float32)
    test_b = c20["test_global"].astype(np.float32)

    d_a = np.linalg.norm(oof_a - y_tr, axis=1)
    d_b = np.linalg.norm(oof_b - y_tr, axis=1)
    pick_b = (d_b < d_a).astype(np.float32)  # 1 = v120 더 정확
    print(f"v112 hit: {(d_a<0.01).mean():.4f}  v120 hit: {(d_b<0.01).mean():.4f}")
    print(f"pick_v120 rate: {pick_b.mean():.4f}  oracle hit: {(np.minimum(d_a,d_b)<0.01).mean():.4f}")

    feats_tr = build_features(X_tr, oof_a, oof_b)
    feats_te = build_features(X_te, test_a, test_b)
    print(f"feats shape: {feats_tr.shape}")

    # 5-fold OOF
    N = len(y_tr)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof_p = np.zeros(N, dtype=np.float32)
    test_p_sum = np.zeros(len(X_te), dtype=np.float32)

    device = torch.device("cpu")
    in_dim = feats_tr.shape[1]
    print(f"\n[selector] training {5}-fold MLP, in_dim={in_dim}, device={device}")
    t0 = time.time()

    for fi, (tr, va) in enumerate(kf.split(np.arange(N))):
        sc = StandardScaler().fit(feats_tr[tr])
        f_tr = sc.transform(feats_tr[tr]).astype(np.float32)
        f_va = sc.transform(feats_tr[va]).astype(np.float32)
        f_te = sc.transform(feats_te).astype(np.float32)
        y_tr_pick = pick_b[tr]
        y_va_pick = pick_b[va]

        x_tr_t = torch.from_numpy(f_tr)
        x_va_t = torch.from_numpy(f_va)
        x_te_t = torch.from_numpy(f_te)
        y_tr_t = torch.from_numpy(y_tr_pick)
        y_va_t = torch.from_numpy(y_va_pick)

        # train across multiple seeds for stability
        n_seeds = 3
        va_p_sum = np.zeros(len(va), dtype=np.float32)
        te_p_sum = np.zeros(len(X_te), dtype=np.float32)
        for sd in range(n_seeds):
            torch.manual_seed(sd); np.random.seed(sd)
            model = SelectorMLP(in_dim, hidden=64).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
            best_auc = -1; bad = 0; patience = 8; best_state = None
            for ep in range(1, 41):
                model.train()
                perm = torch.randperm(len(tr))
                ls = 0; nb = 0
                for s in range(0, len(tr), 256):
                    idx = perm[s:s+256]
                    logits = model(x_tr_t[idx])
                    loss = F.binary_cross_entropy_with_logits(logits, y_tr_t[idx])
                    opt.zero_grad(); loss.backward()
                    opt.step()
                    ls += loss.item(); nb += 1
                sch.step()
                model.eval()
                with torch.no_grad():
                    p_va = torch.sigmoid(model(x_va_t)).cpu().numpy()
                # rank metric: how well does p rank pick_b
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y_va_pick, p_va)
                except Exception:
                    auc = 0.5
                if auc > best_auc:
                    best_auc = auc; bad = 0
                    best_state = {k:v.clone() for k,v in model.state_dict().items()}
                else:
                    bad += 1
                if bad >= patience:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                va_p_sum += torch.sigmoid(model(x_va_t)).cpu().numpy() / n_seeds
                te_p_sum += torch.sigmoid(model(x_te_t)).cpu().numpy() / n_seeds
            print(f"  fold{fi} seed{sd} best AUC={best_auc:.4f}")
        oof_p[va] = va_p_sum
        test_p_sum += te_p_sum / 5

    print(f"\n[selector] total {(time.time()-t0)/60:.2f}m")
    # evaluate
    from sklearn.metrics import roc_auc_score
    auc_total = roc_auc_score(pick_b, oof_p)
    acc_total = ((oof_p > 0.5) == (pick_b > 0.5)).mean()
    print(f"OOF AUC = {auc_total:.4f}")
    print(f"OOF acc (thresh 0.5) = {acc_total:.4f}")

    # blends
    print("\n=== Soft blend (continuous p) ===")
    for clip in [(0.0,1.0), (0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6)]:
        p_clip = np.clip(oof_p, *clip)
        blend = (1 - p_clip[:,None]) * oof_a + p_clip[:,None] * oof_b
        h = (np.linalg.norm(blend - y_tr, axis=1) < 0.01).mean()
        print(f"  clip {clip}: hit = {h:.4f}")

    print("\n=== Hard threshold pick ===")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
        pick = (oof_p > thr).astype(np.float32)[:, None]
        blend = (1 - pick) * oof_a + pick * oof_b
        h = (np.linalg.norm(blend - y_tr, axis=1) < 0.01).mean()
        print(f"  thr={thr:.2f}: hit = {h:.4f}  pick_rate={pick.mean():.3f}")

    print("\n=== Selector + v122c blend (use selector as gate on top of v122c) ===")
    c22 = np.load(CACHE / "v122c_v121diverse_weights.npz", allow_pickle=True)
    oof_v22c = c22["oof_pred"]; test_v22c = c22["test_pred"]
    # 약하게 selector → v122c와 섞기
    for w_sel in [0.1, 0.2, 0.3, 0.5]:
        p_clip = np.clip(oof_p, 0.2, 0.8)
        oof_sel = (1 - p_clip[:,None]) * oof_a + p_clip[:,None] * oof_b
        oof_final = (1 - w_sel) * oof_v22c + w_sel * oof_sel
        h = (np.linalg.norm(oof_final - y_tr, axis=1) < 0.01).mean()
        print(f"  v122c + w_sel={w_sel:.1f} * selector_blend: hit = {h:.4f}")

    # save
    np.savez(CACHE / "v125_selector_state.npz",
              oof_p=oof_p, test_p=test_p_sum,
              oof_pick_b=pick_b)
    print(f"\nsaved cache/v125_selector_state.npz")

if __name__ == "__main__":
    main()

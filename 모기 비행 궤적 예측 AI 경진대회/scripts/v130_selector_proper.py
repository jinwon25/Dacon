"""v130_selector_proper.py — Disagreement/competence-based selector (CV-honest).

EDA가 미검증으로 남긴 카드. v125는 meta-only (AUC 0.5562 dead).
이번엔 |cand_i - cand_j| disagreement + per-axis pred geometry + meta 전부 사용.

두 가지 메커니즘 CV-정직 평가:
  (1) ROUTING: 후보 중 per-sample 가장 가까운 것을 분류기로 예측 → 라우팅.
  (2) COMPETENCE: 각 후보의 per-sample distance를 회귀로 예측 → argmin 선택.
  (3) STACK-OVERRIDE: v122c base에서 selector가 "v122c miss & 다른 후보 hit" 라고
       강하게 판단한 샘플만 override.

baseline = v122c OOF 0.6769. 후보 set oracle = 0.7135.
"""
from __future__ import annotations
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

from v110_de_ensemble import load_pool
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor

CACHE = Path("cache")

def hit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())

def main():
    pool, y = load_pool(include_mdn=True)
    oof = {p[0]: p[1].astype(np.float64) for p in pool}
    test = {p[0]: p[2].astype(np.float64) for p in pool}
    for f, tag in [("v112_v107_diverse_weights.npz", "v112"),
                   ("v122c_v121diverse_weights.npz", "v122c")]:
        st = np.load(CACHE / f); oof[tag] = st["oof_pred"]; test[tag] = st["test_pred"]

    meta = np.load(CACHE / "meta_features.npz")
    mfeat_keys = ["speed_last3","speed_mean","speed_max","accel_mean","accel_max",
                  "jerk_mean","turn_mean","turn_max","kappa_mean","kappa_max","dir_std"]
    M = np.stack([meta[k] for k in mfeat_keys], axis=1)  # (N,11)

    N = y.shape[0]
    print(f"N={N}  baseline v122c={hit(oof['v122c'],y):.4f}")

    # candidate routing set (diverse, strong)
    CANDS = ["v122c", "v120", "v120_big", "v108_15_15_10", "v97c5", "v53"]
    P = np.stack([oof[c] for c in CANDS], axis=0)          # (C,N,3)
    C = len(CANDS)
    D = np.linalg.norm(P - y[None], axis=-1)               # (C,N) true dist
    oracle = float((D.min(0) <= 0.01).mean())
    print(f"candset={CANDS}\n  per-cand oof={[round(hit(oof[c],y),4) for c in CANDS]}")
    print(f"  ORACLE={oracle:.4f}")

    # ---- feature construction ----
    # base geometry: each cand's predicted displacement from last obs is implicit;
    # we use pairwise disagreements + meta + per-cand spread
    feats = [M]
    # pairwise L2 disagreement among candidates
    for i in range(C):
        for j in range(i+1, C):
            dij = np.linalg.norm(P[i]-P[j], axis=-1, keepdims=True)
            feats.append(dij)
    # per-axis disagreement of v122c vs v120 (paradigm pair)
    feats.append(P[CANDS.index("v122c")] - P[CANDS.index("v120")])   # (N,3)
    feats.append(P[CANDS.index("v122c")] - P[CANDS.index("v108_15_15_10")])
    # spread: std across candidates per axis
    feats.append(P.std(0))                                            # (N,3)
    # mean disagreement to centroid
    centroid = P.mean(0)
    feats.append(np.linalg.norm(P - centroid[None], axis=-1).T)       # (N,C)
    Xf = np.concatenate(feats, axis=1)
    print(f"  feature dim={Xf.shape[1]}")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # ============ (2) COMPETENCE: regress per-cand distance, argmin ============
    pred_D = np.zeros((C, N))
    for c in range(C):
        for tr, va in kf.split(np.arange(N)):
            reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                 max_depth=4, l2_regularization=1.0,
                                                 random_state=0)
            reg.fit(Xf[tr], D[c, tr])
            pred_D[c, va] = reg.predict(Xf[va])
    route = pred_D.argmin(0)
    routed = P[route, np.arange(N)]
    print(f"\n[COMPETENCE argmin] routed oof={hit(routed,y):.4f}  (Δ={hit(routed,y)-hit(oof['v122c'],y):+.4f})")
    # accuracy of picking the truly-best
    true_best = D.argmin(0)
    print(f"  route-acc(vs true argmin)={float((route==true_best).mean()):.4f}")

    # ============ (1) ROUTING classifier: predict argmin class ============
    pred_cls = np.zeros(N, dtype=int)
    for tr, va in kf.split(np.arange(N)):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_depth=4, l2_regularization=1.0,
                                              random_state=0)
        clf.fit(Xf[tr], true_best[tr])
        pred_cls[va] = clf.predict(Xf[va])
    routed2 = P[pred_cls, np.arange(N)]
    print(f"[ROUTING clf] routed oof={hit(routed2,y):.4f}  (Δ={hit(routed2,y)-hit(oof['v122c'],y):+.4f})")
    print(f"  clf-acc={float((pred_cls==true_best).mean()):.4f}")

    # ============ (3) STACK-OVERRIDE: only override v122c when confident ============
    # binary: does ANY non-v122c cand hit while v122c misses?
    base_i = CANDS.index("v122c")
    base_hit = (D[base_i] <= 0.01)
    alt_best = np.delete(D, base_i, axis=0).min(0)
    override_gain = (~base_hit) & (alt_best <= 0.01)   # samples where override COULD help
    print(f"\n  v122c miss & alt hit (override-able) = {int(override_gain.sum())} ({override_gain.mean():.4f})")
    # predict competence-argmin only among non-base; override base when predicted alt hits
    # use predicted distance: if min predicted alt dist << predicted base dist, override
    alt_idx = [i for i in range(C) if i != base_i]
    pred_alt_min = pred_D[alt_idx].min(0)
    pred_alt_arg = np.array(alt_idx)[pred_D[alt_idx].argmin(0)]
    for thr in [0.006, 0.007, 0.008, 0.009]:
        do_ov = (pred_alt_min < thr) & (pred_D[base_i] > 0.010)
        out = oof["v122c"].copy()
        out[do_ov] = P[pred_alt_arg[do_ov], np.arange(N)[do_ov]]
        print(f"  [override thr={thr}] n_ov={int(do_ov.sum())}  oof={hit(out,y):.4f}  (Δ={hit(out,y)-hit(oof['v122c'],y):+.4f})")

    # ============ soft per-sample blend weight (v122c vs v120) ============
    # learn w in [0,1]: pred = w*v120 + (1-w)*v122c, target hit
    # proxy: regress indicator(v120 closer) then soft-blend
    closer120 = (np.linalg.norm(oof["v120"]-y,axis=-1) < np.linalg.norm(oof["v122c"]-y,axis=-1)).astype(int)
    prob = np.zeros(N)
    for tr, va in kf.split(np.arange(N)):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(Xf[tr], closer120[tr]); prob[va] = clf.predict_proba(Xf[va])[:,1]
    print(f"\n  closer120 base-rate={closer120.mean():.4f}  clf AUC-proxy acc={float(((prob>0.5).astype(int)==closer120).mean()):.4f}")
    for a in [0.3,0.5,0.7,1.0]:
        w = (prob>0.5)*a
        out = (1-w)[:,None]*oof["v122c"] + w[:,None]*oof["v120"]
        print(f"  [softblend a={a}] oof={hit(out,y):.4f}  (Δ={hit(out,y)-hit(oof['v122c'],y):+.4f})")

if __name__ == "__main__":
    main()

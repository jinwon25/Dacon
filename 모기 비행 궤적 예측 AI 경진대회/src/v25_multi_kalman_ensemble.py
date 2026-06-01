"""v25_multi_kalman_ensemble.py — Multi-Kalman + axis-aware ensemble.

v23/v16 ensemble (v24)에서 발견한 핵심: **두 모델이 축별 강점 다름** (per_axis α=[0.85, 0.5, 0.05]).
v25는 이걸 더 적극적으로 활용한다:

  1. **다양한 Kalman base predictors** (parameter-free, 학습 불필요)
     - CV (등속) σ_obs ∈ {0.1mm, 0.3mm, 1mm, 3mm}
     - CA (등가속) σ_obs ∈ {0.3mm, 1mm}
     - 합쳐서 6 base predictors
  2. **각 base의 axis별 OOF rhit 자동 측정** → 어떤 base가 어떤 축에 강한지 정량화
  3. **Axis-wise per-base mixing**: 각 축마다 base 선택을 학습 (per-axis OOF best)
  4. **v23 + v16 + multi-Kalman 통합 ensemble** (총 8 base)
  5. **Adversarial validation**: train/test shift 측정 → 진단

사용법:
  python scripts/v25_multi_kalman_ensemble.py --v23-mode fast

전제:
  - cache/v23_state_{mode}.npz + cache/kalman.npz + cache/xtrain_xtest.npz
  - archive/v16_stack_oof.npz
출력:
  - open/submission_v25_cpu_{mode}.csv
  - cache/v25_adv_val.npz (adversarial validation 결과)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"

DT = 0.040
T_PRED = 0.080


# ============================================================
# Kalman variants (parameter-free, 학습 불필요)
# ============================================================
def kalman_cv(X, sigma_obs, sigma_proc=1.0, P0=1.0):
    N, T, _ = X.shape
    F = np.array([[1, DT], [0, 1]])
    F_pred = np.array([[1, T_PRED], [0, 1]])
    Q = sigma_proc ** 2 * np.array([[DT**4/4, DT**3/2], [DT**3/2, DT**2]])
    R = sigma_obs ** 2
    pred = np.zeros((N, 3))
    for j in range(3):
        z_all = X[:, :, j]
        state = np.zeros((N, 2)); state[:, 0] = z_all[:, 0]
        P = np.eye(2) * P0
        for t in range(1, T):
            state = state @ F.T
            P = F @ P @ F.T + Q
            innov = z_all[:, t] - state[:, 0]
            S = P[0, 0] + R; K = P[:, 0] / S
            state = state + np.outer(innov, K)
            P = P - np.outer(K, P[0, :])
        pred[:, j] = (state @ F_pred.T)[:, 0]
    return pred


def kalman_ca(X, sigma_obs, sigma_proc=1.0, P0=1.0):
    """Constant acceleration model — 가속도 추가."""
    N, T, _ = X.shape
    F = np.array([[1, DT, DT**2/2], [0, 1, DT], [0, 0, 1]])
    F_pred = np.array([[1, T_PRED, T_PRED**2/2], [0, 1, T_PRED], [0, 0, 1]])
    Q = sigma_proc ** 2 * np.array([[DT**4/4, DT**3/2, DT**2/2],
                                      [DT**3/2, DT**2,   DT    ],
                                      [DT**2/2, DT,      1     ]])
    R = sigma_obs ** 2
    pred = np.zeros((N, 3))
    for j in range(3):
        z_all = X[:, :, j]
        state = np.zeros((N, 3)); state[:, 0] = z_all[:, 0]
        P = np.eye(3) * P0
        for t in range(1, T):
            state = state @ F.T
            P = F @ P @ F.T + Q
            innov = z_all[:, t] - state[:, 0]
            S = P[0, 0] + R; K = P[:, 0] / S
            state = state + np.outer(innov, K)
            P = P - np.outer(K, P[0, :])
        pred[:, j] = (state @ F_pred.T)[:, 0]
    return pred


# ============================================================
# Helper
# ============================================================
def rhit(p, y, mask=None, thr=0.01):
    if mask is not None: p, y = p[mask], y[mask]
    return float((np.linalg.norm(p - y, axis=-1) <= thr).mean())


def axis_rhit(p, y, mask=None, thr=0.01):
    """축별 hit율 (각 축이 1cm 안에 있을 확률 — 절대 distance 기준이 아님)."""
    if mask is not None: p, y = p[mask], y[mask]
    return [float((np.abs(p[:, j] - y[:, j]) <= thr).mean()) for j in range(3)]


# ============================================================
# Adversarial validation
# ============================================================
def adversarial_validation(feat_train, feat_test, n_folds=5):
    """train vs test 구별 LGB classifier → covariate shift 진단."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("[adv] lightgbm not available, skipping")
        return None

    feat = np.concatenate([feat_train, feat_test], axis=0)
    label = np.concatenate([np.zeros(len(feat_train)), np.ones(len(feat_test))])

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(feat))
    feat, label = feat[perm], label[perm]

    kf = KFold(n_folds, shuffle=True, random_state=0)
    aucs, oof = [], np.zeros(len(label))
    for fi, (tr, va) in enumerate(kf.split(feat)):
        gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=300,
                                   min_child_samples=20, verbose=-1, random_state=0)
        gbm.fit(feat[tr], label[tr])
        proba = gbm.predict_proba(feat[va])[:, 1]
        oof[va] = proba
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(label[va], proba)
        aucs.append(auc)

    print(f"[adv] AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "auc_per_fold": aucs, "oof_proba": oof[np.argsort(perm)]}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23-mode", default="fast", choices=["micro", "fast", "full"])
    args = parser.parse_args()
    mode = args.v23_mode

    print("=" * 60)
    print(f"v25 Multi-Kalman + axis-aware ensemble (v23={mode})")
    print("=" * 60)

    # --- Load base data ---
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    # --- Multi-Kalman base predictors ---
    print("\n=== Multi-Kalman base predictors ===")
    cache_kalmans = CACHE_DIR / "kalmans_multi.npz"
    if cache_kalmans.exists():
        kk = np.load(cache_kalmans)
        kalman_specs = [k for k in kk.files if k.startswith("k_")]
        bases_train = {k[2:]: kk[k] for k in kalman_specs if k.endswith("_train")}
        bases_test  = {k[2:].replace("_test", ""): kk[k] for k in kalman_specs if k.endswith("_test")}
        # 재구성
        bases_train = {}
        bases_test = {}
        for k in kk.files:
            if k.startswith("k_") and k.endswith("_train"):
                name = k[2:-6]
                bases_train[name] = kk[k]
            elif k.startswith("k_") and k.endswith("_test"):
                name = k[2:-5]
                bases_test[name] = kk[k]
        print(f"  cache 로드: {len(bases_train)} base predictors")
    else:
        bases_train, bases_test = {}, {}
        kalman_configs = [
            ("cv_so0p1mm", lambda X: kalman_cv(X, sigma_obs=0.1e-3)),
            ("cv_so0p3mm", lambda X: kalman_cv(X, sigma_obs=0.3e-3)),
            ("cv_so1mm",   lambda X: kalman_cv(X, sigma_obs=1.0e-3)),
            ("cv_so3mm",   lambda X: kalman_cv(X, sigma_obs=3.0e-3)),
            ("ca_so0p3mm", lambda X: kalman_ca(X, sigma_obs=0.3e-3)),
            ("ca_so1mm",   lambda X: kalman_ca(X, sigma_obs=1.0e-3)),
        ]
        for name, fn in kalman_configs:
            print(f"  computing {name}…")
            bases_train[name] = fn(X_train)
            bases_test[name]  = fn(X_test)

        save_kwargs = {}
        for name in bases_train:
            save_kwargs[f"k_{name}_train"] = bases_train[name]
            save_kwargs[f"k_{name}_test"]  = bases_test[name]
        np.savez(cache_kalmans, **save_kwargs)
        print(f"  cache 저장: {cache_kalmans}")

    # 각 base의 R-Hit
    print("\n--- base predictor R-Hit (full train) ---")
    for name in sorted(bases_train.keys()):
        r = rhit(bases_train[name], y_train)
        a_rh = axis_rhit(bases_train[name], y_train)
        print(f"  {name:<15}: R-Hit={r:.4f}  axis-hit (x,y,z)=({a_rh[0]:.3f}, {a_rh[1]:.3f}, {a_rh[2]:.3f})")

    # --- v23 ---
    state_path = CACHE_DIR / f"v23_state_{mode}.npz"
    assert state_path.exists(), f"v23 state 없음: {state_path}"
    st = np.load(state_path)
    oof_A, test_A = st["oof_A"], st["test_A"]
    fold_mask_A = st["fold_mask_A"]
    has_B = bool(st.get("has_B", np.array(False)))
    if has_B:
        oof_B, test_B = st["oof_B"], st["test_B"]
        fold_mask_B = st["fold_mask_B"]
        oof_v23_res = (oof_A + oof_B) / 2
        test_v23_res = (test_A + test_B) / 2
        eval_mask = fold_mask_A & fold_mask_B
    else:
        oof_v23_res = oof_A.copy(); test_v23_res = test_A.copy()
        eval_mask = fold_mask_A

    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    k_main_tr = bases_train["cv_so0p3mm"]  # v23이 학습한 base
    k_main_te = bases_test["cv_so0p3mm"]
    oof_v23 = k_main_tr + oof_v23_res * ALPHA
    test_v23 = k_main_te + test_v23_res * ALPHA

    # --- v16 ---
    st16 = np.load(V16_PATH)
    oof_v16  = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)

    # --- 모든 base 합치기 ---
    all_oof  = {**bases_train, "v23": oof_v23, "v16": oof_v16}
    all_test = {**bases_test,  "v23": test_v23, "v16": test_v16}

    print(f"\n--- 모든 8 base predictor on eval_mask (covered {eval_mask.sum()}) ---")
    base_perf = {}
    for name in all_oof:
        r = rhit(all_oof[name], y_train, eval_mask)
        a_rh = axis_rhit(all_oof[name], y_train, eval_mask)
        base_perf[name] = {"rhit": r, "axis_rhit": a_rh}
        print(f"  {name:<15}: R-Hit={r:.4f}  axis (x,y,z)=({a_rh[0]:.3f}, {a_rh[1]:.3f}, {a_rh[2]:.3f})")

    # ============================================================
    # Strategy 1: Axis-wise best base (단순 hardcode per axis)
    # ============================================================
    print("\n=== Strategy 1: per-axis best base (단순 선택) ===")
    best_per_axis = []
    for j, ax_name in enumerate(["x", "y", "z"]):
        scores = {name: axis_rhit(all_oof[name], y_train, eval_mask)[j] for name in all_oof}
        best = max(scores, key=scores.get)
        best_per_axis.append(best)
        print(f"  axis {ax_name}: best={best} ({scores[best]:.4f})")
    ens_axis_best = np.column_stack([all_oof[best_per_axis[j]][:, j] for j in range(3)])
    rh_axis_best = rhit(ens_axis_best, y_train, eval_mask)
    print(f"  → 단순 per-axis pick: {rh_axis_best:.4f}")

    # ============================================================
    # Strategy 2: per-axis linear blending (LGB regress per axis)
    # ============================================================
    print("\n=== Strategy 2: per-axis LGB meta (axis-wise stacking) ===")
    try:
        import lightgbm as lgb
        base_names = list(all_oof.keys())
        # Per-axis: feature = all bases' coord on that axis + |dist| from kalman main, target = y_axis
        ens_lgb = np.zeros((len(y_train), 3))

        kf = KFold(5, shuffle=True, random_state=0)
        for j, ax in enumerate(["x", "y", "z"]):
            feat = np.column_stack([all_oof[n][:, j] for n in base_names])
            # add diff from CV base
            feat = np.column_stack([feat, feat - feat[:, [1]]])  # diff from cv_so0p3mm
            target = y_train[:, j]

            feat_em, target_em = feat[eval_mask], target[eval_mask]
            oof_axis = np.zeros(eval_mask.sum())
            for tr, va in kf.split(feat_em):
                gbm = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=300,
                                         min_child_samples=20, verbose=-1, random_state=0)
                gbm.fit(feat_em[tr], target_em[tr])
                oof_axis[va] = gbm.predict(feat_em[va])
            ens_lgb[eval_mask, j] = oof_axis
            # uncovered: 단순 simple_avg base
            uncov = ~eval_mask
            if uncov.any():
                ens_lgb[uncov, j] = np.mean([all_oof[n][uncov, j] for n in base_names], axis=0)
            ax_r = float((np.abs(oof_axis - target_em) <= 0.01).mean())
            print(f"  axis {ax}: per-axis LGB axis-hit={ax_r:.4f}")
        rh_lgb_axis = rhit(ens_lgb, y_train, eval_mask)
        print(f"  → per-axis LGB joint R-Hit: {rh_lgb_axis:.4f}")
    except Exception as e:
        print(f"  skip: {e}")
        ens_lgb, rh_lgb_axis = None, None

    # ============================================================
    # Strategy 3: Hill climbing greedy ensemble selection
    # ============================================================
    print("\n=== Strategy 3: greedy hill-climbing ensemble ===")
    # 시작: simple_avg of all
    ens_state = np.mean([all_oof[n] for n in all_oof], axis=0)
    rh_state = rhit(ens_state, y_train, eval_mask)
    print(f"  init (all avg): {rh_state:.4f}")

    weights = {n: 1.0 for n in all_oof}
    for it in range(20):
        improved = False
        best_delta = 0; best_action = None
        for n in all_oof:
            for delta in [-0.3, -0.1, 0.1, 0.3]:
                new_w = {k: v + (delta if k == n else 0) for k, v in weights.items()}
                if new_w[n] < 0: continue
                tot = sum(new_w.values())
                if tot <= 0: continue
                ens_new = np.zeros_like(ens_state)
                for k, w in new_w.items():
                    ens_new += w * all_oof[k]
                ens_new /= tot
                rh_new = rhit(ens_new, y_train, eval_mask)
                if rh_new - rh_state > best_delta + 1e-5:
                    best_delta = rh_new - rh_state
                    best_action = (n, delta, new_w, rh_new)
                    improved = True
        if not improved: break
        n, d, new_w, rh_new = best_action
        weights = new_w
        rh_state = rh_new
        print(f"  iter {it+1}: +{n} by {d:+.1f} → R-Hit={rh_state:.4f}")
    # 최종 ensemble
    tot = sum(weights.values())
    ens_greedy = np.zeros_like(oof_v23)
    for k, w in weights.items():
        ens_greedy += (w / tot) * all_oof[k]
    print(f"  → greedy final R-Hit: {rh_state:.4f}, weights:")
    for k, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {k}: {w/tot:.3f}")

    # ============================================================
    # Adversarial validation (진단)
    # ============================================================
    print("\n=== Adversarial validation (train vs test) ===")
    # 단순 feature: last position + velocity + acceleration norm + Kalman delta
    def adv_feats(X):
        last = X[:, -1, :]
        v = np.diff(X, axis=1)[:, -1, :] / DT
        a = np.diff(X, axis=1)[:, -1, :] / DT - np.diff(X, axis=1)[:, -2, :] / DT
        return np.column_stack([
            last, v, np.linalg.norm(v, axis=-1, keepdims=True),
            a, np.linalg.norm(a, axis=-1, keepdims=True),
        ])

    feat_tr = adv_feats(X_train); feat_te = adv_feats(X_test)
    adv_result = adversarial_validation(feat_tr, feat_te)

    if adv_result is not None:
        np.savez(CACHE_DIR / "v25_adv_val.npz",
                  oof_proba=adv_result["oof_proba"], auc=adv_result["auc_mean"])
        print(f"  cache: cache/v25_adv_val.npz")

    # ============================================================
    # 결과 정리 + 최종 선택
    # ============================================================
    print("\n" + "=" * 60)
    print("=== 모든 strategy OOF 결과 ===")
    print("=" * 60)
    results = {
        "v23 alone":            rhit(oof_v23, y_train, eval_mask),
        "v16 alone":            rhit(oof_v16, y_train, eval_mask),
        "kalman_cv_so0p3mm":    rhit(bases_train["cv_so0p3mm"], y_train, eval_mask),
        "strategy_1_axis_best": rh_axis_best,
        "strategy_2_lgb_axis":  rh_lgb_axis if rh_lgb_axis else None,
        "strategy_3_greedy":    rh_state,
    }
    for k, v in results.items():
        if v is not None:
            print(f"  {k:<25}: {v:.4f}")

    # Best strategy 선택
    valid = {k: v for k, v in results.items() if v is not None and k not in ("v23 alone", "v16 alone", "kalman_cv_so0p3mm")}
    best_name = max(valid, key=valid.get)
    best_rh = valid[best_name]
    print(f"\n★★ Best ensemble: {best_name}  {best_rh:.4f}")

    # --- Test prediction ---
    if best_name == "strategy_1_axis_best":
        test_final = np.column_stack([all_test[best_per_axis[j]][:, j] for j in range(3)])
    elif best_name == "strategy_3_greedy":
        test_final = np.zeros_like(test_v23)
        for k, w in weights.items():
            test_final += (w / tot) * all_test[k]
    elif best_name == "strategy_2_lgb_axis" and rh_lgb_axis is not None:
        # 재학습 (전체 eval_mask로) + test 적용
        test_final = np.zeros((10000, 3))
        for j, ax in enumerate(["x", "y", "z"]):
            feat_tr_j = np.column_stack([all_oof[n][:, j] for n in base_names])
            feat_tr_j = np.column_stack([feat_tr_j, feat_tr_j - feat_tr_j[:, [1]]])
            feat_te_j = np.column_stack([all_test[n][:, j] for n in base_names])
            feat_te_j = np.column_stack([feat_te_j, feat_te_j - feat_te_j[:, [1]]])
            gbm_full = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=300,
                                            min_child_samples=20, verbose=-1, random_state=0)
            gbm_full.fit(feat_tr_j[eval_mask], y_train[eval_mask, j])
            test_final[:, j] = gbm_full.predict(feat_te_j)
    else:
        test_final = (test_v23 + test_v16) / 2

    # 정상성 + 제출
    assert test_final.shape == (10000, 3) and np.isfinite(test_final).all()
    out_csv = DATA_DIR / f"submission_v25_cpu_{mode}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_final[:,0], "y": test_final[:,1], "z": test_final[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n[submission] {out_csv}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": f"v25_cpu_{mode}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Multi-Kalman (6) + v23 + v16 = 8 base, axis-aware ensemble",
        "v23_mode": mode,
        "base_perf": {k: v["rhit"] for k, v in base_perf.items()},
        "axis_per_base": {k: v["axis_rhit"] for k, v in base_perf.items()},
        "strategy_results": {k: float(v) for k, v in results.items() if v is not None},
        "best_strategy": best_name,
        "best_rhit": float(best_rh),
        "best_per_axis": best_per_axis if best_name == "strategy_1_axis_best" else None,
        "greedy_weights": {k: float(w/tot) for k, w in weights.items()} if best_name == "strategy_3_greedy" else None,
        "adv_auc": adv_result["auc_mean"] if adv_result else None,
        "covered_rows": int(eval_mask.sum()),
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

    print("\n" + "=" * 60)
    print(f"v25 multi-Kalman ensemble ({mode}) 완료")
    print("=" * 60)
    print(f"  v23 alone : {rhit(oof_v23, y_train, eval_mask):.4f}")
    print(f"  v16 alone : {rhit(oof_v16, y_train, eval_mask):.4f}")
    print(f"  best ens  : {best_rh:.4f}  ({best_name})")
    print(f"  adv AUC   : {adv_result['auc_mean']:.4f}" if adv_result else "  adv AUC   : skipped")
    print(f"  CSV       : {out_csv}")


if __name__ == "__main__":
    main()

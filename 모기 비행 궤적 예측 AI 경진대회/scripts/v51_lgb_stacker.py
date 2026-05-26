"""v51_lgb_stacker.py — LightGBM stacker on OOF residuals.

NN stacker (v46/v48/v50) plateau ~0.6734. Non-linear LGB가 trajectory feats + OOF disagreement에서
다른 signal 찾을 수 있는지 검증.

설계:
- target: y_axis - v35_axis (잔차)
- features: 8 valid models × 3 (24) + global feats (last_pos/v/a/v_mean/v_std/speed/a_norm/kalman) = 24+19=43
- LGB MultiOutput (3 axis 독립 학습, joint R-Hit 측정)
- 5-fold OOF (random_state=0 동일)

v25 per-axis LGB 실패 (joint 0.40) — 그건 target = y_axis 직접 학습이어서 joint corr 손실.
여기서는 target = residual on v35 → 작은 보정만, joint corr 유지.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"

BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rhit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def build_global_features(X, kalman):
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    return np.concatenate([last_pos, v, a, v_mean, v_std, speed, a_norm, kalman], axis=-1).astype(np.float32)


def main():
    np.random.seed(0)

    print("=" * 60)
    print("v51 LightGBM stacker — target = y - v35, MultiOutput per axis")
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

    # Same 8 valid models (skip v32)
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    v30A_oof = kalman_train + st30["oof_A"] * ALPHA
    v30B_oof = kalman_train + st30["oof_B"] * ALPHA
    v30A_te  = kalman_test  + st30["test_A"] * ALPHA
    v30B_te  = kalman_test  + st30["test_B"] * ALPHA

    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v35_oof, v35_te = st35["oof_v35"].astype(np.float64), st35["test_v35"].astype(np.float64)

    st41 = np.load(CACHE_DIR / "v41_state.npz")
    v41A_oof = kalman_train + st41["oof_A"] * ALPHA
    v41B_oof = kalman_train + st41["oof_B"] * ALPHA
    v41A_te  = kalman_test  + st41["test_A"] * ALPHA
    v41B_te  = kalman_test  + st41["test_B"] * ALPHA

    st44 = np.load(CACHE_DIR / "v44_state.npz")
    v44_oof, v44_te = st44["oof_v44"].astype(np.float64), st44["test_v44"].astype(np.float64)

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    gate_te = df_best[["x","y","z"]].values.astype(np.float64)

    st39 = np.load(CACHE_DIR / "v39_state.npz")
    v39_oof, v39_te = st39["oof_v39"].astype(np.float64), st39["test_v39"].astype(np.float64)

    models_train = [v30A_oof, v30B_oof, v35_oof, v41A_oof, v41B_oof, v44_oof, gate_oof, v39_oof]
    models_test  = [v30A_te,  v30B_te,  v35_te,  v41A_te,  v41B_te,  v44_te,  gate_te,  v39_te]
    names = ["v30A", "v30B", "v35", "v41A", "v41B", "v44", "gate", "v39"]

    # Base = v35 (strongest single)
    base_train = v35_oof
    base_test = v35_te

    g_tr = build_global_features(X_train, kalman_train)
    g_te = build_global_features(X_test, kalman_test)

    # Features: 8 model preds (8*3=24) + 8 model residuals vs kalman (24) + global feats (19) = 67
    X_tr_feat = np.concatenate(
        [g_tr] + [m for m in models_train] + [(m - kalman_train) for m in models_train],
        axis=-1,
    ).astype(np.float32)
    X_te_feat = np.concatenate(
        [g_te] + [m for m in models_test] + [(m - kalman_test) for m in models_test],
        axis=-1,
    ).astype(np.float32)

    print(f"\n  feat dim: {X_tr_feat.shape[1]}, N={len(y_train)}")
    print(f"  base = v35 OOF: {rhit(base_train, y_train):.4f}")

    # Per-axis residual target
    target = (y_train - base_train).astype(np.float32)
    print(f"  residual stats: mean abs={np.abs(target).mean()*1000:.2f}mm, "
          f"p95={np.percentile(np.abs(target).max(axis=-1), 95)*1000:.2f}mm")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof_resid = np.zeros((len(y_train), 3), dtype=np.float32)
    test_resid_folds = []

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 0,
    }

    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(X_tr_feat)):
        print(f"\n--- fold {fi+1}/5 ---")
        fold_test_resid = np.zeros((len(X_te_feat), 3), dtype=np.float32)
        for c in range(3):
            dtr = lgb.Dataset(X_tr_feat[tr], target[tr, c])
            dva = lgb.Dataset(X_tr_feat[va], target[va, c])
            mdl = lgb.train(params, dtr,
                            num_boost_round=2000,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            oof_resid[va, c] = mdl.predict(X_tr_feat[va], num_iteration=mdl.best_iteration)
            fold_test_resid[:, c] = mdl.predict(X_te_feat, num_iteration=mdl.best_iteration)
        test_resid_folds.append(fold_test_resid)
        pred_fold = base_train[va] + oof_resid[va]
        rh = rhit(pred_fold, y_train[va])
        print(f"  fold {fi+1} rhit (base+resid): {rh:.4f}  ({(time.time()-t0)/60:.1f}m)")

    test_resid = np.mean(test_resid_folds, axis=0)
    oof_v51 = base_train + oof_resid
    test_v51 = base_test + test_resid
    rh_v51 = rhit(oof_v51, y_train)
    print(f"\n=== v51 LGB stacker 결과 ===")
    print(f"  v35 base OOF : {rhit(base_train, y_train):.4f}")
    print(f"  v51 OOF      : {rh_v51:.4f}  (Δ {rh_v51 - rhit(base_train, y_train):+.4f})")

    # Δ stats
    r_norm = np.linalg.norm(oof_resid, axis=-1)
    print(f"  residual magnitude: mean={r_norm.mean()*1000:.2f}mm, p95={np.percentile(r_norm, 95)*1000:.2f}mm")

    # Hybrid with v35, v48
    print(f"\n=== Hybrid sweep ===")
    best_a35, best_r35 = 1.0, rh_v51
    for alpha in np.linspace(0.0, 1.0, 21):
        ens = alpha * oof_v51 + (1 - alpha) * v35_oof
        r = rhit(ens, y_train)
        if r > best_r35: best_r35, best_a35 = r, alpha
    print(f"  v51+v35: α={best_a35:.2f} OOF={best_r35:.4f}")

    try:
        st48 = np.load(CACHE_DIR / "v48_state.npz")
        oof_v48 = st48["oof_v48"]; test_v48 = st48["test_v48"]
        st46 = np.load(CACHE_DIR / "v46_state.npz")
        oof_v46 = st46["oof_v46"]; test_v46 = st46["test_v46"]

        best_3 = (1.0, 0.0, 0.0); best_3r = rh_v51
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1-a, 11):
                c = 1 - a - b
                ens = a*oof_v51 + b*oof_v48 + c*v35_oof
                r = rhit(ens, y_train)
                if r > best_3r:
                    best_3r, best_3 = r, (a, b, c)
        print(f"  3-way v51/v48/v35: {best_3[0]:.2f}/{best_3[1]:.2f}/{best_3[2]:.2f} → {best_3r:.4f}")

        best_4 = (1.0, 0.0, 0.0, 0.0); best_4r = rh_v51
        for a in np.linspace(0, 1, 6):
            for b in np.linspace(0, 1-a, 6):
                for c in np.linspace(0, 1-a-b, 6):
                    d = 1 - a - b - c
                    if d < 0: continue
                    ens = a*oof_v51 + b*oof_v48 + c*v35_oof + d*oof_v46
                    r = rhit(ens, y_train)
                    if r > best_4r:
                        best_4r, best_4 = r, (a, b, c, d)
        print(f"  4-way v51/v48/v35/v46: {best_4} OOF={best_4r:.4f}")
    except Exception as e:
        print(f"  hybrid cache err: {e}")
        oof_v46 = test_v46 = oof_v48 = test_v48 = None
        best_3, best_3r = (1.0, 0.0, 0.0), rh_v51
        best_4, best_4r = (1.0, 0.0, 0.0, 0.0), rh_v51

    np.savez(CACHE_DIR / "v51_state.npz",
             oof_v51=oof_v51, test_v51=test_v51, rh_v51=rh_v51,
             oof_resid=oof_resid, test_resid=test_resid)

    out_csv = DATA_DIR / "submission_v51_cpu.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v51[:,0], "y": test_v51[:,1], "z": test_v51[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n  [submission] {out_csv}")

    if best_3r > rh_v51 and oof_v48 is not None:
        a, b, c = best_3
        hyb = a*test_v51 + b*test_v48 + c*v35_te
        hyb_csv = DATA_DIR / f"submission_v51_3way_v51x{a:.2f}_v48x{b:.2f}_v35x{c:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb[:,0], "y": hyb[:,1], "z": hyb[:,2]}
                     ).to_csv(hyb_csv, index=False)
        print(f"  [3way] {hyb_csv.name}")

    if best_4r > rh_v51 and oof_v46 is not None:
        a, b, c, d = best_4
        hyb = a*test_v51 + b*test_v48 + c*v35_te + d*test_v46
        hyb_csv = DATA_DIR / f"submission_v51_4way_v51x{a:.2f}_v48x{b:.2f}_v35x{c:.2f}_v46x{d:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb[:,0], "y": hyb[:,1], "z": hyb[:,2]}
                     ).to_csv(hyb_csv, index=False)
        print(f"  [4way] {hyb_csv.name}")

    entry = {
        "version": "v51_lgb_stacker",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "LightGBM stacker per-axis residual on v35 base, joint R-Hit eval",
        "n_models": len(names),
        "v51_oof": float(rh_v51),
        "delta_vs_v35": float(rh_v51 - rhit(base_train, y_train)),
        "hybrid_v35_alpha": float(best_a35),
        "hybrid_v35_oof": float(best_r35),
        "3way_v51_v48_v35": list(best_3),
        "3way_oof": float(best_3r),
        "4way_v51_v48_v35_v46": list(best_4),
        "4way_oof": float(best_4r),
        "submission": str(out_csv),
    }
    log_path = PROJECT_DIR / "run_log.json"
    log = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    log.append(entry)
    json.dump(log, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  [run_log] appended v51_lgb_stacker")


if __name__ == "__main__":
    main()

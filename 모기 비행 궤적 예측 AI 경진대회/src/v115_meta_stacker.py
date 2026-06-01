"""v115_meta_stacker.py - LightGBM meta-stacker on base ensemble + trajectory meta features.

approach:
  - Base = v112_v107_diverse ensemble OOF (LB 0.6888 검증 best)
  - Meta features: 11개 trajectory meta features (accel/jerk/turn/speed/dir/...)
  - 추가 features: 각 paradigm pool model의 OOF residual norm (model별 deviation)
  - Target: residual = y - base_ensemble_pred (clip ±5cm for stability)
  - Model: LightGBM regressor per axis (or multi-output)
  - Nested 5-fold OOF: meta learner도 fold val 예측만 사용 → no leakage
  - Final: corrected = base + meta_residual_pred (clip ±3cm)

usage:
  python scripts/v115_meta_stacker.py --tag v115_basic
  python scripts/v115_meta_stacker.py --tag v115_residual_cap2 --residual-cap 0.02
"""
from __future__ import annotations

import argparse, datetime as _dt, glob, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v110_de_ensemble import load_pool, load_y_and_sub

PROJECT = SCRIPT_DIR.parent
CACHE = PROJECT / "data/cache"
DATA = PROJECT / "data"


def build_meta_features(X):
    """trajectory 메타특징 11개 (TASK 1 와 동일)."""
    DT = 0.040
    v = np.diff(X, axis=1) / DT
    a = np.diff(v, axis=1) / DT
    jerk = np.diff(a, axis=1) / DT

    speed_last3 = np.linalg.norm(v[:, -3:], axis=-1).mean(axis=-1)
    speed_mean = np.linalg.norm(v, axis=-1).mean(axis=-1)
    speed_max = np.linalg.norm(v, axis=-1).max(axis=-1)
    accel_mean = np.linalg.norm(a, axis=-1).mean(axis=-1)
    accel_max = np.linalg.norm(a, axis=-1).max(axis=-1)
    jerk_mean = np.linalg.norm(jerk, axis=-1).mean(axis=-1)

    un = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    cosA = (un[:, :-1] * un[:, 1:]).sum(axis=-1).clip(-1, 1)
    turn = np.arccos(cosA)
    turn_mean = turn.mean(axis=-1); turn_max = turn.max(axis=-1)

    cross_va = np.cross(v[:, 1:], a, axis=-1)
    v_for_k = v[:, 1:]
    v_mag = np.linalg.norm(v_for_k, axis=-1)
    kappa = np.linalg.norm(cross_va, axis=-1) / (v_mag**3 + 1e-12)
    kappa_mean = kappa.mean(axis=-1); kappa_max = kappa.max(axis=-1)

    dir_std = un.std(axis=1).mean(axis=-1)

    return np.stack([
        speed_last3, speed_mean, speed_max,
        accel_mean, accel_max, jerk_mean,
        turn_mean, turn_max,
        kappa_mean, kappa_max,
        dir_std,
    ], axis=-1).astype(np.float32)


def build_pool_features(pool, base_pred):
    """각 pool model 의 deviation from ensemble base. (N, n_models)."""
    devs = []
    for nm, oof, _, _ in pool:
        # per-sample deviation magnitude from base
        d = np.linalg.norm(oof - base_pred, axis=-1)
        devs.append(d)
    return np.stack(devs, axis=-1).astype(np.float32)


def build_pool_test_features(pool, base_test):
    devs = []
    for nm, _, te, _ in pool:
        d = np.linalg.norm(te - base_test, axis=-1)
        devs.append(d)
    return np.stack(devs, axis=-1).astype(np.float32)


def hit(pred, y):
    return float((np.linalg.norm(pred - y, axis=-1) <= 0.01).mean())


def train_meta_stacker(features_tr, features_te, residual_tr, y_tr, base_tr,
                       base_te, n_folds=5, n_seeds=3,
                       residual_cap=0.03, lr=0.05, n_est=200, num_leaves=31,
                       min_data=50, feature_frac=0.8, bagging_frac=0.8,
                       verbose=False):
    """LightGBM meta-stacker per axis, nested OOF."""
    N = len(features_tr)
    oof_resid = np.zeros((N, 3))
    test_resid = np.zeros((len(features_te), 3))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))

    for axis in range(3):
        target_ax = residual_tr[:, axis]
        for s in range(n_seeds):
            test_fold_pred = np.zeros(len(features_te))
            for fi, (tr, va) in enumerate(fold_iter):
                params = dict(
                    objective='regression_l1',  # L1 = MAE, robust to outliers
                    metric='mae',
                    learning_rate=lr,
                    num_leaves=num_leaves,
                    min_data_in_leaf=min_data,
                    feature_fraction=feature_frac,
                    bagging_fraction=bagging_frac,
                    bagging_freq=5,
                    seed=s,
                    verbose=-1,
                )
                tr_ds = lgb.Dataset(features_tr[tr], target_ax[tr])
                va_ds = lgb.Dataset(features_tr[va], target_ax[va], reference=tr_ds)
                model = lgb.train(
                    params, tr_ds, num_boost_round=n_est,
                    valid_sets=[va_ds],
                    callbacks=[lgb.early_stopping(20, verbose=False)],
                )
                p_va = model.predict(features_tr[va], num_iteration=model.best_iteration)
                p_te = model.predict(features_te, num_iteration=model.best_iteration)
                oof_resid[va, axis] += p_va / n_seeds
                test_fold_pred += p_te / n_folds
            test_resid[:, axis] += test_fold_pred / n_seeds
            if verbose:
                rh_partial = hit(base_tr + np.clip(oof_resid, -residual_cap, residual_cap) * (s+1)/n_seeds, y_tr)
                print(f'  axis {axis} seed {s}: partial R-Hit {rh_partial:.4f}', flush=True)

    # cap residual
    oof_resid_c = np.clip(oof_resid, -residual_cap, residual_cap)
    test_resid_c = np.clip(test_resid, -residual_cap, residual_cap)
    return oof_resid_c, test_resid_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v115_basic")
    parser.add_argument("--base", default="v112_v107_diverse",
                        help="base ensemble cache prefix (looks for {base}_weights.npz)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-est", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data", type=int, default=50)
    parser.add_argument("--feature-frac", type=float, default=0.8)
    parser.add_argument("--bagging-frac", type=float, default=0.8)
    parser.add_argument("--residual-cap", type=float, default=0.03)
    parser.add_argument("--include-mdn", type=int, default=1)
    parser.add_argument("--include-pool-devs", type=int, default=1,
                        help="include per-model deviation from base as features")
    args = parser.parse_args()

    np.random.seed(0)
    print("=" * 60)
    print(f"v115 meta-stacker: base={args.base}, residual_cap={args.residual_cap}cm")
    print("=" * 60)

    # base ensemble
    base_npz = np.load(CACHE / f"{args.base}_weights.npz", allow_pickle=True)
    base_oof = base_npz["oof_pred"].astype(np.float64)
    base_test = base_npz["test_pred"].astype(np.float64)

    y, sub, ids = load_y_and_sub()
    nc = np.load(CACHE / "xtrain_xtest.npz")
    X_tr, X_te = nc["X_train"], nc["X_test"]

    rh_base = hit(base_oof, y)
    print(f"\nbase OOF R-Hit: {rh_base:.4f}")

    # features
    meta_tr = build_meta_features(X_tr)
    meta_te = build_meta_features(X_te)
    print(f"meta features: {meta_tr.shape}")

    feat_list_tr = [meta_tr, base_oof.astype(np.float32)]
    feat_list_te = [meta_te, base_test.astype(np.float32)]

    if args.include_pool_devs:
        pool, _ = load_pool(include_mdn=bool(args.include_mdn))
        print(f"pool models for dev features: {len(pool)}")
        devs_tr = build_pool_features(pool, base_oof)
        devs_te = build_pool_test_features(pool, base_test)
        feat_list_tr.append(devs_tr)
        feat_list_te.append(devs_te)

    features_tr = np.concatenate(feat_list_tr, axis=-1).astype(np.float32)
    features_te = np.concatenate(feat_list_te, axis=-1).astype(np.float32)
    print(f"total features: {features_tr.shape[1]}")

    residual_tr = (y - base_oof).astype(np.float32)
    print(f"residual stats (cm): mean abs ({100*np.abs(residual_tr).mean():.3f}), "
          f"std ({100*residual_tr.std():.3f}), max ({100*np.abs(residual_tr).max():.3f})")

    # train
    oof_resid, test_resid = train_meta_stacker(
        features_tr, features_te, residual_tr, y, base_oof, base_test,
        n_folds=args.n_folds, n_seeds=args.n_seeds,
        residual_cap=args.residual_cap, lr=args.lr, n_est=args.n_est,
        num_leaves=args.num_leaves, min_data=args.min_data,
        feature_frac=args.feature_frac, bagging_frac=args.bagging_frac,
    )

    # apply correction (with cap)
    corrected = base_oof + oof_resid
    test_corrected = base_test + test_resid
    rh_corrected = hit(corrected, y)
    print(f"\n=== {args.tag} ===")
    print(f"  base R-Hit:        {rh_base:.4f}")
    print(f"  corrected R-Hit:   {rh_corrected:.4f}  (Δ = {rh_corrected - rh_base:+.4f})")
    print(f"  residual cap: ±{args.residual_cap*100:.1f}cm")
    print(f"  residual applied stats (cm): "
          f"abs mean {100*np.abs(oof_resid).mean():.3f}, "
          f"max {100*np.abs(oof_resid).max():.3f}")

    # subset eval (turn+speed top 20%)
    speed_max = np.linalg.norm(np.diff(X_tr, axis=1)/0.040, axis=-1).max(axis=-1)
    v_arr = np.diff(X_tr, axis=1)/0.040
    un = v_arr / (np.linalg.norm(v_arr, axis=-1, keepdims=True) + 1e-12)
    cosA = (un[:, :-1] * un[:, 1:]).sum(axis=-1).clip(-1, 1)
    turn_max = np.arccos(cosA).max(axis=-1)
    mask = (turn_max >= np.quantile(turn_max, 0.8)) & (speed_max >= np.quantile(speed_max, 0.8))
    rh_sub_base = float((np.linalg.norm(base_oof[mask] - y[mask], axis=-1) <= 0.01).mean())
    rh_sub_corr = float((np.linalg.norm(corrected[mask] - y[mask], axis=-1) <= 0.01).mean())
    print(f"  hard subset (turn+speed top 20%, n={mask.sum()}): "
          f"base {rh_sub_base:.4f} → corrected {rh_sub_corr:.4f} ({rh_sub_corr-rh_sub_base:+.4f})")

    # save
    out_csv = DATA / f"submission_{args.tag}_oof{rh_corrected:.4f}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_corrected[:,0],
                  "y": test_corrected[:,1], "z": test_corrected[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv.name}")

    np.savez(CACHE / f"{args.tag}_state.npz",
             oof=corrected, test_pred=test_corrected, rh=rh_corrected,
             oof_resid=oof_resid, test_resid=test_resid,
             base=args.base, residual_cap=args.residual_cap)
    print(f"  [state] {args.tag}_state.npz")

    entry = {"version": args.tag, "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": f"LightGBM meta-stacker on {args.base} (residual prediction, cap=±{args.residual_cap*100:.1f}cm)",
             "n_features": int(features_tr.shape[1]),
             "rh_base": float(rh_base), "rh_corrected": float(rh_corrected),
             "delta": float(rh_corrected - rh_base),
             "rh_subset_base": float(rh_sub_base), "rh_subset_corr": float(rh_sub_corr)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

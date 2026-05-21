"""CatBoost 회귀 학습 (블렌드 다양성용).

train_solution.py와 동일한 5-fold scenario 분할, log1p 타겟 사용.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

from train_solution import (
    ID_COL, GROUP_COL, TARGET, CAT_COLS,
    LAYOUT_STATIC_COLS, LAYOUT_DERIVED_COLS,
    LEAKY_EXTRA_COLS, LEAKY_SEQ_SUFFIXES,
    add_features, make_folds, seed_everything,
)
from feature_cache import load_cached

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models/cat_log_seed42"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--drop-layout-features", action="store_true")
    parser.add_argument("--drop-leaky-features", action="store_true")
    parser.add_argument("--use-cache", action="store_true", default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_path(args.data_dir)

    print("loading data")
    train_raw = pd.read_csv(data_dir / "train.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    if args.use_cache:
        print("loading cached features")
        train, test = load_cached(data_dir)
    else:
        test_raw = pd.read_csv(data_dir / "test.csv")
        layout = pd.read_csv(data_dir / "layout_info.csv")
        print("building features")
        train = add_features(train_raw, layout)
        test = add_features(test_raw, layout)

    drop_cols = [ID_COL, GROUP_COL, TARGET, "layout_id"]
    if args.drop_layout_features:
        drop_cols.extend(LAYOUT_STATIC_COLS)
        drop_cols.extend(LAYOUT_DERIVED_COLS)
    if args.drop_leaky_features:
        drop_cols.extend(LEAKY_EXTRA_COLS)
        drop_cols.extend([c for c in train.columns if c.endswith(LEAKY_SEQ_SUFFIXES)])
    feature_cols = [c for c in train.columns if c not in drop_cols]
    cat_cols = [c for c in CAT_COLS if c in feature_cols and c != "layout_id"]

    # CatBoost는 NaN 처리하지만 categorical은 string으로 받는 게 안전
    for c in cat_cols:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    y = train[TARGET].to_numpy()
    if args.log_target:
        y_train = np.log1p(np.clip(y, 0, None))
    else:
        y_train = y

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    cat_feature_idx = [feature_cols.index(c) for c in cat_cols]

    params = dict(
        iterations=8000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        loss_function="MAE",
        eval_metric="MAE",
        bagging_temperature=1.0,
        random_seed=args.seed,
        od_type="Iter",
        od_wait=200,
        verbose=200,
        thread_count=-1,
    )

    fold_iters = []
    for fold, (tr_idx, val_idx) in enumerate(make_folds(train, args.n_splits, args.seed), start=1):
        print(f"\n=== fold {fold} ===")
        X_tr = train.iloc[tr_idx][feature_cols]
        X_val = train.iloc[val_idx][feature_cols]
        y_tr = y_train[tr_idx]
        y_val_log = y_train[val_idx]
        y_val_raw = y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_idx)
        val_pool = Pool(X_val, y_val_log, cat_features=cat_feature_idx)
        test_pool = Pool(test[feature_cols], cat_features=cat_feature_idx)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        fold_iters.append(int(model.tree_count_))

        val_pred = model.predict(val_pool)
        tst_pred = model.predict(test_pool)
        if args.log_target:
            val_pred = np.expm1(val_pred)
            tst_pred = np.expm1(tst_pred)
        oof[val_idx] = val_pred
        test_pred += tst_pred / args.n_splits

        fold_mae = mean_absolute_error(y_val_raw, val_pred)
        print(f"fold {fold} MAE (raw): {fold_mae:.6f}, iters: {model.tree_count_}")

    oof = np.clip(oof, 0, None)
    test_pred = np.clip(test_pred, 0, None)
    overall_mae = mean_absolute_error(y, oof)
    print(f"\nOOF MAE clipped: {overall_mae:.6f}")

    sample[TARGET] = test_pred
    sample.to_csv(out_dir / "submission.csv", index=False)

    oof_df = train_raw[[ID_COL, GROUP_COL, "layout_id", TARGET]].copy()
    oof_df["pred"] = oof
    oof_df["abs_error"] = (oof_df[TARGET] - oof_df["pred"]).abs()
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    metadata = {
        "seed": args.seed,
        "n_splits": args.n_splits,
        "log_target": args.log_target,
        "oof_mae": float(overall_mae),
        "feature_count": len(feature_cols),
        "categorical_cols": cat_cols,
        "fold_iterations": fold_iters,
        "python": os.sys.version,
        "catboost": "1.2.8",
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

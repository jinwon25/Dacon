"""Adversarial validation: train과 test 분포 차이 진단.

- train + test를 결합해 이진 분류 (train=0, test=1)
- AUC가 0.5에 가까우면 분포가 비슷, 0.5보다 크게 멀면 분포 차이
- per-feature gain으로 분포를 가르는 주요 피처 식별
- per-train-row P(test=1)을 저장 → 후속 sample_weight 또는 valid 선택용
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_solution import add_features, ID_COL, GROUP_COL, TARGET, CAT_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "models/adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading data")
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    layout = pd.read_csv(data_dir / "layout_info.csv")

    print("building features")
    train = add_features(train_raw, layout)
    test = add_features(test_raw, layout)

    drop_cols = [ID_COL, GROUP_COL, TARGET, "layout_id"]
    feature_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]
    cat_cols = [c for c in CAT_COLS if c in feature_cols and c != "layout_id"]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    X_train["__is_test"] = 0
    X_test["__is_test"] = 1
    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    y_adv = combined["__is_test"].to_numpy()
    X_adv = combined.drop(columns=["__is_test"])

    print(f"combined shape: {X_adv.shape}, train={int((y_adv==0).sum())}, test={int((y_adv==1).sum())}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_adv))
    importances = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_adv, y_adv), start=1):
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=100,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_adv.iloc[tr_idx], y_adv[tr_idx],
            eval_set=[(X_adv.iloc[val_idx], y_adv[val_idx])],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(period=100)],
        )
        oof[val_idx] = model.predict_proba(X_adv.iloc[val_idx])[:, 1]
        importances.append(model.booster_.feature_importance(importance_type="gain"))
        print(f"fold {fold} AUC: {roc_auc_score(y_adv[val_idx], oof[val_idx]):.6f}")

    overall_auc = roc_auc_score(y_adv, oof)
    print(f"\noverall adversarial AUC: {overall_auc:.6f}")
    print("0.5 = 분포 동일, 1.0 = 완벽 분리. 0.6 이상이면 의미 있는 분포 차이.")

    n_train = len(train)
    train_p_test = oof[:n_train]
    test_p_test = oof[n_train:]

    print(f"\ntrain의 P(test) 분포: mean={train_p_test.mean():.4f}, p50={np.median(train_p_test):.4f}, p90={np.quantile(train_p_test, 0.9):.4f}")
    print(f"test의 P(test) 분포: mean={test_p_test.mean():.4f}, p50={np.median(test_p_test):.4f}")

    avg_imp = np.mean(np.stack(importances), axis=0)
    feat_imp = pd.DataFrame({"feature": X_adv.columns, "gain": avg_imp}).sort_values("gain", ascending=False)
    print("\n분포를 가르는 상위 20 피처:")
    print(feat_imp.head(20).to_string(index=False))

    feat_imp.to_csv(out_dir / "adversarial_feature_importance.csv", index=False)

    weights = pd.DataFrame({
        "ID": train_raw[ID_COL].to_numpy(),
        "p_test": train_p_test,
    })
    raw_w = train_p_test / np.clip(1 - train_p_test, 1e-3, 1)
    weights["sample_weight"] = np.clip(raw_w, 0.0, 20.0)
    weights.to_csv(out_dir / "train_adversarial_weights.csv", index=False)
    print(f"\nsaved per-train-row weights to {out_dir / 'train_adversarial_weights.csv'}")

    metadata = {
        "overall_auc": float(overall_auc),
        "n_features": len(feature_cols),
        "top_features": feat_imp.head(20).to_dict(orient="records"),
        "train_p_test_stats": {
            "mean": float(train_p_test.mean()),
            "median": float(np.median(train_p_test)),
            "p90": float(np.quantile(train_p_test, 0.9)),
        },
    }
    with open(out_dir / "adversarial_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

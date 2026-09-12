"""v5 — v4 풍부 피처 + 템포 보강(t_gap2/3, span5, 같은팀 2-터치). 목표: v4(14.02) 깨기.
OOF/테스트 예측을 data/cache/gbdt_oof.npz 에 저장(앙상블용, v4 덮어씀)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, "src")
import lightgbm as lgb  # noqa: E402
from features_v4 import CATEGORICAL_V4, FEATURE_COLS_V4, build_features_v4  # noqa: E402


def oof_encode(train, test, fold, cat, targets, m):
    cols = []
    for t in targets:
        gmean = train[t].mean()
        col = np.empty(len(train))
        for f in np.unique(fold):
            other = fold != f
            g = train.loc[other].groupby(cat)[t].agg(["mean", "count"])
            sm = (g["mean"] * g["count"] + gmean * m) / (g["count"] + m)
            col[fold == f] = train.loc[fold == f, cat].map(sm.to_dict()).fillna(gmean).to_numpy()
        train[f"te_{cat}_{t}"] = col
        g = train.groupby(cat)[t].agg(["mean", "count"])
        sm = (g["mean"] * g["count"] + gmean * m) / (g["count"] + m)
        test[f"te_{cat}_{t}"] = test[cat].map(sm.to_dict()).fillna(gmean).to_numpy()
        cols.append(f"te_{cat}_{t}")
    return cols


train = build_features_v4("'data/train.csv'")
test = build_features_v4("read_csv('data/test/*/*.csv', union_by_name = true)")
train = train[train["tgt_x"].notna()].reset_index(drop=True)
test = test.reset_index(drop=True)
fold = (train["game_id"].astype("int64") % 5).to_numpy()

te = []
te += oof_encode(train, test, fold, "player_id", ["tgt_x", "tgt_y"], m=20)
te += oof_encode(train, test, fold, "team_id", ["tgt_x", "tgt_y"], m=50)

FEATURES = FEATURE_COLS_V4 + te
for col in CATEGORICAL_V4:
    train[col] = train[col].astype("category")
    test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)
print(f"[데이터] train {len(train):,} / test {len(test):,} | 피처 {len(FEATURES)}개")

PARAMS = dict(n_estimators=4000, learning_rate=0.03, num_leaves=63, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, min_child_samples=40,
              random_state=42, n_jobs=-1, verbose=-1)
oof_x, oof_y = np.full(len(train), np.nan), np.full(len(train), np.nan)
pred_x, pred_y = np.zeros(len(test)), np.zeros(len(test))
imp = np.zeros(len(FEATURES))

for axis, oof, pred in [("x", oof_x, pred_x), ("y", oof_y, pred_y)]:
    y = train[f"tgt_{axis}"] - train[f"start_{axis}"]
    for f in range(5):
        tr, va = fold != f, fold == f
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(train.loc[tr, FEATURES], y[tr], eval_set=[(train.loc[va, FEATURES], y[va])],
              categorical_feature=CATEGORICAL_V4,
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[va] = train.loc[va, f"start_{axis}"].to_numpy() + m.predict(train.loc[va, FEATURES])
        pred += (test[f"start_{axis}"].to_numpy() + m.predict(test[FEATURES])) / 5
        imp += m.feature_importances_ / 10

oof_x, oof_y = np.clip(oof_x, 0, 105), np.clip(oof_y, 0, 68)
pred_x, pred_y = np.clip(pred_x, 0, 105), np.clip(pred_y, 0, 68)
cv = np.sqrt((oof_x - train["tgt_x"]) ** 2 + (oof_y - train["tgt_y"]) ** 2).mean()

os.makedirs("data/cache", exist_ok=True)
np.savez("data/cache/gbdt_oof.npz",
         ep=train["game_episode"].to_numpy().astype(str), oof_x=oof_x, oof_y=oof_y,
         test_ep=test["game_episode"].to_numpy().astype(str), pred_x=pred_x, pred_y=pred_y)

print("\n===== 로컬 CV =====")
print(f"  v4 (기준)         : 14.0174")
print(f"  ★ v5 (템포 보강)  : {cv:.4f}")
print("\n----- importance 상위 12 -----")
fi = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values("importance", ascending=False)
print(fi.head(12).to_string(index=False))

sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(pd.DataFrame({"game_episode": test["game_episode"], "end_x": pred_x, "end_y": pred_y}),
                on="game_episode", how="left")
out.to_csv("submissions/lgbm_v5.csv", index=False)
print("\n[완료] submissions/lgbm_v5.csv")

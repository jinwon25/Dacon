"""앙상블 — 서로 다른(decorrelated) 모델을 섞어 '서로의 실수'를 상쇄한다.

[원리(당신이 말한 그대로)]
- 같은 14.4점이라도 모델마다 '틀리는 지점'이 다르다.
- 예측을 평균하면 한 모델이 과하게 틀린 곳을 다른 모델이 당겨줘 평균 오차가 준다.
- 그래서 다양성이 핵심 → ① 다른 알고리즘(LGBM/CatBoost/XGBoost) ② 다른 타깃(disp/abs).

[검증]
- 각 모델 '단독' 로컬 CV와 '앙상블(평균)' 로컬 CV를 같이 출력 → 앙상블이 최고 단일을 이기나?
- 모델 간 예측 상관도 같이 출력 → 덜 닮을수록(낮을수록) 앙상블 이득이 큼.

실행:  python src/train_ensemble.py   (모델 4종 × 2축 × 5fold = 40개 학습, 수 분 소요)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, "src")
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from features import FEATURE_COLS, build_features  # noqa: E402


# ---- v3와 동일한 피처(각도 + 타깃인코딩) ----
def add_geom(df):
    df = df.copy()
    eps = 1e-6
    df["w5_netdist"] = np.hypot(df["w5_dx"], df["w5_dy"])
    df["w5_angle"] = np.arctan2(df["w5_dy"], df["w5_dx"])
    df["w5_straight"] = df["w5_netdist"] / (df["w5_pathlen"] + eps)
    df["dist_to_goal"] = np.hypot(105 - df["start_x"], 34 - df["start_y"])
    df["ang_to_goal"] = np.arctan2(34 - df["start_y"], 105 - df["start_x"])
    df["prev_angle"] = np.arctan2(df["prev_dy"].fillna(0), df["prev_dx"].fillna(0))
    return df


GEOM_COLS = ["w5_netdist", "w5_angle", "w5_straight", "dist_to_goal", "ang_to_goal", "prev_angle"]


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


train = build_features("'data/train.csv'")
test = build_features("read_csv('data/test/*/*.csv', union_by_name = true)")
train = train[train["tgt_x"].notna()].reset_index(drop=True)
test = test.reset_index(drop=True)
train, test = add_geom(train), add_geom(test)
fold = (train["game_id"].astype("int64") % 5).to_numpy()
te = []
te += oof_encode(train, test, fold, "player_id", ["tgt_x", "tgt_y"], m=20)
te += oof_encode(train, test, fold, "team_id", ["tgt_x", "tgt_y"], m=50)

# 앙상블은 라이브러리 3종 공통으로 '숫자 피처'만 사용(범주형 prev_type 제외 → 코드 단순·공정 비교)
FEATURES = [c for c in (FEATURE_COLS + GEOM_COLS + te) if c != "prev_type"]
print(f"[데이터] train {len(train):,} / test {len(test):,} | 숫자 피처 {len(FEATURES)}개")

LGB = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, min_child_samples=40, n_jobs=-1, verbose=-1)
XGB = dict(n_estimators=3000, learning_rate=0.03, max_depth=8, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=20, tree_method="hist", n_jobs=-1)
CAT = dict(iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
           loss_function="RMSE", thread_count=-1)


def fit_predict(kind, Xtr, ytr, Xva, yva, Xte, seed):
    if kind == "lgb":
        mdl = lgb.LGBMRegressor(**LGB, random_state=seed)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(80, verbose=False)])
    elif kind == "xgb":
        mdl = xgb.XGBRegressor(**XGB, random_state=seed, early_stopping_rounds=80)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        mdl = CatBoostRegressor(**CAT, random_seed=seed)
        mdl.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=80, verbose=False)
    return mdl.predict(Xva), mdl.predict(Xte)


def run_cv(kind, mode, seed=42):
    """단일 모델 5-fold CV → (oof_x, oof_y, pred_x, pred_y, cv)"""
    oof_x, oof_y = np.full(len(train), np.nan), np.full(len(train), np.nan)
    pred_x, pred_y = np.zeros(len(test)), np.zeros(len(test))
    for axis, oof, pred in [("x", oof_x, pred_x), ("y", oof_y, pred_y)]:
        y = train[f"tgt_{axis}"] - train[f"start_{axis}"] if mode == "disp" else train[f"tgt_{axis}"]
        for f in range(5):
            tr, va = fold != f, fold == f
            p_va, p_te = fit_predict(kind, train.loc[tr, FEATURES], y[tr],
                                     train.loc[va, FEATURES], y[va], test[FEATURES], seed)
            if mode == "disp":
                p_va = train.loc[va, f"start_{axis}"].to_numpy() + p_va
                p_te = test[f"start_{axis}"].to_numpy() + p_te
            oof[va] = p_va
            pred += p_te / 5
    oof_x, oof_y = np.clip(oof_x, 0, 105), np.clip(oof_y, 0, 68)
    pred_x, pred_y = np.clip(pred_x, 0, 105), np.clip(pred_y, 0, 68)
    cv = np.sqrt((oof_x - train["tgt_x"]) ** 2 + (oof_y - train["tgt_y"]) ** 2).mean()
    return oof_x, oof_y, pred_x, pred_y, cv


MODELS = [
    ("LGBM  disp", "lgb", "disp"),
    ("CatBoost disp", "cat", "disp"),
    ("XGBoost disp", "xgb", "disp"),
    ("LGBM  abs", "lgb", "abs"),
]
res = {}
print("\n----- 단일 모델 학습 중 -----")
for name, kind, mode in MODELS:
    ox, oy, px, py, cv = run_cv(kind, mode)
    res[name] = dict(ox=ox, oy=oy, px=px, py=py, cv=cv)
    print(f"  {name:14s} 로컬 CV {cv:.4f}")

# ---- 앙상블 = 단순 평균 ----
names = list(res)
ens_ox = np.mean([res[n]["ox"] for n in names], axis=0)
ens_oy = np.mean([res[n]["oy"] for n in names], axis=0)
ens_cv = np.sqrt((ens_ox - train["tgt_x"]) ** 2 + (ens_oy - train["tgt_y"]) ** 2).mean()

best_single = min(res[n]["cv"] for n in names)
print(f"\n  ★ 앙상블(4모델 평균) 로컬 CV {ens_cv:.4f}   (최고 단일 {best_single:.4f}, "
      f"이득 {best_single - ens_cv:+.4f})")

# ---- 모델 간 예측 상관 (낮을수록 다양 → 앙상블 이득 큼) ----
print("\n----- 모델 간 OOF 예측 상관(x축, 1에 가까울수록 닮음) -----")
oofx = pd.DataFrame({n: res[n]["ox"] for n in names})
print(oofx.corr().round(3).to_string())

# ---- 제출 파일 ----
ens_px = np.mean([res[n]["px"] for n in names], axis=0)
ens_py = np.mean([res[n]["py"] for n in names], axis=0)
sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(pd.DataFrame({"game_episode": test["game_episode"],
                              "end_x": ens_px, "end_y": ens_py}),
                on="game_episode", how="left")
out.to_csv("submissions/ensemble_v4.csv", index=False)
print("\n[완료] submissions/ensemble_v4.csv")

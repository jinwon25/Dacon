"""v3 — 각도 피처 + 타깃 인코딩(누수 안전) + 변위 타깃 LightGBM.

이번에 쌓는 두 레버
  (1) 기하 피처: 빌드업/골문 방향을 '각도·거리'로. (dx,dy 만으론 트리가 방향을 보기 어려움)
  (2) 타깃 인코딩: player_id / team_id 를 "그 그룹의 평균 도착 좌표"로 바꿔 숫자 피처화.

★ 타깃 인코딩의 함정 = 누수(leakage). "정답으로 만든 피처"라, 어떤 행을 인코딩할 때
   그 행 자신의 정답이 평균에 섞이면 컨닝이 된다(→ 로컬 점수만 좋고 LB는 폭망).
   해결 = Out-Of-Fold 인코딩:
     - train의 각 fold 행은, '자기 fold를 뺀 나머지 fold'의 정답 평균으로 인코딩.
     - test 행은 train 전체 평균으로 인코딩.
   이러면 어떤 행도 '자기 정답'을 자기 피처로 쓰지 않는다.  (= 이번 라운드 핵심 개념)
   + 표본 적은 그룹은 전체 평균 쪽으로 당겨주는 smoothing(m) 으로 과신 방지.

실행:  python src/train_v3.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, "src")
import lightgbm as lgb  # noqa: E402
from features import CATEGORICAL, FEATURE_COLS, build_features  # noqa: E402


def add_geom(df):
    """방향을 각도/거리로 — 트리가 '어느 쪽으로 향하나'를 보기 쉽게."""
    df = df.copy()
    eps = 1e-6
    df["w5_netdist"] = np.hypot(df["w5_dx"], df["w5_dy"])              # 최근5수 직선 이동량
    df["w5_angle"] = np.arctan2(df["w5_dy"], df["w5_dx"])             # 그 방향(각도)
    df["w5_straight"] = df["w5_netdist"] / (df["w5_pathlen"] + eps)    # 빌드업이 얼마나 직선적
    df["dist_to_goal"] = np.hypot(105 - df["start_x"], 34 - df["start_y"])   # 골 중앙까지 거리
    df["ang_to_goal"] = np.arctan2(34 - df["start_y"], 105 - df["start_x"])  # 골 방향 각도
    df["prev_angle"] = np.arctan2(df["prev_dy"].fillna(0), df["prev_dx"].fillna(0))
    return df


GEOM_COLS = ["w5_netdist", "w5_angle", "w5_straight",
             "dist_to_goal", "ang_to_goal", "prev_angle"]


def oof_encode(train, test, fold, cat, targets, m=20):
    """Out-Of-Fold 타깃 인코딩. cat 그룹의 target 평균(스무딩)을 피처로.
    train: 자기 fold 제외하고 계산 / test: train 전체로 계산. 새 컬럼명 리스트도 반환."""
    new_cols = []
    for t in targets:
        gmean = train[t].mean()
        col = np.empty(len(train))
        for f in np.unique(fold):
            other = fold != f          # ★ 자기 fold(f)를 뺀 나머지로만 평균 계산
            g = train.loc[other].groupby(cat)[t].agg(["mean", "count"])
            smooth = (g["mean"] * g["count"] + gmean * m) / (g["count"] + m)
            mp = smooth.to_dict()
            here = fold == f
            col[here] = train.loc[here, cat].map(mp).fillna(gmean).to_numpy()
        name = f"te_{cat}_{t}"
        train[name] = col
        # test: train 전체로 (test엔 정답이 없으니 누수 걱정 없음)
        g = train.groupby(cat)[t].agg(["mean", "count"])
        smooth = (g["mean"] * g["count"] + gmean * m) / (g["count"] + m)
        test[name] = test[cat].map(smooth.to_dict()).fillna(gmean).to_numpy()
        new_cols.append(name)
    return new_cols


# ---- 데이터 준비 ----
train = build_features("'data/train.csv'")
test = build_features("read_csv('data/test/*/*.csv', union_by_name = true)")
train = train[train["tgt_x"].notna()].reset_index(drop=True)
test = test.reset_index(drop=True)
train, test = add_geom(train), add_geom(test)

fold = (train["game_id"].astype("int64") % 5).to_numpy()

te_cols = []
te_cols += oof_encode(train, test, fold, "player_id", ["tgt_x", "tgt_y"], m=20)
te_cols += oof_encode(train, test, fold, "team_id", ["tgt_x", "tgt_y"], m=50)

FEATURES = FEATURE_COLS + GEOM_COLS + te_cols
for col in CATEGORICAL:
    train[col] = train[col].astype("category")
    test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)
print(f"[데이터] train {len(train):,} / test {len(test):,} | 피처 {len(FEATURES)}개 "
      f"(기본 {len(FEATURE_COLS)} + 기하 {len(GEOM_COLS)} + 인코딩 {len(te_cols)})")

# ---- 변위 타깃 LightGBM 5-fold CV (v2에서 이긴 방식) ----
PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=40, random_state=42, n_jobs=-1, verbose=-1,
)
oof_x, oof_y = np.full(len(train), np.nan), np.full(len(train), np.nan)
pred_x, pred_y = np.zeros(len(test)), np.zeros(len(test))
imp = np.zeros(len(FEATURES))

for axis, oof, pred in [("x", oof_x, pred_x), ("y", oof_y, pred_y)]:
    y = train[f"tgt_{axis}"] - train[f"start_{axis}"]      # 변위 타깃
    for f in range(5):
        tr, va = fold != f, fold == f
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(
            train.loc[tr, FEATURES], y[tr],
            eval_set=[(train.loc[va, FEATURES], y[va])],
            categorical_feature=CATEGORICAL,
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        oof[va] = train.loc[va, f"start_{axis}"].to_numpy() + model.predict(train.loc[va, FEATURES])
        pred += (test[f"start_{axis}"].to_numpy() + model.predict(test[FEATURES])) / 5
        imp += model.feature_importances_ / 10

oof_x, oof_y = np.clip(oof_x, 0, 105), np.clip(oof_y, 0, 68)
pred_x, pred_y = np.clip(pred_x, 0, 105), np.clip(pred_y, 0, 68)
cv = np.sqrt((oof_x - train["tgt_x"]) ** 2 + (oof_y - train["tgt_y"]) ** 2).mean()

print("\n===== 로컬 CV (평균 유클리드 거리, 낮을수록 좋음) =====")
print(f"  v2 변위타깃(기준)  : 14.4436")
print(f"  ★ v3 (각도+인코딩) : {cv:.4f}")

print("\n----- importance 상위 12 (te_player vs te_team 확인) -----")
fi = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values(
    "importance", ascending=False)
print(fi.head(12).to_string(index=False))

sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(
    pd.DataFrame({"game_episode": test["game_episode"], "end_x": pred_x, "end_y": pred_y}),
    on="game_episode", how="left")
out.to_csv("submissions/lgbm_v3.csv", index=False)
print("\n[완료] submissions/lgbm_v3.csv")

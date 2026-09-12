"""첫 LightGBM 모델 — '상수 보정(B')'에서 '상황 맞춤 예측'으로.

[우리의 추론]
- EDA에서 본 것: 마지막 패스는 평균 +13.5 전진(B')하지만, *어디서 차느냐에 따라* 다를 것이다.
  · 골 앞(start_x 큼)에서는 더 못 전진(공간이 없음) → 옆/뒤로?
  · 중앙선 부근에서는 시원하게 전진?
  · 빌드업이 향하던 방향(w5_dx/dy)이 도착 방향과 이어질 것이다.
- 가설: start 좌표 + 빌드업 방향을 LightGBM에 주면, '위치마다 다른 전진량'을 스스로 학습해
        상수 B'(로컬 18.16)를 이길 것이다.
- 검증: B' 때와 똑같은 game_id 기반 분할로 로컬 CV(평균 유클리드 거리)를 재서 비교.

실행 (대회 폴더에서):  python src/train_lgbm.py
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

# 1) 피처 만들기 (train: 정답 있음 / test: 정답 NULL)
train = build_features("'data/train.csv'")
test = build_features("read_csv('data/test/*/*.csv', union_by_name = true)")
train = train[train["tgt_x"].notna()].reset_index(drop=True)
test = test.reset_index(drop=True)
print(f"[데이터] train {len(train):,} / test {len(test):,}  | 피처 {len(FEATURE_COLS)}개")

# 범주형(prev_type)은 train/test 카테고리를 맞춰준다
for col in CATEGORICAL:
    train[col] = train[col].astype("category")
    test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

# 2) game_id 기준 5-fold (한 경기는 한 fold에만 → 누수 방지, B' 검증과 동일 철학)
fold = (train["game_id"].astype("int64") % 5).to_numpy()

oof_x = np.full(len(train), np.nan)
oof_y = np.full(len(train), np.nan)
pred_x = np.zeros(len(test))
pred_y = np.zeros(len(test))

params = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=40, random_state=42, n_jobs=-1, verbose=-1,
)
imp = np.zeros(len(FEATURE_COLS))

for f in range(5):
    tr, va = fold != f, fold == f
    for tgt, oof, pred in [("tgt_x", oof_x, pred_x), ("tgt_y", oof_y, pred_y)]:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train.loc[tr, FEATURE_COLS], train.loc[tr, tgt],
            eval_set=[(train.loc[va, FEATURE_COLS], train.loc[va, tgt])],
            categorical_feature=CATEGORICAL,
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        oof[va] = model.predict(train.loc[va, FEATURE_COLS])
        pred += model.predict(test[FEATURE_COLS]) / 5
        imp += model.feature_importances_ / 10  # 2축 × 5fold 평균

# 3) 경기장 밖 예측은 clip (B'에서 효과 봤던 처리)
oof_x, oof_y = np.clip(oof_x, 0, 105), np.clip(oof_y, 0, 68)


def euclid(px, py, tx, ty):
    return np.sqrt((px - tx) ** 2 + (py - ty) ** 2).mean()


cv = euclid(oof_x, oof_y, train["tgt_x"], train["tgt_y"])

# 참고: 같은 데이터에서 A(end=start), B'(start+평균전진+clip) 점수도 같이 계산
DX = (train["tgt_x"] - train["start_x"]).mean()
DY = (train["tgt_y"] - train["start_y"]).mean()
a_cv = euclid(train["start_x"], train["start_y"], train["tgt_x"], train["tgt_y"])
b_cv = euclid(np.clip(train["start_x"] + DX, 0, 105),
              np.clip(train["start_y"] + DY, 0, 68),
              train["tgt_x"], train["tgt_y"])

print("\n===== 로컬 검증 (평균 유클리드 거리, 낮을수록 좋음) =====")
print(f"  A  도착=출발            : {a_cv:6.4f}")
print(f"  B' 출발+평균전진+clip   : {b_cv:6.4f}")
print(f"  ★ LightGBM (피처 모델)  : {cv:6.4f}")

print("\n----- 어떤 피처가 중요했나 (LightGBM importance 상위) -----")
fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
print(fi.sort_values("importance", ascending=False).to_string(index=False))

# 4) 제출 파일 생성
pred_x, pred_y = np.clip(pred_x, 0, 105), np.clip(pred_y, 0, 68)
sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(
    pd.DataFrame({"game_episode": test["game_episode"], "end_x": pred_x, "end_y": pred_y}),
    on="game_episode", how="left",
)
os.makedirs("submissions", exist_ok=True)
out_path = "submissions/lgbm_v1.csv"
out.to_csv(out_path, index=False)
print(f"\n[완료] 제출파일 {out_path}  (행수={len(out)})")

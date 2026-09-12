"""실험: 타깃을 '절대 좌표' vs '변위(도착-출발)'로 두면 뭐가 나을까?

[가설(당신 선택)]
- 절대 좌표(tgt_x)를 바로 예측하는 대신, '출발점에서 얼마나/어디로 움직이는지'(tgt_x - start_x)를
  예측하면 더 쉬울 수 있다. 이유: start_x는 이미 피처라 모델이 그 부분을 다시 배울 필요가 없고,
  변위는 분산이 작고 좌우 대칭(평균≈0)이라 학습이 안정적일 수 있다.
  (= B'/출발점을 1단계로 깔고, 그 위 '잔차'만 모델이 배우는 셈.)

[실험 설계 — 변수 하나만 바꾸기]
- 피처, fold(게임 5분할), LightGBM 파라미터를 '완전히 동일'하게 두고
  타깃 표현(abs vs disp)만 바꿔 로컬 CV(평균 유클리드 거리)를 비교한다.

실행:  python src/exp_target.py
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

train = build_features("'data/train.csv'")
test = build_features("read_csv('data/test/*/*.csv', union_by_name = true)")
train = train[train["tgt_x"].notna()].reset_index(drop=True)
test = test.reset_index(drop=True)
for col in CATEGORICAL:
    train[col] = train[col].astype("category")
    test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

fold = (train["game_id"].astype("int64") % 5).to_numpy()
PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=40, random_state=42, n_jobs=-1, verbose=-1,
)


def run_cv(mode: str):
    """mode='abs'  → 타깃 = tgt_x, tgt_y (절대좌표)
       mode='disp' → 타깃 = tgt - start (변위), 예측 후 start 더해 복원"""
    oof_x, oof_y = np.full(len(train), np.nan), np.full(len(train), np.nan)
    pred_x, pred_y = np.zeros(len(test)), np.zeros(len(test))
    for axis, oof, pred in [("x", oof_x, pred_x), ("y", oof_y, pred_y)]:
        s_col = f"start_{axis}"
        t_col = f"tgt_{axis}"
        # 학습 타깃 만들기
        if mode == "disp":
            y_train = train[t_col] - train[s_col]
        else:
            y_train = train[t_col]
        for f in range(5):
            tr, va = fold != f, fold == f
            model = lgb.LGBMRegressor(**PARAMS)
            model.fit(
                train.loc[tr, FEATURE_COLS], y_train[tr],
                eval_set=[(train.loc[va, FEATURE_COLS], y_train[va])],
                categorical_feature=CATEGORICAL,
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            p_va = model.predict(train.loc[va, FEATURE_COLS])
            p_te = model.predict(test[FEATURE_COLS])
            if mode == "disp":  # 변위 예측 → 절대좌표로 복원
                p_va = train.loc[va, s_col].to_numpy() + p_va
                p_te = test[s_col].to_numpy() + p_te
            oof[va] = p_va
            pred += p_te / 5
    oof_x, oof_y = np.clip(oof_x, 0, 105), np.clip(oof_y, 0, 68)
    pred_x, pred_y = np.clip(pred_x, 0, 105), np.clip(pred_y, 0, 68)
    cv = np.sqrt((oof_x - train["tgt_x"]) ** 2 + (oof_y - train["tgt_y"]) ** 2).mean()
    return cv, pred_x, pred_y


print("[실험] 타깃 표현 A/B 비교 (피처·fold·파라미터 동일)\n")
results = {}
for mode in ("abs", "disp"):
    cv, px, py = run_cv(mode)
    results[mode] = (cv, px, py)
    print(f"  {mode:4s} → 로컬 CV {cv:.4f}")

cv_abs = results["abs"][0]
cv_disp = results["disp"][0]
print(f"\n  차이(disp - abs) = {cv_disp - cv_abs:+.4f}  "
      f"→ {'변위 타깃이 더 좋음 ✅' if cv_disp < cv_abs else '절대좌표가 더 좋음'}")

# 더 좋은 쪽으로 제출 파일 생성
best = "disp" if cv_disp < cv_abs else "abs"
_, px, py = results[best]
sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(
    pd.DataFrame({"game_episode": test["game_episode"], "end_x": px, "end_y": py}),
    on="game_episode", how="left",
)
out_path = f"submissions/lgbm_v2_{best}.csv"
out.to_csv(out_path, index=False)
print(f"\n[완료] 더 나은 타깃({best})으로 제출파일 생성: {out_path}")

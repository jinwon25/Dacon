"""글로벌 LightGBM (direct multi-horizon) — 로컬 홀드아웃 가중 SMAPE 평가.

학습: 전체 train 에서 (anchor t, h) 샘플 생성, target=y[t+h]. log1p 타깃, L1 목적.
검증: test 와 동일 구조의 마지막 7일 홀드아웃 (make_holdout k=0).
요일평균(swl_mean)과의 블렌드도 함께 측정.

    python src/train_lgbm.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from metric import make_holdout, score_long, score_per_store
from features import (build_training_table, build_inference_table,
                      FEATURE_COLS, CAT_COLS)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
HORIZON = 7

LGB_PARAMS = dict(
    objective="regression_l1",
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    max_depth=-1,
    n_jobs=-1,
    verbose=-1,
)


def _prep_cats(df, cats_ref=None):
    df = df.copy()
    ref = {}
    for c in CAT_COLS:
        if cats_ref is None:
            df[c] = df[c].astype("category")
            ref[c] = df[c].cat.categories
        else:
            df[c] = pd.Categorical(df[c], categories=cats_ref[c])
    return (df, ref) if cats_ref is None else df


def train_and_eval():
    train = pd.read_csv(ROOT / "data" / "train" / "train.csv")
    train["영업일자"] = pd.to_datetime(train["영업일자"])
    max_date = train["영업일자"].max()
    cutoff = max_date - pd.Timedelta(days=HORIZON - 1)  # 홀드아웃 target 시작일 (제외 경계)
    print(f"train max={max_date.date()}  holdout target >= {cutoff.date()} (제외)  ...")

    # 학습 테이블
    tbl = build_training_table(train, cutoff_date=cutoff, horizon=HORIZON, stride=1)
    print(f"학습 샘플: {len(tbl):,}  피처 {len(FEATURE_COLS)}개")
    X = tbl[FEATURE_COLS + CAT_COLS]
    y = np.log1p(tbl["target"].clip(lower=0).to_numpy())
    X, cats_ref = _prep_cats(X)

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y, categorical_feature=CAT_COLS)

    # 홀드아웃 추론 (test 구조 동일)
    fold = make_holdout(train, lookback=28, horizon=HORIZON, n_folds=1)[0]
    inp = fold["input_df"]
    ans = fold["answer_df"].copy()
    ans["date"] = pd.to_datetime(ans["date"])

    inf = build_inference_table(inp, horizon=HORIZON)
    Xi = _prep_cats(inf[FEATURE_COLS + CAT_COLS], cats_ref=cats_ref)
    pred = np.expm1(model.predict(Xi)).clip(min=0)
    inf = inf.assign(pred=pred)

    # 정답 결합
    ans_map = {(r.item, pd.Timestamp(r.date)): r.qty for r in ans.itertuples()}
    inf["true"] = [ans_map.get((it, pd.Timestamp(d)), np.nan)
                   for it, d in zip(inf["item"], inf["target_date"])]
    ev = inf.dropna(subset=["true"]).copy()

    # 점수: LGBM 단독 / 요일평균 / 블렌드
    print("\n=== 홀드아웃 가중 SMAPE (fold0, 낮을수록 좋음, LB 스케일) ===")
    print(f"  요일평균(swl_mean) 단독 : {score_long(ev.assign(pred=ev['swl_mean'])):.4f}")
    print(f"  LGBM 단독              : {score_long(ev):.4f}")
    for a in (0.3, 0.5, 0.7):
        blpred = a * ev["pred"] + (1 - a) * ev["swl_mean"]
        print(f"  블렌드 a*LGBM+(1-a)요일 a={a}: {score_long(ev.assign(pred=blpred)):.4f}")
    print("  (참고: 무학습 weekday+mean7 fold0 = 0.7275)")

    print("\n=== LGBM 단독 영업장별 SMAPE (fold0) ===")
    print(score_per_store(ev).to_string(index=False))

    # 피처 중요도 top12
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== feature importance top12 ===")
    print(imp.head(12).to_string())
    return model


if __name__ == "__main__":
    train_and_eval()

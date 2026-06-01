"""강한 무학습 블렌드 제출 생성 — 0.4*요일평균 + 0.6*28일평균, 전역 스케일.

로컬 홀드아웃(4폴드) 가중 SMAPE ≈ 0.677~0.679 (실험: src/experiments.py).
공식 LSTM 베이스라인(LB Public 0.711 / Private 0.694)을 로컬에서 앞서는 첫 제출 후보.
누수 안전: 각 test 샘플의 28일 입력만 사용.

    python src/predict_blend.py   # -> submissions/blend_submission.csv
"""
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd

from baselines import predictors
from metric import make_holdout, score_long

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBM = ROOT / "submissions"

WD_W, M28_W, SCALE = 0.4, 0.6, 1.0  # mean7 가중 0 (28일 평균이 더 안정적)


def blend_pred(input_g: pd.DataFrame) -> np.ndarray:
    p = predictors(input_g)
    return np.clip(SCALE * (WD_W * p["weekday_mean"] + M28_W * p["mean28"]), 0, None)


def holdout_score(n_folds=4) -> float:
    train = pd.read_csv(DATA / "train" / "train.csv")
    folds = make_holdout(train, lookback=28, horizon=7, n_folds=n_folds)
    rows = []
    for fold in folds:
        inp = fold["input_df"].sort_values(["item", "date"])
        ans = {it: g.sort_values("date")["qty"].to_numpy(float)
               for it, g in fold["answer_df"].groupby("item")}
        for it, g in inp.groupby("item"):
            if it in ans:
                rows.append(pd.DataFrame({"item": it, "true": ans[it], "pred": blend_pred(g)}))
    return score_long(pd.concat(rows, ignore_index=True))


def make_submission():
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[sub.columns[1:]] = sub[sub.columns[1:]].astype(float)  # int64→float (예측 대입)
    for path in sorted(glob.glob(str(DATA / "test" / "TEST_*.csv"))):
        prefix = re.search(r"(TEST_\d+)", Path(path).name).group(1)
        t = pd.read_csv(path).rename(
            columns={"영업일자": "date", "영업장명_메뉴명": "item", "매출수량": "qty"})
        t["date"] = pd.to_datetime(t["date"])
        for it, g in t.sort_values(["item", "date"]).groupby("item"):
            pred = blend_pred(g)
            for h in range(1, 8):
                row = f"{prefix}+{h}일"
                if it in sub.columns:
                    sub.loc[sub["영업일자"] == row, it] = pred[h - 1]
    SUBM.mkdir(exist_ok=True)
    out = SUBM / "blend_submission.csv"
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    s = holdout_score()
    print(f"config: {WD_W}*weekday + {M28_W}*mean28, scale={SCALE}")
    print(f"홀드아웃(4폴드) 가중 SMAPE = {s:.4f}")
    out = make_submission()
    print(f"saved -> {out}")

"""SMAPE 최적화 실험 - 요일평균/최근평균 블렌드 가중 + 전역 스케일 그리드 탐색.

SMAPE 는 상대오차이고 타깃 분포가 우편향이라, 예측을 약간 낮추는(scale<1) 것이 유리할 수 있다.
무학습 예측기만으로 블렌드 가중 alpha 와 전역 스케일 c 를 홀드아웃(4폴드)에서 탐색.

    python src/experiments.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from metric import make_holdout, score_long
from baselines import predictors

ROOT = Path(__file__).resolve().parents[1]


def stacked_frame(n_folds=4):
    train = pd.read_csv(ROOT / "data" / "train" / "train.csv")
    folds = make_holdout(train, lookback=28, horizon=7, n_folds=n_folds)
    out = []
    for k, fold in enumerate(folds):
        inp = fold["input_df"].sort_values(["item", "date"])
        ans = fold["answer_df"].sort_values(["item", "date"])
        ans_by_item = {it: g["qty"].to_numpy(float) for it, g in ans.groupby("item")}
        for it, g in inp.groupby("item"):
            if it not in ans_by_item:
                continue
            p = predictors(g)
            out.append(pd.DataFrame({
                "fold": k, "item": it, "true": ans_by_item[it],
                "wd": p["weekday_mean"], "m7": p["mean7"], "m28": p["mean28"],
                "nz": p["nz_mean"], "wd_nz": p["wd_nz"],
            }))
    return pd.concat(out, ignore_index=True)


def main():
    df = stacked_frame(n_folds=4)

    def score(pred):
        return score_long(df.assign(pred=pred))

    # 1) 블렌드 가중 alpha (wd vs m7)
    print("=== alpha*wd + (1-alpha)*m7 (scale=1.0, 4폴드 평균 가중SMAPE) ===")
    best = (None, 9)
    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        s = score(alpha * df["wd"] + (1 - alpha) * df["m7"])
        if s < best[1]:
            best = (alpha, s)
        print(f"  alpha={alpha:.1f}: {s:.4f}")
    alpha = best[0]
    print(f"  -> best alpha={alpha} ({best[1]:.4f})")

    # 2) 전역 스케일 c
    base = alpha * df["wd"] + (1 - alpha) * df["m7"]
    print(f"\n=== c * blend(alpha={alpha}) - 전역 스케일 (4폴드 평균) ===")
    bestc = (None, 9)
    for c in np.round(np.arange(0.70, 1.111, 0.02), 3):
        s = score(c * base)
        if s < bestc[1]:
            bestc = (c, s)
    for c in np.round(np.arange(0.80, 1.05, 0.05), 3):
        print(f"  c={c:.2f}: {score(c * base):.4f}")
    print(f"  -> best c={bestc[0]} ({bestc[1]:.4f})")

    # 3) 3-항 블렌드 + 스케일 (wd, m7, m28)
    print("\n=== wd/m7/m28 단순 3-blend + best c 탐색 ===")
    grid = []
    for a in np.arange(0, 1.01, 0.2):
        for b in np.arange(0, 1.01 - a, 0.2):
            cc = 1 - a - b
            base3 = a * df["wd"] + b * df["m7"] + cc * df["m28"]
            for c in np.round(np.arange(0.80, 1.06, 0.02), 3):
                grid.append((round(a, 2), round(b, 2), round(cc, 2), c, score(c * base3)))
    g = pd.DataFrame(grid, columns=["wd", "m7", "m28", "c", "smape"]).sort_values("smape")
    print(g.head(8).to_string(index=False))
    print(f"\n참고: 기존 weekday+mean7(0.5) 4폴드 평균 = 0.7050")


if __name__ == "__main__":
    main()

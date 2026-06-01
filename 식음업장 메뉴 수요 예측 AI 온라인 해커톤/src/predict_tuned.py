"""업장별 튜닝 블렌드 제출 생성 — (weekday_mean, mean28, nz_mean) 업장별 가중 + 스케일.

설정은 `tune_per_store.search()` 로 4폴드 홀드아웃에서 자동 도출(하드코딩 없음, 재현 가능).
로컬 홀드아웃 4폴드 ≈ 0.565 (전역 블렌드 0.679 대비 -0.11).
핵심: 0 actual 평가 제외 → 비-0 평균(nz_mean)이 SMAPE-최적 수준에 가까움.

    python src/predict_tuned.py   # -> submissions/tuned_submission.csv
"""
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd

from baselines import predictors
from metric import make_holdout, score_long, store_of
from tune_per_store import load_stacked, search, COMP, GLOBAL

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBM = ROOT / "submissions"
GLOBAL_CFG = GLOBAL  # 미지 업장 fallback
PRED_KEY = {"wd": "weekday_mean", "m28": "mean28", "nz": "nz_mean", "wd_nz": "wd_nz"}


def get_configs():
    best = search(load_stacked())
    return {s: cfg for s, (cfg, _) in best.items()}


def blend_pred(input_g: pd.DataFrame, cfg: dict) -> np.ndarray:
    p = predictors(input_g)
    base = sum(cfg.get(k, 0.0) * p[PRED_KEY[k]] for k in COMP)
    return np.clip(cfg["c"] * base, 0, None)


def holdout_score(configs, n_folds=4) -> float:
    train = pd.read_csv(DATA / "train" / "train.csv")
    folds = make_holdout(train, lookback=28, horizon=7, n_folds=n_folds)
    rows = []
    for fold in folds:
        inp = fold["input_df"].sort_values(["item", "date"])
        ans = {it: g.sort_values("date")["qty"].to_numpy(float)
               for it, g in fold["answer_df"].groupby("item")}
        for it, g in inp.groupby("item"):
            if it in ans:
                cfg = configs.get(store_of(it), GLOBAL_CFG)
                rows.append(pd.DataFrame({"item": it, "true": ans[it], "pred": blend_pred(g, cfg)}))
    return score_long(pd.concat(rows, ignore_index=True))


def make_submission(configs):
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[sub.columns[1:]] = sub[sub.columns[1:]].astype(float)
    for path in sorted(glob.glob(str(DATA / "test" / "TEST_*.csv"))):
        prefix = re.search(r"(TEST_\d+)", Path(path).name).group(1)
        t = pd.read_csv(path).rename(
            columns={"영업일자": "date", "영업장명_메뉴명": "item", "매출수량": "qty"})
        t["date"] = pd.to_datetime(t["date"])
        for it, g in t.sort_values(["item", "date"]).groupby("item"):
            if it not in sub.columns:
                continue
            cfg = configs.get(store_of(it), GLOBAL_CFG)
            pred = blend_pred(g, cfg)
            for h in range(1, 8):
                sub.loc[sub["영업일자"] == f"{prefix}+{h}일", it] = pred[h - 1]
    SUBM.mkdir(exist_ok=True)
    out = SUBM / "tuned_submission.csv"
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    configs = get_configs()
    print(f"홀드아웃(4폴드) 가중 SMAPE = {holdout_score(configs):.4f}")
    out = make_submission(configs)
    print(f"saved -> {out}")

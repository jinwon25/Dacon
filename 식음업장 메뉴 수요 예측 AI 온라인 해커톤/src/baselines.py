"""단순 통계 베이스라인 평가 — 28일 입력만으로 7일 예측 (누수 규칙 준수).

로컬 홀드아웃(make_holdout)에서 여러 무학습 예측기를 가중 SMAPE 로 비교한다.
목적: 모델을 키우기 전에 '요일 평균' 같은 강한 단순 베이스라인의 바닥 점수를 확보하고,
LB(베이스라인 ~0.69)와 같은 자릿수가 나오는지로 검증 스케일을 교정한다.

    python src/baselines.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from metric import make_holdout, score_long, score_per_store, store_of  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 7


def future_weekdays(last_date, horizon=HORIZON):
    last_wd = pd.Timestamp(last_date).weekday()
    return [(last_wd + h) % 7 for h in range(1, horizon + 1)]


def predictors(input_g: pd.DataFrame, horizon=HORIZON):
    """input_g: 한 메뉴의 28일 (date,item,qty), 시간순. dict[str -> length-7 ndarray] 반환."""
    v = input_g["qty"].to_numpy(dtype=float)
    wd = input_g["date"].dt.weekday.to_numpy()
    fut_wd = future_weekdays(input_g["date"].iloc[-1], horizon)
    overall = float(v.mean()) if len(v) else 0.0

    # 요일별 평균 (입력 28일 = 각 요일 4회)
    wd_mean = {d: float(v[wd == d].mean()) if (wd == d).any() else overall for d in range(7)}
    weekday = np.array([wd_mean[d] for d in fut_wd])

    # 최근 주 가중 요일평균 (최근 2주에 가중)
    n = len(v)
    recency = np.linspace(0.5, 1.0, n)  # 과거→최근 가중 상승
    wd_wmean = {}
    for d in range(7):
        m = wd == d
        wd_wmean[d] = float(np.average(v[m], weights=recency[m])) if m.any() else overall
    weekday_recent = np.array([wd_wmean[d] for d in fut_wd])

    mean7 = float(v[-7:].mean())
    last = float(v[-1])
    nz = v[v > 0]
    nz_mean = float(nz.mean()) if len(nz) else 0.0  # 0 제외 평균 (비-0 수준)

    # 요일별 비-0 평균 (주간 계절성 + 비-0 수준 결합)
    wd_nz_map = {}
    for d in range(7):
        sel = v[wd == d]
        nzsel = sel[sel > 0]
        wd_nz_map[d] = float(nzsel.mean()) if len(nzsel) else (nz_mean if nz_mean > 0 else overall)
    wd_nz = np.array([wd_nz_map[d] for d in fut_wd])

    return {
        "zero": np.zeros(horizon),
        "last": np.full(horizon, last),
        "mean7": np.full(horizon, mean7),
        "mean28": np.full(horizon, overall),
        "nz_mean": np.full(horizon, nz_mean),
        "wd_nz": wd_nz,
        "weekday_mean": weekday,
        "weekday_recent": weekday_recent,
        "weekday+mean7(0.5)": 0.5 * weekday + 0.5 * mean7,
    }


def evaluate(n_folds=4):
    train = pd.read_csv(ROOT / "data" / "train" / "train.csv")
    folds = make_holdout(train, lookback=28, horizon=HORIZON, n_folds=n_folds)
    method_names = None
    per_fold = {}  # method -> list of fold scores
    last_fold_long = {}  # method -> long df of最근 fold (per-store 진단용)

    for k, fold in enumerate(folds):
        inp = fold["input_df"].sort_values(["item", "date"])
        ans = fold["answer_df"].sort_values(["item", "date"])
        ans_by_item = {it: g["qty"].to_numpy(dtype=float) for it, g in ans.groupby("item")}

        rows_by_method = {}
        for it, g in inp.groupby("item"):
            if it not in ans_by_item:
                continue
            preds = predictors(g)
            if method_names is None:
                method_names = list(preds.keys())
            true = ans_by_item[it]
            for m, p in preds.items():
                rows_by_method.setdefault(m, []).append(
                    pd.DataFrame({"item": it, "true": true, "pred": p})
                )

        for m in method_names:
            long_df = pd.concat(rows_by_method[m], ignore_index=True)
            per_fold.setdefault(m, []).append(score_long(long_df))
            if k == 0:
                last_fold_long[m] = long_df

    # 결과 표
    print(f"\n=== 로컬 홀드아웃 가중 SMAPE (n_folds={n_folds}, 낮을수록 좋음, LB 스케일) ===")
    print(f"{'method':<22} {'fold0(최근)':>12} {'mean':>8} {'std':>7}")
    summary = []
    for m in method_names:
        sc = np.array(per_fold[m])
        summary.append((m, sc[0], sc.mean(), sc.std()))
    for m, f0, mean, std in sorted(summary, key=lambda x: x[2]):
        print(f"{m:<22} {f0:>12.4f} {mean:>8.4f} {std:>7.4f}")

    best = min(summary, key=lambda x: x[2])[0]
    print(f"\n=== 최고 베이스라인 '{best}' 영업장별 SMAPE (fold0) ===")
    ps = score_per_store(last_fold_long[best])
    print(ps.to_string(index=False))
    return per_fold


if __name__ == "__main__":
    evaluate()
